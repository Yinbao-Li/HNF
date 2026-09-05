#!/usr/bin/env python
"""RadHAR v3b: T=90 frames + stronger jack/jump class weights.

Adds over v3:
  - T=90 frames (3 s window, jack/jump period ≈ 0.8 s → 3+ full cycles visible)
  - jump class weight ×2.0, jack ×1.8  (was ×1.5/×1.25)
  - prev-summary points to v3

  python tools/train_radhar_huygens_cls_v3b.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.radhar_cls_dataset import ACTIVITY_TO_IDX, RadHARClsDataset, class_weights_from_samples
from hnf.radhar_huygens_cls import RadHARHuygensClsModel
from hnf.radhar_io import ACTIVITIES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/radhar/Data"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/mmwave/radhar_cls_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mmwave/radhar_huygens_cls_v3b"))
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-range-bins", type=int, default=24)
    p.add_argument("--n-doppler-bins", type=int, default=8)
    p.add_argument("--t-steps", type=int, default=90)
    p.add_argument("--stride-frames", type=int, default=10)
    p.add_argument("--stem-ch", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--mixup-alpha", type=float, default=0.3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prev-summary",
        type=Path,
        default=Path("outputs/mmwave/radhar_huygens_cls_v3/SUMMARY.json"),
    )
    return p.parse_args()


def cosine_warmrestart_lr(epoch: int, total: int, base_lr: float, warmup: int = 3, n_restarts: int = 1) -> float:
    """Cosine with one warm restart at the midpoint."""
    half = total // (n_restarts + 1)
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    # which cycle?
    cycle_len = half
    offset = epoch - warmup
    pos = offset % cycle_len
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * pos / max(cycle_len, 1)))


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None) -> dict:
    model.eval()
    ys, preds = [], []
    n = 0
    total_loss = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        y = batch["y"].to(device)
        logits = model(x, t)["logits"]
        total_loss += float(F.cross_entropy(logits, y, weight=class_weights, reduction="sum").cpu())
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        n += y.numel()
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    acc = float((y_all == p_all).mean())
    per_class = {}
    for name, idx in ACTIVITY_TO_IDX.items():
        m = y_all == idx
        if m.any():
            per_class[name] = float((p_all[m] == idx).mean())
    cm = np.zeros((len(ACTIVITIES), len(ACTIVITIES)), dtype=np.int64)
    for yi, pi in zip(y_all, p_all):
        cm[int(yi), int(pi)] += 1
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "per_class_recall": per_class,
        "n": int(n),
        "confusion": cm.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ds_kw = dict(
        n_range_bins=args.n_range_bins,
        t_steps=args.t_steps,
        stride_frames=args.stride_frames,
        feature_mode="rich",
        include_range_doppler=True,
        n_doppler_bins=args.n_doppler_bins,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    train_ds = RadHARClsDataset(args.data_dir, "train", augment=True, seed=args.seed, **ds_kw)
    test_ds = RadHARClsDataset(args.data_dir, "test", augment=False, **ds_kw)

    # class weights with stronger jump/jack boost
    class_w = class_weights_from_samples(train_ds.samples).to(device)
    for act, boost in (("jump", 2.0), ("jack", 1.8)):
        class_w[ACTIVITY_TO_IDX[act]] *= boost
    class_w = class_w / class_w.mean()

    print(
        f"[v3b] train={len(train_ds)} test={len(test_ds)} "
        f"C={train_ds.n_channels} T={args.t_steps}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = RadHARHuygensClsModel(
        n_channels=train_ds.n_channels,
        n_classes=len(ACTIVITIES),
        stem_ch=args.stem_ch,
        embed_dim=args.embed_dim,
        num_shared_layers=args.num_layers,
        epoch_sec=train_ds.epoch_sec,
        local_window_sec=min(0.5, train_ds.epoch_sec * 0.4),
        dropout=0.3,
        medium_hidden=64,
        residual_energy=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v3b] params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best = {"accuracy": -1.0}

    for epoch in range(1, args.epochs + 1):
        lr = cosine_warmrestart_lr(epoch, args.epochs, args.lr, warmup=3, n_restarts=1)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        run_loss = 0.0
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)

            # MixUp
            x, ya, yb, lam = mixup_batch(x, y, args.mixup_alpha, device)
            logits = model(x, t)["logits"]
            loss = (
                lam * F.cross_entropy(logits, ya, weight=class_w, label_smoothing=args.label_smoothing)
                + (1.0 - lam) * F.cross_entropy(logits, yb, weight=class_w, label_smoothing=args.label_smoothing)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss.detach().cpu()) * y.size(0)
            n += y.size(0)

        te = evaluate(model, test_loader, device, class_w)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": run_loss / max(n, 1),
            "test_loss": te["loss"],
            "test_accuracy": te["accuracy"],
            "per_class_recall": te["per_class_recall"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} lr={lr:.2e} train={row['train_loss']:.4f} "
            f"test_acc={te['accuracy']:.4f} per={te['per_class_recall']}",
            flush=True,
        )
        if te["accuracy"] > best["accuracy"]:
            best = {
                "accuracy": te["accuracy"],
                "epoch": epoch,
                "per_class_recall": te["per_class_recall"],
                "confusion": te["confusion"],
                "n_test": te["n"],
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "activities": list(ACTIVITIES),
                    "n_channels": train_ds.n_channels,
                },
                args.output_dir / "best.pt",
            )

    prev_acc = None
    if args.prev_summary.exists():
        try:
            prev_acc = json.loads(args.prev_summary.read_text()).get("best", {}).get("accuracy")
        except Exception:
            pass

    summary = {
        "dataset": "RadHAR",
        "version": "v3b_t90_strongweights",
        "method": "fixed Huygens/wave frontend",
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_channels": train_ds.n_channels,
        "n_params": n_params,
        "best": best,
        "v2_baseline_accuracy": prev_acc,
        "delta_vs_v2": (best["accuracy"] - prev_acc) if prev_acc is not None else None,
        "history": history,
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = [
        "# RadHAR Huygens/wave v3b (T=90, stronger jack/jump weights)",
        "",
        f"Params: {n_params:,} | Best test acc: **{best['accuracy']:.4f}** (epoch {best.get('epoch')})",
    ]
    if prev_acc is not None:
        lines.append(f"v2 baseline: {prev_acc:.4f} | delta: **{best['accuracy'] - prev_acc:+.4f}**")
    lines += ["", "## Per-class recall", ""]
    for k, v in (best.get("per_class_recall") or {}).items():
        lines.append(f"- {k}: {v:.4f}")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[done] best_acc={best['accuracy']:.4f} → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
