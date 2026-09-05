#!/usr/bin/env python
"""Train RadHAR spatial-energy-pattern head classifier (T=120).

Decision for jack/jump is made in the head by detecting spatial energy
distribution patterns (centroid vs width oscillation), not by near→far mixing.

  python tools/train_radhar_spatial_wave_cls.py --device cuda --t-steps 120
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
from hnf.radhar_io import ACTIVITIES
from hnf.radhar_spatial_wave_cls import RadHARSpatialPatternClsModel

N_CLASSES = len(ACTIVITIES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/radhar/Data"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/mmwave/radhar_cls_cache"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/mmwave/radhar_spatial_pattern_v2_t120"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--t-steps", type=int, default=120)
    p.add_argument("--n-range-bins", type=int, default=24)
    p.add_argument("--n-doppler-bins", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--no-jack-jump-boost", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prev-summary",
        type=Path,
        default=Path("outputs/mmwave/radhar_huygens_cls_v3_t120/SUMMARY.json"),
    )
    return p.parse_args()


def cosine_lr(epoch: int, total: int, base_lr: float, warmup: int = 3) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 1e-12 else 0.0


@torch.no_grad()
def evaluate(model, loader, device, class_w) -> dict:
    model.eval()
    ys, preds = [], []
    w_amp, jack_nf, jump_nf, jump_dw = [], [], [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x)
        logits = out["logits"]
        total_loss += float(F.cross_entropy(logits, y, weight=class_w, reduction="sum").cpu())
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        w_amp.append(out["width_amp"].cpu().numpy())
        jack_nf.append(out["jack_nearfar_amp"].cpu().numpy())
        jump_nf.append(out["jump_nearfar_mean"].cpu().numpy())
        jump_dw.append(out["jump_doppler_width_mean"].cpu().numpy())
        n += y.numel()
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    w_amp = np.concatenate(w_amp)
    jack_nf = np.concatenate(jack_nf)
    jump_nf = np.concatenate(jump_nf)
    jump_dw = np.concatenate(jump_dw)
    acc = float((y_all == p_all).mean())
    per = {
        name: float((p_all[y_all == idx] == idx).mean()) if (y_all == idx).any() else float("nan")
        for name, idx in ACTIVITY_TO_IDX.items()
    }
    j_jack, j_jump = ACTIVITY_TO_IDX["jack"], ACTIVITY_TO_IDX["jump"]
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for yi, pi in zip(y_all, p_all):
        cm[int(yi), int(pi)] += 1
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "per_class_recall": per,
        "n": int(n),
        "confusion": cm.tolist(),
        "width_amp_d_jack_minus_jump": cohens_d(w_amp[y_all == j_jack], w_amp[y_all == j_jump]),
        "jack_nearfar_amp_d": cohens_d(jack_nf[y_all == j_jack], jack_nf[y_all == j_jump]),
        "jump_nearfar_mean_d": cohens_d(jump_nf[y_all == j_jump], jump_nf[y_all == j_jack]),
        "jump_doppler_width_d": cohens_d(jump_dw[y_all == j_jump], jump_dw[y_all == j_jack]),
        "width_amp_by_class": {
            name: float(w_amp[y_all == idx].mean()) if (y_all == idx).any() else float("nan")
            for name, idx in ACTIVITY_TO_IDX.items()
        },
        "jump_nearfar_by_class": {
            name: float(jump_nf[y_all == idx].mean()) if (y_all == idx).any() else float("nan")
            for name, idx in ACTIVITY_TO_IDX.items()
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ds_kw = dict(
        n_range_bins=args.n_range_bins, t_steps=args.t_steps, stride_frames=10,
        feature_mode="rich", include_range_doppler=True,
        n_doppler_bins=args.n_doppler_bins, cache_dir=args.cache_dir,
    )
    train_ds = RadHARClsDataset(args.data_dir, "train", augment=True, seed=args.seed, **ds_kw)
    test_ds = RadHARClsDataset(args.data_dir, "test", augment=False, **ds_kw)

    class_w = class_weights_from_samples(train_ds.samples).to(device)
    for act, boost in (("jump", 1.5), ("jack", 1.25)):
        class_w[ACTIVITY_TO_IDX[act]] *= boost
    class_w = class_w / class_w.mean()

    model = RadHARSpatialPatternClsModel(
        n_channels=train_ds.n_channels,
        n_classes=N_CLASSES,
        n_range_bins=args.n_range_bins,
        n_doppler_bins=args.n_doppler_bins,
        embed_dim=args.embed_dim,
        dropout=0.3,
        jack_jump_boost=not args.no_jack_jump_boost,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[pattern-head] train={len(train_ds)} test={len(test_ds)} "
        f"C={train_ds.n_channels} T={args.t_steps} params={n_params:,} "
        f"jack_jump_boost={not args.no_jack_jump_boost}",
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history, best = [], {"accuracy": -1.0}
    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(epoch, args.epochs, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)["logits"]
            loss = F.cross_entropy(logits, y, weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss.detach().cpu()) * y.size(0)
            n += y.size(0)

        te = evaluate(model, test_loader, device, class_w)
        row = {
            "epoch": epoch, "lr": lr, "train_loss": run_loss / max(n, 1),
            "test_accuracy": te["accuracy"], "per_class_recall": te["per_class_recall"],
            "width_amp_d": te["width_amp_d_jack_minus_jump"],
            "jump_nf_d": te["jump_nearfar_mean_d"],
            "jump_dw_d": te["jump_doppler_width_d"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} lr={lr:.2e} tr={row['train_loss']:.4f} "
            f"acc={te['accuracy']:.4f} jack={te['per_class_recall'].get('jack',0):.3f} "
            f"jump={te['per_class_recall'].get('jump',0):.3f} "
            f"w_d={te['width_amp_d_jack_minus_jump']:.3f} "
            f"jnf_d={te['jump_nearfar_mean_d']:.3f} "
            f"jdw_d={te['jump_doppler_width_d']:.3f}",
            flush=True,
        )
        if te["accuracy"] > best["accuracy"]:
            best = {k: te[k] for k in (
                "accuracy", "per_class_recall", "confusion", "n",
                "width_amp_d_jack_minus_jump", "jack_nearfar_amp_d",
                "jump_nearfar_mean_d", "jump_doppler_width_d",
                "width_amp_by_class", "jump_nearfar_by_class",
            )}
            best["epoch"] = epoch
            best["n_test"] = te["n"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.output_dir / "best.pt")

    prev = None
    if args.prev_summary.exists():
        try:
            prev = json.loads(args.prev_summary.read_text()).get("best", {}).get("accuracy")
        except Exception:
            pass

    summary = {
        "dataset": "RadHAR",
        "version": "spatial_energy_pattern_head_v2_t120",
        "method": "multi-cue E(t,r) + pattern head (jack←bearing nearfar/width; jump←inten nearfar + dop width)",
        "n_train": len(train_ds), "n_test": len(test_ds),
        "n_channels": train_ds.n_channels, "n_params": n_params,
        "t_steps": args.t_steps, "best": best,
        "wave_v3_t120_accuracy": prev,
        "delta_vs_wave_v3_t120": (best["accuracy"] - prev) if prev is not None else None,
        "history": history,
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = [
        "# RadHAR Spatial Energy Pattern Head v2 (T=120)",
        "",
        f"Best acc: **{best['accuracy']:.4f}** (epoch {best.get('epoch')})",
        f"width_amp d (jack−jump): **{best.get('width_amp_d_jack_minus_jump', float('nan')):.3f}**",
        f"jump nearfar_mean d (jump−jack): **{best.get('jump_nearfar_mean_d', float('nan')):.3f}**",
        f"jump doppler_width d (jump−jack): **{best.get('jump_doppler_width_d', float('nan')):.3f}**",
        "",
        "## Per-class recall",
        "",
    ]
    for k, v in (best.get("per_class_recall") or {}).items():
        lines.append(f"- {k}: {v:.4f}")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[done] best={best['accuracy']:.4f} → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
