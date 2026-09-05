#!/usr/bin/env python
"""Fair T=120 baseline for RadHAR: BiGRU and 1D-CNN on the same cache as WaveGRU T=120.

  python tools/train_radhar_baseline_t120.py --device cuda
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

from hnf.radhar_baselines import CNNBaseline, GRUBaseline
from hnf.radhar_cls_dataset import ACTIVITY_TO_IDX, RadHARClsDataset, class_weights_from_samples
from hnf.radhar_io import ACTIVITIES

N_CLASSES = len(ACTIVITIES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/radhar/Data"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/mmwave/radhar_cls_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mmwave/radhar_baseline_t120"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--t-steps", type=int, default=120)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--wavegru-summary",
        type=Path,
        default=Path("outputs/mmwave/radhar_huygens_cls_v4_wavegru_t120/SUMMARY.json"),
    )
    return p.parse_args()


def cosine_lr(epoch: int, total: int, base_lr: float, warmup: int = 3) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, loader, device, class_w) -> dict:
    model.eval()
    ys, preds = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        y = batch["y"].to(device)
        logits = model(x, t)["logits"]
        total_loss += float(F.cross_entropy(logits, y, weight=class_w, reduction="sum").cpu())
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        n += y.numel()
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    acc = float((y_all == p_all).mean())
    per = {
        name: float((p_all[y_all == idx] == idx).mean()) if (y_all == idx).any() else float("nan")
        for name, idx in ACTIVITY_TO_IDX.items()
    }
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for yi, pi in zip(y_all, p_all):
        cm[int(yi), int(pi)] += 1
    return {"loss": total_loss / max(n, 1), "accuracy": acc,
            "per_class_recall": per, "n": n, "confusion": cm.tolist()}


def train_one(model, train_loader, test_loader, *, device, epochs, lr, class_w, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history, best = [], {"accuracy": -1.0}
    for epoch in range(1, epochs + 1):
        cur_lr = cosine_lr(epoch, epochs, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            logits = model(x, t)["logits"]
            loss = F.cross_entropy(logits, y, weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss.detach().cpu()) * y.size(0)
            n += y.size(0)
        te = evaluate(model, test_loader, device, class_w)
        history.append({
            "epoch": epoch, "lr": cur_lr, "train_loss": run_loss / max(n, 1),
            "test_accuracy": te["accuracy"], "per_class_recall": te["per_class_recall"],
        })
        print(
            f"  ep{epoch:02d} lr={cur_lr:.2e} tr={run_loss/max(n,1):.4f} "
            f"acc={te['accuracy']:.4f} jack={te['per_class_recall'].get('jack', 0):.3f} "
            f"jump={te['per_class_recall'].get('jump', 0):.3f}",
            flush=True,
        )
        if te["accuracy"] > best["accuracy"]:
            best = {
                "accuracy": te["accuracy"], "epoch": epoch,
                "per_class_recall": te["per_class_recall"],
                "confusion": te["confusion"], "n_test": te["n"],
            }
    return best, history


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ds_kw = dict(
        n_range_bins=24, t_steps=args.t_steps, stride_frames=10,
        feature_mode="rich", include_range_doppler=True, n_doppler_bins=8,
        cache_dir=args.cache_dir,
    )
    train_ds = RadHARClsDataset(args.data_dir, "train", augment=True, seed=args.seed, **ds_kw)
    test_ds = RadHARClsDataset(args.data_dir, "test", augment=False, **ds_kw)
    n_ch = train_ds.n_channels

    class_w = class_weights_from_samples(train_ds.samples).to(device)
    for act, boost in (("jump", 1.5), ("jack", 1.25)):
        class_w[ACTIVITY_TO_IDX[act]] *= boost
    class_w = class_w / class_w.mean()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    experiments = [
        ("B3_gru_t120", "BiGRU T=120", lambda: GRUBaseline(n_ch, N_CLASSES, hidden=128, n_layers=2)),
        ("B2_cnn_t120", "1D-CNN T=120", lambda: CNNBaseline(n_ch, N_CLASSES, hidden=128, n_layers=4)),
    ]

    rows = []
    for eid, label, build in experiments:
        print(f"\n{'='*60}\n[{eid}] {label}\n{'='*60}", flush=True)
        model = build().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best, history = train_one(
            model, train_loader, test_loader,
            device=device, epochs=args.epochs, lr=args.lr,
            class_w=class_w, seed=args.seed,
        )
        row = {
            "id": eid, "label": label, "best": best, "n_params": n_params,
        }
        rows.append(row)
        (args.output_dir / f"{eid}.json").write_text(json.dumps({
            **row, "history": history,
        }, indent=2, default=str))
        print(f"  → best_acc={best['accuracy']:.4f} params={n_params:,}", flush=True)

    wave_acc = None
    if args.wavegru_summary.exists():
        try:
            wave_acc = json.loads(args.wavegru_summary.read_text()).get("best", {}).get("accuracy")
        except Exception:
            pass

    summary = {
        "dataset": "RadHAR",
        "t_steps": args.t_steps,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_channels": n_ch,
        "wavegru_t120_accuracy": wave_acc,
        "rows": [
            {
                "id": r["id"],
                "label": r["label"],
                "best_accuracy": r["best"]["accuracy"],
                "best_epoch": r["best"]["epoch"],
                "per_class_recall": r["best"]["per_class_recall"],
                "n_params": r["n_params"],
            }
            for r in rows
        ],
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# RadHAR T=120 baselines (same cache as WaveGRU T=120)",
        "",
        f"Train={len(train_ds)} Test={len(test_ds)} T={args.t_steps}",
        "",
        "| Model | Acc | jack | jump | params |",
        "|---|---:|---:|---:|---:|",
    ]
    if wave_acc is not None:
        lines.append(f"| WaveGRU T=120 (ref) | **{wave_acc:.4f}** | — | — | — |")
    for r in rows:
        per = r["best"]["per_class_recall"]
        lines.append(
            f"| {r['label']} | **{r['best']['accuracy']:.4f}** "
            f"| {per.get('jack', float('nan')):.3f} "
            f"| {per.get('jump', float('nan')):.3f} "
            f"| {r['n_params']} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
