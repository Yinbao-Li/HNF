#!/usr/bin/env python
"""Train STEAD-style Huygens/wave classifier on public RadHAR (task C).

Fixed wave frontend — no regime ablation. Official Train/Test activity labels.

  python tools/download_radhar.py
  python tools/train_radhar_huygens_cls.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.radhar_cls_dataset import ACTIVITY_TO_IDX, RadHARClsDataset
from hnf.radhar_huygens_cls import RadHARHuygensClsModel
from hnf.radhar_io import ACTIVITIES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/radhar/Data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mmwave/radhar_huygens_cls_v1"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-range-bins", type=int, default=24)
    p.add_argument("--t-steps", type=int, default=60)
    p.add_argument("--stride-frames", type=int, default=10)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, preds = [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        y = batch["y"].to(device)
        logits = model(x, t)["logits"]
        total_loss += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        pred = logits.argmax(dim=-1)
        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())
        n += y.numel()
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    acc = float((y_all == p_all).mean())
    per_class = {}
    for name, idx in ACTIVITY_TO_IDX.items():
        m = y_all == idx
        if m.any():
            per_class[name] = float((p_all[m] == idx).mean())
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "per_class_recall": per_class,
        "n": int(n),
        "confusion": _confusion(y_all, p_all, len(ACTIVITIES)).tolist(),
    }


def _confusion(y: np.ndarray, p: np.ndarray, n_cls: int) -> np.ndarray:
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for yi, pi in zip(y, p):
        cm[int(yi), int(pi)] += 1
    return cm


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    train_ds = RadHARClsDataset(
        args.data_dir,
        "train",
        n_range_bins=args.n_range_bins,
        t_steps=args.t_steps,
        stride_frames=args.stride_frames,
        max_windows_per_file=None,
        max_files=args.max_files,
        feature_mode="intensity",
        include_range_doppler=False,
        cache_dir=None,
    )
    test_ds = RadHARClsDataset(
        args.data_dir,
        "test",
        n_range_bins=args.n_range_bins,
        t_steps=args.t_steps,
        stride_frames=args.stride_frames,
        max_windows_per_file=None,
        max_files=args.max_files,
        feature_mode="intensity",
        include_range_doppler=False,
        cache_dir=None,
    )
    print(
        f"[radhar] train={len(train_ds)} test={len(test_ds)} "
        f"C={train_ds.n_channels} T={args.t_steps} epoch_sec={train_ds.epoch_sec:.2f}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = RadHARHuygensClsModel(
        n_channels=train_ds.n_channels,
        n_classes=len(ACTIVITIES),
        epoch_sec=train_ds.epoch_sec,
        local_window_sec=min(0.5, train_ds.epoch_sec * 0.4),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = {"accuracy": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            logits = model(x, t)["logits"]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach().cpu()) * y.size(0)
            n += y.size(0)
        te = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": run_loss / max(n, 1),
            "test_loss": te["loss"],
            "test_accuracy": te["accuracy"],
            "per_class_recall": te["per_class_recall"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_loss={row['train_loss']:.4f}  "
            f"test_acc={te['accuracy']:.4f}  per={te['per_class_recall']}",
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
                {"model": model.state_dict(), "args": vars(args), "activities": list(ACTIVITIES)},
                args.output_dir / "best.pt",
            )

    summary = {
        "dataset": "RadHAR",
        "task": "5-class activity detection (official Train/Test)",
        "method": "fixed Huygens/wave frontend (STEAD detection path)",
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "best": best,
        "history": history,
        "note": "Input is range×slow-time from public point clouds; not official voxels.",
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    report = [
        "# RadHAR Huygens/wave 5-class activity detection",
        "",
        f"Train windows: {len(train_ds)} | Test windows: {len(test_ds)}",
        f"Best test accuracy: **{best['accuracy']:.4f}** (epoch {best.get('epoch')})",
        "",
        "## Per-class recall",
        "",
    ]
    for k, v in (best.get("per_class_recall") or {}).items():
        report.append(f"- {k}: {v:.4f}")
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"[done] best_acc={best['accuracy']:.4f} → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
