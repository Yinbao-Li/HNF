#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Braindecode SOTA EEG models on ds004504 (same protocol as HNF)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.eeg_braindecode_models import build_braindecode_model, display_name
from hnf.eeg_dataset import EEGDataset, LABEL_TO_ID
from tools.train_eeg import evaluate, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Braindecode EEG SOTA baseline")
    p.add_argument(
        "--model",
        required=True,
        choices=["eegnetv4", "shallowfbcsp", "deep4net", "eegconformer"],
    )
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--output-dir", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--att-dropout", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="")
    p.add_argument("--sample-rate", type=int, default=128)
    p.add_argument("--epoch-sec", type=float, default=10.0)
    p.add_argument("--stride-sec", type=float, default=5.0)
    p.add_argument("--synthetic-if-missing", action="store_true", default=True)
    p.add_argument("--no-synthetic", action="store_false", dest="synthetic_if_missing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir or f"outputs/eeg/adftd_bd_{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    n_samples = int(round(args.epoch_sec * args.sample_rate))

    common = dict(
        data_dir=args.data_dir,
        test_ratio=0.2,
        val_ratio=0.15,
        seed=args.seed,
        sample_rate=args.sample_rate,
        epoch_sec=args.epoch_sec,
        stride_sec=args.stride_sec,
        synthetic_if_missing=args.synthetic_if_missing,
    )
    train_ds = EEGDataset(split="train", **common)
    val_ds = EEGDataset(split="val", **common)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    extra = {}
    if args.att_dropout is not None:
        extra["att_drop_prob"] = args.att_dropout
    model = build_braindecode_model(
        args.model,
        n_channels=19,
        n_samples=n_samples,
        n_classes=len(LABEL_TO_ID),
        dropout=args.dropout,
        extra=extra,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    wd = args.weight_decay
    lr = args.lr
    if args.model == "eegconformer":
        wd = max(wd, 5e-4)
        lr = min(lr, 3e-4)
    if args.model == "deep4net":
        wd = max(wd, 5e-4)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_auc = -1.0
    best_path = out / "best.pt"
    print(
        f"[EEG-bd] model={display_name(args.model)} device={device} params={n_params} "
        f"train={len(train_ds)} val={len(val_ds)}",
        flush=True,
    )

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"{args.model} {epoch:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.3f}")
        sched.step()
        train_loss = running / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[EEG-bd:{args.model}] ep {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}",
            flush=True,
        )
        score = val_metrics["auc"] if val_metrics["auc"] == val_metrics["auc"] else val_metrics["acc"]
        if score >= best_auc:
            best_auc = score
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "n_params": n_params,
                    "model_name": args.model,
                    "backend": "braindecode",
                    "label_to_id": LABEL_TO_ID,
                },
                best_path,
            )
            print(f"[EEG-bd] saved best → {best_path} (score={best_auc:.4f})", flush=True)

    with (out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "best_auc": best_auc,
                "model": args.model,
                "backend": "braindecode",
            },
            f,
            indent=2,
        )
    print(f"[EEG-bd] done in {time.time() - t0:.1f}s  best_val_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
