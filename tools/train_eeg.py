#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train HNF EEG classifier on OpenNeuro ds004504 (Domain II)."""

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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.eeg_dataset import EEGDataset, LABEL_TO_ID
from hnf.eeg_model import EEGHNFClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEG HNF classifier (AD/FTD/HC)")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--output-dir", default="outputs/eeg/adftd_hnf")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--multi-scale", action="store_true", default=True)
    p.add_argument("--no-multi-scale", action="store_false", dest="multi_scale")
    p.add_argument("--principle", default="huygens_fresnel")
    p.add_argument("--sample-rate", type=int, default=128)
    p.add_argument("--epoch-sec", type=float, default=10.0)
    p.add_argument("--stride-sec", type=float, default=5.0)
    p.add_argument("--synthetic-if-missing", action="store_true", default=True)
    p.add_argument("--no-synthetic", action="store_false", dest="synthetic_if_missing")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
) -> dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits_cat.argmax(dim=-1)
    acc = float((preds == labels).float().mean().item())
    probs = torch.softmax(logits_cat.float(), dim=-1).numpy()
    y_np = labels.numpy()
    auc_macro = _macro_auc(y_np, probs, num_classes)
    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "auc": auc_macro,
    }


def _macro_auc(y: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return _numpy_macro_auc(y, probs, num_classes)
    # One-vs-rest macro AUC; skip classes missing in y
    present = sorted(set(int(v) for v in y.tolist()))
    if len(present) < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(
                y,
                probs,
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes)),
            )
        )
    except ValueError:
        return _numpy_macro_auc(y, probs, num_classes)


def _numpy_macro_auc(y: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """Fallback AUC without sklearn."""
    scores: list[float] = []
    for c in range(num_classes):
        yt = (y == c).astype(np.float64)
        if yt.min() == yt.max():
            continue
        s = probs[:, c]
        order = np.argsort(s)
        yt_sorted = yt[order]
        n_pos = yt.sum()
        n_neg = len(yt) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = np.arange(1, len(yt) + 1, dtype=np.float64)
        # Mann–Whitney via rank sum of positives
        # After sorting by score ascending, use average ranks for ties omitted (simple)
        pos_rank_sum = ranks[yt_sorted == 1].sum()
        auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        scores.append(float(auc))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

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
    test_ds = EEGDataset(split="test", **common)

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

    model = EEGHNFClassifier(
        n_channels=19,
        seq_len=int(round(args.epoch_sec * args.sample_rate)),
        sample_rate=args.sample_rate,
        embed_dim=args.embed_dim,
        num_classes=len(LABEL_TO_ID),
        dropout=args.dropout,
        multi_scale=args.multi_scale,
        principle=args.principle,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[EEG] resumed from {args.resume}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_auc = -1.0
    best_path = out / "best.pt"
    print(
        f"[EEG] device={device} params={n_params} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"subjects_train={len(train_ds.subjects)} principle={args.principle}",
        flush=True,
    )

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
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
            f"[EEG] ep {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}  lr={row['lr']:.2e}",
            flush=True,
        )
        auc = val_metrics["auc"]
        score = auc if auc == auc else val_metrics["acc"]  # NaN → fall back to acc
        if score >= best_auc:
            best_auc = score
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "n_params": n_params,
                    "label_to_id": LABEL_TO_ID,
                },
                best_path,
            )
            print(f"[EEG] saved best → {best_path} (score={best_auc:.4f})", flush=True)

    torch.save({"history": history, "best_auc": best_auc}, out / "history.pt")
    with (out / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history, "best_auc": best_auc}, f, indent=2)
    print(f"[EEG] done in {time.time() - t0:.1f}s  best_val_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
