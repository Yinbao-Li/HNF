#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train HNF on STEAD for earthquake-vs-noise classification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hnf.layers import HuygensAttention
from hnf.stead_dataset import STEADDataset


class STEADHNFClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        gamma: float = 0.5,
        omega: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                HuygensAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    gamma=gamma * (0.9 ** i),
                    omega=omega * (1.05 ** i),
                    causal=True,
                    wave_speed=1.0,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for attn, norm in zip(self.layers, self.norms):
            h = norm(h + self.dropout(attn(h, t=t)))
        pooled = h.mean(dim=1)
        return self.head(pooled)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HNF on STEAD")
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--max-per-class", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=0.3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs/stead_hnf_cls")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = fp = fn = 0
    for batch in loader:
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        y = batch["y"].to(device)
        logits = model(x, t)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        correct += (pred == y).sum().item()
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = STEADDataset("train", seq_len=args.seq_len, max_per_class=args.max_per_class)
    val_ds = STEADDataset("val", seq_len=args.seq_len, max_per_class=max(2000, args.max_per_class // 5))
    test_ds = STEADDataset("test", seq_len=args.seq_len, max_per_class=max(2000, args.max_per_class // 5))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = STEADHNFClassifier(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        gamma=args.gamma,
        omega=args.omega,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(
        f"[STEAD-HNF] device={device} train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} seq_len={args.seq_len}",
        flush=True,
    )

    history_path = out_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"])

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad()
            logits = model(x, t)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()

        sched.step()
        train_metrics = {
            "loss": total_loss / max(total, 1),
            "acc": correct / max(total, 1),
        }
        val_metrics = evaluate(model, val_loader, device)

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["acc"],
                    val_metrics["loss"],
                    val_metrics["acc"],
                    val_metrics["f1"],
                    opt.param_groups[0]["lr"],
                ]
            )

        print(
            f"ep {epoch:03d}  train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['acc']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  lr={opt.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"[STEAD-HNF] done  test_acc={test_metrics['acc']:.4f}  test_f1={test_metrics['f1']:.4f}", flush=True)


if __name__ == "__main__":
    train()

