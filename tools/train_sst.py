#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Huygens Neural Field on NOAA SST (user's HuygensKernel pipeline)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hnf.field import HuygensNeuralField
from hnf.sst_dataset import SSTDataset, SST_H5, SST_TEST, SST_TRAIN, SST_VAL


def parse_args():
    p = argparse.ArgumentParser(description="Train HNF on NOAA SST")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=1e-2)
    p.add_argument("--eps", type=float, default=1e-2, help="Kernel eps (use ~1e-2 for SST coords in [-1,1])")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=6.2831853)
    p.add_argument("--use-density", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs/sst_train_v2")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def masked_l1(pred, target, mask):
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, device, ocean_mask):
    model.eval()
    total, count = 0.0, 0
    mask = ocean_mask.to(device)
    for batch in loader:
        obs_c = batch["obs_coords"][0].to(device)
        tgt_c = batch["target_coords"][0].to(device)
        obs_v = batch["obs_values"].to(device)
        tgt = batch["target_field"].to(device)
        preds = [model(obs_c, obs_v[b], tgt_c) for b in range(obs_v.shape[0])]
        pred = torch.stack(preds, dim=0)
        batch_mask = mask.unsqueeze(0).expand_as(tgt)
        total += masked_l1(pred, tgt, batch_mask).item()
        count += 1
    return total / max(count, 1)


def train():
    args = parse_args()
    if not SST_H5.is_file():
        raise FileNotFoundError(f"SST data not found: {SST_H5}")

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SSTDataset(SST_TRAIN)
    val_ds = SSTDataset(SST_VAL)
    ocean_mask = train_ds.ocean_mask.reshape(-1, 1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = HuygensNeuralField(
        gamma=args.gamma,
        omega=args.omega,
        alpha=args.alpha,
        eps=args.eps,
        causal=False,
        learnable_gamma=True,
        learnable_omega=True,
        use_density=args.use_density,
    ).to(device)

    print(
        f"[HNF-SST] eps={args.eps}  alpha={args.alpha}  "
        f"gamma={model.gamma.item():.4f}  omega={model.omega.item():.4f}",
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)

    history_path = out_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_l1", "val_l1", "gamma", "omega", "lr", "sec"])

    best_val = float("inf")
    print(f"[HNF-SST] device={device}  train={len(train_ds)}  val={len(val_ds)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            obs_c = batch["obs_coords"][0].to(device)
            tgt_c = batch["target_coords"][0].to(device)
            obs_v = batch["obs_values"].to(device)
            tgt = batch["target_field"].to(device)
            mask = ocean_mask.to(device).unsqueeze(0).expand_as(tgt)

            opt.zero_grad()
            preds = [model(obs_c, obs_v[b], tgt_c) for b in range(obs_v.shape[0])]
            pred = torch.stack(preds, dim=0)
            loss = masked_l1(pred, tgt, mask)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            n_batches += 1

        sched.step()
        train_loss /= max(n_batches, 1)
        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_l1": train_loss,
            "gamma": model.gamma.item(),
            "omega": model.omega.item(),
            "lr": opt.param_groups[0]["lr"],
            "sec": elapsed,
        }

        if epoch % args.eval_every == 0 or epoch == 1:
            val_l1 = evaluate(model, val_loader, device, ocean_mask)
            log["val_l1"] = val_l1
            if val_l1 < best_val:
                best_val = val_l1
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_l1": val_l1,
                    "gamma": model.gamma.item(),
                    "omega": model.omega.item(),
                }, out_dir / "best.pt")
            print(
                f"ep {epoch:03d}  train_l1={train_loss:.5f}  val_l1={val_l1:.5f}  "
                f"gamma={model.gamma.item():.4f}  omega={model.omega.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.6f}  {elapsed:.1f}s",
                flush=True,
            )
        else:
            log["val_l1"] = ""
            print(
                f"ep {epoch:03d}  train_l1={train_loss:.5f}  "
                f"gamma={model.gamma.item():.4f}  omega={model.omega.item():.4f}  {elapsed:.1f}s",
                flush=True,
            )

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([
                log["epoch"], log["train_l1"], log.get("val_l1", ""),
                log["gamma"], log["omega"], log["lr"], log["sec"],
            ])

    test_ds = SSTDataset(SST_TEST)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_l1 = evaluate(model, test_loader, device, ocean_mask)
    metrics = {"test_l1": test_l1, "best_val_l1": best_val, "best_epoch": ckpt["epoch"]}
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[HNF-SST] done  test_l1={test_l1:.5f}  best_val={best_val:.5f}", flush=True)


if __name__ == "__main__":
    train()
