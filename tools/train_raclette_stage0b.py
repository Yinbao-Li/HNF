#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Domain-III Stage-0b: RACLETTE sparse→dense velocity reconstruction."""

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

from hnf.fluid_model import FluidHNFReconstructor
from hnf.raclette_dataset import RacletteSliceDataset
from tools.train_fluid import evaluate, rel_err, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RACLETTE Stage-0b")
    p.add_argument("--cache", default="external_data/raclette_cache/gt_slices.npz")
    p.add_argument("--output-dir", default="outputs/fluid/stage0b_raclette")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--keep-frac", type=float, default=0.1)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="")
    p.add_argument("--principle", default="huygens_fresnel")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_ds = RacletteSliceDataset(args.cache, "train", args.keep_frac, args.seed, augment=True)
    val_ds = RacletteSliceDataset(args.cache, "val", args.keep_frac, args.seed, augment=False)
    h = int(train_ds.velocity.shape[2])
    w = int(train_ds.velocity.shape[3])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = FluidHNFReconstructor(
        h=h,
        w=w,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        principle=args.principle,
        predict_eta=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()

    history = []
    best_score = float("-inf")
    best_path = out / "best.pt"
    print(
        f"[raclette-s0b] device={device} params={n_params} "
        f"train={len(train_ds)} val={len(val_ds)} keep={args.keep_frac} grid={h}x{w}",
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
            dense = batch["dense"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            # Emphasize vessel pixels when mask available
            vmask = batch["vessel_mask"].to(device).unsqueeze(1)
            weight = vmask + 0.05
            loss = ((pred - dense).pow(2) * weight).sum() / weight.sum().clamp_min(1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.4f}")
        sched.step()
        train_loss = running / max(n_seen, 1)
        val_m = evaluate(model, val_loader, device, eta_weight=0.0)
        row = {"epoch": float(epoch), "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"[raclette-s0b] ep {epoch:03d}  train={train_loss:.4f}  "
            f"val_vel_rel={val_m['vel_rel']:.4f}  val_mse={val_m['vel_mse']:.6f}",
            flush=True,
        )
        if val_m["score"] >= best_score:
            best_score = val_m["score"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_m,
                    "args": vars(args),
                    "n_params": n_params,
                    "grid": [h, w],
                    "kernel_params": model.collect_kernel_params(),
                },
                best_path,
            )
            print(f"[raclette-s0b] saved best → {best_path} (vel_rel={val_m['vel_rel']:.4f})", flush=True)

    with (out / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history, "best_score": best_score}, f, indent=2)
    print(f"[raclette-s0b] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
