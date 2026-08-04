#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train RACLETTE Stage-0b with spatial HNF (+ rotational sources)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.fluid_losses import (
    curl_weight_at_epoch,
    lr_scale_at_epoch,
    normalized_curl_loss,
    rel_err_masked,
    velocity_recon_loss,
)
from hnf.fluid_spatial import SpatialFluidHNFReconstructor
from hnf.raclette_dataset import RacletteSliceDataset
from tools.train_fluid import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RACLETTE spatial HNF Stage-0b")
    p.add_argument("--cache", default="external_data/raclette_cache/gt_slices.npz")
    p.add_argument("--output-dir", default="outputs/fluid/stage0b_raclette_spatial")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup-epochs", type=int, default=3)
    p.add_argument("--keep-frac", type=float, default=0.1)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=9)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--obs-weight", type=float, default=0.25)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--vessel-weight", type=float, default=1.0, help="extra vessel emphasis on recon")
    p.add_argument("--curl-weight", type=float, default=0.005)
    p.add_argument("--curl-warmup-epochs", type=int, default=8)
    p.add_argument("--no-rotation", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="")
    return p.parse_args()


@torch.no_grad()
def evaluate_raclette(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rels_full: list[float] = []
    rels_vessel: list[float] = []
    mse_sum = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        dense = batch["dense"].to(device)
        vmask = batch["vessel_mask"].to(device)
        pred, _ = model(x, return_aux=True)
        mse_sum += float(((pred - dense) ** 2).mean().item()) * x.size(0)
        n += x.size(0)
        for i in range(x.size(0)):
            rels_full.append(rel_err_masked(pred[i], dense[i]))
            vm = vmask[i].unsqueeze(0).expand(2, -1, -1)
            rels_vessel.append(rel_err_masked(pred[i], dense[i], vm))
    return {
        "vel_mse": mse_sum / max(n, 1),
        "vel_rel": float(np.mean(rels_full)),
        "vel_rel_inside_vessel": float(np.mean(rels_vessel)),
        "score": -float(np.mean(rels_vessel)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_rotation = not args.no_rotation

    train_ds = RacletteSliceDataset(args.cache, "train", args.keep_frac, args.seed, augment=True)
    val_ds = RacletteSliceDataset(args.cache, "val", args.keep_frac, args.seed, augment=False)
    h = int(train_ds.velocity.shape[2])
    w = int(train_ds.velocity.shape[3])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SpatialFluidHNFReconstructor(
        h=h, w=w, embed_dim=args.embed_dim, kernel_size=args.kernel_size,
        num_layers=args.num_layers, dropout=args.dropout,
        predict_eta=False, use_rotation=use_rotation,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    base_lrs = [g["lr"] for g in opt.param_groups]

    history: list[dict] = []
    best_score = float("-inf")
    best_path = out / "best.pt"
    arch = "spatial+rot" if use_rotation else "spatial"
    print(
        f"[raclette-spatial] arch={arch} device={device} params={n_params} "
        f"train={len(train_ds)} val={len(val_ds)} grid={h}x{w} keep={args.keep_frac}",
        flush=True,
    )

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        lr_mul = lr_scale_at_epoch(epoch, args.lr_warmup_epochs)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_mul
        curl_w = curl_weight_at_epoch(epoch, args.curl_weight, args.curl_warmup_epochs)

        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            dense = batch["dense"].to(device)
            obs_mask = batch["mask"].to(device)
            vmask = batch["vessel_mask"].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(x, return_aux=True)
            loss, _ = velocity_recon_loss(
                pred, dense, obs_mask,
                region_mask=vmask,
                obs_weight=args.obs_weight,
                recon_weight=args.recon_weight * args.vessel_weight,
            )
            if curl_w > 0 and use_rotation:
                loss = loss + curl_w * normalized_curl_loss(pred, dense)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.4f}")
        sched.step()
        train_loss = running / max(n_seen, 1)
        val_m = evaluate_raclette(model, val_loader, device)
        row = {"epoch": float(epoch), "train_loss": train_loss, "curl_weight": curl_w, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"[raclette-spatial] ep {epoch:03d}  train={train_loss:.4f}  "
            f"val_vel_rel={val_m['vel_rel']:.4f}  val_inside={val_m['vel_rel_inside_vessel']:.4f}",
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
                    "arch": arch,
                    "kernel_params": model.collect_kernel_params(),
                },
                best_path,
            )
            print(
                f"[raclette-spatial] saved best → {best_path} "
                f"(inside={val_m['vel_rel_inside_vessel']:.4f})",
                flush=True,
            )

    summary = {"arch": arch, "best_score": best_score, "history": history}
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[raclette-spatial] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
