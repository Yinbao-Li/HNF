#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train RACLETTE 3D patches with spatial HNF."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.fluid_losses import lr_scale_at_epoch, rel_err_masked, velocity_recon_loss
from hnf.fluid_spatial3d import Spatial3DFluidHNFReconstructor
from hnf.raclette_volume_dataset import RacletteVolumeDataset
from tools.train_fluid import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="external_data/raclette_cache/gt_volumes.npz")
    p.add_argument("--output-dir", default="outputs/fluid/stage0c_raclette_spatial3d")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep-frac", type=float, default=0.1)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rels_full, rels_in = [], []
    for batch in loader:
        x = batch["x"].to(device)
        dense = batch["dense"].to(device)
        vmask = batch["vessel_mask"].to(device)
        pred, _ = model(x, return_aux=True)
        for i in range(x.size(0)):
            rels_full.append(rel_err_masked(pred[i], dense[i]))
            vm = vmask[i].unsqueeze(0).expand(3, -1, -1, -1)
            rels_in.append(rel_err_masked(pred[i], dense[i], vm))
    return {
        "vel_rel": float(np.mean(rels_full)),
        "vel_rel_inside_vessel": float(np.mean(rels_in)),
        "score": -float(np.mean(rels_in)),
    }


def main():
    args = parse_args()
    set_seed(42)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_ds = RacletteVolumeDataset(args.cache, "train", args.keep_frac, augment=True)
    val_ds = RacletteVolumeDataset(args.cache, "val", args.keep_frac, augment=False)
    _, d, h, w = train_ds.velocity.shape[1:]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    model = Spatial3DFluidHNFReconstructor(
        d=d, h=h, w=w, in_channels=4, embed_dim=48, kernel_size=5,
        predict_eta=False, use_rotation=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_score, best_path = float("-inf"), out / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"ep{epoch}", leave=False):
            x, dense = batch["x"].to(device), batch["dense"].to(device)
            mask, vmask = batch["mask"].to(device), batch["vessel_mask"].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(x, return_aux=True)
            loss, _ = velocity_recon_loss(pred, dense, mask, region_mask=vmask.unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        val_m = evaluate(model, val_loader, device)
        print(f"[raclette-3d] ep{epoch} inside={val_m['vel_rel_inside_vessel']:.4f}", flush=True)
        if val_m["score"] >= best_score:
            best_score = val_m["score"]
            torch.save({"state_dict": model.state_dict(), "val_metrics": val_m, "args": vars(args), "grid": [d, h, w]}, best_path)
    with (out / "summary.json").open("w") as f:
        json.dump({"best_val": json.loads(json.dumps({"score": best_score}))}, f)
    print(f"[raclette-3d] done {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
