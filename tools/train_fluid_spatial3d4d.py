#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train 3D or 4D spatial HNF fluid reconstructors."""

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

from hnf.fluid_dataset3d import SyntheticFluid3DDataset
from hnf.fluid_dataset4d import SyntheticFluid4DDataset
from hnf.fluid_losses import (
    curl_weight_at_epoch,
    lr_scale_at_epoch,
    normalized_curl_loss,
    rel_err_masked,
    velocity_recon_loss,
)
from hnf.fluid_spatial3d import Spatial3DFluidHNFReconstructor
from hnf.fluid_spatial4d import Spatial4DFluidHNFReconstructor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 3D/4D spatial fluid HNF")
    p.add_argument("--mode", choices=["3d", "4d"], default="3d")
    p.add_argument("--output-dir", default="outputs/fluid/spatial3d_synth")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup-epochs", type=int, default=3)
    p.add_argument("--d", type=int, default=12)
    p.add_argument("--h", type=int, default=12)
    p.add_argument("--w", type=int, default=12)
    p.add_argument("--t-steps", type=int, default=4)
    p.add_argument("--keep-frac", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-val", type=int, default=128)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=48)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eta-weight", type=float, default=0.1)
    p.add_argument("--obs-weight", type=float, default=0.25)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--curl-weight", type=float, default=0.01)
    p.add_argument("--curl-warmup-epochs", type=int, default=8)
    p.add_argument("--no-eta", action="store_true")
    p.add_argument("--families", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_err_volume(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return rel_err_masked(pred, gt)


@torch.no_grad()
def evaluate(model, loader, device, eta_weight: float, predict_eta: bool) -> dict:
    model.eval()
    mse = nn.MSELoss()
    rels, eta_rels, by_family = [], [], {}
    tot_v, tot_eta, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        dense = batch["dense"].to(device)
        pred, aux = model(x, return_aux=True)
        tot_v += float(mse(pred, dense).item()) * x.size(0)
        n += x.size(0)
        for i in range(x.size(0)):
            r = rel_err_volume(pred[i], dense[i])
            rels.append(r)
            fam = str(batch["family"][i])
            by_family.setdefault(fam, []).append(r)
        if predict_eta and "eta" in aux:
            eta_t = torch.as_tensor(batch["eta"], device=device, dtype=pred.dtype)
            tot_eta += float(mse(aux["eta"], eta_t).item()) * x.size(0)
            for i in range(x.size(0)):
                eta_rels.append(abs(float(aux["eta"][i]) - float(eta_t[i])) / max(abs(float(eta_t[i])), 1e-6))
    out = {
        "vel_mse": tot_v / max(n, 1),
        "vel_rel": float(np.mean(rels)),
        "eta_rel": float(np.mean(eta_rels)) if eta_rels else float("nan"),
        "score": -float(np.mean(rels)),
    }
    for fam, vals in by_family.items():
        out[f"vel_rel_{fam}"] = float(np.mean(vals))
    return out


def build_loaders(args):
    families = [f.strip() for f in args.families.split(",") if f.strip()] or None
    if args.mode == "3d":
        kw = dict(d=args.d, h=args.h, w=args.w, keep_frac=args.keep_frac, seed=args.seed, families=families)
        train_ds = SyntheticFluid3DDataset("train", args.n_train, **kw)
        val_ds = SyntheticFluid3DDataset("val", args.n_val, **kw)
        test_ds = SyntheticFluid3DDataset("test", args.n_test, **kw)
    else:
        kw = dict(
            t_steps=args.t_steps, d=args.d, h=args.h, w=args.w,
            keep_frac=args.keep_frac, seed=args.seed, families=families,
        )
        train_ds = SyntheticFluid4DDataset("train", args.n_train, **kw)
        val_ds = SyntheticFluid4DDataset("val", args.n_val, **kw)
        test_ds = SyntheticFluid4DDataset("test", args.n_test, **kw)
    nw = 0 if args.batch_size > 2 else 0
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw),
    )


def build_model(args, device):
    predict_eta = not args.no_eta
    if args.mode == "3d":
        model = Spatial3DFluidHNFReconstructor(
            d=args.d, h=args.h, w=args.w, embed_dim=args.embed_dim,
            kernel_size=args.kernel_size, num_layers=args.num_layers,
            dropout=args.dropout, predict_eta=predict_eta, use_rotation=True,
        )
    else:
        model = Spatial4DFluidHNFReconstructor(
            t_steps=args.t_steps, d=args.d, h=args.h, w=args.w,
            embed_dim=args.embed_dim, kernel_size=args.kernel_size,
            num_layers=args.num_layers, dropout=args.dropout,
            predict_eta=predict_eta, use_rotation=True,
        )
    return model.to(device), predict_eta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, test_loader = build_loaders(args)
    model, predict_eta = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()
    base_lrs = [g["lr"] for g in opt.param_groups]
    best_score, best_path = float("-inf"), out / "best.pt"
    history = []

    print(
        f"[fluid-{args.mode}] device={device} params={n_params} "
        f"grid={args.d}x{args.h}x{args.w}" + (f" T={args.t_steps}" if args.mode == "4d" else ""),
        flush=True,
    )

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        lr_mul = lr_scale_at_epoch(epoch, args.lr_warmup_epochs)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_mul
        curl_w = curl_weight_at_epoch(epoch, args.curl_weight, args.curl_warmup_epochs)
        model.train()
        running, n_seen = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs}", leave=False):
            x = batch["x"].to(device)
            dense = batch["dense"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            pred, aux = model(x, return_aux=True)
            loss, _ = velocity_recon_loss(
                pred, dense, mask,
                obs_weight=args.obs_weight, recon_weight=args.recon_weight,
            )
            if predict_eta and "eta" in aux:
                eta_t = torch.as_tensor(batch["eta"], device=device, dtype=pred.dtype)
                loss = loss + args.eta_weight * mse(aux["eta"], eta_t)
            if curl_w > 0:
                loss = loss + curl_w * normalized_curl_loss(pred, dense)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
        sched.step()
        val_m = evaluate(model, val_loader, device, args.eta_weight, predict_eta)
        history.append({"epoch": epoch, "train_loss": running / n_seen, "curl_w": curl_w, **{f"val_{k}": v for k, v in val_m.items()}})
        fam = " ".join(f"{k.split('_')[-1]}={v:.3f}" for k, v in val_m.items() if k.startswith("vel_rel_"))
        print(f"[fluid-{args.mode}] ep {epoch:03d} val_vel_rel={val_m['vel_rel']:.4f} {fam}", flush=True)
        if val_m["score"] >= best_score:
            best_score = val_m["score"]
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_metrics": val_m, "args": vars(args), "n_params": n_params}, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_best = evaluate(model, test_loader, device, args.eta_weight, predict_eta)
    summary = {"mode": args.mode, "best_val": ckpt["val_metrics"], "test_best": test_best, "history": history}
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[fluid-{args.mode}] done {time.time()-t0:.1f}s test_vel_rel={test_best['vel_rel']:.4f}", flush=True)


if __name__ == "__main__":
    main()
