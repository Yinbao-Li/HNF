#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Domain-III spatial HNF (2D Huygens + rotational sources)."""

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

from hnf.fluid_dataset import SyntheticFluidDataset
from hnf.fluid_losses import (
    curl_weight_at_epoch,
    lr_scale_at_epoch,
    normalized_curl_loss,
    rel_err_masked,
    velocity_recon_loss,
)
from hnf.fluid_spatial import SpatialFluidHNFReconstructor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train spatial fluid HNF reconstructor")
    p.add_argument("--output-dir", default="outputs/fluid/spatial_synth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-warmup-epochs", type=int, default=3)
    p.add_argument("--h", type=int, default=32)
    p.add_argument("--w", type=int, default=32)
    p.add_argument("--keep-frac", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=2048)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=9)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eta-weight", type=float, default=0.1)
    p.add_argument("--obs-weight", type=float, default=0.25)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--curl-weight", type=float, default=0.01, help="max curl loss weight after warmup")
    p.add_argument("--curl-warmup-epochs", type=int, default=10)
    p.add_argument("--no-eta", action="store_true")
    p.add_argument("--no-rotation", action="store_true", help="momentum-only spatial ablation")
    p.add_argument("--families", default="", help="comma-separated subset, e.g. vortex")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="")
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
    eta_weight: float,
) -> dict[str, float]:
    model.eval()
    mse = nn.MSELoss()
    tot_v, tot_eta, n = 0.0, 0.0, 0
    rels: list[float] = []
    eta_rels: list[float] = []
    by_family: dict[str, list[float]] = {}
    for batch in loader:
        x = batch["x"].to(device)
        dense = batch["dense"].to(device)
        mask = batch["mask"].to(device)
        pred, aux = model(x, return_aux=True)
        loss_v = mse(pred, dense)
        tot_v += float(loss_v.item()) * x.size(0)
        n += x.size(0)
        for i in range(x.size(0)):
            r = rel_err_masked(pred[i], dense[i])
            rels.append(r)
            fam = str(batch["family"][i]) if isinstance(batch["family"], (list, tuple)) else str(batch["family"])
            by_family.setdefault(fam, []).append(r)
        if "eta" in aux:
            eta_t = torch.as_tensor(batch["eta"], device=device, dtype=pred.dtype)
            loss_eta = mse(aux["eta"], eta_t)
            tot_eta += float(loss_eta.item()) * x.size(0)
            for i in range(x.size(0)):
                gt = float(eta_t[i].item())
                pr = float(aux["eta"][i].item())
                eta_rels.append(abs(pr - gt) / max(abs(gt), 1e-6))
        del mask
    out = {
        "vel_mse": tot_v / max(n, 1),
        "vel_rel": float(np.mean(rels)) if rels else float("nan"),
        "eta_mse": tot_eta / max(n, 1) if eta_rels else float("nan"),
        "eta_rel": float(np.mean(eta_rels)) if eta_rels else float("nan"),
        "score": -float(np.mean(rels)) if rels else float("-inf"),
    }
    for fam, vals in by_family.items():
        out[f"vel_rel_{fam}"] = float(np.mean(vals))
    out["loss"] = out["vel_mse"] + eta_weight * (out["eta_mse"] if out["eta_mse"] == out["eta_mse"] else 0.0)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    predict_eta = not args.no_eta
    families = [f.strip() for f in args.families.split(",") if f.strip()] or None
    use_rotation = not args.no_rotation

    train_ds = SyntheticFluidDataset(
        split="train", n_samples=args.n_train, h=args.h, w=args.w,
        keep_frac=args.keep_frac, seed=args.seed, families=families,
    )
    val_ds = SyntheticFluidDataset(
        split="val", n_samples=args.n_val, h=args.h, w=args.w,
        keep_frac=args.keep_frac, seed=args.seed, families=families,
    )
    test_ds = SyntheticFluidDataset(
        split="test", n_samples=args.n_test, h=args.h, w=args.w,
        keep_frac=args.keep_frac, seed=args.seed, families=families,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SpatialFluidHNFReconstructor(
        h=args.h, w=args.w, embed_dim=args.embed_dim,
        kernel_size=args.kernel_size, num_layers=args.num_layers,
        dropout=args.dropout, predict_eta=predict_eta, use_rotation=use_rotation,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()
    base_lrs = [g["lr"] for g in opt.param_groups]

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_path = out / "best.pt"
    arch = "spatial+rot" if use_rotation else "spatial"
    print(
        f"[fluid-spatial] arch={arch} device={device} params={n_params} "
        f"train={len(train_ds)} val={len(val_ds)} keep={args.keep_frac} "
        f"families={families or 'all'} k={args.kernel_size} "
        f"mask(obs={args.obs_weight},recon={args.recon_weight}) "
        f"curl_max={args.curl_weight} curl_warmup={args.curl_warmup_epochs}",
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
            opt.zero_grad(set_to_none=True)
            pred, aux = model(x, return_aux=True)
            loss, _ = velocity_recon_loss(
                pred, dense, obs_mask,
                obs_weight=args.obs_weight,
                recon_weight=args.recon_weight,
            )
            if predict_eta and "eta" in aux:
                eta_t = torch.as_tensor(batch["eta"], device=device, dtype=pred.dtype)
                loss = loss + args.eta_weight * mse(aux["eta"], eta_t)
            if curl_w > 0 and use_rotation:
                loss = loss + curl_w * normalized_curl_loss(pred, dense)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.4f}", curl=f"{curl_w:.4f}")
        sched.step()
        train_loss = running / max(n_seen, 1)
        val_m = evaluate(model, val_loader, device, args.eta_weight)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "curl_weight": curl_w,
            **{f"val_{k}": v for k, v in val_m.items()},
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        fam_bits = " ".join(f"{k.split('_')[-1]}={v:.3f}" for k, v in val_m.items() if k.startswith("vel_rel_"))
        print(
            f"[fluid-spatial] ep {epoch:03d}  train={train_loss:.4f}  "
            f"val_vel_rel={val_m['vel_rel']:.4f}  {fam_bits}",
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
                    "arch": arch,
                    "kernel_params": model.collect_kernel_params(),
                },
                best_path,
            )
            print(f"[fluid-spatial] saved best → {best_path} (vel_rel={val_m['vel_rel']:.4f})", flush=True)

    test_m = evaluate(model, test_loader, device, args.eta_weight)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_m_best = evaluate(model, test_loader, device, args.eta_weight)

    summary = {
        "arch": arch,
        "best_val": ckpt["val_metrics"],
        "test_last": test_m,
        "test_best": test_m_best,
        "history": history,
        "families": families or "all",
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[fluid-spatial] done in {time.time() - t0:.1f}s  "
        f"test_best_vel_rel={test_m_best['vel_rel']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
