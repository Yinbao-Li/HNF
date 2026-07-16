#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Domain-III Stage-1: constitutive family ID + θ recovery."""

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

from hnf.fluid_constitutive_model import (
    CONST_ID_TO_FAMILY,
    ConstitutiveFluidDataset,
    FluidConstitutiveModel,
    constitutive_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fluid Stage-1 constitutive")
    p.add_argument("--output-dir", default="outputs/fluid/stage1_constitutive")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--h", type=int, default=32)
    p.add_argument("--w", type=int, default=32)
    p.add_argument("--keep-frac", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--vel-weight", type=float, default=1.0)
    p.add_argument("--family-weight", type=float, default=1.0)
    p.add_argument("--theta-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="")
    p.add_argument("--principle", default="huygens_fresnel")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_err_tensor(pred: torch.Tensor, gt: torch.Tensor) -> float:
    num = (pred - gt).pow(2).sum().sqrt()
    den = gt.pow(2).sum().sqrt().clamp_min(1e-8)
    return float((num / den).item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    n = 0
    correct = 0
    vel_rels: list[float] = []
    eta_rels: list[float] = []  # theta[0] = eta / eta0
    theta_rels: list[float] = []
    fam_correct = {"newtonian": [0, 0], "carreau": [0, 0]}

    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)
        dense = batch["dense"].to(device)
        y = torch.as_tensor(batch["family_id"], device=device, dtype=torch.long)
        pred_f = out["family_logits"].argmax(dim=-1)
        correct += int((pred_f == y).sum().item())
        n += x.size(0)
        theta_t = batch["theta"].to(device)
        mask = batch["theta_mask"].to(device)
        for i in range(x.size(0)):
            vel_rels.append(rel_err_tensor(out["dense"][i], dense[i]))
            fam = CONST_ID_TO_FAMILY[int(y[i].item())]
            fam_correct[fam][1] += 1
            if int(pred_f[i].item()) == int(y[i].item()):
                fam_correct[fam][0] += 1
            gt0 = float(theta_t[i, 0].item())
            pr0 = float(out["theta"][i, 0].item())
            eta_rels.append(abs(pr0 - gt0) / max(abs(gt0), 1e-6))
            m = mask[i]
            if float(m.sum()) > 0:
                num = ((out["theta"][i] - theta_t[i]) * m).pow(2).sum().sqrt()
                den = (theta_t[i].abs() * m).pow(2).sum().sqrt().clamp_min(1e-8)
                theta_rels.append(float((num / den).item()))

    fam_acc = {
        k: (v[0] / max(v[1], 1)) for k, v in fam_correct.items()
    }
    return {
        "family_acc": correct / max(n, 1),
        "family_acc_newtonian": fam_acc["newtonian"],
        "family_acc_carreau": fam_acc["carreau"],
        "vel_rel": float(np.mean(vel_rels)) if vel_rels else float("nan"),
        "eta_rel": float(np.mean(eta_rels)) if eta_rels else float("nan"),
        "theta_rel": float(np.mean(theta_rels)) if theta_rels else float("nan"),
        "score": correct / max(n, 1) - 0.25 * float(np.mean(eta_rels) if eta_rels else 1.0),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_ds = ConstitutiveFluidDataset(
        "train", args.n_train, args.h, args.w, args.keep_frac, args.seed
    )
    val_ds = ConstitutiveFluidDataset(
        "val", args.n_val, args.h, args.w, args.keep_frac, args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = FluidConstitutiveModel(
        h=args.h,
        w=args.w,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        principle=args.principle,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_path = out / "best.pt"
    print(
        f"[fluid-s1] device={device} params={n_params} "
        f"train={len(train_ds)} val={len(val_ds)} keep={args.keep_frac}",
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
            opt.zero_grad(set_to_none=True)
            out_m = model(x)
            loss, _ = constitutive_loss(
                out_m,
                batch,
                vel_weight=args.vel_weight,
                family_weight=args.family_weight,
                theta_weight=args.theta_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n_seen, 1):.3f}")
        sched.step()
        train_loss = running / max(n_seen, 1)
        val_m = evaluate(model, val_loader, device, args)
        row = {"epoch": float(epoch), "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"[fluid-s1] ep {epoch:03d}  train={train_loss:.4f}  "
            f"fam_acc={val_m['family_acc']:.3f}  eta_rel={val_m['eta_rel']:.3f}  "
            f"theta_rel={val_m['theta_rel']:.3f}  vel_rel={val_m['vel_rel']:.3f}",
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
                    "kernel_params": model.collect_kernel_params(),
                },
                best_path,
            )
            print(
                f"[fluid-s1] saved best → {best_path} "
                f"(fam_acc={val_m['family_acc']:.3f}, eta_rel={val_m['eta_rel']:.3f})",
                flush=True,
            )

    with (out / "history.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history, "best_score": best_score}, f, indent=2)
    print(f"[fluid-s1] done in {time.time() - t0:.1f}s  best_score={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
