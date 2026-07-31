#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Domain-III R0/R1: Prony Boltzmann memory kernel (σ = K * γ̇)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.rheo_dataset import RheoMemoryDataset, RheoMemoryModel, rheo_memory_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train rheology Boltzmann memory (Prony)")
    p.add_argument("--output-dir", default="outputs/rheo/memory_r0")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-modes", type=int, default=2)
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--anisotropic", action="store_true")
    p.add_argument("--n-train", type=int, default=2048)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--param-reg", type=float, default=0.01)
    p.add_argument("--freq-weight", type=float, default=0.1)
    p.add_argument("--param-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--tag", default="")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _match_modes(
    pred_lam: np.ndarray,
    pred_g: np.ndarray,
    gt_lam: np.ndarray,
    gt_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k = len(gt_lam)
    if k == 1:
        return pred_lam, pred_g
    best = None
    best_cost = 1e18
    for perm in permutations(range(k)):
        pl = pred_lam[list(perm)]
        pg = pred_g[list(perm)] if pred_g.ndim == 1 else pred_g[list(perm)]
        cost = np.mean((np.log(pl + 1e-8) - np.log(gt_lam + 1e-8)) ** 2)
        if cost < best_cost:
            best_cost = cost
            best = (pl, pg)
    assert best is not None
    return best


def _score(m: dict[str, float]) -> float:
    """Composite: prefer stress fit, then identifiable λ/G."""
    return float(m["stress_rel"] + 0.5 * m["lambda_rel"] + 0.3 * m["G_rel"])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mses: list[float] = []
    rels: list[float] = []
    lam_rels: list[float] = []
    g_rels: list[float] = []

    pred_params = model.collect_params()
    pred_lam = np.asarray(pred_params["lambda"], dtype=np.float64)

    for batch in loader:
        gd = batch["gammadot"].to(device)
        st = batch["stress"].to(device)
        dt = batch["dt"].to(device)
        pred = model(gd, dt)
        err = (pred - st).pow(2).mean(dim=tuple(range(1, pred.dim())))
        mses.extend(err.cpu().tolist())
        for i in range(gd.size(0)):
            num = (pred[i] - st[i]).pow(2).sum().sqrt()
            den = st[i].pow(2).sum().sqrt().clamp_min(1e-8)
            rels.append(float((num / den).item()))

            gt_lam = batch["lambda"][i].numpy()
            gt_g = batch["G"][i].numpy()
            if "G" in pred_params:
                pg = np.asarray(pred_params["G"], dtype=np.float64)
            else:
                A = np.asarray(pred_params["A"], dtype=np.float64)
                d = int(pred_params["dim"])
                pg = A.reshape(len(pred_lam), d, d).diagonal(axis1=1, axis2=2).mean(axis=1)
                if gt_g.size == pred_lam.size * d:
                    gt_g = gt_g.reshape(len(pred_lam), d).mean(axis=1)
            pl, pgg = _match_modes(pred_lam, pg, gt_lam, gt_g if gt_g.ndim == 1 else gt_g)
            lam_rels.append(float(np.mean(np.abs(pl - gt_lam) / np.maximum(gt_lam, 1e-6))))
            g_gt = gt_g if gt_g.ndim == 1 else gt_g
            if np.ndim(g_gt) > 1:
                g_gt = g_gt.mean(axis=-1)
            g_rels.append(float(np.mean(np.abs(pgg - g_gt) / np.maximum(np.abs(g_gt), 1e-6))))

    out = {
        "stress_mse": float(np.mean(mses)) if mses else 0.0,
        "stress_rel": float(np.mean(rels)) if rels else 0.0,
        "lambda_rel": float(np.mean(lam_rels)) if lam_rels else 0.0,
        "G_rel": float(np.mean(g_rels)) if g_rels else 0.0,
    }
    out["score"] = _score(out)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Log-spaced init covering typical R0 decades
    if args.n_modes == 1:
        lam0, g0 = [1.0], [1.0]
    elif args.n_modes == 2:
        lam0, g0 = [0.4, 4.0], [1.0, 0.7]
    else:
        lam0 = [10.0 ** (i - (args.n_modes - 1) / 2.0) for i in range(args.n_modes)]
        g0 = [1.0] * args.n_modes

    ds_kw = dict(
        n_steps=args.n_steps,
        dt=args.dt,
        n_modes=args.n_modes,
        dim=args.dim,
        anisotropic=args.anisotropic,
        noise_std=args.noise_std,
        seed=args.seed,
        fixed_material=True,
    )
    train_ds = RheoMemoryDataset("train", n_samples=args.n_train, **ds_kw)
    val_ds = RheoMemoryDataset("val", n_samples=args.n_val, **ds_kw)
    test_ds = RheoMemoryDataset("test", n_samples=args.n_test, **ds_kw)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = RheoMemoryModel(
        n_modes=args.n_modes,
        dim=args.dim,
        anisotropic=args.anisotropic,
        lambda_init=lam0,
        g_init=g0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best = {"score": 1e9}
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"ep{epoch}", leave=False)
        for batch in pbar:
            gd = batch["gammadot"].to(device)
            st = batch["stress"].to(device)
            dt = batch["dt"].to(device)
            pred = model(gd, dt)
            loss, stats = rheo_memory_loss(
                pred,
                st,
                param_reg=args.param_reg,
                freq_weight=0.0 if args.anisotropic else args.freq_weight,
                param_weight=args.param_weight,
                kernel=model.kernel,
                gt_lambda=batch["lambda"].to(device),
                gt_G=batch["G"].to(device),
                gt_g_inf=batch["g_inf"].to(device),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(stats["loss"])
            pbar.set_postfix(loss=f"{stats['loss']:.4g}")
        sched.step()

        val_m = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            "params": model.collect_params(),
        }
        history.append(row)
        print(
            f"[ep {epoch:03d}] loss={row['train_loss']:.4g} "
            f"val_rel={val_m['stress_rel']:.4f} "
            f"λ_rel={val_m['lambda_rel']:.3f} G_rel={val_m['G_rel']:.3f} "
            f"score={val_m['score']:.3f} params={model.collect_params()}",
            flush=True,
        )
        if val_m["score"] < best["score"]:
            best = {**val_m, "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "val": val_m,
                    "params": model.collect_params(),
                    "material": {
                        "lambda": train_ds.material_lambdas,
                        "G": np.asarray(train_ds.material_weights).tolist(),
                        "g_inf": train_ds.material_g_inf,
                    },
                },
                out / "best.pt",
            )

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, device)
    summary = {
        "tag": args.tag or out.name,
        "best_val": best,
        "test": test_m,
        "params": model.collect_params(),
        "material_gt": ckpt.get("material"),
        "elapsed_sec": time.time() - t0,
        "args": vars(args),
    }
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[rheo] → {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
