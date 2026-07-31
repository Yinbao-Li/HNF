#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full anisotropic rheology training + SOTA baseline comparison board.

Protocol: fixed anisotropic Prony material (dim=2, K=2), varied loading
protocols. Primary metric = test stress relative L2 error.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import json
import time
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.rheo_baselines import build_baseline
from hnf.rheo_dataset import RheoMemoryDataset, RheoMemoryModel, rheo_memory_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aniso rheo full train + SOTA board")
    p.add_argument("--output-dir", default="outputs/rheo/aniso_sota_full")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lr-nn", type=float, default=1e-3, help="LR for LSTM/TCN/FIR")
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-modes", type=int, default=2)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--param-weight", type=float, default=0.5)
    p.add_argument("--param-reg", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument(
        "--models",
        default="pnf_aniso,diagonal_prony,isotropic_prony,lstm,tcn,linear_fir",
        help="Comma-separated model keys",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _match_modes(pred_lam, pred_g, gt_lam, gt_g):
    k = len(gt_lam)
    if k == 1:
        return pred_lam, pred_g
    best, best_cost = None, 1e18
    for perm in permutations(range(k)):
        pl = pred_lam[list(perm)]
        pg = pred_g[list(perm)]
        cost = np.mean((np.log(pl + 1e-8) - np.log(gt_lam + 1e-8)) ** 2)
        if cost < best_cost:
            best_cost = cost
            best = (pl, pg)
    return best


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    material_gt: dict | None = None,
) -> dict[str, float]:
    model.eval()
    rels, mses = [], []
    for batch in loader:
        gd = batch["gammadot"].to(device)
        st = batch["stress"].to(device)
        dt = batch["dt"].to(device)
        pred = model(gd, dt)
        err = (pred - st).pow(2)
        # relative L2 per sample
        for i in range(gd.size(0)):
            num = (pred[i] - st[i]).pow(2).sum().sqrt()
            den = st[i].pow(2).sum().sqrt().clamp_min(1e-8)
            rels.append(float((num / den).item()))
            mses.append(float(err[i].mean().item()))

    out = {
        "stress_mse": float(np.mean(mses)),
        "stress_rel": float(np.mean(rels)),
        "stress_rel_median": float(np.median(rels)),
        "stress_rel_p90": float(np.percentile(rels, 90)),
    }

    # Optional λ / A recovery for Prony-family models
    if material_gt is not None and hasattr(model, "kernel"):
        params = model.collect_params() if hasattr(model, "collect_params") else model.kernel.collect_params()
        if "lambda" in params:
            pred_lam = np.asarray(params["lambda"], dtype=np.float64)
            gt_lam = np.asarray(material_gt["lambda"], dtype=np.float64)
            if "G" in params:
                pred_g = np.asarray(params["G"], dtype=np.float64)
                gt_g = np.asarray(material_gt["G"], dtype=np.float64).reshape(-1)[: len(pred_g)]
            elif "A" in params:
                A = np.asarray(params["A"], dtype=np.float64)
                d = int(params.get("dim", 2))
                pred_g = A.reshape(len(pred_lam), d, d).diagonal(axis1=1, axis2=2).mean(1)
                gt_g = np.asarray(material_gt["G"], dtype=np.float64)
                if gt_g.ndim > 1:
                    gt_g = gt_g.mean(axis=-1)
            else:
                pred_g = gt_g = np.ones_like(pred_lam)
            pl, pg = _match_modes(pred_lam, pred_g, gt_lam, gt_g)
            out["lambda_rel"] = float(np.mean(np.abs(pl - gt_lam) / np.maximum(gt_lam, 1e-6)))
            out["G_rel"] = float(np.mean(np.abs(pg - gt_g) / np.maximum(np.abs(gt_g), 1e-6)))
    elif material_gt is not None and hasattr(model, "channels"):
        # diagonal: average channel λ
        ch_lams = []
        for c in model.channels:
            ch_lams.append(np.asarray(c.collect_params()["lambda"], dtype=np.float64))
        pred_lam = np.mean(ch_lams, axis=0)
        gt_lam = np.asarray(material_gt["lambda"], dtype=np.float64)
        pl, _ = _match_modes(pred_lam, pred_lam, gt_lam, gt_lam)
        out["lambda_rel"] = float(np.mean(np.abs(pl - gt_lam) / np.maximum(gt_lam, 1e-6)))
    return out


def build_model(key: str, args: argparse.Namespace) -> nn.Module:
    key = key.strip().lower()
    lam0 = [0.4, 4.0][: args.n_modes]
    while len(lam0) < args.n_modes:
        lam0.append(10.0 ** (len(lam0) - 1))
    g0 = [1.0, 0.7][: args.n_modes]
    while len(g0) < args.n_modes:
        g0.append(0.4)

    if key in {"pnf_aniso", "aniso", "pnf"}:
        return RheoMemoryModel(
            n_modes=args.n_modes,
            dim=args.dim,
            anisotropic=True,
            lambda_init=lam0,
            g_init=g0,
        )
    if key in {"isotropic_prony", "isotropic"}:
        return build_baseline("isotropic", dim=args.dim, n_modes=args.n_modes)
    if key in {"diagonal_prony", "diagonal"}:
        return build_baseline("diagonal", dim=args.dim, n_modes=args.n_modes)
    if key == "lstm":
        return build_baseline("lstm", dim=args.dim, hidden=64)
    if key == "tcn":
        return build_baseline("tcn", dim=args.dim, channels=48)
    if key in {"linear_fir", "fir"}:
        return build_baseline("fir", dim=args.dim, memory=64)
    raise ValueError(key)


def is_prony_family(key: str) -> bool:
    return key.strip().lower() in {
        "pnf_aniso",
        "aniso",
        "pnf",
        "isotropic_prony",
        "isotropic",
        "diagonal_prony",
        "diagonal",
    }


def train_one(
    key: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    material_gt: dict,
    out_root: Path,
) -> dict:
    model = build_model(key, args).to(device)
    lr = args.lr if is_prony_family(key) else args.lr_nn
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    use_param = is_prony_family(key) and key.strip().lower() in {"pnf_aniso", "aniso", "pnf"}
    best = {"stress_rel": 1e9}
    history = []
    t0 = time.time()
    tag = key.strip().lower()
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"{tag} ep{epoch}", leave=False)
        for batch in pbar:
            gd = batch["gammadot"].to(device)
            st = batch["stress"].to(device)
            dt = batch["dt"].to(device)
            pred = model(gd, dt)
            if use_param and hasattr(model, "kernel"):
                loss, stats = rheo_memory_loss(
                    pred,
                    st,
                    param_reg=args.param_reg,
                    freq_weight=0.0,
                    param_weight=args.param_weight,
                    kernel=model.kernel,
                    gt_lambda=batch["lambda"].to(device),
                    gt_G=batch["G"].to(device),
                    gt_g_inf=batch["g_inf"].to(device),
                )
            else:
                loss = (pred - st).pow(2).mean()
                stats = {"loss": float(loss.detach().item()), "mse": float(loss.detach().item())}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(stats["loss"])
            pbar.set_postfix(loss=f"{stats['loss']:.4g}")
        sched.step()

        val_m = evaluate(model, val_loader, device, material_gt=material_gt if use_param else None)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"[{tag} ep{epoch:03d}] loss={row['train_loss']:.4g} "
            f"val_rel={val_m['stress_rel']:.4f}",
            flush=True,
        )
        if val_m["stress_rel"] < best["stress_rel"]:
            best = {**val_m, "epoch": epoch}
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "key": tag, "val": val_m},
                out_dir / "best.pt",
            )

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, device, material_gt=material_gt if use_param else None)
    params = model.collect_params() if hasattr(model, "collect_params") else {}
    summary = {
        "model": tag,
        "n_params": _n_params(model),
        "best_val": best,
        "test": test_m,
        "params": params,
        "elapsed_sec": time.time() - t0,
        "lr": lr,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"model": tag, "test": test_m, "n_params": summary["n_params"]}, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds_kw = dict(
        n_steps=args.n_steps,
        dt=args.dt,
        n_modes=args.n_modes,
        dim=args.dim,
        anisotropic=True,
        noise_std=args.noise_std,
        seed=args.seed,
        fixed_material=True,
    )
    train_ds = RheoMemoryDataset("train", n_samples=args.n_train, **ds_kw)
    val_ds = RheoMemoryDataset("val", n_samples=args.n_val, **ds_kw)
    test_ds = RheoMemoryDataset("test", n_samples=args.n_test, **ds_kw)
    material_gt = {
        "lambda": train_ds.material_lambdas,
        "G": np.asarray(train_ds.material_weights).tolist(),
        "g_inf": train_ds.material_g_inf,
    }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    rows = []
    for key in keys:
        print(f"\n======== training {key} ========", flush=True)
        summary = train_one(key, args, train_loader, val_loader, test_loader, device, material_gt, out_root)
        rows.append(summary)

    rows_sorted = sorted(rows, key=lambda r: r["test"]["stress_rel"])
    board = {
        "protocol": {
            "anisotropic": True,
            "dim": args.dim,
            "n_modes": args.n_modes,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "noise_std": args.noise_std,
            "material_gt": material_gt,
            "epochs": args.epochs,
            "device": str(device),
        },
        "rows": [
            {
                "model": r["model"],
                "n_params": r["n_params"],
                "stress_rel": r["test"]["stress_rel"],
                "stress_rel_median": r["test"]["stress_rel_median"],
                "stress_rel_p90": r["test"]["stress_rel_p90"],
                "stress_mse": r["test"]["stress_mse"],
                "lambda_rel": r["test"].get("lambda_rel"),
                "G_rel": r["test"].get("G_rel"),
                "best_epoch": r["best_val"].get("epoch"),
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in rows_sorted
        ],
        "best": rows_sorted[0]["model"] if rows_sorted else None,
    }
    (out_root / "BOARD.json").write_text(json.dumps(board, indent=2))

    lines = [
        "# Anisotropic rheology — full train vs SOTA baselines",
        "",
        f"- Material GT: λ={material_gt['lambda']}, G/A_diag={material_gt['G']}",
        f"- Protocol: dim={args.dim}, K={args.n_modes}, T={args.n_steps}, "
        f"train/val/test={args.n_train}/{args.n_val}/{args.n_test}, epochs={args.epochs}",
        "",
        "| Model | stress_rel ↓ | median | p90 | λ_rel | n_params |",
        "|-------|-------------:|-------:|----:|------:|---------:|",
    ]
    for r in board["rows"]:
        lam = "—" if r["lambda_rel"] is None else f"{r['lambda_rel']:.4f}"
        lines.append(
            f"| **{r['model']}** | {r['stress_rel']:.4f} | {r['stress_rel_median']:.4f} | "
            f"{r['stress_rel_p90']:.4f} | {lam} | {r['n_params']} |"
            if r["model"] == board["best"]
            else f"| {r['model']} | {r['stress_rel']:.4f} | {r['stress_rel_median']:.4f} | "
            f"{r['stress_rel_p90']:.4f} | {lam} | {r['n_params']} |"
        )
    lines += [
        "",
        f"**Best stress predictor:** `{board['best']}`",
        "",
        "Notes:",
        "- `pnf_aniso`: anisotropic Prony / Boltzmann memory (SPD modal weights A_k).",
        "- `diagonal_prony`: independent per-channel Maxwell (common rheology simplification).",
        "- `isotropic_prony`: shared scalar G_k (misspecified under anisotropy).",
        "- `lstm` / `tcn`: black-box sequence SOTA proxies (not physically identifiable).",
        "- `linear_fir`: classical finite causal linear memory.",
    ]
    (out_root / "BOARD.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[rheo-sota] → {out_root / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
