#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain rheology SOTA board: PNF aniso vs RhINN / classical NLS / EUCLID-lite.

This is the *literature-aligned* comparison (not generic LSTM/TCN).
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.rheo_dataset import RheoMemoryDataset, RheoMemoryModel
from hnf.rheo_domain_sota import ClassicalPronyNLS, RhINNMaxwell, SparsePronyLibrary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/rheo/domain_sota")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-train", type=int, default=2048)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=160)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-modes", type=int, default=2)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--nls-max-nfev", type=int, default=80)
    p.add_argument("--nls-fit-batches", type=int, default=48)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def eval_stress(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rels, mses = [], []
    for batch in loader:
        gd = batch["gammadot"].to(device)
        st = batch["stress"].to(device)
        dt = batch["dt"].to(device)
        pred = model(gd, dt)
        for i in range(gd.size(0)):
            num = (pred[i] - st[i]).pow(2).sum().sqrt()
            den = st[i].pow(2).sum().sqrt().clamp_min(1e-8)
            rels.append(float((num / den).item()))
            mses.append(float((pred[i] - st[i]).pow(2).mean().item()))
    return {
        "stress_rel": float(np.mean(rels)),
        "stress_rel_median": float(np.median(rels)),
        "stress_rel_p90": float(np.percentile(rels, 90)),
        "stress_mse": float(np.mean(mses)),
    }


def train_pnf(args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader = loaders
    model = RheoMemoryModel(
        n_modes=args.n_modes, dim=args.dim, anisotropic=True,
        lambda_init=[0.4, 4.0][: args.n_modes], g_init=[1.0] * args.n_modes,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    best = {"stress_rel": 1e9}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"pnf ep{ep}", leave=False):
            gd, st, dt = batch["gammadot"].to(device), batch["stress"].to(device), batch["dt"].to(device)
            pred = model(gd, dt)
            loss = (pred - st).pow(2).mean()
            # light param match
            lam = model.kernel.relaxation_times()
            gt = batch["lambda"][0].to(device)
            lam_s, _ = torch.sort(lam)
            gt_s, _ = torch.sort(gt)
            loss = loss + 0.3 * (torch.log(lam_s) - torch.log(gt_s.clamp_min(1e-6))).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        val = eval_stress(model, val_loader, device)
        print(f"[pnf_aniso ep{ep:03d}] val_rel={val['stress_rel']:.4f}", flush=True)
        if val["stress_rel"] < best["stress_rel"]:
            best = {**val, "epoch": ep}
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["model"])
    test = eval_stress(model, test_loader, device)
    return {
        "model": "pnf_aniso",
        "cite": "this work (anisotropic Prony / Boltzmann memory)",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "best_val": best,
        "test": test,
        "params": model.collect_params(),
        "elapsed_sec": time.time() - t0,
    }


def train_rhinn(args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader = loaders
    model = RhINNMaxwell(dim=args.dim, n_modes=args.n_modes, hidden=64, phys_weight=1.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best = {"stress_rel": 1e9}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"rhinn ep{ep}", leave=False):
            gd, st, dt = batch["gammadot"].to(device), batch["stress"].to(device), batch["dt"].to(device)
            loss, _ = model.loss(gd, st, dt)
            opt.zero_grad(); loss.backward(); opt.step()
        val = eval_stress(model, val_loader, device)
        print(f"[rhinn_maxwell ep{ep:03d}] val_rel={val['stress_rel']:.4f}", flush=True)
        if val["stress_rel"] < best["stress_rel"]:
            best = {**val, "epoch": ep}
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["model"])
    test = eval_stress(model, test_loader, device)
    return {
        "model": "rhinn_maxwell",
        "cite": "Mahmoudabadbozchelou & Jamali, Sci Rep 2021 (RhINN); Maxwell ODE residual",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "best_val": best,
        "test": test,
        "params": model.collect_params(),
        "elapsed_sec": time.time() - t0,
    }


def train_euclid(args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader = loaders
    model = SparsePronyLibrary(dim=args.dim, n_library=24, l1_weight=1e-3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    best = {"stress_rel": 1e9}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"euclid ep{ep}", leave=False):
            gd, st, dt = batch["gammadot"].to(device), batch["stress"].to(device), batch["dt"].to(device)
            pred = model(gd, dt)
            loss, _ = model.loss(pred, st)
            opt.zero_grad(); loss.backward(); opt.step()
        val = eval_stress(model, val_loader, device)
        print(f"[sparse_prony_euclid ep{ep:03d}] val_rel={val['stress_rel']:.4f}", flush=True)
        if val["stress_rel"] < best["stress_rel"]:
            best = {**val, "epoch": ep}
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["model"])
    test = eval_stress(model, test_loader, device)
    return {
        "model": "sparse_prony_euclid",
        "cite": "EUCLID-inspired Prony library + L1 (Marino et al. 2023; reduced form)",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "best_val": best,
        "test": test,
        "params": model.collect_params(),
        "elapsed_sec": time.time() - t0,
    }


def run_classical_nls(args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader = loaders
    model = ClassicalPronyNLS(n_modes=args.n_modes, dim=args.dim, anisotropic=True).to(device)
    t0 = time.time()
    batches = []
    for i, batch in enumerate(train_loader):
        if i >= args.nls_fit_batches:
            break
        batches.append((batch["gammadot"], batch["stress"], batch["dt"]))
    print(f"[classical_nls] fitting on {len(batches)} batches ...", flush=True)
    model.fit_from_loader(batches, max_nfev=args.nls_max_nfev)
    model = model.to(device)
    val = eval_stress(model, val_loader, device)
    test = eval_stress(model, test_loader, device)
    torch.save({"model": model.state_dict()}, out_dir / "best.pt")
    print(f"[classical_nls] val_rel={val['stress_rel']:.4f} test_rel={test['stress_rel']:.4f}", flush=True)
    return {
        "model": "classical_prony_nls",
        "cite": "Classical nonlinear least-squares Prony (rheometry standard)",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "best_val": {**val, "epoch": 0},
        "test": test,
        "params": model.collect_params(),
        "elapsed_sec": time.time() - t0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    kw = dict(
        n_steps=args.n_steps, dt=args.dt, n_modes=args.n_modes, dim=args.dim,
        anisotropic=True, noise_std=args.noise_std, seed=args.seed, fixed_material=True,
    )
    train_ds = RheoMemoryDataset("train", n_samples=args.n_train, **kw)
    val_ds = RheoMemoryDataset("val", n_samples=args.n_val, **kw)
    test_ds = RheoMemoryDataset("test", n_samples=args.n_test, **kw)
    loaders = (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
    )
    material_gt = {
        "lambda": train_ds.material_lambdas,
        "G": np.asarray(train_ds.material_weights).tolist(),
    }

    rows = []
    for name, fn in [
        ("pnf_aniso", train_pnf),
        ("classical_prony_nls", run_classical_nls),
        ("rhinn_maxwell", train_rhinn),
        ("sparse_prony_euclid", train_euclid),
    ]:
        print(f"\n======== {name} ========", flush=True)
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = fn(args, loaders, device, out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        rows.append(summary)

    rows = sorted(rows, key=lambda r: r["test"]["stress_rel"])
    board = {
        "note": (
            "Domain SOTA board (literature-aligned). Prior LSTM/TCN/FIR board is a "
            "generic ML ablation only and should not be cited as rheology SOTA."
        ),
        "protocol": {
            "dim": args.dim, "n_modes": args.n_modes, "n_steps": args.n_steps,
            "n_train": args.n_train, "epochs": args.epochs, "material_gt": material_gt,
        },
        "rows": [
            {
                "model": r["model"],
                "cite": r["cite"],
                "n_params": r["n_params"],
                "stress_rel": r["test"]["stress_rel"],
                "stress_rel_median": r["test"]["stress_rel_median"],
                "stress_rel_p90": r["test"]["stress_rel_p90"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in rows
        ],
        "best": rows[0]["model"],
    }
    (out_root / "BOARD.json").write_text(json.dumps(board, indent=2))
    lines = [
        "# Domain rheology SOTA board (literature-aligned)",
        "",
        board["note"],
        "",
        f"- Material GT: λ={material_gt['lambda']}, A_diag={material_gt['G']}",
        f"- Protocol: dim={args.dim}, K={args.n_modes}, T={args.n_steps}, "
        f"train={args.n_train}, epochs={args.epochs}",
        "",
        "| Model | stress_rel ↓ | median | p90 | params | Reference |",
        "|-------|-------------:|-------:|----:|-------:|-----------|",
    ]
    for r in board["rows"]:
        mark = "**" if r["model"] == board["best"] else ""
        lines.append(
            f"| {mark}{r['model']}{mark} | {r['stress_rel']:.4f} | {r['stress_rel_median']:.4f} | "
            f"{r['stress_rel_p90']:.4f} | {r['n_params']} | {r['cite']} |"
        )
    lines += [
        "",
        f"**Best:** `{board['best']}`",
        "",
        "References:",
        "- RhINN: Mahmoudabadbozchelou & Jamali, *Sci Rep* (2021).",
        "- Fractional RhINN family: Rheologica Acta (2023).",
        "- EUCLID viscoelastic Prony discovery: Marino et al. (2023).",
        "- Classical Prony NLS: standard rheometry identification.",
    ]
    (out_root / "BOARD.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
