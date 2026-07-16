#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme 1: Bayesian optimization of OBS kernel box params (freeze trunk).

Search multiplicative scales on P/S/MS Huygens (γ, ω, c) with a GP + EI loop.
Objective = pick-only F1 on OBS val split (same disjoint protocol as Step 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import importlib.util
import json
import time

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from tools.analyze_stead_picking import load_model
from tools.obs_kernel_box import (
    GROUP_SCALE_NAMES,
    apply_group_log_scales,
    restore_kernel_raw,
    scales_from_vector,
    snapshot_kernel_state,
    vector_from_scales,
)
from tools.obs_matched_split import load_split_samples
from tools.train_stead_picking import set_seed


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BO kernel-box OBS adapt")
    p.add_argument("--checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--split-json", default="outputs/obs_matched_adapt_split_randoffset/split.json")
    p.add_argument("--output-dir", default="outputs/obs_bayes_kernel_bo")
    p.add_argument("--seq-len", type=int, default=1600)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--n-init", type=int, default=8)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--scale-min", type=float, default=0.35)
    p.add_argument("--scale-max", type=float, default=3.0)
    p.add_argument("--val-max", type=int, default=320, help="Subsample val for fast BO")
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def propose_next(gp: GaussianProcessRegressor, X: np.ndarray, y: np.ndarray, bounds: np.ndarray, rng: np.random.Generator, n_cand: int = 2048) -> np.ndarray:
    cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_cand, bounds.shape[0]))
    mu, sigma = gp.predict(cand, return_std=True)
    ei = expected_improvement(mu, sigma, float(np.max(y)))
    return cand[int(np.argmax(ei))]


@torch.no_grad()
def eval_score(model, samples, device, args, obs_mod) -> dict:
    model.eval()
    # lightweight Namespace for eval_hnf
    po = obs_mod.eval_hnf(
        model,
        samples,
        device,
        args.seq_len,
        args.window_sec,
        args.pick_threshold,
        args.det_threshold,
        args.tol_sec,
        args.batch_size,
    )["pick_only"]
    score = 0.65 * po["p_f1"] + 0.35 * po["s_f1"]
    return {"score": score, "pick_only": po}


def main() -> None:
    args = parse_args()
    if args.n_trials < 50:
        args.n_trials = 50
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    obs_mod = _load_obs_compare_module()

    train_samples, _, split_meta = load_split_samples(args.split_json, "train")
    holdout_samples, _, _ = load_split_samples(args.split_json, "holdout")
    rng = np.random.default_rng(args.seed)
    # BO val: subsample from train pool (disjoint from holdout)
    idxs = rng.choice(len(train_samples), size=min(args.val_max, len(train_samples)), replace=False)
    val_samples = [train_samples[int(i)] for i in idxs]

    model, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    base = snapshot_kernel_state(model)
    (out_dir / "kernel_base.json").write_text(json.dumps(base, indent=2))

    dim = len(GROUP_SCALE_NAMES)
    bounds = np.array([[args.scale_min, args.scale_max]] * dim, dtype=np.float64)
    # Identity start + random Latin-like init
    X_list = [np.ones(dim, dtype=np.float64)]
    for _ in range(args.n_init - 1):
        X_list.append(rng.uniform(bounds[:, 0], bounds[:, 1]))

    history = []
    y_list = []
    t0 = time.time()
    print(
        f"[bayes-bo] val_n={len(val_samples)} holdout_n={len(holdout_samples)} "
        f"dims={GROUP_SCALE_NAMES} protocol={split_meta.get('protocol')}",
        flush=True,
    )

    def evaluate_vec(vec: np.ndarray, tag: str) -> float:
        scales = scales_from_vector(vec)
        restore_kernel_raw(model, base)
        apply_group_log_scales(model, base, scales)
        metrics = eval_score(model, val_samples, device, args, obs_mod)
        row = {
            "tag": tag,
            "scales": scales,
            "score": metrics["score"],
            "pick_only": metrics["pick_only"],
        }
        history.append(row)
        print(
            f"[bayes-bo] {tag} score={metrics['score']:.3f} "
            f"P={metrics['pick_only']['p_f1']:.3f} S={metrics['pick_only']['s_f1']:.3f} "
            f"scales={ {k: round(v,3) for k,v in scales.items()} }",
            flush=True,
        )
        (out_dir / "bo_history.json").write_text(json.dumps(history, indent=2))
        return float(metrics["score"])

    for i, x0 in enumerate(X_list):
        y_list.append(evaluate_vec(x0, f"init{i}"))

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=args.seed)

    n_bo = max(0, args.n_trials - len(X_list))
    for t in range(n_bo):
        gp.fit(X, y)
        x_next = propose_next(gp, X, y, bounds, rng)
        y_next = evaluate_vec(x_next, f"bo{t}")
        X = np.vstack([X, x_next[None, :]])
        y = np.append(y, y_next)

    best_i = int(np.argmax(y))
    best_scales = scales_from_vector(X[best_i])
    restore_kernel_raw(model, base)
    apply_group_log_scales(model, base, best_scales)
    hold = eval_score(model, holdout_samples, device, args, obs_mod)
    # Identity baseline on holdout for comparison
    restore_kernel_raw(model, base)
    base_hold = eval_score(model, holdout_samples, device, args, obs_mod)
    restore_kernel_raw(model, base)
    apply_group_log_scales(model, base, best_scales)

    ckpt = {
        "state_dict": model.state_dict(),
        "args": ckpt_args,
        "bayes_scheme": "kernel_box_bo",
        "best_scales": best_scales,
        "kernel_base": base,
        "val_best_score": float(y[best_i]),
        "holdout_adapted": hold["pick_only"],
        "holdout_identity": base_hold["pick_only"],
        "adapt_args": vars(args),
        "split_json": args.split_json,
    }
    torch.save(ckpt, out_dir / "best.pt")
    report = {
        "scheme": "1_kernel_box_bo",
        "best_scales": best_scales,
        "val_best_score": float(y[best_i]),
        "holdout_bo": hold,
        "holdout_identity": base_hold,
        "n_trials": len(y),
        "elapsed_sec": time.time() - t0,
        "group_scale_names": list(GROUP_SCALE_NAMES),
        "note": "Trunk frozen; only γ/ω/c box scales searched by GP-EI",
    }
    (out_dir / "bayes_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# OBS Bayesian kernel-box BO (scheme 1)",
        "",
        "- Freeze trunk; optimize P/S/MS multiplicative (γ,ω,c) scales",
        f"- trials: {len(y)} | val_best score: {y[best_i]:.3f}",
        f"- holdout identity P/S: {base_hold['pick_only']['p_f1']:.3f} / {base_hold['pick_only']['s_f1']:.3f}",
        f"- holdout BO       P/S: {hold['pick_only']['p_f1']:.3f} / {hold['pick_only']['s_f1']:.3f}",
        f"- best scales: `{json.dumps({k: round(v,3) for k,v in best_scales.items()})}`",
    ]
    (out_dir / "bayes_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
