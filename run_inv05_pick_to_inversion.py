#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inv05: connect synthetic P/S picks to 1D layered inversion.

Pipeline:
  1) True Earth model -> synthetic P/S travel times (as if from run20 picks)
  2) Add pick noise (simulates picking errors)
  3) Invert vp/vs from noisy picks at sparse stations

Usage:
    python run_inv05_pick_to_inversion.py
    python run_inv05_pick_to_inversion.py --pick-noise-std 0.03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hnf.inv_plot import perturb_initial, plot_misfit, plot_velocity_profiles
from hnf.inversion_1d import (
    default_station_distances,
    default_synth_model,
    invert_layered_1d,
    model_rmse,
    synthesize_travel_times,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="inv05: picks -> 1D inversion")
    p.add_argument("--output-dir", default="outputs/inv05_pick_to_inversion")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--pick-noise-std", type=float, default=0.02, help="P/S time noise (s)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_model = default_synth_model(device)
    distances = default_station_distances(device, args.n_stations)
    clean = synthesize_travel_times(true_model, args.source_depth, distances)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    obs = {
        "tp": clean["tp"] + args.pick_noise_std * torch.randn(clean["tp"].shape, generator=gen, device=device),
        "ts": clean["ts"] + args.pick_noise_std * torch.randn(clean["ts"].shape, generator=gen, device=device),
    }

    vp_init, vs_init, q_init = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=args.seed + 1, q_scale=1.0
    )
    recovered, history = invert_layered_1d(
        depths=true_model.depths,
        vp_init=vp_init,
        vs_init=vs_init,
        q_init=q_init,
        source_depth=args.source_depth,
        receiver_distances=distances,
        obs=obs,
        steps=args.steps,
        lr=0.06,
        verbose=True,
    )
    rec = recovered.earth
    init_earth = type(true_model)(depths=true_model.depths, vp=vp_init, vs=vs_init, q=q_init)
    report = {
        "pick_noise_std_s": args.pick_noise_std,
        "source_depth_km": args.source_depth,
        "receiver_distances_km": distances.cpu().tolist(),
        "obs_tp": obs["tp"].cpu().tolist(),
        "obs_ts": obs["ts"].cpu().tolist(),
        "true_tp": clean["tp"].cpu().tolist(),
        "true_ts": clean["ts"].cpu().tolist(),
        "rmse_init": model_rmse(true_model, init_earth),
        "rmse_recovered": model_rmse(true_model, rec),
        "final_loss": history[-1],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    plot_velocity_profiles(
        true_model.depths, true_model.vp, true_model.vs,
        vp_init, vs_init, rec.vp, rec.vs,
        out_dir / "velocity_profiles.png",
        title="inv05: picks -> inversion",
    )
    plot_misfit(history, out_dir / "misfit_curve.png")
    print(json.dumps(report, indent=2))
    print(f"[inv05] -> {out_dir}")


if __name__ == "__main__":
    main()
