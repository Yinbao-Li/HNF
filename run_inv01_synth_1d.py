#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inv01: 1D layered synthetic vp/vs inversion benchmark.

Forward: ray-tracing travel times at 5-10 sparse surface receivers.
Inverse: Adam optimization on layer vp/vs from noisy P/S picks.

Usage:
    python run_inv01_synth_1d.py
    python run_inv01_synth_1d.py --n-stations 10 --noise-std 0.02 --steps 800
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
    p = argparse.ArgumentParser(description="HNF 1D synthetic vp/vs inversion (inv01)")
    p.add_argument("--output-dir", default="outputs/inv01_synth_1d")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--noise-std", type=float, default=0.0, help="travel-time noise (s)")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.06)
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
    obs = synthesize_travel_times(
        true_model,
        source_depth=args.source_depth,
        receiver_distances=distances,
        noise_std=args.noise_std,
        seed=args.seed,
    )

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
        obs={"tp": obs["tp"], "ts": obs["ts"]},
        steps=args.steps,
        lr=args.lr,
        verbose=True,
    )

    rec = recovered.earth
    rmse = model_rmse(true_model, rec)
    init_earth = type(true_model)(
        depths=true_model.depths, vp=vp_init, vs=vs_init, q=q_init
    )
    rmse_init = model_rmse(true_model, init_earth)

    report = {
        "n_stations": args.n_stations,
        "source_depth_km": args.source_depth,
        "noise_std_s": args.noise_std,
        "receiver_distances_km": distances.cpu().tolist(),
        "obs_tp": obs["tp"].cpu().tolist(),
        "obs_ts": obs["ts"].cpu().tolist(),
        "true": {
            "vp": true_model.vp.cpu().tolist(),
            "vs": true_model.vs.cpu().tolist(),
            "q": true_model.q.cpu().tolist(),
        },
        "init": {
            "vp": vp_init.cpu().tolist(),
            "vs": vs_init.cpu().tolist(),
        },
        "recovered": {
            "vp": rec.vp.cpu().tolist(),
            "vs": rec.vs.cpu().tolist(),
            "q": rec.q.cpu().tolist(),
        },
        "rmse_init": rmse_init,
        "rmse_recovered": rmse,
        "final_loss": history[-1],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    plot_velocity_profiles(
        true_model.depths,
        true_model.vp,
        true_model.vs,
        vp_init,
        vs_init,
        rec.vp,
        rec.vs,
        out_dir / "velocity_profiles.png",
    )
    plot_misfit(history, out_dir / "misfit_curve.png")

    print(json.dumps({"rmse_init": rmse_init, "rmse_recovered": rmse, "final_loss": history[-1]}, indent=2))
    print(f"[inv01] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
