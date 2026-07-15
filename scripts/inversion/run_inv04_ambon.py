#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inv04: Ambon regional geometry + catalog-driven 1D inversion benchmark.

Uses:
  - VELEST velocity model from Ambon xlsx (true model)
  - 10 local stations + catalog event geometry
  - Synthetic P/S travel times via ray tracing
  - Compare HNF-Adam vs classical baselines

Usage:
    python run_inv04_ambon.py
    python run_inv04_ambon.py --event-index 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hnf.ambon_data import (
    haversine_km,
    load_ambon_events,
    load_ambon_stations,
    load_ambon_velocity_model,
)
from hnf.inv_plot import perturb_initial, plot_velocity_profiles
from hnf.inversion_1d import synthesize_travel_times
from hnf.inversion_baselines import run_all_baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="inv04 Ambon geometry inversion")
    p.add_argument("--output-dir", default="outputs/inv04_ambon")
    p.add_argument("--event-index", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = load_ambon_stations()
    events = load_ambon_events()
    true_model = load_ambon_velocity_model(use_velest=True)
    ev = events[args.event_index % len(events)]

    distances_km = torch.tensor(
        [haversine_km(ev.longitude, ev.latitude, s.longitude, s.latitude) for s in stations],
        dtype=torch.float32,
    )
    source_depth = max(ev.depth_km, 1.0)

    obs = synthesize_travel_times(
        true_model,
        source_depth=source_depth,
        receiver_distances=distances_km,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    vp_init, vs_init, q_init = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=args.seed + 1, q_scale=1.0
    )

    results = run_all_baselines(
        true_model, vp_init, vs_init, q_init,
        source_depth, distances_km, obs, steps=args.steps,
    )

    rows = []
    for r in results:
        rows.append({
            "method": r.name,
            "vp_rmse": r.rmse["vp_rmse"],
            "vs_rmse": r.rmse["vs_rmse"],
            "time_misfit": r.time_misfit,
            "wall_sec": r.wall_sec,
        })
        sub = out_dir / r.name.replace(" ", "_").replace("(", "").replace(")", "")
        sub.mkdir(exist_ok=True)
        plot_velocity_profiles(
            true_model.depths, true_model.vp, true_model.vs,
            vp_init, vs_init, r.earth.vp, r.earth.vs,
            sub / "velocity_profiles.png",
            title=f"inv04 Ambon — {r.name}",
        )

    report = {
        "event": {
            "index": args.event_index,
            "longitude": ev.longitude,
            "latitude": ev.latitude,
            "depth_km": ev.depth_km,
        },
        "n_stations": len(stations),
        "station_codes": [s.code for s in stations],
        "distances_km": distances_km.tolist(),
        "true_vp": true_model.vp.tolist(),
        "true_vs": true_model.vs.tolist(),
        "methods": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[inv04] -> {out_dir}")


if __name__ == "__main__":
    main()
