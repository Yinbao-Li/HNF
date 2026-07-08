#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inv03: HNF kernel profile regularization for 1D inversion.

Compares:
  A) travel-time only (baseline)
  B) travel-time + HuygensNeuralField smoothness prior

Usage:
    python run_inv03_synth_1d.py
    python run_inv03_synth_1d.py --hnf-weight 0.5 --steps 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hnf.field import HuygensNeuralField
from hnf.inv_plot import perturb_initial, plot_misfit, plot_velocity_profiles
from hnf.inversion_1d import (
    default_station_distances,
    default_synth_model,
    invert_layered_1d,
    model_rmse,
    synthesize_travel_times,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF inv03: kernel-regularized inversion")
    p.add_argument("--output-dir", default="outputs/inv03_synth_1d")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--hnf-weight", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_hnf(device: torch.device) -> HuygensNeuralField:
    return HuygensNeuralField(
        gamma=0.5,
        omega=3.0,
        alpha=1e-2,
        eps=1e-2,
        causal=True,
        wave_speed=1.0,
        learnable_gamma=False,
        learnable_omega=False,
        use_density=False,
    ).to(device)


def run_case(
    name: str,
    true_model,
    distances,
    obs,
    vp_init,
    vs_init,
    q_init,
    args,
    hnf_field=None,
    hnf_weight: float = 0.0,
) -> dict:
    print(f"\n=== {name} (hnf_weight={hnf_weight}) ===", flush=True)
    recovered, history = invert_layered_1d(
        depths=true_model.depths,
        vp_init=vp_init,
        vs_init=vs_init,
        q_init=q_init,
        source_depth=args.source_depth,
        receiver_distances=distances,
        obs=obs,
        steps=args.steps,
        lr=args.lr,
        hnf_field=hnf_field,
        hnf_weight=hnf_weight,
        verbose=True,
    )
    rec = recovered.earth
    return {
        "name": name,
        "hnf_weight": hnf_weight,
        "rmse": model_rmse(true_model, rec),
        "final_loss": history[-1],
        "recovered": {
            "vp": rec.vp.detach().cpu().tolist(),
            "vs": rec.vs.detach().cpu().tolist(),
        },
        "history": history,
        "model": recovered,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_model = default_synth_model(device)
    distances = default_station_distances(device, args.n_stations)
    obs = synthesize_travel_times(
        true_model, args.source_depth, distances,
        noise_std=args.noise_std, seed=args.seed,
    )
    vp_init, vs_init, q_init = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=args.seed + 1, q_scale=1.0
    )

    case_a = run_case(
        "baseline", true_model, distances, obs,
        vp_init, vs_init, q_init, args,
    )
    hnf = build_hnf(device)
    case_b = run_case(
        "hnf_regularized", true_model, distances, obs,
        vp_init, vs_init, q_init, args,
        hnf_field=hnf, hnf_weight=args.hnf_weight,
    )

    init_earth = type(true_model)(depths=true_model.depths, vp=vp_init, vs=vs_init, q=q_init)
    report = {
        "hnf_weight": args.hnf_weight,
        "init_rmse": model_rmse(true_model, init_earth),
        "baseline": {k: v for k, v in case_a.items() if k not in {"history", "model"}},
        "hnf_regularized": {k: v for k, v in case_b.items() if k not in {"history", "model"}},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    for tag, case in [("baseline", case_a), ("hnf_regularized", case_b)]:
        sub = out_dir / tag
        sub.mkdir(exist_ok=True)
        rec = case["model"].earth
        plot_velocity_profiles(
            true_model.depths, true_model.vp, true_model.vs,
            vp_init, vs_init, rec.vp, rec.vs,
            sub / "velocity_profiles.png",
            title=f"{tag}: velocity profiles",
        )
        extra = ["loss_hnf"] if tag == "hnf_regularized" else []
        plot_misfit(case["history"], sub / "misfit_curve.png", extra_keys=extra)

    print(json.dumps({
        "init_rmse": report["init_rmse"],
        "baseline_rmse": case_a["rmse"],
        "hnf_rmse": case_b["rmse"],
    }, indent=2))
    print(f"[inv03] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
