#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inv02: travel-time + amplitude inversion for vp/vs/Q.

Compares:
  A) travel-time only (inv01 baseline)
  B) travel-time + log-amplitude (enables Q recovery)

Usage:
    python run_inv02_synth_1d.py
    python run_inv02_synth_1d.py --amp-weight 2.0 --steps 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hnf.inv_plot import (
    perturb_initial,
    plot_misfit,
    plot_q_profile,
    plot_velocity_profiles,
)
from hnf.inversion_1d import (
    default_station_distances,
    default_synth_model,
    invert_layered_1d,
    model_rmse,
    synthesize_observations,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF inv02: vp/vs/Q with amplitude")
    p.add_argument("--output-dir", default="outputs/inv02_synth_1d")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--frequencies-hz", default="4,8,12,16", help="comma-separated frequencies")
    p.add_argument("--time-noise-std", type=float, default=0.0)
    p.add_argument("--amp-noise-std", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--amp-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_case(
    name: str,
    true_model,
    distances,
    obs,
    vp_init,
    vs_init,
    q_init,
    args,
    amp_weight: float,
    invert_q: bool,
) -> dict:
    print(f"\n=== {name} (amp_weight={amp_weight}, invert_q={invert_q}) ===", flush=True)
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
        amp_weight=amp_weight,
        invert_q=invert_q,
        frequency_hz=obs.get("frequencies_hz", 8.0),
        verbose=True,
    )
    rec = recovered.earth
    rmse = model_rmse(true_model, rec)
    return {
        "name": name,
        "amp_weight": amp_weight,
        "invert_q": invert_q,
        "rmse": rmse,
        "final_loss": history[-1],
        "recovered": {
            "vp": rec.vp.detach().cpu().tolist(),
            "vs": rec.vs.detach().cpu().tolist(),
            "q": rec.q.detach().cpu().tolist(),
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
    obs = synthesize_observations(
        true_model,
        source_depth=args.source_depth,
        receiver_distances=distances,
        frequency_hz=[float(x) for x in args.frequencies_hz.split(",")],
        time_noise_std=args.time_noise_std,
        amp_noise_std=args.amp_noise_std,
        seed=args.seed,
    )
    vp_init, vs_init, q_init = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=args.seed + 1
    )

    case_a = run_case(
        "time_only", true_model, distances, {"tp": obs["tp"], "ts": obs["ts"]},
        vp_init, vs_init, q_init, args, amp_weight=0.0, invert_q=False,
    )
    case_b = run_case(
        "time_plus_amp", true_model, distances, obs,
        vp_init, vs_init, q_init, args, amp_weight=args.amp_weight, invert_q=True,
    )
    # re-run time_plus_amp with two-stage for actual Q recovery
    print("\n=== time_plus_amp_two_stage ===", flush=True)
    recovered_b, history_b = invert_layered_1d(
        depths=true_model.depths,
        vp_init=vp_init,
        vs_init=vs_init,
        q_init=q_init,
        source_depth=args.source_depth,
        receiver_distances=distances,
        obs=obs,
        steps=args.steps,
        lr=args.lr,
        amp_weight=args.amp_weight,
        invert_q=True,
        frequency_hz=obs.get("frequencies_hz", 8.0),
        two_stage_q=True,
        verbose=True,
    )
    case_b2 = {
        "name": "time_plus_amp_two_stage",
        "rmse": model_rmse(true_model, recovered_b.earth),
        "final_loss": history_b[-1],
        "recovered": {
            "vp": recovered_b.earth.vp.detach().cpu().tolist(),
            "vs": recovered_b.earth.vs.detach().cpu().tolist(),
            "q": recovered_b.earth.q.detach().cpu().tolist(),
        },
        "history": history_b,
        "model": recovered_b,
    }

    init_earth = type(true_model)(depths=true_model.depths, vp=vp_init, vs=vs_init, q=q_init)
    report = {
        "frequency_hz": args.frequencies_hz,
        "time_noise_std": args.time_noise_std,
        "amp_noise_std": args.amp_noise_std,
        "true": {
            "vp": true_model.vp.cpu().tolist(),
            "vs": true_model.vs.cpu().tolist(),
            "q": true_model.q.cpu().tolist(),
        },
        "init_rmse": model_rmse(true_model, init_earth),
        "time_only": {k: v for k, v in case_a.items() if k not in {"history", "model"}},
        "time_plus_amp": {k: v for k, v in case_b.items() if k not in {"history", "model"}},
        "time_plus_amp_two_stage": {k: v for k, v in case_b2.items() if k not in {"history", "model"}},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    for tag, case in [
        ("time_only", case_a),
        ("time_plus_amp", case_b),
        ("time_plus_amp_two_stage", case_b2),
    ]:
        sub = out_dir / tag
        sub.mkdir(exist_ok=True)
        rec = case["model"].earth
        plot_velocity_profiles(
            true_model.depths, true_model.vp, true_model.vs,
            vp_init, vs_init, rec.vp, rec.vs,
            sub / "velocity_profiles.png",
            title=f"{tag}: velocity profiles",
        )
        plot_q_profile(
            true_model.depths, true_model.q, q_init, rec.q,
            sub / "q_profile.png",
        )
        extra = ["loss_amp_p", "loss_amp_s"] if "amp" in tag else []
        plot_misfit(case["history"], sub / "misfit_curve.png", extra_keys=extra)

    print(json.dumps({
        "init_rmse": report["init_rmse"],
        "time_only_rmse": case_a["rmse"],
        "time_plus_amp_rmse": case_b["rmse"],
        "time_plus_amp_two_stage_rmse": case_b2["rmse"],
    }, indent=2))
    print(f"[inv02] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
