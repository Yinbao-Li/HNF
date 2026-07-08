#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FWI-lite: 2D acoustic waveform inversion vs travel-time inversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hnf.acoustic_fwi_1d import DirectWaveForward, invert_acoustic_fwi
from hnf.inv_plot import perturb_initial, plot_velocity_profiles
from hnf.inversion_1d import default_station_distances, default_synth_model, model_rmse, synthesize_travel_times
from hnf.inversion_baselines import invert_gauss_newton, invert_hnf_adam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FWI-lite vs travel-time inversion")
    p.add_argument("--output-dir", default="outputs/inv_fwi_lite")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--waveform-noise", type=float, default=0.02)
    p.add_argument("--time-noise", type=float, default=0.02)
    p.add_argument("--fwi-steps", type=int, default=180)
    p.add_argument("--tt-steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_model = default_synth_model(device)
    distances = default_station_distances(device, args.n_stations)
    vp_init, vs_init, q_init = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=args.seed + 1, q_scale=1.0
    )

    print("[fwi-lite] forward true model...", flush=True)
    engine = DirectWaveForward(device=device)
    clean_wf = engine.simulate(true_model, args.source_depth, distances)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    noisy_wf = clean_wf + args.waveform_noise * torch.randn(clean_wf.shape, generator=gen, device=device)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(4, distances.numel())):
        ax.plot(engine.time.cpu().numpy(), noisy_wf[i].cpu().numpy(), alpha=0.8, label=f"rx{i}={distances[i]:.0f}km")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_waveforms.png", dpi=160)
    plt.close(fig)

    print("[fwi-lite] inverting waveforms...", flush=True)
    model, fwi_hist, _ = invert_acoustic_fwi(
        true_model.depths, vp_init, vs_init, q_init,
        true_model, args.source_depth, distances, noisy_wf,
        steps=args.fwi_steps, verbose=True,
    )
    fwi_earth = model.earth
    fwi_rmse = model_rmse(true_model, fwi_earth)

    obs = synthesize_travel_times(
        true_model, args.source_depth, distances,
        noise_std=args.time_noise, seed=args.seed,
    )
    print("[fwi-lite] travel-time inversions...", flush=True)
    hnf_tt = invert_hnf_adam(
        true_model.depths, vp_init, vs_init, q_init,
        args.source_depth, distances, obs, steps=args.tt_steps,
    )
    gn_tt = invert_gauss_newton(
        true_model.depths, vp_init, vs_init, q_init,
        args.source_depth, distances, obs["tp"], obs["ts"],
    )
    hnf_tt.rmse = model_rmse(true_model, hnf_tt.earth)
    gn_tt.rmse = model_rmse(true_model, gn_tt.earth)

    rows = [
        {"method": "FWI-lite (direct-wave)", **fwi_rmse, "waveform_misfit": float(fwi_hist[-1]["waveform"])},
        {"method": "Travel-time HNF-Adam", **hnf_tt.rmse, "time_misfit": hnf_tt.time_misfit},
        {"method": "Travel-time Gauss-Newton", **gn_tt.rmse, "time_misfit": gn_tt.time_misfit},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [r["method"] for r in rows]
    axes[0].bar(range(3), [r["vp_rmse"] for r in rows], color="C0")
    axes[0].set_title("Vp RMSE")
    axes[1].bar(range(3), [r["vs_rmse"] for r in rows], color="C1")
    axes[1].set_title("Vs RMSE")
    for ax in axes:
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["FWI", "TT-Adam", "TT-GN"], fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=160)
    plt.close(fig)

    plot_velocity_profiles(
        true_model.depths, true_model.vp, true_model.vs,
        vp_init, vs_init, fwi_earth.vp, fwi_earth.vs,
        out_dir / "velocity_fwi.png", title="FWI-lite recovered Vp/Vs",
    )

    report = {
        "waveform_noise": args.waveform_noise,
        "time_noise": args.time_noise,
        "true_vp": true_model.vp.cpu().tolist(),
        "init_vp": vp_init.cpu().tolist(),
        "methods": rows,
        "fwi_recovered_vp": fwi_earth.vp.cpu().tolist(),
        "hnf_tt_vp": hnf_tt.earth.vp.cpu().tolist(),
        "gn_tt_vp": gn_tt.earth.vp.cpu().tolist(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"[fwi-lite] -> {out_dir}")


if __name__ == "__main__":
    main()
