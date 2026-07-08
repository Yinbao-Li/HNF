#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inv06: run20 picking -> rho/vp/vs prior -> 1D inversion.

Compare four modes on the same synthetic Earth + multi-station traces:
  1) baseline      — perturb init, true travel times (+ noise) as obs
  2) prior_only    — rho/kernel prior init, same obs
  3) picks_obs     — perturb init, model P/S picks as obs
  4) prior+picks   — rho/kernel prior init + model picks as obs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hnf.inv_plot import perturb_initial, plot_misfit, plot_q_profile, plot_velocity_profiles
from hnf.inversion_1d import (
    LayeredEarth1D,
    default_station_distances,
    default_synth_model,
    invert_layered_1d,
    model_rmse,
    synthesize_travel_times,
)
from hnf.picking_prior import (
    build_picking_prior,
    build_synthetic_prior_fallback,
    load_picking_model_from_checkpoint,
    load_prior_cache,
)
from hnf.synth_waveforms_1d import synthesize_multistation_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="inv06: picking prior -> inversion")
    p.add_argument("--output-dir", default="outputs/inv06_picking_prior")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--trace-noise", type=float, default=0.05)
    p.add_argument("--obs-noise", type=float, default=0.02, help="extra noise on true-time obs")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--synthetic-prior", action="store_true", help="Skip run20; use Q-derived rho fallback")
    p.add_argument("--prior-cache", default="", help="Load pre-exported prior JSON (skip run20 forward)")
    return p.parse_args()


def run_inversion(
    name: str,
    true_model_ref: LayeredEarth1D,
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    distances: torch.Tensor,
    obs: dict[str, torch.Tensor],
    steps: int,
) -> dict:
    recovered, history = invert_layered_1d(
        depths=depths,
        vp_init=vp_init,
        vs_init=vs_init,
        q_init=q_init,
        source_depth=source_depth,
        receiver_distances=distances,
        obs=obs,
        steps=steps,
        lr=0.06,
        verbose=False,
    )
    rec = recovered.earth
    init_earth = type(rec)(depths=depths, vp=vp_init, vs=vs_init, q=q_init)
    true_earth = LayeredEarth1D(depths=depths, vp=true_model_ref.vp, vs=true_model_ref.vs, q=true_model_ref.q)
    return {
        "mode": name,
        "rmse_init": model_rmse(true_earth, init_earth),
        "rmse_recovered": model_rmse(true_earth, rec),
        "final_loss": history[-1],
        "vp_init": vp_init.detach().cpu().tolist(),
        "vs_init": vs_init.detach().cpu().tolist(),
        "q_init": q_init.detach().cpu().tolist(),
        "vp_rec": rec.vp.detach().cpu().tolist(),
        "vs_rec": rec.vs.detach().cpu().tolist(),
        "history": history,
        "rec_earth": rec,
    }


def plot_mode_comparison(rows: list[dict], out_path: Path) -> None:
    names = [r["mode"] for r in rows]
    vp_rmse = [r["rmse_recovered"]["vp_rmse"] for r in rows]
    vs_rmse = [r["rmse_recovered"]["vs_rmse"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(names))
    axes[0].bar(x, vp_rmse, color="C0")
    axes[0].set_title("Vp RMSE after inversion")
    axes[1].bar(x, vs_rmse, color="C1")
    axes[1].set_title("Vs RMSE after inversion")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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
    noisy_true_obs = {
        "tp": clean["tp"] + args.obs_noise * torch.randn(clean["tp"].shape, generator=gen, device=device),
        "ts": clean["ts"] + args.obs_noise * torch.randn(clean["ts"].shape, generator=gen, device=device),
    }

    vp_pert, vs_pert, q_pert = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=args.seed + 1, q_scale=1.0
    )

    ckpt = Path(args.checkpoint)
    prior_cache = Path(args.prior_cache) if args.prior_cache else None
    use_synthetic = args.synthetic_prior or (not ckpt.exists() and prior_cache is None)
    seq_len = args.seq_len

    if prior_cache is not None and prior_cache.exists():
        print(f"[inv06] loading prior cache {prior_cache}", flush=True)
        prior = load_prior_cache(prior_cache, device)
        use_synthetic = False
    elif use_synthetic:
        if not ckpt.exists():
            print(f"[inv06] checkpoint missing ({ckpt}); using synthetic prior fallback", flush=True)
        prior = build_synthetic_prior_fallback(
            true_model, clean["tp"], clean["ts"], vp_perturb_seed=args.seed + 1,
        )
    else:
        model, ckpt_args = load_picking_model_from_checkpoint(ckpt, device, bypass=True)
        seq_len = int(ckpt_args.get("seq_len", args.seq_len))
        x, t, _meta = synthesize_multistation_batch(
            true_model,
            args.source_depth,
            distances,
            seq_len=seq_len,
            noise_std=args.trace_noise,
            seed=args.seed + 10,
        )
        prior = build_picking_prior(
            model, x, t, true_model, clean["tp"], clean["ts"],
            vp_perturb_seed=args.seed + 1,
            infer_seq_len=None,
        )
        del model

    modes = [
        ("baseline", vp_pert, vs_pert, q_pert, noisy_true_obs),
        ("prior_only", prior.vp_init, prior.vs_init, prior.q_init, noisy_true_obs),
        ("picks_obs", vp_pert, vs_pert, q_pert, {"tp": prior.obs_tp, "ts": prior.obs_ts}),
        ("prior+picks", prior.vp_init, prior.vs_init, prior.q_init, {"tp": prior.obs_tp, "ts": prior.obs_ts}),
    ]

    rows = []
    for name, vp_i, vs_i, q_i, obs in modes:
        print(f"[inv06] running {name}...", flush=True)
        res = run_inversion(
            name, true_model, true_model.depths, vp_i, vs_i, q_i,
            args.source_depth, distances, obs, args.steps,
        )
        rows.append(res)
        plot_velocity_profiles(
            true_model.depths, true_model.vp, true_model.vs,
            vp_i, vs_i, res["rec_earth"].vp, res["rec_earth"].vs,
            out_dir / f"velocity_{name}.png",
            title=f"inv06 {name}",
        )
        plot_misfit(res["history"], out_dir / f"misfit_{name}.png")

    plot_q_profile(
        true_model.depths, true_model.q, q_pert, prior.q_init,
        out_dir / "q_prior.png",
    )
    plot_mode_comparison(rows, out_dir / "mode_comparison.png")

    report = {
        "checkpoint": str(ckpt) if ckpt.exists() else None,
        "prior_cache": str(prior_cache) if prior_cache and prior_cache.exists() else None,
        "synthetic_prior": use_synthetic and not (prior_cache and prior_cache.exists()),
        "seq_len": seq_len,
        "trace_noise": args.trace_noise,
        "obs_noise": args.obs_noise,
        "true_vp": true_model.vp.cpu().tolist(),
        "true_vs": true_model.vs.cpu().tolist(),
        "true_q": true_model.q.cpu().tolist(),
        "picking_prior": {
            "kernel_vp": prior.kernel_vp,
            "kernel_vs": prior.kernel_vs,
            "kernel_ratio": prior.kernel_ratio,
            "pick_mae_p_sec": prior.pick_mae_p,
            "pick_mae_s_sec": prior.pick_mae_s,
            "rho_per_layer": prior.rho_per_layer.cpu().tolist(),
            "raw_picks_tp": prior.meta["raw_tp"],
            "raw_picks_ts": prior.meta["raw_ts"],
            "true_tp": clean["tp"].cpu().tolist(),
            "true_ts": clean["ts"].cpu().tolist(),
        },
        "modes": [
            {
                "mode": r["mode"],
                "rmse_init": r["rmse_init"],
                "rmse_recovered": r["rmse_recovered"],
                "final_loss": r["final_loss"],
                "vp_init": r["vp_init"],
                "vp_rec": r["vp_rec"],
            }
            for r in rows
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["modes"], indent=2))
    print(f"[inv06] -> {out_dir}")


if __name__ == "__main__":
    main()
