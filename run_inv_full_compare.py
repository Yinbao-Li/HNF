#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified inversion comparison:
  - Travel-time baselines (HNF-Adam, GN, L-BFGS, grid)
  - FWI-lite (2D acoustic)
  - inv06 picking-prior modes (baseline / prior / picks / both)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hnf.acoustic_fwi_1d import DirectWaveForward, invert_acoustic_fwi
from hnf.inv_plot import perturb_initial
from hnf.inversion_1d import (
    default_station_distances,
    default_synth_model,
    invert_layered_1d,
    model_rmse,
    synthesize_travel_times,
)
from hnf.inversion_baselines import run_all_baselines
from hnf.picking_prior import (
    build_picking_prior,
    build_synthetic_prior_fallback,
    load_picking_model_from_checkpoint,
    load_prior_cache,
)
from hnf.synth_waveforms_1d import synthesize_multistation_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full inversion + FWI + picking prior comparison")
    p.add_argument("--output-dir", default="outputs/inv_full_compare")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--fwi-steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-fwi", action="store_true")
    p.add_argument("--skip-picking", action="store_true")
    p.add_argument("--prior-cache", default="outputs/inv06_run20_prior/prior_cache.json")
    p.add_argument("--synthetic-prior", action="store_true", help="Use Q-derived prior instead of run20")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_model = default_synth_model(device)
    distances = default_station_distances(device, args.n_stations)
    obs = synthesize_travel_times(
        true_model, args.source_depth, distances, noise_std=0.02, seed=args.seed,
    )
    vp_init, vs_init, q_init = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=args.seed + 1, q_scale=1.0
    )

    all_rows: list[dict] = []

    print("[full] travel-time baselines...", flush=True)
    for r in run_all_baselines(
        true_model, vp_init, vs_init, q_init,
        args.source_depth, distances, obs, steps=args.steps,
    ):
        all_rows.append({
            "group": "travel_time",
            "method": r.name,
            "vp_rmse": r.rmse["vp_rmse"],
            "vs_rmse": r.rmse["vs_rmse"],
            "extra": r.time_misfit,
            "wall_sec": r.wall_sec,
        })

    if not args.skip_fwi:
        print("[full] FWI-lite...", flush=True)
        engine = DirectWaveForward(device=device)
        clean_wf = engine.simulate(true_model, args.source_depth, distances)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed)
        noisy_wf = clean_wf + 0.02 * torch.randn(clean_wf.shape, generator=gen, device=device)
        import time
        t0 = time.perf_counter()
        model, hist, _ = invert_acoustic_fwi(
            true_model.depths, vp_init, vs_init, q_init,
            true_model, args.source_depth, distances, noisy_wf,
            steps=args.fwi_steps, verbose=False,
        )
        rec = model.earth
        rmse = model_rmse(true_model, rec)
        all_rows.append({
            "group": "fwi",
            "method": "FWI-lite (direct-wave)",
            "vp_rmse": rmse["vp_rmse"],
            "vs_rmse": rmse["vs_rmse"],
            "extra": hist[-1]["waveform"],
            "wall_sec": time.perf_counter() - t0,
        })

    if not args.skip_picking:
        ckpt = Path(args.checkpoint)
        prior_cache = Path(args.prior_cache) if getattr(args, "prior_cache", "") else None
        if prior_cache is not None and prior_cache.exists() and not args.synthetic_prior:
            print(f"[full] loading prior cache {prior_cache}", flush=True)
            prior = load_prior_cache(prior_cache, device)
            modes = [
                ("prior_only (run20)", prior.vp_init, prior.vs_init, prior.q_init, obs),
                ("prior+picks (run20)", prior.vp_init, prior.vs_init, prior.q_init,
                 {"tp": prior.obs_tp, "ts": prior.obs_ts}),
            ]
            for name, vp_i, vs_i, q_i, obs_m in modes:
                import time
                t0 = time.perf_counter()
                rec_model, _ = invert_layered_1d(
                    true_model.depths, vp_i, vs_i, q_i,
                    args.source_depth, distances, obs_m,
                    steps=args.steps, lr=0.06, verbose=False,
                )
                rmse = model_rmse(true_model, rec_model.earth)
                all_rows.append({
                    "group": "picking_prior",
                    "method": name,
                    "vp_rmse": rmse["vp_rmse"],
                    "vs_rmse": rmse["vs_rmse"],
                    "extra": prior.pick_mae_p,
                    "wall_sec": time.perf_counter() - t0,
                })
        elif args.synthetic_prior or not ckpt.exists():
            if not ckpt.exists():
                print("[full] checkpoint missing; synthetic prior fallback", flush=True)
            else:
                print("[full] --synthetic-prior: skip run20 load", flush=True)
            from hnf.picking_prior import build_synthetic_prior_fallback
            clean = synthesize_travel_times(true_model, args.source_depth, distances)
            prior = build_synthetic_prior_fallback(
                true_model, clean["tp"], clean["ts"], vp_perturb_seed=args.seed + 1,
            )
            modes = [
                ("prior_only", prior.vp_init, prior.vs_init, prior.q_init, obs),
                ("prior+picks", prior.vp_init, prior.vs_init, prior.q_init,
                 {"tp": prior.obs_tp, "ts": prior.obs_ts}),
            ]
            for name, vp_i, vs_i, q_i, obs_m in modes:
                import time
                t0 = time.perf_counter()
                rec_model, _ = invert_layered_1d(
                    true_model.depths, vp_i, vs_i, q_i,
                    args.source_depth, distances, obs_m,
                    steps=args.steps, lr=0.06, verbose=False,
                )
                rmse = model_rmse(true_model, rec_model.earth)
                all_rows.append({
                    "group": "picking_prior",
                    "method": f"{name} (synthetic)",
                    "vp_rmse": rmse["vp_rmse"],
                    "vs_rmse": rmse["vs_rmse"],
                    "extra": prior.pick_mae_p,
                    "wall_sec": time.perf_counter() - t0,
                })
        else:
            print("[full] picking prior modes (run20)...", flush=True)
            model_pk, ckpt_args = load_picking_model_from_checkpoint(ckpt, device, bypass=True)
            seq_len = int(ckpt_args.get("seq_len", 800))
            x, t, _ = synthesize_multistation_batch(
                true_model, args.source_depth, distances,
                seq_len=seq_len, noise_std=0.05, seed=args.seed + 10,
            )
            clean = synthesize_travel_times(true_model, args.source_depth, distances)
            prior = build_picking_prior(
                model_pk, x, t, true_model, clean["tp"], clean["ts"],
                vp_perturb_seed=args.seed + 1,
            )
            modes = [
                ("prior_only", prior.vp_init, prior.vs_init, prior.q_init, obs),
                ("prior+picks", prior.vp_init, prior.vs_init, prior.q_init,
                 {"tp": prior.obs_tp, "ts": prior.obs_ts}),
            ]
            for name, vp_i, vs_i, q_i, obs_m in modes:
                import time
                t0 = time.perf_counter()
                rec_model, _ = invert_layered_1d(
                    true_model.depths, vp_i, vs_i, q_i,
                    args.source_depth, distances, obs_m,
                    steps=args.steps, lr=0.06, verbose=False,
                )
                rmse = model_rmse(true_model, rec_model.earth)
                all_rows.append({
                    "group": "picking_prior",
                    "method": name,
                    "vp_rmse": rmse["vp_rmse"],
                    "vs_rmse": rmse["vs_rmse"],
                    "extra": prior.pick_mae_p,
                    "wall_sec": time.perf_counter() - t0,
                })

    names = [r["method"] for r in all_rows]
    vp = [r["vp_rmse"] for r in all_rows]
    vs = [r["vs_rmse"] for r in all_rows]
    colors = ["C0" if r["group"] == "travel_time" else "C2" if r["group"] == "fwi" else "C3"
              for r in all_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(names)), vp, color=colors)
    axes[0].set_title("Vp RMSE (lower is better)")
    axes[1].bar(range(len(names)), vs, color=colors)
    axes[1].set_title("Vs RMSE (lower is better)")
    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "full_comparison.png", dpi=160)
    plt.close(fig)

    report = {
        "true_vp": true_model.vp.cpu().tolist(),
        "init_vp": vp_init.cpu().tolist(),
        "init_vp_rmse": float(torch.sqrt(torch.mean((vp_init - true_model.vp) ** 2))),
        "methods": all_rows,
        "legend": {
            "travel_time": "走时反演基线",
            "fwi": "FWI-lite 波形反演",
            "picking_prior": "run20 拾取先验",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(all_rows, indent=2))
    print(f"[full_compare] -> {out_dir}")


if __name__ == "__main__":
    main()
