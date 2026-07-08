#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare HNF inversion vs classical baselines on the same synthetic problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hnf.inv_plot import perturb_initial
from hnf.inversion_1d import default_station_distances, default_synth_model, synthesize_travel_times
from hnf.inversion_baselines import run_all_baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inversion baseline comparison")
    p.add_argument("--output-dir", default="outputs/inv_compare_baselines")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def plot_comparison(results, out_path: Path) -> None:
    names = [r.name for r in results]
    vp_rmse = [r.rmse["vp_rmse"] for r in results]
    vs_rmse = [r.rmse["vs_rmse"] for r in results]
    misfit = [r.time_misfit for r in results]
    wall = [r.wall_sec for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = range(len(names))
    axes[0].bar(x, vp_rmse, color="C0")
    axes[0].set_title("Vp RMSE (km/s)")
    axes[1].bar(x, vs_rmse, color="C1")
    axes[1].set_title("Vs RMSE (km/s)")
    axes[2].bar(x, misfit, color="C2")
    axes[2].set_title("Travel-time misfit")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(x, wall, color="C3")
    ax2.set_title("Wall time (s)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_path.with_name("wall_time.png"), dpi=160)
    plt.close(fig2)


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

    results = run_all_baselines(
        true_model, vp_init, vs_init, q_init,
        args.source_depth, distances, obs, steps=args.steps,
    )

    rows = []
    for r in results:
        rows.append({
            "method": r.name,
            "vp_rmse": r.rmse["vp_rmse"],
            "vs_rmse": r.rmse["vs_rmse"],
            "time_misfit": r.time_misfit,
            "wall_sec": r.wall_sec,
            "final_loss": r.history[-1] if r.history else {},
            "recovered_vp": r.earth.vp.detach().cpu().tolist(),
            "recovered_vs": r.earth.vs.detach().cpu().tolist(),
        })

    report = {
        "noise_std_s": args.noise_std,
        "n_stations": args.n_stations,
        "true_vp": true_model.vp.cpu().tolist(),
        "true_vs": true_model.vs.cpu().tolist(),
        "init_vp": vp_init.cpu().tolist(),
        "methods": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    plot_comparison(results, out_dir / "comparison.png")
    print(json.dumps(rows, indent=2))
    print(f"[inv_compare] -> {out_dir}")


if __name__ == "__main__":
    main()
