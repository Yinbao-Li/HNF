#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-scale Ambon (Indonesia) cross-region travel-time inversion.

What this can do without waveforms:
  - real Ambon station geometry + catalog hypocenters
  - VELEST layered true model from the Mendeley release
  - synthetic P/S travel times via ray tracing (+ noise)
  - multi-event method comparison (HNF-Adam vs classical baselines)

This fills the "cross-region geometry / TT generalization" half of Fig5.
Picking zero-shot still needs waveform data (not in this catalog).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hnf.ambon_data import (
    haversine_km,
    load_ambon_events,
    load_ambon_stations,
    load_ambon_velocity_model,
)
from hnf.inv_plot import perturb_initial
from hnf.inversion_1d import synthesize_travel_times
from hnf.inversion_baselines import run_all_baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ambon cross-region TT inversion (paper)")
    p.add_argument("--output-dir", default="outputs/paper_ambon_cross_region")
    p.add_argument("--n-events", type=int, default=64)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max-distance-km", type=float, default=120.0)
    p.add_argument("--min-stations", type=int, default=6)
    return p.parse_args()


def select_events(events, stations, n_events: int, seed: int, max_dist: float, min_sta: int):
    rng = np.random.default_rng(seed)
    eligible = []
    for i, ev in enumerate(events):
        dists = [haversine_km(ev.longitude, ev.latitude, s.longitude, s.latitude) for s in stations]
        keep = [d for d in dists if d <= max_dist]
        if len(keep) >= min_sta and 1.0 <= ev.depth_km <= 40.0:
            eligible.append(i)
    if len(eligible) < n_events:
        chosen = eligible
    else:
        chosen = sorted(rng.choice(eligible, size=n_events, replace=False).tolist())
    return chosen


def run_one_event(
    ev,
    stations,
    true_model,
    *,
    noise_std: float,
    steps: int,
    seed: int,
    max_dist: float,
):
    pairs = []
    for s in stations:
        d = haversine_km(ev.longitude, ev.latitude, s.longitude, s.latitude)
        if d <= max_dist:
            pairs.append((s, d))
    pairs.sort(key=lambda t: t[1])
    dists = torch.tensor([d for _, d in pairs], dtype=torch.float32)
    codes = [s.code for s, _ in pairs]
    source_depth = max(float(ev.depth_km), 1.0)

    obs = synthesize_travel_times(
        true_model,
        source_depth=source_depth,
        receiver_distances=dists,
        noise_std=noise_std,
        seed=seed,
    )
    vp_init, vs_init, q_init = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=seed + 17, q_scale=1.0
    )
    results = run_all_baselines(
        true_model, vp_init, vs_init, q_init,
        source_depth, dists, obs, steps=steps,
    )
    rows = []
    for r in results:
        rows.append({
            "method": r.name,
            "vp_rmse": float(r.rmse["vp_rmse"]),
            "vs_rmse": float(r.rmse["vs_rmse"]),
            "time_misfit": float(r.time_misfit),
            "wall_sec": float(r.wall_sec),
        })
    meta = {
        "longitude": ev.longitude,
        "latitude": ev.latitude,
        "depth_km": source_depth,
        "n_stations": len(codes),
        "station_codes": codes,
        "mean_distance_km": float(dists.mean()),
        "max_distance_km": float(dists.max()),
    }
    return meta, rows


def is_success(row: dict, vp_max: float = 3.0, tt_max: float = 1.0) -> bool:
    vp = float(row["vp_rmse"])
    tt = float(row["time_misfit"])
    return bool(np.isfinite(vp) and np.isfinite(tt) and vp <= vp_max and tt <= tt_max)


def summarize(method_rows: list[dict], vp_max: float = 3.0, tt_max: float = 1.0) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in method_rows:
        by.setdefault(r["method"], []).append(r)
    out = []
    for method, rows in by.items():
        ok = [r for r in rows if is_success(r, vp_max=vp_max, tt_max=tt_max)]
        vp_all = np.array([r["vp_rmse"] for r in rows], dtype=np.float64)
        vp_ok = np.array([r["vp_rmse"] for r in ok], dtype=np.float64) if ok else np.array([])
        vs_ok = np.array([r["vs_rmse"] for r in ok], dtype=np.float64) if ok else np.array([])
        tt_ok = np.array([r["time_misfit"] for r in ok], dtype=np.float64) if ok else np.array([])
        out.append({
            "method": method,
            "n": len(rows),
            "n_success": len(ok),
            "success_rate": float(len(ok) / max(len(rows), 1)),
            "vp_rmse_median_all": float(np.nanmedian(vp_all)),
            "vp_rmse_mean_success": float(vp_ok.mean()) if len(vp_ok) else float("nan"),
            "vp_rmse_median_success": float(np.median(vp_ok)) if len(vp_ok) else float("nan"),
            "vp_rmse_std_success": float(vp_ok.std()) if len(vp_ok) else float("nan"),
            "vs_rmse_mean_success": float(vs_ok.mean()) if len(vs_ok) else float("nan"),
            "time_misfit_mean_success": float(tt_ok.mean()) if len(tt_ok) else float("nan"),
            "wall_sec_mean": float(np.mean([r["wall_sec"] for r in rows])),
            "success_rule": f"finite and vp_rmse<={vp_max} and time_misfit<={tt_max}",
        })
    # rank by success rate then median success Vp RMSE
    out.sort(key=lambda r: (-r["success_rate"], r["vp_rmse_median_success"] if np.isfinite(r["vp_rmse_median_success"]) else 1e9))
    return out


def plot_board(
    stations,
    event_metas: list[dict],
    summary: list[dict],
    method_rows: list[dict],
    true_vp: list[float],
    out_dir: Path,
) -> dict[str, str]:
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)

    # A: map
    ax = axes[0, 0]
    ax.scatter([s.longitude for s in stations], [s.latitude for s in stations],
               c="C3", marker="^", s=70, label="stations", zorder=3)
    ax.scatter([e["longitude"] for e in event_metas], [e["latitude"] for e in event_metas],
               c="C0", s=28, alpha=0.75, label="events", zorder=2)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("A. Ambon geometry (catalog events + stations)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # B: success-rate + median Vp RMSE on successes
    ax = axes[0, 1]
    names = [s["method"] for s in summary]
    rates = [100.0 * s["success_rate"] for s in summary]
    colors = ["C2" if "HNF" in n else "C0" for n in names]
    ax.bar(range(len(names)), rates, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"B. Stable recoveries (n={len(event_metas)}; Vp≤3 & TT≤1)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, s in enumerate(summary):
        ax.text(i, rates[i] + 1.5, f"{s['n_success']}/{s['n']}", ha="center", fontsize=7)

    # C: median Vp RMSE among successes
    ax = axes[1, 0]
    meds = [s["vp_rmse_median_success"] if np.isfinite(s["vp_rmse_median_success"]) else 0.0 for s in summary]
    stds = [s["vp_rmse_std_success"] if np.isfinite(s["vp_rmse_std_success"]) else 0.0 for s in summary]
    ax.bar(range(len(names)), meds, yerr=stds, color=colors, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median Vp RMSE (successes)")
    ax.set_title("C. Accuracy on successful inversions")
    ax.grid(True, axis="y", alpha=0.3)

    # D: HNF success vs depth (clip display)
    ax = axes[1, 1]
    hnf = [r for r in method_rows if r["method"] == "HNF-Adam"]
    if hnf:
        ok = [r for r in hnf if is_success(r)]
        bad = [r for r in hnf if not is_success(r)]
        if ok:
            ax.scatter([r["depth_km"] for r in ok], [r["vp_rmse"] for r in ok], s=28, alpha=0.85, c="C2", label="success")
        if bad:
            ax.scatter([r["depth_km"] for r in bad], [min(float(r["vp_rmse"]), 5.0) for r in bad],
                       s=28, alpha=0.55, c="C3", marker="x", label="fail (clipped)")
        ax.axhline(3.0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("source depth (km)")
        ax.set_ylabel("HNF-Adam Vp RMSE")
        ax.set_title("D. HNF stability vs catalog depth")
        ax.set_ylim(0, 5.2)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    fig.suptitle("Fig5 (right): Ambon Indonesia cross-region travel-time generalization", fontsize=13)
    p_board = out_dir / "ambon_cross_region_board.png"
    fig.savefig(p_board, dpi=170)
    plt.close(fig)
    (docs / "fig5_ambon_cross_region.png").write_bytes(p_board.read_bytes())

    # also a compact fig5 companion strip
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].scatter([s.longitude for s in stations], [s.latitude for s in stations], c="C3", marker="^", s=70)
    axes[0].scatter([e["longitude"] for e in event_metas], [e["latitude"] for e in event_metas], c="C0", s=22, alpha=0.75)
    axes[0].set_title("Ambon events / stations")
    axes[0].set_xlabel("lon")
    axes[0].set_ylabel("lat")
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(range(len(names)), rates, color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("success rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Stable recoveries on Ambon geometry")
    axes[1].grid(True, axis="y", alpha=0.3)
    p_strip = out_dir / "ambon_cross_region_strip.png"
    fig.savefig(p_strip, dpi=160)
    plt.close(fig)
    (docs / "fig5_ambon_cross_region_strip.png").write_bytes(p_strip.read_bytes())

    return {
        "board": str(p_board),
        "strip": str(p_strip),
        "fig5_board": "docs/figures/fig5_ambon_cross_region.png",
        "fig5_strip": "docs/figures/fig5_ambon_cross_region_strip.png",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = load_ambon_stations()
    events = load_ambon_events()
    true_model = load_ambon_velocity_model(use_velest=True)
    idxs = select_events(
        events, stations, args.n_events, args.seed, args.max_distance_km, args.min_stations
    )
    print(
        f"[ambon-paper] stations={len(stations)} catalog={len(events)} "
        f"selected={len(idxs)} steps={args.steps}",
        flush=True,
    )

    event_metas = []
    method_rows = []
    for k, idx in enumerate(idxs):
        ev = events[idx]
        meta, rows = run_one_event(
            ev, stations, true_model,
            noise_std=args.noise_std,
            steps=args.steps,
            seed=args.seed + 1000 + k,
            max_dist=args.max_distance_km,
        )
        meta["event_index"] = int(idx)
        event_metas.append(meta)
        for r in rows:
            method_rows.append({**r, **{kk: meta[kk] for kk in (
                "event_index", "depth_km", "mean_distance_km", "n_stations"
            )}})
        if (k + 1) % 8 == 0 or (k + 1) == len(idxs):
            print(f"[ambon-paper] finished {k+1}/{len(idxs)}", flush=True)

    summary = summarize(method_rows)
    figs = plot_board(
        stations, event_metas, summary, method_rows,
        true_model.vp.tolist(), out_dir,
    )

    with (out_dir / "per_event_methods.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(method_rows[0].keys()))
        w.writeheader()
        w.writerows(method_rows)

    report = {
        "dataset": "2019 Ambon aftershocks (Mendeley / BMKG+ITB)",
        "data_root": "external_data/ambon_mendeley",
        "n_catalog_events": len(events),
        "n_stations_total": len(stations),
        "n_selected_events": len(idxs),
        "noise_std": args.noise_std,
        "steps": args.steps,
        "max_distance_km": args.max_distance_km,
        "true_vp": true_model.vp.tolist(),
        "true_vs": true_model.vs.tolist(),
        "true_depths": true_model.depths.tolist(),
        "summary": summary,
        "figures": figs,
        "notes": {
            "scope": "Cross-region travel-time / geometry generalization; not picking zero-shot.",
            "waveforms": "Ambon release is catalog+velocity tables only; P/S times are ray-traced synthetics.",
            "ranking": "Primary: success rate; secondary: median Vp RMSE on successes.",
            "best_method": summary[0]["method"] if summary else None,
            "hnf_note": "HNF-Adam can diverge on some Ambon geometries; report success rate, not raw mean.",
        },
    }
    (out_dir / "ambon_cross_region_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Ambon Cross-Region Travel-Time Generalization",
        "",
        f"- catalog events: {len(events)}",
        f"- selected events: {len(idxs)}",
        f"- stations: {len(stations)} (distance <= {args.max_distance_km} km)",
        f"- noise_std: {args.noise_std} s, steps: {args.steps}",
        "",
        "## Method summary (success = Vp RMSE≤3 and TT≤1)",
    ]
    for s in summary:
        md.append(
            f"- `{s['method']}`: success={s['n_success']}/{s['n']} ({100*s['success_rate']:.0f}%), "
            f"median Vp(success)={s['vp_rmse_median_success']:.3f}, "
            f"median Vp(all)={s['vp_rmse_median_all']:.3f}"
        )
    md += [
        "",
        "## Scope",
        "- Uses real Ambon geometry + VELEST true model",
        "- Travel times are synthetic (catalog has no waveforms)",
        "- Complements STEAD SNR Fig5 as the cross-region TT half",
        "",
        f"## Figures",
        f"- `{Path(figs['board']).name}`",
        f"- `{figs['fig5_board']}`",
    ]
    (out_dir / "ambon_cross_region_report.md").write_text("\n".join(md))
    print(json.dumps({
        "n_events": len(idxs),
        "summary": summary,
        "best": summary[0] if summary else None,
        "report": str(out_dir / "ambon_cross_region_report.json"),
        "figure": figs["fig5_board"],
    }, indent=2))


if __name__ == "__main__":
    main()
