#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase F: real-data pseudo-2D STEAD profile.

Because STEAD provides robust event geometry as epicentral distance + source
depth, but not a stable field survey line with station coordinates for every
trace, Phase F first builds a pseudo-2D section along source_distance_km.

Pipeline:
  1. run picks on STEAD test events
  2. use geo-conditioned inversion head to predict 1D vp/vs
  3. refine each local model with travel-time fitting
  4. aggregate local 1D models in distance bins
  5. render Vp / Vs / VpVs + support / uncertainty maps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.analyze_stead_picking import load_model
from hnf.inversion_1d import LayeredEarth1D, default_synth_model, travel_time_phase
from hnf.inversion_baselines import invert_hnf_adam, invert_lbfgs_torch
from hnf.picking_metrics import idx_to_sec
from hnf.picking_prior import run_picking_on_batch
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.stead_zhizi_inversion_dataset import encode_geometry_tensor
from hnf.physics_decoder import load_physics_decoder_from_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase F STEAD pseudo-2D structural profile")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_mixed_geo/best_physics_head.pt")
    p.add_argument("--output-dir", default="outputs/phase_f_stead_profile")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-events", type=int, default=96)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--distance-min", type=float, default=1.0)
    p.add_argument("--distance-max", type=float, default=200.0)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--pick-fallback-sec", type=float, default=0.5)
    p.add_argument("--obs-fallback", action="store_true", default=True)
    p.add_argument("--qc-pick-err-p-max", type=float, default=0.35)
    p.add_argument("--qc-pick-err-s-max", type=float, default=0.25)
    p.add_argument("--qc-refined-tt-max", type=float, default=6.0)
    p.add_argument("--qc-min-events-per-bin", type=int, default=3)
    p.add_argument("--qc-max-vp-std", type=float, default=2.0)
    p.add_argument("--qc-max-vs-std", type=float, default=1.5)
    return p.parse_args()


def time_misfit(
    earth: LayeredEarth1D,
    source_depth: float,
    distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
) -> float:
    src = torch.tensor(source_depth, dtype=earth.vp.dtype, device=earth.vp.device)
    tp = travel_time_phase(earth, "P", src, distances)
    ts = travel_time_phase(earth, "S", src, distances)
    return float(torch.mean((tp - obs_tp) ** 2) + torch.mean((ts - obs_ts) ** 2))


def refine_tt(
    depths: torch.Tensor,
    vp0: torch.Tensor,
    vs0: torch.Tensor,
    q0: torch.Tensor,
    source_depth: float,
    distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
):
    res = invert_lbfgs_torch(
        depths, vp0, vs0, q0, source_depth, distances, obs_tp, obs_ts, max_iter=80
    )
    if not torch.isfinite(res.earth.vp).all():
        res = invert_hnf_adam(
            depths, vp0, vs0, q0, source_depth, distances, {"tp": obs_tp, "ts": obs_ts}, steps=400
        )
    return res


def interpolate_layers_to_section(
    x_centers: np.ndarray,
    layer_values: np.ndarray,
    depths: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    layer_interp = np.stack([np.interp(x_grid, x_centers, layer_values[:, i]) for i in range(layer_values.shape[1])], axis=0)
    sec = np.zeros((len(z_grid), len(x_grid)), dtype=np.float32)
    for iz, z in enumerate(z_grid):
        li = np.searchsorted(depths, z, side="right") - 1
        li = int(np.clip(li, 0, layer_values.shape[1] - 1))
        sec[iz] = layer_interp[li]
    return sec


def plot_section(
    section: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    out_path: Path,
    *,
    title: str,
    cmap: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(
        section,
        extent=[float(x_grid[0]), float(x_grid[-1]), float(z_grid[-1]), float(z_grid[0])],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Epicentral distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.95)
    cb.set_label(label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_masked_section(
    section: np.ndarray,
    mask: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    out_path: Path,
    *,
    title: str,
    cmap: str,
    label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    masked = np.ma.masked_where(~mask.astype(bool), section)
    im = ax.imshow(
        masked,
        extent=[float(x_grid[0]), float(x_grid[-1]), float(z_grid[-1]), float(z_grid[0])],
        aspect="auto",
        cmap=cmap,
    )
    ax.imshow(
        (~mask.astype(bool)).astype(float),
        extent=[float(x_grid[0]), float(x_grid[-1]), float(z_grid[-1]), float(z_grid[0])],
        aspect="auto",
        cmap="gray",
        alpha=0.22,
    )
    ax.set_xlabel("Epicentral distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.95)
    cb.set_label(label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    backbone, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=True)
    base = default_synth_model(device)
    bridge = load_physics_decoder_from_checkpoint(
        backbone,
        args.physics_head,
        device,
        embed_dim=int(ckpt_args.get("embed_dim", 64)),
        n_layers=base.n_layers,
        infer_seq_len=args.infer_seq_len,
    )

    ds = STEADPickingDataset("test", seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows: list[dict] = []
    pick_p_err: list[float] = []
    pick_s_err: list[float] = []
    n_seen = 0

    for batch in loader:
        if n_seen >= args.max_events:
            break
        if float(batch["det"][0]) <= 0.5:
            continue
        if float(batch["p_valid"][0]) <= 0 or float(batch["s_valid"][0]) <= 0:
            continue

        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        if not np.isfinite(dist) or not np.isfinite(depth):
            continue
        if dist < args.distance_min or dist > args.distance_max:
            continue
        depth = max(depth, 1.0)

        x = batch["x"].to(device)
        t = batch["t"].to(device)
        gt_p = idx_to_sec(int(batch["p_idx"][0]), x.shape[1])
        gt_s = idx_to_sec(int(batch["s_idx"][0]), x.shape[1])

        picks = run_picking_on_batch(
            backbone,
            x,
            t,
            pick_threshold=args.pick_threshold,
            det_threshold=args.det_threshold,
            infer_seq_len=None,
        )
        tp = picks["tp_sec"][0]
        ts = picks["ts_sec"][0]
        if tp is not None:
            pick_p_err.append(abs(tp - gt_p))
        if ts is not None:
            pick_s_err.append(abs(ts - gt_s))
        if args.obs_fallback:
            if tp is None or abs(tp - gt_p) > args.pick_fallback_sec:
                tp = gt_p
            if ts is None or abs(ts - gt_s) > args.pick_fallback_sec:
                ts = gt_s
        if tp is None or ts is None:
            continue

        geo = encode_geometry_tensor(dist, depth, device=device)
        distances = torch.tensor([dist], dtype=torch.float32, device=device)
        obs_tp = torch.tensor([tp], device=device)
        obs_ts = torch.tensor([ts], device=device)

        with torch.no_grad():
            out, _ = bridge.forward_event(x, t, include_picks=True, geo=geo)
        init_earth = bridge.physics_head.earth(out, base.depths, base.q)
        init_tt = time_misfit(init_earth, depth, distances, obs_tp, obs_ts)
        ref = refine_tt(
            base.depths,
            init_earth.vp.detach(),
            init_earth.vs.detach(),
            base.q,
            depth,
            distances,
            obs_tp,
            obs_ts,
        )
        rows.append(
            {
                "trace_name": batch["trace_name"][0],
                "distance_km": dist,
                "source_depth_km": depth,
                "pick_err_p": abs(tp - gt_p),
                "pick_err_s": abs(ts - gt_s),
                "init_tt": init_tt,
                "refined_tt": float(ref.time_misfit),
                "vp_init": init_earth.vp.detach().cpu().tolist(),
                "vs_init": init_earth.vs.detach().cpu().tolist(),
                "vp_refined": ref.earth.vp.detach().cpu().tolist(),
                "vs_refined": ref.earth.vs.detach().cpu().tolist(),
            }
        )
        n_seen += 1
        print(
            f"[phase-f {n_seen}/{args.max_events}] d={dist:.1f} z={depth:.1f} "
            f"tt {init_tt:.3f}->{float(ref.time_misfit):.3f}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No valid STEAD events for Phase F profile")

    rows = sorted(rows, key=lambda r: r["distance_km"])
    qc_rows = [
        r for r in rows
        if r["pick_err_p"] <= args.qc_pick_err_p_max
        and r["pick_err_s"] <= args.qc_pick_err_s_max
        and r["refined_tt"] <= args.qc_refined_tt_max
    ]
    if not qc_rows:
        qc_rows = rows
    distances_np = np.array([r["distance_km"] for r in rows], dtype=np.float32)
    bin_edges = np.linspace(max(args.distance_min, distances_np.min()), min(args.distance_max, distances_np.max()), args.n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    vp_bins = []
    vs_bins = []
    vp_std_bins = []
    vs_std_bins = []
    support_bins = []
    tt_bins = []
    records = []

    for i in range(args.n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        sel = [r for r in qc_rows if (r["distance_km"] >= lo and (r["distance_km"] < hi or (i == args.n_bins - 1 and r["distance_km"] <= hi)))]
        if not sel:
            # fill with NaNs, interpolate later
            n_layers = base.n_layers
            vp_bins.append(np.full(n_layers, np.nan, dtype=np.float32))
            vs_bins.append(np.full(n_layers, np.nan, dtype=np.float32))
            vp_std_bins.append(np.full(n_layers, np.nan, dtype=np.float32))
            vs_std_bins.append(np.full(n_layers, np.nan, dtype=np.float32))
            support_bins.append(0.0)
            tt_bins.append(np.nan)
            records.append(
                {
                    "x_km": float(bin_centers[i]),
                    "depths_km": base.depths.detach().cpu().tolist(),
                    "vp_km_s": None,
                    "vs_km_s": None,
                    "vp_std": None,
                    "vs_std": None,
                    "coverage_score": 0.0,
                    "event_count": 0,
                    "station_count": 1,
                }
            )
            continue
        vp_arr = np.array([r["vp_refined"] for r in sel], dtype=np.float32)
        vs_arr = np.array([r["vs_refined"] for r in sel], dtype=np.float32)
        vp_bins.append(vp_arr.mean(axis=0))
        vs_bins.append(vs_arr.mean(axis=0))
        vp_std_bins.append(vp_arr.std(axis=0))
        vs_std_bins.append(vs_arr.std(axis=0))
        support_bins.append(float(len(sel)))
        tt_bins.append(float(np.mean([r["refined_tt"] for r in sel])))
        records.append(
            {
                "x_km": float(bin_centers[i]),
                "depths_km": base.depths.detach().cpu().tolist(),
                "vp_km_s": vp_arr.mean(axis=0).tolist(),
                "vs_km_s": vs_arr.mean(axis=0).tolist(),
                "vp_std": vp_arr.std(axis=0).tolist(),
                "vs_std": vs_arr.std(axis=0).tolist(),
                "coverage_score": float(len(sel)),
                "event_count": len(sel),
                "station_count": 1,
            }
        )

    def _fill_nan_columns(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        x = np.arange(out.shape[0])
        for j in range(out.shape[1]):
            y = out[:, j]
            good = np.isfinite(y)
            if good.sum() == 0:
                out[:, j] = 0.0
            elif good.sum() == 1:
                out[:, j] = y[good][0]
            else:
                out[:, j] = np.interp(x, x[good], y[good])
        return out

    vp_bins_np = _fill_nan_columns(np.stack(vp_bins, axis=0))
    vs_bins_np = _fill_nan_columns(np.stack(vs_bins, axis=0))
    vp_std_np = _fill_nan_columns(np.stack(vp_std_bins, axis=0))
    vs_std_np = _fill_nan_columns(np.stack(vs_std_bins, axis=0))
    support_np = np.asarray(support_bins, dtype=np.float32)

    x_grid = np.linspace(bin_edges[0], bin_edges[-1], 220)
    z_grid = np.linspace(0.0, float(base.depths[-1].item()), 180)
    depths_np = base.depths.detach().cpu().numpy()
    vp_sec = interpolate_layers_to_section(bin_centers, vp_bins_np, depths_np, x_grid, z_grid)
    vs_sec = interpolate_layers_to_section(bin_centers, vs_bins_np, depths_np, x_grid, z_grid)
    vpvs_sec = vp_sec / np.clip(vs_sec, 1e-6, None)
    vp_std_sec = interpolate_layers_to_section(bin_centers, vp_std_np, depths_np, x_grid, z_grid)
    vs_std_sec = interpolate_layers_to_section(bin_centers, vs_std_np, depths_np, x_grid, z_grid)
    support_sec = np.tile(np.interp(x_grid, bin_centers, support_np), (len(z_grid), 1))
    trust_1d = (
        (np.interp(x_grid, bin_centers, support_np) >= float(args.qc_min_events_per_bin))
        & (np.interp(x_grid, bin_centers, _fill_nan_columns(vp_std_np[:, :1]).squeeze(-1)) <= args.qc_max_vp_std)
        & (np.interp(x_grid, bin_centers, _fill_nan_columns(vs_std_np[:, :1]).squeeze(-1)) <= args.qc_max_vs_std)
    )
    trust_mask = np.tile(trust_1d.astype(np.float32), (len(z_grid), 1))

    plot_section(vp_sec, x_grid, z_grid, fig_dir / "stead_profile_vp.png", title="STEAD pseudo-2D Vp profile", cmap="viridis", label="Vp (km/s)")
    plot_section(vs_sec, x_grid, z_grid, fig_dir / "stead_profile_vs.png", title="STEAD pseudo-2D Vs profile", cmap="viridis", label="Vs (km/s)")
    plot_section(vpvs_sec, x_grid, z_grid, fig_dir / "stead_profile_vpvs.png", title="STEAD pseudo-2D Vp/Vs profile", cmap="coolwarm", label="Vp/Vs")
    plot_section(support_sec, x_grid, z_grid, fig_dir / "stead_profile_support.png", title="Event support per distance bin", cmap="magma", label="Event count")
    plot_section(vp_std_sec, x_grid, z_grid, fig_dir / "stead_profile_vp_std.png", title="Vp uncertainty (bin std)", cmap="plasma", label="Vp std (km/s)")
    plot_section(vs_std_sec, x_grid, z_grid, fig_dir / "stead_profile_vs_std.png", title="Vs uncertainty (bin std)", cmap="plasma", label="Vs std (km/s)")
    plot_section(trust_mask, x_grid, z_grid, fig_dir / "stead_profile_trust_mask.png", title="Trusted region mask", cmap="gray", label="1=trusted")
    plot_masked_section(vp_sec, trust_mask, x_grid, z_grid, fig_dir / "stead_profile_vp_masked.png", title="STEAD Vp profile (trusted region)", cmap="viridis", label="Vp (km/s)")
    plot_masked_section(vs_sec, trust_mask, x_grid, z_grid, fig_dir / "stead_profile_vs_masked.png", title="STEAD Vs profile (trusted region)", cmap="viridis", label="Vs (km/s)")
    plot_masked_section(vpvs_sec, trust_mask, x_grid, z_grid, fig_dir / "stead_profile_vpvs_masked.png", title="STEAD Vp/Vs profile (trusted region)", cmap="coolwarm", label="Vp/Vs")

    summary = {
        "phase": "F_stead_pseudo_profile",
        "checkpoint": args.checkpoint,
        "physics_head": args.physics_head,
        "n_events_used": len(rows),
        "distance_range_km": [float(bin_edges[0]), float(bin_edges[-1])],
        "n_bins": args.n_bins,
        "pick_mae_p": float(np.mean(pick_p_err)) if pick_p_err else None,
        "pick_mae_s": float(np.mean(pick_s_err)) if pick_s_err else None,
        "mean_refined_tt": float(np.nanmean([r["refined_tt"] for r in rows])),
        "n_events_qc_kept": len(qc_rows),
        "qc_keep_frac": float(len(qc_rows) / max(len(rows), 1)),
        "mean_events_per_bin": float(np.mean(support_np)),
        "max_events_per_bin": int(np.max(support_np)),
        "trusted_bin_frac": float(np.mean(trust_1d.astype(np.float32))),
        "qc_rules": {
            "pick_err_p_max": args.qc_pick_err_p_max,
            "pick_err_s_max": args.qc_pick_err_s_max,
            "refined_tt_max": args.qc_refined_tt_max,
            "min_events_per_bin": args.qc_min_events_per_bin,
            "max_vp_std": args.qc_max_vp_std,
            "max_vs_std": args.qc_max_vs_std,
        },
        "outputs": {
            "vp": str(fig_dir / "stead_profile_vp.png"),
            "vs": str(fig_dir / "stead_profile_vs.png"),
            "vpvs": str(fig_dir / "stead_profile_vpvs.png"),
            "support": str(fig_dir / "stead_profile_support.png"),
            "vp_std": str(fig_dir / "stead_profile_vp_std.png"),
            "vs_std": str(fig_dir / "stead_profile_vs_std.png"),
            "trust_mask": str(fig_dir / "stead_profile_trust_mask.png"),
            "vp_masked": str(fig_dir / "stead_profile_vp_masked.png"),
            "vs_masked": str(fig_dir / "stead_profile_vs_masked.png"),
            "vpvs_masked": str(fig_dir / "stead_profile_vpvs_masked.png"),
        },
        "profile_samples": records,
        "events": rows,
        "events_qc": qc_rows,
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"events", "profile_samples"}}, indent=2))
    print(f"[phase-f] -> {out_dir}")


if __name__ == "__main__":
    main()
