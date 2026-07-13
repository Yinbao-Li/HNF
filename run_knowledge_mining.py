#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical knowledge mining over HNF latent / kernel / physical outputs.

Exports:
  - sample-level CSV/JSON
  - screened pairwise / partial statistics with bootstrap CI
  - bucket summaries
  - compact figures for README / analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.picking_metrics import idx_to_sec
from hnf.picking_prior import run_picking_on_batch
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.stead_zhizi_inversion_dataset import encode_geometry_tensor
from hnf.physics_decoder import load_physics_decoder_from_checkpoint
from run_phase_f_stead_profile import refine_tt, time_misfit
from hnf.inversion_1d import default_synth_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF knowledge mining export")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_mixed_geo/best_physics_head.pt")
    p.add_argument("--output-dir", default="outputs/knowledge_mining")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--max-events", type=int, default=96)
    p.add_argument("--distance-min", type=float, default=1.0)
    p.add_argument("--distance-max", type=float, default=200.0)
    p.add_argument("--n-bootstrap", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trim-quantile", type=float, default=0.95)
    p.add_argument(
        "--compare-physics-heads",
        nargs="*",
        default=[
            "outputs/zhizi_inversion_bridge_macro/best_physics_head.pt",
            "outputs/zhizi_inversion_bridge_residual/best_physics_head.pt",
            "outputs/zhizi_inversion_mixed_geo/best_physics_head.pt",
            "outputs/zhizi_inversion_stead_macro/best_physics_head.pt",
        ],
    )
    p.add_argument("--compare-max-events", type=int, default=24)
    return p.parse_args()


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for idx, c in enumerate(counts):
        if c > 1:
            loc = np.where(inv == idx)[0]
            ranks[loc] = ranks[loc].mean()
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if den <= 1e-12:
        return float("nan")
    return float(np.sum(x0 * y0) / den)


def spearman_corr(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    good = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[good]
    ya = ya[good]
    if len(xa) < 3:
        return float("nan")
    return _pearson(_rankdata(xa), _rankdata(ya))


def bootstrap_ci(x: list[float], y: list[float], *, n_boot: int, seed: int) -> tuple[float, float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    good = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[good]
    ya = ya[good]
    if len(xa) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xa), size=len(xa))
        vals.append(spearman_corr(xa[idx].tolist(), ya[idx].tolist()))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _linear_residual(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.ndim == 1:
        controls = controls[:, None]
    good = np.all(np.isfinite(controls), axis=1) & np.isfinite(y)
    out = np.full_like(y, np.nan, dtype=np.float64)
    if good.sum() < controls.shape[1] + 2:
        return out
    x = controls[good]
    yy = y[good]
    x = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    out[good] = yy - x @ beta
    return out


def partial_spearman_corr(x: list[float], y: list[float], controls: list[list[float]]) -> tuple[float, int]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ca = np.asarray(controls, dtype=np.float64).T
    good = np.isfinite(xa) & np.isfinite(ya) & np.all(np.isfinite(ca), axis=1)
    xa = xa[good]
    ya = ya[good]
    ca = ca[good]
    if len(xa) < max(8, ca.shape[1] + 3):
        return float("nan"), 0
    xr = _linear_residual(_rankdata(xa), ca)
    yr = _linear_residual(_rankdata(ya), ca)
    good2 = np.isfinite(xr) & np.isfinite(yr)
    if good2.sum() < 4:
        return float("nan"), int(good2.sum())
    return _pearson(xr[good2], yr[good2]), int(good2.sum())


def bootstrap_partial_ci(
    x: list[float],
    y: list[float],
    controls: list[list[float]],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ca = np.asarray(controls, dtype=np.float64).T
    good = np.isfinite(xa) & np.isfinite(ya) & np.all(np.isfinite(ca), axis=1)
    xa = xa[good]
    ya = ya[good]
    ca = ca[good]
    if len(xa) < max(8, ca.shape[1] + 3):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xa), size=len(xa))
        val, _ = partial_spearman_corr(xa[idx].tolist(), ya[idx].tolist(), [ca[idx, j].tolist() for j in range(ca.shape[1])])
        vals.append(val)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _normal_p_from_r(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 4:
        return float("nan")
    r = float(np.clip(r, -0.999999, 0.999999))
    z = abs(r) * np.sqrt(max(n - 3, 1))
    # normal approx using erfc
    return float(np.math.erfc(z / np.sqrt(2.0)))


def fdr_bh(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.asarray(pvals)[order]
    adj = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adj[i] = val
        prev = val
    out = np.empty(n, dtype=np.float64)
    out[order] = adj
    return out.tolist()


def bucket_summary(rows: list[dict], key: str, fields: list[str], n_bins: int = 4) -> list[dict]:
    vals = np.asarray([r[key] for r in rows], dtype=np.float64)
    good = np.isfinite(vals)
    vals_good = vals[good]
    if len(vals_good) < n_bins:
        return []
    edges = np.quantile(vals_good, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        sel = [r for r in rows if np.isfinite(r[key]) and r[key] >= lo and r[key] < hi]
        if not sel:
            continue
        rec = {"bucket_key": key, "bucket_id": i, "lo": lo, "hi": hi, "n": len(sel)}
        for f in fields:
            arr = np.asarray([r[f] for r in sel], dtype=np.float64)
            rec[f"{f}_mean"] = float(np.nanmean(arr))
            rec[f"{f}_std"] = float(np.nanstd(arr))
        out.append(rec)
    return out


def mediation_screen(rows: list[dict]) -> list[dict]:
    chains = [
        ("rho_mean", "p_prob_lag", "refined_tt"),
        ("rho_mean", "s_prob_lag", "refined_tt"),
        ("rho_p_lag", "p_prob_lag", "refined_tt"),
        ("rho_s_lag", "s_prob_lag", "refined_tt"),
    ]
    out = []
    for x, m, y in chains:
        xm = spearman_corr([r[x] for r in rows], [r[m] for r in rows])
        my = spearman_corr([r[m] for r in rows], [r[y] for r in rows])
        xy = spearman_corr([r[x] for r in rows], [r[y] for r in rows])
        out.append({
            "x": x,
            "m": m,
            "y": y,
            "rho_xm": xm,
            "rho_my": my,
            "rho_xy": xy,
            "chain_score": float(xm * my) if np.isfinite(xm) and np.isfinite(my) else float("nan"),
        })
    return out


def plot_overview(rows: list[dict], out_dir: Path) -> dict:
    dist = [r["distance_km"] for r in rows]
    depth = [r["source_depth_km"] for r in rows]
    rho = [r["rho_mean"] for r in rows]
    vpvs = [r["vpvs_mean"] for r in rows]
    tt = [r["refined_tt"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    sc0 = axes[0, 0].scatter(dist, rho, c=depth, cmap="viridis", s=44)
    axes[0, 0].set_xlabel("distance_km")
    axes[0, 0].set_ylabel("rho_mean")
    axes[0, 0].set_title("rho_mean vs distance (color=depth)")
    axes[0, 0].grid(True, alpha=0.3)
    fig.colorbar(sc0, ax=axes[0, 0], fraction=0.046)

    sc1 = axes[0, 1].scatter(rho, vpvs, c=tt, cmap="plasma", s=44)
    axes[0, 1].set_xlabel("rho_mean")
    axes[0, 1].set_ylabel("vpvs_mean")
    axes[0, 1].set_title("rho_mean vs vp/vs (color=refined_tt)")
    axes[0, 1].grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=axes[0, 1], fraction=0.046)

    axes[1, 0].scatter([r["p_prob_lag"] for r in rows], [r["s_prob_lag"] for r in rows], c=dist, cmap="magma", s=44)
    axes[1, 0].axvline(0.0, color="k", ls="--", lw=1)
    axes[1, 0].axhline(0.0, color="k", ls="--", lw=1)
    axes[1, 0].set_xlabel("p_prob_lag (s)")
    axes[1, 0].set_ylabel("s_prob_lag (s)")
    axes[1, 0].set_title("Pick lag coupling")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(tt, bins=min(12, len(tt)), color="C0", alpha=0.85)
    axes[1, 1].set_xlabel("refined_tt")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("Refined TT distribution")
    axes[1, 1].grid(True, alpha=0.3)
    p = out_dir / "knowledge_overview.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return {"figure": str(p)}


def plot_bucket_panels(distance_buckets: list[dict], depth_buckets: list[dict], out_dir: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    for ax, buckets, title in [
        (axes[0], distance_buckets, "Distance buckets"),
        (axes[1], depth_buckets, "Depth buckets"),
    ]:
        if not buckets:
            ax.axis("off")
            continue
        x = np.arange(len(buckets))
        ax.errorbar(x, [b["rho_mean_mean"] for b in buckets], yerr=[b["rho_mean_std"] for b in buckets], marker="o", label="rho_mean")
        ax.errorbar(x, [b["vpvs_mean_mean"] for b in buckets], yerr=[b["vpvs_mean_std"] for b in buckets], marker="s", label="vpvs_mean")
        ax.errorbar(x, [b["refined_tt_mean"] for b in buckets], yerr=[b["refined_tt_std"] for b in buckets], marker="^", label="refined_tt")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b['lo']:.1f}\n{b['hi']:.1f}" for b in buckets], fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    p = out_dir / "knowledge_bucket_panels.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return {"figure": str(p)}


def robust_subset(rows: list[dict], trim_quantile: float) -> tuple[list[dict], dict[str, float]]:
    if not rows:
        return rows, {"n_total": 0, "n_kept": 0}
    q = float(np.clip(trim_quantile, 0.5, 0.999))
    cut_tt = float(np.quantile([r["refined_tt"] for r in rows], q))
    cut_pp = float(np.quantile([r["pick_err_p"] for r in rows], q))
    cut_ps = float(np.quantile([r["pick_err_s"] for r in rows], q))
    kept = [
        r for r in rows
        if r["refined_tt"] <= cut_tt and r["pick_err_p"] <= cut_pp and r["pick_err_s"] <= cut_ps
    ]
    return kept, {
        "trim_quantile": q,
        "tt_cut": cut_tt,
        "pick_err_p_cut": cut_pp,
        "pick_err_s_cut": cut_ps,
        "n_total": len(rows),
        "n_kept": len(kept),
    }


def summarize_relations(rows: list[dict], pairs: list[tuple[str, str]], n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    pvals = []
    for i, (a, b) in enumerate(pairs):
        x = [r[a] for r in rows]
        y = [r[b] for r in rows]
        rho_sp = spearman_corr(x, y)
        ci_lo, ci_hi = bootstrap_ci(x, y, n_boot=n_bootstrap, seed=seed + i * 17)
        n_eff = int(np.isfinite(np.asarray(x)).astype(int) @ np.isfinite(np.asarray(y)).astype(int))
        p = _normal_p_from_r(rho_sp, n_eff)
        rels.append({"x": a, "y": b, "n": n_eff, "spearman": rho_sp, "ci95": [ci_lo, ci_hi], "p_approx": p})
        pvals.append(p)
    adj = fdr_bh([1.0 if not np.isfinite(p) else p for p in pvals])
    for r, q in zip(rels, adj):
        r["fdr_q"] = q
    return rels


def summarize_partials(
    rows: list[dict],
    partial_specs: list[tuple[str, str, list[str]]],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    partials: list[dict[str, Any]] = []
    pvals_partial = []
    for i, (xk, yk, ck) in enumerate(partial_specs):
        x = [r[xk] for r in rows]
        y = [r[yk] for r in rows]
        controls = [[r[c] for r in rows] for c in ck]
        rho_pc, n_eff = partial_spearman_corr(x, y, controls)
        ci_lo, ci_hi = bootstrap_partial_ci(x, y, controls, n_boot=n_bootstrap, seed=seed + 1000 + i * 19)
        p = _normal_p_from_r(rho_pc, n_eff)
        partials.append({
            "x": xk,
            "y": yk,
            "controls": ck,
            "n": n_eff,
            "partial_spearman": rho_pc,
            "ci95": [ci_lo, ci_hi],
            "p_approx": p,
        })
        pvals_partial.append(p)
    adj_partial = fdr_bh([1.0 if not np.isfinite(p) else p for p in pvals_partial])
    for r, q in zip(partials, adj_partial):
        r["fdr_q"] = q
    return partials


def compare_heads(
    rows: list[dict],
    backbone,
    ckpt_args: dict,
    base,
    device,
    infer_seq_len: int,
    head_paths: list[str],
    max_events: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    out = []
    # Reuse the same backbone while swapping heads; compare on exported sample traces only.
    ds = STEADPickingDataset("test", seq_len=int(ckpt_args.get("seq_len", 800)))
    target_rows = rows[: max(1, int(max_events))]
    wanted = {r["trace_name"] for r in target_rows}
    matched_items: dict[str, Any] = {}
    for i in range(len(ds)):
        item = ds[i]
        trace = str(item["trace_name"])
        if trace in wanted:
            matched_items[trace] = item
            if len(matched_items) == len(wanted):
                break
    selected = [r for r in target_rows if r["trace_name"] in matched_items]
    if not selected:
        return []
    for head_path in head_paths:
        hp = Path(head_path)
        if not hp.exists():
            continue
        bridge = load_physics_decoder_from_checkpoint(
            backbone,
            str(hp),
            device,
            embed_dim=int(ckpt_args.get("embed_dim", 64)),
            n_layers=base.n_layers,
            infer_seq_len=infer_seq_len,
        )
        bridge.eval()
        vp_means = []
        vs_means = []
        vpvs_means = []
        tt_vals = []
        for row in selected:
            item = matched_items[row["trace_name"]]
            x = item["x"].unsqueeze(0).to(device)
            t = item["t"].unsqueeze(0).to(device) if item["t"].dim() == 2 else item["t"].to(device)
            dist = float(item["source_distance_km"])
            depth = max(float(item["source_depth_km"]), 1.0)
            geo = encode_geometry_tensor(dist, depth, device=device) if getattr(bridge, "geo_condition", False) else None
            gt_p = idx_to_sec(int(item["p_idx"]), x.shape[1])
            gt_s = idx_to_sec(int(item["s_idx"]), x.shape[1])
            picks = run_picking_on_batch(backbone, x, t, infer_seq_len=None)
            tp = picks["tp_sec"][0] if picks["tp_sec"][0] is not None else gt_p
            ts = picks["ts_sec"][0] if picks["ts_sec"][0] is not None else gt_s
            distances = torch.tensor([dist], dtype=torch.float32, device=device)
            obs_tp = torch.tensor([tp], device=device)
            obs_ts = torch.tensor([ts], device=device)
            with torch.no_grad():
                out_head, _ = bridge.forward_event(x, t[0] if t.dim() == 3 else t, include_picks=True, geo=geo)
            init_earth = bridge.physics_head.earth(out_head, base.depths, base.q)
            vp = init_earth.vp.detach().cpu().numpy()
            vs = init_earth.vs.detach().cpu().numpy()
            vp_means.append(float(np.mean(vp)))
            vs_means.append(float(np.mean(vs)))
            vpvs_means.append(float(np.mean(vp / np.clip(vs, 1e-6, None))))
            tt_vals.append(float(time_misfit(init_earth, depth, distances, obs_tp, obs_ts)))
        out.append({
            "physics_head": str(hp),
            "n_rows": len(vp_means),
            "vp_mean_avg": float(np.mean(vp_means)),
            "vs_mean_avg": float(np.mean(vs_means)),
            "vpvs_mean_avg": float(np.mean(vpvs_means)),
            "init_tt_avg": float(np.mean(tt_vals)),
            "init_tt_std": float(np.std(tt_vals)),
        })
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    bridge.eval()
    ds = STEADPickingDataset("test", seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    params = backbone.collect_kernel_params()
    rows: list[dict] = []
    n_seen = 0
    for batch in loader:
        if n_seen >= args.max_events:
            break
        if float(batch["det"][0]) <= 0.5 or float(batch["p_valid"][0]) <= 0 or float(batch["s_valid"][0]) <= 0:
            continue
        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        if not np.isfinite(dist) or not np.isfinite(depth) or dist < args.distance_min or dist > args.distance_max:
            continue
        depth = max(depth, 1.0)
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        gt_p = idx_to_sec(int(batch["p_idx"][0]), x.shape[1])
        gt_s = idx_to_sec(int(batch["s_idx"][0]), x.shape[1])
        picks = run_picking_on_batch(backbone, x, t, infer_seq_len=None)
        tp = picks["tp_sec"][0] if picks["tp_sec"][0] is not None else gt_p
        ts = picks["ts_sec"][0] if picks["ts_sec"][0] is not None else gt_s

        with torch.no_grad():
            feat = bridge.extract_station_features(x, t, include_picks=True)
            geo = encode_geometry_tensor(dist, depth, device=device) if getattr(bridge, "geo_condition", False) else None
            out, _ = bridge.forward_event(x, t[0] if t.dim() == 3 else t, include_picks=True, geo=geo)
        rho = feat["rho"][0].detach().cpu().numpy()
        p_prob = torch.sigmoid(feat["p_logits"][0]).detach().cpu().numpy()
        s_prob = torch.sigmoid(feat["s_logits"][0]).detach().cpu().numpy()
        h_real = feat["h_real"][0].detach().cpu().numpy()
        h_imag = feat["h_imag"][0].detach().cpu().numpy()
        env = np.sqrt((h_real ** 2 + h_imag ** 2).sum(axis=-1) + 1e-8)
        t_sec = t[0, :, 0].detach().cpu().numpy()

        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0])
        def _peak(arr: np.ndarray, center_idx: int, left: int = 30, right: int = 50) -> float:
            i0 = max(0, center_idx - left)
            i1 = min(len(arr), center_idx + right)
            loc = int(np.argmax(arr[i0:i1])) + i0
            return float(t_sec[loc])

        distances = torch.tensor([dist], dtype=torch.float32, device=device)
        obs_tp = torch.tensor([tp], device=device)
        obs_ts = torch.tensor([ts], device=device)
        init_earth = bridge.physics_head.earth(out, base.depths, base.q)
        init_tt = time_misfit(init_earth, depth, distances, obs_tp, obs_ts)
        ref = refine_tt(base.depths, init_earth.vp.detach(), init_earth.vs.detach(), base.q, depth, distances, obs_tp, obs_ts)
        vp = ref.earth.vp.detach().cpu().numpy()
        vs = ref.earth.vs.detach().cpu().numpy()
        row = {
            "trace_name": str(batch["trace_name"][0]),
            "distance_km": dist,
            "source_depth_km": depth,
            "pick_err_p": abs(tp - gt_p),
            "pick_err_s": abs(ts - gt_s),
            "init_tt": float(init_tt),
            "refined_tt": float(ref.time_misfit),
            "rho_mean": float(np.mean(rho)),
            "rho_peak": float(np.max(rho)),
            "rho_p_lag": float(_peak(rho, p_idx) - gt_p),
            "rho_s_lag": float(_peak(rho, s_idx) - gt_s),
            "env_p_lag": float(_peak(env, p_idx) - gt_p),
            "env_s_lag": float(_peak(env, s_idx) - gt_s),
            "p_prob_lag": float(_peak(p_prob, p_idx) - gt_p),
            "s_prob_lag": float(_peak(s_prob, s_idx) - gt_s),
            "gamma_p0": float(params["p_branch_0"]["gamma"]),
            "omega_p0": float(params["p_branch_0"]["omega"]),
            "gamma_s0": float(params["s_branch_0"]["gamma"]),
            "omega_s0": float(params["s_branch_0"]["omega"]),
            "kernel_vp": float(feat["kernel_vp"][0]),
            "kernel_vs": float(feat["kernel_vs"][0]),
            "vp_mean": float(np.mean(vp)),
            "vs_mean": float(np.mean(vs)),
            "vpvs_mean": float(np.mean(vp / np.clip(vs, 1e-6, None))),
            "vp_std_layers": float(np.std(vp)),
            "vs_std_layers": float(np.std(vs)),
        }
        rows.append(row)
        n_seen += 1

    if not rows:
        raise RuntimeError("No rows collected for knowledge mining")

    fields = list(rows[0].keys())
    with (out_dir / "sample_level_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "sample_level_stats.json").write_text(json.dumps(rows, indent=2))

    pairs = [
        ("distance_km", "rho_mean"),
        ("source_depth_km", "rho_mean"),
        ("rho_mean", "vp_mean"),
        ("rho_mean", "vpvs_mean"),
        ("kernel_vp", "vp_mean"),
        ("kernel_vs", "vs_mean"),
        ("p_prob_lag", "refined_tt"),
        ("s_prob_lag", "refined_tt"),
        ("pick_err_p", "refined_tt"),
        ("pick_err_s", "refined_tt"),
    ]
    rels = summarize_relations(rows, pairs, args.n_bootstrap, args.seed)

    partial_specs = [
        ("rho_mean", "vp_mean", ["distance_km", "source_depth_km"]),
        ("rho_mean", "vpvs_mean", ["distance_km", "source_depth_km"]),
        ("p_prob_lag", "refined_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("s_prob_lag", "refined_tt", ["distance_km", "source_depth_km", "pick_err_s"]),
        ("rho_p_lag", "refined_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("rho_s_lag", "refined_tt", ["distance_km", "source_depth_km", "pick_err_s"]),
    ]
    partials = summarize_partials(rows, partial_specs, args.n_bootstrap, args.seed)

    robust_rows, robust_info = robust_subset(rows, args.trim_quantile)
    robust_relations = summarize_relations(robust_rows, pairs, args.n_bootstrap, args.seed + 2000) if robust_rows else []
    robust_partials = summarize_partials(robust_rows, partial_specs, args.n_bootstrap, args.seed + 3000) if robust_rows else []

    distance_buckets = bucket_summary(rows, "distance_km", ["rho_mean", "vpvs_mean", "refined_tt", "pick_err_p", "pick_err_s"], n_bins=4)
    depth_buckets = bucket_summary(rows, "source_depth_km", ["rho_mean", "vpvs_mean", "refined_tt", "pick_err_p", "pick_err_s"], n_bins=4)
    mediation = mediation_screen(rows)
    head_compare = compare_heads(rows, backbone, ckpt_args, base, device, args.infer_seq_len, args.compare_physics_heads, args.compare_max_events)

    overview = plot_overview(rows, out_dir)
    bucket_fig = plot_bucket_panels(distance_buckets, depth_buckets, out_dir)
    report = {
        "checkpoint": args.checkpoint,
        "physics_head": args.physics_head,
        "n_rows": len(rows),
        "relations": rels,
        "partial_relations": partials,
        "robust_subset": robust_info,
        "robust_relations": robust_relations,
        "robust_partial_relations": robust_partials,
        "distance_buckets": distance_buckets,
        "depth_buckets": depth_buckets,
        "mediation_screen": mediation,
        "head_compare": head_compare,
        "overview": overview,
        "bucket_overview": bucket_fig,
        "field_names": fields,
        "notes": {
            "kernel_global": "Current gamma/omega/kernel_vp/kernel_vs are global learned branch parameters; event-level variation is expected mainly in rho/picks/bridge outputs.",
            "causal_chain": "Subsequent mining should condition on the mechanism chain gamma/omega -> kernel behavior -> rho/picks -> macro conditioning -> vp/vs.",
        },
    }
    (out_dir / "knowledge_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# HNF Knowledge Mining Report",
        "",
        f"- n rows: {len(rows)}",
        f"- sample table: `sample_level_stats.csv`",
        f"- overview: `{Path(overview['figure']).name}`",
        "",
        "## Screened relations",
    ]
    for r in rels:
        md.append(
            f"- `{r['x']} -> {r['y']}`: spearman={r['spearman']:.3f}, "
            f"95% CI=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}], "
            f"p≈{r['p_approx']:.3g}, q≈{r['fdr_q']:.3g}"
        )
    md += [
        "",
        "## Partial relations",
    ]
    for r in partials:
        md.append(
            f"- `{r['x']} -> {r['y']} | {', '.join(r['controls'])}`: partial={r['partial_spearman']:.3f}, "
            f"95% CI=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}], "
            f"p≈{r['p_approx']:.3g}, q≈{r['fdr_q']:.3g}"
        )
    md += [
        "",
        "## Robust subset",
        f"- kept: {robust_info['n_kept']} / {robust_info['n_total']}",
        f"- trim_quantile: {robust_info['trim_quantile']:.3f}",
        f"- tt_cut: {robust_info['tt_cut']:.3f}",
        f"- pick_err_p_cut: {robust_info['pick_err_p_cut']:.3f}",
        f"- pick_err_s_cut: {robust_info['pick_err_s_cut']:.3f}",
        "",
        "## Robust partial relations",
    ]
    for r in robust_partials:
        md.append(
            f"- `{r['x']} -> {r['y']} | {', '.join(r['controls'])}`: partial={r['partial_spearman']:.3f}, "
            f"95% CI=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}], "
            f"p≈{r['p_approx']:.3g}, q≈{r['fdr_q']:.3g}"
        )
    md += [
        "",
        "## Mediation-style screen",
    ]
    for r in mediation:
        md.append(
            f"- `{r['x']} -> {r['m']} -> {r['y']}`: "
            f"rho_xm={r['rho_xm']:.3f}, rho_my={r['rho_my']:.3f}, rho_xy={r['rho_xy']:.3f}, chain_score={r['chain_score']:.3f}"
        )
    md += [
        "",
        "## Bucket outputs",
        f"- overview: `{Path(bucket_fig['figure']).name}`",
        f"- distance buckets: {len(distance_buckets)}",
        f"- depth buckets: {len(depth_buckets)}",
        "",
        "## Head compare",
    ]
    for r in head_compare:
        md.append(
            f"- `{Path(r['physics_head']).parent.name}`: "
            f"vp_mean_avg={r['vp_mean_avg']:.3f}, vs_mean_avg={r['vs_mean_avg']:.3f}, "
            f"vpvs_mean_avg={r['vpvs_mean_avg']:.3f}, init_tt_avg={r['init_tt_avg']:.3f}±{r['init_tt_std']:.3f}"
        )
    (out_dir / "knowledge_report.md").write_text("\n".join(md))
    print(json.dumps({"n_rows": len(rows), "overview": overview["figure"], "report": str(out_dir / "knowledge_report.json")}, indent=2))


if __name__ == "__main__":
    main()
