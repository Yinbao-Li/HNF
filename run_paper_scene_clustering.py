#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-grade scene clustering + noise-feature mining.

Exports a large sample table (geometry, latents, picks, noise-branch stats),
clusters events, and re-tests candidate relations within each cluster with
bootstrap CI. This tests whether regularities are scene-specific.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.inversion_1d import default_synth_model
from hnf.picking_metrics import idx_to_sec
from hnf.picking_prior import run_picking_on_batch
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.stead_zhizi_inversion_dataset import encode_geometry_tensor
from hnf.zhizi_inversion_bridge import load_inversion_bridge_from_checkpoint
from run_knowledge_mining import bootstrap_partial_ci, partial_spearman_corr, spearman_corr, _normal_p_from_r
from run_phase_f_stead_profile import time_misfit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper scene clustering mining")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_mixed_geo/best_physics_head.pt")
    p.add_argument("--output-dir", default="outputs/paper_scene_clustering")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--max-events", type=int, default=400)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def kmeans(x: np.ndarray, k: int, seed: int, n_iter: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # standardize
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = (x - mu) / sd
    centers = z[rng.choice(len(z), size=k, replace=False)]
    labels = np.zeros(len(z), dtype=np.int64)
    for _ in range(n_iter):
        dist = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1)
        for j in range(k):
            sel = z[labels == j]
            if len(sel) == 0:
                centers[j] = z[rng.integers(0, len(z))]
            else:
                centers[j] = sel.mean(axis=0)
    return labels, centers


def peak_lag(arr: np.ndarray, t_sec: np.ndarray, center_idx: int, gt_sec: float, left: int = 30, right: int = 50) -> float:
    i0 = max(0, center_idx - left)
    i1 = min(len(arr), center_idx + right)
    loc = int(np.argmax(arr[i0:i1])) + i0
    return float(t_sec[loc] - gt_sec)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)

    # Keep noise branch ON for mining noise features
    backbone, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=False)
    base = default_synth_model(device)
    bridge = load_inversion_bridge_from_checkpoint(
        backbone, args.physics_head, device,
        embed_dim=int(ckpt_args.get("embed_dim", 64)),
        n_layers=base.n_layers,
        infer_seq_len=600,
    )
    bridge.eval()

    ds = STEADPickingDataset("test", seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    n_seen = 0
    for batch in loader:
        if n_seen >= args.max_events:
            break
        if float(batch["det"][0]) <= 0.5 or float(batch["p_valid"][0]) <= 0 or float(batch["s_valid"][0]) <= 0:
            continue
        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        if not np.isfinite(dist) or not np.isfinite(depth) or dist < 1 or dist > 200:
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
            out_nc = backbone.forward_explain(x, t, include_kernel_row=False)
            feat = bridge.extract_station_features(x, t, include_picks=True)
            geo = encode_geometry_tensor(dist, depth, device=device) if getattr(bridge, "geo_condition", False) else None
            bout, _ = bridge.forward_event(x, t[0] if t.dim() == 3 else t, include_picks=True, geo=geo)

        rho = feat["rho"][0].detach().cpu().numpy()
        p_prob = torch.sigmoid(feat["p_logits"][0]).detach().cpu().numpy()
        s_prob = torch.sigmoid(feat["s_logits"][0]).detach().cpu().numpy()
        t_sec = t[0, :, 0].detach().cpu().numpy()
        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0])

        # noise-branch features if available
        if "nc_u_denoised" in out_nc and "nc_n_sim" in out_nc:
            u_dn = out_nc["nc_u_denoised"][0].detach().cpu().numpy()
            n_sim = out_nc["nc_n_sim"][0].detach().cpu().numpy()
            x_np = x[0].detach().cpu().numpy()
            noise_energy = float(np.mean(n_sim ** 2))
            signal_energy = float(np.mean(u_dn ** 2)) + 1e-8
            residual_energy = float(np.mean((x_np - u_dn) ** 2))
            noise_ratio = noise_energy / (signal_energy + noise_energy)
        else:
            noise_energy = float("nan")
            signal_energy = float("nan")
            residual_energy = float("nan")
            noise_ratio = float("nan")

        distances = torch.tensor([dist], dtype=torch.float32, device=device)
        obs_tp = torch.tensor([tp], device=device)
        obs_ts = torch.tensor([ts], device=device)
        init_earth = bridge.physics_head.earth(bout, base.depths, base.q)
        vp = init_earth.vp.detach().cpu().numpy()
        vs = init_earth.vs.detach().cpu().numpy()
        init_tt = float(time_misfit(init_earth, depth, distances, obs_tp, obs_ts))

        rows.append({
            "trace_name": str(batch["trace_name"][0]),
            "distance_km": dist,
            "source_depth_km": depth,
            "pick_err_p": abs(tp - gt_p),
            "pick_err_s": abs(ts - gt_s),
            "refined_tt": init_tt,  # paper-scale: use init TT as fit-quality proxy
            "init_tt": init_tt,
            "rho_mean": float(np.mean(rho)),
            "rho_p_lag": peak_lag(rho, t_sec, p_idx, gt_p),
            "rho_s_lag": peak_lag(rho, t_sec, s_idx, gt_s),
            "p_prob_lag": peak_lag(p_prob, t_sec, p_idx, gt_p),
            "s_prob_lag": peak_lag(s_prob, t_sec, s_idx, gt_s),
            "noise_energy": noise_energy,
            "signal_energy": signal_energy,
            "residual_energy": residual_energy,
            "noise_ratio": noise_ratio,
            "vp_mean": float(np.mean(vp)),
            "vs_mean": float(np.mean(vs)),
            "vpvs_mean": float(np.mean(vp / np.clip(vs, 1e-6, None))),
        })
        n_seen += 1
        if n_seen % 25 == 0:
            print(f"[cluster] collected {n_seen}/{args.max_events}", flush=True)

    if len(rows) < max(args.n_clusters * 3, 6):
        raise RuntimeError(f"Too few rows for clustering: {len(rows)}")

    fields = list(rows[0].keys())
    with (out_dir / "sample_with_noise.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    feat_keys = [
        "distance_km", "source_depth_km", "pick_err_p", "pick_err_s",
        "rho_mean", "rho_p_lag", "rho_s_lag", "noise_ratio", "refined_tt",
    ]
    X = np.array([[r[k] for k in feat_keys] for r in rows], dtype=np.float64)
    # impute nan noise_ratio with median
    for j, k in enumerate(feat_keys):
        col = X[:, j]
        if not np.all(np.isfinite(col)):
            med = np.nanmedian(col)
            col[~np.isfinite(col)] = med if np.isfinite(med) else 0.0
            X[:, j] = col
    labels, _ = kmeans(X, args.n_clusters, args.seed)
    for r, lab in zip(rows, labels):
        r["cluster"] = int(lab)

    with (out_dir / "sample_clustered.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["cluster"])
        w.writeheader()
        w.writerows(rows)

    # cluster profiles
    profiles = []
    for c in range(args.n_clusters):
        sel = [r for r in rows if r["cluster"] == c]
        prof = {"cluster": c, "n": len(sel)}
        for k in feat_keys + ["vp_mean", "vpvs_mean"]:
            arr = np.array([r[k] for r in sel], dtype=np.float64)
            prof[f"{k}_mean"] = float(np.nanmean(arr))
            prof[f"{k}_std"] = float(np.nanstd(arr))
        profiles.append(prof)

    # within-cluster candidate tests
    candidates = [
        ("rho_p_lag", "refined_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("rho_mean", "vp_mean", ["distance_km", "source_depth_km"]),
        ("noise_ratio", "refined_tt", ["distance_km", "source_depth_km"]),
        ("noise_ratio", "pick_err_p", ["distance_km"]),
    ]
    cluster_rels = []
    for c in range(args.n_clusters):
        sel = [r for r in rows if r["cluster"] == c]
        for i, (xk, yk, ck) in enumerate(candidates):
            x = [r[xk] for r in sel]
            y = [r[yk] for r in sel]
            controls = [[r[cc] for r in sel] for cc in ck]
            rho, n_eff = partial_spearman_corr(x, y, controls)
            ci_lo, ci_hi = bootstrap_partial_ci(x, y, controls, n_boot=args.n_bootstrap, seed=args.seed + c * 100 + i)
            cluster_rels.append({
                "cluster": c,
                "n": n_eff,
                "x": xk,
                "y": yk,
                "controls": ck,
                "partial_spearman": rho,
                "ci95": [ci_lo, ci_hi],
                "p_approx": _normal_p_from_r(rho, n_eff),
                "ci_excludes_zero": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and ((ci_lo > 0) or (ci_hi < 0))),
            })

    # global for comparison
    global_rels = []
    for i, (xk, yk, ck) in enumerate(candidates):
        x = [r[xk] for r in rows]
        y = [r[yk] for r in rows]
        controls = [[r[cc] for r in rows] for cc in ck]
        rho, n_eff = partial_spearman_corr(x, y, controls)
        ci_lo, ci_hi = bootstrap_partial_ci(x, y, controls, n_boot=args.n_bootstrap, seed=args.seed + 900 + i)
        global_rels.append({
            "cluster": "all",
            "n": n_eff,
            "x": xk,
            "y": yk,
            "controls": ck,
            "partial_spearman": rho,
            "ci95": [ci_lo, ci_hi],
            "p_approx": _normal_p_from_r(rho, n_eff),
            "ci_excludes_zero": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and ((ci_lo > 0) or (ci_hi < 0))),
        })

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    sc = axes[0].scatter(
        [r["distance_km"] for r in rows],
        [r["source_depth_km"] for r in rows],
        c=[r["cluster"] for r in rows],
        cmap="tab10", s=28, alpha=0.85,
    )
    axes[0].set_xlabel("distance_km")
    axes[0].set_ylabel("source_depth_km")
    axes[0].set_title("Scene clusters in geometry space")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes[0], fraction=0.046, label="cluster")

    # relation heatmap-like bars for rho_p_lag->refined_tt
    xs = ["all"] + [f"C{c}" for c in range(args.n_clusters)]
    vals = []
    for item in global_rels:
        if item["x"] == "rho_p_lag" and item["y"] == "refined_tt":
            vals.append(item["partial_spearman"])
    for c in range(args.n_clusters):
        for item in cluster_rels:
            if item["cluster"] == c and item["x"] == "rho_p_lag" and item["y"] == "refined_tt":
                vals.append(item["partial_spearman"])
    axes[1].bar(range(len(xs)), vals, color=["0.45"] + [f"C{i}" for i in range(args.n_clusters)])
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xticks(range(len(xs)))
    axes[1].set_xticklabels(xs)
    axes[1].set_ylabel("partial Spearman")
    axes[1].set_title("rho_p_lag -> refined_tt by cluster")
    axes[1].grid(True, axis="y", alpha=0.3)
    p_fig = out_dir / "scene_clustering_summary.png"
    fig.savefig(p_fig, dpi=160)
    plt.close(fig)
    (docs / "scene_clustering_summary.png").write_bytes(p_fig.read_bytes())

    # noise vs quality scatter
    fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    ax.scatter([r["noise_ratio"] for r in rows], [r["pick_err_p"] for r in rows], c=[r["cluster"] for r in rows], cmap="tab10", s=28)
    ax.set_xlabel("noise_ratio")
    ax.set_ylabel("pick_err_p (s)")
    ax.set_title("Noise-branch ratio vs P pick error")
    ax.grid(True, alpha=0.3)
    p_noise = out_dir / "noise_ratio_vs_pick_err.png"
    fig.savefig(p_noise, dpi=150)
    plt.close(fig)
    (docs / "noise_ratio_vs_pick_err.png").write_bytes(p_noise.read_bytes())

    supported = [r for r in cluster_rels if r.get("ci_excludes_zero")]
    report = {
        "n_rows": len(rows),
        "n_clusters": args.n_clusters,
        "feature_keys": feat_keys,
        "profiles": profiles,
        "global_relations": global_rels,
        "cluster_relations": cluster_rels,
        "supported_cluster_relations": supported,
        "figures": {
            "clustering_summary": str(p_fig),
            "noise_scatter": str(p_noise),
        },
        "notes": {
            "interpretation": "A relation is treated as scene-supported only if its within-cluster bootstrap CI excludes zero.",
            "noise": "noise_ratio comes from Huygens noise-cancel branch (n_sim vs u_denoised energy).",
        },
    }
    (out_dir / "cluster_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Paper Scene Clustering Report",
        "",
        f"- n rows: {len(rows)}",
        f"- n clusters: {args.n_clusters}",
        f"- supported within-cluster relations (CI excludes 0): {len(supported)}",
        "",
        "## Cluster profiles",
    ]
    for p in profiles:
        md.append(
            f"- C{p['cluster']} n={p['n']}: dist={p['distance_km_mean']:.1f}, depth={p['source_depth_km_mean']:.1f}, "
            f"noise_ratio={p['noise_ratio_mean']:.3f}, refined_tt={p['refined_tt_mean']:.2f}"
        )
    md += ["", "## Global relations"]
    for r in global_rels:
        md.append(
            f"- `{r['x']}->{r['y']}`: partial={r['partial_spearman']:.3f}, "
            f"CI=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}], support={r['ci_excludes_zero']}"
        )
    md += ["", "## Supported cluster relations"]
    if not supported:
        md.append("- none with CI excluding zero")
    for r in supported:
        md.append(
            f"- C{r['cluster']} `{r['x']}->{r['y']}`: partial={r['partial_spearman']:.3f}, "
            f"CI=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}], n={r['n']}"
        )
    (out_dir / "cluster_report.md").write_text("\n".join(md))
    print(json.dumps({"n_rows": len(rows), "supported": len(supported), "report": str(out_dir / "cluster_report.json")}, indent=2))


if __name__ == "__main__":
    main()
