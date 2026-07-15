#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proof-of-superiority package:
  1) STEAD geometry-aware inv05-real (macro vs perturb refine)
  2) Unified synthetic baselines + paired significance (Route A2 style)
  3) Latent visualization: rho / envelope / kernel wave_speed / paths
  4) Training curve + comparison dashboards

Usage:
  python scripts/inversion/run_proof_suite.py --device cuda --max-events 48 --n-synth 32
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.acoustic_fwi_1d import DirectWaveForward, invert_acoustic_fwi
from hnf.inversion_1d import LayeredEarth1D, default_synth_model, model_rmse, travel_time_phase
from hnf.inversion_baselines import invert_gauss_newton, invert_hnf_adam, invert_lbfgs_torch
from hnf.inv_plot import perturb_initial
from hnf.picking_metrics import idx_to_sec
from hnf.picking_prior import run_picking_on_batch
from hnf.ray_paths import direct_ray_path
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.dual_path_inversion import DualPathInversionBridge, load_dual_path_bridge
from hnf.stead_zhizi_inversion_dataset import encode_geometry_tensor
from hnf.physics_decoder import (
    PhysicsDecoder,
    load_physics_decoder_from_checkpoint,
    load_physics_head_state,
)
from hnf.zhizi_inversion_dataset import ZhiziInversionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Proof suite for Zhizi inversion claims")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_bridge_macro/best_physics_head.pt")
    p.add_argument(
        "--physics-head-stead",
        default="",
        help="Geo head for STEAD (dual-path); defaults to --physics-head when set alone",
    )
    p.add_argument(
        "--physics-head-synth",
        default="",
        help="Macro head for synthetic Route A2 (dual-path)",
    )
    p.add_argument(
        "--dual-path",
        action="store_true",
        help="STEAD=physics-head-stead or mixed_geo; synth=macro baseline",
    )
    p.add_argument("--head-mode", choices=["residual", "macro"], default="macro")
    p.add_argument("--output-dir", default="outputs/proof_suite")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-events", type=int, default=48)
    p.add_argument("--n-synth", type=int, default=32)
    p.add_argument("--fwi-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--obs-fallback", action="store_true", default=True)
    p.add_argument("--pick-fallback-sec", type=float, default=0.5)
    p.add_argument("--skip-stead", action="store_true")
    p.add_argument("--skip-synth", action="store_true")
    p.add_argument("--skip-latent", action="store_true")
    return p.parse_args()


def time_misfit(earth, source_depth, distances, obs_tp, obs_ts):
    src = torch.tensor(source_depth, dtype=earth.vp.dtype, device=earth.vp.device)
    tp = travel_time_phase(earth, "P", src, distances)
    ts = travel_time_phase(earth, "S", src, distances)
    return float(torch.mean((tp - obs_tp) ** 2) + torch.mean((ts - obs_ts) ** 2))


def refine_tt(depths, vp0, vs0, q0, source_depth, distances, obs_tp, obs_ts):
    res = invert_lbfgs_torch(
        depths, vp0, vs0, q0, source_depth, distances, obs_tp, obs_ts, max_iter=80
    )
    if not torch.isfinite(res.earth.vp).all():
        res = invert_hnf_adam(
            depths, vp0, vs0, q0, source_depth, distances,
            {"tp": obs_tp, "ts": obs_ts}, steps=400,
        )
    return res


def wilcoxon_greater(a: list[float], b: list[float]) -> dict:
    """Paired Wilcoxon signed-rank style: H1 = median(a-b) < 0  (a better if lower error)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return {"n": int(n), "stat": None, "p_approx": None, "note": "too_few"}
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    w = min(w_pos, w_neg)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w - mean) / math.sqrt(var)
    # one-sided-ish normal approx for visualization only
    p = 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "n": int(n),
        "w_min": w,
        "w_pos": w_pos,
        "w_neg": w_neg,
        "z_approx": z,
        "p_approx_two_sided": p,
        "mean_delta_a_minus_b": float(np.mean(np.asarray(a) - np.asarray(b))),
    }


def plot_training_curves(history_path: Path, out_path: Path) -> None:
    hist = json.loads(history_path.read_text())
    epochs = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    keys = [
        ("val", "rmse_vp_rmse", "Val Vp RMSE"),
        ("val", "loss", "Val total loss"),
        ("val", "loss_unrolled_vp", "Val unrolled Vp MSE"),
    ]
    for ax, (split, key, title) in zip(axes, keys):
        ys = [h[split].get(key, float("nan")) for h in hist]
        ax.plot(epochs, ys, marker="o", ms=3)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latent_panel(
    out_path: Path,
    t: np.ndarray,
    waveform: np.ndarray,
    rho: np.ndarray,
    env: np.ndarray,
    p_prob: np.ndarray,
    s_prob: np.ndarray,
    gt_p,  # Optional[float]
    gt_s,  # Optional[float]
    meta: dict,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, waveform[:, 2], color="0.2", lw=0.8, label="Z")
    axes[0].set_ylabel("amp")
    axes[0].set_title(
        f"trace={meta.get('trace_name','?')}  dist={meta.get('distance_km', float('nan')):.1f}km  "
        f"depth={meta.get('source_depth_km', float('nan')):.1f}km"
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(t, rho, color="C3", label="rho(t) latent density")
    axes[1].set_ylabel("rho")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t, env, color="C0", label="wavefield envelope")
    axes[2].set_ylabel("env")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[3].plot(t, p_prob, color="C2", label="P pick")
    axes[3].plot(t, s_prob, color="C1", label="S pick")
    if gt_p is not None:
        axes[3].axvline(gt_p, color="C2", ls="--", alpha=0.7)
    if gt_s is not None:
        axes[3].axvline(gt_s, color="C1", ls="--", alpha=0.7)
    axes[3].set_ylabel("pick")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_latent_viz(args, bridge, backbone, device, out_dir: Path) -> dict:
    ds = STEADPickingDataset("test", seq_len=800, max_event_traces=200)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    plotted = 0
    for batch in loader:
        if plotted >= 8:
            break
        if float(batch["det"][0]) <= 0.5:
            continue
        if batch["p_valid"][0] <= 0:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        with torch.no_grad():
            feat = bridge.extract_station_features(x, t, include_picks=True)
            picks = run_picking_on_batch(backbone, x, t, infer_seq_len=None)
        t_np = t[:, 0].detach().cpu().numpy() if t.dim() == 2 else t.detach().cpu().numpy().reshape(-1)
        if t_np.ndim > 1:
            t_np = t_np[:, 0]
        rho = feat["rho"][0].detach().cpu().numpy()
        hr = feat["h_real"][0].detach().cpu().numpy()
        hi = feat["h_imag"][0].detach().cpu().numpy()
        env = np.sqrt((hr ** 2 + hi ** 2).sum(axis=-1) + 1e-8)
        p_prob = 1.0 / (1.0 + np.exp(-feat["p_logits"][0].detach().cpu().numpy()))
        s_prob = 1.0 / (1.0 + np.exp(-feat["s_logits"][0].detach().cpu().numpy()))
        x_np = x[0].detach().cpu().numpy()
        # align lengths if bridge downsampled
        seq = min(len(t_np), len(rho), len(env), x_np.shape[0], len(p_prob))
        t_use = np.linspace(0, 60, seq)
        gt_p = idx_to_sec(int(batch["p_idx"][0]), x.shape[1]) if batch["p_valid"][0] > 0 else None
        gt_s = idx_to_sec(int(batch["s_idx"][0]), x.shape[1]) if batch["s_valid"][0] > 0 else None
        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        meta = {
            "trace_name": batch["trace_name"][0] if isinstance(batch["trace_name"], (list, tuple)) else str(batch["trace_name"][0]),
            "distance_km": dist,
            "source_depth_km": depth,
            "kernel_vp": float(feat["kernel_vp"][0]),
            "kernel_vs": float(feat["kernel_vs"][0]),
            "rho_mean": float(np.mean(rho[:seq])),
            "rho_std": float(np.std(rho[:seq])),
            "env_peak": float(np.max(env[:seq])),
        }
        plot_latent_panel(
            out_dir / f"latent_case_{plotted:02d}.png",
            t_use,
            x_np[:seq],
            rho[:seq],
            env[:seq],
            p_prob[:seq],
            s_prob[:seq],
            gt_p,
            gt_s,
            meta,
        )
        rows.append(meta)
        plotted += 1

    # aggregate rho vs distance
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        dvals = [r["distance_km"] for r in rows if np.isfinite(r["distance_km"])]
        rvals = [r["rho_mean"] for r in rows if np.isfinite(r["distance_km"])]
        ax.scatter(dvals, rvals, c="C3")
        ax.set_xlabel("source_distance_km")
        ax.set_ylabel("mean rho(t)")
        ax.set_title("Latent density vs epicentral distance (sample)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "rho_vs_distance.png", dpi=140)
        plt.close(fig)
    return {"n_latent_cases": len(rows), "cases": rows}


def run_stead_geom(args, bridge, backbone, device, out_dir: Path) -> dict:
    ds = STEADPickingDataset("test", seq_len=800)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    base = default_synth_model(device)
    vp_pert, vs_pert, q_init = perturb_initial(base.vp, base.vs, base.q, seed=42, q_scale=1.0)

    pick_p_err, pick_s_err = [], []
    rows = []
    n_seen = 0
    for batch in loader:
        if n_seen >= args.max_events:
            break
        if float(batch["det"][0]) <= 0.5:
            continue
        if batch["p_valid"][0] <= 0 or batch["s_valid"][0] <= 0:
            continue
        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        if not np.isfinite(dist) or not np.isfinite(depth):
            continue
        if dist < 1.0 or dist > 200.0:
            continue
        depth = max(depth, 1.0)

        n_seen += 1
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        gt_p = idx_to_sec(int(batch["p_idx"][0]), x.shape[1])
        gt_s = idx_to_sec(int(batch["s_idx"][0]), x.shape[1])
        picks = run_picking_on_batch(backbone, x, t, infer_seq_len=None)
        tp, ts = picks["tp_sec"][0], picks["ts_sec"][0]
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

        distances = torch.tensor([dist], device=device, dtype=torch.float32)
        obs_tp = torch.tensor([tp], device=device)
        obs_ts = torch.tensor([ts], device=device)
        geo = (
            encode_geometry_tensor(dist, depth, device=device)
            if getattr(bridge, "geo_condition", False)
            else None
        )
        with torch.no_grad():
            out, _ = bridge.forward_event(x, t, include_picks=True, geo=geo)
        zh = bridge.physics_head.earth(out, base.depths, base.q)
        zh_init_tt = time_misfit(zh, depth, distances, obs_tp, obs_ts)
        pe_earth = LayeredEarth1D(base.depths, vp_pert, vs_pert, q_init)
        pe_init_tt = time_misfit(pe_earth, depth, distances, obs_tp, obs_ts)
        zh_ref = refine_tt(base.depths, zh.vp.detach(), zh.vs.detach(), q_init, depth, distances, obs_tp, obs_ts)
        pe_ref = refine_tt(base.depths, vp_pert, vs_pert, q_init, depth, distances, obs_tp, obs_ts)
        rows.append({
            "idx": n_seen - 1,
            "distance_km": dist,
            "source_depth_km": depth,
            "pick_err_p": abs((picks["tp_sec"][0] or gt_p) - gt_p) if picks["tp_sec"][0] is not None else None,
            "pick_err_s": abs((picks["ts_sec"][0] or gt_s) - gt_s) if picks["ts_sec"][0] is not None else None,
            "zhizi_init_tt": zh_init_tt,
            "perturb_init_tt": pe_init_tt,
            "zhizi_refined_tt": zh_ref.time_misfit,
            "perturb_refined_tt": pe_ref.time_misfit,
            "zhizi_refine_wins": zh_ref.time_misfit < pe_ref.time_misfit,
        })
        print(
            f"[stead {n_seen}/{args.max_events}] d={dist:.1f}km z={depth:.1f}km "
            f"ref z={zh_ref.time_misfit:.4f} p={pe_ref.time_misfit:.4f}",
            flush=True,
        )

    zh_ref = [r["zhizi_refined_tt"] for r in rows]
    pe_ref = [r["perturb_refined_tt"] for r in rows]
    summary = {
        "n": len(rows),
        "pick_mae_p": float(np.mean(pick_p_err)) if pick_p_err else None,
        "pick_mae_s": float(np.mean(pick_s_err)) if pick_s_err else None,
        "zhizi_refined_tt_mean": float(np.mean(zh_ref)) if rows else None,
        "perturb_refined_tt_mean": float(np.mean(pe_ref)) if rows else None,
        "zhizi_refine_win_frac": float(np.mean([r["zhizi_refine_wins"] for r in rows])) if rows else None,
        "wilcoxon_zhizi_vs_perturb_refined_tt": wilcoxon_greater(zh_ref, pe_ref),
        "events": rows,
    }
    (out_dir / "stead_geom_report.json").write_text(json.dumps(summary, indent=2))

    # scatter plot
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(pe_ref, zh_ref, alpha=0.7, c="C0")
        lim = max(max(pe_ref + zh_ref), 1e-3)
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("perturb refined TT misfit")
        ax.set_ylabel("Zhizi refined TT misfit")
        ax.set_title(f"STEAD geom-aware refine (n={len(rows)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "stead_refine_scatter.png", dpi=140)
        plt.close(fig)
    return summary


def run_synth_full_compare(args, bridge, device, out_dir: Path) -> dict:
    ds = ZhiziInversionDataset(n_samples=args.n_synth, seq_len=600, seed=args.seed, device=device)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    for idx, batch in enumerate(loader):
        x = batch["x"][0].to(device)
        t = batch["t"].to(device)
        true_vp = batch["true_vp"][0].to(device)
        true_vs = batch["true_vs"][0].to(device)
        true_q = batch["true_q"][0].to(device)
        depths = batch["depths"][0].to(device)
        distances = batch["distances"][0].to(device)
        source_depth = float(batch["source_depth"][0])
        true_earth = LayeredEarth1D(depths, true_vp, true_vs, true_q)
        engine = DirectWaveForward(device=device, nt=x.shape[0], dt=60.0 / max(x.shape[0] - 1, 1))
        observed = engine.simulate(true_earth, source_depth, distances)
        obs_tt = {
            "tp": travel_time_phase(true_earth, "P", torch.tensor(source_depth, device=device), distances),
            "ts": travel_time_phase(true_earth, "S", torch.tensor(source_depth, device=device), distances),
        }
        base = default_synth_model(device)
        vp_pert, vs_pert, q_init = perturb_initial(base.vp, base.vs, base.q, seed=args.seed + idx * 17, q_scale=1.0)
        with torch.no_grad():
            out, _ = bridge.forward_event(x, t, include_picks=True)
        zh = bridge.physics_head.earth(out, depths, true_q)

        methods = {}
        # wave refine from zhizi / perturb
        zh_w, _, _ = invert_acoustic_fwi(depths, zh.vp.detach(), zh.vs.detach(), q_init, true_earth, source_depth, distances, observed, steps=args.fwi_steps)
        pe_w, _, _ = invert_acoustic_fwi(depths, vp_pert, vs_pert, q_init, true_earth, source_depth, distances, observed, steps=args.fwi_steps)
        methods["zhizi_wave"] = model_rmse(true_earth, zh_w.earth)["vp_rmse"]
        methods["perturb_wave"] = model_rmse(true_earth, pe_w.earth)["vp_rmse"]
        methods["zhizi_init"] = model_rmse(true_earth, zh)["vp_rmse"]
        methods["perturb_init"] = model_rmse(true_earth, LayeredEarth1D(depths, vp_pert, vs_pert, q_init))["vp_rmse"]

        gn = invert_gauss_newton(depths, vp_pert, vs_pert, q_init, source_depth, distances, obs_tt["tp"], obs_tt["ts"], n_iter=25)
        methods["gn_tt"] = model_rmse(true_earth, gn.earth)["vp_rmse"]
        lbfgs = invert_lbfgs_torch(depths, vp_pert, vs_pert, q_init, source_depth, distances, obs_tt["tp"], obs_tt["ts"], max_iter=80)
        methods["lbfgs_tt"] = model_rmse(true_earth, lbfgs.earth)["vp_rmse"]
        adam = invert_hnf_adam(depths, vp_pert, vs_pert, q_init, source_depth, distances, obs_tt, steps=400)
        methods["adam_tt"] = model_rmse(true_earth, adam.earth)["vp_rmse"]

        methods["idx"] = idx
        rows.append(methods)
        print(
            f"[synth {idx+1}/{args.n_synth}] zh_wave={methods['zhizi_wave']:.3f} "
            f"pe_wave={methods['perturb_wave']:.3f} gn={methods['gn_tt']:.3f}",
            flush=True,
        )

        if idx == 0:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
            for ax, earth, title in [
                (axes[0], true_earth, "True"),
                (axes[1], zh, "Zhizi init"),
                (axes[2], zh_w.earth, "Zhizi+wave"),
            ]:
                for phase, ls in [("P", "-"), ("S", "--")]:
                    for d in distances[:4]:
                        xx, zz = direct_ray_path(earth, phase, source_depth, float(d))
                        ax.plot(xx.detach().cpu(), zz.detach().cpu(), ls, alpha=0.7)
                ax.invert_yaxis()
                ax.set_title(title)
                ax.set_xlabel("x (km)")
                ax.grid(True, alpha=0.2)
            axes[0].set_ylabel("z (km)")
            fig.tight_layout()
            fig.savefig(out_dir / "example_paths.png", dpi=140)
            plt.close(fig)

    def col(k):
        return [r[k] for r in rows]

    summary = {
        "n": len(rows),
        "means": {k: float(np.mean(col(k))) for k in rows[0] if k != "idx"},
        "medians": {k: float(np.median(col(k))) for k in rows[0] if k != "idx"},
        "zhizi_wave_better_than_perturb_frac": float(np.mean([r["zhizi_wave"] < r["perturb_wave"] for r in rows])),
        "wilcoxon_zhizi_wave_vs_perturb_wave": wilcoxon_greater(col("zhizi_wave"), col("perturb_wave")),
        "wilcoxon_zhizi_wave_vs_gn": wilcoxon_greater(col("zhizi_wave"), col("gn_tt")),
        "per_event": rows,
    }
    (out_dir / "synth_full_compare.json").write_text(json.dumps(summary, indent=2))

    # bar chart
    labels = ["zhizi_init", "perturb_init", "zhizi_wave", "perturb_wave", "gn_tt", "lbfgs_tt", "adam_tt"]
    fig, ax = plt.subplots(figsize=(9, 4))
    vals = [summary["means"][k] for k in labels]
    ax.bar(range(len(labels)), vals, color=["C0", "C1", "C0", "C1", "C2", "C3", "C4"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Vp RMSE mean")
    ax.set_title(f"Synthetic full compare (n={len(rows)})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "synth_full_compare_bars.png", dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = out_dir / "latent"
    latent_dir.mkdir(exist_ok=True)

    backbone, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=True)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    n_layers = default_synth_model(device).n_layers

    dual: DualPathInversionBridge | None = None
    bridge: PhysicsDecoder
    if args.dual_path or args.physics_head_stead or args.physics_head_synth:
        stead_head = args.physics_head_stead or args.physics_head
        synth_head = args.physics_head_synth or "outputs/zhizi_inversion_bridge_macro/best_physics_head.pt"
        dual = load_dual_path_bridge(
            backbone, device,
            stead_head=stead_head,
            synth_head=synth_head,
            embed_dim=embed_dim,
            n_layers=n_layers,
        )
        bridge = dual.stead
        report_heads = {
            "dual_path": True,
            "physics_head_stead": dual.stead_head,
            "physics_head_synth": dual.synth_head,
        }
    else:
        bridge = load_physics_decoder_from_checkpoint(
            backbone, args.physics_head, device,
            head_mode=args.head_mode,
            embed_dim=embed_dim,
            n_layers=n_layers,
        )
        report_heads = {"physics_head": args.physics_head}

    report: dict = {
        "checkpoint": args.checkpoint,
        "head_mode": args.head_mode,
        **report_heads,
    }

    hist = Path("outputs/zhizi_inversion_bridge_macro/history.json")
    if hist.exists():
        plot_training_curves(hist, out_dir / "training_curves.png")
        report["training_curves"] = str(out_dir / "training_curves.png")

    if not args.skip_latent:
        print("[proof] latent visualization...", flush=True)
        report["latent"] = run_latent_viz(args, bridge, backbone, device, latent_dir)
    if not args.skip_stead:
        print("[proof] STEAD geometry-aware evaluation...", flush=True)
        report["stead"] = run_stead_geom(args, bridge, backbone, device, out_dir)
        # trim events for top-level readability
        stead_slim = {k: v for k, v in report["stead"].items() if k != "events"}
        report["stead_summary"] = stead_slim
    if not args.skip_synth:
        print("[proof] synthetic full compare...", flush=True)
        synth_bridge = dual.synth if dual is not None else bridge
        report["synth"] = run_synth_full_compare(args, synth_bridge, device, out_dir)
        report["synth_summary"] = {k: v for k, v in report["synth"].items() if k != "per_event"}

    slim = {k: v for k, v in report.items() if k not in ("stead", "synth", "latent")}
    slim["stead_summary"] = report.get("stead_summary")
    slim["synth_summary"] = report.get("synth_summary")
    slim["latent_n"] = (report.get("latent") or {}).get("n_latent_cases")
    stead_s = report.get("stead_summary") or {}
    synth_s = report.get("synth_summary") or {}
    slim["verdict"] = {
        "stead_geom_gate": (
            "PASS" if (stead_s.get("zhizi_refine_win_frac") or 0) >= 0.65 else "FAIL"
        ),
        "route_a2_wave_init": (
            "PASS"
            if (synth_s.get("zhizi_wave_better_than_perturb_frac") or 0) >= 0.8
            else "FAIL"
        ),
        "dual_path": bool(dual is not None),
        "absolute_tt_oracle": (
            "GN/LBFGS still lower VpRMSE on synth travel-time oracle; "
            "Zhizi claim is FWI-lite init (Route A2), not TT-oracle solve."
        ),
    }
    (out_dir / "proof_report.json").write_text(json.dumps(slim, indent=2))
    print(json.dumps({
        "stead_summary": report.get("stead_summary"),
        "synth_summary": report.get("synth_summary"),
        "latent_n": slim["latent_n"],
        "verdict": slim["verdict"],
    }, indent=2))
    print(f"[proof] -> {out_dir}")


if __name__ == "__main__":
    main()
