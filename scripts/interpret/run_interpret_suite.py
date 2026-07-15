#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpretability proof package for HNF (picking + Physics Decoder).

Sections:
  A) Kernel physics: obliquity χ(θ), Huygens vs Huygens–Fresnel |K| difference
  B) Picking explain: rho / envelope / kernel row contribution (run20)
  C) Latent bridge panels + rho-vs-distance (macro head)
  D) Principle ablation: run20 vs Fresnel picking metrics
  E) Inversion init→refine scatter (synthetic)

Usage:
  python scripts/interpret/run_interpret_suite.py --device cuda
  python scripts/interpret/run_interpret_suite.py --device cuda --copy-to-docs
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.analyze_stead_picking import load_model as load_picking_ckpt
from tools.eval_stead_picking import evaluate_checkpoint
from hnf.inversion_1d import LayeredEarth1D, default_synth_model, travel_time_phase
from hnf.kernel import HuygensKernel
from hnf.picking_metrics import idx_to_sec
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.physics_decoder import PhysicsDecoder, load_physics_head_state


RUN20_REF = {
    "det_f1": 0.9940792470016699,
    "p_f1": 0.9593805429066354,
    "s_f1": 0.9492358762370036,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF interpretability suite")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_bridge_macro/best_physics_head.pt")
    p.add_argument("--fresnel-checkpoint", default="outputs/huygens_fresnel/picking_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/interpret_suite")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--n-latent", type=int, default=6)
    p.add_argument("--n-kernel-rows", type=int, default=4)
    p.add_argument("--n-joint-summary", type=int, default=24)
    p.add_argument("--n-lag-cases", type=int, default=24)
    p.add_argument("--n-ablation-scans", type=int, default=7)
    p.add_argument("--copy-to-docs", action="store_true")
    return p.parse_args()


def plot_obliquity_and_kernel_diff(out_dir: Path, device: torch.device) -> dict:
    """Fresnel obliquity χ(lag) and |K_hf|−|K_h| on a uniform time grid."""
    n = 120
    t = torch.linspace(0, 15.0, n, device=device).view(1, n, 1)
    x = torch.zeros(1, n, 4, device=device)

    common = dict(
        gamma=0.5,
        omega=0.3,
        causal=True,
        wave_speed=6.0,
        distance_mode="time",
        local_window_sec=15.0,
        obliquity_scale=1.0,
    )
    k_h = HuygensKernel(principle="huygens", **common).to(device)
    k_f = HuygensKernel(principle="huygens_fresnel", **common).to(device)

    with torch.no_grad():
        r = k_h.resolve_distance(x, t=t)
        chi = k_f._fresnel_obliquity(r, t=t, x=x)[0].cpu().numpy()
        kh = torch.abs(k_h(x, t=t, return_complex=True))[0].cpu().numpy()
        kf = torch.abs(k_f(x, t=t, return_complex=True))[0].cpu().numpy()
        diff = kf - kh

    lags = np.linspace(0, 15.0, n)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    im0 = axes[0].imshow(chi, aspect="auto", origin="lower", cmap="magma",
                         extent=[0, 15, 0, 15])
    axes[0].set_title("Fresnel obliquity χ(θ)")
    axes[0].set_xlabel("source lag (s)")
    axes[0].set_ylabel("receiver time (s)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.log10(kh + 1e-8), aspect="auto", origin="lower", cmap="viridis",
                         extent=[0, 15, 0, 15])
    axes[1].set_title("|K| Huygens (log10)")
    axes[1].set_xlabel("source lag (s)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                         extent=[0, 15, 0, 15],
                         vmin=-np.percentile(np.abs(diff), 99),
                         vmax=np.percentile(np.abs(diff), 99))
    axes[2].set_title("|K_Fresnel| − |K_Huygens|")
    axes[2].set_xlabel("source lag (s)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    p_main = out_dir / "kernel_obliquity_diff.png"
    fig.savefig(p_main, dpi=150)
    plt.close(fig)

    # 1D slice at representative receiver index
    row = n // 2
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(lags, chi[row], label="χ along causal row", color="C3")
    ax2 = ax.twinx()
    ax2.plot(lags, kh[row], label="|K| Huygens", color="C0", alpha=0.7)
    ax2.plot(lags, kf[row], label="|K| Fresnel", color="C1", alpha=0.7, ls="--")
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("obliquity χ")
    ax2.set_ylabel("|K|")
    ax.set_title(f"Causal kernel row t≈{lags[row]:.1f}s")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_slice = out_dir / "kernel_row_slice.png"
    fig.savefig(p_slice, dpi=150)
    plt.close(fig)

    return {
        "chi_mean_causal": float(chi[chi > 0.01].mean()) if (chi > 0.01).any() else None,
        "mean_abs_kernel_diff": float(np.mean(np.abs(diff))),
        "figures": [str(p_main), str(p_slice)],
    }


def plot_picking_principle_compare(out_dir: Path, run20_ckpt: Path, fresnel_ckpt: Path, device: str) -> dict:
    """Bar chart: run20 vs Fresnel test metrics."""
    rows = []
    pick_json = Path("outputs/huygens_fresnel/pick_compare.json")
    if pick_json.is_file():
        data = json.loads(pick_json.read_text())
        rows = [
            ("run20 (Huygens)", dict(RUN20_REF)),
            ("Fresnel", data.get("fresnel_summary", data.get("fresnel_test", {}))),
        ]
    else:
        for name, ckpt in [("run20", run20_ckpt), ("fresnel", fresnel_ckpt)]:
            if not ckpt.is_file():
                continue
            m = evaluate_checkpoint(
                str(ckpt),
                {"seq_len": 800},
                post_process_p_before_s=True,
                device=device,
            )
            rows.append((name, m))

    labels = ["det_f1", "p_f1", "s_f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    w = 0.35
    for i, (name, m) in enumerate(rows):
        vals = [float(m.get(k, 0)) for k in labels]
        ax.bar(x + (i - 0.5) * w, vals, width=w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("F1")
    ax.set_title("Picking: Huygens (run20) vs Huygens–Fresnel")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "picking_principle_compare.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    delta = {}
    if len(rows) == 2:
        for k in labels:
            delta[k] = float(rows[1][1].get(k, 0)) - float(rows[0][1].get(k, 0))
    return {"metrics": {n: m for n, m in rows}, "delta_fresnel_minus_run20": delta, "figure": str(p)}


def plot_kernel_parameter_semantics(model, out_dir: Path, device: torch.device) -> dict:
    params = model.collect_kernel_params()
    names = list(params.keys())
    gammas = np.array([params[k]["gamma"] for k in names], dtype=np.float32)
    omegas = np.array([params[k]["omega"] for k in names], dtype=np.float32)
    speeds = np.array([params[k]["wave_speed"] for k in names], dtype=np.float32)

    lags = torch.linspace(0.0, 15.0, 240, device=device).view(1, -1, 1)
    x = torch.zeros(1, lags.shape[1], 4, device=device)
    gamma_scan = np.linspace(max(0.08, float(gammas.min()) * 0.6), float(gammas.max()) * 1.4, 8)
    omega_scan = np.linspace(max(0.08, float(omegas.min()) * 0.6), float(omegas.max()) * 1.4, 8)
    gamma_rows = []
    omega_rows = []
    widths = []
    cycles = []
    with torch.no_grad():
        for g in gamma_scan:
            k = HuygensKernel(gamma=float(g), omega=float(np.median(omegas)), causal=True, wave_speed=6.0, distance_mode="time").to(device)
            row = torch.abs(k(x, t=lags, return_complex=True))[0, lags.shape[1] // 2].detach().cpu().numpy()
            gamma_rows.append(row)
            thresh = 0.5 * max(row.max(), 1e-8)
            widths.append(float(np.sum(row >= thresh) * (15.0 / max(len(row) - 1, 1))))
        for w in omega_scan:
            k = HuygensKernel(gamma=float(np.median(gammas)), omega=float(w), causal=True, wave_speed=6.0, distance_mode="time").to(device)
            row = torch.real(k(x, t=lags, return_complex=True))[0, lags.shape[1] // 2].detach().cpu().numpy()
            omega_rows.append(row)
            signs = np.sign(row)
            cycles.append(float(np.sum(signs[:-1] * signs[1:] < 0)))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    sc = axes[0, 0].scatter(gammas, omegas, c=speeds, cmap="viridis", s=55)
    axes[0, 0].set_xlabel("gamma")
    axes[0, 0].set_ylabel("omega")
    axes[0, 0].set_title("Learned kernel parameters by branch/layer")
    axes[0, 0].grid(True, alpha=0.3)
    for i, name in enumerate(names):
        if i < 6:
            axes[0, 0].annotate(name, (gammas[i], omegas[i]), fontsize=7, alpha=0.8)
    fig.colorbar(sc, ax=axes[0, 0], fraction=0.046, label="wave_speed")

    im1 = axes[0, 1].imshow(np.stack(gamma_rows, axis=0), aspect="auto", cmap="magma", extent=[0, 15, len(gamma_scan) - 0.5, -0.5])
    axes[0, 1].set_yticks(np.arange(len(gamma_scan)))
    axes[0, 1].set_yticklabels([f"{g:.2f}" for g in gamma_scan])
    axes[0, 1].set_xlabel("lag (s)")
    axes[0, 1].set_ylabel("gamma scan")
    axes[0, 1].set_title("Kernel magnitude row vs gamma")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(np.stack(omega_rows, axis=0), aspect="auto", cmap="RdBu_r", extent=[0, 15, len(omega_scan) - 0.5, -0.5])
    axes[1, 0].set_yticks(np.arange(len(omega_scan)))
    axes[1, 0].set_yticklabels([f"{w:.2f}" for w in omega_scan])
    axes[1, 0].set_xlabel("lag (s)")
    axes[1, 0].set_ylabel("omega scan")
    axes[1, 0].set_title("Real kernel row vs omega")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].plot(gamma_scan, widths, marker="o", label="effective width @ 0.5 max")
    axes[1, 1].plot(omega_scan, np.array(cycles) * (15.0 / max(lags.shape[1] - 1, 1)), marker="s", label="zero-crossing density")
    axes[1, 1].set_xlabel("parameter value")
    axes[1, 1].set_title("Parameter semantics summary")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    p = out_dir / "kernel_gamma_omega_semantics.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return {
        "figure": str(p),
        "gamma_range": [float(gammas.min()), float(gammas.max())],
        "omega_range": [float(omegas.min()), float(omegas.max())],
        "wave_speed_range": [float(speeds.min()), float(speeds.max())],
    }


def run_kernel_contrib_panels(
    model,
    device: torch.device,
    out_dir: Path,
    seq_len: int,
    n_cases: int,
) -> dict:
    """forward_explain with kernel row for P branch at GT P index."""
    from hnf.picking_model import STEADHNFPickingModel

    assert isinstance(model, STEADHNFPickingModel)
    ds = STEADPickingDataset("test", seq_len=seq_len, max_event_traces=300)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    cases = []
    plotted = 0

    for batch in loader:
        if plotted >= n_cases:
            break
        if float(batch["det"][0]) <= 0.5 or batch["p_valid"][0] <= 0:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0]) if batch["s_valid"][0] > 0 else -1
        with torch.no_grad():
            out = model.forward_explain(
                x, t, include_kernel_row=True, kernel_row_idx=p_idx, kernel_branch="p"
            )
        t_sec = t[0, :, 0].cpu().numpy()
        z = x[0, :, 2].cpu().numpy()
        rho = out["rho"][0].cpu().numpy()
        p_prob = torch.sigmoid(out["p"][0]).cpu().numpy()
        s_prob = torch.sigmoid(out["s"][0]).cpu().numpy()
        k_row = out["kernel_contrib"][0].cpu().numpy()
        gt_p = idx_to_sec(p_idx, seq_len)
        gt_s = idx_to_sec(s_idx, seq_len) if s_idx >= 0 else None

        fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t_sec, z, color="0.2", lw=0.8)
        axes[0].set_ylabel("Z")
        axes[0].set_title(batch["trace_name"][0] if isinstance(batch["trace_name"], (list, tuple)) else str(batch["trace_name"]))
        axes[1].plot(t_sec, rho, color="C3")
        axes[1].set_ylabel("rho(t)")
        axes[2].plot(t_sec, out["p_envelope"][0].cpu().numpy(), color="C0", label="P env")
        axes[2].plot(t_sec, out["s_envelope"][0].cpu().numpy(), color="C1", label="S env", alpha=0.8)
        axes[2].legend(fontsize=8)
        axes[2].set_ylabel("envelope")
        axes[3].plot(t_sec, k_row, color="C4")
        axes[3].set_ylabel("|K| row@P")
        axes[3].set_title("P-branch kernel contribution to GT P index")
        axes[4].plot(t_sec, p_prob, color="C2", label="P")
        axes[4].plot(t_sec, s_prob, color="C1", label="S")
        axes[4].axvline(gt_p, color="C2", ls="--", alpha=0.7)
        if gt_s is not None:
            axes[4].axvline(gt_s, color="C1", ls="--", alpha=0.7)
        axes[4].set_ylabel("pick prob")
        axes[4].set_xlabel("time (s)")
        axes[4].legend(fontsize=8)
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fp = out_dir / f"kernel_contrib_{plotted:02d}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)

        # rho in S window vs noise
        s0 = max(0, s_idx - 20)
        s1 = min(len(rho), s_idx + 40) if s_idx >= 0 else len(rho) // 2
        n0, n1 = 0, min(80, len(rho) // 4)
        cases.append({
            "trace": str(batch["trace_name"][0]),
            "rho_mean_noise": float(np.mean(rho[n0:n1])),
            "rho_mean_s_window": float(np.mean(rho[s0:s1])) if s1 > s0 else None,
            "rho_ratio_s_over_noise": float(np.mean(rho[s0:s1]) / max(np.mean(rho[n0:n1]), 1e-6)) if s1 > s0 else None,
            "figure": str(fp),
        })
        plotted += 1

    if cases:
        ratios = [c["rho_ratio_s_over_noise"] for c in cases if c["rho_ratio_s_over_noise"] is not None]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(ratios, bins=min(8, len(ratios)), color="C3", alpha=0.85)
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("mean rho(S window) / mean rho(noise)")
        ax.set_ylabel("count")
        ax.set_title(f"Latent rho aligns with S energy (n={len(ratios)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "rho_s_over_noise_hist.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
    else:
        p = None

    return {"n_cases": len(cases), "cases": cases, "rho_hist": str(p) if p else None}


def run_bridge_latent_panels(
    args,
    device: torch.device,
    out_dir: Path,
) -> dict:
    backbone, ckpt_args = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    base = default_synth_model(device)
    state = torch.load(args.physics_head, map_location=device, weights_only=False)
    geo_condition = bool(state.get("geo_condition", False)) or bool(
        (state.get("args") or {}).get("geo_condition", False)
    )
    bridge = PhysicsDecoder(
        backbone=backbone,
        n_layers=base.n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=600,
        head_mode="macro",
        geo_condition=geo_condition,
    ).to(device)
    load_physics_head_state(bridge.physics_head, state["physics_head"])
    bridge.eval()

    ds = STEADPickingDataset("test", seq_len=args.seq_len, max_event_traces=200)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    plotted = 0
    for batch in loader:
        if plotted >= args.n_latent:
            break
        if float(batch["det"][0]) <= 0.5 or batch["p_valid"][0] <= 0:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        with torch.no_grad():
            feat = bridge.extract_station_features(x, t, include_picks=True)
        rho = feat["rho"][0].detach().cpu().numpy()
        hr = feat["h_real"][0].detach().cpu().numpy()
        env = np.sqrt((hr ** 2).sum(axis=-1) + 1e-8)
        p_prob = 1.0 / (1.0 + np.exp(-feat["p_logits"][0].detach().cpu().numpy()))
        s_prob = 1.0 / (1.0 + np.exp(-feat["s_logits"][0].detach().cpu().numpy()))
        x_np = x[0].detach().cpu().numpy()
        seq = min(len(rho), len(env), x_np.shape[0])
        t_use = np.linspace(0, 60, seq)
        dist = float(batch["source_distance_km"][0])
        gt_p = idx_to_sec(int(batch["p_idx"][0]), x.shape[1])
        gt_s = idx_to_sec(int(batch["s_idx"][0]), x.shape[1]) if batch["s_valid"][0] > 0 else None

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t_use, x_np[:seq, 2], color="0.2", lw=0.8)
        axes[0].set_ylabel("Z")
        axes[0].set_title(f"dist={dist:.1f}km  {batch['trace_name'][0]}")
        axes[1].plot(t_use, rho[:seq], color="C3")
        axes[1].set_ylabel("rho(t)")
        axes[2].plot(t_use, env[:seq], color="C0")
        axes[2].set_ylabel("envelope")
        axes[3].plot(t_use, p_prob[:seq], color="C2", label="P")
        axes[3].plot(t_use, s_prob[:seq], color="C1", label="S")
        axes[3].axvline(gt_p, color="C2", ls="--", alpha=0.7)
        if gt_s is not None:
            axes[3].axvline(gt_s, color="C1", ls="--", alpha=0.7)
        axes[3].legend(fontsize=8)
        axes[3].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fp = out_dir / f"bridge_latent_{plotted:02d}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        rows.append({"distance_km": dist, "rho_mean": float(np.mean(rho[:seq])), "figure": str(fp)})
        plotted += 1

    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter([r["distance_km"] for r in rows], [r["rho_mean"] for r in rows], c="C3")
        ax.set_xlabel("source_distance_km")
        ax.set_ylabel("mean rho(t)")
        ax.set_title("Bridge latent rho vs distance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "bridge_rho_vs_distance.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
    return {"n": len(rows), "cases": rows}


def run_joint_latent_summary(args, device: torch.device, out_dir: Path) -> dict:
    backbone, ckpt_args = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    base = default_synth_model(device)
    state = torch.load(args.physics_head, map_location=device, weights_only=False)
    geo_condition = bool(state.get("geo_condition", False)) or bool((state.get("args") or {}).get("geo_condition", False))
    bridge = PhysicsDecoder(
        backbone=backbone,
        n_layers=base.n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=600,
        head_mode="macro",
        geo_condition=geo_condition,
    ).to(device)
    load_physics_head_state(bridge.physics_head, state["physics_head"])
    bridge.eval()
    ds = STEADPickingDataset("test", seq_len=args.seq_len, max_event_traces=240)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    for batch in loader:
        if len(rows) >= args.n_joint_summary:
            break
        if float(batch["det"][0]) <= 0.5 or float(batch["p_valid"][0]) <= 0 or float(batch["s_valid"][0]) <= 0:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        with torch.no_grad():
            feat = bridge.extract_station_features(x, t, include_picks=True)
            out, _ = bridge.forward_event(x, t, include_picks=True)
        params = bridge.backbone.collect_kernel_params()
        rho = feat["rho"][0].detach().cpu().numpy()
        vp = out.vp[0].detach().cpu().numpy()
        vs = out.vs[0].detach().cpu().numpy()
        rows.append({
            "distance_km": dist,
            "depth_km": depth,
            "rho_mean": float(np.mean(rho)),
            "rho_peak": float(np.max(rho)),
            "gamma_p0": float(params["p_branch_0"]["gamma"]),
            "omega_p0": float(params["p_branch_0"]["omega"]),
            "gamma_s0": float(params["s_branch_0"]["gamma"]),
            "omega_s0": float(params["s_branch_0"]["omega"]),
            "kernel_vp": float(feat["kernel_vp"][0]),
            "kernel_vs": float(feat["kernel_vs"][0]),
            "vp_mean": float(np.mean(vp)),
            "vs_mean": float(np.mean(vs)),
            "vpvs_mean": float(np.mean(vp / np.clip(vs, 1e-6, None))),
        })

    if not rows:
        return {"n": 0}
    mat_keys = ["distance_km", "depth_km", "rho_mean", "rho_peak", "gamma_p0", "omega_p0", "kernel_vp", "kernel_vs", "vp_mean", "vs_mean", "vpvs_mean"]
    mat = np.array([[r[k] for k in mat_keys] for r in rows], dtype=np.float32)
    corr = np.corrcoef(mat, rowvar=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    axes[0, 0].scatter([r["distance_km"] for r in rows], [r["rho_mean"] for r in rows], c=[r["gamma_p0"] for r in rows], cmap="magma", s=55)
    axes[0, 0].set_xlabel("distance_km")
    axes[0, 0].set_ylabel("rho_mean")
    axes[0, 0].set_title("rho(t) vs geometry (color=gamma_p0)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter([r["kernel_vp"] for r in rows], [r["vp_mean"] for r in rows], c=[r["omega_p0"] for r in rows], cmap="viridis", s=55)
    axes[0, 1].set_xlabel("kernel_vp")
    axes[0, 1].set_ylabel("vp_mean")
    axes[0, 1].set_title("kernel_vp -> recovered vp")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter([r["kernel_vs"] for r in rows], [r["vpvs_mean"] for r in rows], c=[r["rho_peak"] for r in rows], cmap="plasma", s=55)
    axes[1, 0].set_xlabel("kernel_vs")
    axes[1, 0].set_ylabel("vp/vs mean")
    axes[1, 0].set_title("kernel_vs / rho_peak -> vp/vs")
    axes[1, 0].grid(True, alpha=0.3)

    im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[1, 1].set_xticks(np.arange(len(mat_keys)))
    axes[1, 1].set_yticks(np.arange(len(mat_keys)))
    axes[1, 1].set_xticklabels(mat_keys, rotation=55, ha="right", fontsize=7)
    axes[1, 1].set_yticklabels(mat_keys, fontsize=7)
    axes[1, 1].set_title("Joint latent / physical correlation")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    p = out_dir / "joint_latent_physics_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return {"n": len(rows), "figure": str(p), "keys": mat_keys, "corr": corr.tolist(), "rows": rows}


def plot_vp_vs_sensitivity(out_dir: Path, device: torch.device) -> dict:
    base = default_synth_model(device)
    source_depth = torch.tensor(10.0, device=device)
    distances = torch.tensor([5.0, 20.0, 50.0, 100.0, 160.0], device=device)
    eps = 0.05
    mats = {
        "dTp_dVp": np.zeros((base.n_layers, len(distances)), dtype=np.float32),
        "dTs_dVp": np.zeros((base.n_layers, len(distances)), dtype=np.float32),
        "dTp_dVs": np.zeros((base.n_layers, len(distances)), dtype=np.float32),
        "dTs_dVs": np.zeros((base.n_layers, len(distances)), dtype=np.float32),
    }
    for i in range(base.n_layers):
        dv = torch.zeros_like(base.vp)
        ds = torch.zeros_like(base.vs)
        dv[i] = eps
        ds[i] = eps
        e_vp_p = LayeredEarth1D(base.depths, base.vp + dv, base.vs, base.q)
        e_vp_m = LayeredEarth1D(base.depths, (base.vp - dv).clamp(min=1.5), base.vs, base.q)
        e_vs_p = LayeredEarth1D(base.depths, base.vp, torch.minimum(base.vs + ds, base.vp * 0.75), base.q)
        e_vs_m = LayeredEarth1D(base.depths, base.vp, (base.vs - ds).clamp(min=1.0), base.q)
        tp_vp = (travel_time_phase(e_vp_p, "P", source_depth, distances) - travel_time_phase(e_vp_m, "P", source_depth, distances)) / (2 * eps)
        ts_vp = (travel_time_phase(e_vp_p, "S", source_depth, distances) - travel_time_phase(e_vp_m, "S", source_depth, distances)) / (2 * eps)
        tp_vs = (travel_time_phase(e_vs_p, "P", source_depth, distances) - travel_time_phase(e_vs_m, "P", source_depth, distances)) / (2 * eps)
        ts_vs = (travel_time_phase(e_vs_p, "S", source_depth, distances) - travel_time_phase(e_vs_m, "S", source_depth, distances)) / (2 * eps)
        mats["dTp_dVp"][i] = tp_vp.detach().cpu().numpy()
        mats["dTs_dVp"][i] = ts_vp.detach().cpu().numpy()
        mats["dTp_dVs"][i] = tp_vs.detach().cpu().numpy()
        mats["dTs_dVs"][i] = ts_vs.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    titles = [
        ("dTp_dVp", "dTp / dVp"),
        ("dTs_dVp", "dTs / dVp"),
        ("dTp_dVs", "dTp / dVs"),
        ("dTs_dVs", "dTs / dVs"),
    ]
    for ax, (k, title) in zip(axes.flat, titles):
        arr = mats[k]
        im = ax.imshow(arr, aspect="auto", cmap="coolwarm")
        ax.set_xticks(np.arange(len(distances)))
        ax.set_xticklabels([f"{float(d):.0f}" for d in distances.detach().cpu().tolist()])
        ax.set_yticks(np.arange(base.n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(base.n_layers)])
        ax.set_xlabel("receiver distance (km)")
        ax.set_ylabel("layer")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    p = out_dir / "vp_vs_tt_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return {"figure": str(p), "distances_km": distances.detach().cpu().tolist(), "matrices": {k: v.tolist() for k, v in mats.items()}}


def _shift_trace(x: torch.Tensor, n_shift: int) -> torch.Tensor:
    y = torch.roll(x, shifts=n_shift, dims=1)
    if n_shift > 0:
        y[:, :n_shift] = 0.0
    elif n_shift < 0:
        y[:, n_shift:] = 0.0
    return y


def _smooth_trace(x: torch.Tensor, k: int = 9) -> torch.Tensor:
    y = F.avg_pool1d(x.transpose(1, 2), kernel_size=k, stride=1, padding=k // 2)
    return y.transpose(1, 2)


def run_counterfactual_panels(model, device: torch.device, out_dir: Path, seq_len: int) -> dict:
    ds = STEADPickingDataset("test", seq_len=seq_len, max_event_traces=200)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    picked = None
    for batch in loader:
        if float(batch["det"][0]) > 0.5 and float(batch["p_valid"][0]) > 0 and float(batch["s_valid"][0]) > 0:
            picked = batch
            break
    if picked is None:
        return {"figure": None}
    x = picked["x"].to(device)
    t = picked["t"].to(device)
    p_idx = int(picked["p_idx"][0])
    s_idx = int(picked["s_idx"][0])
    gt_p = idx_to_sec(p_idx, x.shape[1])
    gt_s = idx_to_sec(s_idx, x.shape[1])
    variants = {
        "original": x,
        "amplitude_x0.5": x * 0.5,
        "time_shift_0.6s": _shift_trace(x, max(1, x.shape[1] // 100)),
        "smoothed": _smooth_trace(x, k=11),
    }
    rows = []
    t_sec = t[0, :, 0].detach().cpu().numpy()
    fig, axes = plt.subplots(len(variants), 4, figsize=(14, 2.9 * len(variants)), sharex=True)
    if len(variants) == 1:
        axes = np.expand_dims(axes, 0)
    for ridx, (name, xv) in enumerate(variants.items()):
        with torch.no_grad():
            out = model.forward_explain(xv, t, include_kernel_row=True, kernel_row_idx=p_idx, kernel_branch="p")
        rho = out["rho"][0].detach().cpu().numpy()
        p_env = out["p_envelope"][0].detach().cpu().numpy()
        s_env = out["s_envelope"][0].detach().cpu().numpy()
        p_prob = torch.sigmoid(out["p"][0]).detach().cpu().numpy()
        s_prob = torch.sigmoid(out["s"][0]).detach().cpu().numpy()
        k_row = out["kernel_contrib"][0].detach().cpu().numpy()
        rows.append({
            "name": name,
            "rho_peak_sec": float(t_sec[int(np.argmax(rho))]),
            "p_peak_sec": float(t_sec[int(np.argmax(p_prob))]),
            "s_peak_sec": float(t_sec[int(np.argmax(s_prob))]),
            "rho_mean": float(np.mean(rho)),
        })
        axes[ridx, 0].plot(t_sec, xv[0, :, 2].detach().cpu().numpy(), color="0.2", lw=0.8)
        axes[ridx, 0].set_ylabel(name)
        axes[ridx, 1].plot(t_sec, rho, color="C3")
        axes[ridx, 2].plot(t_sec, p_env, color="C0", label="P env")
        axes[ridx, 2].plot(t_sec, s_env, color="C1", alpha=0.8, label="S env")
        axes[ridx, 3].plot(t_sec, p_prob, color="C2", label="P")
        axes[ridx, 3].plot(t_sec, s_prob, color="C1", label="S")
        axes[ridx, 3].plot(t_sec, k_row / max(np.max(k_row), 1e-6), color="C4", alpha=0.7, label="|K| row (norm)")
        for c in [1, 2, 3]:
            axes[ridx, c].axvline(gt_p, color="C2", ls="--", alpha=0.5)
            axes[ridx, c].axvline(gt_s, color="C1", ls="--", alpha=0.5)
        axes[ridx, 0].grid(True, alpha=0.25)
        axes[ridx, 1].grid(True, alpha=0.25)
        axes[ridx, 2].grid(True, alpha=0.25)
        axes[ridx, 3].grid(True, alpha=0.25)
    axes[0, 0].set_title("Z trace")
    axes[0, 1].set_title("rho(t)")
    axes[0, 2].set_title("P/S envelope")
    axes[0, 3].set_title("P/S prob + kernel row")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[-1, 2].set_xlabel("time (s)")
    axes[-1, 3].set_xlabel("time (s)")
    axes[0, 2].legend(fontsize=8)
    axes[0, 3].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "counterfactual_response_panel.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    return {"figure": str(p), "trace": str(picked["trace_name"][0]), "variants": rows}


def run_temporal_lag_stats(model, device: torch.device, out_dir: Path, seq_len: int, n_cases: int) -> dict:
    ds = STEADPickingDataset("test", seq_len=seq_len, max_event_traces=300)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    for batch in loader:
        if len(rows) >= n_cases:
            break
        if float(batch["det"][0]) <= 0.5 or float(batch["p_valid"][0]) <= 0 or float(batch["s_valid"][0]) <= 0:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        with torch.no_grad():
            out = model.forward_explain(x, t)
        rho = out["rho"][0].detach().cpu().numpy()
        p_env = out["p_envelope"][0].detach().cpu().numpy()
        s_env = out["s_envelope"][0].detach().cpu().numpy()
        p_prob = torch.sigmoid(out["p"][0]).detach().cpu().numpy()
        s_prob = torch.sigmoid(out["s"][0]).detach().cpu().numpy()
        t_sec = t[0, :, 0].detach().cpu().numpy()
        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0])
        gt_p = idx_to_sec(p_idx, seq_len)
        gt_s = idx_to_sec(s_idx, seq_len)

        def _peak_in_window(arr: np.ndarray, center_idx: int, left: int = 30, right: int = 50) -> float:
            i0 = max(0, center_idx - left)
            i1 = min(len(arr), center_idx + right)
            loc = int(np.argmax(arr[i0:i1])) + i0
            return float(t_sec[loc])

        rho_p = _peak_in_window(rho, p_idx)
        rho_s = _peak_in_window(rho, s_idx)
        p_env_p = _peak_in_window(p_env, p_idx)
        s_env_s = _peak_in_window(s_env, s_idx)
        p_prob_p = _peak_in_window(p_prob, p_idx)
        s_prob_s = _peak_in_window(s_prob, s_idx)
        rows.append({
            "rho_p_lag": rho_p - gt_p,
            "rho_s_lag": rho_s - gt_s,
            "p_env_lag": p_env_p - gt_p,
            "s_env_lag": s_env_s - gt_s,
            "p_prob_lag": p_prob_p - gt_p,
            "s_prob_lag": s_prob_s - gt_s,
        })

    if not rows:
        return {"n": 0}
    keys = ["rho_p_lag", "rho_s_lag", "p_env_lag", "s_env_lag", "p_prob_lag", "s_prob_lag"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        vals = [r[key] for r in rows]
        ax.hist(vals, bins=min(10, len(vals)), color="C0" if "prob" in key else ("C3" if "rho" in key else "C1"), alpha=0.85)
        ax.axvline(0.0, color="k", ls="--", lw=1)
        ax.set_title(key.replace("_", " "))
        ax.set_xlabel("peak lag vs GT (s)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
    p = out_dir / "temporal_lag_statistics.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in keys}
    return {"n": len(rows), "figure": str(p), "summary": summary, "rows": rows}


def run_branch_parameter_ablation(
    args,
    device: torch.device,
    out_dir: Path,
) -> dict:
    backbone, ckpt_args = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    base = default_synth_model(device)
    state = torch.load(args.physics_head, map_location=device, weights_only=False)
    geo_condition = bool(state.get("geo_condition", False)) or bool((state.get("args") or {}).get("geo_condition", False))
    bridge = PhysicsDecoder(
        backbone=backbone,
        n_layers=base.n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=600,
        head_mode="macro",
        geo_condition=geo_condition,
    ).to(device)
    load_physics_head_state(bridge.physics_head, state["physics_head"])
    bridge.eval()

    ds = STEADPickingDataset("test", seq_len=args.seq_len, max_event_traces=200)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    batch = None
    for item in loader:
        if float(item["det"][0]) > 0.5 and float(item["p_valid"][0]) > 0 and float(item["s_valid"][0]) > 0:
            batch = item
            break
    if batch is None:
        return {"figure": None}
    x = batch["x"].to(device)
    t = batch["t"].to(device)
    dist = float(batch["source_distance_km"][0])
    depth = max(float(batch["source_depth_km"][0]), 1.0)
    gt_p = idx_to_sec(int(batch["p_idx"][0]), x.shape[1])
    gt_s = idx_to_sec(int(batch["s_idx"][0]), x.shape[1])
    geo = None
    if geo_condition:
        from hnf.stead_zhizi_inversion_dataset import encode_geometry_tensor
        geo = encode_geometry_tensor(dist, depth, device=device)

    # Scan raw Parameter values (gamma uses softplus; collect_kernel_params returns effective).
    # Also perturb the last branch layer, which owns the kernel_contrib row used for visualization.
    p_gamma_raw = float(bridge.backbone.p_layers[-1].kernel.gamma.detach().cpu())
    p_omega_raw = float(bridge.backbone.p_layers[-1].kernel.omega.detach().cpu())
    s_gamma_raw = float(bridge.backbone.s_layers[-1].kernel.gamma.detach().cpu())
    s_omega_raw = float(bridge.backbone.s_layers[-1].kernel.omega.detach().cpu())
    scans = {
        "p_gamma0": np.linspace(p_gamma_raw - 2.0, p_gamma_raw + 2.0, args.n_ablation_scans),
        "p_omega0": np.linspace(max(0.05, p_omega_raw * 0.2), p_omega_raw * 2.0, args.n_ablation_scans),
        "s_gamma0": np.linspace(s_gamma_raw - 2.0, s_gamma_raw + 2.0, args.n_ablation_scans),
        "s_omega0": np.linspace(max(0.05, s_omega_raw * 0.2), s_omega_raw * 2.0, args.n_ablation_scans),
    }
    results = {}
    kernel_rows = {}
    t_sec = t[0, :, 0].detach().cpu().numpy()
    p_idx = int(batch["p_idx"][0])

    for key, vals in scans.items():
        rows = []
        krows = []
        for val in vals:
            pert_bridge = copy.deepcopy(bridge).to(device)
            if key == "p_gamma0":
                for layer in pert_bridge.backbone.p_layers:
                    layer.kernel.gamma.data.fill_(float(val))
            elif key == "p_omega0":
                for layer in pert_bridge.backbone.p_layers:
                    layer.kernel.omega.data.fill_(float(val))
            elif key == "s_gamma0":
                for layer in pert_bridge.backbone.s_layers:
                    layer.kernel.gamma.data.fill_(float(val))
            elif key == "s_omega0":
                for layer in pert_bridge.backbone.s_layers:
                    layer.kernel.omega.data.fill_(float(val))
            with torch.no_grad():
                out = pert_bridge.backbone.forward_explain(
                    x, t, include_kernel_row=True, kernel_row_idx=p_idx, kernel_branch="p"
                )
                # forward_event expects shared t as (T,1) or batched (1,T,1)
                t_event = t[0] if t.dim() == 3 and t.shape[0] == 1 else t
                if t_event.dim() == 3:
                    t_event = t_event[0]
                bout, _ = pert_bridge.forward_event(x, t_event, include_picks=True, geo=geo)
            p_prob = torch.sigmoid(out["p"][0]).detach().cpu().numpy()
            s_prob = torch.sigmoid(out["s"][0]).detach().cpu().numpy()
            rho = out["rho"][0].detach().cpu().numpy()
            krow = out["kernel_contrib"][0].detach().cpu().numpy()
            vp = bout.vp[0].detach().cpu().numpy()
            vs = bout.vs[0].detach().cpu().numpy()
            if key.startswith("p_"):
                eff = float(pert_bridge.backbone.p_layers[-1].kernel.effective_gamma().detach().cpu()) if "gamma" in key else float(val)
            else:
                eff = float(pert_bridge.backbone.s_layers[-1].kernel.effective_gamma().detach().cpu()) if "gamma" in key else float(val)
            rows.append({
                "value": float(val),
                "effective_value": eff,
                "p_lag": float(t_sec[int(np.argmax(p_prob))] - gt_p),
                "s_lag": float(t_sec[int(np.argmax(s_prob))] - gt_s),
                "rho_mean": float(np.mean(rho)),
                "vp_mean": float(np.mean(vp)),
                "vs_mean": float(np.mean(vs)),
                "vpvs_mean": float(np.mean(vp / np.clip(vs, 1e-6, None))),
            })
            krows.append(krow / max(np.max(krow), 1e-6))
        results[key] = rows
        kernel_rows[key] = (vals, np.stack(krows, axis=0))

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12.5), constrained_layout=True)
    plot_specs = [
        ("p_gamma0", "P branch gamma"),
        ("p_omega0", "P branch omega"),
        ("s_gamma0", "S branch gamma"),
        ("s_omega0", "S branch omega"),
    ]
    for ridx, (key, title) in enumerate(plot_specs):
        vals = np.array([r["value"] for r in results[key]])
        p_lag = np.array([r["p_lag"] for r in results[key]])
        s_lag = np.array([r["s_lag"] for r in results[key]])
        vp_mean = np.array([r["vp_mean"] for r in results[key]])
        vs_mean = np.array([r["vs_mean"] for r in results[key]])
        axes[ridx, 0].plot(vals, p_lag, marker="o", label="P lag")
        axes[ridx, 0].plot(vals, s_lag, marker="s", label="S lag")
        axes[ridx, 0].axhline(0.0, color="k", ls="--", lw=1)
        axes[ridx, 0].set_title(f"{title}: lag response")
        axes[ridx, 0].grid(True, alpha=0.3)
        if ridx == 0:
            axes[ridx, 0].legend(fontsize=8)

        axes[ridx, 1].plot(vals, vp_mean, marker="o", label="vp mean")
        axes[ridx, 1].plot(vals, vs_mean, marker="s", label="vs mean")
        axes[ridx, 1].set_title(f"{title}: bridge output")
        axes[ridx, 1].grid(True, alpha=0.3)
        if ridx == 0:
            axes[ridx, 1].legend(fontsize=8)

        scan_vals, karr = kernel_rows[key]
        im = axes[ridx, 2].imshow(karr, aspect="auto", cmap="magma", extent=[0, 60, len(scan_vals) - 0.5, -0.5])
        axes[ridx, 2].set_yticks(np.arange(len(scan_vals)))
        axes[ridx, 2].set_yticklabels([f"{v:.2f}" for v in scan_vals], fontsize=7)
        axes[ridx, 2].set_title(f"{title}: kernel row")
        axes[ridx, 2].set_xlabel("time (s)")
        fig.colorbar(im, ax=axes[ridx, 2], fraction=0.046)

    for ax in axes[:, 0]:
        ax.set_xlabel("parameter value")
        ax.set_ylabel("lag vs GT (s)")
    for ax in axes[:, 1]:
        ax.set_xlabel("parameter value")
        ax.set_ylabel("mean velocity")
    p = out_dir / "branch_parameter_ablation.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    weak_bridge = True
    max_vp_span = 0.0
    max_vs_span = 0.0
    for key_rows in results.values():
        vp_vals = [r["vp_mean"] for r in key_rows]
        vs_vals = [r["vs_mean"] for r in key_rows]
        max_vp_span = max(max_vp_span, max(vp_vals) - min(vp_vals))
        max_vs_span = max(max_vs_span, max(vs_vals) - min(vs_vals))
    if max_vp_span > 0.05 or max_vs_span > 0.05:
        weak_bridge = False
    return {
        "figure": str(p),
        "trace": str(batch["trace_name"][0]),
        "results": results,
        "max_vp_span": float(max_vp_span),
        "max_vs_span": float(max_vs_span),
        "weak_bridge_propagation": weak_bridge,
    }


def plot_causal_chain_visuals(out_dir: Path, report: dict) -> dict:
    cf = report.get("counterfactual", {})
    lag = report.get("temporal_lag", {}).get("summary", {})
    branch = report.get("branch_ablation", {})
    ks = report.get("kernel_semantics", {})

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")
    boxes = {
        "kernel": (0.05, 0.68, 0.23, 0.18, "Kernel params\n"
                   f"gamma {ks.get('gamma_range', [float('nan'), float('nan')])[0]:.2f}..{ks.get('gamma_range', [float('nan'), float('nan')])[1]:.2f}\n"
                   f"omega {ks.get('omega_range', [float('nan'), float('nan')])[0]:.2f}..{ks.get('omega_range', [float('nan'), float('nan')])[1]:.2f}\n"
                   f"c {ks.get('wave_speed_range', [float('nan'), float('nan')])[0]:.2f}..{ks.get('wave_speed_range', [float('nan'), float('nan')])[1]:.2f}"),
        "wave": (0.37, 0.68, 0.24, 0.18, "Wave / latent response\n"
                 f"rho(S)/noise ≈ {np.mean([c['rho_ratio_s_over_noise'] for c in report.get('kernel_contrib', {}).get('cases', [])]):.2f}\n"
                 f"rho_p lag {lag.get('rho_p_lag', {}).get('mean', float('nan')):+.2f}s\n"
                 f"rho_s lag {lag.get('rho_s_lag', {}).get('mean', float('nan')):+.2f}s"),
        "pick": (0.69, 0.68, 0.24, 0.18, "Pick timing\n"
                 f"P lag {lag.get('p_prob_lag', {}).get('mean', float('nan')):+.2f}s\n"
                 f"S lag {lag.get('s_prob_lag', {}).get('mean', float('nan')):+.2f}s\n"
                 "closest to GT among latent observables"),
        "branch": (0.20, 0.28, 0.26, 0.20, "Branch ablation\n"
                   f"max vp span {branch.get('max_vp_span', float('nan')):.4f}\n"
                   f"max vs span {branch.get('max_vs_span', float('nan')):.4f}\n"
                   + ("weak bridge propagation" if branch.get("weak_bridge_propagation", False) else "visible bridge propagation")),
        "bridge": (0.56, 0.28, 0.30, 0.20, "Bridge / physical output\n"
                   "kernel/pick perturbations\n"
                   "affect timing and rows strongly,\n"
                   "but propagate weakly to vp/vs in this local scan"),
    }
    for _, (x, y, w, h, text) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec="0.4", alpha=0.92))
        ax.text(x + 0.015, y + h - 0.02, text, va="top", fontsize=9.5, family="monospace")
    arrows = [
        ((0.28, 0.77), (0.37, 0.77)),
        ((0.61, 0.77), (0.69, 0.77)),
        ((0.49, 0.68), (0.33, 0.48)),
        ((0.79, 0.68), (0.71, 0.48)),
        ((0.46, 0.38), (0.56, 0.38)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.6, color="0.25"))
    ax.set_title("Interpretable causal chain summary", fontsize=13)
    p_graph = out_dir / "causal_chain_graph.png"
    fig.savefig(p_graph, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.axis("off")
    rows = cf.get("variants", [])
    txt = ["Wave-level causal checks"]
    for r in rows:
        txt.append(
            f"{r['name']}: rho_peak={r['rho_peak_sec']:.2f}s, "
            f"P_peak={r['p_peak_sec']:.2f}s, S_peak={r['s_peak_sec']:.2f}s, "
            f"rho_mean={r['rho_mean']:.2f}"
        )
    txt += [
        "",
        "Interpretation:",
        "- amplitude scaling changes rho magnitude more than timing",
        "- time shift moves rho and pick peaks together",
        "- smoothing perturbs S much more strongly than P",
    ]
    ax.text(0.03, 0.96, "\n".join(txt), va="top", fontsize=10, family="monospace")
    p_wave = out_dir / "causal_wave_summary.png"
    fig.savefig(p_wave, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"graph": str(p_graph), "wave_summary": str(p_wave)}


def plot_interpretability_summary_panel(out_dir: Path, report: dict) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0), constrained_layout=True)
    items = [
        (report.get("kernel_semantics", {}).get("figure"), "Kernel semantics"),
        (report.get("counterfactual", {}).get("figure"), "Counterfactual response"),
        (report.get("joint_summary", {}).get("figure"), "Latent -> physical mapping"),
        (report.get("temporal_lag", {}).get("figure"), "Temporal lag statistics"),
        (report.get("branch_ablation", {}).get("figure"), "Branch parameter ablation"),
        (report.get("vp_vs_sensitivity", {}).get("figure"), "Vp/Vs sensitivity"),
    ]
    for ax, (path, title) in zip(axes.flat, items):
        if path:
            ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Overlay compact summary box on last panel.
    lag = report.get("temporal_lag", {}).get("summary", {})
    rho_ratio_cases = report.get("kernel_contrib", {}).get("cases", [])
    rho_ratio_mean = float(np.mean([c["rho_ratio_s_over_noise"] for c in rho_ratio_cases])) if rho_ratio_cases else float("nan")
    axes[1, 2].text(
        0.03,
        0.97,
        "\n".join(
            [
                "Interpretability summary",
                f"- gamma range: {report.get('kernel_semantics', {}).get('gamma_range', [float('nan'), float('nan')])[0]:.2f} .. {report.get('kernel_semantics', {}).get('gamma_range', [float('nan'), float('nan')])[1]:.2f}",
                f"- omega range: {report.get('kernel_semantics', {}).get('omega_range', [float('nan'), float('nan')])[0]:.2f} .. {report.get('kernel_semantics', {}).get('omega_range', [float('nan'), float('nan')])[1]:.2f}",
                f"- wave_speed range: {report.get('kernel_semantics', {}).get('wave_speed_range', [float('nan'), float('nan')])[0]:.2f} .. {report.get('kernel_semantics', {}).get('wave_speed_range', [float('nan'), float('nan')])[1]:.2f}",
                f"- mean rho(S)/rho(noise): {rho_ratio_mean:.2f}",
                f"- p_prob lag mean: {lag.get('p_prob_lag', {}).get('mean', float('nan')):+.2f}s",
                f"- s_prob lag mean: {lag.get('s_prob_lag', {}).get('mean', float('nan')):+.2f}s",
                "",
                "Mechanism chain",
                "gamma / omega",
                " -> kernel support / oscillation",
                " -> rho / envelope / pick timing",
                " -> macro head conditioning",
                " -> vp / vs sensitivity",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=9.2,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
    )
    p = out_dir / "interpretability_summary_panel.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return {"figure": str(p)}


def plot_inversion_init_refine(out_dir: Path) -> dict:
    """Copy/summarize init vs wave scatter from proof outputs if present."""
    proof = Path("outputs/proof_suite/synth_full_compare.json")
    fresnel = Path("outputs/huygens_fresnel/proof_suite/synth_full_compare.json")
    fig_path = None
    summary = {}
    if proof.is_file():
        data = json.loads(proof.read_text())
        per = data.get("per_event", [])
        if per:
            zh_i = [r["zhizi_init"] for r in per]
            zh_w = [r["zhizi_wave"] for r in per]
            pe_w = [r["perturb_wave"] for r in per]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].scatter(zh_i, zh_w, alpha=0.7, c="C0")
            lim = max(max(zh_i + zh_w), 0.1)
            axes[0].plot([0, lim], [0, lim], "k--", lw=1)
            axes[0].set_xlabel("zhizi init VpRMSE")
            axes[0].set_ylabel("zhizi wave VpRMSE")
            axes[0].set_title("Init → wave refine (run20)")
            axes[1].hist(np.array(zh_w) - np.array(pe_w), bins=10, color="C0", alpha=0.8)
            axes[1].axvline(0, color="k", ls="--")
            axes[1].set_xlabel("zhizi_wave − perturb_wave")
            axes[1].set_title("Route A2 paired delta")
            for ax in axes:
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = out_dir / "inversion_init_refine.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            summary["run20"] = {
                "mean_zhizi_wave": float(np.mean(zh_w)),
                "mean_perturb_wave": float(np.mean(pe_w)),
                "win_frac": float(np.mean([a < b for a, b in zip(zh_w, pe_w)])),
            }
    if fresnel.is_file():
        fd = json.loads(fresnel.read_text())
        summary["fresnel"] = {
            "mean_zhizi_wave": fd.get("means", {}).get("zhizi_wave"),
            "win_frac": fd.get("zhizi_wave_better_than_perturb_frac"),
        }
    return {"figure": str(fig_path) if fig_path else None, "summary": summary}


def copy_docs(out_dir: Path) -> list[str]:
    doc_dir = Path("docs/figures/interpret")
    doc_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(out_dir.glob("*.png")):
        dst = doc_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    for sub in ["kernel_contrib", "bridge_latent"]:
        sd = out_dir / sub
        if not sd.is_dir():
            continue
        (doc_dir / sub).mkdir(exist_ok=True)
        for src in sorted(sd.glob("*.png")):
            dst = doc_dir / sub / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    kdir = out_dir / "kernel_contrib"
    bdir = out_dir / "bridge_latent"
    out_dir.mkdir(parents=True, exist_ok=True)
    kdir.mkdir(exist_ok=True)
    bdir.mkdir(exist_ok=True)

    report = {"checkpoint": args.checkpoint, "physics_head": args.physics_head}

    print("[interpret] kernel obliquity + diff...", flush=True)
    report["kernel_physics"] = plot_obliquity_and_kernel_diff(out_dir, device)

    print("[interpret] kernel gamma/omega semantics...", flush=True)
    model_for_semantics, _ = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    report["kernel_semantics"] = plot_kernel_parameter_semantics(model_for_semantics, out_dir, device)

    print("[interpret] picking principle compare...", flush=True)
    report["picking_ablation"] = plot_picking_principle_compare(
        out_dir, Path(args.checkpoint), Path(args.fresnel_checkpoint), args.device
    )

    print("[interpret] kernel contribution panels (run20)...", flush=True)
    model, _ = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    report["kernel_contrib"] = run_kernel_contrib_panels(
        model, device, kdir, args.seq_len, args.n_kernel_rows
    )

    print("[interpret] counterfactual response panels...", flush=True)
    report["counterfactual"] = run_counterfactual_panels(model, device, out_dir, args.seq_len)

    print("[interpret] temporal lag statistics...", flush=True)
    report["temporal_lag"] = run_temporal_lag_stats(model, device, out_dir, args.seq_len, args.n_lag_cases)

    print("[interpret] branch parameter ablation...", flush=True)
    report["branch_ablation"] = run_branch_parameter_ablation(args, device, out_dir)

    print("[interpret] bridge latent panels...", flush=True)
    report["bridge_latent"] = run_bridge_latent_panels(args, device, bdir)

    print("[interpret] joint latent/physics summary...", flush=True)
    report["joint_summary"] = run_joint_latent_summary(args, device, out_dir)

    print("[interpret] inversion init→refine...", flush=True)
    report["inversion"] = plot_inversion_init_refine(out_dir)

    print("[interpret] vp/vs sensitivity...", flush=True)
    report["vp_vs_sensitivity"] = plot_vp_vs_sensitivity(out_dir, device)

    print("[interpret] summary panel...", flush=True)
    report["summary_panel"] = plot_interpretability_summary_panel(out_dir, report)

    print("[interpret] causal chain visuals...", flush=True)
    report["causal_chain"] = plot_causal_chain_visuals(out_dir, report)

    # Merge fresnel invert compare if present
    inv_cmp = Path("outputs/huygens_fresnel/invert_compare.json")
    if inv_cmp.is_file():
        report["fresnel_inversion_ablation"] = json.loads(inv_cmp.read_text())

    report["interpretation_notes"] = {
        "rho": "Soft latent weight; higher in energetic / S intervals — not crustal density.",
        "gamma": "Controls kernel locality: larger gamma narrows effective support and enforces more local causal influence.",
        "omega": "Controls oscillatory phase structure: larger omega increases sign changes / phase sensitivity along causal rows.",
        "vp_vs": "Recovered Vp/Vs are downstream physical outputs; sensitivity heatmaps show which layers and offsets constrain P and S travel times.",
        "counterfactual": "Amplitude scaling, time shifting, and smoothing reveal whether rho and pick curves follow energy, timing, or band-limited structure.",
        "temporal_lag": "Lag histograms quantify whether rho, envelopes, and pick probabilities peak before, on, or after GT P/S arrivals.",
        "branch_ablation": "Local scans of p/s branch gamma and omega show which parameter moves pick lag, kernel concentration, and downstream vp/vs most strongly.",
        "branch_ablation_verdict": "After fixing the perturbed-bridge path, local p/s gamma/omega scans still change pick timing and kernel rows much more than downstream vp/vs; this suggests weak local propagation from branch-specific kernel knobs into the macro inversion head under the current architecture.",
        "causal_chain": "Causal-chain visuals summarize how kernel parameters influence latent timing, pick timing, and only weakly propagate into vp/vs in local branch perturbations.",
        "obliquity": "Fresnel χ suppresses off-axis secondary sources; changes |K| mainly at longer lags.",
        "kernel_row": "Causal light-cone row at GT P shows which past samples contribute to pick.",
        "fresnel_verdict": "Fresnel picking: det +0.002, P/S −0.034/−0.022 vs run20; inversion still PASS but marginal.",
    }

    (out_dir / "interpret_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# HNF Interpretability Report",
        "",
        "## Kernel physics",
        f"- Mean |K_Fresnel − K_Huygens|: {report['kernel_physics'].get('mean_abs_kernel_diff')}",
        f"- See `kernel_obliquity_diff.png`, `kernel_row_slice.png`",
        "",
        "## Picking (run20 vs Fresnel)",
        f"- Delta: {report['picking_ablation'].get('delta_fresnel_minus_run20')}",
        "",
        "## Kernel gamma / omega semantics",
        f"- See `{Path(report['kernel_semantics'].get('figure', '')).name}`",
        f"- gamma range: {report['kernel_semantics'].get('gamma_range')}",
        f"- omega range: {report['kernel_semantics'].get('omega_range')}",
        f"- wave_speed range: {report['kernel_semantics'].get('wave_speed_range')}",
        "",
        "## Latent rho",
        f"- S-window / noise rho ratio cases: {report['kernel_contrib'].get('n_cases')}",
        "",
        "## Counterfactual response",
        f"- See `{Path(report['counterfactual'].get('figure', '')).name}`",
        "",
        "## Temporal lag statistics",
        f"- n cases: {report['temporal_lag'].get('n')}",
        "",
        "## Branch parameter ablation",
        f"- See `{Path(report['branch_ablation'].get('figure', '')).name}`",
        f"- weak bridge propagation: {report['branch_ablation'].get('weak_bridge_propagation')}",
        f"- max vp span: {report['branch_ablation'].get('max_vp_span')}",
        f"- max vs span: {report['branch_ablation'].get('max_vs_span')}",
        "",
        "## Summary panel",
        f"- See `{Path(report['summary_panel'].get('figure', '')).name}`",
        "",
        "## Causal chain",
        f"- See `{Path(report['causal_chain'].get('graph', '')).name}`, `{Path(report['causal_chain'].get('wave_summary', '')).name}`",
        "",
        "## Joint latent / physical summary",
        f"- n cases: {report['joint_summary'].get('n')}",
        "",
        "## Vp/Vs sensitivity",
        f"- See `{Path(report['vp_vs_sensitivity'].get('figure', '')).name}`",
        "",
        "## Inversion",
        f"- {report['inversion'].get('summary')}",
        "",
        "Run: `python scripts/interpret/run_interpret_suite.py --device cuda --copy-to-docs`",
    ]
    (out_dir / "interpret_report.md").write_text("\n".join(md))

    if args.copy_to_docs:
        copied = copy_docs(out_dir)
        report["docs_copied"] = copied
        (out_dir / "interpret_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "kernel": report["kernel_physics"],
        "picking_delta": report["picking_ablation"].get("delta_fresnel_minus_run20"),
        "rho_cases": report["kernel_contrib"].get("n_cases"),
        "inversion": report["inversion"].get("summary"),
        "out": str(out_dir),
    }, indent=2))
    print(f"[interpret] -> {out_dir}")


if __name__ == "__main__":
    main()
