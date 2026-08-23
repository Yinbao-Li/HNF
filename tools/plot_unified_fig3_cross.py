#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 3 — Θ* landings + EEG/SST cases + Θ(λ) continuum."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.eeg_geometry import electrode_xyz, pairwise_chord_distance
from hnf.propagation_dynamics.sst_sensor_dataset import build_sensor_coords

C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
BG = "#FFFFFF"
C_WAVE = "#0072B2"
C_INST = "#E69F00"
C_GDIFF = "#009E73"
C_DIFF = "#D55E00"
C_DAMP = "#CC79A7"
C_STEAD = "#0072B2"
C_EEG = "#E69F00"
C_SST = "#009E73"

REGIME_ORDER = ["wave", "instantaneous", "graph_diffusion", "diffusion"]
REGIME_LABEL = {
    "wave": "wave",
    "instantaneous": "instantaneous",
    "graph_diffusion": "graph diff.",
    "diffusion": "diffusion",
}
REGIME_COLOR = {
    "wave": C_WAVE,
    "instantaneous": C_INST,
    "graph_diffusion": C_GDIFF,
    "diffusion": C_DIFF,
}
COORD_KEYS = ["delay", "spatial_mix", "heat_mix", "norm_mix"]
COORD_SHORT = {
    "delay": r"$\delta$",
    "spatial_mix": r"$\sigma$",
    "heat_mix": r"$\eta$",
    "norm_mix": r"$\nu$",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/figures/unified")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--stead", default="outputs/propagation_dynamics/unified_stead_regime_ablation_v1/AGGREGATE.json")
    p.add_argument("--eeg", default="outputs/propagation_dynamics/unified_empirical_instances_v1/eeg_summary.json")
    p.add_argument("--sst", default="outputs/propagation_dynamics/unified_empirical_instances_v1/sst_summary.json")
    p.add_argument("--eeg-ablation", default="outputs/propagation_dynamics/eeg_regime_ablation_v1")
    p.add_argument("--sst-ablation", default="outputs/propagation_dynamics/sst_regime_ablation_v1")
    p.add_argument("--eeg-case", default="sub-008_t50176")
    p.add_argument("--sst-case", default="sst_t1500")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--theta-lambda", default="outputs/propagation_dynamics/unified_theta_lambda_v1/wave_to_diffusion.json")
    return p.parse_args()


def resolve(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else _REPO / p


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "axes.linewidth": 0.7,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, width=0.6, colors=C_INK)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK)
        ax.spines[side].set_linewidth(0.7)


def hdr(ax, letter, title):
    ax.set_axis_off()
    # Centered in the title band so title→panel gap is controlled by GridSpec, not text y.
    ax.text(0.0, 0.42, letter, fontsize=11, fontweight="bold", color=C_INK, va="center", transform=ax.transAxes)
    ax.text(0.07, 0.42, title, fontsize=8.0, color=C_INK, va="center", transform=ax.transAxes)



def add_cbar(im, ax, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4.2%", pad=0.05)
    cb = ax.figure.colorbar(im, cax=cax)
    if label:
        cb.set_label(label, fontsize=5.6)
    cb.ax.tick_params(labelsize=5.0, length=1.6, width=0.5)
    return cb


def telegraph_impulse(t, delay, damping, heat):
    t = np.asarray(t, float)
    t0 = 0.15 + 1.55 * delay
    width = 0.08 + 0.22 * (1.0 - 0.7 * delay) + 0.35 * heat
    ball = np.exp(-0.5 * ((t - t0) / max(width, 1e-3)) ** 2)
    ball *= np.exp(-1.8 * damping * max(t0, 0.0))
    local = heat * np.exp(-((t - 0.05) / 0.12) ** 2)
    g = np.maximum((1.0 - 0.55 * heat) * ball + local, 0.0)
    m = float(g.max()) if g.max() > 0 else 1.0
    return g / m


def unified_rows(summary):
    return [r for r in (summary.get("rows") or []) if r.get("backend") == "unified"]


def domain_pack(stead, eeg, sst):
    packs = []
    for name, blob, color in (("STEAD", stead, C_STEAD), ("EEG", eeg, C_EEG), ("SST", sst, C_SST)):
        v = blob.get("verdict") or blob
        rows = {r["regime"]: r for r in unified_rows(blob)}
        preferred = v["preferred_unified"]
        mses = {k: float(rows[k]["mse_mean"]) for k in REGIME_ORDER}
        best = min(mses.values())
        lags = {k: float(rows[k]["mean_lag"]) for k in REGIME_ORDER}
        coords_src = rows[preferred].get("regime_coords") or v.get("regime_coords") or {}
        packs.append({
            "name": name,
            "color": color,
            "preferred": preferred,
            "delta": {k: mses[k] - best for k in REGIME_ORDER},
            "lags": lags,
            "coords": {k: float(coords_src.get(k, 0.0)) for k in COORD_KEYS},
        })
    return packs


def load_case(root: Path, prior: str, seed: int, sid: str):
    path = root / prior / f"seed_{seed}" / "alpha" / f"{sid}.npz"
    z = np.load(path)
    return {
        "amp": np.asarray(z["prior_amp"], float),
        "lag": np.asarray(z["prior_lag"], float),
        "prior": str(z["prior"]),
        "source_id": str(z["source_id"]),
    }


def draw_a(ax_heat, ax_bar, packs):
    clean(ax_heat)
    mat = np.array([[p["delta"][r] for r in REGIME_ORDER] for p in packs], float)
    mat_n = mat / np.maximum(mat.max(axis=1, keepdims=True), 1e-9)
    cmap = LinearSegmentedColormap.from_list("delta", ["#FFFFFF", "#FEE2E2", "#FCA5A5", "#DC2626", "#7F1D1D"], N=128)
    im = ax_heat.imshow(mat_n, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_heat.set_xticks(range(4))
    ax_heat.set_xticklabels([REGIME_LABEL[r] for r in REGIME_ORDER], rotation=0, ha="center")
    ax_heat.set_yticks(range(3))
    ax_heat.set_yticklabels([p["name"] for p in packs])
    for i, p in enumerate(packs):
        for j, r in enumerate(REGIME_ORDER):
            d = p["delta"][r]
            txt = "★" if r == p["preferred"] else f"{d:.3f}"
            col = C_INK if mat_n[i, j] < 0.55 else "#FFFFFF"
            ax_heat.text(j, i, txt, ha="center", va="center",
                         fontsize=6.2 if r == p["preferred"] else 5.6, color=col,
                         fontweight="bold" if r == p["preferred"] else "normal")
        ax_heat.yaxis.get_ticklabels()[i].set_color(p["color"])
        ax_heat.yaxis.get_ticklabels()[i].set_fontweight("bold")
    ax_heat.set_xlim(-0.5, 3.5)
    ax_heat.set_ylim(2.5, -0.5)
    ax_heat.set_title(r"$\Delta$MSE excess (★ = $\Theta^{\star}$)", fontsize=6.2, color=C_MUTED, pad=2)
    add_cbar(im, ax_heat, "rel. excess")

    clean(ax_bar)
    y = np.arange(len(packs))[::-1]
    ax_bar.set_ylim(-0.70, len(packs) - 0.05)
    ax_bar.set_xlim(0.0, 1.28)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(
        [f"{p['name']} → {REGIME_LABEL[p['preferred']]}" for p in packs[::-1]], fontsize=6.2
    )
    for k, key in enumerate(COORD_KEYS):
        ax_bar.text(
            0.02 + k * 0.22 + 0.45,
            len(packs) - 0.18,
            COORD_SHORT[key],
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=C_MUTED,
        )
    for yi, p in zip(y, packs[::-1]):
        for k, key in enumerate(COORD_KEYS):
            val = p["coords"].get(key, 0.0)
            x0 = 0.02 + k * 0.22
            ax_bar.barh(
                yi,
                min(val, 1.0) * 0.90,
                left=x0,
                height=0.42,
                color=REGIME_COLOR[p["preferred"]],
                alpha=0.85 if val > 0.05 else 0.22,
                edgecolor=C_INK,
                linewidth=0.4,
            )
            ax_bar.text(x0 + 0.45, yi, f"{val:.0f}", ha="center", va="center", fontsize=5.8, color=C_INK)
        lag = p["lags"][p["preferred"]]
        ax_bar.text(1.10, yi, f"lag={lag:.2f}", ha="left", va="center", fontsize=6.0, color=C_MUTED)
    ax_bar.set_xticks([])
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.set_xlabel(r"$\Theta^{\star}$ $(\delta,\sigma,\eta,\nu)$ + lag", fontsize=6.2, labelpad=8)


def draw_eeg_case(axes, root, sid, seed):
    ax_amp, ax_lw, ax_li, ax_prof = axes
    inst = load_case(root, "instantaneous", seed, sid)
    wave = load_case(root, "wave", seed, sid)

    amp = inst["amp"].copy()
    np.fill_diagonal(amp, np.nan)
    cmap_a = LinearSegmentedColormap.from_list("amp", ["#F8FAFC", "#93C5FD", "#1D4ED8", "#1E3A8A"], N=128)
    im = ax_amp.imshow(amp, cmap=cmap_a, vmin=0, vmax=max(float(np.nanpercentile(amp, 98)), 1e-6),
                       interpolation="nearest", aspect="equal")
    ax_amp.set_box_aspect(1)
    ax_amp.set_title(r"EEG $A_{ij}$", fontsize=6.4, pad=2)
    ax_amp.set_xticks([]); ax_amp.set_yticks([])
    for s in ax_amp.spines.values():
        s.set_color(C_LINE)
    add_cbar(im, ax_amp)

    cmap_l = LinearSegmentedColormap.from_list("lag", ["#FFF7ED", "#FDBA74", "#EA580C", "#7C2D12"], N=128)
    for ax, case, title, favor in (
        (ax_lw, wave, r"wave $\tau_{ij}$", False),
        (ax_li, inst, r"inst. $\tau\equiv0$", True),
    ):
        L = case["lag"].astype(float)
        np.fill_diagonal(L, np.nan)
        iml = ax.imshow(L, cmap=cmap_l, vmin=0, vmax=max(13.0, float(np.nanmax(L))),
                        interpolation="nearest", aspect="equal")
        ax.set_box_aspect(1)
        ax.set_title(title, fontsize=6.2, pad=2, color=C_INST if favor else C_WAVE)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(1.3 if favor else 0.6)
            s.set_color(C_INST if favor else C_LINE)
        cb = add_cbar(iml, ax)
        cb.set_label("samples", fontsize=5.2)

    clean(ax_prof)
    xyz = electrode_xyz()
    D = np.asarray(pairwise_chord_distance(xyz), float)
    seed_i = int(np.argmax(np.nansum(inst["amp"], axis=1)))
    order = np.argsort(D[seed_i])
    dist = D[seed_i, order]
    ax_prof.plot(dist, wave["amp"][seed_i, order], color=C_WAVE, lw=1.5, label="amp", zorder=2)
    ax2 = ax_prof.twinx()
    ax2.spines["top"].set_visible(False)
    lag_row = wave["lag"][seed_i, order]
    ax2.vlines(dist, 0, lag_row, color=C_WAVE, alpha=0.30, lw=0.9)
    ax2.plot(dist, lag_row, "o", color=C_WAVE, ms=2.2, label="wave lag")
    ax2.axhline(0, color=C_INST, lw=1.1, ls="--", label="inst. lag")
    ax_prof.set_xlabel("chord distance from seed", fontsize=6.2)
    ax_prof.set_ylabel(r"$A$", fontsize=6.6)
    ax2.set_ylabel("lag (samples)", fontsize=5.8, color=C_WAVE, labelpad=2)
    ax2.yaxis.set_label_coords(1.14, 0.45)
    ax2.tick_params(axis="y", colors=C_WAVE, labelsize=5.2)
    ax_prof.set_title(f"case {sid}", fontsize=6.4, color=C_MUTED, pad=3)
    h1, l1 = ax_prof.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_prof.legend(
        h1 + h2, l1 + l2, fontsize=5.0, loc="upper left",
        bbox_to_anchor=(0.38, 0.98), borderaxespad=0.0, ncol=1,
        frameon=True, fancybox=False, edgecolor=C_LINE, framealpha=0.95,
        handlelength=1.2, borderpad=0.35, labelspacing=0.25,
    )


def draw_sst_case(axes, root, sid, seed):
    ax_g, ax_w, ax_prof = axes
    gdiff = load_case(root, "graph_diffusion", seed, sid)
    wave = load_case(root, "wave", seed, sid)
    coords = build_sensor_coords().detach().cpu().numpy()
    order = np.argsort(np.arctan2(coords[:, 1], coords[:, 0]))

    def show_A(ax, A, title, favor=False, color=C_GDIFF):
        M = A[np.ix_(order, order)].copy()
        np.fill_diagonal(M, np.nan)
        lo, hi = np.nanpercentile(M, [5, 99])
        cmap = (LinearSegmentedColormap.from_list("sst", ["#F0FDF4", "#86EFAC", "#16A34A", "#14532D"], N=128)
                if favor else
                LinearSegmentedColormap.from_list("w", ["#EFF6FF", "#93C5FD", "#1D4ED8", "#1E3A8A"], N=128))
        im = ax.imshow(np.log1p(np.maximum(M, 0)), cmap=cmap,
                       vmin=np.log1p(max(float(lo), 0)), vmax=np.log1p(max(float(hi), 1e-6)),
                       interpolation="nearest", aspect="equal")
        ax.set_box_aspect(1)
        ax.set_title(title, fontsize=6.2, pad=2, color=color if favor else C_WAVE)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(1.3 if favor else 0.6)
            s.set_color(color if favor else C_LINE)
        cb = add_cbar(im, ax)
        cb.set_label(r"$\log(1+A)$", fontsize=5.2)

    show_A(ax_g, gdiff["amp"], r"SST $A$ graph diff. ($\Theta^{\star}$)", favor=True, color=C_GDIFF)
    show_A(ax_w, wave["amp"], r"SST $A$ wave (wrong)", favor=False)

    clean(ax_prof)
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    seed_i = int(np.argmin(np.sum((coords - coords.mean(0)) ** 2, axis=1)))
    order_d = np.argsort(dmat[seed_i])
    dist = dmat[seed_i, order_d]
    ag = gdiff["amp"][seed_i, order_d]
    aw = wave["amp"][seed_i, order_d]
    ax_prof.plot(dist, ag / (ag.max() + 1e-12), color=C_GDIFF, lw=1.7, label="graph diffusion")
    ax_prof.plot(dist, aw / (aw.max() + 1e-12), color=C_WAVE, lw=1.4, ls="--", label="wave")
    lag_row = wave["lag"][seed_i, order_d]
    if np.any(lag_row > 0):
        ax_prof.scatter(dist[lag_row > 0], (aw / (aw.max() + 1e-12))[lag_row > 0],
                        s=7, color=C_WAVE, zorder=3, label="wave lag>0")
    ax_prof.set_xlabel("sensor distance from seed", fontsize=6.2)
    ax_prof.set_ylabel("normalized $A$", fontsize=6.6)
    ax_prof.set_title(f"case {sid}", fontsize=6.4, color=C_MUTED, pad=3)
    ax_prof.set_ylim(-0.02, 1.08)
    ax_prof.legend(fontsize=5.0, loc="upper right", frameon=True, fancybox=False,
                   edgecolor=C_LINE, framealpha=0.92, handlelength=1.2,
                   borderpad=0.3, labelspacing=0.2)


def draw_d(ax_img, ax_curve, theta_rows):
    t = np.linspace(0.0, 3.2, 320)
    mats, lams, phases = [], [], []
    for row in theta_rows:
        proxy = row.get("telegraph_proxy") or row.get("regime_coords") or {}
        mats.append(telegraph_impulse(t, float(proxy.get("delay", 0.0)),
                                      float(proxy.get("damping", 0.0)),
                                      float(proxy.get("heat_mix", 0.0))))
        lams.append(float(row["lambda"]))
        phases.append(str(row.get("phase", "")))
    G = np.asarray(mats)
    cmap = LinearSegmentedColormap.from_list("green", ["#FFFFFF", "#DBEAFE", "#93C5FD", "#1D4ED8", "#1E3A8A"], N=128)
    im = ax_img.imshow(G, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], lams[0], lams[-1]],
                       cmap=cmap, interpolation="bilinear", vmin=0, vmax=1)
    peaks = [t[int(np.argmax(row))] for row in G]
    ax_img.plot(peaks, lams, color="#F97316", lw=1.1, ls="--")
    ax_img.set_xlabel("time after impulse", fontsize=6.6)
    ax_img.set_ylabel(r"path $\lambda$", fontsize=6.6)
    ax_img.set_title(r"Dense Green fingerprint $G(t;\lambda)$", fontsize=6.6, pad=6)

    phase_col = {
        "wave_like": C_WAVE,
        "damped_wave": C_DAMP,
        "instantaneous_like": C_INST,
        "diffusive": C_DIFF,
    }
    # Labels sit in the near-empty late-time region to avoid covering the ballistic lobe.
    shown = set()
    for lam, ph in zip(lams, phases):
        if not ph or ph in shown:
            continue
        shown.add(ph)
        ax_img.text(
            t[-1] - 0.04, lam, ph.replace("_", "-"),
            ha="right", va="center", fontsize=5.2,
            color=phase_col.get(ph, C_MUTED),
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
        )
    add_cbar(im, ax_img, "norm. env.")

    clean(ax_curve)
    ax2 = ax_curve.twinx()
    ax2.spines["top"].set_visible(False)
    lam = np.array(lams, float)
    lag = np.array([float(r["mean_lag"]) for r in theta_rows], float)
    damp = np.array([
        float((r.get("regime_coords") or {}).get("damping", (r.get("telegraph_proxy") or {}).get("damping", 0.0)))
        for r in theta_rows
    ], float)
    peak_t = np.array([float((r.get("impulse") or {}).get("mean_peak_time", np.nan)) for r in theta_rows], float)
    ax_curve.plot(lam, lag, color=C_WAVE, lw=1.7, marker="o", ms=2.5, label="mean lag")
    ax_curve.fill_between(lam, 0, damp * 0.55, color=C_DAMP, alpha=0.15, label="damping (scaled)")
    ax2.plot(lam, peak_t, color="#F97316", lw=1.3, ls="--", marker="s", ms=2.2, label="impulse peak time")
    ax_curve.set_xlabel(r"$\lambda$", fontsize=6.6)
    ax_curve.set_ylabel("mean lag (samples)", fontsize=6.4)
    ax2.set_ylabel("impulse peak time", fontsize=6.0, color="#C2410C")
    ax2.tick_params(axis="y", colors="#C2410C", labelsize=5.2)
    ax_curve.set_xlim(0, 1)
    ax_curve.set_title("Empirical continuum observables", fontsize=6.2, color=C_MUTED, pad=4)
    h1, l1 = ax_curve.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_curve.legend(h1 + h2, l1 + l2, fontsize=5.0, loc="upper right",
                    frameon=True, fancybox=False, edgecolor=C_LINE, framealpha=0.92,
                    handlelength=1.2, borderpad=0.3, labelspacing=0.2)


def main():
    args = parse_args()
    style()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    packs = domain_pack(load_json(resolve(args.stead)), load_json(resolve(args.eeg)), load_json(resolve(args.sst)))
    theta_rows = load_json(resolve(args.theta_lambda))["rows"]

    # Row1 = a|b, Row2 = c|d. Shared title rhythm; row1 gets a slightly larger
    # upper↔lower gap; row2 matrix row is shorter so square kernels are not stretched.
    title_h, title_gap = 0.10, 0.12
    row1_ratios, row1_gap = (1.10, 1.00), 0.28
    row2_ratios, row2_gap = (0.78, 1.00), 0.22
    fig = plt.figure(figsize=(12.4, 12.2))
    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.00, 1.05],
        left=0.12, right=0.93, top=0.945, bottom=0.07, hspace=0.20,
    )
    top = outer[0].subgridspec(2, 2, height_ratios=[title_h, 1.0], hspace=title_gap, wspace=0.40)
    bot = outer[1].subgridspec(2, 2, height_ratios=[title_h, 1.0], hspace=title_gap, wspace=0.40)

    ax_ta = fig.add_subplot(top[0, 0])
    ax_tb = fig.add_subplot(top[0, 1])
    hdr(ax_ta, "a", r"Cross-domain $\Theta^{\star}$ under one unified operator")
    hdr(ax_tb, "b", r"Telegrapher path $\Theta(\lambda)$: Green fingerprint")

    gs_a = top[1, 0].subgridspec(2, 1, height_ratios=list(row1_ratios), hspace=row1_gap)
    ax_heat = fig.add_subplot(gs_a[0])
    ax_bar = fig.add_subplot(gs_a[1])
    draw_a(ax_heat, ax_bar, packs)

    gs_b = top[1, 1].subgridspec(2, 1, height_ratios=list(row1_ratios), hspace=row1_gap)
    ax_img = fig.add_subplot(gs_b[0])
    ax_curve = fig.add_subplot(gs_b[1])
    draw_d(ax_img, ax_curve, theta_rows)

    ax_tc = fig.add_subplot(bot[0, 0])
    ax_td = fig.add_subplot(bot[0, 1])
    hdr(ax_tc, "c", r"EEG case: instantaneous $\Theta^{\star}$ vs forced wave lag")
    hdr(ax_td, "d", r"SST case: graph-diffusion $\Theta^{\star}$ vs wave amp")

    gs_c = bot[1, 0].subgridspec(2, 3, height_ratios=list(row2_ratios), hspace=row2_gap, wspace=0.32)
    ax_c = [
        fig.add_subplot(gs_c[0, 0]),
        fig.add_subplot(gs_c[0, 1]),
        fig.add_subplot(gs_c[0, 2]),
        fig.add_subplot(gs_c[1, :]),
    ]
    draw_eeg_case(ax_c, resolve(args.eeg_ablation), args.eeg_case, args.seed)

    gs_d = bot[1, 1].subgridspec(2, 2, height_ratios=list(row2_ratios), hspace=row2_gap, wspace=0.30)
    ax_d = [
        fig.add_subplot(gs_d[0, 0]),
        fig.add_subplot(gs_d[0, 1]),
        fig.add_subplot(gs_d[1, :]),
    ]
    draw_sst_case(ax_d, resolve(args.sst_ablation), args.sst_case, args.seed)

    png = out_dir / "fig3_cross.png"
    pdf = out_dir / "fig3_cross.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print({"png": str(png), "pdf": str(pdf)})


if __name__ == "__main__":
    main()
