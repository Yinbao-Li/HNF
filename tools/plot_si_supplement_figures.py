#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate SI placeholder figures for sn-article.tex.

Outputs (PDF+PNG) under docs/figures/si/:
  decoder_detailed.{pdf,png}
  eeg_residualizer_pipeline.{pdf,png}
  tube_mwd_alignment.{pdf,png}
  fig_s4_stead_examples.{pdf,png}
  fig_s5_fluid_combined.{pdf,png}   # primary SI plate (a–f)
  fig_s5_fluid_3d_reconstruction.{pdf,png}  # optional row-only
  fig_s6_fluid_baselines.{pdf,png}          # optional baselines-only

Example
-------
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/plot_si_supplement_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.causal_chain import TAU_HI, TAU_LO, TAU_N, _resample_to_tau, has_valid_chain
from hnf.stead_picking_dataset import STEAD_DIR
from tools.analyze_stead_picking import load_model
from tools.expand_shape_labels_regional import observables_from_batch

BG = "#FFFFFF"
C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
C_TEAL = "#0F766E"
C_GOLD = "#B45309"
C_BLUE = "#1D4ED8"
C_RED = "#B91C1C"
C_BOX = "#F8FAFC"
C_SOFT = "#F3F4F6"

SHAPE_DISPLAY = {
    "impulsive_fastQ": "impulsive_fast_decay",
}

SHAPE_COLORS = {
    "impulsive_fastQ": "#B91C1C",
    "emergent": "#B45309",
    "multipath": "#1D4ED8",
    "slow_coda": "#0F766E",
    "standard": "#57534E",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/figures/si")
    p.add_argument("--checkpoint", default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--skip-stead", action="store_true")
    p.add_argument("--skip-fluid", action="store_true")
    return p.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BG,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "text.color": C_INK,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
        }
    )


def _save(fig, out: Path, dpi: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[si] → {out.with_suffix('.pdf')}", flush=True)


def _card(
    ax,
    xy,
    w,
    h,
    title,
    *,
    body: str | None = None,
    accent=C_TEAL,
    fc="#FFFFFF",
    title_fs=8.5,
    body_fs=7.0,
):
    """Journal-style card: soft shadow, left accent rail, title + optional body."""
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.007, y - 0.009),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor="#D1D5DB",
            edgecolor="none",
            alpha=0.55,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=fc,
            edgecolor="#CBD5E1",
            linewidth=1.15,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle((x + 0.005, y + 0.05 * h), 0.010, 0.90 * h, facecolor=accent, edgecolor="none", zorder=3)
    )
    tx = x + 0.030
    if body:
        ax.text(tx, y + h * 0.66, title, ha="left", va="center", fontsize=title_fs, fontweight="bold", color=C_INK, zorder=4)
        ax.text(tx, y + h * 0.30, body, ha="left", va="center", fontsize=body_fs, color=C_MUTED, zorder=4, linespacing=1.3)
    else:
        ax.text(
            x + w / 2 + 0.004,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color=C_INK,
            zorder=4,
            linespacing=1.25,
        )
    return {
        "c": (x + w / 2, y + h / 2),
        "r": (x + w, y + h / 2),
        "l": (x, y + h / 2),
        "t": (x + w / 2, y + h),
        "b": (x + w / 2, y),
        "box": (x, y, w, h),
    }


def _arrow(ax, p0, p1, *, color=C_MUTED, rad=0.0):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.35,
            mutation_scale=11,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=5,
    )


def _stage_label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=7.0, color=C_MUTED, fontweight="bold", zorder=4)


# ---------------------------------------------------------------------------
# B. Physics Decoder
# ---------------------------------------------------------------------------
def plot_decoder(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.0), facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Main title block (top)
    ax.text(0.02, 0.985, "Physics Decoder", fontsize=13, fontweight="bold", color=C_INK, va="top")
    ax.text(
        0.02,
        0.940,
        "Frozen probe features  →  event-level pooling  →  layered velocity for FWI",
        fontsize=7.8,
        color=C_MUTED,
        va="top",
    )

    # Stage labels lower (clear of main title); pipeline hugs labels; content sits near footer
    y_stage = 0.820
    y_sep = 0.790
    y_pipe_top = 0.768
    y_band0 = 0.055
    band_h = y_pipe_top - y_band0

    for x0, w in ((0.02, 0.18), (0.26, 0.22), (0.54, 0.20), (0.78, 0.18)):
        ax.add_patch(
            Rectangle((x0, y_band0), w, band_h, facecolor=C_SOFT, edgecolor="none", alpha=0.45, zorder=0)
        )
    ax.plot([0.02, 0.96], [y_sep, y_sep], color=C_LINE, lw=0.8, zorder=1)

    for x, lab in ((0.11, "1  PROBE"), (0.37, "2  EXPORTS"), (0.64, "3  POOL"), (0.87, "4  DECODE")):
        ax.text(x, y_stage, lab, ha="center", va="center", fontsize=7.2, color=C_INK, fontweight="bold", zorder=4)

    feat_h = 0.138
    gap = 0.016
    feats = [
        (y_pipe_top - feat_h, "Wavefield stats", r"$h_r$, $h_i$ envelopes"),
        (y_pipe_top - 2 * (feat_h + gap), r"Density $\rho(t)$", "layer-bucket summaries"),
        (y_pipe_top - 3 * (feat_h + gap), "Kernel params", r"$(\gamma,\omega,c)$ readable"),
        (y_pipe_top - 4 * (feat_h + gap), "P / S picks", "arrival times"),
    ]
    feat_nodes = []
    for y, title, body in feats:
        feat_nodes.append(
            _card(ax, (0.27, y), 0.20, feat_h, title, body=body, accent=C_BLUE, fc="#EFF6FF", title_fs=8.0, body_fs=6.6)
        )

    y_stack0 = feats[-1][0]
    y_stack1 = feats[0][0] + feat_h
    mid_y = 0.5 * (y_stack0 + y_stack1)
    bb_h = 0.40
    bb = _card(
        ax,
        (0.03, mid_y - 0.5 * bb_h),
        0.16,
        bb_h,
        "Frozen PNF\nbackbone",
        body="wave / Fresnel\nprobe (fixed)",
        accent=C_TEAL,
        fc="#ECFDF5",
        title_fs=9,
    )
    pool_h = 0.34
    pool = _card(
        ax,
        (0.55, mid_y - 0.5 * pool_h),
        0.18,
        pool_h,
        "Feature pool",
        body="per-station concat\n→ event-mean",
        accent=C_GOLD,
        fc="#FFFBEB",
        title_fs=9,
    )
    head_h = 0.28
    head = _card(
        ax,
        (0.79, mid_y - 0.5 * head_h + 0.04),
        0.17,
        head_h,
        "Macro head",
        body=r"MLP $48\!\to\!48\!\to\!3$" "\nscale, contrast, $V_S/V_P$",
        accent=C_RED,
        fc="#FEF2F2",
        title_fs=9,
        body_fs=6.8,
    )

    for node in feat_nodes:
        _arrow(ax, bb["r"], node["l"], color=C_TEAL, rad=0.05 if node["c"][1] > mid_y else -0.05)
        _arrow(ax, node["r"], pool["l"], color=C_BLUE, rad=-0.04 if node["c"][1] > mid_y else 0.04)
    _arrow(ax, pool["r"], head["l"], color=C_GOLD)

    # FWI output chip just above footer
    out_y = 0.070
    ax.add_patch(
        FancyBboxPatch(
            (0.79, out_y),
            0.17,
            0.110,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#FFFFFF",
            edgecolor=C_RED,
            linewidth=1.2,
            zorder=2,
        )
    )
    ax.text(0.875, out_y + 0.070, r"layered $v_P,\,v_S$", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(0.875, out_y + 0.032, "FWI initialisation", ha="center", va="center", fontsize=7, color=C_MUTED, zorder=4)
    _arrow(ax, head["b"], (0.875, out_y + 0.110), color=C_RED)

    ax.text(
        0.03,
        0.022,
        "Tanh / softplus maps keep velocities physical; backbone weights stay frozen at inference.",
        fontsize=7.2,
        color=C_MUTED,
    )
    _save(fig, out_dir / "decoder_detailed", dpi)


# ---------------------------------------------------------------------------
# C. EEG residualizer
# ---------------------------------------------------------------------------
def plot_eeg_pipeline(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8), facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, "EEG leftover construction", fontsize=13, fontweight="bold", color=C_INK, va="top")
    ax.text(
        0.02,
        0.915,
        "Train-only residualizer: probe medium orthogonalised against classical voltmeter covariates",
        fontsize=7.8,
        color=C_MUTED,
        va="top",
    )

    for x0, w in ((0.02, 0.16), (0.22, 0.20), (0.48, 0.20), (0.74, 0.22)):
        ax.add_patch(Rectangle((x0, 0.14), w, 0.64, facecolor=C_SOFT, edgecolor="none", alpha=0.4, zorder=0))

    _stage_label(ax, 0.10, 0.76, "INPUT")
    _stage_label(ax, 0.32, 0.76, "PARALLEL READOUTS")
    _stage_label(ax, 0.58, 0.76, "PROBE SCALAR")
    _stage_label(ax, 0.85, 0.76, "RESIDUALIZE")

    raw = _card(
        ax,
        (0.03, 0.32),
        0.14,
        0.36,
        "Raw EEG",
        body=r"$19\times1280$" "\n10 s @ 128 Hz",
        accent=C_TEAL,
        fc="#ECFDF5",
        title_fs=9,
    )
    probe = _card(
        ax,
        (0.23, 0.50),
        0.18,
        0.22,
        "Spatial PNF probe",
        body="Fresnel / anisotropic",
        accent=C_BLUE,
        fc="#EFF6FF",
        title_fs=8.5,
        body_fs=6.8,
    )
    volt = _card(
        ax,
        (0.23, 0.18),
        0.18,
        0.22,
        "Voltmeter",
        body=r"age, sex, $\theta/\alpha$, BP-$\alpha$",
        accent=C_GOLD,
        fc="#FFFBEB",
        title_fs=8.5,
        body_fs=6.8,
    )
    deff = _card(
        ax,
        (0.49, 0.50),
        0.18,
        0.22,
        r"$D_{\mathrm{eff}}$",
        body=r"$1/\rho_{\mathrm{std}}$ + topography",
        accent=C_INK,
        fc="#FFFFFF",
        title_fs=9,
        body_fs=6.8,
    )
    left = _card(
        ax,
        (0.75, 0.30),
        0.20,
        0.38,
        "Train-only OLS",
        body=r"leftover $=D_{\mathrm{eff}}-f(\mathrm{volt})$" "\nheld-out $n{=}31$",
        accent=C_RED,
        fc="#FEF2F2",
        title_fs=9,
        body_fs=6.8,
    )

    _arrow(ax, raw["r"], probe["l"], color=C_TEAL, rad=0.12)
    _arrow(ax, raw["r"], volt["l"], color=C_TEAL, rad=-0.12)
    _arrow(ax, probe["r"], deff["l"], color=C_BLUE)
    _arrow(ax, deff["r"], left["l"], color=C_INK, rad=0.08)
    _arrow(ax, volt["r"], left["l"], color=C_GOLD, rad=-0.12)

    # note strip
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.04),
            0.92,
            0.08,
            boxstyle="round,pad=0.008,rounding_size=0.01",
            facecolor="#F9FAFB",
            edgecolor=C_LINE,
            linewidth=0.8,
            zorder=2,
        )
    )
    ax.text(
        0.49,
        0.08,
        r"LEMON control: within-cohort residualization only — no raw AHEPA $\beta$ transport"
        r"  ($D_{\mathrm{eff}}$ scale ≈0.35 vs ≈1.42).",
        ha="center",
        va="center",
        fontsize=7.2,
        color=C_MUTED,
        zorder=4,
    )
    _save(fig, out_dir / "eeg_residualizer_pipeline", dpi)


# ---------------------------------------------------------------------------
# D. Tube–MWD schematic
# ---------------------------------------------------------------------------
def plot_tube_mwd(out_dir: Path, dpi: int) -> None:
    rng = np.random.default_rng(0)
    lam = np.logspace(-2, 2, 8)
    gk = np.array([0.05, 0.12, 0.28, 0.22, 0.15, 0.10, 0.05, 0.03])
    gk = gk / gk.sum()
    alpha = 3.4
    logM = np.log10(1e5) + (1 / alpha) * (np.log10(lam) - np.average(np.log10(lam), weights=gk))
    # synthetic GPC
    m = np.logspace(3.5, 6.5, 200)
    gpc = np.exp(-0.5 * ((np.log10(m) - 5.1) / 0.35) ** 2)
    gpc = gpc / np.trapz(gpc, np.log10(m))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), facecolor=BG)
    # left: mode masses
    ax = axes[0]
    ax.bar(np.arange(8), gk, color=C_TEAL, edgecolor=C_INK, linewidth=0.5)
    ax.set_xticks(np.arange(8), [f"$\\lambda_{k+1}$" for k in range(8)], fontsize=7)
    ax.set_ylabel(r"mode mass $G_k/\sum G$")
    ax.set_title("(a) PNF / Prony spectrum", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # middle: map
    ax = axes[1]
    ax.scatter(np.log10(lam), logM, s=55, c=C_GOLD, edgecolors=C_INK, zorder=3)
    xx = np.linspace(np.log10(lam).min() - 0.3, np.log10(lam).max() + 0.3, 50)
    yy = np.log10(1e5) + (1 / alpha) * (xx - np.average(np.log10(lam), weights=gk))
    ax.plot(xx, yy, color=C_MUTED, lw=1.2, ls="--")
    ax.set_xlabel(r"$\log_{10}\lambda$")
    ax.set_ylabel(r"$\log_{10} M$")
    ax.set_title(r"(b) Tube map $M\propto\lambda^{1/\alpha}$", loc="left", fontsize=10)
    ax.text(0.05, 0.92, r"$\alpha=3.4$", transform=ax.transAxes, fontsize=8, color=C_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # right: GPC + deposited masses
    ax = axes[2]
    ax.fill_between(np.log10(m), gpc, color="#DBEAFE", alpha=0.9, label="GPC $w(M)$")
    ax.plot(np.log10(m), gpc, color=C_BLUE, lw=1.2)
    # deposit
    dep = np.zeros(8)
    for i, lm in enumerate(logM):
        j = int(np.argmin(np.abs(np.log10(m) - lm)))
        # mass near mode
        dep[i] = gpc[max(0, j - 3) : j + 4].sum()
    dep = dep / (dep.sum() + 1e-12)
    ax2 = ax.twinx()
    ax2.bar(logM, dep, width=0.12, color=C_TEAL, alpha=0.75, edgecolor=C_INK, linewidth=0.4, label="GPC→modes")
    ax.set_xlabel(r"$\log_{10} M$")
    ax.set_ylabel("GPC density", color=C_BLUE)
    ax2.set_ylabel("deposited mass", color=C_TEAL)
    ax.set_title(r"(c) Align → Pearson $r$", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    r = float(np.corrcoef(gk, dep)[0, 1])
    ax.text(0.05, 0.92, f"toy $r={r:.2f}$", transform=ax.transAxes, fontsize=8, color=C_MUTED)

    fig.suptitle("Tube–MWD alignment schematic (illustrative)", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "tube_mwd_alignment", dpi)


# ---------------------------------------------------------------------------
# I. STEAD facies examples
# ---------------------------------------------------------------------------
def plot_stead_examples(out_dir: Path, dpi: int, args: argparse.Namespace) -> None:
    labels = Path("outputs/shape_labels_expanded/socal/traces_labeled.csv")
    if not labels.is_file():
        print("[si] skip STEAD examples: missing labels", flush=True)
        return
    df = pd.read_csv(labels)
    order = ["impulsive_fastQ", "emergent", "multipath", "slow_coda", "standard"]
    picks = []
    for sh in order:
        sub = df[df["shape"] == sh].copy()
        if sub.empty:
            continue
        sub = sub[(sub["ps_gap"] > 2.5) & (sub["ps_gap"] < 35) & (sub["snr_db"] > 8)]
        if sub.empty:
            sub = df[df["shape"] == sh]
        # prefer clear prototypes
        if sh == "multipath":
            sub = sub.sort_values(["n_rho_peaks", "snr_db"], ascending=[False, False])
        elif sh == "slow_coda":
            sub = sub.sort_values(["coda_slope", "snr_db"], ascending=[False, False])
        elif sh == "impulsive_fastQ":
            sub = sub.sort_values(["onset_sharp", "coda_slope"], ascending=[False, True])
        elif sh == "emergent":
            sub = sub.sort_values(["onset_sharp", "snr_db"], ascending=[True, False])
        else:
            sub = sub.sort_values(["onset_sharp", "snr_db"], ascending=[False, False])
        picks.append(sub.iloc[0])

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()
    stead = Path(STEAD_DIR)
    handles = {}

    def wave(chunk, name):
        if chunk not in handles:
            handles[chunk] = h5py.File(stead / f"chunk{chunk}_eofextract" / f"chunk{chunk}.hdf5", "r")
        return np.asarray(handles[chunk]["data"][name][()], dtype=np.float32)

    rows = []
    with torch.no_grad():
        for r in picks:
            w = wave(int(r["chunk"]), str(r["trace_name"]))  # T,3
            x = torch.from_numpy(w).transpose(0, 1)  # 3,T
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std
            x = F.interpolate(x.unsqueeze(0), size=800, mode="linear", align_corners=False).squeeze(0)
            x = x.transpose(0, 1).unsqueeze(0)  # 1,T,3
            t = torch.linspace(0, 60.0, 800).view(1, 800, 1)
            out = model(x.to(device), t.to(device))
            obs = observables_from_batch(out, x.to(device), 0, window_sec=60.0, pick_threshold=0.3)
            tau = np.linspace(TAU_LO, TAU_HI, TAU_N)
            if has_valid_chain(obs):
                rho_tau = _resample_to_tau(obs.rho, obs.p_sec, obs.ps_gap_sec, obs.window_sec)
                rho_tau = rho_tau / max(float(np.abs(rho_tau).max()), 1e-12)
            else:
                rho_tau = np.zeros(TAU_N, dtype=float)
            rows.append(
                {
                    "shape": str(r["shape"]),
                    "wave": w,
                    "rho_tau": rho_tau,
                    "tau": tau,
                    "p": obs.p_prob,
                    "s": obs.s_prob,
                    "p_sec": obs.p_sec,
                    "s_sec": obs.s_sec,
                }
            )

    for h in handles.values():
        h.close()

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 1.55 * len(rows) + 0.4), facecolor=BG)
    if len(rows) == 1:
        axes = np.array([axes])
    t_nat = np.linspace(0, 60, rows[0]["wave"].shape[0])
    t_prob = np.linspace(0, 60, len(rows[0]["p"]))
    for i, rec in enumerate(rows):
        c = SHAPE_COLORS.get(rec["shape"], C_INK)
        # crop waveform around P–S
        p_sec = float(rec["p_sec"]) if rec["p_sec"] >= 0 else 5.0
        s_sec = float(rec["s_sec"]) if rec["s_sec"] >= 0 else p_sec + 10.0
        t0 = max(0.0, p_sec - 4.0)
        t1 = min(60.0, max(s_sec + 12.0, p_sec + 18.0))
        m = (t_nat >= t0) & (t_nat <= t1)
        mp = (t_prob >= t0) & (t_prob <= t1)

        ax = axes[i, 0]
        wave = rec["wave"]
        scale = np.max(np.abs(wave[m])) + 1e-9
        for k, col in [(0, "#57534E"), (1, "#A8A29E"), (2, C_INK)]:
            ax.plot(t_nat[m], wave[m, k] / scale * 0.7 + k * 2.2, color=col, lw=0.6)
        if rec["p_sec"] >= 0:
            ax.axvline(rec["p_sec"], color=C_RED, ls="--", lw=0.8, alpha=0.8)
        if rec["s_sec"] >= 0:
            ax.axvline(rec["s_sec"], color=C_BLUE, ls="--", lw=0.8, alpha=0.8)
        ax.set_xlim(t0, t1)
        label = SHAPE_DISPLAY.get(rec["shape"], rec["shape"]).replace("_", "\n")
        ax.set_ylabel(label, fontsize=7.5, color=c, fontweight="bold")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_title("3C waveform", loc="left", fontsize=9)

        ax = axes[i, 1]
        ax.plot(rec["tau"], rec["rho_tau"], color=c, lw=1.2)
        ax.axvline(0.0, color=C_RED, ls=":", lw=0.7, alpha=0.7)
        ax.axvline(1.0, color=C_BLUE, ls=":", lw=0.7, alpha=0.7)
        ax.set_xlim(TAU_LO, TAU_HI)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_title(r"$\rho(\tau)$ ($P{=}0$, $S{=}1$)", loc="left", fontsize=9)

        ax = axes[i, 2]
        ax.plot(t_prob[mp], rec["p"][mp], color=C_RED, lw=1.0, label="P")
        ax.plot(t_prob[mp], rec["s"][mp], color=C_BLUE, lw=1.0, label="S")
        ax.set_xlim(t0, t1)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_title("pick probabilities", loc="left", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        if i == len(rows) - 1:
            ax.set_xlabel("time (s)")
            axes[i, 0].set_xlabel("time (s)")
            axes[i, 1].set_xlabel(r"delay time $\tau$")

    fig.suptitle("Representative STEAD facies (frozen PNF ceilings)", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig_s4_stead_examples", dpi)


# ---------------------------------------------------------------------------
# Fluid figures
# ---------------------------------------------------------------------------
def _fluid_vortex_payload(device: str):
    """Return (rho, speed_g, err, fam_stats) or None if checkpoint missing."""
    ckpt = Path("outputs/fluid/spatial_3d4d_suite/synth3d_vortex/best.pt")
    if not ckpt.is_file():
        return None
    from hnf.fluid_spatial3d import Spatial3DFluidHNFReconstructor
    from hnf.fluid_synth3d import make_sample3d

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    grid = state.get("args", {}).get("d", 12)
    model = Spatial3DFluidHNFReconstructor(d=grid, h=grid, w=grid, embed_dim=48, kernel_size=5).to(dev)
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        s = make_sample3d(d=grid, h=grid, w=grid, keep_frac=0.1, family="vortex_tube", seed=1)
        x = torch.from_numpy(np.concatenate([s["sparse"], s["mask"]], axis=0)).unsqueeze(0).to(dev)
        pred, aux = model(x, return_aux=True)
        rho = aux["rho"][0, 0].cpu().numpy()
        pred_np = pred[0].cpu().numpy()
        dense = s["dense"]
        speed_p = np.linalg.norm(pred_np, axis=0)
        speed_g = np.linalg.norm(dense, axis=0)
        err = np.abs(speed_p - speed_g)
        fam_stats = []
        for fam in ["pipe", "shear3d", "vortex_tube"]:
            mags_v, mags_m = [], []
            for seed in (1, 7, 13):
                ss = make_sample3d(d=grid, h=grid, w=grid, keep_frac=0.1, family=fam, seed=seed)
                xx = torch.from_numpy(np.concatenate([ss["sparse"], ss["mask"]], axis=0)).unsqueeze(0).to(dev)
                feat = model.patch(xx)
                layer0 = model.encoder.layers[0]
                mom = layer0.source_mom(feat)
                vort = layer0.source_vort(feat)
                mags_v.append(float(vort.pow(2).mean().sqrt().cpu()))
                mags_m.append(float(mom.pow(2).mean().sqrt().cpu()))
            fam_stats.append((fam, float(np.mean(mags_v)), float(np.mean(mags_m))))
    return rho, speed_g, err, fam_stats


def _draw_baseline_bars(ax, board: dict, keys, title: str) -> None:
    labels, vals, colors = [], [], []
    for key, lab, col in keys:
        if key in board and "vel_rel" in board[key]:
            labels.append(lab)
            vals.append(float(board[key]["vel_rel"]))
            colors.append(col)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.62, edgecolor=C_INK, linewidth=0.4)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"test vel. relative error (lower better)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)


def plot_fluid_combined(out_dir: Path, dpi: int, device: str) -> None:
    """Single SI plate: vortex reconstruction (a–d) + baselines (e–f)."""
    board_path = Path("outputs/fluid/baseline3d4d_board.json")
    payload = _fluid_vortex_payload(device)
    if payload is None or not board_path.is_file():
        print("[si] skip fluid combined: missing ckpt or board", flush=True)
        return
    rho, speed_g, err, fam_stats = payload
    board = json.loads(board_path.read_text())
    zmid = rho.shape[0] // 2

    fig = plt.figure(figsize=(12.2, 7.2), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.45)

    heatmaps = [
        (0, rho[zmid], "magma", r"(a) $\rho$ mid-$z$", r"$\rho$"),
        (1, speed_g[zmid], "viridis", r"(b) $|v|$ GT", r"$|v|$"),
        (2, err[zmid], "hot", r"(c) $|v|$ error", r"abs.\ error"),
    ]
    for col, arr, cmap, title, cblabel in heatmaps:
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.04)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(cblabel, fontsize=7)
        cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)

    ax = fig.add_subplot(gs[0, 3])
    names = [f[0].replace("_", "\n") for f in fam_stats]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, [f[1] for f in fam_stats], w, color=C_BLUE, label="vorticity DOF", edgecolor=C_INK, linewidth=0.4)
    ax.bar(x + w / 2, [f[2] for f in fam_stats], w, color=C_GOLD, label="momentum DOF", edgecolor=C_INK, linewidth=0.4)
    ax.set_xticks(x, names, fontsize=7)
    ax.set_ylabel("mean source magnitude", fontsize=8)
    ax.set_title("(d) secondary-source mag.", loc="left", fontsize=9)
    ymax = max(max(f[1] for f in fam_stats), max(f[2] for f in fam_stats))
    ax.set_ylim(0, ymax * 1.28)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", borderaxespad=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_e = fig.add_subplot(gs[1, 0:2])
    ax_f = fig.add_subplot(gs[1, 2:4])
    _draw_baseline_bars(
        ax_e,
        board,
        [
            ("unet3d", "U-Net 3D", "#4C72B0"),
            ("recfno3d", "RecFNO3D", "#DD8452"),
            ("flowmri_net3d", "FlowMRI-Net3D", "#55A868"),
            ("hnf_spatial3d_rot", "PNF spatial3D+rot", "#C44E52"),
        ],
        "(e) vortex_tube @10% keep",
    )
    _draw_baseline_bars(
        ax_f,
        board,
        [
            ("unet3d_all", "U-Net 3D", "#4C72B0"),
            ("recfno3d_all", "RecFNO3D", "#DD8452"),
            ("flowmri_net3d_all", "FlowMRI-Net3D", "#55A868"),
            ("hnf_spatial3d_all", "PNF spatial3D", "#C44E52"),
        ],
        "(f) all families @10% keep",
    )

    fig.suptitle("3D fluid reconstruction supplements", fontsize=12, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, hspace=0.42, wspace=0.55)
    _save(fig, out_dir / "fig_s5_fluid_combined", dpi)


def plot_fluid_s5(out_dir: Path, dpi: int, device: str) -> None:
    """Legacy single-row vortex plate (kept for regeneration / debugging)."""
    payload = _fluid_vortex_payload(device)
    if payload is None:
        print("[si] skip fluid s5: missing ckpt", flush=True)
        return
    rho, speed_g, err, fam_stats = payload
    zmid = rho.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.35), facecolor=BG, gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    heatmaps = [
        (axes[0], rho[zmid], "magma", r"(a) $\rho$ mid-$z$", r"$\rho$"),
        (axes[1], speed_g[zmid], "viridis", r"(b) $|v|$ GT", r"$|v|$"),
        (axes[2], err[zmid], "hot", r"(c) $|v|$ error", r"abs.\ error"),
    ]
    for ax, arr, cmap, title, cblabel in heatmaps:
        im = ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.04)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(cblabel, fontsize=7)
        cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)
    ax = axes[3]
    names = [f[0].replace("_", "\n") for f in fam_stats]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, [f[1] for f in fam_stats], w, color=C_BLUE, label="vorticity DOF", edgecolor=C_INK, linewidth=0.4)
    ax.bar(x + w / 2, [f[2] for f in fam_stats], w, color=C_GOLD, label="momentum DOF", edgecolor=C_INK, linewidth=0.4)
    ax.set_xticks(x, names, fontsize=7)
    ax.set_ylabel("mean source magnitude", fontsize=8)
    ax.set_title("(d) secondary-source mag.", loc="left", fontsize=9)
    ymax = max(max(f[1] for f in fam_stats), max(f[2] for f in fam_stats))
    ax.set_ylim(0, ymax * 1.28)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", borderaxespad=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("3D fluid reconstruction — vortex tube + source semantics", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "fig_s5_fluid_3d_reconstruction", dpi)


def plot_fluid_s6(out_dir: Path, dpi: int) -> None:
    board_path = Path("outputs/fluid/baseline3d4d_board.json")
    if not board_path.is_file():
        print("[si] skip fluid baselines: missing board", flush=True)
        return
    board = json.loads(board_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=BG)
    _draw_baseline_bars(
        axes[0],
        board,
        [
            ("unet3d", "U-Net 3D", "#4C72B0"),
            ("recfno3d", "RecFNO3D", "#DD8452"),
            ("flowmri_net3d", "FlowMRI-Net3D", "#55A868"),
            ("hnf_spatial3d_rot", "PNF spatial3D+rot", "#C44E52"),
        ],
        "(e) vortex_tube @10% keep",
    )
    _draw_baseline_bars(
        axes[1],
        board,
        [
            ("unet3d_all", "U-Net 3D", "#4C72B0"),
            ("recfno3d_all", "RecFNO3D", "#DD8452"),
            ("flowmri_net3d_all", "FlowMRI-Net3D", "#55A868"),
            ("hnf_spatial3d_all", "PNF spatial3D", "#C44E52"),
        ],
        "(f) all families @10% keep",
    )
    fig.suptitle("Sparse 3D fluid reconstruction — baselines vs PNF", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "fig_s6_fluid_baselines", dpi)


def main() -> None:
    args = parse_args()
    _style()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_decoder(out, args.dpi)
    plot_eeg_pipeline(out, args.dpi)
    plot_tube_mwd(out, args.dpi)
    if not args.skip_stead:
        try:
            plot_stead_examples(out, args.dpi, args)
        except Exception as exc:
            print(f"[si] STEAD examples failed: {exc}", flush=True)
    if not args.skip_fluid:
        try:
            plot_fluid_combined(out, args.dpi, args.device)
            plot_fluid_s5(out, args.dpi, args.device)
            plot_fluid_s6(out, args.dpi)
        except Exception as exc:
            print(f"[si] fluid figures failed: {exc}", flush=True)
    print(f"[si] done → {out}", flush=True)


if __name__ == "__main__":
    main()
