#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nature-Communications display figures for coda-facies path domains.

Writes four main plates + a one-page talk hero to docs/figures/nc_main/.

  Fig. 1  Method schematic + facies + what β_res means
  Fig. 2  SoCal discovery (map, same-station, station forest)
  Fig. 3  Independence: classical Qc, Berg Vs split, Lin QS anti-align
  Fig. 4  Cascadia replica, jackknife, absorbing vs ringing synthesis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
BG = "#FFFFFF"
C_ABS = "#B91C1C"  # absorbing / faster decay
C_RING = "#1D4ED8"  # ringing / slower decay
C_TEAL = "#0F766E"
C_GOLD = "#B45309"
C_PURPLE = "#6D28D9"

SHAPE_COLORS = {
    "impulsive_fastQ": "#0072B2",
    "emergent": "#E69F00",
    "multipath": "#009E73",
    "slow_coda": "#D55E00",
    "standard": "#6B7280",
}
SHAPE_LABEL = {
    "impulsive_fastQ": "impulsive\nfast-Q",
    "emergent": "emergent",
    "multipath": "multipath",
    "slow_coda": "slow\ncoda",
    "standard": "standard",
}


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BG,
            "figure.facecolor": BG,
        }
    )


def _panel(ax, letter: str, x=-0.08, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", color=C_INK, va="bottom", ha="left")


def _clean(ax, grid=False):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, colors=C_INK)
    if grid:
        ax.grid(True, color=C_LINE, lw=0.4, zorder=0)


def _load_polylines(path: Path):
    if not path.is_file():
        return []
    geo = json.loads(path.read_text(encoding="utf-8"))
    lines = []

    def add(coords):
        arr = np.asarray(coords, float)
        if arr.ndim == 2 and len(arr) >= 2:
            lines.append(arr)

    for feat in geo.get("features", []):
        g = feat.get("geometry") or {}
        t, c = g.get("type"), g.get("coordinates")
        if t == "LineString":
            add(c)
        elif t == "MultiLineString":
            for part in c:
                add(part)
        elif t == "Polygon":
            add(c[0])
        elif t == "MultiPolygon":
            for poly in c:
                add(poly[0])
    return lines


def _load_coast(path: Path):
    if not path.is_file():
        return []
    geo = json.loads(path.read_text(encoding="utf-8"))
    rings = []
    for feat in geo.get("features", []):
        g = feat.get("geometry") or {}
        if g.get("type") == "Polygon":
            rings.append(np.asarray(g["coordinates"][0], float))
        elif g.get("type") == "MultiPolygon":
            for poly in g["coordinates"]:
                rings.append(np.asarray(poly[0], float))
    return rings


def fig1_method(out: Path, dpi: int):
    fig = plt.figure(figsize=(180 / 25.4, 145 / 25.4), facecolor=BG)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.35, left=0.06, right=0.98, top=0.90, bottom=0.08)

    # --- a schematic pipeline ---
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    _panel(ax, "a", 0.0, 1.02)
    ax.set_title("From waveform facies to a structure-controlled path residual", loc="left", fontsize=9, color=C_INK, pad=2)

    boxes = [
        (0.15, 1.35, 1.7, 1.35, "Local EQ\nwaveform\n(STEAD 60 s)", "#F3F4F6"),
        (2.15, 1.35, 1.85, 1.35, "Frozen Huygens\npicker +\ninterpretable\nobservables", "#ECFDF5"),
        (4.30, 1.35, 1.85, 1.35, "Five-class\nfacies\n(ceiling, not\nclustering)", "#EFF6FF"),
        (6.45, 1.35, 1.7, 1.35, "Coda slope\n→ distance-\ndetrended β", "#FFF7ED"),
        (8.45, 0.35, 3.3, 2.55, "", "#F5F3FF"),
    ]
    for x, y, w, h, txt, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec="#D1D5DB", lw=0.8))
        if txt:
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2, color=C_INK)
    ax.text(10.1, 2.55, "Structure expectation", ha="center", fontsize=8, fontweight="bold", color=C_PURPLE)
    ax.text(
        10.1,
        1.45,
        "ridge / station FE\n"
        r"$\beta \sim \log d + z + M$" "\n"
        r"$+\;\log(1+d_{\mathrm{fault}})$" "\n"
        "+ station",
        ha="center",
        va="center",
        fontsize=7.2,
        color=C_INK,
    )
    ax.text(10.1, 0.55, r"leftover  $\beta_{\mathrm{res}}$  at path midpoint", ha="center", fontsize=7.4, color=C_PURPLE, fontweight="bold")
    for x0, x1 in [(1.85, 2.15), (4.00, 4.30), (6.15, 6.45), (8.15, 8.45)]:
        ax.annotate("", xy=(x1, 2.02), xytext=(x0, 2.02), arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.1))

    # mini waveform in first box
    t = np.linspace(0, 1, 200)
    w = np.exp(-((t - 0.28) / 0.04) ** 2) * np.sin(40 * t) + 0.45 * np.exp(-((t - 0.42) / 0.08) ** 2) * np.sin(28 * t)
    ax.plot(0.28 + t * 1.45, 1.58 + 0.35 * w, color=C_INK, lw=0.7)

    # --- b five facies ---
    axb = fig.add_subplot(gs[1, 0])
    _clean(axb)
    _panel(axb, "b")
    axb.set_title("Frozen facies", fontsize=9, color=C_INK, loc="left")
    t = np.linspace(0, 1, 300)
    cartoons = {
        "impulsive_fastQ": np.exp(-((t - 0.25) / 0.03) ** 2) * np.exp(-8 * np.clip(t - 0.25, 0, None)),
        "emergent": np.exp(-((t - 0.38) / 0.12) ** 2) * (0.4 + 0.6 * t),
        "multipath": np.exp(-((t - 0.28) / 0.04) ** 2) + 0.7 * np.exp(-((t - 0.48) / 0.05) ** 2) + 0.45 * np.exp(-((t - 0.66) / 0.06) ** 2),
        "slow_coda": np.exp(-((t - 0.30) / 0.06) ** 2) * np.exp(-1.3 * np.clip(t - 0.30, 0, None)),
        "standard": np.exp(-((t - 0.30) / 0.05) ** 2) * np.exp(-3.2 * np.clip(t - 0.30, 0, None)),
    }
    for i, name in enumerate(SHAPE_COLORS):
        y0 = 4.4 - i
        env = cartoons[name]
        env = env / env.max()
        axb.fill_between(t, y0, y0 + 0.75 * env, color=SHAPE_COLORS[name], alpha=0.85, lw=0)
        axb.text(1.04, y0 + 0.25, SHAPE_LABEL[name], fontsize=6.5, color=C_INK, va="center")
    axb.set_xlim(0, 1.55)
    axb.set_ylim(-0.15, 5.2)
    axb.set_xlabel("time after P (schematic)", fontsize=7)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)

    # --- c what beta means ---
    axc = fig.add_subplot(gs[1, 1])
    _clean(axc)
    _panel(axc, "c")
    axc.set_title(r"What $\beta_{\mathrm{res}}$ measures", fontsize=9, color=C_INK, loc="left")
    t = np.linspace(0, 1, 400)
    fast = np.exp(-((t - 0.2) / 0.05) ** 2) * np.exp(-6.5 * np.clip(t - 0.2, 0, None))
    slow = np.exp(-((t - 0.2) / 0.05) ** 2) * np.exp(-2.0 * np.clip(t - 0.2, 0, None))
    axc.fill_between(t, 0, fast / fast.max(), color=C_ABS, alpha=0.25)
    axc.plot(t, fast / fast.max(), color=C_ABS, lw=1.6, label=r"$\beta_{\mathrm{res}}<0$  faster decay")
    axc.fill_between(t, 0, slow / slow.max(), color=C_RING, alpha=0.18)
    axc.plot(t, slow / slow.max(), color=C_RING, lw=1.6, label=r"$\beta_{\mathrm{res}}>0$  longer ringing")
    axc.set_xlabel("time after S (schematic)", fontsize=7)
    axc.set_ylabel("envelope", fontsize=7)
    axc.legend(frameon=False, fontsize=6.8, loc="upper right")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1.15)

    # --- d claim cartoon ---
    axd = fig.add_subplot(gs[1, 2])
    _clean(axd)
    _panel(axd, "d")
    axd.set_title("Why this is not scalar Q", fontsize=9, color=C_INK, loc="left")
    axd.set_xlim(0, 10)
    axd.set_ylim(0, 10)
    axd.axis("off")
    axd.add_patch(FancyBboxPatch((0.3, 5.6), 4.2, 3.8, boxstyle="round,pad=0.05,rounding_size=0.1", fc="#F3F4F6", ec=C_LINE))
    axd.add_patch(FancyBboxPatch((5.3, 5.6), 4.4, 3.8, boxstyle="round,pad=0.05,rounding_size=0.1", fc="#F3F4F6", ec=C_LINE))
    axd.text(2.4, 8.9, "Same published Vs / Q", ha="center", fontsize=7.2, color=C_MUTED)
    axd.text(7.5, 8.9, "Opposite coda facies", ha="center", fontsize=7.2, color=C_MUTED)
    axd.plot([1.1, 3.7], [7.4, 7.4], color="#9CA3AF", lw=6, solid_capstyle="round")
    axd.plot([1.1, 3.7], [6.6, 6.6], color="#9CA3AF", lw=6, solid_capstyle="round")
    axd.text(2.4, 7.4, "Vs band", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    axd.text(2.4, 6.6, "direct-wave QS", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    axd.plot([6.0, 9.2], [7.55, 6.55], color=C_ABS, lw=2.2)
    axd.plot([6.0, 9.2], [7.55, 7.55], color=C_RING, lw=2.2)
    axd.text(9.35, 6.55, "absorb", color=C_ABS, fontsize=7, va="center")
    axd.text(9.35, 7.55, "ring", color=C_RING, fontsize=7, va="center")
    axd.text(7.5, 6.05, "unmixed by β_res", ha="center", fontsize=7, color=C_INK, fontweight="bold")
    axd.add_patch(FancyBboxPatch((0.4, 0.35), 9.2, 4.55, boxstyle="round,pad=0.06,rounding_size=0.1", fc="#FAFAF9", ec=C_LINE, zorder=0))
    axd.text(
        5.0,
        2.55,
        "Direct-wave QS sees extinction along the ray.\n"
        "Coda facies see how energy is trapped\n"
        "and released after S.",
        ha="center",
        va="center",
        fontsize=7.3,
        color=C_INK,
        zorder=1,
    )

    fig.savefig(out / "Fig1_method.pdf", dpi=dpi)
    fig.savefig(out / "Fig1_method.png", dpi=dpi)
    plt.close(fig)


def _map_base(ax, coast, faults, xlim, ylim):
    _clean(ax)
    for ring in coast:
        ax.plot(ring[:, 0], ring[:, 1], color="#D1D5DB", lw=0.5, zorder=0)
    if faults:
        lc = LineCollection(faults, colors="#6B7280", linewidths=0.35, alpha=0.7, zorder=1)
        ax.add_collection(lc)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°)", fontsize=7)
    ax.set_ylabel("Latitude (°)", fontsize=7)


def fig2_discovery(out: Path, dpi: int):
    cells = pd.read_csv(_REPO / "outputs/structure_residual_socal/grid_beta_resid.csv")
    ss = pd.read_csv(_REPO / "outputs/structure_residual_socal/same_station_validation.csv")
    bysite = pd.read_csv(_REPO / "outputs/structure_residual_socal/same_station_by_site.csv")
    coast = _load_coast(_REPO / "docs/figures/geo/ne_110m_land.geojson")
    faults = _load_polylines(_REPO / "docs/figures/geo/qfaults_socal.geojson") + _load_polylines(
        _REPO / "docs/figures/geo/qfaults_socal_offshore.geojson"
    )

    fig = plt.figure(figsize=(180 / 25.4, 155 / 25.4), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1], height_ratios=[1.15, 1], hspace=0.38, wspace=0.28, left=0.07, right=0.98, top=0.90, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    _map_base(ax, coast, faults, (-118.6, -115.3), (32.55, 34.55))
    _panel(ax, "a")
    ax.set_title(r"SoCal path-midpoint  $\beta_{\mathrm{res}}$", fontsize=9, color=C_INK, loc="left")
    nrm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    sc = ax.scatter(
        cells["lon"],
        cells["lat"],
        c=cells["mean"],
        s=np.clip(cells["n"] * 0.55, 18, 160),
        cmap="coolwarm_r",
        norm=nrm,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"cell-mean $\beta_{\mathrm{res}}$" "\n(red = faster decay)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    marks = {2: (-116.375, 33.375, C_ABS, "#2 Salton"), 5: (-116.375, 33.875, C_RING, "#5 ETR")}
    for _, (lon, lat, col, lab) in marks.items():
        ax.plot(lon, lat, marker="s", ms=11, mfc="none", mec=col, mew=1.6, zorder=5)
        ax.text(lon + 0.08, lat + 0.08, lab, color=col, fontsize=7.5, fontweight="bold", zorder=5)
    ax.text(-118.45, 32.68, "red = faster decay    blue = longer ringing", fontsize=6.5, color=C_MUTED)

    axb = fig.add_subplot(gs[0, 1])
    _clean(axb)
    _panel(axb, "b")
    axb.set_title("Same-station test (site-median)", fontsize=9, color=C_INK, loc="left")
    ok = ss[ss["n_sites_paired"] >= 8].copy().sort_values("rank")
    labels, vals, cols, ps, ns = [], [], [], [], []
    short = {
        "Salton Trough / S. San Andreas–Imperial junction": "#2 Salton",
        "E. Transverse Ranges / Little San Bernardino Mts": "#5 E. Transverse R.",
        "San Jacinto FZ / Anza–Coyote Creek step": "#1 San Jacinto step",
        "San Andreas–San Jacinto junction (San Gorgonio–Banning)": "#6 SA–SJ junction",
        "San Jacinto FZ central (Anza–Hemet)": "#7 SJ central",
    }
    for r in ok.itertuples():
        labels.append(short.get(r.geo_tag, r.geo_tag.split("/")[0][:18]))
        vals.append(r.site_delta_med)
        cols.append(C_ABS if r.site_delta_med < 0 else C_RING)
        ps.append(r.mw_p)
        ns.append(int(r.n_sites_paired))
    ys = np.arange(len(labels))[::-1]
    axb.axvline(0, color=C_LINE, lw=0.8)
    axb.barh(ys, vals, color=cols, height=0.62, zorder=2)
    axb.set_yticks(ys)
    axb.set_yticklabels(labels, fontsize=7)
    axb.set_xlabel(r"same-station median $\Delta\beta$", fontsize=7)
    for y, v, p, n in zip(ys, vals, ps, ns):
        axb.text(v + (0.0015 if v >= 0 else -0.0015), y, f"n={n}  p≈{p:.1g}", va="center", ha="left" if v >= 0 else "right", fontsize=6, color=C_MUTED)
    axb.set_xlim(-0.055, 0.055)

    axc = fig.add_subplot(gs[1, 0])
    _clean(axc)
    _panel(axc, "c")
    axc.set_title("#2 Salton · per-station Δβ", fontsize=9, color=C_INK, loc="left")
    s2 = bysite[bysite["rank"] == 2].sort_values("delta")
    ys = np.arange(len(s2))
    axc.axvline(0, color=C_LINE, lw=0.8)
    axc.barh(ys, s2["delta"], color=[C_ABS if v < 0 else C_RING for v in s2["delta"]], height=0.7, zorder=2)
    axc.set_yticks(ys)
    axc.set_yticklabels(s2["site"], fontsize=5.8)
    axc.set_xlabel(r"$\Delta\beta$  (in-cell − other paths)", fontsize=7)
    axc.text(0.02, 0.04, "80% of stations faster in-cell", transform=axc.transAxes, fontsize=7, color=C_ABS)

    axd = fig.add_subplot(gs[1, 1])
    _clean(axd)
    _panel(axd, "d")
    axd.set_title("#5 E. Transverse Ranges · per-station Δβ", fontsize=9, color=C_INK, loc="left")
    s5 = bysite[bysite["rank"] == 5].sort_values("delta")
    ys = np.arange(len(s5))
    axd.axvline(0, color=C_LINE, lw=0.8)
    axd.barh(ys, s5["delta"], color=[C_ABS if v < 0 else C_RING for v in s5["delta"]], height=0.7, zorder=2)
    axd.set_yticks(ys)
    axd.set_yticklabels(s5["site"], fontsize=5.8)
    axd.set_xlabel(r"$\Delta\beta$  (in-cell − other paths)", fontsize=7)
    axd.text(0.02, 0.04, "93% of stations slower in-cell", transform=axd.transAxes, fontsize=7, color=C_RING)

    fig.suptitle(
        "Path domains survive station control   ·   not a seismicity or site-mix artefact",
        fontsize=9.5,
        color=C_INK,
        y=0.975,
    )
    fig.savefig(out / "Fig2_socal_discovery.pdf", dpi=dpi)
    fig.savefig(out / "Fig2_socal_discovery.png", dpi=dpi)
    plt.close(fig)


def fig3_independence(out: Path, dpi: int):
    qc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_classical_qc.csv")
    qcc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_classical_qc_cells.csv")
    sep = pd.read_csv(_REPO / "outputs/structure_residual_socal/shape_separation_within_qc_bins.csv")
    berg = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_berg2021.csv")
    linq = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_lin2023_q.csv")
    bergc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_berg2021_cells.csv")
    linc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_lin2023_q_cells.csv")

    fig = plt.figure(figsize=(180 / 25.4, 168 / 25.4), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28, left=0.08, right=0.98, top=0.90, bottom=0.08)

    def _mark(ax, df, xcol):
        m2 = np.isclose(df["gx"], -116.375) & np.isclose(df["gy"], 33.375)
        m5 = np.isclose(df["gx"], -116.375) & np.isclose(df["gy"], 33.875)
        ax.scatter(df[xcol], df["beta_resid"], s=4, c="#D1D5DB", alpha=0.35, linewidths=0, rasterized=True, zorder=1)
        ax.scatter(df.loc[m2, xcol], df.loc[m2, "beta_resid"], s=10, c=C_ABS, alpha=0.75, linewidths=0, label="#2 Salton", zorder=3)
        ax.scatter(df.loc[m5, xcol], df.loc[m5, "beta_resid"], s=10, c=C_RING, alpha=0.8, linewidths=0, label="#5 ETR", zorder=3)

    ax = fig.add_subplot(gs[0, 0])
    _clean(ax)
    _panel(ax, "a")
    ax.set_title(r"Classical single-trace $q_c$   $r=0.64$", fontsize=9, color=C_INK, loc="left")
    _mark(ax, qc, "qc_slope")
    x = qc["qc_slope"].to_numpy()
    y = qc["beta_resid"].to_numpy()
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    xx = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 50)
    ax.plot(xx, coef[0] + coef[1] * xx, color=C_INK, lw=1.1, zorder=2)
    ax.set_xlabel(r"classical coda decay slope  (more negative = faster)", fontsize=7)
    ax.set_ylabel(r"$\beta_{\mathrm{res}}$", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.axhline(0, color=C_LINE, lw=0.6)

    axb = fig.add_subplot(gs[0, 1])
    _clean(axb)
    _panel(axb, "b")
    axb.set_title("Facies still split inside matched $q_c$ terciles", fontsize=9, color=C_INK, loc="left")
    xs = np.arange(len(sep))
    axb.bar(xs - 0.18, sep["beta_mp"], 0.36, color="#009E73", label="multipath")
    axb.bar(xs + 0.18, sep["beta_sc"], 0.36, color="#D55E00", label="slow coda")
    axb.axhline(0, color=C_LINE, lw=0.7)
    axb.set_xticks(xs)
    axb.set_xticklabels(sep["qc_bin"], fontsize=8)
    axb.set_ylabel(r"median $\beta_{\mathrm{res}}$", fontsize=8)
    axb.legend(frameon=False, fontsize=7)

    axc = fig.add_subplot(gs[1, 0])
    _clean(axc)
    _panel(axc, "c")
    axc.set_title(r"Berg 2021 $V_S$  ·  SPLIT   $r=-0.08$", fontsize=9, color=C_INK, loc="left")
    _mark(axc, berg, "vs_0to8")
    axc.axvspan(3.05, 3.21, color="#FEF3C7", alpha=0.55, zorder=0)
    axc.text(3.13, 0.175, "shared Vs band\nΔβ = +0.046", ha="center", va="top", fontsize=6.8, color=C_GOLD)
    axc.set_xlabel(r"path-midpoint mean $V_S$ 0–8 km  (km s$^{-1}$)", fontsize=7)
    axc.set_ylabel(r"$\beta_{\mathrm{res}}$", fontsize=8)
    axc.axhline(0, color=C_LINE, lw=0.6)
    axc.legend(frameon=False, fontsize=7)
    axc.set_xlim(2.55, 3.45)

    axd = fig.add_subplot(gs[1, 1])
    _clean(axd)
    _panel(axd, "d")
    axd.set_title(r"Lin & Jordan 2023 $Q_S$  ·  orthogonal / anti-aligned   $r=-0.07$", fontsize=9, color=C_INK, loc="left")
    _mark(axd, linq, "qs_0to8")
    axd.set_xlabel(r"path-midpoint mean $Q_S$ 0–8 km", fontsize=7)
    axd.set_ylabel(r"$\beta_{\mathrm{res}}$", fontsize=8)
    axd.axhline(0, color=C_LINE, lw=0.6)
    axd.legend(frameon=False, fontsize=7)
    axd.annotate(
        "high QS\nyet faster coda",
        xy=(790, -0.03),
        xytext=(680, -0.16),
        fontsize=6.8,
        color=C_ABS,
        arrowprops=dict(arrowstyle="->", color=C_ABS, lw=0.8),
    )
    axd.annotate(
        "lower QS\nyet rings",
        xy=(560, 0.04),
        xytext=(430, 0.16),
        fontsize=6.8,
        color=C_RING,
        arrowprops=dict(arrowstyle="->", color=C_RING, lw=0.8),
    )

    # leftover summary strip as fig text
    fig.text(
        0.08,
        0.955,
        "Leftover cell Δβ after removing each observable    "
        f"qc:  #2 {qcc.loc[qcc['rank']==2,'delta_beta_after_qc'].iloc[0]:+.3f} / #5 {qcc.loc[qcc['rank']==5,'delta_beta_after_qc'].iloc[0]:+.3f}     "
        f"Vs:  #2 {bergc.loc[bergc['rank']==2,'delta_beta_after_vs'].iloc[0]:+.3f} / #5 {bergc.loc[bergc['rank']==5,'delta_beta_after_vs'].iloc[0]:+.3f}     "
        f"QS:  #2 {linc.loc[linc['rank']==2,'delta_beta_after_qs'].iloc[0]:+.3f} / #5 {linc.loc[linc['rank']==5,'delta_beta_after_qs'].iloc[0]:+.3f}",
        fontsize=7.2,
        color=C_INK,
    )
    fig.savefig(out / "Fig3_independence.pdf", dpi=dpi)
    fig.savefig(out / "Fig3_independence.png", dpi=dpi)
    plt.close(fig)


def fig4_replica(out: Path, dpi: int):
    cells = pd.read_csv(_REPO / "outputs/structure_residual_cascadia_volc/grid_beta_resid.csv")
    ss = pd.read_csv(_REPO / "outputs/structure_residual_cascadia_volc/same_station_validation.csv")
    jk = pd.read_csv(_REPO / "outputs/structure_residual_cascadia_volc/sthelens_source_jackknife.csv")
    coast = _load_coast(_REPO / "docs/figures/geo/ne_110m_land.geojson")
    faults = []
    for p in [
        "docs/figures/geo/qfaults_pnw_wa.geojson",
        "docs/figures/geo/qfaults_pnw_wa_offshore.geojson",
        "docs/figures/geo/qfaults_pnw_or.geojson",
    ]:
        faults += _load_polylines(_REPO / p)

    fig = plt.figure(figsize=(180 / 25.4, 160 / 25.4), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1], height_ratios=[1.12, 1], hspace=0.36, wspace=0.30, left=0.08, right=0.98, top=0.90, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    _map_base(ax, coast, faults, (-124.7, -120.3), (45.3, 48.7))
    _panel(ax, "a")
    ax.set_title(r"Cascadia replica  $\beta_{\mathrm{res}}$", fontsize=9, color=C_INK, loc="left")
    nrm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    sc = ax.scatter(
        cells["lon"],
        cells["lat"],
        c=cells["mean"],
        s=np.clip(cells["n"] * 0.35, 14, 140),
        cmap="coolwarm_r",
        norm=nrm,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(r"$\beta_{\mathrm{res}}$  (red = faster decay)", fontsize=7)
    ax.plot(-122.375, 46.375, marker="s", ms=10, mfc="none", mec=C_ABS, mew=1.6)
    ax.text(-122.28, 46.18, "# St Helens–Cowlitz\nΔβ = −0.014", color=C_ABS, fontsize=7, fontweight="bold")
    ax.plot(-122.375, 47.375, marker="s", ms=10, mfc="none", mec=C_RING, mew=1.6)
    ax.text(-122.28, 47.48, "# Seattle basin\nΔβ = +0.045", color=C_RING, fontsize=7, fontweight="bold")

    axb = fig.add_subplot(gs[0, 1])
    _clean(axb)
    _panel(axb, "b")
    axb.set_title("Same-station Δβ  (cells with ≥8 stations)", fontsize=9, color=C_INK, loc="left")
    ok = ss[ss["n_sites_paired"] >= 8].copy()
    short = {
        "SW Washington / Mt St Helens–Cowlitz": "St Helens–Cowlitz",
        "Seattle basin / central Puget Sound": "Seattle basin",
        "NE Olympic / Juan de Fuca": "NE Olympic",
    }
    labels = [short.get(r.geo_tag, r.geo_tag[:20]) for r in ok.itertuples()]
    vals = ok["site_delta_med"].to_numpy()
    cols = [C_ABS if v < 0 else C_RING for v in vals]
    xs = np.arange(len(vals))
    axb.axhline(0, color=C_LINE, lw=0.8)
    axb.bar(xs, vals, color=cols, width=0.62)
    axb.set_xticks(xs)
    axb.set_xticklabels(labels, fontsize=7)
    axb.set_ylabel(r"same-station median $\Delta\beta$", fontsize=8)
    for i, r in enumerate(ok.itertuples()):
        axb.text(i, r.site_delta_med + np.sign(r.site_delta_med or 1) * 0.003, f"n={int(r.n_sites_paired)}\np≈{r.mw_p:.1g}", ha="center", fontsize=6.2, color=C_MUTED)

    axc = fig.add_subplot(gs[1, 0])
    _clean(axc)
    _panel(axc, "c")
    axc.set_title("St Helens source / year jackknife  ·  STABLE", fontsize=9, color=C_INK, loc="left")
    focus_tests = [
        "full",
        "drop_top1_srcbins",
        "drop_top5_srcbins",
        "drop_top10_srcbins",
        "cap20_per_srcbin",
        "cap10_per_srcbin",
        "in_cell_mag>=1.0",
        "in_cell_mag>=1.5",
        "drop_year_2011",
        "drop_year_2014",
        "drop_year_2017",
    ]
    pretty = {
        "full": "full",
        "drop_top1_srcbins": "drop top1\nsource",
        "drop_top5_srcbins": "drop top5\nsources",
        "drop_top10_srcbins": "drop top10\nsources",
        "cap20_per_srcbin": "cap 20 /\nsource",
        "cap10_per_srcbin": "cap 10 /\nsource",
        "in_cell_mag>=1.0": "M≥1.0",
        "in_cell_mag>=1.5": "M≥1.5",
        "drop_year_2011": "drop\n2011",
        "drop_year_2014": "drop\n2014",
        "drop_year_2017": "drop\n2017",
    }
    sub = jk.set_index("test").loc[focus_tests]
    xs = np.arange(len(sub))
    axc.axhline(0, color=C_LINE, lw=0.8)
    axc.bar(xs, sub["site_delta_med"], color=C_ABS, width=0.72)
    axc.set_xticks(xs)
    axc.set_xticklabels([pretty[t] for t in focus_tests], fontsize=6.2)
    axc.set_ylabel(r"same-station $\Delta\beta$", fontsize=8)
    axc.set_ylim(-0.05, 0.01)

    axd = fig.add_subplot(gs[1, 1])
    axd.set_xlim(0, 10)
    axd.set_ylim(0, 10)
    axd.axis("off")
    _panel(axd, "d", 0.0, 1.02)
    axd.set_title("Synthesis: absorbing vs ringing paths", fontsize=9, color=C_INK, loc="left")
    # two basins
    axd.add_patch(FancyBboxPatch((0.3, 5.3), 4.5, 4.2, boxstyle="round,pad=0.06,rounding_size=0.12", fc="#FEF2F2", ec=C_ABS, lw=1.1))
    axd.add_patch(FancyBboxPatch((5.2, 5.3), 4.5, 4.2, boxstyle="round,pad=0.06,rounding_size=0.12", fc="#EFF6FF", ec=C_RING, lw=1.1))
    axd.text(2.55, 8.95, "ABSORBING", ha="center", fontsize=8, fontweight="bold", color=C_ABS)
    axd.text(7.45, 8.95, "RINGING", ha="center", fontsize=8, fontweight="bold", color=C_RING)
    axd.text(2.55, 7.7, "Salton–Imperial\nMt St Helens–Cowlitz", ha="center", fontsize=7.5, color=C_INK)
    axd.text(7.45, 7.7, "E. Transverse Ranges\nSeattle basin", ha="center", fontsize=7.5, color=C_INK)
    axd.text(2.55, 6.05, r"$\Delta\beta < 0$" "\nfaster coda decay\nsame stations, other paths", ha="center", fontsize=6.8, color=C_ABS)
    axd.text(7.45, 6.05, r"$\Delta\beta > 0$" "\nenergy persists\nsame stations, other paths", ha="center", fontsize=6.8, color=C_RING)
    axd.add_patch(FancyBboxPatch((0.3, 0.35), 9.4, 4.5, boxstyle="round,pad=0.06,rounding_size=0.12", fc="#FAFAF9", ec=C_LINE))
    axd.text(
        5.0,
        2.55,
        "Same crust. Three observables.\n"
        "• Berg Vs  — split (same velocity band)\n"
        "• Lin QS   — orthogonal / anti-aligned\n"
        "• Coda facies residual — station-controlled\n"
        "   and replicated outside SoCal.\n\n"
        "Prediction: new SCSN Salton-cell paths\n"
        r"site-median $\Delta\beta \leq -0.02$  ($n_{\mathrm{st}}\geq 8$).",
        ha="center",
        va="center",
        fontsize=7.1,
        color=C_INK,
    )

    fig.suptitle("Out-of-region replication  ·  two basins, opposite coda facies", fontsize=9.5, color=C_INK, y=0.975)
    fig.savefig(out / "Fig4_replication.pdf", dpi=dpi)
    fig.savefig(out / "Fig4_replication.png", dpi=dpi)
    plt.close(fig)


def fig_hero(out: Path, dpi: int):
    """One-page talk / cover plate compressing the claim."""
    fig = plt.figure(figsize=(180 / 25.4, 95 / 25.4), facecolor=BG)
    gs = GridSpec(1, 3, figure=fig, wspace=0.18, left=0.05, right=0.98, top=0.82, bottom=0.14)
    cells = pd.read_csv(_REPO / "outputs/structure_residual_socal/grid_beta_resid.csv")
    coast = _load_coast(_REPO / "docs/figures/geo/ne_110m_land.geojson")
    faults = _load_polylines(_REPO / "docs/figures/geo/qfaults_socal.geojson")
    ax = fig.add_subplot(gs[0, 0])
    _map_base(ax, coast, faults, (-118.3, -115.4), (32.7, 34.4))
    nrm = TwoSlopeNorm(vmin=-0.04, vcenter=0, vmax=0.04)
    ax.scatter(cells.lon, cells.lat, c=cells["mean"], s=np.clip(cells.n * 0.45, 12, 110), cmap="coolwarm_r", norm=nrm, edgecolors="white", lw=0.2, zorder=3)
    ax.plot(-116.375, 33.375, "s", ms=9, mfc="none", mec=C_ABS, mew=1.5)
    ax.plot(-116.375, 33.875, "s", ms=9, mfc="none", mec=C_RING, mew=1.5)
    ax.text(-116.28, 33.28, "#2", color=C_ABS, fontsize=8, fontweight="bold")
    ax.text(-116.28, 33.95, "#5", color=C_RING, fontsize=8, fontweight="bold")
    ax.set_title("SoCal  β_res domains", fontsize=9, color=C_INK)

    ax2 = fig.add_subplot(gs[0, 1])
    _clean(ax2)
    labs = ["classical qc", "Berg Vs", "Lin QS"]
    d2 = [-0.014, -0.028, -0.026]
    d5 = [0.019, 0.037, 0.038]
    xs = np.arange(3)
    ax2.axhline(0, color=C_LINE, lw=0.8)
    ax2.bar(xs - 0.18, d2, 0.36, color=C_ABS, label="#2 Salton")
    ax2.bar(xs + 0.18, d5, 0.36, color=C_RING, label="#5 ETR")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labs, fontsize=7.5)
    ax2.set_ylabel(r"leftover $\Delta\beta$ after removal", fontsize=8)
    ax2.set_title("Not absorbed by Qc / Vs / QS", fontsize=9, color=C_INK)
    ax2.legend(frameon=False, fontsize=7)

    ax3 = fig.add_subplot(gs[0, 2])
    _clean(ax3)
    ax3.axhline(0, color=C_LINE, lw=0.8)
    ax3.bar([0, 1], [-0.014, 0.045], color=[C_ABS, C_RING], width=0.55)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["St Helens\n(absorb)", "Seattle\n(ring)"], fontsize=8)
    ax3.set_ylabel(r"same-station $\Delta\beta$", fontsize=8)
    ax3.set_title("Cascadia replica", fontsize=9, color=C_INK)

    fig.suptitle(
        "Coda facies unmix path domains that Vs and direct-wave Q do not resolve",
        fontsize=11,
        color=C_INK,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(out / "Fig0_hero.pdf", dpi=dpi)
    fig.savefig(out / "Fig0_hero.png", dpi=dpi)
    plt.close(fig)


def main():
    _style()
    out = _REPO / "docs/figures/nc_main"
    out.mkdir(parents=True, exist_ok=True)
    dpi = 300
    print("[fig1]", flush=True)
    fig1_method(out, dpi)
    print("[fig2]", flush=True)
    fig2_discovery(out, dpi)
    print("[fig3]", flush=True)
    fig3_independence(out, dpi)
    print("[fig4]", flush=True)
    fig4_replica(out, dpi)
    print("[hero]", flush=True)
    fig_hero(out, dpi)
    print(f"[done] {out}", flush=True)
    for p in sorted(out.glob("Fig*")):
        print(" ", p.name, flush=True)


if __name__ == "__main__":
    main()
