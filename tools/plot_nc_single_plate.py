#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One Nature-style display figure (panels a–h). Typography matches PNF large plates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

C_INK = "#0B3D4A"
C_TEAL = "#1B6B93"
C_ACCENT = "#C45C26"
C_MUTED = "#6B7C80"
C_LINE = "#D5DADF"
C_FILL = "#F4F7F8"
BG = "#FFFFFF"
C_ABS = "#B91C1C"
C_RING = "#1D4ED8"
C_GOLD = "#B45309"

SHAPE_COLORS = {
    "impulsive": "#0072B2",
    "emergent": "#E69F00",
    "multipath": "#009E73",
    "slow_coda": "#D55E00",
    "standard": "#7F7F7F",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.linewidth": 0.6,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "text.color": C_INK,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel_header(ax, letter: str, title: str, y: float = 0.50, title_dx_pt: float = 16.0) -> None:
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=C_INK,
        va="center",
        ha="left",
        zorder=20,
        clip_on=False,
    )
    ax.text(
        0.0,
        y,
        title,
        transform=ax.transAxes + ScaledTranslation(title_dx_pt / 72.0, 0.0, ax.figure.dpi_scale_trans),
        fontsize=9,
        color=C_INK,
        va="center",
        ha="left",
        zorder=20,
        clip_on=False,
        linespacing=1.25,
    )


def _round(ax, xy, w, h, fc, ec="#D5DADF", lw=0.8, r=0.08, z=2):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
        mutation_aspect=1,
    )
    ax.add_patch(box)
    return box


def _polylines(path: Path):
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
            for p in c:
                add(p)
        elif t == "Polygon":
            add(c[0])
        elif t == "MultiPolygon":
            for poly in c:
                add(poly[0])
    return lines


def _coast(path: Path):
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


def _basemap(ax, coast, faults, xlim, ylim):
    ax.set_facecolor(BG)
    for ring in coast:
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#F4F7F8", edgecolor=C_LINE, lw=0.35, zorder=0)
    if faults:
        ax.add_collection(LineCollection(faults, colors="#8A857A", linewidths=0.32, alpha=0.8, zorder=1))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, color=C_LINE, lw=0.35, zorder=0.4)
    ax.tick_params(labelsize=7)


def _inset_cbar(fig, ax, mappable, label):
    cax = inset_axes(
        ax,
        width="48%",
        height="4.0%",
        loc="lower left",
        bbox_to_anchor=(0.05, 0.17, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=7.0, labelpad=1.0)
    cb.ax.tick_params(labelsize=5.8, length=2.0, pad=0.8)
    cb.outline.set_linewidth(0.5)
    return cb


def _header_axes(fig, spec_row, ncols, wspace, width_ratios=None, head=0.24):
    kwargs = dict(height_ratios=[head, 1.0], hspace=0.08, wspace=wspace)
    if width_ratios is not None:
        kwargs["width_ratios"] = width_ratios
    inner = GridSpecFromSubplotSpec(2, ncols, subplot_spec=spec_row, **kwargs)
    heads, bodies = [], []
    for j in range(ncols):
        ah = fig.add_subplot(inner[0, j])
        ah.set_facecolor(BG)
        ah.axis("off")
        ah.set_xlim(0, 1)
        ah.set_ylim(0, 1)
        heads.append(ah)
        bodies.append(fig.add_subplot(inner[1, j]))
    return heads, bodies


def _arrow(ax, x0, x1, y):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.05, mutation_scale=8),
        zorder=5,
    )


def _draw_panel_a(ax):
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)
    ax.axis("off")
    t = np.linspace(0, 1, 360)
    rng = np.random.default_rng(1)

    # unboxed data input
    ax.text(2.05, 9.55, "Local EQ waveform", ha="center", fontsize=7.0, color=C_MUTED)
    noise = 0.05 * rng.normal(size=t.size)
    p = np.exp(-((t - 0.22) / 0.018) ** 2) * np.sin(55 * np.pi * t)
    s = 0.85 * np.exp(-((t - 0.42) / 0.035) ** 2) * np.sin(38 * np.pi * t)
    coda = 0.35 * np.exp(-3.8 * np.clip(t - 0.42, 0, None)) * np.sin(30 * np.pi * t)
    wave = p + s + coda + noise
    env = np.sqrt(np.clip(np.convolve(wave**2, np.ones(9) / 9.0, mode="same"), 0, None))
    env /= env.max() + 1e-9
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    xw = 0.25 + t * 3.60
    ax.plot(xw, 8.35 + 0.55 * wave, color=C_INK, lw=0.55, zorder=3)
    ax.fill_between(xw, 5.85, 5.85 + 1.55 * env, color=C_TEAL, alpha=0.20, zorder=2)
    ax.plot(xw, 5.85 + 1.55 * env, color=C_TEAL, lw=0.95, zorder=3)
    xp, xs = 0.25 + 0.22 * 3.60, 0.25 + 0.42 * 3.60
    ax.plot([xp, xp], [5.75, 8.95], color=C_MUTED, lw=0.5)
    ax.plot([xs, xs], [5.75, 8.95], color=C_MUTED, lw=0.5, ls="--")
    ax.text(xp, 9.08, "P", ha="center", fontsize=6.5, color=C_MUTED)
    ax.text(xs, 9.08, "S", ha="center", fontsize=6.5, color=C_MUTED)
    ax.text(2.05, 5.45, "STEAD  ·  3-comp + envelope", ha="center", fontsize=6.1, color=C_MUTED)

    top_y, top_h = 5.25, 4.55
    cards = [
        (4.55, 4.35, "#EAF4F8", C_TEAL, "1  Frozen picker"),
        (9.30, 4.55, "#F3F8F4", "#009E73", "2  Five facies"),
        (14.25, 4.20, "#FBF3EC", C_ACCENT, r"3  Coda slope $\rightarrow$ $\beta$"),
        (18.85, 4.95, "#F4F1FA", C_INK, "4  Structure + residual"),
    ]
    for x, w, fc, ec, title in cards:
        _round(ax, (x, top_y), w, top_h, fc, ec, lw=0.85, r=0.09)
        ax.text(x + 0.22, 9.48, title, ha="left", va="center", fontsize=7.0, color=C_INK, fontweight="bold")

    _arrow(ax, 3.90, 4.55, 7.50)
    for x0, x1 in [(8.90, 9.30), (13.85, 14.25), (18.45, 18.85)]:
        _arrow(ax, x0, x1, 7.50)

    ax.text(6.72, 8.15, "interpretable observables", ha="center", fontsize=6.5, color=C_INK)
    ax.text(
        6.72,
        6.85,
        "coda slope\nonset  ·  ρ(τ)\nthresholds frozen",
        ha="center",
        va="center",
        fontsize=6.6,
        color=C_TEAL,
        linespacing=1.55,
    )

    cartoons = {
        "impulsive": np.exp(-((t - 0.22) / 0.03) ** 2) * np.exp(-9 * np.clip(t - 0.22, 0, None)),
        "emergent": np.exp(-((t - 0.34) / 0.11) ** 2),
        "multipath": (
            np.exp(-((t - 0.22) / 0.03) ** 2)
            + 0.70 * np.exp(-((t - 0.42) / 0.04) ** 2)
            + 0.45 * np.exp(-((t - 0.62) / 0.05) ** 2)
        ),
        "slow_coda": np.exp(-((t - 0.26) / 0.05) ** 2) * np.exp(-1.4 * np.clip(t - 0.26, 0, None)),
        "standard": np.exp(-((t - 0.26) / 0.04) ** 2) * np.exp(-3.4 * np.clip(t - 0.26, 0, None)),
    }
    labels = ["impulsive", "emergent", "multipath", "slow coda", "standard"]
    keys = ["impulsive", "emergent", "multipath", "slow_coda", "standard"]
    for i, (k, lab) in enumerate(zip(keys, labels)):
        y0 = 8.58 - i * 0.56
        e = cartoons[k] / cartoons[k].max()
        xx = 9.52 + t * 2.15
        ax.fill_between(xx, y0, y0 + 0.42 * e, color=SHAPE_COLORS[k], alpha=0.88, lw=0, zorder=3)
        ax.text(11.82, y0 + 0.06, lab, fontsize=6.0, color=C_INK, va="center")

    tt = np.linspace(0.0, 1.0, 80)
    ax.scatter(
        14.55 + rng.random(22) * 3.15,
        8.40 - 1.15 * rng.random(22) - 0.15 * rng.random(22),
        s=7,
        c="#C9D0D3",
        zorder=2,
        linewidths=0,
    )
    ax.plot(14.55 + tt * 3.40, 8.45 - 1.35 * tt, color=C_ACCENT, lw=1.45, zorder=3)
    ax.plot(14.55 + tt * 3.40, 7.75 - 0.40 * tt, color=C_TEAL, lw=1.10, ls="--", zorder=3)
    ax.text(14.65, 8.55, "obs", fontsize=6.0, color=C_ACCENT, ha="left")
    ax.text(14.65, 7.45, r"$\mathbb{E}[\log d]$", fontsize=6.0, color=C_TEAL, ha="left")
    ax.text(16.35, 6.05, r"$\beta=$ slope − distance trend", ha="center", fontsize=6.4, color=C_INK)

    ax.text(
        21.32,
        8.20,
        r"$\beta\sim\log d+z+M$" "\n"
        r"$+\log(1+d_{\mathrm{flt}})$" "\n"
        r"$+$ station FE",
        ha="center",
        va="center",
        fontsize=6.6,
        color=C_INK,
        linespacing=1.50,
    )
    ax.plot([19.45, 23.20], [6.45, 6.45], color=C_INK, lw=1.0, zorder=3)
    ax.plot(19.45, 6.45, "o", color=C_INK, ms=5.2, zorder=4)
    ax.plot(23.20, 6.45, "s", color=C_TEAL, ms=5.2, zorder=4)
    ax.plot(21.32, 6.45, "D", color=C_ABS, ms=5.8, zorder=4)
    ax.text(19.45, 6.85, "src", ha="center", fontsize=6.0, color=C_MUTED)
    ax.text(23.20, 6.85, "sta", ha="center", fontsize=6.0, color=C_MUTED)
    ax.text(21.32, 6.05, "mid", ha="center", fontsize=6.0, color=C_ABS)
    ax.text(21.32, 5.65, r"leftover $\beta_{\mathrm{res}}$", ha="center", fontsize=6.4, color=C_ABS, fontweight="bold")

    _round(ax, (0.20, 0.20), 7.50, 4.70, "#FDECEC", C_ABS, lw=0.85)
    _round(ax, (8.00, 0.20), 7.50, 4.70, "#EAF0FB", C_RING, lw=0.85)
    _round(ax, (15.80, 0.20), 8.00, 4.70, C_FILL, C_LINE, lw=0.75)
    ax.text(3.95, 4.48, r"Absorbing   $\beta_{\mathrm{res}}<0$", ha="center", fontsize=7.1, color=C_ABS, fontweight="bold")
    ax.text(11.75, 4.48, r"Ringing   $\beta_{\mathrm{res}}>0$", ha="center", fontsize=7.1, color=C_RING, fontweight="bold")
    ax.text(19.80, 4.48, "Same-station control", ha="center", fontsize=7.1, color=C_INK, fontweight="bold")

    t2 = np.linspace(0, 1, 200)
    fast = np.exp(-((t2 - 0.18) / 0.045) ** 2) * np.exp(-6.8 * np.clip(t2 - 0.18, 0, None))
    slow = np.exp(-((t2 - 0.18) / 0.045) ** 2) * np.exp(-2.0 * np.clip(t2 - 0.18, 0, None))
    fast /= fast.max()
    slow /= slow.max()
    ax.fill_between(0.58 + t2 * 6.74, 1.10, 1.10 + 2.60 * fast, color=C_ABS, alpha=0.18, lw=0)
    ax.plot(0.58 + t2 * 6.74, 1.10 + 2.60 * fast, color=C_ABS, lw=1.4)
    ax.fill_between(8.38 + t2 * 6.74, 1.10, 1.10 + 2.60 * slow, color=C_RING, alpha=0.16, lw=0)
    ax.plot(8.38 + t2 * 6.74, 1.10 + 2.60 * slow, color=C_RING, lw=1.4)
    ax.text(3.95, 0.55, "faster coda decay", ha="center", fontsize=6.3, color=C_ABS)
    ax.text(11.75, 0.55, "energy persists after S", ha="center", fontsize=6.3, color=C_RING)

    ax.text(19.80, 3.85, "Same site, other paths = control", ha="center", fontsize=6.3, color=C_INK)
    ax.plot([17.70, 19.80], [2.45, 1.15], color=C_ABS, lw=1.1, zorder=3)
    ax.plot([17.70, 22.00], [2.45, 1.15], color=C_RING, lw=1.1, zorder=3)
    ax.plot(17.70, 2.45, "s", color=C_TEAL, ms=6.3, zorder=4)
    ax.plot(19.80, 1.15, "o", color=C_ABS, ms=5.2, zorder=4)
    ax.plot(22.00, 1.15, "o", color=C_RING, ms=5.2, zorder=4)
    ax.plot(18.75, 1.80, "D", color=C_ABS, ms=4.4, zorder=4)
    ax.plot(19.85, 1.80, "D", color=C_RING, ms=4.4, zorder=4)
    ax.text(17.70, 2.88, "one station", ha="center", fontsize=6.0, color=C_TEAL)
    ax.text(19.80, 0.68, "path A", ha="center", fontsize=5.9, color=C_ABS)
    ax.text(22.00, 0.68, "path B", ha="center", fontsize=5.9, color=C_RING)


def main():
    _style()
    out = _REPO / "docs/figures/nc_main"
    out.mkdir(parents=True, exist_ok=True)

    socal_cells = pd.read_csv(_REPO / "outputs/structure_residual_socal/grid_beta_resid.csv")
    socal_ss = pd.read_csv(_REPO / "outputs/structure_residual_socal/same_station_validation.csv")
    qc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_classical_qc.csv")
    qcc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_classical_qc_cells.csv")
    berg = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_berg2021.csv")
    linq = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_lin2023_q.csv")
    bergc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_berg2021_cells.csv")
    linc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_lin2023_q_cells.csv")
    cas_cells = pd.read_csv(_REPO / "outputs/structure_residual_cascadia_volc/grid_beta_resid.csv")
    cas_ss = pd.read_csv(_REPO / "outputs/structure_residual_cascadia_volc/same_station_validation.csv")

    coast = _coast(_REPO / "docs/figures/geo/ne_110m_land.geojson")
    faults_s = _polylines(_REPO / "docs/figures/geo/qfaults_socal.geojson") + _polylines(
        _REPO / "docs/figures/geo/qfaults_socal_offshore.geojson"
    )
    faults_c = []
    for p in (
        "docs/figures/geo/qfaults_pnw_wa.geojson",
        "docs/figures/geo/qfaults_pnw_wa_offshore.geojson",
        "docs/figures/geo/qfaults_pnw_or.geojson",
    ):
        faults_c += _polylines(_REPO / p)

    # Wider plate; 3 content rows with asymmetric gaps (tight a→row2, roomier row2→row3)
    fig = plt.figure(figsize=(252 / 25.4, 245 / 25.4), facecolor=BG)
    gs = GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[1.22, 0.035, 1.12, 0.13, 1.28],
        hspace=0.0,
        left=0.070,
        right=0.975,
        top=0.970,
        bottom=0.042,
    )

    # ---- Row 1: a method ----
    h0, b0 = _header_axes(fig, gs[0], ncols=1, wspace=0.0, head=0.18)
    _panel_header(h0[0], "a", "From waveform facies to a structure-controlled path residual")
    _draw_panel_a(b0[0])

    # ---- Row 2: same-station + independence  (letters b–e; widths 1:2:2:2) ----
    # content order: old c, d, e, f
    h1, b1 = _header_axes(
        fig,
        gs[2],
        ncols=4,
        wspace=0.28,
        width_ratios=[1, 1, 1, 1],
        head=0.20,
    )
    ax_ss, ax_qc, ax_vs, ax_qs = b1
    _panel_header(h1[0], "b", "Same-station test")
    _panel_header(h1[1], "c", r"More than scalar $q_c$")
    _panel_header(h1[2], "d", r"Berg 2021 $V_S$")
    _panel_header(h1[3], "e", r"Lin 2023 $Q_S$")

    rows = []
    for rank, lab in [(2, "#2 Salton"), (5, "#5 ETR"), (1, "#1 SJ"), (6, "#6 SA–SJ")]:
        r = socal_ss.loc[socal_ss["rank"] == rank].iloc[0]
        rows.append((lab, float(r.site_delta_med), int(r.n_sites_paired), float(r.mw_p)))
    for rank, lab in [(7, "St Helens"), (2, "Seattle")]:
        r = cas_ss.loc[cas_ss["rank"] == rank].iloc[0]
        rows.append((lab, float(r.site_delta_med), int(r.n_sites_paired), float(r.mw_p)))
    ys = np.arange(len(rows))[::-1]
    ax_ss.axvline(0, color=C_LINE, lw=0.7)
    ax_ss.barh(ys, [r[1] for r in rows], color=[C_ABS if r[1] < 0 else C_RING for r in rows], height=0.58, zorder=2)
    ax_ss.set_yticks(ys)
    ax_ss.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    ax_ss.set_xlabel(r"median $\Delta\beta$", fontsize=7.5)
    ax_ss.tick_params(labelsize=6.2)
    ax_ss.set_xlim(-0.048, 0.072)
    ax_ss.set_ylim(-0.70, 5.70)
    for y, r in zip(ys, rows):
        x = r[1]
        # place n on the far side of zero so it does not sit on the bar
        if x >= 0:
            ax_ss.text(-0.002, y, f"n={r[2]}", va="center", ha="right", fontsize=5.4, color=C_MUTED, clip_on=True)
        else:
            ax_ss.text(0.002, y, f"n={r[2]}", va="center", ha="left", fontsize=5.4, color=C_MUTED, clip_on=True)

    def _mark(ax, df, xcol, legend=True):
        m2 = np.isclose(df.gx, -116.375) & np.isclose(df.gy, 33.375)
        m5 = np.isclose(df.gx, -116.375) & np.isclose(df.gy, 33.875)
        ax.scatter(df[xcol], df.beta_resid, s=3.2, c="#C9D0D3", alpha=0.35, linewidths=0, rasterized=True, zorder=1)
        ax.scatter(df.loc[m2, xcol], df.loc[m2, "beta_resid"], s=9, c=C_ABS, alpha=0.85, linewidths=0, zorder=3, label="#2")
        ax.scatter(df.loc[m5, xcol], df.loc[m5, "beta_resid"], s=9, c=C_RING, alpha=0.88, linewidths=0, zorder=3, label="#5")
        ax.axhline(0, color=C_LINE, lw=0.55)
        if legend:
            ax.legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.3, borderaxespad=0.15)
        ax.tick_params(labelsize=6.5)

    _mark(ax_qc, qc, "qc_slope")
    x = qc.qc_slope.to_numpy()
    y = qc.beta_resid.to_numpy()
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)
    xx = np.linspace(np.nanpercentile(x, 3), np.nanpercentile(x, 97), 40)
    ax_qc.plot(xx, coef[0] + coef[1] * xx, color=C_INK, lw=1.0, zorder=2)
    ax_qc.set_xlabel("classical coda-decay slope", fontsize=7.5)
    ax_qc.set_ylabel(r"$\beta_{\mathrm{res}}$", fontsize=7.5)
    ax_qc.text(0.97, 0.97, r"$r=0.64$", transform=ax_qc.transAxes, ha="right", va="top", fontsize=7.0, color=C_INK)
    left = [
        float(qcc.loc[qcc["rank"] == 2, "delta_beta_after_qc"].iloc[0]),
        float(qcc.loc[qcc["rank"] == 5, "delta_beta_after_qc"].iloc[0]),
    ]
    ax_qc.text(
        0.97,
        0.14,
        f"leftover Δβ after $q_c$\n#2 {left[0]:+.3f}   #5 {left[1]:+.3f}",
        transform=ax_qc.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=C_INK,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=C_LINE, linewidth=0.5, alpha=0.40),
    )

    _mark(ax_vs, berg, "vs_0to8", legend=False)
    ax_vs.axvspan(3.05, 3.21, color="#F7E7C6", alpha=0.55, zorder=0)
    ax_vs.text(0.97, 0.97, r"SPLIT   $r=-0.08$", transform=ax_vs.transAxes, ha="right", va="top", fontsize=7.0, color=C_INK)
    ax_vs.text(2.78, 0.168, "same $V_S$ band\nΔβ = +0.046", ha="center", va="top", fontsize=6.1, color=C_GOLD, linespacing=1.25)
    ax_vs.set_xlabel(r"path-mid $V_S$ 0–8 km (km s$^{-1}$)", fontsize=7.5)
    ax_vs.set_xlim(2.62, 3.40)
    ax_vs.tick_params(labelleft=False)

    _mark(ax_qs, linq, "qs_0to8", legend=False)
    ax_qs.set_xlabel(r"path-mid $Q_S$ 0–8 km", fontsize=7.5)
    ax_qs.tick_params(labelleft=False)
    ax_qs.text(0.97, 0.97, r"anti-aligned   $r=-0.07$", transform=ax_qs.transAxes, ha="right", va="top", fontsize=7.0, color=C_INK)
    ax_qs.annotate(
        "high $Q_S$,\nfaster coda",
        xy=(790, -0.03),
        xytext=(640, -0.145),
        fontsize=6.0,
        color=C_ABS,
        ha="center",
        linespacing=1.20,
        arrowprops=dict(arrowstyle="->", color=C_ABS, lw=0.7),
    )
    ax_qs.annotate(
        "lower $Q_S$,\nrings",
        xy=(555, 0.045),
        xytext=(430, 0.150),
        fontsize=6.0,
        color=C_RING,
        ha="center",
        linespacing=1.20,
        arrowprops=dict(arrowstyle="->", color=C_RING, lw=0.7),
    )

    # ---- Row 3: SoCal map + Cascadia + synthesis  (letters f–h; widths 4:4:3) ----
    h2, b2 = _header_axes(
        fig,
        gs[4],
        ncols=3,
        wspace=0.22,
        width_ratios=[4, 4, 3],
        head=0.18,
    )
    ax_soc, ax_cas, ax_syn = b2
    _panel_header(h2[0], "f", r"SoCal: path-midpoint $\beta_{\mathrm{res}}$")
    _panel_header(h2[1], "g", "Out-of-region replica (Cascadia)")
    _panel_header(h2[2], "h", "Absorbing vs ringing")

    # Horizontal separation only (keep map rendering unchanged): f left, g/h right
    fig.canvas.draw()
    for ax in (h2[0], b2[0]):
        p = ax.get_position()
        ax.set_position([p.x0 - 0.012, p.y0, p.width, p.height])
    for ax in (h2[1], b2[1], h2[2], b2[2]):
        p = ax.get_position()
        ax.set_position([p.x0 + 0.022, p.y0, p.width, p.height])

    _basemap(ax_soc, coast, faults_s, (-118.55, -115.35), (32.55, 34.55))
    nrm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    sc = ax_soc.scatter(
        socal_cells.lon,
        socal_cells.lat,
        c=socal_cells["mean"],
        s=np.clip(socal_cells.n * 0.48, 14, 120),
        cmap="coolwarm_r",
        norm=nrm,
        edgecolors="white",
        linewidths=0.2,
        zorder=3,
    )
    _inset_cbar(fig, ax_soc, sc, r"$\beta_{\mathrm{res}}$  (red = faster decay)")
    ax_soc.plot(-116.375, 33.375, marker="s", ms=8, mfc="none", mec=C_ABS, mew=1.35, zorder=5)
    ax_soc.plot(-116.375, 33.875, marker="s", ms=8, mfc="none", mec=C_RING, mew=1.35, zorder=5)
    ax_soc.text(-116.55, 33.16, "#2 Salton\nΔβ = −0.032", color=C_ABS, fontsize=6.6, fontweight="bold", ha="right", zorder=5, linespacing=1.25)
    ax_soc.text(-116.55, 34.10, "#5 ETR\nΔβ = +0.025", color=C_RING, fontsize=6.6, fontweight="bold", ha="right", zorder=5, linespacing=1.25)

    _basemap(ax_cas, coast, faults_c, (-124.65, -120.35), (45.35, 48.65))
    sc2 = ax_cas.scatter(
        cas_cells.lon,
        cas_cells.lat,
        c=cas_cells["mean"],
        s=np.clip(cas_cells.n * 0.26, 10, 105),
        cmap="coolwarm_r",
        norm=nrm,
        edgecolors="white",
        linewidths=0.2,
        zorder=3,
    )
    _inset_cbar(fig, ax_cas, sc2, r"$\beta_{\mathrm{res}}$  (red = faster decay)")
    ax_cas.plot(-122.375, 46.375, marker="s", ms=8, mfc="none", mec=C_ABS, mew=1.35, zorder=5)
    ax_cas.plot(-122.375, 47.375, marker="s", ms=8, mfc="none", mec=C_RING, mew=1.35, zorder=5)
    ax_cas.text(-123.95, 46.55, "St Helens–Cowlitz\nΔβ = −0.014", color=C_ABS, fontsize=6.4, fontweight="bold", ha="left", linespacing=1.25)
    ax_cas.text(-123.95, 47.90, "Seattle basin\nΔβ = +0.045", color=C_RING, fontsize=6.4, fontweight="bold", ha="left", linespacing=1.25)

    # Compact synthesis panel h: fill canvas leftward; text kept inside boxes
    ax_syn.set_xlim(0, 10)
    ax_syn.set_ylim(0, 10)
    ax_syn.axis("off")
    # top cards — slightly taller, nearly full width, small gap
    top_y, top_h = 6.55, 3.20
    left_x, right_x, card_w = 0.10, 5.10, 4.80
    _round(ax_syn, (left_x, top_y), card_w, top_h, "#FDECEC", C_ABS, lw=0.90, r=0.07)
    _round(ax_syn, (right_x, top_y), card_w, top_h, "#EAF0FB", C_RING, lw=0.90, r=0.07)
    cx1, cx2 = left_x + 0.5 * card_w, right_x + 0.5 * card_w
    ax_syn.text(cx1, top_y + top_h - 0.38, "ABSORBING", ha="center", fontsize=7.0, fontweight="bold", color=C_ABS)
    ax_syn.text(cx2, top_y + top_h - 0.38, "RINGING", ha="center", fontsize=7.0, fontweight="bold", color=C_RING)
    ax_syn.text(
        cx1,
        top_y + 0.52 * top_h,
        "Salton–Imperial\nSt Helens",
        ha="center",
        va="center",
        fontsize=6.0,
        color=C_INK,
        linespacing=1.40,
    )
    ax_syn.text(
        cx2,
        top_y + 0.52 * top_h,
        "Transverse Ranges\nSeattle basin",
        ha="center",
        va="center",
        fontsize=6.0,
        color=C_INK,
        linespacing=1.40,
    )
    ax_syn.text(cx1, top_y + 0.18 * top_h, r"$\beta_{\mathrm{res}}<0$", ha="center", fontsize=5.8, color=C_ABS)
    ax_syn.text(cx2, top_y + 0.18 * top_h, r"$\beta_{\mathrm{res}}>0$", ha="center", fontsize=5.8, color=C_RING)

    bot_x, bot_y, bot_w, bot_h = 0.10, 0.18, 9.80, 5.95
    _round(ax_syn, (bot_x, bot_y), bot_w, bot_h, C_FILL, C_LINE, lw=0.70, r=0.07)
    ax_syn.text(
        bot_x + 0.5 * bot_w,
        bot_y + bot_h - 0.38,
        r"Leftover $\Delta\beta$ after removing each observable",
        ha="center",
        fontsize=6.0,
        color=C_MUTED,
    )
    d2q, d5q = left
    d2v = float(bergc.loc[bergc["rank"] == 2, "delta_beta_after_vs"].iloc[0])
    d5v = float(bergc.loc[bergc["rank"] == 5, "delta_beta_after_vs"].iloc[0])
    d2s = float(linc.loc[linc["rank"] == 2, "delta_beta_after_qs"].iloc[0])
    d5s = float(linc.loc[linc["rank"] == 5, "delta_beta_after_qs"].iloc[0])

    y0_bar = bot_y + 0.42 * bot_h

    def ymap(v):
        return y0_bar + v * (0.36 * bot_h / 0.05)

    ax_syn.plot([bot_x + 0.55, bot_x + 0.92 * bot_w], [y0_bar, y0_bar], color=C_LINE, lw=0.7, zorder=2)
    for i, (lab, a, b) in enumerate(
        zip([r"after $q_c$", r"after $V_S$", r"after $Q_S$"], [d2q, d2v, d2s], [d5q, d5v, d5s])
    ):
        x0 = bot_x + 1.55 + i * 2.75
        ax_syn.add_patch(Rectangle((x0 - 0.36, min(y0_bar, ymap(a))), 0.32, abs(ymap(a) - y0_bar), fc=C_ABS, zorder=3, lw=0))
        ax_syn.add_patch(Rectangle((x0 + 0.10, min(y0_bar, ymap(b))), 0.32, abs(ymap(b) - y0_bar), fc=C_RING, zorder=3, lw=0))
        ax_syn.text(x0 + 0.03, bot_y + 0.14 * bot_h, lab, ha="center", fontsize=5.9, color=C_INK)
    ax_syn.plot(bot_x + 0.75, bot_y + 0.82 * bot_h, "s", color=C_ABS, ms=4.8)
    ax_syn.text(bot_x + 1.00, bot_y + 0.82 * bot_h, "#2", color=C_ABS, fontsize=5.9, va="center")
    ax_syn.plot(bot_x + 1.85, bot_y + 0.82 * bot_h, "s", color=C_RING, ms=4.8)
    ax_syn.text(bot_x + 2.10, bot_y + 0.82 * bot_h, "#5", color=C_RING, fontsize=5.9, va="center")
    ax_syn.text(
        bot_x + 0.5 * bot_w,
        bot_y + 0.06 * bot_h,
        r"Predict SCSN Salton: $\Delta\beta\leq-0.02$ ($n_{\mathrm{st}}\geq8$)",
        ha="center",
        fontsize=5.8,
        color=C_INK,
    )

    # Final left/right separation after datalim maps settle (widths unchanged)
    fig.canvas.draw()
    pf = ax_soc.get_position()
    pg = ax_cas.get_position()
    need = 0.040
    dx = (pf.x1 + need) - pg.x0
    if dx > 0:
        lock_left = {h2[0], ax_soc}
        for ax in fig.axes:
            if ax in lock_left:
                continue
            p = ax.get_position()
            # move g/h columns and their insets only
            if p.x0 >= pg.x0 - 0.03:
                ax.set_position([p.x0 + dx, p.y0, p.width, p.height])
        # if h would exceed right margin, pull f further left instead of clipping h
        ph = ax_syn.get_position()
        overflow = ph.x1 - 0.985
        if overflow > 0:
            for ax in (h2[0], ax_soc):
                p = ax.get_position()
                ax.set_position([p.x0 - overflow, p.y0, p.width, p.height])
            for ax in fig.axes:
                if ax in (h2[0], ax_soc):
                    continue
                p = ax.get_position()
                if p.x0 >= pg.x0 - 0.05:
                    ax.set_position([p.x0 - overflow, p.y0, p.width, p.height])

    # Nudge h a little left (header + body only)
    for ax in (h2[2], ax_syn):
        p = ax.get_position()
        ax.set_position([p.x0 - 0.012, p.y0, p.width, p.height])

    fig.savefig(out / "Fig_main_single.pdf", dpi=320, pad_inches=0.03)
    fig.savefig(out / "Fig_main_single.png", dpi=320, pad_inches=0.03)
    plt.close(fig)
    print(f"[done] {out / 'Fig_main_single.png'}", flush=True)


if __name__ == "__main__":
    main()
