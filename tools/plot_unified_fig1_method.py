#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified-propagation Fig. 1 — Nature-style method schematic.

Panels (unified-propagation-article.tex / fig:method):
  a  Recursive local sources (landscape cards; no arcs)
  b  Operator P(X; G, Θ, α) — X/G/Θ/α parallel column
  c  Mechanism axes + domain landings (chart | table)
  d  Mini Θ(λ) phase strip (real wave→diffusion path)

Outputs:
  docs/figures/unified/fig1_method.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
C_SOFT = "#F3F4F6"
BG = "#FFFFFF"
C_TEAL = "#0F766E"
C_BLUE = "#1D4ED8"
C_ORANGE = "#C2410C"
C_GOLD = "#B45309"
C_GREEN = "#047857"
C_WAVE = "#0072B2"
C_INST = "#E69F00"
C_GDIFF = "#009E73"
C_DIFF = "#D55E00"
C_DAMP = "#56B4E9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/figures/unified")
    p.add_argument(
        "--theta-lambda",
        default="outputs/propagation_dynamics/unified_theta_lambda_v1/wave_to_diffusion.json",
    )
    p.add_argument("--dpi", type=int, default=400)
    return p.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BG,
            "figure.facecolor": BG,
        }
    )


def _panel(ax, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=C_INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.annotate(
        title,
        xy=(0.0, 1.04),
        xycoords=ax.transAxes,
        xytext=(14, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=C_INK,
        va="bottom",
        ha="left",
        annotation_clip=False,
    )


def _round(ax, xy, w, h, fc=BG, ec=C_INK, lw=1.0, r=0.02, z=2):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, p0, p1, color=C_INK, lw=1.2, ms=9, z=4):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0",
        zorder=z,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)



# Shared soft background (a and c use identical geometry)
SOFT_BOX = (0.03, 0.08, 0.94, 0.84)  # x, y, w, h


def _soft_bg(ax, box=SOFT_BOX, alpha=0.35):
    x, y, w, h = box
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=C_SOFT,
            edgecolor="none",
            linewidth=0,
            alpha=alpha,
            zorder=0,
        )
    )


def draw_a(ax) -> None:
    """Vivid recursive graph; titles share axes box with other panels (no equal-aspect shrink)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel(ax, "a", "Recursive local sources")

    _soft_bg(ax)


    # Enlarged, more open layout
    nodes = {
        "i": (0.24, 0.50),
        "j1": (0.72, 0.78),
        "j2": (0.84, 0.50),
        "j3": (0.68, 0.20),
    }
    for key, col in (("j1", C_TEAL), ("j2", C_BLUE), ("j3", C_ORANGE)):
        p0, p1 = nodes["i"], nodes[key]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=1.7, alpha=0.9, zorder=2)
        mx = 0.52 * p0[0] + 0.48 * p1[0]
        my = 0.52 * p0[1] + 0.48 * p1[1]
        off = {"j1": (-0.01, 0.045), "j2": (0.0, 0.04), "j3": (-0.01, -0.05)}[key]
        ax.text(
            mx + off[0],
            my + off[1],
            r"$(A_{ij},\tau_{ij})$",
            fontsize=6.8,
            color=col,
            ha="center",
            va="center",
            zorder=5,
        )

    xi, yi = nodes["i"]
    ax.add_patch(Circle((xi, yi), 0.065, fc="#ECFDF5", ec=C_TEAL, lw=1.5, zorder=4))
    ax.text(xi, yi, r"$x_i$", fontsize=10, ha="center", va="center", color=C_INK, zorder=5)
    ax.text(xi, yi + 0.12, "local state", fontsize=7.2, ha="center", color=C_MUTED)
    ax.text(xi, yi - 0.12, r"$\to$ local response", fontsize=7.2, ha="center", color=C_TEAL)

    for k, (key, col) in enumerate((("j1", C_TEAL), ("j2", C_BLUE), ("j3", C_ORANGE))):
        x, y = nodes[key]
        ax.add_patch(Circle((x, y), 0.042, fc=BG, ec=col, lw=1.35, zorder=4))
        ax.plot(x, y, "o", color=col, ms=5.0, zorder=5)
        for rr, a in ((0.065, 0.65), (0.100, 0.40), (0.135, 0.22)):
            th0, th1 = -30 + 28 * k, 160 + 28 * k
            ax.add_patch(
                Arc(
                    (x, y),
                    2 * rr,
                    2 * rr,
                    angle=0,
                    theta1=th0,
                    theta2=th1,
                    color=col,
                    lw=1.15,
                    alpha=a,
                    zorder=3,
                )
            )

    ax.text(0.82, 0.92, "secondary sources", fontsize=7.5, color=C_INK, ha="center")


def draw_b(ax) -> None:
    """X, G, Θ, α in one row; straight arrows to spaced landings on P."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel(ax, "b", r"Operator $\hat{X}=P(X;G,\Theta,\alpha)$")

    specs = [
        (r"$X$", "field", "#EFF6FF", C_BLUE),
        (r"$G$", "geometry", "#ECFDF5", C_TEAL),
        (r"$\Theta$", "mechanism", "#FFF7ED", C_ORANGE),
        (r"$\alpha$", "gate", "#FEF3C7", C_GOLD),
    ]
    iw, ih, iy = 0.16, 0.14, 0.72
    xs = np.linspace(0.04, 0.80, len(specs))
    lands = np.linspace(0.26, 0.62, len(specs))
    for x, land, (lab, sub, fc, ec) in zip(xs, lands, specs):
        _round(ax, (float(x), iy), iw, ih, fc=fc, ec=ec, lw=1.1, r=0.02)
        ax.text(float(x) + iw / 2, iy + ih * 0.62, lab, fontsize=11, ha="center", va="center", color=C_INK)
        ax.text(float(x) + iw / 2, iy + ih * 0.28, sub, fontsize=6.3, ha="center", va="center", color=C_MUTED)
        # Straight arrows (no arc)
        _arrow(ax, (float(x) + iw / 2, iy), (float(land), 0.50), color=ec, lw=1.05, ms=8)

    _round(ax, (0.22, 0.24), 0.44, 0.26, fc=BG, ec=C_INK, lw=1.5, r=0.03)
    ax.text(0.44, 0.42, r"$P$", fontsize=15, ha="center", va="center", color=C_INK, fontweight="bold")
    ax.text(0.44, 0.31, "unified propagator", fontsize=6.8, ha="center", va="center", color=C_MUTED)

    _round(ax, (0.76, 0.26), 0.18, 0.22, fc="#F0FDF4", ec=C_GREEN, lw=1.3, r=0.025)
    ax.text(0.85, 0.40, r"$\hat{X}$", fontsize=12, ha="center", va="center", color=C_INK)
    ax.text(0.85, 0.30, "next state", fontsize=6.4, ha="center", va="center", color=C_MUTED)
    _arrow(ax, (0.66, 0.37), (0.76, 0.37), color=C_GREEN, lw=1.25, ms=10)


def draw_c(ax) -> None:
    """Θ plane with three domain landings; same soft box as panel a; no extra legend."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _panel(ax, "c", r"Physical axes of $\Theta$ + domain landings")
    _soft_bg(ax)  # identical geometry to panel a

    # Soft clouds only for landed domains (no unused heat cloud)
    for cx, cy, w, h, col, a in (
        (0.26, 0.30, 0.36, 0.32, C_INST, 0.16),   # EEG
        (0.72, 0.30, 0.40, 0.34, C_WAVE, 0.15),   # STEAD
        (0.48, 0.62, 0.38, 0.34, C_GDIFF, 0.16),  # SST
    ):
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=col, edgecolor="none", alpha=a, zorder=1))
        ax.add_patch(Ellipse((cx, cy), w * 0.55, h * 0.55, facecolor=col, edgecolor="none", alpha=a * 0.65, zorder=1))

    ox, oy = 0.16, 0.22
    ax.annotate(
        "",
        xy=(0.88, oy),
        xytext=(ox, oy),
        arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.35, mutation_scale=10),
    )
    ax.annotate(
        "",
        xy=(ox, 0.88),
        xytext=(ox, oy),
        arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.35, mutation_scale=10),
    )
    # Axis title placed clear of STEAD marker/label
    ax.text(0.88, oy - 0.06, r"delay $\delta$", fontsize=8.5, ha="right", color=C_INK)
    ax.text(ox - 0.04, 0.88, r"heat $\eta$", fontsize=8.5, ha="right", va="top", color=C_INK, rotation=90)

    ax.annotate(
        "",
        xy=(0.58, 0.58),
        xytext=(ox + 0.03, oy + 0.03),
        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.1, mutation_scale=9),
    )
    ax.text(0.60, 0.60, r"spatial $\sigma$", fontsize=8, color=C_MUTED, ha="left")

    # Huygens arcs near STEAD only
    for rr, a in ((0.055, 0.65), (0.09, 0.40), (0.125, 0.22)):
        ax.add_patch(
            Arc(
                (0.70, 0.32),
                2 * rr,
                2 * rr,
                angle=0,
                theta1=-20,
                theta2=200,
                color=C_WAVE,
                lw=1.15,
                alpha=a,
                zorder=2,
            )
        )

    # Landings — labels offset to avoid axis-title clash (STEAD above-right, not below)
    landings = [
        (0.26, 0.32, C_INST, "EEG", (-0.02, 0.08)),
        (0.70, 0.32, C_WAVE, "STEAD", (0.02, 0.10)),
        (0.48, 0.62, C_GDIFF, "SST", (0.0, 0.09)),
    ]
    for x, y, col, name, (dx, dy) in landings:
        ax.plot(x, y, "o", ms=20, color=col, alpha=0.20, zorder=3)
        ax.plot(x, y, "o", ms=12, color=col, markeredgecolor=C_INK, markeredgewidth=0.75, zorder=5)
        ax.plot(x, y, "o", ms=3.6, color=BG, zorder=6)
        if name == "SST":
            ax.add_patch(Ellipse((x, y), 0.14, 0.12, fill=False, ec=col, lw=1.25, ls="--", zorder=4))
        ax.text(
            x + dx,
            y + dy,
            name,
            fontsize=8.5,
            fontweight="bold",
            color=col,
            ha="center",
            va="center",
            zorder=6,
        )


def draw_d(ax_chart, ax_leg, path_json: Path) -> None:
    """Path curves on ax_chart; legend on ax_leg — same vertical budget as panel c."""
    ax = ax_chart
    _panel(ax, "d", r"Telegrapher path $\Theta(\lambda)$: wave $\to$ damped-wave $\to$ diffusion")

    rows = json.loads(path_json.read_text(encoding="utf-8"))["rows"]
    lam = np.array([r["lambda"] for r in rows], float)
    lag = np.array([r["mean_lag"] for r in rows], float)
    damp = np.array([r["regime_coords"]["damping"] for r in rows], float)
    phases = [r["phase"] for r in rows]

    phase_color = {
        "wave_like": C_WAVE,
        "damped_wave": C_DAMP,
        "instantaneous_like": C_INST,
        "diffusive": C_DIFF,
        "transitional": C_MUTED,
    }
    for i in range(len(lam) - 1):
        ax.axvspan(lam[i], lam[i + 1], color=phase_color.get(phases[i], C_MUTED), alpha=0.12, lw=0, zorder=0)
    ax.axvspan(lam[-1] - 0.025, 1.0, color=phase_color.get(phases[-1], C_MUTED), alpha=0.12, lw=0, zorder=0)

    ax2 = ax.twinx()
    ax.plot(lam, lag, color=C_WAVE, lw=2.0, marker="o", ms=3.2, zorder=3)
    ax2.plot(lam, damp, color=C_ORANGE, lw=1.8, marker="s", ms=2.8, zorder=3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, max(lag) * 1.25 + 0.02)
    ax2.set_ylim(-0.05, 1.15)
    ax.set_xlabel(r"path coordinate $\lambda$", fontsize=7.5, color=C_INK, labelpad=6)
    ax.set_ylabel("mean lag (samples)", fontsize=7.5, color=C_WAVE, labelpad=2)
    ax2.set_ylabel(r"travel-time damping $\zeta$", fontsize=7.5, color=C_ORANGE, labelpad=2)
    ax.tick_params(axis="y", colors=C_WAVE, labelsize=7)
    ax2.tick_params(axis="y", colors=C_ORANGE, labelsize=7)
    ax.tick_params(axis="x", labelsize=7, colors=C_INK)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.spines["left"].set_color(C_WAVE)
    ax2.spines["right"].set_color(C_ORANGE)
    ax.spines["bottom"].set_color(C_LINE)

    ax.text(0.0, max(lag) * 1.08, r"wave ($\lambda{=}0$)", fontsize=6.8, color=C_WAVE, ha="left")
    ax.text(1.0, max(lag) * 1.08, r"diffusion ($\lambda{=}1$)", fontsize=6.8, color=C_DIFF, ha="right")

    damp_idx = [i for i, p in enumerate(phases) if p == "damped_wave"]
    if damp_idx:
        i_mid = damp_idx[len(damp_idx) // 2]
        ax.annotate(
            "damped-wave",
            xy=(lam[i_mid], lag[i_mid]),
            xytext=(0.28, max(lag) * 0.72),
            fontsize=6.8,
            color=C_DAMP,
            arrowprops=dict(arrowstyle="-|>", color=C_DAMP, lw=0.8, mutation_scale=7),
        )
    i_peak = int(np.argmax(damp))
    ax2.plot(lam[i_peak], damp[i_peak], "D", color=C_ORANGE, ms=4.5, zorder=4)
    ax2.text(lam[i_peak] + 0.02, damp[i_peak] - 0.12, r"$\zeta$ peak", fontsize=6.5, color=C_ORANGE, ha="left")

    # Legend inside dedicated strip (keeps c/d axes boxes the same height)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    legend = [
        Line2D([0], [0], color=C_WAVE, lw=2.0, marker="o", ms=3.5, label="mean lag"),
        Line2D([0], [0], color=C_ORANGE, lw=1.8, marker="s", ms=3.0, label=r"damping $\zeta$"),
        Line2D([0], [0], color=C_WAVE, lw=6, alpha=0.25, label="wave-like"),
        Line2D([0], [0], color=C_DAMP, lw=6, alpha=0.25, label="damped-wave"),
        Line2D([0], [0], color=C_INST, lw=6, alpha=0.25, label="instantaneous-like"),
        Line2D([0], [0], color=C_DIFF, lw=6, alpha=0.25, label="diffusive"),
    ]
    ax_leg.legend(
        handles=legend,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handlelength=1.6,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def main() -> None:
    args = parse_args()
    _style()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path_json = Path(args.theta_lambda)
    if not path_json.is_absolute():
        path_json = _REPO / path_json
    if not path_json.is_file():
        raise FileNotFoundError(path_json)

    # Taller canvas; bottom row uses identical nested chart/legend ratios for c and d
    fig = plt.figure(figsize=(7.6, 7.8), dpi=150)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.95, 1.25],
        width_ratios=[1.0, 1.0],
        left=0.06,
        right=0.96,
        top=0.94,
        bottom=0.06,
        wspace=0.22,
        hspace=0.14,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    ax_c = fig.add_subplot(gs[1, 0])

    # d keeps a legend strip, with extra gap so it clears the x-label
    gs_d = gs[1, 1].subgridspec(2, 1, height_ratios=[5.6, 1.05], hspace=0.22)
    ax_d_chart = fig.add_subplot(gs_d[0])
    ax_d_leg = fig.add_subplot(gs_d[1])

    draw_a(ax_a)
    draw_b(ax_b)
    draw_c(ax_c)
    draw_d(ax_d_chart, ax_d_leg, path_json)

    png = out_dir / "fig1_method.png"
    pdf = out_dir / "fig1_method.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print({"png": str(png), "pdf": str(pdf)})


if __name__ == "__main__":
    main()
