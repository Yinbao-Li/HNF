#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PNF Figure 1 — principle, latents, and discovery paradigm.

Layout:
  a  Light-cone Huygens field + learnable kernel K(γ, ω, c)
  b  STEAD forward path (3C → field → det/P/S)
  c  Interpretable latent variables (native redraw, no nested letters)
  d  Causal operator · kernel families · discovery workflow

Outputs:
  docs/figures/pnf/pnf_journal_fig1_principle.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Ellipse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
import math
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

C_INK = "#0B3D4A"
C_TEAL = "#1B6B93"
C_ACCENT = "#C45C26"
C_MUTED = "#6B7C80"
C_LINE = "#D5DADF"
C_FILL = "#F4F7F8"
C_WAVE = "#3D5A5B"
C_P = "#4C7A5A"
C_S = "#8B5E3C"
BG = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fig-dir", default="docs/figures/pnf")
    p.add_argument("--stem", default="pnf_journal_fig1_principle")
    p.add_argument(
        "--latents-cache",
        default="docs/figures/pnf/seismic_latents_cache.npz",
    )
    return p.parse_args()


def _round_box(ax, xy, w, h, fc, ec, lw=1.0, r=0.018, z=2):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
        mutation_aspect=0.4,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, p0, p1, color=C_INK, lw=1.15, style="-|>", rad=0.0, z=3):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arr)


def _panel_header(ax, letter: str, title: str, y: float = 0.85, title_dx_pt: float = 16.0):
    """Panel letter + title. Title offset in points so gap matches across panel widths."""
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=C_INK,
        va="center",
        ha="left",
        zorder=20,
        clip_on=False,
    )
    ax.annotate(
        title,
        xy=(0.0, y),
        xycoords=ax.transAxes,
        xytext=(title_dx_pt, 0.0),
        textcoords="offset points",
        fontsize=10,
        fontweight="normal",
        color=C_INK,
        va="center",
        ha="left",
        zorder=20,
        annotation_clip=False,
    )


def _kernel_profiles(r, gamma=2.8, omega=9.0, eps=0.08):
    amp = np.exp(-gamma * r**2) / (r + eps)
    re = amp * np.cos(omega * r)
    im = amp * np.sin(omega * r)
    return re, im, amp


def draw_principle(fig, outer_spec):
    """a  Borderless light cone (full-cell, circular wavefronts) + kernel."""
    # Dedicated header band so title never overlaps the cone / kernel
    gs_outer = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_spec,
        height_ratios=[0.22, 1.0],
        hspace=0.12,
    )
    ax_h = fig.add_subplot(gs_outer[0])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    _panel_header(ax_h, "a", "Huygens principle → learnable kernel", y=0.50)

    gs = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_outer[1],
        width_ratios=[1.55, 0.85],
        wspace=0.06,
    )
    ax_c = fig.add_subplot(gs[0, 0])
    ax_k = fig.add_subplot(gs[0, 1])
    for a in (ax_c, ax_k):
        a.set_facecolor(BG)
        a.set_frame_on(False)
        a.patch.set_visible(False)

    # Equal-aspect light cone: |x|=c(t-t0) at true 45°, circular secondary waves.
    # (imshow aspect="auto" was the squeeze source; do not use it here.)
    t0 = 0.06
    y_max = 1.0 - t0  # cone reaches ±y_max at t=1 with c=1
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(-y_max, y_max)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    ax_c.patch.set_visible(False)

    n = 280
    tt = np.linspace(0.0, 1.0, n)
    xx = np.linspace(-y_max, y_max, n)
    T, X = np.meshgrid(tt, xx)
    R = np.sqrt(np.maximum(T - t0, 0.0) ** 2 + X**2) + 1e-6
    amp = np.exp(-2.5 * R**2) / (R + 0.045)
    field = amp * np.cos(15.5 * R)
    inside = (np.abs(X) <= np.maximum(T - t0, 0.0)) & (T > t0)
    field = np.where(inside, field, np.nan)
    vmax = float(np.nanpercentile(np.abs(field), 96))
    cmap = LinearSegmentedColormap.from_list(
        "pnf_cone",
        ["#041E28", "#0E5A78", "#7EB7C8", "#F7FAFB", "#F0A060", "#C45C26", "#7A2208"],
        N=256,
    )
    ax_c.imshow(
        field,
        origin="lower",
        extent=[0.0, 1.0, -y_max, y_max],
        aspect="equal",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
        zorder=1,
        clip_on=True,
    )
    ax_c.fill_between([t0, 1], [0, y_max], y_max * 1.15, color="#FFFFFF", zorder=2, lw=0)
    ax_c.fill_between([t0, 1], [0, -y_max], -y_max * 1.15, color="#FFFFFF", zorder=2, lw=0)
    ax_c.fill_between([-0.05, t0], y_max * 1.15, -y_max * 1.15, color="#FFFFFF", zorder=2, lw=0)
    ax_c.plot([t0, 1], [0, y_max], color=C_ACCENT, lw=2.1, zorder=4)
    ax_c.plot([t0, 1], [0, -y_max], color=C_ACCENT, lw=2.1, zorder=4)
    ax_c.axhline(0, color=C_MUTED, lw=0.75, zorder=3)
    ax_c.set_aspect("equal", adjustable="box", anchor="C")

    ax_c.text(1.03, 0.0, r"$t$", fontsize=8, color=C_MUTED, va="center", ha="left", clip_on=False)
    ax_c.annotate(
        "light cone",
        xy=(0.72, 0.72 - t0),
        xytext=(0.32, y_max * 0.72),
        fontsize=8.5,
        color=C_ACCENT,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=0.9, mutation_scale=8),
        zorder=7,
    )
    ax_c.text(0.90, y_max * 0.70, r"$c$", fontsize=9, color=C_ACCENT, fontweight="bold", ha="center")

    for i, st in enumerate([0.22, 0.40, 0.58, 0.74]):
        ax_c.plot(st, 0, "o", color=C_INK, ms=3.5, zorder=6)
        col = C_TEAL if i % 2 == 0 else C_INK
        for k, rr in enumerate((0.06, 0.11, 0.16)):
            th = np.linspace(12, 168, 55)
            ax_c.plot(
                st + rr * np.cos(np.deg2rad(th)),
                rr * np.sin(np.deg2rad(th)),
                color=col,
                lw=1.05,
                alpha=0.88 - 0.22 * k,
                zorder=5,
            )
    ax_c.plot(t0, 0, "o", color=C_ACCENT, ms=5.5, zorder=6)
    ax_c.annotate(
        "secondary waves",
        xy=(0.40, 0.12),
        xytext=(0.52, 0.42),
        fontsize=8,
        color=C_TEAL,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=0.85, mutation_scale=7),
        zorder=7,
        clip_on=False,
    )

    # ---- kernel: no frame ----
    ax_k.set_xlim(0, 1)
    ax_k.set_ylim(0, 1)
    ax_k.axis("off")
    r = np.linspace(0.0, 1.4, 480)
    re, im, amp1 = _kernel_profiles(r, gamma=1.9, omega=12.0, eps=0.055)
    scale = np.max(np.abs(re)) + 1e-8
    xs = 0.06 + 0.88 * (r / r.max())
    py0, ph = 0.58, 0.28
    ax_k.plot([0.06, 0.94], [py0, py0], color=C_LINE, lw=0.7, zorder=2)
    env = amp1 / scale
    ax_k.fill_between(xs, py0 - ph * env, py0 + ph * env, color="#1B6B9322", lw=0, zorder=2)
    ax_k.plot(xs, py0 + ph * (re / scale), color=C_TEAL, lw=1.9, zorder=4)
    ax_k.plot(xs, py0 + ph * (im / scale), color=C_ACCENT, lw=1.7, zorder=4)
    ax_k.plot([0.08, 0.16], [0.92, 0.92], color=C_TEAL, lw=1.7)
    ax_k.text(0.18, 0.92, r"$\mathrm{Re}$", fontsize=8, color=C_TEAL, va="center")
    ax_k.plot([0.38, 0.46], [0.92, 0.92], color=C_ACCENT, lw=1.7)
    ax_k.text(0.48, 0.92, r"$\mathrm{Im}$", fontsize=8, color=C_ACCENT, va="center")
    ax_k.text(0.94, py0 - 0.05, r"$r$", fontsize=8, color=C_MUTED, ha="right", va="top")

    chips = [(r"$\gamma$", C_TEAL), (r"$\omega$", C_ACCENT), (r"$c$", C_WAVE)]
    for i, (lab, col) in enumerate(chips):
        cx = 0.20 + i * 0.28
        ax_k.text(cx, 0.22, lab, fontsize=11, color=col, ha="center", va="center", fontweight="bold")

    ax_k.text(
        0.50,
        0.08,
        r"$K\!\propto\!\dfrac{e^{-\gamma r^{2}}e^{i\omega r}}{r+\varepsilon}\,\chi$",
        fontsize=9,
        color=C_INK,
        ha="center",
        va="center",
    )


def _draw_huygens_inset(ax, x0, y0, w, h):
    """Gently curved Earth section — flatter arc, surface near top of box."""
    # Large radius → mild curvature; span ~90% of inset width.
    y_surf = y0 + 0.86 * h
    R_surf = 2.4 * w
    cx = x0 + 0.50 * w
    cy = y_surf - R_surf
    half_chord = 0.46 * w
    ang_half = np.rad2deg(np.arcsin(min(half_chord / R_surf, 0.99)))
    a0, a1 = 90.0 - ang_half, 90.0 + ang_half
    th = np.linspace(a0, a1, 220)

    # Layer thicknesses as fractions of inset height (half of previous).
    depth = [0.0, 0.07 * h, 0.15 * h, 0.24 * h, 0.33 * h]
    R_layers = [R_surf - d for d in depth]
    layer_cols = ["#C5D9C8", "#E8D4A8", "#C9A06C", "#7A5A3A"]

    for i, fc in enumerate(layer_cols):
        r_out, r_in = R_layers[i], R_layers[i + 1]
        xo = cx + r_out * np.cos(np.deg2rad(th))
        yo = cy + r_out * np.sin(np.deg2rad(th))
        xi = cx + r_in * np.cos(np.deg2rad(th[::-1]))
        yi = cy + r_in * np.sin(np.deg2rad(th[::-1]))
        ax.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]), color=fc, lw=0, zorder=2)
        ax.plot(xo, yo, color="#5A4636", lw=0.7, alpha=0.7, zorder=3)

    xs = cx + R_surf * np.cos(np.deg2rad(th))
    ys = cy + R_surf * np.sin(np.deg2rad(th))
    ax.plot(xs, ys, color=C_INK, lw=1.45, zorder=4)
    ax.text(xs[12], ys[12] + 0.012 * h, "surface", fontsize=6.0, color=C_MUTED, ha="left", va="bottom", zorder=9)

    # title top-left, shifted slightly down
    ax.text(
        x0 + 0.02 * w,
        y0 + 0.90 * h,
        "Huygens wavefield (schematic)",
        fontsize=6.2,
        color=C_MUTED,
        ha="left",
        va="top",
        zorder=10,
    )

    # source near bottom of 4-layer stack so the cone/waves span all layers
    sang = 90.0
    sR = R_surf - 0.31 * h
    sx = cx + sR * np.cos(np.deg2rad(sang))
    sy = cy + sR * np.sin(np.deg2rad(sang))
    ax.plot(sx, sy, "*", color=C_ACCENT, ms=15, zorder=8, markeredgecolor="#7A2208", markeredgewidth=0.4)
    ax.text(sx - 0.025 * w, sy - 0.03 * h, "source", fontsize=8.5, color=C_ACCENT, ha="right", va="top", fontweight="bold", zorder=9)

    rad = np.deg2rad(sang)
    er = np.array([np.cos(rad), np.sin(rad)])
    et = np.array([-np.sin(rad), np.cos(rad)])

    def local(perp, along):
        return sx + along * er[0] + perp * et[0], sy + along * er[1] + perp * et[1]

    L = R_surf - sR  # distance source → surface (~full crust thickness)

    # Primary wavefronts: concentric arcs, uniform spacing, spanning full crust
    wave_fracs = (0.22, 0.44, 0.66, 0.88)
    wave_half = 52.0
    for k, s in enumerate(wave_fracs):
        phi = np.linspace(-wave_half, wave_half, 72)
        pts = [local(s * L * np.sin(np.deg2rad(p)), s * L * np.cos(np.deg2rad(p))) for p in phi]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=C_TEAL,
            lw=1.15,
            alpha=0.92 - 0.12 * k,
            zorder=5,
            solid_capstyle="round",
        )

    # Short light cone (~1/3 of previous full-crust length)
    cone_len = 0.32 * L
    half = 34.0
    phi_cone = np.linspace(-half, half, 40)
    arc = [local(cone_len * np.tan(np.deg2rad(p)), cone_len) for p in phi_cone]
    ax.fill(
        [sx] + [p[0] for p in arc] + [sx],
        [sy] + [p[1] for p in arc] + [sy],
        color="#C45C2648",
        zorder=4,
        lw=0,
    )
    left = local(cone_len * np.tan(np.deg2rad(-half)), cone_len)
    right = local(cone_len * np.tan(np.deg2rad(half)), cone_len)
    ax.plot([sx, left[0]], [sy, left[1]], color=C_ACCENT, lw=1.55, zorder=6, solid_capstyle="round")
    ax.plot([sx, right[0]], [sy, right[1]], color=C_ACCENT, lw=1.55, zorder=6, solid_capstyle="round")

    ax.annotate(
        "light cone",
        xy=local(0.10 * L, 0.18 * L),
        xytext=local(0.42 * L, 0.08 * L),
        fontsize=7.5,
        fontweight="bold",
        color=C_ACCENT,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=1.05, mutation_scale=8),
        zorder=9,
    )

    # Secondary wavelets along the full source→probe center line (sparse, even spacing)
    n_sec = 6
    sec_fracs = np.linspace(0.12, 0.88, n_sec)
    spacing = (sec_fracs[1] - sec_fracs[0]) * L
    r_s = 0.32 * spacing
    for frac in sec_fracs:
        bx, by = local(0.0, float(frac) * L)
        ax.plot(bx, by, "o", color=C_INK, ms=2.2, zorder=7)
        cdir = er.copy()  # outward along center ray toward probe
        cperp = et.copy()
        phi2 = np.linspace(-85.0, 85.0, 40)
        wx = [bx + r_s * (np.cos(np.deg2rad(q)) * cdir[0] + np.sin(np.deg2rad(q)) * cperp[0]) for q in phi2]
        wy = [by + r_s * (np.cos(np.deg2rad(q)) * cdir[1] + np.sin(np.deg2rad(q)) * cperp[1]) for q in phi2]
        ax.plot(wx, wy, color=C_TEAL, lw=0.95, alpha=0.92, zorder=6, solid_capstyle="round")
    ax.annotate(
        "secondary waves",
        xy=local(0.0, 0.58 * L),
        xytext=local(0.42 * L, 0.52 * L),
        fontsize=6.5,
        color=C_TEAL,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=0.9, mutation_scale=7),
        zorder=9,
    )

    def hit_surface(perp):
        direction = np.array(local(perp, 1.15 * L)) - np.array([sx, sy])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        for t in np.linspace(0.05 * L, 1.6 * L, 140):
            px = sx + t * direction[0]
            py = sy + t * direction[1]
            if (px - cx) ** 2 + (py - cy) ** 2 >= (R_surf * 0.998) ** 2:
                ang = np.arctan2(py - cy, px - cx)
                return cx + R_surf * np.cos(ang), cy + R_surf * np.sin(ang)
        return None

    # Probes stay widely spaced on surface (independent of short cone)
    probe_span = 0.42 * L
    probe_perps = (-probe_span, 0.0, probe_span)
    for j, ps in enumerate(probe_perps):
        pt = hit_surface(ps)
        if pt is None:
            continue
        ax.plot(pt[0], pt[1], "v", color=C_WAVE, ms=6.5, zorder=8)
        # solid segment only within short cone, then dotted to probe
        ray = np.array(pt) - np.array([sx, sy])
        ray = ray / (np.linalg.norm(ray) + 1e-9)
        mid = (sx + cone_len * ray[0], sy + cone_len * ray[1])
        ax.plot([sx, mid[0]], [sy, mid[1]], color=C_WAVE, lw=0.85, alpha=0.50, zorder=4)
        ax.plot([mid[0], pt[0]], [mid[1], pt[1]], color=C_WAVE, lw=0.75, ls=":", alpha=0.75, zorder=4)
        if j == 1:
            ax.text(pt[0], pt[1] + 0.018 * h, "probe", fontsize=6.2, color=C_WAVE, ha="center", va="bottom", zorder=9)


def draw_architecture(fig, outer_spec):
    """b  STEAD forward path (fully visible) + tall Huygens inset in lower blank."""
    gs_outer = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_spec,
        height_ratios=[0.18, 1.0],
        hspace=0.08,
    )
    ax_h = fig.add_subplot(gs_outer[0])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    _panel_header(ax_h, "b", "Forward path: STEAD 3C picking", y=0.50)

    ax = fig.add_subplot(gs_outer[1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- forward path: original positions, fully inside [0, 1] ---
    for i, (lab, col) in enumerate([("E", C_TEAL), ("N", "#2E86AB"), ("Z", C_MUTED)]):
        y = 0.92 - i * 0.070
        tt = np.linspace(0, 1, 60)
        ww = np.exp(-((tt - 0.45) ** 2) / 0.02) * np.sin(2 * np.pi * (10 + i) * tt)
        ax.plot(0.04 + 0.13 * tt, y + 0.022 * ww, color=col, lw=0.9)
        ax.text(0.005, y, lab, fontsize=7, color=col, va="center", ha="left", fontweight="bold")
    ax.text(0.10, 0.975, "3C", fontsize=7, color=C_MUTED, ha="center")

    _arrow(ax, (0.19, 0.85), (0.24, 0.85), color=C_LINE, lw=1.1)

    _round_box(ax, (0.24, 0.84), 0.145, 0.13, "#EAF3F5", C_TEAL, lw=1.0, r=0.01)
    ax.text(0.312, 0.935, "embed", fontsize=6.5, color=C_TEAL, ha="center")
    ax.text(0.312, 0.88, r"$h$", fontsize=9.5, color=C_INK, ha="center", fontweight="bold")

    _round_box(ax, (0.24, 0.68), 0.145, 0.13, "#F7F3EE", C_WAVE, lw=1.0, r=0.01)
    ax.text(0.312, 0.775, "probe", fontsize=7, color=C_MUTED, ha="center", va="center", style="italic")
    ax.text(0.312, 0.72, r"$\rho(t)$", fontsize=9, color=C_WAVE, ha="center", va="center")

    _arrow(ax, (0.385, 0.905), (0.44, 0.86), color=C_LINE, lw=1.0)
    _arrow(ax, (0.385, 0.745), (0.44, 0.78), color=C_LINE, lw=1.0)

    _round_box(ax, (0.44, 0.70), 0.22, 0.26, "#FFFFFF", C_ACCENT, lw=1.15, r=0.01)
    ax.text(0.55, 0.91, "Huygens", fontsize=8, color=C_ACCENT, ha="center", fontweight="bold")
    ax.text(0.55, 0.825, r"fine $\omega\!\uparrow$", fontsize=6.5, color=C_TEAL, ha="center")
    ax.text(0.55, 0.755, r"coarse $\omega\!\downarrow$", fontsize=6.5, color=C_MUTED, ha="center")

    # Fan-out: one arrow from Huygens to each of det / P / S
    outs = [("det", C_MUTED, 0.93), ("P", C_TEAL, 0.78), ("S", C_ACCENT, 0.63)]
    hx = 0.66
    hy = 0.83  # mid-right of Huygens box
    for lab, col, y in outs:
        _round_box(ax, (0.74, y - 0.055), 0.22, 0.11, "#FFFFFF", col, lw=0.95, r=0.008)
        tt = np.linspace(0, 1, 36)
        pk = np.exp(-((tt - 0.55) ** 2) / 0.02)
        ax.plot(0.76 + 0.12 * tt, y + 0.028 * pk - 0.01, color=col, lw=1.0)
        ax.text(0.92, y, lab, fontsize=7.0, color=col, ha="left", va="center", fontweight="bold")
        # slight curve so the three arrows separate cleanly
        rad = 0.18 if y > hy else (-0.18 if y < hy else 0.0)
        _arrow(ax, (hx, hy), (0.74, y), color=col, lw=1.05, rad=rad)

    # Shift inset slightly up and right.
    fig.canvas.draw()
    bbox = ax.get_position()
    inset_w = bbox.width * 0.72
    inset_x = bbox.x0 + bbox.width * 0.16
    inset_top = bbox.y0 + bbox.height * 0.68
    inset_bot = bbox.y0 - bbox.height * 0.79
    ax_in = fig.add_axes([inset_x, inset_bot, inset_w, inset_top - inset_bot], zorder=2)
    ax_in.set_facecolor(BG)
    ax_in.set_xlim(0, 1)
    ax_in.set_ylim(0, 1)
    ax_in.axis("off")
    ax_in.set_navigate(False)
    _draw_huygens_inset(ax_in, x0=0.0, y0=0.0, w=1.0, h=1.0)


def _mark_ps(ax, t, p_true, s_true):
    ax.axvline(t[p_true], color=C_P, ls="--", lw=0.9, alpha=0.85, zorder=2)
    ax.axvline(t[s_true], color=C_S, ls="--", lw=0.9, alpha=0.85, zorder=2)


def draw_latents(fig, outer_spec, cache_path: Path):
    """c  Native 4-row latent panel; content kept inside figure width."""
    gs = GridSpecFromSubplotSpec(
        5,
        1,
        subplot_spec=outer_spec,
        height_ratios=[0.32, 1.15, 0.90, 0.95, 0.95],
        hspace=0.52,
    )

    ax_h = fig.add_subplot(gs[0])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    _panel_header(ax_h, "c", "Interpretable latent variables in the seismic domain", y=0.50)

    data = np.load(cache_path)
    t = data["t_sec"]
    x = data["x"]
    rho = data["rho"]
    energy = data["wave_energy"]
    kernel = data["kernel"]
    p_prob = data["p_prob"]
    s_prob = data["s_prob"]
    p_true = int(data["p_true"])
    s_true = int(data["s_true"])
    p_pred = int(data["p_pred"])
    s_pred = int(data["s_pred"])

    # tighter window → less empty left space / less “over-wide” feel
    t0 = max(0.0, float(t[p_true]) - 5.0)
    t1 = min(float(t[-1]), float(t[s_true]) + 8.0)

    xd = x - x.mean(axis=0, keepdims=True)
    xd = xd / (np.max(np.abs(xd)) + 1e-8)

    ax0 = fig.add_subplot(gs[1])
    ax1 = fig.add_subplot(gs[2], sharex=ax0)
    ax2 = fig.add_subplot(gs[3], sharex=ax0)
    ax3 = fig.add_subplot(gs[4], sharex=ax0)
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_clip_on(True)

    colors = [C_TEAL, "#2E86AB", C_MUTED]
    names = ["E", "N", "Z"]
    offsets = [1.15, 0.0, -1.15]
    for i in range(3):
        ax0.plot(t, xd[:, i] + offsets[i], color=colors[i], lw=0.7, label=names[i])
    _mark_ps(ax0, t, p_true, s_true)
    ax0.text(t[p_true] + 0.25, 2.0, "P", color=C_P, fontsize=8, fontweight="bold")
    ax0.text(t[s_true] + 0.25, 2.0, "S", color=C_S, fontsize=8, fontweight="bold")
    ax0.set_ylabel("Amp.", fontsize=7.5, labelpad=2, color=C_INK)
    ax0.set_title("3C waveform with P / S arrivals", loc="left", fontsize=8.5, pad=2, color=C_INK)
    ax0.set_yticks(offsets)
    ax0.set_yticklabels(names, fontsize=7)
    ax0.set_ylim(-2.35, 2.4)
    ax0.legend(loc="upper right", ncol=3, fontsize=6.5, frameon=False, handlelength=1.0)

    en = energy / (energy.max() + 1e-8)
    rn = rho / (rho.max() + 1e-8)
    ax1.fill_between(t, 0, en, color="#C8C3B8", alpha=0.45, lw=0, label="energy")
    ax1.plot(t, rn, color="#8B4513", lw=1.2, label=r"$\rho(t)$")
    _mark_ps(ax1, t, p_true, s_true)
    ax1.set_ylabel(r"$\rho$", fontsize=7.5, labelpad=2, color=C_INK)
    ax1.set_title(r"Learned density $\rho(t)$ tracks waveform energy", loc="left", fontsize=8.5, pad=2, color=C_INK)
    ax1.set_ylim(-0.02, 1.15)
    ax1.legend(loc="upper right", ncol=2, fontsize=6.5, frameon=False)

    k = np.abs(kernel)
    k = k / (k.max() + 1e-12)
    ax2.fill_between(t, 0, k, color="#5E3C99", alpha=0.18, lw=0)
    ax2.plot(t, k, color="#5E3C99", lw=1.15, label=r"$|K[P,:]|$")
    _mark_ps(ax2, t, p_true, s_true)
    # light pre-onset shading only (no text annotation)
    ax2.axvspan(t0, t[p_true], color="#5E3C99", alpha=0.05, zorder=0, clip_on=True)
    ax2.set_ylabel(r"$|K|$", fontsize=7.5, labelpad=2, color=C_INK)
    ax2.set_title("Kernel row at ground-truth P index", loc="left", fontsize=8.5, pad=2, color=C_INK)
    ax2.set_ylim(-0.02, 1.15)
    ax2.legend(loc="upper right", fontsize=6.5, frameon=False)

    ax3.plot(t, p_prob, color="#0072B2", lw=1.2, label="P pick")
    ax3.plot(t, s_prob, color="#D55E00", lw=1.2, label="S pick")
    _mark_ps(ax3, t, p_true, s_true)
    ax3.scatter([t[p_pred]], [p_prob[p_pred]], s=18, color="#0072B2", zorder=4)
    ax3.scatter([t[s_pred]], [s_prob[s_pred]], s=18, color="#D55E00", zorder=4)
    ax3.set_ylabel("Prob.", fontsize=7.5, labelpad=2, color=C_INK)
    ax3.set_xlabel("Time (s)", fontsize=8, color=C_INK)
    ax3.set_title("P and S pick probability", loc="left", fontsize=8.5, pad=2, color=C_INK)
    ax3.set_ylim(-0.02, 1.12)
    ax3.legend(loc="upper right", ncol=2, fontsize=6.5, frameon=False)

    ax0.set_xlim(t0, t1)
    for ax in (ax0, ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)


def draw_paradigm(ax):
    """d  Three named rows; tall workflow row so markers stay circular."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Leave clear gap below panel title before row-1 heading
    _panel_header(ax, "d", "PNF: causal physics kernel for new discoveries", y=0.97)

    # ========== Row 1: Causal integral operator (compact) ==========
    ax.text(0.02, 0.915, "Causal integral operator", fontsize=8.5, color=C_TEAL, ha="left", va="center")
    _round_box(ax, (0.02, 0.825), 0.96, 0.07, "#EAF3F5", C_TEAL, lw=1.1, r=0.012)
    tri = Polygon(
        [(0.06, 0.837), (0.13, 0.837), (0.13, 0.880)],
        closed=True,
        facecolor="#1B6B9344",
        edgecolor=C_TEAL,
        lw=1.0,
        zorder=4,
    )
    ax.add_patch(tri)
    ax.text(
        0.56,
        0.860,
        r"$u=\mathcal{K}_{\theta}[s]=\int_{\tau\leq t} K_{\theta}(t-\tau)\,s(\tau)\,d\tau$"
        r"   ·   light-cone support   ·   $\theta$ readable",
        fontsize=8.0,
        color=C_INK,
        ha="center",
        va="center",
    )

    # ========== Row 2: Physics kernel family (slightly shorter) ==========
    ax.text(0.02, 0.795, "Physics kernel family", fontsize=8.5, color=C_INK, ha="left", va="center")

    _round_box(ax, (0.02, 0.575), 0.31, 0.19, "#FFFFFF", C_TEAL, lw=1.1, r=0.012)
    ax.text(0.175, 0.735, "Wave · seismic", fontsize=7.5, color=C_TEAL, ha="center")
    ax.text(0.175, 0.700, r"$K_{\gamma,\omega,c}$", fontsize=8, color=C_INK, ha="center")
    rr = np.linspace(0, 1, 90)
    yy = np.exp(-2.8 * rr**2) * np.cos(14 * rr)
    yy /= np.max(np.abs(yy)) + 1e-8
    ax.plot(0.05 + 0.25 * rr, 0.650 + 0.038 * yy, color=C_TEAL, lw=1.15)
    t3 = np.linspace(0, 1, 70)
    w3 = np.exp(-((t3 - 0.34) ** 2) / 0.0035) * np.sin(2 * np.pi * 16 * t3)
    ax.plot(0.05 + 0.25 * t3, 0.600 + 0.020 * w3, color=C_INK, lw=0.85)

    _round_box(ax, (0.345, 0.575), 0.31, 0.19, "#FFFFFF", C_WAVE, lw=1.1, r=0.012)
    ax.text(0.50, 0.735, "Diffusion · EEG", fontsize=7.5, color=C_WAVE, ha="center")
    ax.text(0.50, 0.700, r"$K_{D_{\parallel},D_{\perp},\alpha}$", fontsize=7.5, color=C_INK, ha="center")
    ax.add_patch(
        Ellipse((0.44, 0.640), 0.12, 0.048, angle=35, fc="#3D5A5B22", ec=C_WAVE, lw=1.15, zorder=3)
    )
    for i, col in enumerate([C_TEAL, C_WAVE, C_ACCENT]):
        te = np.linspace(0, 1, 80)
        we = 0.7 * np.sin(2 * np.pi * (9 + 0.7 * i) * te) * (0.65 + 0.35 * np.sin(np.pi * te))
        ax.plot(0.545 + 0.09 * te, 0.640 - i * 0.020 + 0.011 * we, color=col, lw=0.8)

    _round_box(ax, (0.67, 0.575), 0.31, 0.19, "#FFFFFF", C_ACCENT, lw=1.1, r=0.012)
    ax.text(0.825, 0.735, "Memory · rheology", fontsize=7.5, color=C_ACCENT, ha="center")
    ax.text(0.825, 0.700, r"$K=\sum_k G_k e^{-t/\lambda_k}$", fontsize=7.0, color=C_INK, ha="center")
    for lam, G in zip([0.15, 0.35, 0.55, 0.75], [0.55, 0.9, 0.45, 0.7]):
        x = 0.70 + 0.12 * lam
        ax.plot([x, x], [0.600, 0.600 + 0.07 * G], color=C_ACCENT, lw=1.6, solid_capstyle="round")
        ax.plot(x, 0.600 + 0.07 * G, "o", color=C_ACCENT, ms=3.0)
    w = np.linspace(0.05, 1.0, 60)
    gp = 0.3 + 0.55 / (1 + (0.45 / w) ** 2)
    gpp = 0.55 * (0.45 * w) / (1 + (0.45 * w) ** 2) + 0.15
    ax.plot(0.845 + 0.11 * w, 0.610 + 0.07 * gp, color=C_TEAL, lw=1.0)
    ax.plot(0.845 + 0.11 * w, 0.610 + 0.07 * gpp, color=C_ACCENT, lw=1.0)

    # ========== Row 3: Workflow (taller — circular markers) ==========
    ax.text(0.02, 0.540, "Workflow for discoveries", fontsize=8.5, color=C_ACCENT, ha="left", va="center")

    # Learn
    _round_box(ax, (0.02, 0.02), 0.30, 0.49, "#EAF3F5", C_TEAL, lw=1.25, r=0.012)
    ax.text(0.17, 0.46, "Learn", fontsize=9.5, color=C_TEAL, ha="center", fontweight="bold")
    for i, col in enumerate([C_TEAL, C_WAVE, C_ACCENT]):
        tt = np.linspace(0, 1, 50)
        ww = np.exp(-((tt - 0.4) ** 2) / 0.01) * np.sin(2 * np.pi * (12 - i) * tt)
        ax.plot(0.05 + 0.12 * tt, 0.38 - i * 0.036 + 0.015 * ww, color=col, lw=0.75)
    for i, (lab, col) in enumerate([("det", C_MUTED), ("P", C_TEAL), ("S", C_ACCENT)]):
        x = 0.20 + i * 0.04
        ax.plot(x, 0.335, "o", color=col, ms=9, zorder=4, clip_on=False)
        ax.text(x, 0.290, lab, fontsize=6, color=col, ha="center")
    ep = np.linspace(0, 1, 50)
    loss = np.exp(-3.0 * ep) + 0.05
    ax.fill_between(0.05 + 0.22 * ep, 0.07, 0.07 + 0.15 * loss, color="#1B6B9333", lw=0)
    ax.plot(0.05 + 0.22 * ep, 0.07 + 0.15 * loss, color=C_TEAL, lw=1.55)
    ax.annotate(
        r"$\theta$",
        xy=(0.24, 0.11),
        xytext=(0.27, 0.20),
        fontsize=9,
        color=C_TEAL,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=0.9, mutation_scale=7),
    )

    _arrow(ax, (0.32, 0.25), (0.345, 0.25), color=C_LINE, lw=1.3)

    # Probe
    _round_box(ax, (0.345, 0.02), 0.31, 0.49, "#F7F3EE", C_WAVE, lw=1.25, r=0.012)
    ax.text(0.50, 0.46, "Probe", fontsize=9.5, color=C_WAVE, ha="center", fontweight="bold")
    tw = np.linspace(0, 1, 120)
    ww = np.exp(-((tw - 0.30) ** 2) / 0.0025) * np.sin(2 * np.pi * 18 * tw)
    ww += 0.75 * np.exp(-((tw - 0.58) ** 2) / 0.005) * np.sin(2 * np.pi * 10 * tw)
    ax.plot(0.37 + 0.26 * tw, 0.30 + 0.060 * ww, color=C_INK, lw=1.05, zorder=3)
    for tp, lab, col in [(0.30, "P", C_TEAL), (0.58, "S", C_ACCENT)]:
        x = 0.37 + 0.26 * tp
        ax.plot([x, x], [0.18, 0.39], color=col, lw=2.1, zorder=5, solid_capstyle="round")
        ax.plot(x, 0.39, "v", color=col, ms=10, zorder=6, clip_on=False)
        ax.plot(x, 0.39, "o", color=col, ms=15, alpha=0.18, zorder=4, clip_on=False)
        ax.text(x, 0.425, lab, fontsize=7.5, color=col, ha="center", fontweight="bold")
    orbs = [(0.40, r"$\gamma$", C_TEAL), (0.47, r"$\omega$", C_ACCENT), (0.54, r"$c$", C_WAVE), (0.61, r"$\rho$", C_INK)]
    for i, (ox, lab, col) in enumerate(orbs):
        src = 0.37 + 0.26 * (0.30 if i < 2 else 0.58)
        ax.plot([src, ox], [0.18, 0.135], color=col, lw=0.75, alpha=0.55, zorder=2)
        ax.plot(ox, 0.100, "o", color="#FFFFFF", ms=18, markeredgecolor=col, markeredgewidth=1.5, zorder=5, clip_on=False)
        ax.text(ox, 0.100, lab, fontsize=8, color=col, ha="center", va="center", zorder=6)

    _arrow(ax, (0.655, 0.25), (0.68, 0.25), color=C_LINE, lw=1.3)

    # Discover — evenly spaced output cards (no overlap)
    _round_box(ax, (0.68, 0.02), 0.30, 0.49, "#FBF0E8", C_ACCENT, lw=1.25, r=0.012)
    ax.text(0.83, 0.46, "Discover", fontsize=9.5, color=C_ACCENT, ha="center", fontweight="bold")
    ax.add_patch(
        Polygon(
            [(0.74, 0.36), (0.92, 0.36), (0.875, 0.245), (0.785, 0.245)],
            closed=True,
            facecolor="#C45C2633",
            edgecolor=C_ACCENT,
            lw=1.1,
            zorder=3,
        )
    )
    ax.text(0.83, 0.30, "Physics\nDecoder", fontsize=7, color=C_ACCENT, ha="center", va="center", zorder=4)
    card_w = 0.055
    card_left, card_right = 0.695, 0.965
    gap = (card_right - card_left - 4 * card_w) / 3.0
    cards = [
        (r"$v_P$", C_TEAL),
        (r"$M_w$", C_ACCENT),
        (r"$\lambda_k$", C_WAVE),
        (r"$\alpha$", C_INK),
    ]
    for i, (lab, col) in enumerate(cards):
        x = card_left + i * (card_w + gap)
        ax.plot([0.83, x + 0.5 * card_w], [0.245, 0.17], color=col, lw=0.85, alpha=0.7, zorder=2)
        _round_box(ax, (x, 0.055), card_w, 0.095, "#FFFFFF", col, lw=1.15, r=0.012, z=5)
        ax.text(x + 0.5 * card_w, 0.102, lab, fontsize=7.5, color=col, ha="center", va="center", zorder=6)


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.latents_cache)
    if not cache.is_file():
        raise SystemExit(f"missing latents cache: {cache}")

    fig = plt.figure(figsize=(11.2, 12.6), facecolor=BG)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.55, 1.15, 1.55],  # taller a|b for taller Huygens inset
        width_ratios=[0.85, 1.35],
        hspace=0.10,
        wspace=0.09,
        left=0.08,
        right=0.97,
        top=0.935,
        bottom=0.025,
    )

    draw_principle(fig, gs[0, 0])
    draw_architecture(fig, gs[0, 1])
    draw_latents(fig, gs[1, :], cache)

    ax_d = fig.add_subplot(gs[2, :])
    ax_d.set_facecolor(BG)
    draw_paradigm(ax_d)

    fig.suptitle(
        "Physics Neural Field (PNF): principle, latents, and discovery paradigm",
        fontsize=12,
        fontweight="bold",
        color=C_INK,
        y=0.975,
    )

    out = fig_dir / args.stem
    # no bbox_inches='tight' — prevents panel c from expanding past figure width
    fig.savefig(f"{out}.png", facecolor=BG, pad_inches=0.04)
    fig.savefig(f"{out}.pdf", facecolor=BG, pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {out}.png / {out}.pdf")


if __name__ == "__main__":
    main()
