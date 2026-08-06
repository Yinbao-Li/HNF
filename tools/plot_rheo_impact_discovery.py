#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Impact-discovery plate for rheology spectrum ↔ MWD claim.

Style locked to EEG impact plate: 3D boom pies, bold panel letters,
EEG-like method boxes/fonts. Model name is PNF throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hnf.rheo_elliott_compat import MWD_X, normalize_mwd_on_x
from hnf.rheo_gpc import load_leeds_gpc_all
from hnf.rheo_leeds import load_leeds_saos_all

BG = "#FFFFFF"
C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
C_TEAL = "#0F766E"
C_GOLD = "#B45309"
C_LIN = "#2c6e8a"
C_BR = "#c45c26"
C_NULL = "#9CA3AF"
C_LOO = "#6b4c9a"
C_UMN = "#B45309"
C_PW = "#4B5563"

OUT = _REPO / "docs/figures/rheo"
HARDEN = _REPO / "outputs/rheo/spectrum_mwd_harden/HARDEN.json"
DISCOVERY = _REPO / "outputs/rheo/spectrum_mwd_mine/DISCOVERY.json"

BRANCHED = {"A1PS", "PSA"}


def _style() -> None:
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
            "axes.facecolor": BG,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "text.color": C_INK,
        }
    )


def _panel_header(ax, letter: str, title: str, y: float = 1.02, title_dx_pt: float = 16.0) -> None:
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight=700,
        fontfamily="DejaVu Sans",
        color="#000000",
        va="bottom",
        ha="left",
        zorder=30,
        clip_on=False,
    )
    ax.annotate(
        title,
        xy=(0.0, y),
        xycoords=ax.transAxes,
        xytext=(title_dx_pt, 0.0),
        textcoords="offset points",
        fontsize=9,
        color=C_INK,
        va="bottom",
        ha="left",
        zorder=30,
        clip_on=False,
    )


def _clean(ax) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, colors=C_INK)


def _box(ax, x, y, w, h, fc, ec="#D1D5DB", lw=0.85, r=0.08, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={r}",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=z,
            clip_on=False,
        )
    )


def _boom_pie_3d(fig, host_ax, labels, sizes, colors, total_label: str) -> None:
    host_ax.set_xlim(0, 1)
    host_ax.set_ylim(0, 1)
    host_ax.axis("off")
    host_ax.set_facecolor(BG)

    sizes = np.asarray(sizes, dtype=float)
    total = float(sizes.sum())
    fracs = sizes / max(total, 1.0)

    fig.canvas.draw()
    bbox = host_ax.get_position()
    leg_h = 0.18 * bbox.height
    bot_h = 0.08 * bbox.height
    avail_h = bbox.height - leg_h - bot_h
    side = min(1.08 * avail_h, 1.40 * bbox.width)
    x0 = bbox.x0 + 0.5 * (bbox.width - side)
    y0 = bbox.y0 + bot_h * 0.35
    ax3d = fig.add_axes([x0, y0, side, side], projection="3d", facecolor=BG)
    ax3d.set_zorder(3)
    ax3d.set_axis_off()
    ax3d.set_box_aspect((1.0, 1.0, 0.42))
    ax3d.view_init(elev=28, azim=-50)
    ax3d.set_xlim(-1.55, 1.55)
    ax3d.set_ylim(-1.55, 1.55)
    ax3d.set_zlim(0, 0.55)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BG)
    ax3d.grid(False)

    height = 0.34
    # Match EEG boom look with clear radial separation
    explode0 = 0.30
    theta0 = 0.5 * np.pi
    for frac, col in zip(fracs, colors):
        dtheta = 2.0 * np.pi * float(frac)
        nseg = max(18, int(72 * frac) + 2)
        thetas = np.linspace(theta0, theta0 - dtheta, nseg)
        theta_mid = theta0 - 0.5 * dtheta
        explode = explode0 + (0.14 if frac < 0.25 else 0.0)
        ox = explode * np.cos(theta_mid)
        oy = explode * np.sin(theta_mid)
        rgb = np.array(to_rgb(col))
        top_rgb = np.clip(rgb * 1.08 + 0.05, 0, 1)
        side_rgb = np.clip(rgb * 0.68, 0, 1)
        xt = ox + np.concatenate([[0.0], np.cos(thetas), [0.0]])
        yt = oy + np.concatenate([[0.0], np.sin(thetas), [0.0]])
        zt = np.full_like(xt, height)
        ax3d.add_collection3d(
            Poly3DCollection(
                [list(zip(xt, yt, zt))],
                facecolors=[top_rgb],
                edgecolors="white",
                linewidths=0.4,
                alpha=1.0,
            )
        )
        for i in range(len(thetas) - 1):
            xa, ya = ox + np.cos(thetas[i]), oy + np.sin(thetas[i])
            xb, yb = ox + np.cos(thetas[i + 1]), oy + np.sin(thetas[i + 1])
            verts = [(xa, ya, 0.0), (xb, yb, 0.0), (xb, yb, height), (xa, ya, height)]
            ax3d.add_collection3d(
                Poly3DCollection(
                    [verts],
                    facecolors=[side_rgb],
                    edgecolors=[side_rgb],
                    linewidths=0.03,
                    alpha=1.0,
                )
            )
        for th in (thetas[0], thetas[-1]):
            x, y = ox + np.cos(th), oy + np.sin(th)
            verts = [(ox, oy, 0), (x, y, 0), (x, y, height), (ox, oy, height)]
            ax3d.add_collection3d(
                Poly3DCollection(
                    [verts],
                    facecolors=[np.clip(side_rgb * 0.88, 0, 1)],
                    edgecolors="none",
                    alpha=1.0,
                )
            )
        theta0 -= dtheta

    ax_leg = fig.add_axes([bbox.x0, bbox.y0 + bbox.height - leg_h, bbox.width, leg_h], facecolor=BG)
    ax_leg.set_zorder(6)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    n = len(labels)
    xs = np.linspace(0.14, 0.86, n) if n > 1 else [0.5]
    for i, lab in enumerate(labels):
        lx = float(xs[i])
        ax_leg.plot(lx, 0.20, "o", color=colors[i], markersize=7.5, clip_on=False)
        ax_leg.text(lx, 0.72, lab, fontsize=7.4, color=C_INK, va="center", ha="center", fontweight="bold")
        ax_leg.text(
            lx,
            0.42,
            f"{int(sizes[i])} ({100.0 * fracs[i]:.0f}%)",
            fontsize=5.8,
            color=C_MUTED,
            va="center",
            ha="center",
        )

    ax_bot = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bot_h], facecolor=BG)
    ax_bot.set_zorder(6)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis("off")
    ax_bot.text(0.5, 0.40, total_label, ha="center", va="center", fontsize=7.6, color=C_INK, fontweight="bold")


def _maxwell_gp_gpp(omega, lam, g, g_inf=0.0):
    omega = np.asarray(omega, dtype=float)
    lam = np.asarray(lam, dtype=float)
    g = np.asarray(g, dtype=float)
    x = omega[:, None] * lam[None, :]
    x2 = x * x
    den = 1.0 + x2
    gp = g_inf + np.sum(g[None, :] * x2 / den, axis=1)
    gpp = np.sum(g[None, :] * x / den, axis=1)
    return gp, gpp


def panel_method(ax) -> None:
    """EEG-matched method panel: same coords, box geometry, 1.3× fonts.
    Header is drawn by the shared top row (aligned with panel a).
    """
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.05)
    ax.axis("off")

    # Top pipeline — equal height, clear gaps (mirror EEG)
    top_y, top_h = 1.55, 1.25
    boxes = [
        (0.15, top_y, 2.00, top_h, "#F3F4F6", "SAOS\n$\\omega$, $G'$, $G''$"),
        (2.55, top_y, 2.05, top_h, "#ECFDF5", "PNF\nfixed $\\tau$ library\n$\\rightarrow G_k$"),
        (5.00, top_y, 2.20, top_h, "#EFF6FF", "Tube map\n$M\\propto\\lambda^{1/\\alpha}$\nmode$\\leftrightarrow$GPC"),
        (7.70, top_y, 4.00, top_h, "#F5F3FF", ""),
    ]
    for x, y, w, h, fc, txt in boxes:
        _box(ax, x, y, w, h, fc)
        if txt:
            ax.text(
                x + w / 2,
                y + h / 2,
                txt,
                ha="center",
                va="center",
                fontsize=7.2 * 1.3,
                color=C_INK,
                linespacing=1.15,
            )

    # mini moduli sketch in first box (EEG places a wave here)
    t = np.linspace(0, 1, 80)
    ax.plot(0.35 + t * 1.55, 1.78 + 0.22 * (0.6 + 0.9 * t), color=C_LIN, lw=0.9, zorder=4)
    ax.plot(0.35 + t * 1.55, 2.15 - 0.12 * t, color=C_MUTED, lw=0.85, ls="--", zorder=4)

    ax.text(9.70, 2.50, "Discovery test", ha="center", fontsize=7.6 * 1.3, fontweight="bold", color="#6D28D9")
    ax.text(
        9.70,
        1.95,
        r"corr$(G_k/\sum G,\;\mathrm{GPC})$"
        "\n"
        r"vs shuffle-$G_k$ null",
        ha="center",
        va="center",
        fontsize=7.0 * 1.3,
        color=C_INK,
        linespacing=1.15,
    )

    mid = top_y + 0.5 * top_h
    for x0, x1 in [(2.15, 2.55), (4.60, 5.00), (7.20, 7.70)]:
        ax.annotate("", xy=(x1, mid), xytext=(x0, mid), arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.05))

    gap = 0.40
    cw = (2.55, 2.55, 2.70)
    total = sum(cw) + 2 * gap
    x0 = 0.5 * (12.0 - total)
    ctrls = [
        (x0, 0.10, cw[0], 0.95, "#D1FAE5", "shuffle null\n(within sample)"),
        (x0 + cw[0] + gap, 0.10, cw[1], 0.95, "#FEE2E2", r"$\alpha$ & free-$\lambda$" "\nrobustness"),
        (x0 + cw[0] + gap + cw[1] + gap, 0.10, cw[2], 0.95, "#DBEAFE", "Elliott NN\n(for compare)"),
    ]
    for x, y, w, h, fc, txt in ctrls:
        _box(ax, x, y, w, h, fc, r=0.06)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.0 * 1.3, color=C_INK, linespacing=1.15)


def _draw_topo_inset(ax, branched: bool, color: str) -> None:
    """Polymer topology cartoon in axes fraction (white chip, clear stroke)."""
    # chip background
    # lower inset so it sits mid-right, clear of titles and legend
    y0 = 0.40
    ax.add_patch(
        FancyBboxPatch(
            (0.72, y0),
            0.26,
            0.26,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            transform=ax.transAxes,
            fc=(1.0, 1.0, 1.0, 0.35),
            ec="#D1D5DB",
            lw=0.7,
            zorder=8,
            clip_on=False,
        )
    )
    if not branched:
        xs = np.linspace(0.76, 0.94, 7)
        ys = y0 + 0.13 + 0.04 * np.sin(np.linspace(0, 2.2 * np.pi, 7))
        ax.plot(xs, ys, "-", color=color, lw=2.0, solid_capstyle="round", transform=ax.transAxes, zorder=9, clip_on=False)
        ax.plot(xs, ys, "o", color=color, ms=3.0, mew=0, transform=ax.transAxes, zorder=10, clip_on=False)
    else:
        cx, cy = 0.85, y0 + 0.12
        arms = [
            (0.76, y0 + 0.22),
            (0.94, y0 + 0.22),
            (0.75, y0 + 0.02),
            (0.95, y0 + 0.02),
            (0.85, y0 + 0.00),
        ]
        for x, y in arms:
            ax.plot([cx, x], [cy, y], "-", color=color, lw=1.5, transform=ax.transAxes, zorder=9, clip_on=False)
            ax.plot(x, y, "o", color=color, ms=2.8, mew=0, transform=ax.transAxes, zorder=10, clip_on=False)
        ax.plot(cx, cy, "o", color=color, ms=3.6, mew=0, transform=ax.transAxes, zorder=10, clip_on=False)


def panel_cases(fig, outer_spec, saos, gpc, harden, disc_rows) -> None:
    """Three cases: moduli + compressed GPC; no blank spacer row."""
    gs = GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=outer_spec,
        height_ratios=[0.09, 1.0, 0.40],
        hspace=0.32,
        wspace=0.28,
    )
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis("off")
    ax_h.set_facecolor(BG)
    _panel_header(
        ax_h,
        "c",
        "Key cases — SAOS + PNF reconstruction + GPC (topology inset)",
        y=0.15,
    )

    per = harden["by_alpha"]["3.4"]["per_sample"]
    by = {r["sample"]: r for r in disc_rows}
    cases = [
        ("PS3", "strong align", C_LIN, False),
        ("PS1", "high null (n.s.)", C_GOLD, False),
        ("A1PS", "branched", C_BR, True),
    ]

    for j, (sid, tag, col, branched) in enumerate(cases):
        ax = fig.add_subplot(gs[1, j])
        ax_g = fig.add_subplot(gs[2, j])
        _clean(ax)
        _clean(ax_g)

        s = saos[sid]
        row = by[sid]
        lam = np.asarray(row["lambda"], dtype=float)
        gk = np.asarray(row["g"], dtype=float)
        gp_hat, gpp_hat = _maxwell_gp_gpp(s.omega, lam, gk)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(s.omega, s.g_prime, "-", color=col, lw=1.7, label=r"$G'$ data", zorder=3)
        ax.plot(s.omega, s.g_double_prime, "-", color=C_MUTED, lw=1.35, alpha=0.9, label=r"$G''$ data", zorder=3)
        ax.plot(s.omega, gp_hat, "--", color=col, lw=1.15, alpha=0.95, label=r"$G'$ PNF", zorder=4)
        ax.plot(s.omega, gpp_hat, "--", color="#1F2937", lw=1.05, alpha=0.8, label=r"$G''$ PNF", zorder=4)
        ax.plot(s.omega[::3], gp_hat[::3], "o", color=col, ms=2.8, mew=0.0, alpha=0.85, zorder=5)
        ax.plot(s.omega[::3], gpp_hat[::3], "s", color="#1F2937", ms=2.4, mew=0.0, alpha=0.7, zorder=5)
        ax.set_xlabel(r"$\omega$ (rad/s)", fontsize=7.5, labelpad=3)
        if j == 0:
            ax.set_ylabel(r"$G'$, $G''$ (Pa)", fontsize=7.5, labelpad=5)
            ax.yaxis.set_label_coords(-0.20, 0.5)
        ax.legend(fontsize=5.2, frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0.15)
        ax.set_title(f"{sid}  ·  {tag}", fontsize=8.5, color=C_INK, loc="left", pad=3)
        ax.tick_params(axis="x", labelsize=6.5, pad=1)
        _draw_topo_inset(ax, branched=branched, color=col)

        obs = per[sid]["obs"]
        nul = per[sid]["null_mean"]
        p = per[sid]["p_ge_obs"]
        sig = r"$p{<}0.05$" if p < 0.05 else "n.s."

        y = normalize_mwd_on_x(gpc[sid].M, gpc[sid].w, MWD_X)
        ax_g.set_xscale("log")
        ax_g.fill_between(MWD_X, y, color=col, alpha=0.35, lw=0)
        ax_g.plot(MWD_X, y, color=col, lw=1.3)
        ymax = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
        ax_g.set_ylim(0.0, ymax * 1.18)
        ax_g.set_xlabel(r"$M$ (g/mol)", fontsize=7)
        if j == 0:
            ax_g.set_ylabel(r"$dW/d\ln M$", fontsize=7.5, labelpad=5)
            ax_g.yaxis.set_label_coords(-0.20, 0.5)
        ax_g.set_yticks([])
        ax_g.text(
            0.03,
            0.92,
            f"GPC  ·  tube $r$={obs:.2f} · null={nul:.2f} · {sig}",
            transform=ax_g.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            color=C_INK,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=(1.0, 1.0, 1.0, 0.22),
                ec=(0.75, 0.78, 0.82, 0.45),
                lw=0.45,
            ),
            zorder=6,
        )


def panel_outcomes_tube(ax, harden: dict) -> None:
    _clean(ax)
    _panel_header(ax, "d", "Outcome — tube alignment ≫ shuffle null", y=1.06)
    per = harden["by_alpha"]["3.4"]["per_sample"]
    ids = harden["samples"]
    obs = [per[s]["obs"] for s in ids]
    nul = [per[s]["null_mean"] for s in ids]
    cols = [C_BR if per[s]["branched"] else C_LIN for s in ids]
    x = np.arange(len(ids))
    ax.bar(x - 0.18, obs, 0.36, color=cols, label="observed", zorder=3)
    ax.bar(x + 0.18, nul, 0.36, color=C_NULL, alpha=0.75, label=r"shuffle $G_k$", zorder=3)
    c = harden["claim"]
    ax.axhline(c["fixed_lambda_alpha_3p4_mean_r"], color=C_LIN, ls="--", lw=1.0, alpha=0.85)
    ax.axhline(c["null_mean_r"], color=C_NULL, ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=50, ha="right", fontsize=6.5)
    ax.set_ylabel("tube corr  $r$", fontsize=8)
    ax.set_ylim(0, 1.08)
    leg = ax.legend(
        fontsize=6.5,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        fancybox=True,
        edgecolor=C_LINE,
        framealpha=0.28,
        borderpad=0.35,
        facecolor="white",
    )
    leg.get_frame().set_linewidth(0.55)
    leg.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.28))
    ax.text(
        0.98,
        0.02,
        f"mean $r$={c['fixed_lambda_alpha_3p4_mean_r']:.3f}\n"
        f"null={c['null_mean_r']:.3f}\n"
        f"$\\Delta$={c['delta']:.2f}  ·  {int(round(100 * c['frac_significant']))}% $p{{<}}0.05$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.28", fc=(1, 1, 1, 0.55), ec=C_LINE, lw=0.7),
    )


def panel_robust(ax, harden: dict) -> None:
    _clean(ax)
    _panel_header(ax, "e", "Robustness — $\\alpha$, free-$\\lambda$, bin proxy", y=1.06)
    alphas = [float(a) for a in harden["alphas"]]
    means = [harden["by_alpha"][str(a)]["obs_boot"]["mean"] for a in alphas]
    lo = [harden["by_alpha"][str(a)]["obs_boot"]["ci95"][0] for a in alphas]
    hi = [harden["by_alpha"][str(a)]["obs_boot"]["ci95"][1] for a in alphas]
    yerr = np.vstack([np.array(means) - lo, np.array(hi) - np.array(means)])
    ax.errorbar(alphas, means, yerr=yerr, fmt="o-", color=C_LIN, lw=1.6, ms=7, label="fixed-$\\lambda$", zorder=3)
    ax.axhline(harden["claim"]["null_mean_r"], color=C_NULL, ls="--", lw=1.1, label="null @ $\\alpha{=}3.4$")
    ax.axhline(harden["saos_bin_null"]["mean_obs"], color=C_GOLD, ls=":", lw=1.3, label="SAOS-bin")
    if harden.get("free_lambda"):
        ax.scatter(
            [3.4],
            [harden["free_lambda"]["obs_boot"]["mean"]],
            marker="D",
            s=55,
            color=C_LOO,
            zorder=5,
            label="free-$\\lambda$",
        )
    ax.set_xlabel(r"tube exponent $\alpha$", fontsize=8)
    ax.set_ylabel("mean $r$", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(
        fontsize=6,
        frameon=False,
        loc="lower right",
        labelspacing=1.85,
        handlelength=1.7,
        handletextpad=0.6,
        borderaxespad=0.45,
    )
    ax.text(
        0.02,
        0.98,
        "abs. $r$ partly structural;\n$\\Delta$ vs null is primary",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=C_MUTED,
    )


def main() -> None:
    _style()
    harden = json.loads(HARDEN.read_text())
    disc = json.loads(DISCOVERY.read_text())
    saos = {s.sample_id: s for s in load_leeds_saos_all()}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all()}

    # Compact after dropping redundant discovery panel
    fig = plt.figure(figsize=(13.2, 13.2), facecolor=BG)
    gs = GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[1.05, 0.08, 2.25, 0.30, 1.22],
        hspace=0.0,
        left=0.082,
        right=0.940,
        top=0.952,
        bottom=0.040,
    )

    # Row 1: shared header strip so a|b letters+titles share one baseline
    gs0 = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs[0],
        height_ratios=[0.14, 1.0],
        width_ratios=[1.25, 1.75],
        hspace=0.04,
        wspace=0.12,
    )
    ax_ha = fig.add_subplot(gs0[0, 0])
    ax_hb = fig.add_subplot(gs0[0, 1])
    for axh in (ax_ha, ax_hb):
        axh.set_facecolor(BG)
        axh.axis("off")
    _panel_header(ax_ha, "a", "Cohorts — paired melts vs external probes", y=0.15)
    _panel_header(ax_hb, "b", "From SAOS master curves to tube-aligned PNF mode mass", y=0.15)

    gs_pie = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1, 0], wspace=0.14)
    ax_pie1 = fig.add_subplot(gs_pie[0, 0])
    ax_pie2 = fig.add_subplot(gs_pie[0, 1])
    ax_b = fig.add_subplot(gs0[1, 1])

    _boom_pie_3d(
        fig,
        ax_pie1,
        ["linear", "branched"],
        [7, 2],
        [C_LIN, C_BR],
        "Leeds  n=9  SAOS∩GPC",
    )
    _boom_pie_3d(
        fig,
        ax_pie2,
        ["paired", "UMN SAOS", "PW OOD"],
        [9, 8, 1],
        [C_TEAL, C_UMN, C_PW],
        "Roles  (only paired enter dig)",
    )
    panel_method(ax_b)

    # Row 2: cases
    panel_cases(fig, gs[2], saos, gpc, harden, disc["rows"])

    # Row 3: d | e only (g dropped — duplicated d/e claim); width 6:4
    gs2 = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs[4],
        wspace=0.26,
        width_ratios=[6, 4],
    )
    ax_d = fig.add_subplot(gs2[0, 0])
    ax_e = fig.add_subplot(gs2[0, 1])
    panel_outcomes_tube(ax_d, harden)
    panel_robust(ax_e, harden)

    fig.suptitle(
        "Readable PNF amplitudes are molecularly aligned with MWD",
        fontsize=11.5,
        fontweight="bold",
        color=C_INK,
        x=0.055,
        ha="left",
        y=0.975,
    )

    fig.canvas.draw()
    ax_ha.set_zorder(90)
    ax_ha.patch.set_visible(True)
    ax_ha.patch.set_facecolor(BG)
    ax_ha.patch.set_alpha(1.0)
    ax_hb.set_zorder(90)
    ax_hb.patch.set_visible(True)
    ax_hb.patch.set_facecolor(BG)
    ax_hb.patch.set_alpha(1.0)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "rheo_impact_discovery"
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print(f"[fig] → {OUT / f'{stem}.png'}")


if __name__ == "__main__":
    main()
