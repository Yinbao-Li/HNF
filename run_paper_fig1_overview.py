#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper Fig1: Huygens concept + HNF kernel formula + framework flowchart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def main() -> None:
    out_dir = Path("outputs/paper_fig1")
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])

    # A: Huygens schematic
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(-0.2, 1.2)
    ax0.set_ylim(-0.15, 1.05)
    ax0.axis("off")
    ax0.set_title("A. Huygens secondary sources", loc="left", fontsize=12)
    # wavefront arcs
    for r in [0.25, 0.45, 0.65]:
        arc = mpatches.Arc((0.15, 0.5), 2 * r, 2 * r, angle=0, theta1=-40, theta2=40, lw=1.6, color="C0")
        ax0.add_patch(arc)
    # secondary sources on a front
    xs = np.linspace(0.35, 0.35, 7)
    ys = np.linspace(0.2, 0.8, 7)
    ax0.scatter(xs, ys, s=36, c="C3", zorder=3)
    for x, y in zip(xs, ys):
        circ = plt.Circle((x + 0.18, y), 0.08, fill=False, ls="--", color="0.35", lw=1.0)
        ax0.add_patch(circ)
    ax0.annotate("primary\nwavefront", xy=(0.42, 0.88), fontsize=9)
    ax0.annotate("secondary\nsources", xy=(0.28, 0.05), fontsize=9, color="C3")
    ax0.annotate("re-radiated\nwavelets", xy=(0.72, 0.55), fontsize=9)
    ax0.text(0.02, 0.98, "Each point on a wavefront acts as a source of spherical wavelets.", fontsize=9, va="top")

    # B: kernel formula
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ax1.set_title("B. HNF causal kernel", loc="left", fontsize=12)
    formula = (
        r"$K(i,j)=\frac{1}{r_{ij}^{2}+\varepsilon}\,e^{-\gamma r_{ij}^{2}}\,e^{i\omega r_{ij}}$"
        "\n\n"
        r"$r_{ij}=\mathrm{dist}(x_i,x_j)$"
        "\n"
        "causal support: past samples only"
        "\n\n"
        r"$\gamma$: locality / support width"
        "\n"
        r"$\omega$: oscillatory / phase sensitivity"
        "\n"
        r"$\rho(t)$: soft latent weight (not crustal density)"
    )
    ax1.text(0.04, 0.78, formula, fontsize=12, va="top", family="serif")
    ax1.add_patch(mpatches.FancyBboxPatch((0.03, 0.08), 0.94, 0.28, boxstyle="round,pad=0.02", fc="#f4f7fb", ec="0.7"))
    ax1.text(
        0.06,
        0.30,
        "Learning target: discover which past samples causally contribute\n"
        "to detection / P / S decisions under Huygens locality.",
        fontsize=10,
        va="top",
    )

    # C: framework flowchart
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("C. End-to-end HNF framework", loc="left", fontsize=12)

    boxes = [
        (0.02, 0.55, 0.14, 0.28, "3C waveform\n(+ geometry)"),
        (0.20, 0.55, 0.15, 0.28, "HNF backbone\nkernel + ρ(t)"),
        (0.39, 0.70, 0.14, 0.18, "Noise cancel\nbranch"),
        (0.39, 0.42, 0.14, 0.18, "P/S pick\nheads"),
        (0.58, 0.55, 0.16, 0.28, "Physics head\nmacro / geo"),
        (0.78, 0.55, 0.18, 0.28, "1D Vp/Vs (+Q*)\nTT / wave refine"),
    ]
    for x, y, w, h, txt in boxes:
        ax2.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", fc="white", ec="0.35", lw=1.3))
        ax2.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.5)

    arrows = [
        ((0.16, 0.69), (0.20, 0.69)),
        ((0.35, 0.75), (0.39, 0.79)),
        ((0.35, 0.62), (0.39, 0.51)),
        ((0.53, 0.79), (0.58, 0.72)),
        ((0.53, 0.51), (0.58, 0.65)),
        ((0.74, 0.69), (0.78, 0.69)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax2.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.5, color="0.25"))

    ax2.add_patch(mpatches.FancyBboxPatch((0.20, 0.08), 0.58, 0.24, boxstyle="round,pad=0.02", fc="#f7f4ef", ec="0.65"))
    ax2.text(
        0.22,
        0.28,
        "Imaging path: local 1D models → distance-binned pseudo-2D section\n"
        "+ ray coverage / uncertainty / trust mask\n"
        "*Q is optional / staged; current production head focuses on Vp/Vs.",
        fontsize=9.5,
        va="top",
    )

    p = out_dir / "fig1_hnf_overview.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    docs_p = docs / "fig1_hnf_overview.png"
    docs_p.write_bytes(p.read_bytes())
    print({"figure": str(p), "docs": str(docs_p)})


if __name__ == "__main__":
    main()
