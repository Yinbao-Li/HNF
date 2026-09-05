#!/usr/bin/env python
"""Plot NMI Paper B figures (Fig.1 protocol schematic + Fig.2–4 gate/results).

  python tools/plot_nmi_paper_b_figures.py
 → outputs/nmi_paper_b/figures/*.pdf|.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "outputs" / "nmi_paper_b" / "figures"
NUM = _REPO / "outputs" / "nmi_paper_b" / "NUMBERS.json"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def fig1_protocol():
    """Three-column protocol schematic."""
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title(
        "Fig. 1  Huygens learning protocol (shared slots, domain-specific distance)",
        fontsize=11, pad=8, loc="left",
    )

    cols = [
        ("STEAD / seismic", "Sources: channels\nAxis: inter-sample time\nSpace@t: simultaneous\nNull: time shuffle", "#1f4e79"),
        ("RadHAR mmWave", "Sources: range×cue\nAxis: inter-frame time\nSpace@frame: simultaneous\nNull: frame shuffle", "#2e7d4f"),
        ("QM9 molecules", "Sources: atoms Z\nAxis: none (static)\nDistance: r_ij (Å)\nNull: geom. scramble", "#8b4513"),
    ]
    slots = ["Secondary sources", "Distance kernel K", "Gather / pool", "Task head"]

    for i, (title, body, color) in enumerate(cols):
        x0 = 0.4 + i * 3.4
        # column header
        ax.add_patch(FancyBboxPatch(
            (x0, 3.35), 3.0, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor="none", alpha=0.9,
        ))
        ax.text(x0 + 1.5, 3.62, title, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(x0 + 1.5, 2.95, body, ha="center", va="top", fontsize=7.5, color="#222", linespacing=1.35)

        for j, slot in enumerate(slots):
            y = 2.15 - j * 0.48
            ax.add_patch(FancyBboxPatch(
                (x0 + 0.25, y), 2.5, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#f4f6f8", edgecolor=color, linewidth=1.2,
            ))
            ax.text(x0 + 1.5, y + 0.19, slot, ha="center", va="center", fontsize=8)
            if j < len(slots) - 1:
                ax.annotate(
                    "", xy=(x0 + 1.5, y - 0.02), xytext=(x0 + 1.5, y - 0.08),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                )

    ax.text(
        5.25, 0.18,
        "Shared protocol language  ·  Not a claim of universal spacetime light-cone validation",
        ha="center", va="center", fontsize=8, style="italic", color="#444",
    )
    fig.savefig(OUT / "fig1_protocol.pdf")
    fig.savefig(OUT / "fig1_protocol.png")
    plt.close(fig)


def fig2_gates(n: dict):
    """Three-domain gate panel: relative retention after null."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=False)

    # STEAD
    ax = axes[0]
    s = n["domains"]["STEAD"]
    labels = ["clean", "time\nshuffle", "block\nshuffle", "circular\nshift", "time\nreverse"]
    vals = [
        s["clean_pick_focus"], s["time_shuffle"], s["block_shuffle"],
        s["circular_shift"], s["time_reverse"],
    ]
    colors = ["#1f4e79"] + ["#c0392b"] * 4
    ax.bar(labels, vals, color=colors, width=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean P/S F1")
    ax.set_title("a  STEAD temporal nulls", loc="left", fontsize=10)
    ax.axhline(0.1, color="#888", ls="--", lw=0.8)

    # RadHAR
    ax = axes[1]
    r = n["domains"]["RadHAR"]
    labels = ["clean", "frame\nshuffle", "hist\nT=60", "hist\nT=30"]
    vals = [r["clean_acc"], r["frame_shuffle_acc"], r["history_T60"], r["history_T30"]]
    colors = ["#2e7d4f", "#c0392b", "#e67e22", "#e67e22"]
    ax.bar(labels, vals, color=colors, width=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("b  RadHAR frame nulls", loc="left", fontsize=10)
    ax.axhline(r["bigru_acc"], color="#555", ls=":", lw=1, label=f"BiGRU {r['bigru_acc']:.3f}")
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    # QM9
    ax = axes[2]
    q = n["domains"]["QM9_gap_full"]
    labels = ["M0", "M1", "H1", "H1-scr", "shell1", "shell2", "H1-feat"]
    vals = [
        q["M0_mae"], q["M1_mae"], q["H1_mae"], q["H1_scr_mae"],
        q["shell1_mae"], q["shell2_mae"], q["H1_feat_mae"],
    ]
    colors = ["#999", "#555", "#8b4513", "#c0392b", "#c0392b", "#e67e22", "#c0392b"]
    ax.bar(labels, vals, color=colors, width=0.7)
    ax.set_ylabel("test MAE (Hartree)")
    ax.set_title("c  QM9 gap gates", loc="left", fontsize=10)
    ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Fig. 2  Protocol gates: break the intended distance → performance collapses", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_gates.pdf")
    fig.savefig(OUT / "fig2_gates.png")
    plt.close(fig)


def fig3_radhar(n: dict):
    r = n["domains"]["RadHAR"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    names = ["WaveGRU", "Wave+Pattern\n(claim)", "BiGRU", "frame\nshuffle"]
    vals = [r["wavegru_acc"], r["clean_acc"], r["bigru_acc"], r["frame_shuffle_acc"]]
    colors = ["#2e7d4f", "#1b5e20", "#555555", "#c0392b"]
    bars = ax.bar(names, vals, color=colors, width=0.65)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("test accuracy")
    ax.set_title("Fig. 3  RadHAR: matched temporal baseline vs protocol null", loc="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_radhar.pdf")
    fig.savefig(OUT / "fig3_radhar.png")
    plt.close(fig)


def fig4_qm9(n: dict):
    q = n["domains"]["QM9_gap_full"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))

    ax = axes[0]
    labels = ["M0\nno geom", "M1\nCFConv", "H1\nHuygens"]
    vals = [q["M0_mae"], q["M1_mae"], q["H1_mae"]]
    ax.bar(labels, vals, color=["#999", "#555", "#8b4513"], width=0.65)
    ax.set_ylabel("test MAE (Hartree)")
    ax.set_title("a  Primary models (full QM9 gap)", loc="left")

    ax = axes[1]
    labels = ["H1", "scramble", "shell-1", "shell-2", "feat-dist"]
    vals = [q["H1_mae"], q["H1_scr_mae"], q["shell1_mae"], q["shell2_mae"], q["H1_feat_mae"]]
    ax.bar(labels, vals, color=["#8b4513", "#c0392b", "#c0392b", "#e67e22", "#c0392b"], width=0.65)
    ax.set_ylabel("test MAE (Hartree)")
    ax.set_title("b  Geometry / locality gates", loc="left")

    fig.suptitle("Fig. 4  QM9 geometric Huygens (gap)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_qm9.pdf")
    fig.savefig(OUT / "fig4_qm9.png")
    plt.close(fig)


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    n = json.loads(NUM.read_text())
    fig1_protocol()
    fig2_gates(n)
    fig3_radhar(n)
    fig4_qm9(n)
    print(f"wrote figures → {OUT}", flush=True)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name}", flush=True)


if __name__ == "__main__":
    main()
