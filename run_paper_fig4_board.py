#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package existing method-comparison assets into a single Fig4 board.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Fig4 method-comparison board")
    p.add_argument("--inv-report", default="outputs/inv_full_compare/report.json")
    p.add_argument("--proof-report", default="outputs/proof_suite/proof_report.json")
    p.add_argument("--bars", default="docs/figures/synth_full_compare_bars.png")
    p.add_argument("--stead-scatter", default="docs/figures/stead_refine_scatter.png")
    p.add_argument("--wave-hist", default="docs/figures/synth_wave_delta_hist.png")
    p.add_argument("--output", default="docs/figures/fig4_method_comparison.png")
    return p.parse_args()


def load_methods(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return list(data.get("methods", []))


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    methods = load_methods(Path(args.inv_report))
    proof = {}
    pr = Path(args.proof_report)
    if pr.exists():
        proof = json.loads(pr.read_text())

    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])

    # A: method bars from report (re-draw for crispness)
    ax0 = fig.add_subplot(gs[0, 0])
    if methods:
        names = [m["method"] for m in methods]
        vp = [m["vp_rmse"] for m in methods]
        colors = []
        for m in methods:
            g = m.get("group", "")
            colors.append({"travel_time": "C0", "fwi": "C1", "picking_prior": "C2"}.get(g, "0.6"))
        ax0.barh(range(len(names)), vp, color=colors)
        ax0.set_yticks(range(len(names)))
        ax0.set_yticklabels(names, fontsize=8)
        ax0.set_xlabel("Vp RMSE")
        ax0.set_title("A. Synthetic method comparison (Vp RMSE)")
        ax0.grid(True, axis="x", alpha=0.3)
        # highlight best travel-time
        tt = [m for m in methods if m.get("group") == "travel_time"]
        if tt:
            best = min(tt, key=lambda m: m["vp_rmse"])
            ax0.annotate(
                f"best TT: {best['method']}",
                xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom", fontsize=8,
            )
    else:
        ax0.axis("off")
        ax0.text(0.1, 0.5, "Missing inv_full_compare/report.json", fontsize=11)

    # B: existing bars image if present
    ax1 = fig.add_subplot(gs[0, 1])
    bars = Path(args.bars)
    if bars.exists():
        ax1.imshow(mpimg.imread(bars))
        ax1.set_title("B. Full comparison board (existing)")
        ax1.axis("off")
    else:
        ax1.axis("off")
        ax1.text(0.1, 0.5, "Missing synth_full_compare_bars.png", fontsize=11)

    # C: STEAD refine scatter
    ax2 = fig.add_subplot(gs[1, 0])
    sc = Path(args.stead_scatter)
    if sc.exists():
        ax2.imshow(mpimg.imread(sc))
        ax2.set_title("C. STEAD init → refine travel-time")
        ax2.axis("off")
    else:
        ax2.axis("off")
        ax2.text(0.1, 0.5, "Missing stead_refine_scatter.png", fontsize=11)

    # D: narrative + wave hist / proof bullets
    ax3 = fig.add_subplot(gs[1, 1])
    wh = Path(args.wave_hist)
    if wh.exists():
        ax3.imshow(mpimg.imread(wh))
        ax3.set_title("D. Waveform-aware delta (Route A2 / proof)")
        ax3.axis("off")
    else:
        ax3.axis("off")
        bullets = [
            "Fig4 packaging notes",
            "",
            "• Travel-time oracles remain strong on layered synth",
            "• HNF value is waveform-aware init + STEAD refine path",
            "• Use Route A2 / proof_suite for complex-model claims",
        ]
        if proof:
            bullets.append(f"• proof_report keys: {', '.join(list(proof)[:6])}")
        ax3.text(0.05, 0.95, "\n".join(bullets), va="top", fontsize=10, family="DejaVu Sans")

    fig.suptitle("Figure 4. Method comparison materials", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=170)
    # also copy into outputs
    out2 = Path("outputs/paper_fig4_board")
    out2.mkdir(parents=True, exist_ok=True)
    (out2 / "fig4_method_comparison.png").write_bytes(out.read_bytes())
    meta = {
        "output": str(out),
        "n_methods": len(methods),
        "sources": {
            "inv_report": args.inv_report,
            "proof_report": args.proof_report,
            "bars": args.bars,
            "stead_scatter": args.stead_scatter,
            "wave_hist": args.wave_hist,
        },
    }
    (out2 / "fig4_board_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
