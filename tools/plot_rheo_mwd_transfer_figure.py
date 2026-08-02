#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure: interpretable spectrum→MWD transfer vs Elliott NN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

C_NN = "#2c6e8a"
C_RIDGE = "#c45c26"
C_TUBE = "#6a6a6a"
C_GT = "#1a1a1a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--board", default="outputs/rheo/mwd_transfer/TRANSFER.json")
    p.add_argument("--out", default="docs/figures/rheo/rheo_journal_mwd_transfer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = json.loads(Path(args.board).read_text())
    rows = d["rows"]
    x = np.asarray(d["mwd_x"])
    ids = [r["sample"] for r in rows]

    fig = plt.figure(figsize=(11.2, 7.0), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # (a) RMSE bars
    ax = fig.add_subplot(gs[0, 0])
    methods = ["elliott_nn", "synth_ridge", "tube"]
    labels = ["Elliott NN", "Synth Ridge", "Tube proj."]
    colors = [C_NN, C_RIDGE, C_TUBE]
    means = [d["summary"]["methods"][m]["mean_rmse"] for m in methods]
    ax.bar(np.arange(3), means, color=colors, width=0.65)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("mean MWD RMSE")
    ax.set_title("(a) Full MWD error", loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) per-sample RMSE
    ax = fig.add_subplot(gs[0, 1:])
    xs = np.arange(len(ids))
    ax.plot(xs, [r["elliott_nn"]["rmse"] for r in rows], "o-", color=C_NN, label="Elliott NN", ms=5)
    ax.plot(xs, [r["synth_ridge"]["rmse"] for r in rows], "s--", color=C_RIDGE, label="Synth Ridge", ms=5)
    ax.plot(xs, [r["tube"]["rmse"] for r in rows], "^:", color=C_TUBE, label="Tube proj.", ms=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ids, rotation=30, ha="right")
    ax.set_ylabel("MWD RMSE")
    ax.set_title("(b) Per-sample", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c–e) example MWDs
    for i, sid in enumerate(["M1", "PS2", "A1PS"]):
        ax = fig.add_subplot(gs[1, i])
        r = next(z for z in rows if z["sample"] == sid)
        ax.semilogx(x, r["mwd_gt"], color=C_GT, lw=1.8, label="GPC")
        ax.semilogx(x, r["mwd_nn"], color=C_NN, lw=1.4, label="Elliott NN")
        ax.semilogx(x, r["mwd_ridge"], color=C_RIDGE, lw=1.2, ls="--", label="Synth Ridge")
        ax.set_xlim(1e3, 1e7)
        ax.set_xlabel("M (g/mol)")
        if i == 0:
            ax.set_ylabel(r"$dW/d\ln M$")
            ax.legend(frameon=False, fontsize=7)
        letter = chr(ord("c") + i)
        ax.set_title(
            f"({letter}) {sid}  NN={r['elliott_nn']['rmse']:.3f}  Ridge={r['synth_ridge']['rmse']:.3f}",
            loc="left",
            fontsize=10,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", f"{out}.png")


if __name__ == "__main__":
    main()
