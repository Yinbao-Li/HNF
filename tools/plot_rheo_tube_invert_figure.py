#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure for G4 tube-invert / LOO Nature dig."""

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

C = {
    "elliott_nn": "#222222",
    "loo_mlp_mwd": "#6b4c9a",
    "loo_ridge_mwd": "#2c6e8a",
    "synth_ridge": "#3d7a5a",
    "synth_mlp": "#5a9aa8",
    "tube": "#888888",
    "tube_invert": "#c45c26",
    "hybrid_invert": "#a67c52",
}
LABEL = {
    "elliott_nn": "Elliott NN",
    "loo_mlp_mwd": "LOO real MLP",
    "loo_ridge_mwd": "LOO real Ridge",
    "synth_ridge": "Synth Ridge",
    "synth_mlp": "Synth MLP",
    "tube": "Tube deposit",
    "tube_invert": "Tube invert",
    "hybrid_invert": "Hybrid invert",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--invert", default="outputs/rheo/tube_invert_g4/INVERT.json")
    p.add_argument("--out", default="docs/figures/rheo/rheo_nature_g4_invert")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = json.loads(Path(args.invert).read_text())
    rows = d["rows"]
    x = np.asarray(d["mwd_x"], dtype=float)
    methods = [
        m
        for m in [
            "elliott_nn",
            "loo_mlp_mwd",
            "loo_ridge_mwd",
            "synth_ridge",
            "tube",
            "tube_invert",
        ]
        if m in d["summary"]["methods"]
    ]

    fig = plt.figure(figsize=(11.2, 6.8), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    means = [d["summary"]["methods"][m]["mean_rmse"] for m in methods]
    ax.barh(np.arange(len(methods)), means, color=[C[m] for m in methods])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([LABEL[m] for m in methods], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean MWD RMSE")
    ax.set_title("(a) Accuracy ladder", loc="left", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    show = [m for m in ["elliott_nn", "loo_mlp_mwd", "loo_ridge_mwd", "synth_ridge", "tube"] if m in methods]
    ids = [r["sample"] for r in rows]
    xpos = np.arange(len(ids))
    width = 0.16
    for i, m in enumerate(show):
        vals = [r[m]["rmse"] for r in rows]
        ax.bar(xpos + (i - len(show) / 2) * width + width / 2, vals, width, color=C[m], label=LABEL[m])
    ax.set_xticks(xpos)
    ax.set_xticklabels(ids, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("MWD RMSE")
    ax.set_title("(b) Per-sample", loc="left", fontsize=10)
    ax.legend(fontsize=6, frameon=False, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    nn = d["summary"]["methods"]["elliott_nn"]["mean_rmse"]
    loo = d["summary"]["methods"]["loo_mlp_mwd"]["mean_rmse"]
    ridge = d["summary"]["methods"]["synth_ridge"]["mean_rmse"]
    inv = d["summary"]["methods"]["tube_invert"]["mean_rmse"]
    tu = d["summary"]["methods"]["tube"]["mean_rmse"]
    frac = (tu - loo) / (tu - nn)
    txt = (
        "(c) Nature verdict\n\n"
        f"NN RMSE          {nn:.3f}\n"
        f"LOO real MLP     {loo:.3f}  ({100*frac:.0f}% gap)\n"
        f"Synth Ridge      {ridge:.3f}\n"
        f"Tube deposit     {tu:.3f}\n"
        f"Zero-shot invert {inv:.3f}  (worse)\n\n"
        "Zero-shot nonlinear invert on the\n"
        "α=3.4 heuristic cannot replace\n"
        "Elliott tube-sim pretraining.\n\n"
        "Nearest Nature step: readable\n"
        "log G_k + few GPC labels ≈ NN."
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8.5)

    order = sorted(rows, key=lambda r: r["elliott_nn"]["rmse"])
    picks = [order[0], order[len(order) // 2], order[-1]]
    for j, r in enumerate(picks):
        ax = fig.add_subplot(gs[1, j])
        ax.plot(x, r["mwd_gt"], "k-", lw=1.6, label="GPC")
        ax.plot(x, r["mwd_nn"], color=C["elliott_nn"], lw=1.2, label="NN")
        ax.plot(x, r["mwd_loo_mlp"], color=C["loo_mlp_mwd"], lw=1.2, label="LOO MLP")
        ax.plot(x, r["mwd_ridge"], color=C["synth_ridge"], lw=1.0, ls="--", label="Synth Ridge")
        ax.plot(x, r["mwd_invert"], color=C["tube_invert"], lw=1.0, ls=":", label="Invert")
        ax.set_xscale("log")
        ax.set_xlabel("M")
        if j == 0:
            ax.set_ylabel(r"$dW/d\ln M$")
        ax.set_title(f"({'def'[j]}) {r['sample']}", loc="left", fontsize=10)
        if j == 2:
            ax.legend(fontsize=6, frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", f"{out}.png")


if __name__ == "__main__":
    main()
