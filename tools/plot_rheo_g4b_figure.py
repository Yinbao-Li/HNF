#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure: G4b RDP+CLF vs heuristic vs Elliott NN."""

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
    "elliott_nn": "#222",
    "rdp_mlp": "#6b4c9a",
    "rdp_ridge": "#a67c52",
    "heur_ridge": "#3d7a5a",
    "heur_mlp": "#5a9aa8",
}
LAB = {
    "elliott_nn": "Elliott NN",
    "rdp_mlp": "RDP+CLF MLP",
    "rdp_ridge": "RDP+CLF Ridge",
    "heur_ridge": "Heur Ridge",
    "heur_mlp": "Heur MLP",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--g4b", default="outputs/rheo/tube_g4b_rdp/G4B.json")
    p.add_argument("--out", default="docs/figures/rheo/rheo_nature_g4b_rdp")
    args = p.parse_args()
    d = json.loads(Path(args.g4b).read_text())
    rows = d["rows"]
    x = np.asarray(d["mwd_x"])
    methods = [m for m in LAB if m in d["summary"]["methods"]]

    fig = plt.figure(figsize=(11.0, 6.4), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    means = [d["summary"]["methods"][m]["mean_rmse"] for m in methods]
    ax.barh(np.arange(len(methods)), means, color=[C[m] for m in methods])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([LAB[m] for m in methods], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean MWD RMSE")
    ax.set_title("(a) Zero-shot ladder", loc="left", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    show = [m for m in ["elliott_nn", "rdp_mlp", "heur_ridge"] if m in methods]
    ids = [r["sample"] for r in rows]
    xpos = np.arange(len(ids))
    w = 0.25
    for i, m in enumerate(show):
        ax.bar(xpos + (i - 1) * w, [r[m]["rmse"] for r in rows], w, color=C[m], label=LAB[m])
    ax.set_xticks(xpos)
    ax.set_xticklabels(ids, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("MWD RMSE")
    ax.set_title("(b) Per-sample", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    s = d["summary"]["methods"]
    nn, rdp, heur = s["elliott_nn"]["mean_rmse"], s["rdp_mlp"]["mean_rmse"], s["heur_ridge"]["mean_rmse"]
    frac = s["rdp_mlp"].get("frac_gap_closed_vs_heur_to_nn", 0)
    ax.text(
        0.02,
        0.98,
        "(c) G4b verdict\n\n"
        f"NN          {nn:.3f}\n"
        f"RDP+CLF MLP {rdp:.3f}  ({100*frac:.0f}% of heur→NN)\n"
        f"Heur Ridge  {heur:.3f}\n\n"
        "Open RDP+CLF helps a little.\n"
        "Remaining gap ≈ Elliott BoB\n"
        "+ ~8e5 sims (not public).\n\n"
        "Still not Nature-alone.",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
    )

    order = sorted(rows, key=lambda r: r["elliott_nn"]["rmse"])
    for j, r in enumerate([order[0], order[len(order) // 2], order[-1]]):
        ax = fig.add_subplot(gs[1, j])
        ax.plot(x, r["mwd_gt"], "k-", lw=1.5, label="GPC")
        ax.plot(x, r["mwd_nn"], color=C["elliott_nn"], lw=1.2, label="NN")
        ax.plot(x, r["mwd_rdp_mlp"], color=C["rdp_mlp"], lw=1.2, label="RDP MLP")
        ax.plot(x, r["mwd_heur_ridge"], color=C["heur_ridge"], lw=1.0, ls="--", label="Heur")
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
