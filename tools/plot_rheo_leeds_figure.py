#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Journal figure: Leeds PS SAOS — PNF vs Classical Prony NLS."""

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

# Match rheo journal style (muted, print-friendly)
C_GT = "#1a1a1a"
C_PNF = "#c45c26"
C_NLS = "#2c6e8a"
C_BAR_PNF = "#c45c26"
C_BAR_NLS = "#2c6e8a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--board-dir", default="outputs/rheo/leeds_real_saos")
    p.add_argument("--out", default="docs/figures/rheo/rheo_journal_leeds_saos")
    p.add_argument("--samples", default="PS1,M1,PS8", help="comma samples for spectra panels")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.board_dir)
    board = json.loads((root / "BOARD.json").read_text())
    preds = json.loads((root / "predictions.json").read_text())

    fixed = [r for r in board["rows"] if r["setting"] == "fixed_lambda"]
    samples = sorted({r["sample"] for r in fixed})
    nls_by = {r["sample"]: r for r in fixed if r["method"] == "classical_prony_nls"}
    pnf_by = {r["sample"]: r for r in fixed if r["method"] == "pnf"}

    show = [s.strip() for s in args.samples.split(",") if s.strip()]
    show = [s for s in show if s in preds]
    if not show:
        show = samples[:3]

    fig = plt.figure(figsize=(11.2, 7.2), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.2], hspace=0.38, wspace=0.32)

    # (a) bar: mean/median rel_log
    ax0 = fig.add_subplot(gs[0, 0])
    methods = ["classical_prony_nls", "pnf"]
    labels = ["Prony NLS", "PNF"]
    means = [board["summary_fixed_lambda"][m]["mean_rel_log"] for m in methods]
    meds = [board["summary_fixed_lambda"][m]["median_rel_log"] for m in methods]
    x = np.arange(2)
    w = 0.36
    ax0.bar(x - w / 2, means, w, color=[C_BAR_NLS, C_BAR_PNF], label="mean", alpha=0.9)
    ax0.bar(x + w / 2, meds, w, color=[C_BAR_NLS, C_BAR_PNF], alpha=0.45, label="median", hatch="//")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel(r"rel. log$_{10}$ error")
    ax0.set_title("(a) Fixed-λ library fit", loc="left", fontsize=11)
    ax0.legend(frameon=False, fontsize=8)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # (b) per-sample paired comparison
    ax1 = fig.add_subplot(gs[0, 1:])
    xs = np.arange(len(samples))
    nls_v = [nls_by[s]["rel_log"] for s in samples]
    pnf_v = [pnf_by[s]["rel_log"] for s in samples]
    ax1.plot(xs, nls_v, "o-", color=C_NLS, label="Prony NLS", ms=5, lw=1.2)
    ax1.plot(xs, pnf_v, "s--", color=C_PNF, label="PNF", ms=5, lw=1.2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(samples, rotation=30, ha="right")
    ax1.set_ylabel(r"rel. log$_{10}$ error")
    ax1.set_title("(b) Per-sample (PS melts)", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (c–e) spectra
    for i, sid in enumerate(show[:3]):
        ax = fig.add_subplot(gs[1, i])
        d = preds[sid]
        w = np.asarray(d["omega"])
        gp = np.asarray(d["g_prime"])
        gpp = np.asarray(d["g_double_prime"])
        pnf = d["fixed_lambda"]["pnf"]
        ax.loglog(w, gp, "-", color=C_GT, lw=1.6, label=r"GT $G'$")
        ax.loglog(w, gpp, "-", color=C_GT, lw=1.0, alpha=0.55, label=r"GT $G''$")
        ax.loglog(w, pnf["gp_hat"], "o", color=C_PNF, ms=3.2, mfc="none", mew=1.0, label=r"PNF $G'$")
        ax.loglog(w, pnf["gpp_hat"], "s", color=C_PNF, ms=2.8, mfc="none", mew=0.9, alpha=0.85, label=r"PNF $G''$")
        letter = chr(ord("c") + i)
        t = d.get("temperature_c", "")
        err = pnf["metrics"]["rel_log"]
        ax.set_title(f"({letter}) {sid}  T={t:.0f}°C  rel_log={err:.3f}", loc="left", fontsize=10)
        ax.set_xlabel(r"$\omega$ (rad/s)")
        if i == 0:
            ax.set_ylabel(r"$G',\ G''$ (Pa)")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", f"{out}.png", f"{out}.pdf")


if __name__ == "__main__":
    main()
