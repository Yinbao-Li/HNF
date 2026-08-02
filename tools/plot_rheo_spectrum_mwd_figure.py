#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure: PNF spectrum ↔ MWD discovery (Leeds)."""

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

from hnf.rheo_gpc import load_leeds_gpc_all, mwd_on_log_grid

C_LIN = "#2c6e8a"
C_BR = "#c45c26"
C_NULL = "#888888"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--discovery", default="outputs/rheo/spectrum_mwd_mine/DISCOVERY.json")
    p.add_argument("--out", default="docs/figures/rheo/rheo_journal_spectrum_mwd")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = json.loads(Path(args.discovery).read_text())
    rows = d["rows"]
    gpc = {s.sample_id: s for s in load_leeds_gpc_all()}
    logM_grid = np.linspace(3.0, 7.0, 80)

    fig = plt.figure(figsize=(11.0, 7.0), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

    # (a) tube corr vs shuffle null
    ax = fig.add_subplot(gs[0, 0])
    ids = [r["sample"] for r in rows]
    obs = [r["tube_corr"] for r in rows]
    # approximate null means from saved note — recompute light null for plot
    rng = np.random.default_rng(0)

    def tube_corr(lam, g, mwd, alpha=3.4):
        g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
        lam = np.asarray(lam, dtype=np.float64)
        w = g / max(g.sum(), 1e-30)
        logM_med = float(np.average(logM_grid, weights=np.maximum(mwd, 1e-12)))
        log_lam_med = float(np.average(np.log10(lam), weights=w))
        logM_of_lam = logM_med + (1.0 / alpha) * (np.log10(lam) - log_lam_med)
        mwd_mass = np.zeros(len(lam))
        dlog = float(logM_grid[1] - logM_grid[0])
        for j, lm in enumerate(logM_grid):
            k = int(np.argmin(np.abs(logM_of_lam - lm)))
            mwd_mass[k] += mwd[j] * dlog
        mwd_mass /= max(mwd_mass.sum(), 1e-30)
        if w.std() < 1e-12 or mwd_mass.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(w, mwd_mass)[0, 1])

    null_m = []
    for r in rows:
        mwd = mwd_on_log_grid(gpc[r["sample"]], logM_grid)
        ns = [
            tube_corr(r["lambda"], rng.permutation(r["g"]), mwd)
            for _ in range(400)
        ]
        null_m.append(float(np.mean(ns)))

    x = np.arange(len(ids))
    ax.bar(x - 0.18, obs, 0.36, color=[C_BR if r["branched"] else C_LIN for r in rows], label="PNF vs MWD")
    ax.bar(x + 0.18, null_m, 0.36, color=C_NULL, alpha=0.55, label="shuffle-G null")
    ax.axhline(np.mean(obs), color=C_LIN, ls="--", lw=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("corr(mode mass, MWD mass)")
    ax.set_title(r"(a) Tube map $M\propto\lambda^{1/3.4}$", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) example alignment PS3
    ax = fig.add_subplot(gs[0, 1])
    r = next(z for z in rows if z["sample"] == "PS3")
    lam = np.asarray(r["lambda"])
    g = np.asarray(r["g"])
    w = g / g.sum()
    mwd = mwd_on_log_grid(gpc["PS3"], logM_grid)
    logM_med = float(np.average(logM_grid, weights=np.maximum(mwd, 1e-12)))
    log_lam_med = float(np.average(np.log10(lam), weights=w))
    logM_of_lam = logM_med + (1.0 / 3.4) * (np.log10(lam) - log_lam_med)
    ax.plot(logM_grid, mwd / mwd.max(), color="k", lw=1.6, label="GPC MWD")
    ax.stem(logM_of_lam, w / w.max(), linefmt="C1-", markerfmt="C1o", basefmt=" ", label="PNF modes")
    ax.set_xlabel(r"$\log_{10} M$")
    ax.set_ylabel("normalized mass")
    ax.set_title(f"(b) PS3 alignment  r={r['tube_corr']:.2f}", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) LOO: honest failure on Mw
    ax = fig.add_subplot(gs[0, 2])
    L = d["loo"]["log10_Mw"]
    y = np.array([r["log10_Mw"] for r in rows])
    yhat = np.array(L["loo_pnf_uni"]["yhat"])
    colors = [C_BR if r["branched"] else C_LIN for r in rows]
    ax.scatter(y, yhat, c=colors, s=40, zorder=3)
    lims = [min(y.min(), yhat.min()) - 0.05, max(y.max(), yhat.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"true $\log_{10} M_w$")
    ax.set_ylabel(r"LOO pred (best PNF feat)")
    ax.set_title(
        f"(c) Mw LOO fails  R²={L['loo_pnf_uni']['r2_loo']:.2f}",
        loc="left",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (d) spectrum geometry vs ω span confound
    ax = fig.add_subplot(gs[1, 0])
    for r in rows:
        ax.scatter(
            r["saos_log10_omega_span"],
            r["pnf_mean_log10_lam"],
            c=C_BR if r["branched"] else C_LIN,
            s=50,
            zorder=3,
        )
        ax.text(
            r["saos_log10_omega_span"] + 0.05,
            r["pnf_mean_log10_lam"],
            r["sample"],
            fontsize=7,
        )
    ax.set_xlabel(r"SAOS $\log_{10}(\omega_{max}/\omega_{min})$")
    ax.set_ylabel(r"PNF $\langle\log_{10}\lambda\rangle_w$")
    ax.set_title("(d) Window confound for branching", loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (e) A1PS vs linear: mode spectrum
    ax = fig.add_subplot(gs[1, 1])
    for sid, c, ls in [("PS1", C_LIN, "-"), ("A1PS", C_BR, "--")]:
        r = next(z for z in rows if z["sample"] == sid)
        lam = np.asarray(r["lambda"])
        g = np.asarray(r["g"])
        ax.loglog(lam, g / g.max(), ls, color=c, marker="o", ms=4, label=sid)
    ax.set_xlabel(r"$\lambda$ (s)")
    ax.set_ylabel(r"$G_k$ / max")
    ax.set_title("(e) Linear PS1 vs branched A1PS", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (f) summary text panel
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    txt = (
        "Takeaways (n=9, hypothesis-grade)\n\n"
        f"1. Tube alignment mean r={d['tube']['mean_all']:.2f}\n"
        "   ≫ amplitude-shuffle null (~0.57);\n"
        "   most samples p<0.05.\n\n"
        "2. LOO Mw/Ð from descriptors fails:\n"
        "   need synthetic pretrain (Elliott)\n"
        "   or more samples — not a bug of PNF.\n\n"
        "3. Raw 'branching AUC=1' is partly\n"
        "   ω-window confound (panel d);\n"
        "   A1PS remains a spectral outlier.\n\n"
        "Increment vs Elliott NN: readable\n"
        "Maxwell weights as molecularly\n"
        "aligned sufficient statistics, plus\n"
        "explicit failure modes."
    )
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="DejaVu Sans", wrap=True)
    ax.set_title("(f) Claim boundary", loc="left", fontsize=11)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", f"{out}.png")


if __name__ == "__main__":
    main()
