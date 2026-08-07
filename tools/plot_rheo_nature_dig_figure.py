#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nature dig plate: Leeds tube harden + UMN topology SAOS."""

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

C_LIN = "#2c6e8a"
C_BR = "#c45c26"
C_NULL = "#8a8a8a"
C_STAR = "#3d7a5a"
C_BOT = "#6b4c9a"
C_TR = "#b0892e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--harden", default="outputs/rheo/spectrum_mwd_harden/HARDEN.json")
    p.add_argument("--umn", default="outputs/rheo/umn_bottlebrush/BOARD.json")
    p.add_argument("--out", default="docs/figures/rheo/rheo_nature_dig")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    h = json.loads(Path(args.harden).read_text())
    u = json.loads(Path(args.umn).read_text())

    fig = plt.figure(figsize=(11.2, 6.6), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.38)

    # (a) obs vs null per sample α=3.4
    ax = fig.add_subplot(gs[0, 0])
    per = h["by_alpha"]["3.4"]["per_sample"]
    ids = h["samples"]
    obs = [per[s]["obs"] for s in ids]
    nul = [per[s]["null_mean"] for s in ids]
    cols = [C_BR if per[s]["branched"] else C_LIN for s in ids]
    x = np.arange(len(ids))
    ax.bar(x - 0.18, obs, 0.36, color=cols, label="PNF vs MWD")
    ax.bar(x + 0.18, nul, 0.36, color=C_NULL, alpha=0.55, label="shuffle-\(G_k\)")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("tube corr \(r\)")
    ax.set_title("(a) Leeds tube alignment", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 1.05)

    # (b) α sensitivity + controls
    ax = fig.add_subplot(gs[0, 1])
    alphas = [float(a) for a in h["alphas"]]
    means = [h["by_alpha"][str(a)]["obs_boot"]["mean"] for a in alphas]
    lo = [h["by_alpha"][str(a)]["obs_boot"]["ci95"][0] for a in alphas]
    hi = [h["by_alpha"][str(a)]["obs_boot"]["ci95"][1] for a in alphas]
    yerr = np.vstack([np.array(means) - np.array(lo), np.array(hi) - np.array(means)])
    ax.errorbar(alphas, means, yerr=yerr, fmt="o-", color=C_LIN, lw=1.5, ms=6, label="fixed-λ obs")
    null_m = h["by_alpha"]["3.4"]["null_boot"]["mean"]
    ax.axhline(null_m, color=C_NULL, ls="--", lw=1.2, label="null @α=3.4")
    ax.axhline(h["saos_bin_null"]["mean_obs"], color=C_BR, ls=":", lw=1.4, label="SAOS-bin (no Prony)")
    if h.get("free_lambda"):
        ax.scatter([3.4], [h["free_lambda"]["obs_boot"]["mean"]], marker="D", s=45, color=C_BOT, zorder=5, label="free-λ")
    ax.set_xlabel("tube exponent α")
    ax.set_ylabel("mean \(r\)")
    ax.set_title("(b) Robustness controls", loc="left", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_ylim(0, 1.05)

    # (c) claim strip
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    c = h["claim"]
    txt = (
        "Leeds n=9 harden\n\n"
        f"obs r = {c['fixed_lambda_alpha_3p4_mean_r']:.3f}\n"
        f"  CI [{c['fixed_lambda_alpha_3p4_ci95'][0]:.2f}, {c['fixed_lambda_alpha_3p4_ci95'][1]:.2f}]\n"
        f"null r = {c['null_mean_r']:.3f}\n"
        f"Δ = {c['delta']:.3f}\n"
        f"frac p<0.05 = {c['frac_significant']:.2f}\n"
        f"SAOS-bin r = {c['saos_bin_mean_r']:.3f}\n"
    )
    if c.get("free_lambda_mean_r") is not None:
        txt += f"free-λ r = {c['free_lambda_mean_r']:.3f}\n"
    ax.text(0.02, 0.95, "(c) Locked numbers", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
    ax.text(0.02, 0.82, txt, transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    # (d) UMN PNF vs NLS
    ax = fig.add_subplot(gs[1, 0])
    rows = u["rows"]
    ax.scatter(
        [r["nls_rel_log"] for r in rows],
        [r["pnf_rel_log"] for r in rows],
        c=[
            {"star-like": C_STAR, "bottlebrush": C_BOT, "transition": C_TR}[r["regime"]]
            for r in rows
        ],
        s=42,
        zorder=3,
    )
    lims = [
        min(min(r["nls_rel_log"] for r in rows), min(r["pnf_rel_log"] for r in rows)) * 0.9,
        max(max(r["nls_rel_log"] for r in rows), max(r["pnf_rel_log"] for r in rows)) * 1.1,
    ]
    ax.plot(lims, lims, "--", color="#999", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("NLS rel_log")
    ax.set_ylabel("PNF rel_log")
    ax.set_title("(d) UMN PNF ≡ NLS", loc="left", fontsize=10)

    # (e) 〈log λ〉 vs Nbb
    ax = fig.add_subplot(gs[1, 1])
    for r in rows:
        col = {"star-like": C_STAR, "bottlebrush": C_BOT, "transition": C_TR}[r["regime"]]
        ax.scatter(r["nbb"], r["pnf_mean_log10_lam"], c=col, s=48, zorder=3)
        ax.annotate(f"N{r['nbb']}", (r["nbb"], r["pnf_mean_log10_lam"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(50, color="#bbb", ls="--", lw=0.9)
    ax.axvline(70, color="#bbb", ls="--", lw=0.9)
    sp = u["spearman_nbb_vs_mean_loglam"]
    ax.set_xlabel("backbone \(N_{bb}\)")
    ax.set_ylabel(r"mean $\log_{10}\lambda$")
    ax.set_title(f"(e) Spectrum vs topology  ρ={sp['rho']:.2f}", loc="left", fontsize=10)

    # (f) Nature gate
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0.02, 0.95, "(f) Top-end discovery gates", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        0.82,
        "NOW (publishable core)\n"
        "• Mode mass ≫ shuffle-G null (Δ≈0.35)\n"
        "• α-robust; free-λ preserved\n"
        "• PNF≡NLS on PS + PLA grafts\n\n"
        "CEILINGS (keep in paper)\n"
        "• Abs. r partly structural (bin≈0.86)\n"
        "• n=9; LOO Mw fails; Elliott wins MWD\n"
        "• UMN: no w(M); no Nbb–spectrum link\n\n"
        "NATURE GATES\n"
        "• Unique map / larger paired n\n"
        "• Cross-chemistry nontrivial predict\n"
        "• Close gap to nonlinear tube / NN",
        transform=ax.transAxes,
        va="top",
        fontsize=8.0,
        family="sans-serif",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", f"{out}.png")


if __name__ == "__main__":
    main()
