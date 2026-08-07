#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anomaly taxonomy figure: voltmeter false-alarm HC vs young-EEG patients."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)
C = {"HC": "#1B6B93", "FTD": "#C45C26", "AD": "#3D5A5B"}


def main() -> None:
    an = pd.read_csv("outputs/eeg/probe_publishable/anomaly_cases_full.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.7))
    fig.subplots_adjust(wspace=0.48, left=0.07, right=0.98, top=0.82, bottom=0.2)

    ax = axes[0]
    for g, d in an.groupby("clinical_group"):
        ax.scatter(d.theta_alpha_ratio, d.PC1, s=16, c=C[g], alpha=0.35, edgecolors="none")
    rings = an[an.ring]
    ax.scatter(
        rings.theta_alpha_ratio,
        rings.PC1,
        s=70,
        facecolors="none",
        edgecolors="#111",
        linewidths=1.1,
        zorder=5,
        label="black ring",
    )
    for _, r in rings.iterrows():
        ax.annotate(r.subject_id.replace("sub-", ""), (r.theta_alpha_ratio, r.PC1), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"voltmeter  $\theta/\alpha$")
    ax.set_ylabel("PC1 (manifold)")
    ax.set_title("a  Rings follow the voltmeter", loc="left", fontsize=9, pad=6)

    ax = axes[1]
    hc_p = an[an.taxon.str.contains("HC_looks_disease")]
    hc_ok = an[(an.clinical_group == "HC") & (~an.taxon.str.contains("HC_looks_disease"))]
    dis_s = an[an.taxon.str.contains("disease_looks_HC")]
    dis_ok = an[(an.clinical_group != "HC") & (~an.taxon.str.contains("disease_looks_HC"))]
    series = [hc_ok.D_eff_res_trainfit, hc_p.D_eff_res_trainfit, dis_s.D_eff_res_trainfit, dis_ok.D_eff_res_trainfit]
    labels = ["HC\nconcord", "HC\nparadox", "pts\nyoung-EEG", "pts\nconcord"]
    cols = ["#1B6B93", "#7BA3B5", "#C45C26", "#3D5A5B"]
    bp = ax.boxplot(series, labels=labels, patch_artist=True, showfliers=False, widths=0.62)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(0, color="#888", lw=0.6, ls="--")
    ax.set_ylabel(r"probe leftover $D_\mathrm{eff}$")
    ax.set_title("b  Leftover splits the two paradoxes", loc="left", fontsize=9, pad=6)

    ax = axes[2]
    ax.scatter(hc_ok.theta_alpha_ratio, hc_ok.D_eff_res_trainfit, s=18, c="#1B6B93", alpha=0.35, label="HC concord")
    ax.scatter(hc_p.theta_alpha_ratio, hc_p.D_eff_res_trainfit, s=28, c="#1B6B93", edgecolors="#111", linewidths=0.6, label="HC paradox")
    ax.scatter(dis_ok.theta_alpha_ratio, dis_ok.D_eff_res_trainfit, s=18, c="#3D5A5B", alpha=0.35, label="pt concord")
    ax.scatter(dis_s.theta_alpha_ratio, dis_s.D_eff_res_trainfit, s=28, c="#C45C26", edgecolors="#111", linewidths=0.6, label="pt young-EEG")
    for sid in ("sub-043", "sub-059", "sub-061", "sub-080", "sub-025", "sub-003"):
        r = an.loc[an.subject_id == sid].iloc[0]
        ax.annotate(sid.replace("sub-", ""), (r.theta_alpha_ratio, r.D_eff_res_trainfit), fontsize=6)
    ax.axhline(0, color="#888", lw=0.5, ls="--")
    ax.axvline(float(an.loc[an.clinical_group == "HC", "theta_alpha_ratio"].median()), color="#888", lw=0.5, ls=":")
    ax.set_xlabel(r"$\theta/\alpha$")
    ax.set_ylabel(r"probe leftover")
    ax.set_title("c  Named cases", loc="left", fontsize=9, pad=6)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    out = Path("docs/figures/eeg")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "eeg_anomaly_cases.png")
    fig.savefig(out / "eeg_anomaly_cases.pdf")
    plt.close(fig)
    print(f"[fig] → {out / 'eeg_anomaly_cases.png'}")


if __name__ == "__main__":
    main()
