#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-panel EEG Nature-track figure: residual / transfer / jackknife."""

from __future__ import annotations

import json
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
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)

C = {"HC": "#1B6B93", "FTD": "#C45C26", "AD": "#3D5A5B"}
GROUPS = ("HC", "FTD", "AD")


def main() -> None:
    v3 = pd.read_csv("outputs/eeg/structure_residual/subjects_native_v3_residual.csv")
    xfer = json.loads(Path("outputs/eeg/longitudinal_ds005385_rho/TRANSFER.json").read_text())
    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.07, right=0.98, top=0.82, bottom=0.22)

    ax = fig.add_subplot(gs[0, 0])
    data = [v3.loc[v3.clinical_group == g, "D_eff_res"].to_numpy() for g in GROUPS]
    bp = ax.boxplot(data, labels=GROUPS, patch_artist=True, widths=0.55, showfliers=False)
    for patch, g in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(C[g])
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="#888", lw=0.6, ls="--")
    ax.set_ylabel(r"$D_\mathrm{eff}$ residual")
    ax.set_title("a  v3 leftover after age+sex+θ/α", loc="left", fontsize=9, pad=8)

    ax = fig.add_subplot(gs[0, 1])
    rows = xfer["rows"]["aniso_phase_off"]
    s1 = np.asarray([r["ses1"]["rho_std"] for r in rows], float)
    s2 = np.asarray([r["ses2"]["rho_std"] for r in rows], float)
    ax.scatter(s1, s2, s=18, c="#0B3D4A", alpha=0.75, edgecolors="none")
    lo = float(np.nanmin([s1.min(), s2.min()]))
    hi = float(np.nanmax([s1.max(), s2.max()]))
    ax.plot([lo, hi], [lo, hi], color="#888", lw=0.7, ls="--")
    ax.set_xlabel(r"ses-1 $\rho_\mathrm{std}$")
    ax.set_ylabel(r"ses-2 $\rho_\mathrm{std}$")
    ax.set_title("b  ds005385 aniso ρ test–retest", loc="left", fontsize=9, pad=8)

    ax = fig.add_subplot(gs[0, 2])
    # recreate LOSO quickly for the histogram
    from hnf.eeg_subject_diffusion import residualize, sex_to_float, spearman_r

    y = v3["D_eff"].to_numpy(float)
    X = np.column_stack(
        [
            v3.age.to_numpy(float),
            sex_to_float(v3.gender.tolist()),
            v3.theta_alpha_ratio.to_numpy(float),
            v3.bp_alpha.to_numpy(float),
        ]
    )
    stage = v3.clinical_group.map({"HC": 0, "FTD": 1, "AD": 2}).to_numpy(float)
    rs = []
    for i in range(len(v3)):
        keep = np.ones(len(v3), bool)
        keep[i] = False
        resid_i, _, _ = residualize(y[keep], X[keep])
        r, _ = spearman_r(resid_i, stage[keep])
        if np.isfinite(r):
            rs.append(r)
    ax.hist(rs, bins=12, color="#0B3D4A", alpha=0.8)
    ax.axvline(0.0, color="#888", lw=0.7, ls="--")
    ax.set_xlabel("LOSO leftover Spearman r")
    ax.set_ylabel("folds")
    ax.set_title("c  v3 jackknife (all r>0)", loc="left", fontsize=9, pad=8)

    out = Path("docs/figures/eeg")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "eeg_nature_track.png")
    fig.savefig(out / "eeg_nature_track.pdf")
    plt.close(fig)
    print(f"[fig] → {out / 'eeg_nature_track.png'}", flush=True)


if __name__ == "__main__":
    main()
