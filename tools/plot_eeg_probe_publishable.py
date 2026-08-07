#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publishable probe-mine figure: OOS stage fail / MMSE leftover / dual probe / phase control."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hnf.eeg_subject_diffusion import residualize, sex_to_float

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


def mmse_voltmeter_residual(df: pd.DataFrame) -> np.ndarray:
    y = df["mmse"].to_numpy(float)
    X = np.column_stack(
        [
            df["age"].to_numpy(float),
            sex_to_float(df["gender"].tolist()),
            df["theta_alpha_ratio"].to_numpy(float),
            df["bp_alpha"].to_numpy(float),
        ]
    )
    resid, _, _ = residualize(y, X)
    return resid


def main() -> None:
    v3 = pd.read_csv("outputs/eeg/probe_publishable/subjects_native_v3_trainfit_residual.csv")
    an = pd.read_csv("outputs/eeg/probe_publishable/subjects_aniso_phase_off_trainfit_residual.csv")
    on = pd.read_csv("outputs/eeg/probe_publishable/subjects_aniso_phase_on_trainfit_residual.csv")
    m = v3.merge(an[["subject_id", "D_eff_res_trainfit"]], on="subject_id", suffixes=("_v3", "_aniso"))

    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.55))
    fig.subplots_adjust(wspace=0.55, left=0.06, right=0.99, top=0.82, bottom=0.22)

    # a train vs valtest leftover by group
    ax = axes[0]
    positions = []
    data = []
    colors = []
    labels = []
    pos = 1
    for split, slabel in (("train", "train"), ("valtest", "val+test")):
        sub = v3 if split == "train" else v3[v3.split.isin(["val", "test"])]
        for g in ("HC", "FTD", "AD"):
            data.append(sub.loc[sub.clinical_group == g, "D_eff_res_trainfit"].to_numpy())
            positions.append(pos)
            colors.append(C[g])
            labels.append(f"{slabel}\n{g}" if g == "FTD" else "")
            pos += 1
        pos += 0.6
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(0, color="#888", lw=0.6, ls="--")
    ax.set_xticks([2, 6.6])
    ax.set_xticklabels(["train", "held-out"])
    ax.set_ylabel(r"probe leftover $D_\mathrm{eff}$")
    ax.set_title("a  vs diagnosis (OOS fails)", loc="left", fontsize=8.5, pad=6)

    # b MMSE residual vs leftover on valtest
    ax = axes[1]
    vt = v3[v3.split.isin(["val", "test"])].copy()
    vt["mmse_res"] = mmse_voltmeter_residual(vt)
    for g, d in vt.groupby("clinical_group"):
        ax.scatter(d.D_eff_res_trainfit, d.mmse_res, s=22, c=C[g], label=g, alpha=0.85, edgecolors="none")
    ax.axhline(0, color="#888", lw=0.5, ls="--")
    ax.axvline(0, color="#888", lw=0.5, ls="--")
    ax.set_xlabel(r"probe leftover")
    ax.set_ylabel("MMSE | voltmeter")
    ax.set_title("b  vs cognition (OOS holds)", loc="left", fontsize=8.5, pad=6)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    # c dual probe
    ax = axes[2]
    for g, d in m.groupby("clinical_group"):
        ax.scatter(
            d.D_eff_res_trainfit_v3,
            d.D_eff_res_trainfit_aniso,
            s=18,
            c=C[g],
            alpha=0.8,
            edgecolors="none",
        )
    lim = np.nanpercentile(
        np.r_[m.D_eff_res_trainfit_v3.to_numpy(), m.D_eff_res_trainfit_aniso.to_numpy()],
        [2, 98],
    )
    ax.plot(lim, lim, color="#888", lw=0.6, ls="--")
    ax.set_xlabel("v3 leftover")
    ax.set_ylabel("aniso leftover")
    ax.set_title("c  two kernels, one axis", loc="left", fontsize=8.5, pad=6)

    # d phase control MMSE ΔR²
    ax = axes[3]

    def delta(df):
        y = df.mmse.to_numpy(float)
        ok = np.isfinite(y)
        X = np.column_stack(
            [
                df.age.to_numpy(float),
                sex_to_float(df.gender),
                df.theta_alpha_ratio.to_numpy(float),
                df.bp_alpha.to_numpy(float),
            ]
        )
        pr = df.D_eff_res_trainfit.to_numpy(float)
        _, r0, _ = residualize(y[ok], X[ok])
        _, r1, _ = residualize(y[ok], np.c_[X[ok], pr[ok]])
        return r1 - r0

    names = ["v3", "aniso\nphase off", "aniso\nphase on"]
    vals = [delta(v3), delta(an), delta(on)]
    cols = ["#0B3D4A", "#1B6B93", "#9AA3A7"]
    ax.bar(np.arange(3), vals, color=cols, width=0.62)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names)
    ax.set_ylabel(r"MMSE $\Delta R^2$ | voltmeter")
    ax.set_title("d  probe-physics control", loc="left", fontsize=8.5, pad=6)

    out = Path("docs/figures/eeg")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "eeg_probe_publishable.png")
    fig.savefig(out / "eeg_probe_publishable.pdf")
    plt.close(fig)
    print(f"[fig] → {out / 'eeg_probe_publishable.png'}")


if __name__ == "__main__":
    main()
