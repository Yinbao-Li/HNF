#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Journal figure: EEG classification and interpretable biomarkers (a–b).

a  Subject accuracy & macro-AUC vs literature / Braindecode baselines.
b  Subject-level effective diffusivity D_eff ∝ 1/ρ_std increases along
    HC → FTD → AD (Spearman), a PNF-native proxy linked to reduced
    medium-density dynamics / synchrony — NOT the shared global tensor D.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Restrained palette (no purple / cream AI defaults)
C_HNF = "#0B3D4A"
C_BASE = "#9AA3A7"
C_HC = "#1B6B93"
C_FTD = "#C45C26"
C_AD = "#3D5A5B"
BG = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clinical-dir",
        default="outputs/eeg/aniso_diffusion_ablation/phase_off_clinical",
    )
    p.add_argument("--fig-dir", default="docs/figures/eeg")
    p.add_argument("--stem", default="aniso_journal_classify_biomarker")
    return p.parse_args()


def _load_deff(clinical_dir: Path) -> tuple[pd.DataFrame, float, float, dict]:
    frames = [pd.read_csv(clinical_dir / f"subjects_{s}.csv") for s in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True)
    gcol = "clinical_group" if "clinical_group" in df.columns else "group"
    df = df.dropna(subset=["rho_std", gcol]).copy()
    # Effective diffusivity proxy from PNF medium density dynamics.
    # Lower ρ_std (disease) → higher D_eff: less temporally localized medium response.
    df["D_eff"] = 1.0 / (df["rho_std"].astype(float).clip(lower=1e-6))
    order = {"HC": 0, "FTD": 1, "AD": 2}
    df["stage"] = df[gcol].map(order)
    r, p = stats.spearmanr(df["D_eff"], df["stage"])
    # Kruskal–Wallis + pairwise HC vs AD
    groups = [df.loc[df[gcol] == g, "D_eff"].to_numpy() for g in ("HC", "FTD", "AD")]
    kw = stats.kruskal(*groups)
    mw = stats.mannwhitneyu(groups[0], groups[2], alternative="two-sided")
    summary = {
        "spearman_r": float(r),
        "spearman_p": float(p),
        "kruskal_p": float(kw.pvalue),
        "mw_hc_ad_p": float(mw.pvalue),
        "means": {g: float(df.loc[df[gcol] == g, "D_eff"].mean()) for g in ("HC", "FTD", "AD")},
        "ns": {g: int((df[gcol] == g).sum()) for g in ("HC", "FTD", "AD")},
        "definition": "D_eff = 1 / rho_std (subject-level; PNF medium-density dynamics)",
    }
    return df, float(r), float(p), summary


def panel_a(ax: plt.Axes) -> None:
    """Classification: subject accuracy + macro-AUC vs baselines."""
    # Canonical held-out test numbers (n=18), aniso-only comparison
    models = [
        "PNF aniso\ndiffusion",
        "EEGNetv4",
        "EEG\nConformer",
        "Deep4Net",
        "Shallow\nFBCSP",
    ]
    subject_acc = [0.833, 0.722, 0.667, 0.611, 0.556]
    macro_auc = [0.917, 0.780, 0.773, 0.811, 0.828]
    x = np.arange(len(models))
    w = 0.38
    b1 = ax.bar(
        x - w / 2,
        subject_acc,
        width=w,
        color=[C_HNF] + [C_BASE] * 4,
        label="Subject accuracy",
        edgecolor="none",
        zorder=3,
    )
    b2 = ax.bar(
        x + w / 2,
        macro_auc,
        width=w,
        color=[C_HNF] + [C_BASE] * 4,
        alpha=0.45,
        label="Macro-AUC",
        edgecolor="none",
        hatch="////",
        zorder=3,
    )
    # hatch only on AUC bars for visual separation while keeping HNF dark
    for i, bar in enumerate(b2):
        bar.set_facecolor(C_HNF if i == 0 else C_BASE)
        bar.set_alpha(0.55 if i == 0 else 0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("a  Classification vs baselines", loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.axhline(subject_acc[0], color=C_HNF, ls="--", lw=0.7, alpha=0.35, zorder=1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.7)
    # value labels on HNF
    ax.text(0 - w / 2, subject_acc[0] + 0.018, f"{subject_acc[0]:.3f}", ha="center", fontsize=7, color=C_HNF)
    ax.text(0 + w / 2, macro_auc[0] + 0.018, f"{macro_auc[0]:.3f}", ha="center", fontsize=7, color=C_HNF)


def panel_b(ax: plt.Axes, df: pd.DataFrame, spearman_p: float, summary: dict) -> None:
    """D_eff by clinical stage."""
    gcol = "clinical_group"
    order = ["HC", "FTD", "AD"]
    colors = [C_HC, C_FTD, C_AD]
    data = [df.loc[df[gcol] == g, "D_eff"].to_numpy() for g in order]
    positions = np.arange(1, 4)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#111", lw=1.2),
        whiskerprops=dict(color="#444", lw=0.8),
        capprops=dict(color="#444", lw=0.8),
        boxprops=dict(lw=0.8),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.78)

    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, data, colors):
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=18,
            color=c,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )

    means = [float(np.mean(v)) for v in data]
    ax.plot(positions, means, color="#222", lw=1.2, marker="o", ms=5, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{g}\n(n={summary['ns'][g]})" for g in order],
        fontsize=8,
    )
    ax.set_ylabel(r"Effective diffusivity  $D_{\mathrm{eff}}=1/\rho_{\mathrm{std}}$")
    ax.set_title(
        "b  Learned diffusivity proxy vs disease stage",
        loc="left",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    # Stats annotation — only claim p<0.001 if true
    p = spearman_p
    p_str = "p < 0.001" if p < 1e-3 else f"p = {p:.2e}"
    r = summary["spearman_r"]
    ax.text(
        0.98,
        0.98,
        f"Spearman ρ = {r:.2f}\n{p_str}\n(HC → FTD → AD)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#DDD", lw=0.6),
    )
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    clin = Path(args.clinical_dir)

    df, r, p, summary = _load_deff(clin)
    print(json.dumps(summary, indent=2), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.85), gridspec_kw={"wspace": 0.32})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)

    panel_a(axes[0])
    panel_b(axes[1], df, p, summary)

    fig.suptitle(
        "EEG classification and interpretable biomarkers",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    # footnote under (b) definition
    fig.text(
        0.98,
        -0.02,
        r"$D_{\mathrm{eff}}$: subject-level PNF proxy from medium density ($1/\rho_{\mathrm{std}}$);"
        " global tensor $D$ is shared. OpenNeuro ds004504.",
        ha="right",
        fontsize=7,
        color="#555",
    )

    stem = fig_dir / args.stem
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor=BG)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    p_claim = "p<0.001" if summary["spearman_p"] < 1e-3 else f"p={summary['spearman_p']:.2e}"
    meta = {
        "figure": str(stem.with_suffix(".png")),
        "panel_a": {
            "metric": ["subject_accuracy", "macro_AUC"],
            "models": ["PNF aniso", "EEGNetv4", "EEG Conformer", "Deep4Net", "ShallowFBCSP"],
            "subject_acc": [0.833, 0.722, 0.667, 0.611, 0.556],
            "macro_auc": [0.917, 0.780, 0.773, 0.811, 0.828],
            "protocol": "held-out test n=18, seed=42",
        },
        "panel_b": summary,
        "caption_suggestion": (
            "EEG classification and interpretable biomarkers. "
            "a Subject accuracy and macro-AUC versus Braindecode / literature baselines "
            "(held-out test, n=18). "
            "b Subject-level effective diffusivity D_eff = 1/rho_std "
            f"increases with disease stage HC→FTD→AD (Spearman ρ={summary['spearman_r']:.2f}, "
            f"{p_claim}), consistent with reduced PNF medium-density dynamics "
            "(and reduced neural synchrony) in Alzheimer's disease."
        ),
    }
    (stem.with_name(stem.name + "_meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[journal-fig] → {stem.with_suffix('.png')}", flush=True)
    print(f"[journal-fig] → {stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
