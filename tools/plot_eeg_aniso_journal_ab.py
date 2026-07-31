#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Journal figure: EEG subject manifold + classification + biomarker (a–c).

a  Subject router-feature PCA with LDA decision regions (HC / FTD / AD).
b  Subject accuracy & macro-AUC vs literature / Braindecode baselines.
c  Subject-level effective diffusivity D_eff ∝ 1/ρ_std along HC → FTD → AD.
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
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

from hnf.eeg_pattern_library import EEG_ROUTER_FEATURES, subject_router_features
from hnf.pattern_library import features_to_vector

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

C_PNF = "#0B3D4A"
C_BASE = "#9AA3A7"
C_HC = "#1B6B93"
C_FTD = "#C45C26"
C_AD = "#3D5A5B"
BG = "#FFFFFF"
GROUPS = ("HC", "FTD", "AD")
COLORS = {"HC": C_HC, "FTD": C_FTD, "AD": C_AD}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clinical-dir",
        default="outputs/eeg/aniso_diffusion_ablation/phase_off_clinical",
    )
    p.add_argument("--fig-dir", default="docs/figures/eeg")
    p.add_argument("--stem", default="aniso_journal_classify_biomarker")
    return p.parse_args()


def _load_subjects(clinical_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(clinical_dir / f"subjects_{split}.csv")
        d["split"] = split
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _load_deff(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    gcol = "clinical_group" if "clinical_group" in df.columns else "group"
    out = df.dropna(subset=["rho_std", gcol]).copy()
    out["D_eff"] = 1.0 / out["rho_std"].astype(float).clip(lower=1e-6)
    order = {"HC": 0, "FTD": 1, "AD": 2}
    out["stage"] = out[gcol].map(order)
    r, p = stats.spearmanr(out["D_eff"], out["stage"])
    groups = [out.loc[out[gcol] == g, "D_eff"].to_numpy() for g in GROUPS]
    kw = stats.kruskal(*groups)
    mw = stats.mannwhitneyu(groups[0], groups[2], alternative="two-sided")
    summary = {
        "spearman_r": float(r),
        "spearman_p": float(p),
        "kruskal_p": float(kw.pvalue),
        "mw_hc_ad_p": float(mw.pvalue),
        "means": {g: float(out.loc[out[gcol] == g, "D_eff"].mean()) for g in GROUPS},
        "ns": {g: int((out[gcol] == g).sum()) for g in GROUPS},
        "definition": "D_eff = 1 / rho_std (subject-level; PNF medium-density dynamics)",
    }
    return out, summary


def _pca_lda(df: pd.DataFrame) -> dict:
    """Router-feature PCA + 3-class LDA in PC1–PC2."""
    vecs = []
    for r in df.to_dict("records"):
        feat = subject_router_features(dict(r))
        vecs.append(features_to_vector(feat, EEG_ROUTER_FEATURES))
    X = np.asarray(vecs, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xz = (X - mu) / sd
    _, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    pc = Xz @ Vt[:2].T
    var = (S**2) / (S**2).sum()

    y = df["clinical_group"].astype(str).to_numpy()
    means = {g: pc[y == g].mean(axis=0) for g in GROUPS}
    Sw = np.zeros((2, 2))
    for g in GROUPS:
        d = pc[y == g] - means[g]
        Sw += d.T @ d
    Sw /= max(len(pc) - 3, 1)
    Sw_inv = np.linalg.pinv(Sw)
    priors = {g: float((y == g).mean()) for g in GROUPS}

    def scores(xy: np.ndarray) -> np.ndarray:
        out = []
        for g in GROUPS:
            w = Sw_inv @ means[g]
            b = -0.5 * means[g] @ Sw_inv @ means[g] + np.log(priors[g] + 1e-12)
            out.append(xy @ w + b)
        return np.stack(out, axis=-1)

    lda_pred = np.array(GROUPS)[scores(pc).argmax(1)]
    lda_acc = float((lda_pred == y).mean())

    # HC vs disease PC1 threshold (midpoint of HC mean and disease centroid)
    dis_c = 0.5 * (means["AD"][0] + means["FTD"][0])
    thr = 0.5 * (means["HC"][0] + dis_c)
    hc_side = pc[:, 0] >= thr
    hc_acc = float((hc_side == (y == "HC")).mean())

    loadings = {
        "PC1": [
            (EEG_ROUTER_FEATURES[j], float(Vt[0, j]))
            for j in np.argsort(-np.abs(Vt[0]))[:6]
        ],
        "PC2": [
            (EEG_ROUTER_FEATURES[j], float(Vt[1, j]))
            for j in np.argsort(-np.abs(Vt[1]))[:6]
        ],
    }
    return {
        "pc": pc,
        "y": y,
        "means": means,
        "Sw_inv": Sw_inv,
        "priors": priors,
        "scores_fn": scores,
        "thr_pc1": float(thr),
        "lda_acc": lda_acc,
        "hc_vs_disease_acc": hc_acc,
        "var": {"PC1": float(var[0]), "PC2": float(var[1]), "PC1+PC2": float(var[:2].sum())},
        "loadings": loadings,
        "subject_id": df["subject_id"].astype(str).to_numpy(),
        "pred": df["pred"].astype(int).map({0: "HC", 1: "FTD", 2: "AD"}).to_numpy()
        if "pred" in df.columns
        else lda_pred,
    }


def panel_a(ax: plt.Axes, pca: dict) -> None:
    """PCA scatter + LDA regions + PC1 HC-vs-disease threshold."""
    pc = pca["pc"]
    y = pca["y"]
    means = pca["means"]
    Sw_inv = pca["Sw_inv"]
    priors = pca["priors"]
    scores_fn = pca["scores_fn"]
    thr = pca["thr_pc1"]
    pred = pca["pred"]

    pad = 0.6
    xmin, xmax = float(pc[:, 0].min() - pad), float(pc[:, 0].max() + pad)
    ymin, ymax = float(pc[:, 1].min() - pad), float(pc[:, 1].max() + pad)
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 320), np.linspace(ymin, ymax, 320))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    Z = scores_fn(grid).argmax(axis=1).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        Z,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["#1B6B9333", "#C45C2633", "#3D5A5B33"],
        antialiased=True,
    )
    # Pairwise LDA boundaries
    for i in range(3):
        for j in range(i + 1, 3):
            gi, gj = GROUPS[i], GROUPS[j]
            wi = Sw_inv @ means[gi]
            bi = -0.5 * means[gi] @ Sw_inv @ means[gi] + np.log(priors[gi] + 1e-12)
            wj = Sw_inv @ means[gj]
            bj = -0.5 * means[gj] @ Sw_inv @ means[gj] + np.log(priors[gj] + 1e-12)
            w = wi - wj
            b = bi - bj
            if abs(w[1]) < 1e-9:
                continue
            xs = np.linspace(xmin, xmax, 240)
            ys = -(w[0] * xs + b) / w[1]
            m = (ys >= ymin) & (ys <= ymax)
            ax.plot(xs[m], ys[m], color="#222", ls="--", lw=0.85, alpha=0.65)

    ax.axvline(thr, color="#444", ls=":", lw=1.1, zorder=2)

    # class centroids
    for g in GROUPS:
        ax.scatter(
            means[g][0],
            means[g][1],
            marker="X",
            s=70,
            c=COLORS[g],
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )

    for g in GROUPS:
        m = y == g
        ax.scatter(
            pc[m, 0],
            pc[m, 1],
            s=28,
            c=COLORS[g],
            label=g,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )

    # outline model-misclassified subjects
    mis = pred != y
    if mis.any():
        ax.scatter(
            pc[mis, 0],
            pc[mis, 1],
            s=78,
            facecolors="none",
            edgecolors="#111",
            linewidths=1.1,
            zorder=5,
            label="Model error",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(f"PC1 ({100 * pca['var']['PC1']:.0f}% var)")
    ax.set_ylabel(f"PC2 ({100 * pca['var']['PC2']:.0f}% var)")
    ax.set_title(
        "a  Subject router-feature PCA with LDA decision regions",
        loc="left",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncols=2)
    ax.text(
        0.01,
        0.02,
        f"LDA acc={pca['lda_acc']:.2f}   PC1 HC↔disease thr={thr:.2f} "
        f"(acc={pca['hc_vs_disease_acc']:.2f})   n={len(y)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F7F7F7", edgecolor="#DDD", lw=0.5),
    )


def panel_b(ax: plt.Axes) -> None:
    """Classification: subject accuracy + macro-AUC vs baselines."""
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
    ax.bar(
        x - w / 2,
        subject_acc,
        width=w,
        color=[C_PNF] + [C_BASE] * 4,
        label="Subject accuracy",
        edgecolor="none",
        zorder=3,
    )
    b2 = ax.bar(
        x + w / 2,
        macro_auc,
        width=w,
        color=[C_PNF] + [C_BASE] * 4,
        alpha=0.45,
        label="Macro-AUC",
        edgecolor="none",
        hatch="////",
        zorder=3,
    )
    for i, bar in enumerate(b2):
        bar.set_facecolor(C_PNF if i == 0 else C_BASE)
        bar.set_alpha(0.55 if i == 0 else 0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("b  Classification vs baselines", loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.axhline(subject_acc[0], color=C_PNF, ls="--", lw=0.7, alpha=0.35, zorder=1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.7)
    ax.text(0 - w / 2, subject_acc[0] + 0.018, f"{subject_acc[0]:.3f}", ha="center", fontsize=7, color=C_PNF)
    ax.text(0 + w / 2, macro_auc[0] + 0.018, f"{macro_auc[0]:.3f}", ha="center", fontsize=7, color=C_PNF)


def panel_c(ax: plt.Axes, df: pd.DataFrame, summary: dict) -> None:
    """D_eff by clinical stage."""
    gcol = "clinical_group"
    data = [df.loc[df[gcol] == g, "D_eff"].to_numpy() for g in GROUPS]
    positions = np.arange(1, 4)
    colors = [COLORS[g] for g in GROUPS]

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
    ax.set_xticklabels([f"{g}\n(n={summary['ns'][g]})" for g in GROUPS], fontsize=8)
    ax.set_ylabel(r"Effective diffusivity  $D_{\mathrm{eff}}=1/\rho_{\mathrm{std}}$")
    ax.set_title(
        "c  Learned diffusivity proxy vs disease stage",
        loc="left",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    p = summary["spearman_p"]
    p_str = "p < 0.001" if p < 1e-3 else f"p = {p:.2e}"
    ax.text(
        0.98,
        0.98,
        f"Spearman ρ = {summary['spearman_r']:.2f}\n{p_str}\n(HC → FTD → AD)",
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

    df_all = _load_subjects(clin)
    df_deff, summary = _load_deff(df_all)
    pca = _pca_lda(df_all)
    print(json.dumps({"deff": summary, "pca_var": pca["var"], "lda_acc": pca["lda_acc"]}, indent=2), flush=True)

    fig = plt.figure(figsize=(10.4, 7.6))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_facecolor(BG)

    panel_a(ax_a, pca)
    panel_b(ax_b)
    panel_c(ax_c, df_deff, summary)

    fig.suptitle(
        "EEG subject manifold, classification, and interpretable biomarkers",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "a: 21 router features (ρ / band / HNF energies / soft scores), z-scored PCA + LDA regions; "
        "black rings = classifier errors. "
        r"$D_{\mathrm{eff}}=1/\rho_{\mathrm{std}}$ is a subject proxy; global tensor $D$ is shared. "
        "OpenNeuro ds004504.",
        ha="center",
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
        "layout": "a full-width top; b,c bottom row",
        "panel_a": {
            "type": "router_feature_PCA_LDA",
            "n_subjects": int(len(df_all)),
            "var": pca["var"],
            "lda_acc": pca["lda_acc"],
            "pc1_hc_vs_disease_thr": pca["thr_pc1"],
            "pc1_hc_vs_disease_acc": pca["hc_vs_disease_acc"],
            "loadings_top": pca["loadings"],
            "features": list(EEG_ROUTER_FEATURES),
        },
        "panel_b": {
            "metric": ["subject_accuracy", "macro_AUC"],
            "models": ["PNF aniso", "EEGNetv4", "EEG Conformer", "Deep4Net", "ShallowFBCSP"],
            "subject_acc": [0.833, 0.722, 0.667, 0.611, 0.556],
            "macro_auc": [0.917, 0.780, 0.773, 0.811, 0.828],
            "protocol": "held-out test n=18, seed=42",
        },
        "panel_c": summary,
        "caption_suggestion": (
            "EEG subject manifold, classification, and interpretable biomarkers. "
            "a Subject-level PCA of 21 router features with LDA decision regions "
            f"(3-class acc={pca['lda_acc']:.2f}; PC1 HC↔disease thr={pca['thr_pc1']:.2f}, "
            f"acc={pca['hc_vs_disease_acc']:.2f}; black rings = classifier errors). "
            "b Subject accuracy and macro-AUC versus Braindecode / literature baselines "
            "(held-out test, n=18). "
            "c Subject-level effective diffusivity D_eff = 1/rho_std "
            f"increases with disease stage HC→FTD→AD (Spearman ρ={summary['spearman_r']:.2f}, "
            f"{p_claim})."
        ),
    }
    (stem.with_name(stem.name + "_meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[journal-fig] → {stem.with_suffix('.png')}", flush=True)
    print(f"[journal-fig] → {stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
