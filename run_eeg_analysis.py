# -*- coding: utf-8 -*-
"""Figures for Domain II EEG analysis (omega / ROC / CM / rho)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from hnf.eeg_dataset import EEGDataset, ID_TO_LABEL, LABEL_TO_ID
from hnf.eeg_model import EEGHNFClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG analysis figures")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--checkpoint", default="outputs/eeg/adftd_hnf/best.pt")
    p.add_argument("--metrics-json", default="outputs/eeg/adftd_hnf/test_metrics.json")
    p.add_argument("--fig-dir", default="docs/figures/eeg")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-if-missing", action="store_true", default=True)
    return p.parse_args()


def _anova_pvalue(groups: list[np.ndarray]) -> float:
    """One-way ANOVA p-value (pure numpy)."""
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return float("nan")
    all_v = np.concatenate(groups)
    grand = all_v.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b = len(groups) - 1
    df_w = len(all_v) - len(groups)
    if df_w <= 0 or ss_within <= 0:
        return float("nan")
    msb = ss_between / df_b
    msw = ss_within / df_w
    f = msb / msw
    # Regularized incomplete beta approximation is heavy; report F as text and
    # a coarse p via survival of F using scipy if present.
    try:
        from scipy.stats import f as f_dist

        return float(f_dist.sf(f, df_b, df_w))
    except Exception:
        return float("nan")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    principle = str(ckpt_args.get("principle", "huygens_fresnel"))

    model = EEGHNFClassifier(
        sample_rate=sample_rate,
        seq_len=int(round(epoch_sec * sample_rate)),
        embed_dim=embed_dim,
        principle=principle,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # --- Fig1: omega distribution (model-level params; group via subject rho proxy) ---
    kparams = model.collect_kernel_params()
    omegas = [v["omega"] for v in kparams.values()]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([omegas], labels=["all layers"])
    ax.set_ylabel("ω")
    ax.set_title("Learned Huygens ω (model kernels)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_omega_boxplot.png", dpi=150)
    plt.close(fig)

    # Per-subject mean |rho| grouped by label for a group-contrast panel
    ds = EEGDataset(
        data_dir=args.data_dir,
        split="test",
        seed=args.seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,
        synthetic_if_missing=args.synthetic_if_missing,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    rho_by_label: dict[int, list[np.ndarray]] = defaultdict(list)
    omega_proxy_by_label: dict[int, list[float]] = defaultdict(list)
    all_probs: list[np.ndarray] = []
    all_y: list[int] = []

    mean_omega = float(np.mean(omegas)) if omegas else 0.0
    for batch in loader:
        x = batch["x"].to(device)
        logits, aux = model(x, return_aux=True)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        rho = aux["rho"].cpu().numpy()  # (B, T, 1)
        for i, lab in enumerate(batch["label"]):
            lab_i = int(lab)
            all_y.append(lab_i)
            all_probs.append(probs[i])
            rho_by_label[lab_i].append(rho[i, :, 0])
            # Proxy: mean rho energy as subject-level “activity” marker next to ω
            omega_proxy_by_label[lab_i].append(mean_omega * float(np.mean(rho[i])))

    # Fig1b group boxplot of omega*rho proxy + ANOVA
    fig, ax = plt.subplots(figsize=(6, 4))
    order = [0, 2]  # HC vs AD
    data = [np.asarray(omega_proxy_by_label.get(c, []), dtype=np.float64) for c in order]
    labels = [ID_TO_LABEL[c] for c in order]
    ax.boxplot([d for d in data if len(d)], labels=[lb for lb, d in zip(labels, data) if len(d)])
    p = _anova_pvalue([d for d in data if len(d)])
    ax.set_title(f"ω·⟨ρ⟩ proxy  HC vs AD  (ANOVA p={p:.3g})")
    ax.set_ylabel("ω × mean ρ")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_omega_hc_vs_ad.png", dpi=150)
    plt.close(fig)

    # --- Fig2: ROC HC vs AD ---
    y = np.asarray(all_y)
    probs = np.stack(all_probs, axis=0) if all_probs else np.zeros((0, 3))
    fig, ax = plt.subplots(figsize=(5, 5))
    mask = (y == 0) | (y == 2)
    if mask.sum() >= 2 and probs.size:
        yt = (y[mask] == 2).astype(np.int64)
        s = probs[mask, 2]
        # Simple ROC
        thr = np.unique(s)
        tprs, fprs = [0.0], [0.0]
        for t in sorted(thr, reverse=True):
            pred = (s >= t).astype(np.int64)
            tp = np.sum((pred == 1) & (yt == 1))
            fp = np.sum((pred == 1) & (yt == 0))
            fn = np.sum((pred == 0) & (yt == 1))
            tn = np.sum((pred == 0) & (yt == 0))
            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            tprs.append(tpr)
            fprs.append(fpr)
        tprs.append(1.0)
        fprs.append(1.0)
        auc = float(np.trapz(tprs, fprs))
        ax.plot(fprs, tprs, label=f"HC vs AD AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (HC vs AD)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_roc_hc_ad.png", dpi=150)
    plt.close(fig)

    # --- Fig3: confusion matrix ---
    cm = None
    metrics_path = Path(args.metrics_json)
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            cm = np.asarray(json.load(f).get("confusion_matrix"), dtype=np.int64)
    if cm is None or cm.size == 0:
        pred = probs.argmax(axis=1) if probs.size else np.zeros_like(y)
        cm = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(y, pred):
            cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), [ID_TO_LABEL[i] for i in range(3)])
    ax.set_yticks(range(3), [ID_TO_LABEL[i] for i in range(3)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Fig4: mean rho(t) HC vs AD ---
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(int(round(epoch_sec * sample_rate))) / float(sample_rate)
    for lab, color in ((0, "C0"), (2, "C3")):
        series = rho_by_label.get(lab, [])
        if not series:
            continue
        arr = np.stack(series, axis=0)
        mu = arr.mean(axis=0)
        se = arr.std(axis=0) / np.sqrt(arr.shape[0])
        ax.plot(t, mu, color=color, label=ID_TO_LABEL[lab])
        ax.fill_between(t, mu - se, mu + se, color=color, alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ρ(t)")
    ax.set_title("Mean latent density ρ(t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_rho_mean_hc_ad.png", dpi=150)
    plt.close(fig)

    meta = {
        "kernel_omegas": kparams,
        "anova_p_omega_proxy_hc_ad": p,
        "n_test_epochs": int(len(y)),
    }
    with (fig_dir / "analysis_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[EEG-analysis] figures → {fig_dir}")


if __name__ == "__main__":
    main()
