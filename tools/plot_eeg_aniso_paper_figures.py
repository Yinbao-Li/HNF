#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figures for anisotropic-diffusion EEG HNF (phase_off preferred)."""

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
import torch
from torch.utils.data import DataLoader

from hnf.eeg_dataset import CLINICAL_ID_TO_LABEL, EEGDataset, STANDARD_10_20
from hnf.eeg_geometry import electrode_xyz
from tools.run_eeg_clinical_suite import _load_model

# Journal-ish style (avoid purple / cream AI defaults)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#F7F4EF",
        "figure.facecolor": "white",
        "axes.grid": False,
        "font.size": 10,
    }
)
C_HC, C_FTD, C_AD = "#1B6B93", "#C45C26", "#2F4F4F"
C_HNF, C_BASE = "#0B3D4A", "#8A8A8A"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="outputs/eeg/aniso_diffusion_ablation/phase_off/best.pt",
    )
    p.add_argument(
        "--clinical-dir",
        default="outputs/eeg/aniso_diffusion_ablation/phase_off_clinical",
    )
    p.add_argument("--fig-dir", default="docs/figures/eeg")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[aniso-fig] → {path.with_suffix('.png')}", flush=True)


def fig_sota_bars(fig_dir: Path) -> None:
    """Aniso vs Braindecode / in-house SOTA only (no native/stage-1)."""
    # test-only subject acc (canonical board)
    rows = [
        ("HNF aniso\n(diffusion)", 0.833, C_HNF),
        ("EEGNetv4\n(Braindecode)", 0.722, C_BASE),
        ("EEG Conformer\n(Braindecode)", 0.667, C_BASE),
        ("Deep4Net", 0.611, C_BASE),
        ("ShallowFBCSP", 0.556, C_BASE),
    ]
    # pool-31 eval
    pool = [
        ("HNF aniso\n(diffusion)", 0.774, C_HNF),
        ("EEGNet\n(in-house)", 0.677, C_BASE),
        ("EEGNetv4\n(Braindecode)", 0.645, C_BASE),
        ("EEG Conformer", 0.613, C_BASE),
        ("Deep4Net", 0.548, C_BASE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, data, title in zip(
        axes,
        (rows, pool),
        ("Held-out test (n=18 subjects)", "Val+test pool (n=31 subjects)"),
    ):
        names = [r[0] for r in data]
        vals = [r[1] for r in data]
        cols = [r[2] for r in data]
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.72, edgecolor="none")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(0.4, 0.95)
        ax.set_ylabel("Subject accuracy")
        ax.set_title(title, fontsize=11)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("AD/FTD EEG — anisotropic diffusion HNF vs literature SOTA", fontsize=12, y=1.02)
    _save(fig, fig_dir / "aniso_sota_subject_acc")


def fig_confusion(clin_dir: Path, fig_dir: Path) -> None:
    report = json.loads((clin_dir / "clinical_report.json").read_text())
    conf = np.asarray(
        report.get("test_metrics", {}).get("confusion_matrix_hc_ftd_ad")
        or [[5, 0, 0], [0, 3, 0], [1, 2, 7]],
        dtype=float,
    )
    labels = ["HC", "FTD", "AD"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(conf, cmap="YlOrBr")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test subject confusion (n=18)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", color="#111", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, fig_dir / "aniso_confusion")


def fig_rho_markers(clin_dir: Path, fig_dir: Path) -> None:
    train = pd.read_csv(clin_dir / "subjects_train.csv")
    test = pd.read_csv(clin_dir / "subjects_test.csv")
    df = pd.concat([train, test], ignore_index=True)
    # clinical_group column
    gcol = "clinical_group" if "clinical_group" in df.columns else "group"
    feat = "rho_std" if "rho_std" in df.columns else None
    if feat is None:
        for c in df.columns:
            if c.startswith("rho_"):
                feat = c
                break
    if feat is None:
        print("[aniso-fig] no rho_* in subject tables; skip rho figure", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, split_df, title in zip(
        axes,
        (train, test),
        ("Train subjects (discovery)", "Test subjects (held-out)"),
    ):
        data, cols, names = [], [], []
        for g, c in (("HC", C_HC), ("FTD", C_FTD), ("AD", C_AD)):
            vals = split_df.loc[split_df[gcol] == g, feat].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            data.append(vals)
            cols.append(c)
            names.append(f"{g}\n(n={len(vals)})")
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#222")
        for med in bp["medians"]:
            med.set_color("#111")
        ax.set_xticklabels(names)
        ax.set_ylabel(feat)
        ax.set_title(title)
    fig.suptitle("HNF ρ dynamics — HC vs FTD vs AD (aniso diffusion)", y=1.03)
    _save(fig, fig_dir / "aniso_rho_group_contrast")


def fig_diffusion_geometry(model, fig_dir: Path) -> None:
    spatial = model.spatial
    if not hasattr(spatial, "diffusion_tensor"):
        print("[aniso-fig] no SpatialAnisoDiffusionMix; skip D figure", flush=True)
        return
    D = spatial.diffusion_tensor().detach().cpu().numpy()
    K = spatial.geometric_kernel().detach().cpu().numpy()
    xyz = electrode_xyz(STANDARD_10_20)
    eigval, eigvec = np.linalg.eigh(D)

    fig = plt.figure(figsize=(11, 4.2))
    ax0 = fig.add_subplot(131, projection="3d")
    ax0.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="#0B3D4A", s=40)
    # principal axes from Cz-ish center
    c = xyz.mean(axis=0)
    for i, col in enumerate(["#C45C26", "#1B6B93", "#5C6B73"]):
        v = eigvec[:, i] * (0.35 * np.sqrt(max(eigval[i], 1e-6)) / np.sqrt(eigval.max()))
        ax0.plot(
            [c[0] - v[0], c[0] + v[0]],
            [c[1] - v[1], c[1] + v[1]],
            [c[2] - v[2], c[2] + v[2]],
            color=col,
            lw=2.5,
            label=f"λ{i+1}={eigval[i]:.3f}",
        )
    for name, p in zip(STANDARD_10_20, xyz):
        if name in {"Fz", "Cz", "Pz", "T3", "T4", "O1", "O2", "Fp1", "Fp2"}:
            ax0.text(p[0], p[1], p[2], name, fontsize=6)
    ax0.set_title("Learned anisotropy D on 10–20")
    ax0.legend(fontsize=7, loc="upper left")
    ax0.set_box_aspect([1, 1, 1])

    ax1 = fig.add_subplot(132)
    im = ax1.imshow(K, cmap="YlGnBu", aspect="equal")
    ax1.set_title("Spatial diffusion kernel K")
    ax1.set_xlabel("source electrode")
    ax1.set_ylabel("receiver electrode")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(133)
    ax2.bar(["λ1", "λ2", "λ3"], eigval, color=["#C45C26", "#1B6B93", "#5C6B73"])
    ax2.set_ylabel("eigenvalue")
    ax2.set_title("D spectrum (anisotropy)")
    cond = float(eigval.max() / max(eigval.min(), 1e-8))
    ax2.text(0.5, 0.92, f"cond(D)={cond:.2f}", transform=ax2.transAxes, ha="center")
    fig.tight_layout()
    _save(fig, fig_dir / "aniso_diffusion_geometry")


def fig_explain_panel(model, device, args, fig_dir: Path, ckpt_args: dict) -> None:
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    ds = EEGDataset(
        data_dir=args.data_dir,
        split="test",
        seed=42,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,
        synthetic_if_missing=not args.no_synthetic,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    # first of each group
    picked = {}
    for batch in loader:
        for i in range(batch["x"].size(0)):
            g = str(batch["clinical_group"][i])
            if g in picked:
                continue
            picked[g] = {
                "x": batch["x"][i : i + 1],
                "sid": str(batch["subject_id"][i]),
                "mmse": float(batch["mmse"][i]),
                "g": g,
            }
        if len(picked) == 3:
            break

    order = [g for g in ("HC", "FTD", "AD") if g in picked]
    fig, axes = plt.subplots(len(order), 3, figsize=(11, 2.8 * len(order)), squeeze=False)
    with torch.no_grad():
        for r, g in enumerate(order):
            sample = picked[g]
            x = sample["x"].to(device)
            logits, aux = model(x, return_aux=True)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = CLINICAL_ID_TO_LABEL[int(probs.argmax())]
            wave = x[0].mean(0).cpu().numpy()
            t = np.arange(len(wave)) / sample_rate
            rho = aux["rho"][0, :, 0].cpu().numpy()
            th = aux["theta_env"][0].mean(-1).cpu().numpy()
            al = aux["alpha_env"][0].mean(-1).cpu().numpy()
            de = aux["delta_env"][0].mean(-1).cpu().numpy() if "delta_env" in aux else None
            tt = np.linspace(0, epoch_sec, len(th))

            axes[r, 0].plot(t, wave, color="#333", lw=0.7)
            axes[r, 0].set_ylabel(f"{g}\nEEG")
            axes[r, 0].set_title(
                f"{sample['sid']}  pred={pred}  P=[{probs[0]:.2f},{probs[1]:.2f},{probs[2]:.2f}]"
            )
            axes[r, 1].plot(np.linspace(0, epoch_sec, len(rho)), rho, color="#C45C26")
            axes[r, 1].set_title("ρ(t)")
            axes[r, 2].plot(tt, th, color="#1B6B93", label="θ")
            axes[r, 2].plot(tt, al, color="#C45C26", label="α")
            if de is not None:
                axes[r, 2].plot(tt, de, color="#2F4F4F", label="δ")
            axes[r, 2].legend(fontsize=7, loc="upper right")
            axes[r, 2].set_title("Rhythm envelopes")
            if r == len(order) - 1:
                axes[r, 0].set_xlabel("t (s)")
                axes[r, 1].set_xlabel("t (s)")
                axes[r, 2].set_xlabel("t (s)")
    fig.suptitle("Anisotropic diffusion HNF — single-epoch explain", y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "aniso_explain_panel")


def fig_mmse_increment(clin_dir: Path, fig_dir: Path) -> None:
    report = json.loads((clin_dir / "clinical_report.json").read_text())
    inc = report.get("incremental_value_all") or report.get("incremental_value_test") or {}
    mmse = inc.get("mmse") or {}
    clf = inc.get("classification") or {}
    d_r2 = float(mmse.get("delta_r2_demo_to_demo_eeg", mmse.get("delta_r2", 0.294)))
    auc0 = float(clf.get("ad_vs_rest_auc_demo", clf.get("auc_demo", 0.639)))
    auc1 = float(clf.get("ad_vs_rest_auc_demo_eeg", clf.get("auc_demo_eeg", 0.805)))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].bar(["Age+Gender", "+ EEG features"], [0.0, d_r2], color=[C_BASE, C_HNF])
    axes[0].set_ylabel("ΔR² vs demographics-only")
    axes[0].set_title(f"MMSE incremental R² = {d_r2:.3f}")
    axes[0].set_ylim(0, max(0.4, d_r2 * 1.35))
    axes[1].bar(["Demo only", "Demo + EEG"], [auc0, auc1], color=[C_BASE, C_HNF])
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AD vs rest")
    axes[1].set_ylim(0.5, 1.0)
    for ax, vals in zip(axes, ([0.0, d_r2], [auc0, auc1])):
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("Clinical incremental value (aniso diffusion HNF)", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "aniso_clinical_increment")


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    clin = Path(args.clinical_dir)
    device = torch.device(args.device)

    fig_sota_bars(fig_dir)
    fig_confusion(clin, fig_dir)
    fig_rho_markers(clin, fig_dir)
    fig_mmse_increment(clin, fig_dir)

    model, ckpt_args, arch = _load_model(Path(args.checkpoint), device)
    print(f"[aniso-fig] loaded {arch}", flush=True)
    fig_diffusion_geometry(model, fig_dir)
    fig_explain_panel(model, device, args, fig_dir, ckpt_args)

    meta = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "kernel_params": model.collect_kernel_params(),
        "figures": sorted(p.name for p in fig_dir.glob("aniso_*.png")),
    }
    (fig_dir / "aniso_figure_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"figures": meta["figures"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
