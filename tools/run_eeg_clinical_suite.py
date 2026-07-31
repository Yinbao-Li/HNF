#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEG clinical-breakthrough suite on frozen Stage-1 HNF.

Closes gaps against `.cursor/rules/eeg-clinical-standards.mdc`:
  1. Report HC / FTD / AD (class-1 = FTD; do not call it MCI)
  2. Align Age / Gender / MMSE; demography-only vs +EEG incremental value
  3. Subject-level primary endpoints
  4. FDR-controlled marker mining (train discovery → test confirmation)
  5. Clinical operating points + AD↔FTD hard differential

Usage:
  PYTHONPATH=. python tools/run_eeg_clinical_suite.py \\
    --checkpoint outputs/eeg/adftd_hnf_stage1/best.pt \\
    --device cuda --output-dir outputs/eeg/clinical_breakthrough_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from hnf.eeg_clinical import (
    ad_vs_ftd_mask,
    benjamini_hochberg,
    binary_operating_points,
    confusion_matrix,
    epoch_feature_vector,
    logistic_acc_auc,
    one_way_anova_pvalue,
    ridge_r2,
)
from hnf.eeg_dataset import (
    CLINICAL_ID_TO_LABEL as CLIN_MAP,
    CLINICAL_LABEL_TO_ID,
    EEGDataset,
)
from hnf.eeg_model import EEGHNFClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG clinical breakthrough suite")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--checkpoint", default="outputs/eeg/adftd_hnf_stage1/best.pt")
    p.add_argument("--output-dir", default="outputs/eeg/clinical_breakthrough_v1")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--max-epochs-per-split", type=int, default=0, help="0 = all")
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    arch = str(ckpt.get("arch") or a.get("arch") or "")
    sample_rate = int(a.get("sample_rate", 128))
    epoch_sec = float(a.get("epoch_sec", 10.0))
    embed_dim = int(a.get("embed_dim", 64))
    principle = str(a.get("principle", "huygens_fresnel"))
    rhythm_phase = bool(a.get("rhythm_phase", True))
    dropout = float(a.get("dropout", 0.2))
    seq_len = int(round(epoch_sec * sample_rate))

    if (
        "native" in arch
        or "eeg_hnf_native" in arch
        or "eeg_hnf_aniso" in arch
        or "aniso" in arch
        or any(k.startswith("spatial.diff_L") for k in ckpt.get("state_dict", {}))
        or any(k.startswith("theta.stack.") for k in ckpt.get("state_dict", {}))
    ):
        from hnf.eeg_native_model import EEGHNFNativeClassifier

        sd = ckpt["state_dict"]
        use_delta = any(k.startswith("delta.") for k in sd)
        segment_pool = any(k.startswith("pool_theta.") for k in sd)
        head_in = int(sd["head.0.weight"].shape[1]) if "head.0.weight" in sd else -1
        n_branches = 3 if use_delta else 2
        extras = 4 + 4  # rho_stats + band
        if head_in == embed_dim * n_branches + extras + 6:
            include_region = True
        elif head_in == embed_dim * n_branches + extras:
            include_region = False
        else:
            # Fall back: prefer matching the larger (v5) layout.
            include_region = True
        # State-dict heuristic if args missing: aniso spatial has diff_L.
        if any(k.startswith("spatial.diff_L") for k in sd):
            principle = "aniso_diffusion"
        model = EEGHNFNativeClassifier(
            n_channels=19,
            seq_len=seq_len,
            sample_rate=sample_rate,
            embed_dim=embed_dim,
            num_classes=3,
            dropout=dropout,
            principle=principle,
            use_spatial=not bool(a.get("no_spatial", False)),
            use_delta=use_delta,
            segment_pool=segment_pool,
            include_region_in_head=include_region,
            rhythm_phase=rhythm_phase,
        ).to(device)
    else:
        model = EEGHNFClassifier(
            n_channels=19,
            seq_len=seq_len,
            sample_rate=sample_rate,
            embed_dim=embed_dim,
            num_classes=3,
            dropout=dropout,
            principle=principle,
        ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, a, arch or type(model).__name__


@torch.no_grad()
def _collect_split(
    model: EEGHNFClassifier,
    split: str,
    *,
    data_dir: str,
    seed: int,
    sample_rate: int,
    epoch_sec: float,
    batch_size: int,
    device: torch.device,
    synthetic_if_missing: bool,
    max_epochs: int,
    mean_omega: float,
    swap_val_test: bool = False,
) -> dict[str, Any]:
    ds = EEGDataset(
        data_dir=data_dir,
        split=split,
        seed=seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,  # non-overlap
        synthetic_if_missing=synthetic_if_missing,
        swap_val_test=swap_val_test,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []
    n_seen = 0
    for batch in loader:
        x = batch["x"].to(device)
        logits, aux = model(x, return_aux=True)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        rho = aux["rho"].cpu().numpy()
        theta_env = aux.get("theta_env")
        alpha_env = aux.get("alpha_env")
        delta_env = aux.get("delta_env")
        band_proxy = aux.get("band_proxy")
        x_spatial = aux.get("x_spatial")
        region_energy = aux.get("region_energy")
        theta_np = theta_env.detach().cpu().numpy() if theta_env is not None else None
        alpha_np = alpha_env.detach().cpu().numpy() if alpha_env is not None else None
        delta_np = delta_env.detach().cpu().numpy() if delta_env is not None else None
        band_np = band_proxy.detach().cpu().numpy() if band_proxy is not None else None
        spatial_np = x_spatial.detach().cpu().numpy() if x_spatial is not None else None
        region_np = region_energy.detach().cpu().numpy() if region_energy is not None else None
        x_np = batch["x"].numpy()
        for i in range(x_np.shape[0]):
            feats = epoch_feature_vector(
                x_np[i],
                rho[i],
                sample_rate=float(sample_rate),
                mean_omega=mean_omega,
            )
            if theta_np is not None and alpha_np is not None:
                theta_energy = float(np.mean(theta_np[i]))
                alpha_energy = float(np.mean(alpha_np[i]))
                feats.update(
                    {
                        "hnf_theta_energy": theta_energy,
                        "hnf_alpha_energy": alpha_energy,
                        "hnf_theta_alpha_ratio": float(
                            np.log((theta_energy + 1e-8) / (alpha_energy + 1e-8))
                        ),
                    }
                )
            if delta_np is not None:
                feats["hnf_delta_energy"] = float(np.mean(delta_np[i]))
            if band_np is not None:
                for j, name in enumerate(("delta", "theta", "alpha", "beta")):
                    feats[f"hnf_band_{name}"] = float(band_np[i, j])
            if spatial_np is not None:
                feats["spatial_mix_delta"] = float(
                    np.mean(np.abs(spatial_np[i] - x_np[i]))
                    / (np.mean(np.abs(x_np[i])) + 1e-8)
                )
            if region_np is not None:
                feats.update(
                    {
                        "region_frontal": float(region_np[i, 0]),
                        "region_temporal": float(region_np[i, 1]),
                        "region_central": float(region_np[i, 2]),
                        "region_posterior": float(region_np[i, 3]),
                        "region_ft_contrast": float(region_np[i, 4]),
                        "region_pf_contrast": float(region_np[i, 5]),
                    }
                )
            rows.append(
                {
                    "split": split,
                    "subject_id": str(batch["subject_id"][i]),
                    "label": int(batch["label"][i]),
                    "clinical_group": str(batch["clinical_group"][i]),
                    "age": float(batch["age"][i]),
                    "gender": str(batch["gender"][i]),
                    "mmse": float(batch["mmse"][i]),
                    "prob_hc": float(probs[i, 0]),
                    "prob_ftd": float(probs[i, 1]),
                    "prob_ad": float(probs[i, 2]),
                    "pred": int(probs[i].argmax()),
                    **feats,
                }
            )
            n_seen += 1
            if max_epochs and n_seen >= max_epochs:
                return {"epochs": rows, "subjects": ds.subjects}
    return {"epochs": rows, "subjects": ds.subjects}


def _aggregate_subjects(epoch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in epoch_rows:
        by[r["subject_id"]].append(r)
    out: list[dict[str, Any]] = []
    feat_keys = [
        k
        for k in epoch_rows[0].keys()
        if k
        not in {
            "split",
            "subject_id",
            "label",
            "clinical_group",
            "age",
            "gender",
            "mmse",
            "pred",
        }
        and not k.startswith("prob_")
    ] if epoch_rows else []
    for sid, rows in by.items():
        probs = np.asarray([[r["prob_hc"], r["prob_ftd"], r["prob_ad"]] for r in rows], dtype=np.float64)
        mean_p = probs.mean(axis=0)
        pred = int(mean_p.argmax())
        # majority vote of epoch preds as secondary
        maj = int(np.bincount([r["pred"] for r in rows], minlength=3).argmax())
        agg = {
            "subject_id": sid,
            "split": rows[0]["split"],
            "label": int(rows[0]["label"]),
            "clinical_group": rows[0]["clinical_group"],
            "age": float(rows[0]["age"]),
            "gender": rows[0]["gender"],
            "mmse": float(rows[0]["mmse"]),
            "n_epochs": len(rows),
            "prob_hc": float(mean_p[0]),
            "prob_ftd": float(mean_p[1]),
            "prob_ad": float(mean_p[2]),
            "pred": pred,
            "pred_majority": maj,
        }
        for k in feat_keys:
            vals = np.asarray([r[k] for r in rows], dtype=np.float64)
            agg[k] = float(np.nanmean(vals))
        out.append(agg)
    return out


def _subject_metrics(subjects: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([s["label"] for s in subjects], dtype=np.int64)
    pred = np.asarray([s["pred"] for s in subjects], dtype=np.int64)
    cm = confusion_matrix(y, pred, 3)
    acc = float((y == pred).mean()) if len(y) else float("nan")
    # per-class recall
    recalls = {}
    for c, name in CLIN_MAP.items():
        mask = y == c
        recalls[name] = float((pred[mask] == c).mean()) if mask.any() else float("nan")
    # hard differential AD vs FTD among true AD∪FTD
    m = ad_vs_ftd_mask(y)
    ad_ftd = {
        "n": int(m.sum()),
        "accuracy": float((pred[m] == y[m]).mean()) if m.any() else float("nan"),
        "ad_as_ftd": int(((y == CLINICAL_LABEL_TO_ID["AD"]) & (pred == CLINICAL_LABEL_TO_ID["FTD"])).sum()),
        "ftd_as_ad": int(((y == CLINICAL_LABEL_TO_ID["FTD"]) & (pred == CLINICAL_LABEL_TO_ID["AD"])).sum()),
    }
    # binary AD vs rest / FTD vs AD scores
    score_ad = np.asarray([s["prob_ad"] for s in subjects], dtype=np.float64)
    score_ftd = np.asarray([s["prob_ftd"] for s in subjects], dtype=np.float64)
    ops = {
        "ad_vs_rest": binary_operating_points((y == CLINICAL_LABEL_TO_ID["AD"]).astype(np.int64), score_ad),
        "ftd_vs_rest": binary_operating_points((y == CLINICAL_LABEL_TO_ID["FTD"]).astype(np.int64), score_ftd),
    }
    if m.any():
        y_af = (y[m] == CLINICAL_LABEL_TO_ID["AD"]).astype(np.int64)
        # score favoring AD over FTD
        score_af = score_ad[m] / np.maximum(score_ad[m] + score_ftd[m], 1e-8)
        ops["ad_vs_ftd"] = binary_operating_points(y_af, score_af)
    return {
        "n_subjects": int(len(subjects)),
        "subject_accuracy": acc,
        "confusion_matrix_hc_ftd_ad": cm.tolist(),
        "recall": recalls,
        "ad_ftd_differential": ad_ftd,
        "operating_points": ops,
        "id_to_clinical_label": dict(CLIN_MAP),
        "note": "Class-1 is FTD (Stage-1 slot historically labeled MCI).",
    }


MARKER_KEYS = [
    "rho_mean",
    "rho_std",
    "rho_p90",
    "rho_cv",
    "omega_rho",
    "bp_delta",
    "bp_theta",
    "bp_alpha",
    "bp_beta",
    "theta_alpha_ratio",
    # EEG-native HNF readouts (absent for Stage-1 temporal port)
    "hnf_theta_energy",
    "hnf_alpha_energy",
    "hnf_theta_alpha_ratio",
    "hnf_delta_energy",
    "hnf_band_delta",
    "hnf_band_theta",
    "hnf_band_alpha",
    "hnf_band_beta",
    "spatial_mix_delta",
    "region_frontal",
    "region_temporal",
    "region_central",
    "region_posterior",
    "region_ft_contrast",
    "region_pf_contrast",
]


def _fdr_mine(subjects: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    contrasts = [
        ("HC_vs_AD", ["HC", "AD"]),
        ("HC_vs_FTD", ["HC", "FTD"]),
        ("AD_vs_FTD", ["AD", "FTD"]),
        ("HC_vs_disease", ["HC", "AD", "FTD"]),  # HC vs pooled disease via 2 groups below
    ]
    for key in MARKER_KEYS:
        for cname, groups in contrasts:
            if cname == "HC_vs_disease":
                gvals = [
                    np.asarray([s.get(key, np.nan) for s in subjects if s["clinical_group"] == "HC"], dtype=np.float64),
                    np.asarray(
                        [s.get(key, np.nan) for s in subjects if s["clinical_group"] in {"AD", "FTD"}],
                        dtype=np.float64,
                    ),
                ]
            else:
                gvals = [
                    np.asarray([s.get(key, np.nan) for s in subjects if s["clinical_group"] == g], dtype=np.float64)
                    for g in groups
                ]
            means = [float(np.nanmean(g)) if g.size else float("nan") for g in gvals]
            p = one_way_anova_pvalue(gvals)
            results.append(
                {
                    "feature": key,
                    "contrast": cname,
                    "p": p,
                    "means": means,
                    "groups": groups if cname != "HC_vs_disease" else ["HC", "disease"],
                    "n": [int(np.isfinite(g).sum()) for g in gvals],
                }
            )
    pvals = np.asarray([r["p"] for r in results], dtype=np.float64)
    # nan → 1 for FDR ranking
    p_for_fdr = np.where(np.isfinite(pvals), pvals, 1.0)
    rejected, q = benjamini_hochberg(p_for_fdr, alpha=alpha)
    for i, r in enumerate(results):
        r["q"] = float(q[i])
        r["reject_fdr"] = bool(rejected[i] and np.isfinite(pvals[i]))
    results.sort(key=lambda r: (0 if r["reject_fdr"] else 1, r["q"] if np.isfinite(r["q"]) else 9.0))
    return results


def _incremental(subjects: list[dict[str, Any]]) -> dict[str, Any]:
    """Demography-only vs demography+EEG for MMSE and AD-vs-rest."""
    age = np.asarray([s["age"] for s in subjects], dtype=np.float64)
    sex = np.asarray([1.0 if s["gender"] == "M" else 0.0 for s in subjects], dtype=np.float64)
    mmse = np.asarray([s["mmse"] for s in subjects], dtype=np.float64)
    eeg_keys = ["rho_mean", "theta_alpha_ratio", "bp_theta", "bp_alpha", "omega_rho"]
    # Include model-native responses only when available for every subject.
    native_keys = [
        "hnf_theta_energy",
        "hnf_alpha_energy",
        "hnf_theta_alpha_ratio",
        "hnf_band_theta",
        "hnf_band_alpha",
        "spatial_mix_delta",
        "region_ft_contrast",
        "region_pf_contrast",
        "hnf_delta_energy",
    ]
    if subjects and all(all(np.isfinite(s.get(k, np.nan)) for k in native_keys) for s in subjects):
        eeg_keys += native_keys
    eeg = np.asarray(
        [[s.get(k, np.nan) for k in eeg_keys] for s in subjects],
        dtype=np.float64,
    )
    demo = np.c_[age, sex]
    demo_eeg = np.c_[demo, eeg]
    # MMSE continuous
    mmse_block = {
        "demo_r2": ridge_r2(demo, mmse),
        "demo_eeg_r2": ridge_r2(demo_eeg, mmse),
    }
    mmse_block["delta_r2"] = (
        float(mmse_block["demo_eeg_r2"] - mmse_block["demo_r2"])
        if np.isfinite(mmse_block["demo_r2"]) and np.isfinite(mmse_block["demo_eeg_r2"])
        else float("nan")
    )
    # AD vs rest classification
    y_ad = np.asarray([1 if s["clinical_group"] == "AD" else 0 for s in subjects], dtype=np.float64)
    y_ftd = np.asarray([1 if s["clinical_group"] == "FTD" else 0 for s in subjects], dtype=np.float64)
    # AD vs FTD only
    m = ad_vs_ftd_mask(np.asarray([s["label"] for s in subjects]))
    clf = {
        "ad_vs_rest_demo": logistic_acc_auc(demo, y_ad),
        "ad_vs_rest_demo_eeg": logistic_acc_auc(demo_eeg, y_ad),
        "ftd_vs_rest_demo": logistic_acc_auc(demo, y_ftd),
        "ftd_vs_rest_demo_eeg": logistic_acc_auc(demo_eeg, y_ftd),
    }
    if m.any():
        y_af = (np.asarray([s["label"] for s in subjects])[m] == CLINICAL_LABEL_TO_ID["AD"]).astype(np.float64)
        clf["ad_vs_ftd_demo"] = logistic_acc_auc(demo[m], y_af)
        clf["ad_vs_ftd_demo_eeg"] = logistic_acc_auc(demo_eeg[m], y_af)
    return {"mmse": mmse_block, "classification": clf}


def _plot_confusion(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2], ["HC", "FTD", "AD"])
    ax.set_yticks([0, 1, 2], ["HC", "FTD", "AD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Subject-level confusion (clinical taxonomy)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_markers(mine: list[dict], out: Path, top_k: int = 12) -> None:
    hits = [r for r in mine if r["reject_fdr"]][:top_k] or mine[:top_k]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['feature']}\n{r['contrast']}" for r in hits]
    qs = [-np.log10(max(r["q"], 1e-12)) for r in hits]
    colors = ["#2a6f97" if r["reject_fdr"] else "#adb5bd" for r in hits]
    ax.barh(range(len(hits)), qs[::-1], color=colors[::-1])
    ax.set_yticks(range(len(hits)), labels[::-1], fontsize=8)
    ax.set_xlabel(r"$-\log_{10} q$ (BH-FDR)")
    ax.set_title("Interpretable marker contrasts (train discovery)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _checklist_progress(report: dict) -> dict[str, Any]:
    test = report.get("test_metrics", {})
    mine_test = report.get("marker_mining_test_confirmation", [])
    incr = report.get("incremental_value_all", {})
    n_fdr_train = sum(1 for r in report.get("marker_mining_train", []) if r.get("reject_fdr"))
    n_fdr_test = sum(1 for r in mine_test if r.get("reject_fdr"))
    delta = (incr.get("mmse") or {}).get("delta_r2")
    ad_ftd = (test.get("ad_ftd_differential") or {})
    return {
        "1_differential_taxonomy": {
            "status": "closed",
            "detail": "Reports HC/FTD/AD; class-1 renamed from MCI→FTD",
        },
        "2_clinical_covariates": {
            "status": "partial" if (delta is None or not np.isfinite(delta) or delta <= 0) else "progress",
            "mmse_delta_r2_demo_to_demo_eeg": delta,
            "detail": "Age/Gender/MMSE aligned; incremental EEG value measured",
        },
        "3_subject_level_endpoints": {
            "status": "progress",
            "test_n_subjects": test.get("n_subjects"),
            "subject_accuracy": test.get("subject_accuracy"),
            "detail": "Primary metrics are subject-level; N still modest (ds004504)",
        },
        "4_fdr_markers": {
            "status": "progress" if n_fdr_train else "open",
            "n_reject_train": n_fdr_train,
            "n_reject_test_confirmation": n_fdr_test,
            "detail": (
                "BH-FDR train discovery; same feature×contrast retested on "
                "held-out subjects (small-N confirmation)"
            ),
        },
        "5_decision_operating_points": {
            "status": "progress",
            "ad_vs_ftd_n": ad_ftd.get("n"),
            "ad_vs_ftd_accuracy": ad_ftd.get("accuracy"),
            "detail": "Youden + sens≥0.8 points for AD vs rest / AD vs FTD",
        },
        "6_transfer_fewshot": {
            "status": "open",
            "detail": "Deferred until marker + differential claims strengthen",
        },
        "claim_discipline": (
            "Aiming for breakthrough; not claiming clinical breakthrough yet. "
            "This report measures progress against the checklist."
        ),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, ckpt_args, arch_name = _load_model(Path(args.checkpoint), device)
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    kparams = model.collect_kernel_params()
    mean_omega = float(np.mean([v["omega"] for v in kparams.values() if "omega" in v])) if kparams else 0.0

    print(f"[clinical] device={device} arch={arch_name} ckpt={args.checkpoint}", flush=True)
    packs = {}
    swap_val_test = bool(ckpt_args.get("swap_val_test", False))
    pool_val_test = bool(ckpt_args.get("pool_val_test", False))
    if pool_val_test:
        print(
            "[clinical] pool_val_test=True — val and test both use merged val+test pool",
            flush=True,
        )
    elif swap_val_test:
        print("[clinical] swap_val_test=True (original test→val, original val→test)", flush=True)
    for split in ("train", "val", "test"):
        print(f"[clinical] collecting {split} …", flush=True)
        ds_split = split
        if pool_val_test and split in {"val", "test"}:
            ds_split = "valtest"
        packs[split] = _collect_split(
            model,
            ds_split,
            data_dir=args.data_dir,
            seed=args.seed,
            sample_rate=sample_rate,
            epoch_sec=epoch_sec,
            batch_size=args.batch_size,
            device=device,
            synthetic_if_missing=not args.no_synthetic,
            max_epochs=args.max_epochs_per_split,
            mean_omega=mean_omega,
            swap_val_test=swap_val_test and not pool_val_test,
        )

    subj = {sp: _aggregate_subjects(packs[sp]["epochs"]) for sp in packs}
    # write subject tables
    import csv

    for sp, rows in subj.items():
        if not rows:
            continue
        path = out / f"subjects_{sp}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    test_metrics = _subject_metrics(subj["test"])
    train_mine = _fdr_mine(subj["train"], args.fdr_alpha)
    # confirmatory: same features on test, report which train hits still p<alpha (descriptive)
    test_mine = _fdr_mine(subj["test"], args.fdr_alpha)
    train_hits = {(r["feature"], r["contrast"]) for r in train_mine if r["reject_fdr"]}
    test_confirm = [
        {**r, "confirmed_from_train": (r["feature"], r["contrast"]) in train_hits}
        for r in test_mine
        if (r["feature"], r["contrast"]) in train_hits
    ]

    all_subjects = subj["train"] + subj["val"] + subj["test"]
    incremental = _incremental(all_subjects)
    incremental_test = _incremental(subj["test"])

    _plot_confusion(np.asarray(test_metrics["confusion_matrix_hc_ftd_ad"]), out / "confusion_hc_ftd_ad.png")
    _plot_markers(train_mine, out / "marker_fdr_train.png")

    report = {
        "goal": "breakthrough clinical help (AD/FTD EEG) — progress against checklist",
        "checkpoint": str(args.checkpoint),
        "clinical_taxonomy": dict(CLIN_MAP),
        "taxonomy_note": (
            "Stage-1 head class-1 was trained with FTD occupying the historical MCI slot. "
            "All clinical reports here name it FTD."
        ),
        "kernel_omegas": kparams,
        "mean_omega": mean_omega,
        "cohort_counts": {
            sp: {
                "n_subjects": len(subj[sp]),
                "by_group": {
                    g: sum(1 for s in subj[sp] if s["clinical_group"] == g)
                    for g in ("HC", "FTD", "AD")
                },
            }
            for sp in subj
        },
        "test_metrics": test_metrics,
        "marker_mining_train": train_mine,
        "marker_mining_test_confirmation": test_confirm,
        "incremental_value_all": incremental,
        "incremental_value_test": incremental_test,
    }
    report["checklist_progress"] = _checklist_progress(report)

    (out / "clinical_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # short markdown for humans
    cm = test_metrics["confusion_matrix_hc_ftd_ad"]
    af = test_metrics["ad_ftd_differential"]
    hits = [r for r in train_mine if r["reject_fdr"]][:8]
    md = [
        "# EEG clinical breakthrough — progress report",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        "## Taxonomy",
        "Subject-level labels reported as **HC / FTD / AD** (class-1 = FTD, not MCI).",
        "",
        "## Test subject metrics",
        f"- n = **{test_metrics['n_subjects']}**",
        f"- subject accuracy = **{test_metrics['subject_accuracy']:.3f}**",
        f"- recalls: {test_metrics['recall']}",
        f"- AD↔FTD differential accuracy = **{af.get('accuracy')}** "
        f"(n={af.get('n')}; AD→FTD confusions={af.get('ad_as_ftd')}, FTD→AD={af.get('ftd_as_ad')})",
        "",
        "Confusion (rows=true HC/FTD/AD):",
        f"```\n{np.asarray(cm)}\n```",
        "",
        "## FDR marker hits (train discovery)",
    ]
    if not hits:
        md.append("_No BH-FDR rejections at α=0.05 on train._")
    else:
        for r in hits:
            md.append(
                f"- `{r['feature']}` / {r['contrast']}: p={r['p']:.3g}, q={r['q']:.3g}, means={r['means']}"
            )
    md += [
        "",
        f"Train FDR hits retested on held-out test: **{len(test_confirm)}**; "
        f"survive test BH-FDR: **{sum(1 for r in test_confirm if r.get('reject_fdr'))}**",
        "",
        "## Incremental value (Age+Gender → +EEG features)",
        f"- MMSE ΔR² (all subjects) = **{(incremental.get('mmse') or {}).get('delta_r2')}**",
        f"- AD vs rest AUC demo → demo+EEG = "
        f"{(incremental.get('classification') or {}).get('ad_vs_rest_demo', {}).get('auc')} → "
        f"{(incremental.get('classification') or {}).get('ad_vs_rest_demo_eeg', {}).get('auc')}",
        "",
        "## Checklist progress",
        "```json",
        json.dumps(report["checklist_progress"], indent=2),
        "```",
        "",
        "> Ambition: breakthrough clinical help. Claim: **not yet** — this is a progress board.",
    ]
    (out / "CLINICAL_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report["checklist_progress"], indent=2))
    print(f"[clinical] wrote {out/'CLINICAL_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
