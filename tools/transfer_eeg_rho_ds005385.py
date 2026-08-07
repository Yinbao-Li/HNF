#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen HNF ρ / D_eff transfer onto healthy-aging ds005385.

Tests marker reliability (test–retest) and age direction vs ds004504 disease.
Not an AD classification transfer claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from hnf.eeg_ckpt import load_native_checkpoint
from hnf.eeg_clinical import epoch_feature_vector, welch_band_powers
from hnf.eeg_ds005385 import iter_paired_recordings, load_edf_ahepa19
from hnf.eeg_subject_diffusion import icc_abs_agreement, pearson_r

MARKERS = (
    "rho_std",
    "rho_mean",
    "rho_p90",
    "rho_cv",
    "D_eff",
    "theta_alpha_ratio",
    "bp_alpha",
    "bp_theta",
)

DS004504_DIRECTION = {
    "rho_std": "disease_down",
    "rho_mean": "disease_down",
    "rho_p90": "disease_down",
    "rho_cv": "disease_down",
    "D_eff": "disease_up",
    "theta_alpha_ratio": "disease_up",
    "bp_alpha": "disease_down",
    "bp_theta": "disease_up",
}

CKPTS = {
    "native_v3": "outputs/eeg/adftd_hnf_native_v3/best.pt",
    "aniso_phase_off": "outputs/eeg/aniso_diffusion_ablation/phase_off/best.pt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="external_data/eeg_ds005385")
    p.add_argument("--output-dir", default="outputs/eeg/longitudinal_ds005385_rho")
    p.add_argument("--max-subjects", type=int, default=60)
    p.add_argument("--max-epochs", type=int, default=12)
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def _zscore_epochs(epochs: np.ndarray) -> np.ndarray:
    # Match EEGDataset: z-score per channel on the concatenated recording, approx via per-epoch.
    out = []
    for ep in epochs:
        mu = ep.mean(axis=1, keepdims=True)
        sd = ep.std(axis=1, keepdims=True) + 1e-6
        out.append((ep - mu) / sd)
    return np.stack(out, axis=0).astype(np.float32)


@torch.no_grad()
def session_features(
    model: torch.nn.Module,
    epochs: np.ndarray,
    *,
    sample_rate: float,
    mean_omega: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    x = torch.from_numpy(_zscore_epochs(epochs))
    acc: dict[str, list[float]] = {k: [] for k in MARKERS}
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        logits, aux = model(xb, return_aux=True)
        rho = aux["rho"].detach().cpu().numpy()
        x_np = xb.detach().cpu().numpy()
        for i in range(x_np.shape[0]):
            feats = epoch_feature_vector(
                x_np[i],
                rho[i],
                sample_rate=float(sample_rate),
                mean_omega=mean_omega,
            )
            feats["D_eff"] = 1.0 / max(float(feats["rho_std"]), 1e-6)
            bp = welch_band_powers(x_np[i], float(sample_rate))
            feats.update(bp)
            for k in MARKERS:
                v = feats.get(k, float("nan"))
                if np.isfinite(v):
                    acc[k].append(float(v))
    return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in acc.items()}


def _direction_note(marker: str, r_age: float) -> str:
    dirc = DS004504_DIRECTION.get(marker)
    if dirc is None or not np.isfinite(r_age):
        return "—"
    if dirc == "disease_up":
        return "age↑ matches disease↑" if r_age > 0 else "age trend opposite to disease↑"
    return "age↓ matches disease↓" if r_age < 0 else "age trend opposite to disease↓"


def run_ckpt(
    tag: str,
    ckpt_path: Path,
    pairs,
    *,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    model, args, arch = load_native_checkpoint(ckpt_path, device)
    sample_rate = int(args.get("sample_rate", 128))
    kparams = model.collect_kernel_params()
    mean_omega = (
        float(np.mean([v["omega"] for v in kparams.values() if "omega" in v])) if kparams else 0.0
    )
    rows = []
    n_skip = 0
    for rec1, rec2 in pairs:
        try:
            e1, _ = load_edf_ahepa19(rec1.path, target_sfreq=sample_rate, max_epochs=max_epochs)
            e2, _ = load_edf_ahepa19(rec2.path, target_sfreq=sample_rate, max_epochs=max_epochs)
            f1 = session_features(
                model, e1, sample_rate=sample_rate, mean_omega=mean_omega,
                device=device, batch_size=batch_size,
            )
            f2 = session_features(
                model, e2, sample_rate=sample_rate, mean_omega=mean_omega,
                device=device, batch_size=batch_size,
            )
        except Exception as exc:
            n_skip += 1
            print(f"[skip {tag}] {rec1.subject_id}: {exc}", flush=True)
            continue
        rows.append(
            {
                "subject_id": rec1.subject_id,
                "age_baseline": rec1.age,
                "sex": rec1.sex,
                "year_ses1": rec1.recording_year,
                "year_ses2": rec2.recording_year,
                "ses1": f1,
                "ses2": f2,
            }
        )
        print(f"[ok {tag}] {rec1.subject_id} age={rec1.age}", flush=True)

    ages = np.asarray([r["age_baseline"] for r in rows], dtype=np.float64)
    summary = {
        "checkpoint": str(ckpt_path),
        "arch": arch,
        "n_paired": len(rows),
        "n_skip": n_skip,
        "markers": {},
    }
    for k in MARKERS:
        v1 = np.asarray([r["ses1"].get(k, np.nan) for r in rows], dtype=np.float64)
        v2 = np.asarray([r["ses2"].get(k, np.nan) for r in rows], dtype=np.float64)
        r_tr = pearson_r(v1, v2)
        icc = icc_abs_agreement(v1, v2)
        r_age = pearson_r(ages, v1)
        summary["markers"][k] = {
            "test_retest_r": r_tr,
            "icc": icc,
            "mean_ses1": float(np.nanmean(v1)) if len(v1) else float("nan"),
            "mean_ses2": float(np.nanmean(v2)) if len(v2) else float("nan"),
            "mean_delta_ses2_minus_ses1": float(np.nanmean(v2 - v1)) if len(v1) else float("nan"),
            "age_corr_ses1": r_age,
            "ds004504_disease_direction": DS004504_DIRECTION.get(k),
            "age_vs_disease_note": _direction_note(k, r_age),
        }
    return summary, rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (root / "participants.tsv").is_file():
        raise SystemExit(f"Missing {root}/participants.tsv")
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pairs = list(
        iter_paired_recordings(root, max_subjects=args.max_subjects, task="EyesClosed", acq="pre")
    )
    # materialize once so both ckpts see the same subject list
    board = {"device": str(device), "n_pair_candidates": len(pairs), "models": {}}
    md = [
        "# ds005385 frozen ρ / D_eff transfer",
        "",
        "Healthy aging (no AD). Frozen ds004504 checkpoints, forward-only.",
        "Claim: marker **stability + age direction**, not diagnosis transfer.",
        "",
    ]
    all_rows = {}
    for tag, ckpt in CKPTS.items():
        path = Path(ckpt)
        if not path.is_file():
            print(f"[warn] missing {path}", flush=True)
            continue
        summary, rows = run_ckpt(
            tag,
            path,
            pairs,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        board["models"][tag] = summary
        all_rows[tag] = rows
        md += [
            f"## {tag}",
            "",
            f"checkpoint `{ckpt}`  n_paired={summary['n_paired']}  skipped={summary['n_skip']}",
            "",
            "| Marker | test–retest r | ICC | mean Δ | age corr | vs ds004504 |",
            "|--------|--------------:|----:|-------:|---------:|-------------|",
        ]
        for k, s in summary["markers"].items():
            md.append(
                f"| `{k}` | {s['test_retest_r']:.3f} | {s['icc']:.3f} | "
                f"{s['mean_delta_ses2_minus_ses1']:.4f} | {s['age_corr_ses1']:.3f} | "
                f"{s['age_vs_disease_note']} |"
            )
        md.append("")

    (out / "TRANSFER.json").write_text(json.dumps({"board": board, "rows": all_rows}, indent=2))
    md += [
        "## Interpretation template",
        "",
        "- ICC / r high → ρ is a stable person-level trait in healthy adults.",
        "- Age direction matching ds004504 disease → soft independent closure",
        "  (aging continuum), **not** an AD diagnostic claim.",
        "- If leftover ρ on ds004504 is real but ICC here is ~0, the marker is unstable.",
        "",
        "Regenerate: `PYTHONPATH=. python tools/transfer_eeg_rho_ds005385.py`",
    ]
    (out / "BOARD.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[transfer] → {out / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
