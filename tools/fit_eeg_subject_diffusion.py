#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-subject anisotropic D from resting EEG covariance (ds004504).

Global checkpoint D is not a biomarker. Here each subject gets D, τ, regional
coupling, then residuals vs age+sex+θ/α, plus literature atrophy-template match.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import DataLoader

from hnf.eeg_clinical import welch_band_powers
from hnf.eeg_dataset import EEGDataset
from hnf.eeg_subject_diffusion import (
    ATROPHY_TEMPLATES,
    atrophy_template_scores,
    fit_subject_diffusion,
    residualize,
    sex_to_float,
    spearman_r,
)

GROUPS = ("HC", "FTD", "AD")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--output-dir", default="outputs/eeg/subject_diffusion")
    p.add_argument("--max-epochs-per-subject", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clinical-join", default="outputs/eeg/clinical_breakthrough_native_v3")
    return p.parse_args()


def collect_subject_epochs(
    data_dir: str,
    *,
    seed: int,
    max_epochs: int,
) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        ds = EEGDataset(
            data_dir=data_dir,
            split=split,
            seed=seed,
            sample_rate=128,
            epoch_sec=10.0,
            stride_sec=10.0,
            synthetic_if_missing=False,
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        counts: dict[str, int] = defaultdict(int)
        for batch in loader:
            sid = str(batch["subject_id"][0])
            if counts[sid] >= max_epochs:
                continue
            x = batch["x"][0].numpy()
            rec = by.setdefault(
                sid,
                {
                    "subject_id": sid,
                    "split": split,
                    "clinical_group": str(batch["clinical_group"][0]),
                    "age": float(batch["age"][0]),
                    "gender": str(batch["gender"][0]),
                    "mmse": float(batch["mmse"][0]),
                    "epochs": [],
                },
            )
            rec["epochs"].append(x)
            counts[sid] += 1
    return by


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("[subject-D] loading epochs …", flush=True)
    by = collect_subject_epochs(
        args.data_dir, seed=args.seed, max_epochs=args.max_epochs_per_subject
    )
    rows = []
    for i, (sid, rec) in enumerate(sorted(by.items())):
        epochs = np.stack(rec["epochs"], axis=0)
        # classical spectra on same epochs (control topography)
        bp_acc = defaultdict(list)
        for ep in epochs:
            bp = welch_band_powers(ep, 128.0)
            for k, v in bp.items():
                if np.isfinite(v):
                    bp_acc[k].append(float(v))
        fit = fit_subject_diffusion(epochs, maxiter=100)
        tmpl = atrophy_template_scores(
            {k: float(fit[k]) for k in fit if str(k).startswith("couple_")}
        )
        row = {
            "subject_id": sid,
            "split": rec["split"],
            "clinical_group": rec["clinical_group"],
            "age": rec["age"],
            "gender": rec["gender"],
            "mmse": rec["mmse"],
            "n_epochs": int(epochs.shape[0]),
            **{k: fit[k] for k in fit if k != "D_flat"},
            **tmpl,
        }
        for k, vs in bp_acc.items():
            row[k] = float(np.mean(vs))
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[subject-D] {i+1}/{len(by)} {sid} {rec['clinical_group']} "
                f"fit_r={fit['fit_r']:.2f} FA={fit['D_fa']:.2f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    # join v3 leftover if present
    join_path = Path(args.clinical_join)
    clin_frames = []
    for split in ("train", "val", "test"):
        p = join_path / f"subjects_{split}.csv"
        if p.is_file():
            clin_frames.append(pd.read_csv(p))
    if clin_frames:
        clin = pd.concat(clin_frames, ignore_index=True)
        keep = [
            c
            for c in (
                "subject_id",
                "rho_std",
                "rho_mean",
                "theta_alpha_ratio",
                "bp_alpha",
                "region_frontal",
                "region_temporal",
                "region_posterior",
                "region_ft_contrast",
                "region_pf_contrast",
            )
            if c in clin.columns
        ]
        df = df.merge(clin[keep], on="subject_id", how="left", suffixes=("", "_clin"))
        if "theta_alpha_ratio_clin" in df.columns and "theta_alpha_ratio" not in df.columns:
            df["theta_alpha_ratio"] = df["theta_alpha_ratio_clin"]
        if "bp_alpha_clin" in df.columns:
            df["bp_alpha"] = df["bp_alpha"].fillna(df["bp_alpha_clin"])

    if "theta_alpha_ratio" not in df.columns:
        df["theta_alpha_ratio"] = df.get("bp_theta", np.nan) - df.get("bp_alpha", np.nan)

    X = np.column_stack(
        [
            df["age"].to_numpy(dtype=np.float64),
            sex_to_float(df["gender"].tolist()),
            df["theta_alpha_ratio"].to_numpy(dtype=np.float64),
            df["bp_alpha"].to_numpy(dtype=np.float64),
        ]
    )
    stage = df["clinical_group"].map({"HC": 0, "FTD": 1, "AD": 2}).to_numpy(dtype=np.float64)
    groups = df["clinical_group"].astype(str).to_numpy()

    metrics = [
        "D_trace",
        "D_aniso",
        "D_fa",
        "tau",
        "fit_r",
        "couple_frontal",
        "couple_temporal",
        "couple_posterior",
        "couple_pf_contrast",
        "couple_ft_contrast",
        "tmpl_AD",
        "tmpl_FTD",
        "tmpl_ftd_minus_ad",
    ]
    board = {
        "n": int(len(df)),
        "counts": {g: int((groups == g).sum()) for g in GROUPS},
        "templates": ATROPHY_TEMPLATES,
        "markers": {},
    }
    for col in metrics:
        if col not in df.columns:
            continue
        y = df[col].to_numpy(dtype=np.float64)
        r_raw, p_raw = spearman_r(y, stage)
        resid, r2, _ = residualize(y, X)
        df[f"{col}_res"] = resid
        r, p = spearman_r(resid, stage)
        gvals = [resid[groups == g] for g in GROUPS]
        gvals = [v[np.isfinite(v)] for v in gvals]
        mw_ad = (
            stats.mannwhitneyu(gvals[0], gvals[2], alternative="two-sided")
            if min(len(gvals[0]), len(gvals[2])) >= 3
            else None
        )
        mw_af = (
            stats.mannwhitneyu(gvals[1], gvals[2], alternative="two-sided")
            if min(len(gvals[1]), len(gvals[2])) >= 3
            else None
        )
        board["markers"][col] = {
            "raw_vs_stage_r": r_raw,
            "raw_p": p_raw,
            "structure_R2": r2,
            "leftover_r": r,
            "leftover_p": p,
            "mw_hc_ad_p": float(mw_ad.pvalue) if mw_ad else float("nan"),
            "mw_ftd_ad_p": float(mw_af.pvalue) if mw_af else float("nan"),
            "means": {g: float(np.nanmean(y[groups == g])) for g in GROUPS},
            "means_res": {g: float(np.nanmean(resid[groups == g])) for g in GROUPS},
        }

    df.to_csv(out / "subject_diffusion.csv", index=False)
    (out / "DIFFUSION_BOARD.json").write_text(json.dumps(board, indent=2))

    md = [
        "# Subject-level anisotropic diffusion (ds004504)",
        "",
        "Each subject: fit \(D=LL^\\top\), \(\\tau\) so the diffusion Green kernel",
        "matches off-diagonal channel correlation. Literature atrophy templates are",
        "**rank maps, not same-subject MRI**.",
        "",
        f"n={board['n']}  {board['counts']}",
        "",
        "| marker | vs stage r | p | structure R² | leftover r | leftover p | FTD vs AD p |",
        "|--------|-----------:|--:|-------------:|-----------:|-----------:|------------:|",
    ]
    for col, s in board["markers"].items():
        md.append(
            f"| `{col}` | {s['raw_vs_stage_r']:.3f} | {s['raw_p']:.2e} | "
            f"{s['structure_R2']:.3f} | {s['leftover_r']:.3f} | {s['leftover_p']:.2e} | "
            f"{s['mw_ftd_ad_p']:.2e} |"
        )
    md += [
        "",
        "## Template prediction",
        "",
        "- `tmpl_ftd_minus_ad`: FTD-like minus AD-like match of the **decoupling** profile.",
        "- Pre-registered: FTD > AD on this delta after residualization.",
        "",
        "Regenerate: `PYTHONPATH=. python tools/fit_eeg_subject_diffusion.py`",
    ]
    (out / "BOARD.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[subject-D] → {out / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
