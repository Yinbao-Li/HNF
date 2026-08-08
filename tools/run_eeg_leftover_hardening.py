#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B4: EEG leftover LOSO intervals + capacity-matched / shuffle nulls.

Uses frozen train-fit leftovers from probe_publishable boards (matches Domain II).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
import sys

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.eeg_subject_diffusion import residualize, sex_to_float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--residual-csv",
        default="outputs/eeg/probe_publishable/subjects_native_v3_trainfit_residual.csv",
    )
    p.add_argument("--output-dir", default="outputs/eeg/leftover_hardening")
    p.add_argument("--n-null", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def covariates(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["age"].to_numpy(float),
            sex_to_float(df["gender"].tolist()),
            df["theta_alpha_ratio"].to_numpy(float),
            df["bp_alpha"].to_numpy(float),
        ]
    )


def corr(a, b, method="spearman"):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan"), float("nan"), int(m.sum())
    if method == "pearson":
        r, p = stats.pearsonr(a[m], b[m])
    else:
        r, p = stats.spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.residual_csv)
    X = covariates(df)
    tr = df["split"].eq("train").to_numpy()
    ho = df["split"].isin(["val", "test"]).to_numpy()
    leftover = df["D_eff_res_trainfit"].to_numpy(float)
    mmse = df["mmse"].to_numpy(float)
    _, _, b_m = residualize(mmse[tr], X[tr])
    mmse_res = mmse - np.column_stack([np.ones(len(df)), X]) @ b_m

    obs = {
        "spearman": corr(leftover[ho], mmse_res[ho], "spearman"),
        "pearson": corr(leftover[ho], mmse_res[ho], "pearson"),
        "spearman_all": corr(leftover, mmse_res, "spearman"),
        "pearson_test_only": corr(
            leftover[df["split"].eq("test").to_numpy()],
            mmse_res[df["split"].eq("test").to_numpy()],
            "pearson",
        ),
    }

    # LOSO on held-out pool using frozen leftover
    loso_sp, loso_pe = [], []
    ho_idx = np.where(ho)[0]
    for i in ho_idx:
        mask = ho.copy()
        mask[i] = False
        r, _, _ = corr(leftover[mask], mmse_res[mask], "spearman")
        rp, _, _ = corr(leftover[mask], mmse_res[mask], "pearson")
        if np.isfinite(r):
            loso_sp.append(r)
        if np.isfinite(rp):
            loso_pe.append(rp)
    loso_sp = np.asarray(loso_sp, float)
    loso_pe = np.asarray(loso_pe, float)

    def boot_ci(x):
        boots = [float(np.median(rng.choice(x, size=x.size, replace=True))) for _ in range(1000)]
        return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # Nulls: shuffle leftover on train-fit board by permuting D_eff_res among held-out
    # and capacity-matched random residual with same df
    r_obs = obs["spearman"][0]
    null_shuf, null_cap = [], []
    y = df["D_eff"].to_numpy(float)
    for _ in range(int(args.n_null)):
        y_s = y.copy()
        y_s[tr] = rng.permutation(y_s[tr])
        _, _, beta = residualize(y_s[tr], X[tr])
        left_s = y_s - np.column_stack([np.ones(len(df)), X]) @ beta
        r, _, _ = corr(left_s[ho], mmse_res[ho], "spearman")
        if np.isfinite(r):
            null_shuf.append(r)
        R = rng.standard_normal((len(df), X.shape[1]))
        _, _, beta_r = residualize(y[tr], R[tr])
        left_r = y - np.column_stack([np.ones(len(df)), R]) @ beta_r
        r2, _, _ = corr(left_r[ho], mmse_res[ho], "spearman")
        if np.isfinite(r2):
            null_cap.append(abs(r2))
    null_shuf = np.asarray(null_shuf, float)
    null_cap = np.asarray(null_cap, float)

    def p_ge(null_abs_or_signed, obs_r, signed=True):
        if signed:
            return float((np.sum(np.abs(null_abs_or_signed) >= abs(obs_r)) + 1) / (len(null_abs_or_signed) + 1))
        return float((np.sum(null_abs_or_signed >= abs(obs_r)) + 1) / (len(null_abs_or_signed) + 1))

    report = {
        "residual_csv": str(args.residual_csv),
        "n_heldout": int(ho.sum()),
        "observed": {
            k: {"r": v[0], "p": v[1], "n": v[2]} for k, v in obs.items()
        },
        "loso_spearman": {
            "n_folds": int(loso_sp.size),
            "median_r": float(np.median(loso_sp)),
            "p10": float(np.percentile(loso_sp, 10)),
            "p90": float(np.percentile(loso_sp, 90)),
            "ci95_median_bootstrap": boot_ci(loso_sp),
        },
        "loso_pearson": {
            "n_folds": int(loso_pe.size),
            "median_r": float(np.median(loso_pe)),
            "p10": float(np.percentile(loso_pe, 10)),
            "p90": float(np.percentile(loso_pe, 90)),
            "ci95_median_bootstrap": boot_ci(loso_pe),
        },
        "null_train_shuffle_Deff": {
            "n": int(null_shuf.size),
            "mean_abs_r": float(np.mean(np.abs(null_shuf))),
            "p_ge_obs_spearman": p_ge(null_shuf, r_obs, signed=True),
        },
        "null_capacity_matched_random_features": {
            "n": int(null_cap.size),
            "mean_abs_r": float(np.mean(null_cap)),
            "p_ge_obs_spearman": p_ge(null_cap, r_obs, signed=False),
        },
    }
    (out / "LEFTOVER_HARDENING.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", out / "LEFTOVER_HARDENING.json")


if __name__ == "__main__":
    main()
