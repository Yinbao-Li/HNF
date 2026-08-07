#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structure-residual board for Domain-II EEG (seismic β_res analogue).

Residualize ρ / D_eff against age + sex + classical θ/α, then jackknife.
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
import pandas as pd
from scipy import stats

from hnf.eeg_subject_diffusion import residualize, sex_to_float, spearman_r

GROUPS = ("HC", "FTD", "AD")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--v3-dir",
        default="outputs/eeg/clinical_breakthrough_native_v3",
    )
    p.add_argument(
        "--aniso-dir",
        default="outputs/eeg/aniso_diffusion_ablation/phase_off_clinical",
    )
    p.add_argument("--output-dir", default="outputs/eeg/structure_residual")
    return p.parse_args()


def load_subjects(clinical_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        path = clinical_dir / f"subjects_{split}.csv"
        if not path.is_file():
            continue
        d = pd.read_csv(path)
        d["split"] = split
        frames.append(d)
    if not frames:
        raise SystemExit(f"No subjects_*.csv under {clinical_dir}")
    return pd.concat(frames, ignore_index=True)


def _covariates(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["age"].to_numpy(dtype=np.float64),
            sex_to_float(df["gender"].tolist()),
            df["theta_alpha_ratio"].to_numpy(dtype=np.float64),
            df["bp_alpha"].to_numpy(dtype=np.float64),
        ]
    )


def _stage(df: pd.DataFrame) -> np.ndarray:
    return df["clinical_group"].map({"HC": 0, "FTD": 1, "AD": 2}).to_numpy(dtype=np.float64)


def _group_means(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    out = {}
    for g in GROUPS:
        m = groups == g
        out[g] = float(np.nanmean(values[m])) if m.any() else float("nan")
    return out


def analyze_one(name: str, df: pd.DataFrame) -> dict:
    df = df.copy()
    df["D_eff"] = 1.0 / np.clip(df["rho_std"].to_numpy(dtype=np.float64), 1e-6, None)
    X = _covariates(df)
    stage = _stage(df)
    groups = df["clinical_group"].astype(str).to_numpy()
    report: dict = {
        "name": name,
        "n": int(len(df)),
        "counts": {g: int((groups == g).sum()) for g in GROUPS},
        "markers": {},
        "mmse": {},
        "jackknife": {},
    }
    for col in ("D_eff", "rho_std", "rho_mean", "theta_alpha_ratio", "bp_alpha"):
        if col not in df.columns:
            continue
        r, p = spearman_r(df[col].to_numpy(), stage)
        report["markers"][col] = {"spearman_vs_stage_r": r, "p": p}

    for col in ("D_eff", "rho_std", "rho_mean"):
        y = df[col].to_numpy(dtype=np.float64)
        resid, r2, _ = residualize(y, X)
        df[f"{col}_res"] = resid
        r, p = spearman_r(resid, stage)
        stage_res, _, _ = residualize(stage, X)
        rp, pp = spearman_r(resid, stage_res)
        gvals = [resid[groups == g] for g in GROUPS]
        gvals = [v[np.isfinite(v)] for v in gvals]
        kw = stats.kruskal(*gvals) if all(len(v) >= 2 for v in gvals) else None
        mw_ad = (
            stats.mannwhitneyu(gvals[0], gvals[2], alternative="two-sided")
            if len(gvals[0]) and len(gvals[2])
            else None
        )
        mw_af = (
            stats.mannwhitneyu(gvals[1], gvals[2], alternative="two-sided")
            if len(gvals[1]) and len(gvals[2])
            else None
        )
        report["markers"][f"{col}_residual"] = {
            "structure_R2": r2,
            "leftover_vs_stage_r": r,
            "leftover_vs_stage_p": p,
            "partial_vs_stage_res_r": rp,
            "partial_vs_stage_res_p": pp,
            "kruskal_p": float(kw.pvalue) if kw is not None else float("nan"),
            "mw_hc_ad_p": float(mw_ad.pvalue) if mw_ad is not None else float("nan"),
            "mw_ftd_ad_p": float(mw_af.pvalue) if mw_af is not None else float("nan"),
            "means_raw": _group_means(y, groups),
            "means_residual": _group_means(resid, groups),
        }

    mmse = df["mmse"].to_numpy(dtype=np.float64)
    for mask_name, mask in (
        ("all", np.isfinite(mmse)),
        ("patients", np.isfinite(mmse) & (groups != "HC")),
    ):
        if int(mask.sum()) < 12:
            continue
        y = mmse[mask]
        X0 = X[mask]
        _, r20, _ = residualize(y, X0)
        block = {"n": int(mask.sum()), "demo_spec_R2": r20, "increments": {}}
        for col in ("D_eff", "rho_std", "rho_mean"):
            X1 = np.column_stack([X0, df[col].to_numpy(dtype=np.float64)[mask]])
            _, r21, _ = residualize(y, X1)
            block["increments"][col] = {"R2": r21, "delta_R2": float(r21 - r20)}
        report["mmse"][mask_name] = block

    # LOSO jackknife on D_eff residual Spearman
    y = df["D_eff"].to_numpy(dtype=np.float64)
    n = len(df)
    rs = []
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        resid_i, _, beta = residualize(y[keep], X[keep])
        # apply beta to kept rows only (already have resid_i on keep mask positions)
        r, _ = spearman_r(resid_i, stage[keep])
        if np.isfinite(r):
            rs.append(r)
    rs = np.asarray(rs, dtype=np.float64)
    report["jackknife"]["D_eff_res_spearman"] = {
        "n_folds": int(rs.size),
        "median_r": float(np.median(rs)),
        "mean_r": float(np.mean(rs)),
        "min_r": float(np.min(rs)),
        "max_r": float(np.max(rs)),
        "frac_positive": float(np.mean(rs > 0)),
        "frac_r_gt_0p15": float(np.mean(rs > 0.15)),
    }
    # MMSE ΔR² jackknife (all subjects with MMSE)
    mask = np.isfinite(mmse)
    idx = np.where(mask)[0]
    deltas = []
    for j, i in enumerate(idx):
        keep = mask.copy()
        keep[i] = False
        if int(keep.sum()) < 12:
            continue
        y_m = mmse[keep]
        X0 = X[keep]
        _, r20, _ = residualize(y_m, X0)
        X1 = np.column_stack([X0, y[keep]])
        _, r21, _ = residualize(y_m, X1)
        deltas.append(float(r21 - r20))
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size:
        report["jackknife"]["mmse_delta_R2_Deff"] = {
            "n_folds": int(deltas.size),
            "median": float(np.median(deltas)),
            "mean": float(np.mean(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "frac_positive": float(np.mean(deltas > 0)),
        }
    report["subject_table_preview_cols"] = [
        "subject_id",
        "clinical_group",
        "split",
        "D_eff",
        "D_eff_res",
        "rho_std_res",
    ]
    return report, df


def _md_section(rep: dict) -> list[str]:
    lines = [
        f"## {rep['name']}",
        "",
        f"n={rep['n']}  counts={rep['counts']}",
        "",
        "| marker | vs stage r | p | structure R² | leftover r | leftover p | HC−AD MW | FTD−AD MW |",
        "|--------|-----------:|--:|-------------:|-----------:|-----------:|---------:|----------:|",
    ]
    for key in ("D_eff", "rho_std", "rho_mean"):
        raw = rep["markers"].get(key, {})
        res = rep["markers"].get(f"{key}_residual", {})
        lines.append(
            f"| `{key}` | {raw.get('spearman_vs_stage_r', float('nan')):.3f} | "
            f"{raw.get('p', float('nan')):.2e} | "
            f"{res.get('structure_R2', float('nan')):.3f} | "
            f"{res.get('leftover_vs_stage_r', float('nan')):.3f} | "
            f"{res.get('leftover_vs_stage_p', float('nan')):.2e} | "
            f"{res.get('mw_hc_ad_p', float('nan')):.2e} | "
            f"{res.get('mw_ftd_ad_p', float('nan')):.2e} |"
        )
    lines += ["", "### MMSE increment beyond age+sex+θ/α", ""]
    for mask_name, block in rep.get("mmse", {}).items():
        lines.append(
            f"- **{mask_name}** (n={block['n']}): demo+spec R²={block['demo_spec_R2']:.3f}"
        )
        for col, inc in block["increments"].items():
            lines.append(f"  - +`{col}` → R²={inc['R2']:.3f} (Δ={inc['delta_R2']:+.3f})")
    jk = rep.get("jackknife", {}).get("D_eff_res_spearman", {})
    if jk:
        lines += [
            "",
            "### LOSO jackknife (`D_eff` residual vs stage)",
            "",
            f"median r={jk['median_r']:.3f}  range [{jk['min_r']:.3f}, {jk['max_r']:.3f}]  "
            f"frac(r>0)={jk['frac_positive']:.2f}  frac(r>0.15)={jk['frac_r_gt_0p15']:.2f}",
        ]
    jk2 = rep.get("jackknife", {}).get("mmse_delta_R2_Deff", {})
    if jk2:
        lines += [
            "",
            "### LOSO jackknife (MMSE ΔR² from `D_eff`)",
            "",
            f"median ΔR²={jk2['median']:+.3f}  range [{jk2['min']:+.3f}, {jk2['max']:+.3f}]  "
            f"frac(Δ>0)={jk2['frac_positive']:.2f}",
        ]
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, path in (
        ("native_v3", Path(args.v3_dir)),
        ("aniso_phase_off", Path(args.aniso_dir)),
    ):
        df = load_subjects(path)
        rep, df_out = analyze_one(name, df)
        reports.append(rep)
        df_out.to_csv(out / f"subjects_{name}_residual.csv", index=False)

    payload = {"protocol": "age+sex+theta_alpha_ratio+bp_alpha OLS residual", "models": reports}
    (out / "RESIDUAL_BOARD.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# EEG structure residual board",
        "",
        "Same claim discipline as seismic β_res: a marker is only a candidate",
        "physical fact after classical covariates are removed.",
        "",
        "Covariates: **age + sex + θ/α + bp_α**.  "
        "`D_eff = 1 / rho_std`.  FTD vs AD leftover is expected to be null on scalars.",
        "",
    ]
    for rep in reports:
        md.extend(_md_section(rep))
    md += [
        "## Claim discipline",
        "",
        "- Do **not** headline raw `D_eff` HC→AD gradients (most is classical slowing + age).",
        "- Keep leftover ρ / `D_eff` only if jackknife sign-stable and MMSE ΔR² > 0.",
        "- Scalar leftover **cannot** separate FTD vs AD — that needs subject-level D topography.",
        "",
        "Regenerate: `PYTHONPATH=. python tools/analyze_eeg_structure_residual.py`",
    ]
    (out / "BOARD.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[residual] → {out / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
