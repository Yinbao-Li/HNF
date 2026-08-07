#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mine publishable probe facts on frozen EEG clinical tables.

Protocol (seismic-style, probe language):
  1. Fit voltmeter residualization on TRAIN only → confirm on val / test / val+test
  2. Dual-probe (v3 Huygens ρ vs aniso diffusion ρ): shared medium vs orthogonal axes
  3. MMSE path: voltmeter vs probe leftover (incremental + partial)
  4. Rhythm-branch probe vs classical Welch (different instrument)
  5. Phase ablation as probe-physics control
  6. Boundary subjects (probe–label mismatch)
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
    p.add_argument("--v3-dir", default="outputs/eeg/clinical_breakthrough_native_v3")
    p.add_argument("--aniso-off", default="outputs/eeg/aniso_diffusion_ablation/phase_off_clinical")
    p.add_argument("--aniso-on", default="outputs/eeg/aniso_diffusion_ablation/phase_on_clinical")
    p.add_argument("--output-dir", default="outputs/eeg/probe_publishable")
    p.add_argument("--n-perm", type=int, default=2000)
    return p.parse_args()


def load_split_tables(clinical_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        p = clinical_dir / f"subjects_{split}.csv"
        d = pd.read_csv(p)
        d["split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["D_eff"] = 1.0 / np.clip(df["rho_std"].to_numpy(float), 1e-6, None)
    df["sex"] = sex_to_float(df["gender"].tolist())
    df["stage"] = df["clinical_group"].map({"HC": 0, "FTD": 1, "AD": 2}).astype(float)
    df["disease"] = (df["clinical_group"] != "HC").astype(float)
    return df


def covar_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["age"].to_numpy(float),
            df["sex"].to_numpy(float),
            df["theta_alpha_ratio"].to_numpy(float),
            df["bp_alpha"].to_numpy(float),
        ]
    )


def apply_residual(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), X])
    out = np.full(len(y), np.nan)
    m = np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(beta).all()
    out[m] = y[m] - A[m] @ beta
    return out


def fit_train_residualizer(df: pd.DataFrame, col: str) -> np.ndarray:
    tr = df[df.split == "train"]
    _, _, beta = residualize(tr[col].to_numpy(float), covar_matrix(tr))
    return beta


def perm_spearman(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    r_obs, p_para = spearman_r(a, b)
    m = np.isfinite(a) & np.isfinite(b)
    aa, bb = a[m], b[m]
    if aa.size < 6 or not np.isfinite(r_obs):
        return {"r": r_obs, "p_parametric": p_para, "p_perm": float("nan"), "n": int(aa.size)}
    null = np.empty(n_perm, float)
    for i in range(n_perm):
        null[i] = spearman_r(aa, rng.permutation(bb))[0]
    p_perm = float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1))
    return {"r": r_obs, "p_parametric": p_para, "p_perm": p_perm, "n": int(aa.size)}


def split_eval(df: pd.DataFrame, resid: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    out = {}
    for name, mask in (
        ("train", df.split == "train"),
        ("val", df.split == "val"),
        ("test", df.split == "test"),
        ("valtest", df.split.isin(["val", "test"])),
        ("all", np.ones(len(df), dtype=bool)),
    ):
        m = np.asarray(mask, dtype=bool)
        sub_r = resid[m]
        st = df.loc[m, "stage"].to_numpy(float)
        dis = df.loc[m, "disease"].to_numpy(float)
        g = df.loc[m, "clinical_group"].astype(str).to_numpy()
        block = {
            "n": int(mask.sum()),
            "vs_stage": perm_spearman(sub_r, st, n_perm, rng),
            "vs_disease": perm_spearman(sub_r, dis, n_perm, rng),
            "means": {gg: float(np.nanmean(sub_r[g == gg])) for gg in GROUPS},
        }
        hc, ad = sub_r[g == "HC"], sub_r[g == "AD"]
        ftd = sub_r[g == "FTD"]
        if len(hc) >= 3 and len(ad) >= 3:
            block["mw_hc_ad_p"] = float(stats.mannwhitneyu(hc, ad, alternative="two-sided").pvalue)
        if len(ftd) >= 3 and len(ad) >= 3:
            block["mw_ftd_ad_p"] = float(stats.mannwhitneyu(ftd, ad, alternative="two-sided").pvalue)
        out[name] = block
    return out


def mmse_paths(df: pd.DataFrame, probe_res: np.ndarray) -> dict:
    mmse = df["mmse"].to_numpy(float)
    Xv = covar_matrix(df)
    stage = df["stage"].to_numpy(float)
    out = {}
    for name, mask in (
        ("all", np.isfinite(mmse)),
        ("patients", np.isfinite(mmse) & (df.clinical_group.to_numpy() != "HC")),
        ("test", np.isfinite(mmse) & (df.split.to_numpy() == "test")),
    ):
        if int(mask.sum()) < 12:
            continue
        y = mmse[mask]
        paths = {}
        specs = {
            "voltmeter": Xv[mask],
            "stage": stage[mask].reshape(-1, 1),
            "voltmeter+stage": np.column_stack([Xv[mask], stage[mask]]),
            "voltmeter+probe": np.column_stack([Xv[mask], probe_res[mask]]),
            "voltmeter+stage+probe": np.column_stack([Xv[mask], stage[mask], probe_res[mask]]),
            "stage+probe": np.column_stack([stage[mask], probe_res[mask]]),
        }
        for k, X in specs.items():
            _, r2, _ = residualize(y, X)
            paths[k] = r2
        paths["delta_probe_beyond_voltmeter"] = paths["voltmeter+probe"] - paths["voltmeter"]
        paths["delta_probe_beyond_voltmeter_stage"] = (
            paths["voltmeter+stage+probe"] - paths["voltmeter+stage"]
        )
        r, p = spearman_r(probe_res[mask], y)
        paths["probe_vs_mmse_r"] = r
        paths["probe_vs_mmse_p"] = p
        # voltmeter-residual MMSE vs probe
        mmse_res, _, _ = residualize(y, Xv[mask])
        rr, pp = spearman_r(probe_res[mask], mmse_res)
        paths["probe_vs_mmse_res_r"] = rr
        paths["probe_vs_mmse_res_p"] = pp
        paths["n"] = int(mask.sum())
        out[name] = paths
    return out


def boundary_subjects(df: pd.DataFrame, resid: np.ndarray, k: int = 6) -> list[dict]:
    tmp = df.copy()
    tmp["probe_res"] = resid
    rows = []
    # disease-like HC: highest leftover D_eff among HC
    hc = tmp[tmp.clinical_group == "HC"].nlargest(k, "probe_res")
    for _, r in hc.iterrows():
        rows.append(
            {
                "kind": "HC_disease_like_probe",
                "subject_id": r.subject_id,
                "split": r.split,
                "group": r.clinical_group,
                "age": float(r.age),
                "mmse": float(r.mmse),
                "probe_res": float(r.probe_res),
                "pred": int(r.pred) if "pred" in r else None,
            }
        )
    dis = tmp[tmp.clinical_group != "HC"].nsmallest(k, "probe_res")
    for _, r in dis.iterrows():
        rows.append(
            {
                "kind": "disease_HC_like_probe",
                "subject_id": r.subject_id,
                "split": r.split,
                "group": r.clinical_group,
                "age": float(r.age),
                "mmse": float(r.mmse),
                "probe_res": float(r.probe_res),
                "pred": int(r.pred) if "pred" in r else None,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    v3 = load_split_tables(Path(args.v3_dir))
    aniso = load_split_tables(Path(args.aniso_off))
    phase_on = load_split_tables(Path(args.aniso_on))

    report: dict = {"protocol": "train-fit voltmeter residual → confirm on held-out splits", "probes": {}}

    # --- single-probe train/test ---
    for tag, df in (("native_v3", v3), ("aniso_phase_off", aniso), ("aniso_phase_on", phase_on)):
        probe_block = {"markers": {}}
        for col in ("D_eff", "rho_std", "rho_mean"):
            beta = fit_train_residualizer(df, col)
            resid = apply_residual(df[col].to_numpy(float), covar_matrix(df), beta)
            df[f"{col}_res_trainfit"] = resid
            probe_block["markers"][col] = {
                "train_beta": [float(x) for x in beta],
                "splits": split_eval(df, resid, args.n_perm, rng),
                "mmse": mmse_paths(df, resid),
            }
        probe_block["boundaries_Deff"] = boundary_subjects(df, df["D_eff_res_trainfit"].to_numpy())
        report["probes"][tag] = probe_block
        df.to_csv(out / f"subjects_{tag}_trainfit_residual.csv", index=False)

    # --- dual probe on matched subjects ---
    m = v3.merge(
        aniso[
            [
                "subject_id",
                "rho_std",
                "rho_mean",
                "D_eff",
                "rho_std_res_trainfit",
                "D_eff_res_trainfit",
                "hnf_theta_energy",
                "hnf_alpha_energy",
                "hnf_theta_alpha_ratio",
            ]
        ],
        on="subject_id",
        suffixes=("_v3", "_aniso"),
    )
    dual = {}
    for split_name, mask in (
        ("train", m.split == "train"),
        ("valtest", m.split.isin(["val", "test"])),
        ("all", np.ones(len(m), bool)),
    ):
        a = m.loc[mask, "D_eff_res_trainfit_v3"].to_numpy(float)
        b = m.loc[mask, "D_eff_res_trainfit_aniso"].to_numpy(float)
        st = m.loc[mask, "stage"].to_numpy(float)
        mmse = m.loc[mask, "mmse"].to_numpy(float)
        r_ab, p_ab = spearman_r(a, b)
        # unique vs shared for stage: residualize each leftover on the other
        a_u, _, _ = residualize(a, b.reshape(-1, 1))
        b_u, _, _ = residualize(b, a.reshape(-1, 1))
        dual[split_name] = {
            "n": int(mask.sum()),
            "v3_vs_aniso_leftover_r": r_ab,
            "v3_vs_aniso_leftover_p": p_ab,
            "v3_unique_vs_stage": perm_spearman(a_u, st, args.n_perm, rng),
            "aniso_unique_vs_stage": perm_spearman(b_u, st, args.n_perm, rng),
            "v3_vs_stage": perm_spearman(a, st, args.n_perm, rng),
            "aniso_vs_stage": perm_spearman(b, st, args.n_perm, rng),
        }
        if np.isfinite(mmse).sum() >= 12:
            dual[split_name]["v3_vs_mmse"] = perm_spearman(a, mmse, args.n_perm, rng)
            dual[split_name]["aniso_vs_mmse"] = perm_spearman(b, mmse, args.n_perm, rng)
            # joint MMSE R2
            Xv = covar_matrix(m.loc[mask])
            y = mmse
            ok = np.isfinite(y)
            _, r2v, _ = residualize(y[ok], Xv[ok])
            _, r2a, _ = residualize(y[ok], np.column_stack([Xv[ok], a[ok]]))
            _, r2b, _ = residualize(y[ok], np.column_stack([Xv[ok], b[ok]]))
            _, r2ab, _ = residualize(y[ok], np.column_stack([Xv[ok], a[ok], b[ok]]))
            dual[split_name]["mmse_R2"] = {
                "voltmeter": r2v,
                "+v3": r2a,
                "+aniso": r2b,
                "+both": r2ab,
                "delta_v3": r2a - r2v,
                "delta_aniso": r2b - r2v,
                "delta_both": r2ab - r2v,
                "delta_second_probe": r2ab - max(r2a, r2b),
            }
    report["dual_probe"] = dual

    # --- HNF rhythm branch vs Welch voltmeter ---
    rhythm = {}
    for tag, df in (("native_v3", v3), ("aniso_phase_off", aniso)):
        if "hnf_theta_alpha_ratio" not in df.columns:
            continue
        beta = fit_train_residualizer(df, "hnf_theta_alpha_ratio")
        # residualize HNF θ/α on classical voltmeter+demo (train beta via same covar which already has θ/α)
        resid = apply_residual(df["hnf_theta_alpha_ratio"].to_numpy(float), covar_matrix(df), beta)
        rhythm[tag] = split_eval(df, resid, args.n_perm, rng)
    report["hnf_rhythm_leftover"] = rhythm

    # --- spatial mix / region leftover on aniso (has region cols) ---
    spatial = {}
    for col in (
        "spatial_mix_delta",
        "region_ft_contrast",
        "region_pf_contrast",
        "region_frontal",
        "region_posterior",
    ):
        if col not in aniso.columns:
            continue
        beta = fit_train_residualizer(aniso, col)
        resid = apply_residual(aniso[col].to_numpy(float), covar_matrix(aniso), beta)
        spatial[col] = split_eval(aniso, resid, args.n_perm, rng)
    report["spatial_probe_leftover"] = spatial

    (out / "PROBE_MINE.json").write_text(json.dumps(report, indent=2))
    md = _to_markdown(report)
    (out / "BOARD.md").write_text(md)
    print(md, flush=True)
    print(f"[probe-mine] → {out / 'BOARD.md'}", flush=True)


def _fmt_sp(d: dict) -> str:
    if not d:
        return "—"
    r = d.get("r", float("nan"))
    pp = d.get("p_perm", float("nan"))
    return f"r={r:.3f} p_perm={pp:.3g} n={d.get('n', '—')}"


def _to_markdown(report: dict) -> str:
    lines = [
        "# EEG probe publishable mine",
        "",
        "Voltmeter = age + sex + θ/α + bp_α. Residualizer **fit on train only**.",
        "Permutation p on Spearman (abs, two-sided).",
        "",
        "## 1. Train-fit → held-out confirmation (`D_eff` leftover vs stage / disease)",
        "",
    ]
    for tag, block in report["probes"].items():
        de = block["markers"]["D_eff"]["splits"]
        lines += [f"### {tag}", "", "| split | vs stage | vs disease (HC↔rest) | HC−AD MW |", "|-------|----------|----------------------|---------:|"]
        for sp in ("train", "val", "test", "valtest"):
            b = de[sp]
            lines.append(
                f"| {sp} (n={b['n']}) | {_fmt_sp(b['vs_stage'])} | {_fmt_sp(b['vs_disease'])} | "
                f"{b.get('mw_hc_ad_p', float('nan')):.3g} |"
            )
        mm = block["markers"]["D_eff"].get("mmse", {})
        if "all" in mm:
            a = mm["all"]
            lines.append(
                f"- MMSE all: ΔR²(probe|voltmeter)={a['delta_probe_beyond_voltmeter']:+.3f}; "
                f"ΔR²(probe|voltmeter+stage)={a['delta_probe_beyond_voltmeter_stage']:+.3f}; "
                f"probe vs MMSE_res r={a['probe_vs_mmse_res_r']:.3f} p={a['probe_vs_mmse_res_p']:.3g}"
            )
        if "patients" in mm:
            a = mm["patients"]
            lines.append(
                f"- MMSE patients: ΔR²(probe|voltmeter)={a['delta_probe_beyond_voltmeter']:+.3f}; "
                f"probe vs MMSE_res r={a['probe_vs_mmse_res_r']:.3f} p={a['probe_vs_mmse_res_p']:.3g}"
            )
        lines.append("")

    lines += ["## 2. Dual probe (v3 Huygens ρ vs aniso diffusion ρ)", ""]
    for sp, b in report["dual_probe"].items():
        lines += [
            f"### {sp} n={b['n']}",
            f"- leftover–leftover Spearman r={b['v3_vs_aniso_leftover_r']:.3f} p={b['v3_vs_aniso_leftover_p']:.3g}",
            f"- v3 vs stage {_fmt_sp(b['v3_vs_stage'])}",
            f"- aniso vs stage {_fmt_sp(b['aniso_vs_stage'])}",
            f"- v3 unique|aniso vs stage {_fmt_sp(b['v3_unique_vs_stage'])}",
            f"- aniso unique|v3 vs stage {_fmt_sp(b['aniso_unique_vs_stage'])}",
        ]
        if "mmse_R2" in b:
            m = b["mmse_R2"]
            lines.append(
                f"- MMSE R² voltmeter={m['voltmeter']:.3f} +v3={m['+v3']:.3f} "
                f"(Δ={m['delta_v3']:+.3f}) +aniso={m['+aniso']:.3f} (Δ={m['delta_aniso']:+.3f}) "
                f"+both={m['+both']:.3f} (second-probe Δ={m['delta_second_probe']:+.3f})"
            )
        lines.append("")

    lines += ["## 3. HNF rhythm-branch leftover after voltmeter", ""]
    for tag, splits in report.get("hnf_rhythm_leftover", {}).items():
        vt = splits.get("valtest", {})
        lines.append(f"- **{tag} valtest** hnf θ/α leftover vs stage {_fmt_sp(vt.get('vs_stage', {}))}")
    lines += ["", "## 4. Spatial / regional probe leftover (aniso)", ""]
    for col, splits in report.get("spatial_probe_leftover", {}).items():
        vt = splits.get("valtest", {})
        al = splits.get("all", {})
        lines.append(
            f"- `{col}` valtest {_fmt_sp(vt.get('vs_stage', {}))} · all {_fmt_sp(al.get('vs_stage', {}))}"
        )

    lines += [
        "",
        "## 5. Boundary subjects (v3 `D_eff` train-fit leftover)",
        "",
        "| kind | id | split | group | age | MMSE | leftover |",
        "|------|----|-------|-------|----:|-----:|---------:|",
    ]
    for row in report["probes"]["native_v3"]["boundaries_Deff"]:
        lines.append(
            f"| {row['kind']} | {row['subject_id']} | {row['split']} | {row['group']} | "
            f"{row['age']:.0f} | {row['mmse']:.0f} | {row['probe_res']:+.3f} |"
        )
    lines += [
        "",
        "## Claim gate",
        "",
        "- Publish leftover only if **valtest** stage or disease contrast keeps sign and p_perm<0.05.",
        "- Dual-probe: if leftover–leftover r high → one medium axis, two kernels; if low and both unique-stage → two axes.",
        "- Phase_on should be weaker than phase_off if the claim is diffusion transport.",
        "- Boundaries are **predictions**, not subtypes, until follow-up / imaging.",
        "",
        "Regenerate: `PYTHONPATH=. python tools/mine_eeg_probe_publishable.py`",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
