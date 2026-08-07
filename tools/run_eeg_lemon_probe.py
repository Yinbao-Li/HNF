#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen ds004504 probe → MPI-LEMON (healthy EEG+T1).

Pre-registered (see docs/EEG_LEMON_PREREG.md):
  H1  leftover D_eff (AHEPA train β) vs GM/ICV after age+sex on structure  → r < 0
  H2  leftover rho_std vs GM/ICV after age+sex                              → r > 0
  H3  EC–EO ICC of ρ / D_eff  (reliability)
  H4  raw D_eff vs age  (aging continuum, disease↑)
  H5  leftover D_eff vs TMT-B | age+sex  → r > 0 (worse cognition)

Does not retrain. Does not claim AD diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from hnf.eeg_ckpt import load_native_checkpoint
from hnf.eeg_clinical import epoch_feature_vector, welch_band_powers
from hnf.eeg_lemon import list_lemon_subjects, load_lemon_ahepa19
from hnf.eeg_subject_diffusion import pearson_r, residualize, sex_to_float, spearman_r

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

CKPTS = {
    "native_v3": "outputs/eeg/adftd_hnf_native_v3/best.pt",
    "aniso_phase_off": "outputs/eeg/aniso_diffusion_ablation/phase_off/best.pt",
}

VOLTMETER = ("age", "sex_f", "theta_alpha_ratio", "bp_alpha")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="external_data/eeg_lemon")
    p.add_argument("--output-dir", default="outputs/eeg/lemon_probe")
    p.add_argument("--mine-json", default="outputs/eeg/probe_publishable/PROBE_MINE.json")
    p.add_argument("--morph-csv", default="outputs/eeg/lemon_morphometry/lemon_t1_morphometry.csv")
    p.add_argument("--max-subjects", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="")
    p.add_argument("--skip-forward", action="store_true", help="reuse subjects_probe.csv")
    return p.parse_args()


def _fmt(x: float, nd: int = 3) -> str:
    return "NA" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


@torch.no_grad()
def cond_features(
    model: torch.nn.Module,
    epochs: np.ndarray,
    *,
    sample_rate: float,
    mean_omega: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    x = torch.from_numpy(np.asarray(epochs, dtype=np.float32))
    acc: dict[str, list[float]] = {k: [] for k in MARKERS}
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        _, aux = model(xb, return_aux=True)
        rho = aux["rho"].detach().cpu().numpy()
        x_np = xb.detach().cpu().numpy()
        for i in range(x_np.shape[0]):
            feats = epoch_feature_vector(
                x_np[i], rho[i], sample_rate=float(sample_rate), mean_omega=mean_omega
            )
            feats["D_eff"] = 1.0 / max(float(feats["rho_std"]), 1e-6)
            feats.update(welch_band_powers(x_np[i], float(sample_rate)))
            for k in MARKERS:
                v = feats.get(k, float("nan"))
                if np.isfinite(v):
                    acc[k].append(float(v))
    return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in acc.items()}


def load_train_betas(mine_json: Path) -> dict[str, dict[str, list[float]]]:
    blob = json.loads(Path(mine_json).read_text())
    out: dict[str, dict[str, list[float]]] = {}
    for tag, rec in blob.get("probes", {}).items():
        out[tag] = {}
        for marker, mrec in rec.get("markers", {}).items():
            beta = mrec.get("train_beta")
            if beta:
                out[tag][marker] = [float(x) for x in beta]
    return out


def apply_train_beta(y: float, age: float, sex_f: float, tha: float, bpa: float, beta: list[float]) -> float:
    if not (np.isfinite(y) and np.isfinite(age) and np.isfinite(sex_f) and np.isfinite(tha) and np.isfinite(bpa)):
        return float("nan")
    if len(beta) < 5 or not all(math.isfinite(b) for b in beta[:5]):
        return float("nan")
    hat = beta[0] + beta[1] * age + beta[2] * sex_f + beta[3] * tha + beta[4] * bpa
    return float(y - hat)


def load_morph(path: Path) -> dict[str, dict]:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open() as fh:
        return {str(r["subject_id"]): r for r in csv.DictReader(fh)}


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def corr_block(a: np.ndarray, b: np.ndarray) -> dict:
    r_p = pearson_r(a, b)
    r_s, p_s = spearman_r(a, b)
    m = np.isfinite(a) & np.isfinite(b)
    return {"n": int(m.sum()), "pearson_r": r_p, "spearman_r": r_s, "spearman_p": p_s}


def partial_corr(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> dict:
    y_res, _, _ = residualize(y, Z)
    x_res, _, _ = residualize(x, Z)
    return corr_block(y_res, x_res)


def analyze(rows: list[dict]) -> dict:
    age = np.asarray([r["age"] for r in rows], dtype=np.float64)
    sex = sex_to_float([r["sex"] for r in rows])
    demo = np.column_stack([age, sex])
    gm_icv = np.asarray([r.get("gm_icv", np.nan) for r in rows], dtype=np.float64)
    brain = np.asarray([r.get("brain_cm3", np.nan) for r in rows], dtype=np.float64)
    tmt_b = np.asarray([r.get("tmt_b", np.nan) for r in rows], dtype=np.float64)
    cvlt = np.asarray([r.get("cvlt_long_delay", np.nan) for r in rows], dtype=np.float64)
    out: dict = {"n": len(rows), "tests": {}}

    for tag in ("native_v3", "aniso_phase_off"):
        prefix = f"{tag}_ec_"
        deff = np.asarray([r.get(prefix + "D_eff", np.nan) for r in rows], dtype=np.float64)
        rstd = np.asarray([r.get(prefix + "rho_std", np.nan) for r in rows], dtype=np.float64)
        deff_res = np.asarray([r.get(prefix + "D_eff_res", np.nan) for r in rows], dtype=np.float64)
        rstd_res = np.asarray([r.get(prefix + "rho_std_res", np.nan) for r in rows], dtype=np.float64)
        deff_eo = np.asarray([r.get(f"{tag}_eo_D_eff", np.nan) for r in rows], dtype=np.float64)
        rstd_eo = np.asarray([r.get(f"{tag}_eo_rho_std", np.nan) for r in rows], dtype=np.float64)
        tests = {
            "H1_Deff_leftover_vs_gm_icv_partial_age_sex": {
                **partial_corr(deff_res, gm_icv, demo),
                "predicted": "spearman_r < 0",
            },
            "H1b_Deff_leftover_vs_brain_cm3_partial_age_sex": {
                **partial_corr(deff_res, brain, demo),
                "predicted": "spearman_r < 0",
            },
            "H2_rho_std_leftover_vs_gm_icv_partial_age_sex": {
                **partial_corr(rstd_res, gm_icv, demo),
                "predicted": "spearman_r > 0",
            },
            "H3_ec_eo_icc_Deff": {
                "pearson_r": pearson_r(deff, deff_eo),
                "spearman_r": spearman_r(deff, deff_eo)[0],
                "n": int((np.isfinite(deff) & np.isfinite(deff_eo)).sum()),
                "predicted": "r high (reliability)",
            },
            "H3b_ec_eo_icc_rho_std": {
                "pearson_r": pearson_r(rstd, rstd_eo),
                "spearman_r": spearman_r(rstd, rstd_eo)[0],
                "n": int((np.isfinite(rstd) & np.isfinite(rstd_eo)).sum()),
                "predicted": "r high (reliability)",
            },
            "H4_raw_Deff_vs_age": {
                **corr_block(deff, age),
                "predicted": "pearson_r > 0 (disease↑ continuum)",
            },
            "H4b_raw_rho_std_vs_age": {
                **corr_block(rstd, age),
                "predicted": "pearson_r < 0",
            },
            "H5_Deff_leftover_vs_TMTB_partial_age_sex": {
                **partial_corr(deff_res, tmt_b, demo),
                "predicted": "spearman_r > 0",
            },
            "H5b_Deff_leftover_vs_CVLT_long_partial_age_sex": {
                **partial_corr(deff_res, cvlt, demo),
                "predicted": "spearman_r < 0",
            },
            "ctrl_leftover_vs_age": {
                **corr_block(deff_res, age),
                "predicted": "~0 (age already in voltmeter)",
            },
        }
        # young vs old leftover (should be weak)
        yg = np.asarray([r.get("age_group") == "young" for r in rows], dtype=bool)
        og = np.asarray([r.get("age_group") == "old" for r in rows], dtype=bool)
        if yg.sum() >= 8 and og.sum() >= 8:
            from scipy import stats

            a, b = deff_res[yg], deff_res[og]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) >= 8 and len(b) >= 8:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                tests["ctrl_leftover_young_vs_old_mw"] = {
                    "n_young": int(len(a)),
                    "n_old": int(len(b)),
                    "mean_young": float(np.mean(a)),
                    "mean_old": float(np.mean(b)),
                    "U": float(u),
                    "p": float(p),
                    "predicted": "ns after age residualization",
                }
        out["tests"][tag] = tests
    return out


def write_board(path: Path, analysis: dict, n_forward: int, n_skip: int) -> None:
    md = [
        "# LEMON frozen probe × T1 morphometry",
        "",
        "Healthy lifespan (Babayan 2019). Frozen ds004504 v3 / aniso `phase_off`.",
        "Voltmeter leftover uses **AHEPA train β only** (no refit on LEMON).",
        "T1 metrics are GMM tissue proxies, **not** FreeSurfer thickness.",
        "",
        f"n_forward_rows={n_forward}  n_skip={n_skip}",
        "",
        "Pre-reg: `docs/EEG_LEMON_PREREG.md`",
        "",
    ]
    for tag, tests in analysis.get("tests", {}).items():
        md += [f"## {tag}", "", "| Test | n | Spearman r | p | Pearson r | predicted |", "|------|--:|-----------:|--:|----------:|-----------|"]
        for name, rec in tests.items():
            md.append(
                f"| `{name}` | {rec.get('n', rec.get('n_young', '—'))} | "
                f"{_fmt(float(rec.get('spearman_r', float('nan'))))} | "
                f"{_fmt(float(rec.get('spearman_p', rec.get('p', float('nan')))))} | "
                f"{_fmt(float(rec.get('pearson_r', float('nan'))))} | "
                f"{rec.get('predicted', '')} |"
            )
        md.append("")
    md += [
        "## Claim discipline",
        "",
        "- Pass H1/H2 → leftover tracks structure beyond age/sex/voltmeter in healthy adults.",
        "- Fail H1/H2 with pass H3/H4 → marker is reliable + aging-related, structure proxy too crude (need FastSurfer).",
        "- Do **not** call this AD/FTD closure. That needs AHEPA MRI or a clinical site.",
        "",
        "Regenerate:",
        "`PYTHONPATH=. python tools/run_eeg_lemon_probe.py --device cuda`",
        "`PYTHONPATH=. python tools/extract_lemon_t1_morphometry.py`",
        "",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


def run_forward(args, device: torch.device) -> tuple[list[dict], int]:
    subjects = list_lemon_subjects(Path(args.root))
    if args.max_subjects and args.max_subjects > 0:
        subjects = subjects[: int(args.max_subjects)]
    betas = load_train_betas(Path(args.mine_json))
    morph = load_morph(Path(args.morph_csv))
    models = {}
    for tag, ckpt in CKPTS.items():
        path = Path(ckpt)
        if not path.is_file():
            print(f"[warn] missing {path}", flush=True)
            continue
        model, ckpt_args, arch = load_native_checkpoint(path, device)
        sample_rate = int(ckpt_args.get("sample_rate", 128))
        kparams = model.collect_kernel_params()
        mean_omega = (
            float(np.mean([v["omega"] for v in kparams.values() if "omega" in v])) if kparams else 0.0
        )
        models[tag] = (model, sample_rate, mean_omega, arch)

    rows = []
    n_skip = 0
    for i, s in enumerate(subjects, 1):
        if s.ec_path is None:
            n_skip += 1
            continue
        row = {
            "subject_id": s.subject_id,
            "age": s.age,
            "sex": s.sex,
            "age_bin": s.age_bin,
            "age_group": s.age_group,
            "tmt_a": s.tmt_a,
            "tmt_b": s.tmt_b,
            "cvlt_long_delay": s.cvlt_long_delay,
            "n_ch_note": "",
        }
        mrow = morph.get(s.subject_id, {})
        for k in (
            "brain_cm3",
            "gm_cm3",
            "wm_cm3",
            "icv_cm3",
            "gm_icv",
            "wm_icv",
            "ok",
            "source",
        ):
            if k in mrow:
                try:
                    row[k] = float(mrow[k]) if k not in {"ok", "source"} else mrow[k]
                except (TypeError, ValueError):
                    row[k] = mrow[k]
        sex_f = float(sex_to_float([s.sex])[0])
        row["sex_f"] = sex_f
        try:
            ec, _ = load_lemon_ahepa19(s.ec_path, target_sfreq=128.0, max_epochs=args.max_epochs)
        except Exception as exc:
            n_skip += 1
            print(f"[skip EC] {s.subject_id}: {exc}", flush=True)
            continue
        eo = None
        if s.eo_path is not None:
            try:
                eo, _ = load_lemon_ahepa19(s.eo_path, target_sfreq=128.0, max_epochs=args.max_epochs)
            except Exception as exc:
                print(f"[warn EO] {s.subject_id}: {exc}", flush=True)
        for tag, (model, sr, mean_omega, _arch) in models.items():
            fec = cond_features(
                model, ec, sample_rate=sr, mean_omega=mean_omega, device=device, batch_size=args.batch_size
            )
            for k, v in fec.items():
                row[f"{tag}_ec_{k}"] = v
            if eo is not None:
                feo = cond_features(
                    model, eo, sample_rate=sr, mean_omega=mean_omega, device=device, batch_size=args.batch_size
                )
                for k, v in feo.items():
                    row[f"{tag}_eo_{k}"] = v
            bmap = betas.get(tag, {})
            for marker in ("D_eff", "rho_std", "rho_mean"):
                beta = bmap.get(marker)
                if not beta:
                    continue
                y = float(row.get(f"{tag}_ec_{marker}", float("nan")))
                row[f"{tag}_ec_{marker}_res"] = apply_train_beta(
                    y, float(s.age), sex_f, float(fec.get("theta_alpha_ratio", np.nan)), float(fec.get("bp_alpha", np.nan)), beta
                )
        rows.append(row)
        if i % 10 == 0 or i == len(subjects):
            print(f"[lemon-probe] {i}/{len(subjects)} ok={len(rows)} skip={n_skip}", flush=True)
    return rows, n_skip


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "subjects_probe.csv"
    if args.skip_forward and csv_path.is_file():
        with csv_path.open() as fh:
            rows = []
            for r in csv.DictReader(fh):
                conv = {}
                for k, v in r.items():
                    if k in {"subject_id", "sex", "age_bin", "age_group", "source", "ok", "n_ch_note"}:
                        conv[k] = v
                    else:
                        try:
                            conv[k] = float(v) if v not in {"", None} else float("nan")
                        except ValueError:
                            conv[k] = v
                rows.append(conv)
        n_skip = 0
        print(f"[lemon-probe] reused {csv_path} n={len(rows)}", flush=True)
    else:
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[lemon-probe] device={device}", flush=True)
        rows, n_skip = run_forward(args, device)
        write_csv(csv_path, rows)
    # re-merge morph if it appeared after forward
    morph = load_morph(Path(args.morph_csv))
    if morph:
        for r in rows:
            m = morph.get(str(r["subject_id"]), {})
            for k in ("brain_cm3", "gm_cm3", "wm_cm3", "icv_cm3", "gm_icv", "wm_icv"):
                if k in m and str(m.get(k, "")).strip() not in {"", "nan"}:
                    try:
                        r[k] = float(m[k])
                    except ValueError:
                        pass
        write_csv(csv_path, rows)
    analysis = analyze(rows)
    (out / "LEMON_PROBE.json").write_text(
        json.dumps({"n": len(rows), "n_skip": n_skip, "analysis": analysis}, indent=2),
        encoding="utf-8",
    )
    write_board(out / "BOARD.md", analysis, len(rows), n_skip)
    print((out / "BOARD.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
