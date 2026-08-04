#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Longitudinal validation of Domain-II EEG markers on OpenNeuro ds005385.

Dataset is *healthy* aging (no AD). We test:
  1) 5-year test–retest stability (Pearson r / ICC) of AD-relevant spectral markers
  2) Age association of the same markers at baseline
  3) Directional consistency vs ds004504 disease findings
     (e.g. higher theta_alpha_ratio in disease → expect older healthy adults higher?)

Outputs → ``outputs/eeg/longitudinal_ds005385/``
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

from hnf.eeg_clinical import welch_band_powers
from hnf.eeg_ds005385 import AHEPA_19, iter_paired_recordings, load_edf_ahepa19
from hnf.eeg_geometry import REGION_CHANNELS


MARKERS = (
    "theta_alpha_ratio",
    "bp_theta",
    "bp_alpha",
    "bp_beta",
    "front_post_alpha",
    "cov_aniso",
)

# Disease-direction findings from ds004504 knowledge cards (HC vs disease)
DS004504_DIRECTION = {
    "theta_alpha_ratio": "disease_up",  # disease > HC
    "bp_alpha": "disease_down",  # HC > disease
    "bp_theta": "disease_up",  # typically slowing
}


def _region_idx(names=AHEPA_19) -> dict[str, np.ndarray]:
    name_to_i = {n: i for i, n in enumerate(names)}
    out = {}
    for reg, chs in REGION_CHANNELS.items():
        idx = [name_to_i[c] for c in chs if c in name_to_i]
        out[reg] = np.asarray(idx, dtype=np.int64)
    return out


def epoch_extra_feats(x: np.ndarray, sfreq: float, reg_idx: dict[str, np.ndarray]) -> dict[str, float]:
    """Extra spatial features on epoch ``(C,T)``."""
    # front/post alpha ratio from channel FFT
    t = x.shape[1]
    window = np.hanning(t)
    spec = np.fft.rfft(x * window[None, :], axis=1)
    power = (np.abs(spec) ** 2) / max(float(np.sum(window**2)), 1e-12)
    freqs = np.fft.rfftfreq(t, d=1.0 / float(sfreq))
    a_mask = (freqs >= 8.0) & (freqs < 13.0)
    a_pow = power[:, a_mask].sum(axis=1)
    fi = reg_idx.get("frontal", np.asarray([], dtype=np.int64))
    pi = reg_idx.get("posterior", np.asarray([], dtype=np.int64))
    front = float(np.log10(max(a_pow[fi].mean() if fi.size else 1e-20, 1e-20)))
    post = float(np.log10(max(a_pow[pi].mean() if pi.size else 1e-20, 1e-20)))
    # covariance anisotropy: λ_max / λ_mean of channel cov
    xc = x - x.mean(axis=1, keepdims=True)
    cov = (xc @ xc.T) / max(t - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.maximum(ev, 0.0)
    cov_aniso = float(ev[-1] / (ev.mean() + 1e-12))
    return {"front_post_alpha": front - post, "cov_aniso": cov_aniso}


def subject_session_features(path: Path) -> dict[str, float]:
    epochs, sfreq = load_edf_ahepa19(path)
    reg = _region_idx()
    acc: dict[str, list[float]] = {k: [] for k in MARKERS}
    for ep in epochs:
        bp = welch_band_powers(ep, sfreq)
        extra = epoch_extra_feats(ep, sfreq, reg)
        row = {**bp, **extra}
        for k in MARKERS:
            v = row.get(k, float("nan"))
            if np.isfinite(v):
                acc[k].append(float(v))
    return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in acc.items()}


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def icc_abs_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1)-like absolute agreement for two repeated measures."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    x = np.stack([a[m], b[m]], axis=1)  # (n, 2)
    n = x.shape[0]
    grand = x.mean()
    ms_rows = ((x.mean(axis=1) - grand) ** 2).sum() * 2 / max(n - 1, 1)
    ms_err = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0) + grand) ** 2).sum() / max(
        (n - 1) * 1, 1
    )
    # simplified ICC(A,1)
    return float((ms_rows - ms_err) / (ms_rows + ms_err + 1e-12))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="external_data/eeg_ds005385")
    p.add_argument("--output-dir", default="outputs/eeg/longitudinal_ds005385")
    p.add_argument("--max-subjects", type=int, default=60)
    p.add_argument("--task", default="EyesClosed")
    p.add_argument("--acq", default="pre")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (root / "participants.tsv").is_file():
        raise SystemExit(f"Missing {root}/participants.tsv — run tools/download_eeg_ds005385.py first")

    rows = []
    n_skip = 0
    for rec1, rec2 in iter_paired_recordings(
        root, max_subjects=args.max_subjects, task=args.task, acq=args.acq
    ):
        try:
            f1 = subject_session_features(rec1.path)
            f2 = subject_session_features(rec2.path)
        except Exception as exc:
            n_skip += 1
            print(f"[skip] {rec1.subject_id}: {exc}", flush=True)
            continue
        years = None
        if rec1.recording_year and rec2.recording_year:
            years = int(rec2.recording_year - rec1.recording_year)
        rows.append(
            {
                "subject_id": rec1.subject_id,
                "age_baseline": rec1.age,
                "sex": rec1.sex,
                "year_ses1": rec1.recording_year,
                "year_ses2": rec2.recording_year,
                "delta_years": years,
                "ses1": f1,
                "ses2": f2,
            }
        )
        print(f"[ok] {rec1.subject_id} age={rec1.age} Δy={years}", flush=True)

    if not rows:
        raise SystemExit("No paired recordings found. Download EyesClosed/pre EDFs first.")

    summary = {"n_paired": len(rows), "n_skip": n_skip, "markers": {}}
    ages = np.asarray([r["age_baseline"] for r in rows], dtype=np.float64)
    for k in MARKERS:
        v1 = np.asarray([r["ses1"].get(k, np.nan) for r in rows], dtype=np.float64)
        v2 = np.asarray([r["ses2"].get(k, np.nan) for r in rows], dtype=np.float64)
        delta = v2 - v1
        r_tr = pearson_r(v1, v2)
        icc = icc_abs_agreement(v1, v2)
        r_age = pearson_r(ages, v1)
        mean_delta = float(np.nanmean(delta))
        # interpret vs disease direction (exploratory for healthy aging)
        note = ""
        if k in DS004504_DIRECTION:
            dirc = DS004504_DIRECTION[k]
            if np.isfinite(r_age):
                if dirc == "disease_up":
                    note = "age↑ matches disease↑" if r_age > 0 else "age trend opposite to disease↑"
                else:
                    note = "age↓ matches disease↓" if r_age < 0 else "age trend opposite to disease↓"
        summary["markers"][k] = {
            "test_retest_r": r_tr,
            "icc": icc,
            "mean_ses1": float(np.nanmean(v1)),
            "mean_ses2": float(np.nanmean(v2)),
            "mean_delta_ses2_minus_ses1": mean_delta,
            "age_corr_ses1": r_age,
            "ds004504_disease_direction": DS004504_DIRECTION.get(k),
            "age_vs_disease_note": note,
        }

    (out / "longitudinal_pairs.json").write_text(json.dumps({"rows": rows}, indent=2))
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Longitudinal EEG validation — OpenNeuro ds005385",
        "",
        "Healthy aging cohort (Dortmund Vital Study). **No AD patients.**",
        f"Paired EyesClosed/acq-{args.acq} recordings: **n={len(rows)}** (skipped {n_skip}).",
        "",
        "Claim discipline: this tests **marker stability / age association** of Domain-II",
        "spectral & spatial features — **not** AD classification transfer.",
        "",
        "| Marker | test–retest r | ICC | mean Δ (ses2−ses1) | age corr (ses1) | vs ds004504 disease dir |",
        "|--------|--------------:|----:|-------------------:|----------------:|-------------------------|",
    ]
    for k, s in summary["markers"].items():
        md.append(
            f"| `{k}` | {s['test_retest_r']:.3f} | {s['icc']:.3f} | "
            f"{s['mean_delta_ses2_minus_ses1']:.4f} | {s['age_corr_ses1']:.3f} | "
            f"{s['age_vs_disease_note'] or '—'} |"
        )
    md += [
        "",
        "## Takeaway template",
        "- High test–retest r / ICC → marker is longitudinally reliable in healthy adults.",
        "- Age correlation aligned with ds004504 disease direction → soft support that the",
        "  marker tracks aging-related slowing (not a disease diagnosis claim).",
        "",
        "Regenerate: `PYTHONPATH=. python tools/run_eeg_longitudinal_ds005385.py`",
    ]
    (out / "BOARD.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[longitudinal] → {out / 'BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
