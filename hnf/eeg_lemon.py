# -*- coding: utf-8 -*-
"""MPI-LEMON phenotype + 62ch EEGLAB → AHEPA-19 loader."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hnf.eeg_ds005385 import AHEPA_19, _ALIAS

_AGE_BIN_RE = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$")


def age_bin_midpoint(age_str: object) -> float:
    s = str(age_str or "").strip()
    m = _AGE_BIN_RE.match(s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return float("nan")
    lo, hi = float(m.group(1)), float(m.group(2))
    return 0.5 * (lo + hi)


def lemon_sex_code(gender: object) -> str:
    """LEMON META: 1=female, 2=male → F/M for ``sex_to_float``."""
    s = str(gender or "").strip()
    if s in {"1", "1.0", "F", "f", "female"}:
        return "F"
    if s in {"2", "2.0", "M", "m", "male"}:
        return "M"
    return ""


@dataclass
class LemonSubject:
    subject_id: str
    age: float
    sex: str
    age_bin: str
    age_group: str  # young / old / mid
    ec_path: Optional[Path]
    eo_path: Optional[Path]
    t1w_path: Optional[Path]
    inv2_path: Optional[Path]
    brain_path: Optional[Path]
    tmt_a: float
    tmt_b: float
    cvlt_long_delay: float


def _first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p is not None and p.is_file():
            return p
    return None


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        sid = str(r.get("ID") or r.get("participant_id") or "").strip()
        if sid:
            out[sid] = r
    return out


def _float_cell(row: Optional[dict], key: str) -> float:
    if not row:
        return float("nan")
    raw = str(row.get(key, "")).strip()
    if raw in {"", "nan", "NaN", "NA", "None"}:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def list_lemon_subjects(root: Path) -> list[LemonSubject]:
    root = Path(root)
    inv_path = root / "INVENTORY.json"
    if inv_path.is_file():
        inv = json.loads(inv_path.read_text())
        ids = list(inv.get("eeg_and_mri_ids") or inv.get("eeg_ids") or [])
    else:
        ids = sorted(
            {p.name.split("_")[0] for p in (root / "eeg" / "preprocessed").glob("sub-*.set")}
        )
    meta = _read_csv_by_id(
        root / "behavioural" / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    )
    if not meta:
        meta = _read_csv_by_id(root / "mri_meta" / "Participants_LEMON.csv")
    tmt = _read_csv_by_id(
        root / "behavioural" / "Cognitive_Test_Battery_LEMON" / "TMT" / "TMT.csv"
    )
    cvlt_dir = root / "behavioural" / "Cognitive_Test_Battery_LEMON"
    cvlt_path = cvlt_dir / "CVLT.csv"
    if not cvlt_path.is_file():
        matches = list(cvlt_dir.glob("CVLT*/CVLT.csv"))
        cvlt_path = matches[0] if matches else cvlt_path
    cvlt = _read_csv_by_id(cvlt_path)

    eeg_dir = root / "eeg" / "preprocessed"
    out: list[LemonSubject] = []
    for sid in ids:
        row = meta.get(sid, {})
        age_bin = str(row.get("Age") or "").strip()
        age = age_bin_midpoint(age_bin)
        sex = lemon_sex_code(row.get("Gender_ 1=female_2=male") or row.get("Gender"))
        if np.isfinite(age) and age < 45:
            age_group = "young"
        elif np.isfinite(age) and age >= 55:
            age_group = "old"
        else:
            age_group = "mid"
        ec = _first_existing(eeg_dir / f"{sid}_EC.set")
        eo = _first_existing(eeg_dir / f"{sid}_EO.set", eeg_dir / f"{sid}_E0.set")
        anat = root / "mri" / "raw" / sid / "ses-01" / "anat"
        brain = (
            root
            / "mri"
            / "derivatives"
            / sid
            / "anat"
            / f"{sid}_ses-01_acq-mp2rage_brain.nii.gz"
        )
        tmt_row, cvlt_row = tmt.get(sid), cvlt.get(sid)
        out.append(
            LemonSubject(
                subject_id=sid,
                age=float(age),
                sex=sex,
                age_bin=age_bin,
                age_group=age_group,
                ec_path=ec,
                eo_path=eo,
                t1w_path=_first_existing(anat / f"{sid}_ses-01_acq-mp2rage_T1w.nii.gz"),
                inv2_path=_first_existing(anat / f"{sid}_ses-01_inv-2_mp2rage.nii.gz"),
                brain_path=_first_existing(brain),
                tmt_a=_float_cell(tmt_row, "TMT_1"),
                tmt_b=_float_cell(tmt_row, "TMT_5"),
                cvlt_long_delay=_float_cell(cvlt_row, "CVLT_11"),
            )
        )
    return out


def load_lemon_ahepa19(
    path: Path,
    *,
    target_sfreq: float = 128.0,
    epoch_sec: float = 10.0,
    max_epochs: int = 24,
    filter_hz: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, float]:
    """EEGLAB .set → pad/reorder AHEPA-19 → resample → middle non-overlap epochs.

    Does **not** fall back to the first 19 physical channels. Missing 10–20
    sites (common after LEMON ICA) are zero-filled.
    """
    import mne

    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    raw.pick("eeg")
    rename = {}
    for ch in list(raw.ch_names):
        base = ch.split(" ")[-1].replace("EEG", "").strip()
        key = _ALIAS.get(base, _ALIAS.get(base.upper(), base))
        if key in AHEPA_19:
            rename[ch] = key
    if rename:
        raw.rename_channels({k: v for k, v in rename.items() if k in raw.ch_names})
    keep = [c for c in AHEPA_19 if c in raw.ch_names]
    if len(keep) < 10:
        raise RuntimeError(f"Too few mapped 10–20 channels ({len(keep)}) in {path}")
    raw.pick(keep)
    if filter_hz is not None:
        raw.filter(float(filter_hz[0]), float(filter_hz[1]), verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-3:
        raw.resample(target_sfreq, verbose="ERROR")
    data = raw.get_data()
    C_full = np.zeros((len(AHEPA_19), data.shape[1]), dtype=np.float64)
    name_to_i = {n: i for i, n in enumerate(keep)}
    for j, name in enumerate(AHEPA_19):
        if name in name_to_i:
            C_full[j] = data[name_to_i[name]]
    # recording-level z-score (ds004504 EEGDataset), then epoch
    mu = C_full.mean(axis=1, keepdims=True)
    sd = C_full.std(axis=1, keepdims=True) + 1e-6
    present = np.array([name in name_to_i for name in AHEPA_19], dtype=bool)
    C_full[present] = (C_full[present] - mu[present]) / sd[present]
    sfreq = float(raw.info["sfreq"])
    win = int(round(epoch_sec * sfreq))
    if win < 8 or C_full.shape[1] < win:
        raise RuntimeError(f"Recording too short: {path}")
    n_possible = C_full.shape[1] // win
    n_ep = min(int(max_epochs), max(n_possible, 1))
    start0 = max(0, (C_full.shape[1] - n_ep * win) // 2)
    eps = [C_full[:, start0 + i * win : start0 + (i + 1) * win] for i in range(n_ep)]
    return np.stack(eps, axis=0).astype(np.float32), sfreq
