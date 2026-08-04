# -*- coding: utf-8 -*-
"""Minimal loader for OpenNeuro ds005385 longitudinal resting-state EEG."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Map common 10–20 / 10–10 names onto the 19-channel AHEPA montage used in Domain II.
AHEPA_19 = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
)
# Some caps use T7/T8/P7/P8 instead of T3/T4/T5/T6.
_ALIAS = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "FP1": "Fp1", "FP2": "Fp2",
}


@dataclass
class LongeegRecording:
    subject_id: str
    session: str  # ses-1 / ses-2
    age: float
    sex: str
    recording_year: Optional[int]
    path: Path


def list_longitudinal_subjects(root: Path) -> list[dict]:
    rows = list(csv.DictReader((root / "participants.tsv").open(), delimiter="\t"))
    out = []
    for r in rows:
        if str(r.get("session1", "")).lower() != "yes":
            continue
        if str(r.get("session2", "")).lower() != "yes":
            continue
        out.append(
            {
                "subject_id": str(r["participant_id"]),
                "age": float(r["age"]) if str(r.get("age", "")).strip() not in ("", "n/a") else float("nan"),
                "sex": str(r.get("sex", "")),
            }
        )
    return out


def session_year(root: Path, subject_id: str, session: str) -> Optional[int]:
    p = root / subject_id / f"{subject_id}_sessions.tsv"
    if not p.is_file():
        return None
    for r in csv.DictReader(p.open(), delimiter="\t"):
        if str(r.get("session_id")) == session:
            try:
                return int(r["recording_year"])
            except (KeyError, ValueError, TypeError):
                return None
    return None


def edf_path(
    root: Path,
    subject_id: str,
    session: str,
    *,
    task: str = "EyesClosed",
    acq: str = "pre",
) -> Path:
    return (
        root
        / subject_id
        / session
        / "eeg"
        / f"{subject_id}_{session}_task-{task}_acq-{acq}_eeg.edf"
    )


def iter_paired_recordings(
    root: Path,
    *,
    max_subjects: int = 0,
    task: str = "EyesClosed",
    acq: str = "pre",
) -> Iterator[tuple[LongeegRecording, LongeegRecording]]:
    subjects = list_longitudinal_subjects(root)
    if max_subjects and max_subjects > 0:
        subjects = subjects[:max_subjects]
    for s in subjects:
        sid = s["subject_id"]
        p1 = edf_path(root, sid, "ses-1", task=task, acq=acq)
        p2 = edf_path(root, sid, "ses-2", task=task, acq=acq)
        if not (p1.is_file() and p2.is_file()):
            continue
        yield (
            LongeegRecording(
                subject_id=sid,
                session="ses-1",
                age=float(s["age"]),
                sex=str(s["sex"]),
                recording_year=session_year(root, sid, "ses-1"),
                path=p1,
            ),
            LongeegRecording(
                subject_id=sid,
                session="ses-2",
                age=float(s["age"]),
                sex=str(s["sex"]),
                recording_year=session_year(root, sid, "ses-2"),
                path=p2,
            ),
        )


def load_edf_ahepa19(
    path: Path,
    *,
    target_sfreq: float = 128.0,
    epoch_sec: float = 10.0,
    max_epochs: int = 12,
) -> tuple[np.ndarray, float]:
    """Load EDF → pick/rename to AHEPA-19 → resample → return epochs ``(N, 19, T)``."""
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    raw.pick("eeg")
    rename = {}
    for ch in list(raw.ch_names):
        base = ch.split(" ")[-1].replace("EEG", "").replace("-", "").strip()
        for pref in ("EEG", "E"):
            if base.upper().startswith(pref) and len(base) > len(pref):
                cand = base[len(pref) :]
                if cand:
                    base = cand
        key = _ALIAS.get(base, base)
        if key in AHEPA_19:
            rename[ch] = key
    raw.rename_channels({k: v for k, v in rename.items() if k in raw.ch_names})
    keep = [c for c in AHEPA_19 if c in raw.ch_names]
    if len(keep) < 10:
        raise RuntimeError(f"Too few mapped channels ({len(keep)}) in {path}")
    raw.pick(keep)
    # reorder to AHEPA order where present; pad missing later
    raw.filter(0.5, 40.0, verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-3:
        raw.resample(target_sfreq, verbose="ERROR")
    data = raw.get_data()  # (C, T)
    # pad / align to 19
    C_full = np.zeros((len(AHEPA_19), data.shape[1]), dtype=np.float64)
    name_to_i = {n: i for i, n in enumerate(keep)}
    for j, name in enumerate(AHEPA_19):
        if name in name_to_i:
            C_full[j] = data[name_to_i[name]]
    sfreq = float(raw.info["sfreq"])
    win = int(round(epoch_sec * sfreq))
    if win < 8 or C_full.shape[1] < win:
        raise RuntimeError(f"Recording too short: {path}")
    # non-overlapping epochs from the middle of the recording
    n_possible = C_full.shape[1] // win
    n_ep = min(max_epochs, max(n_possible, 1))
    start0 = max(0, (C_full.shape[1] - n_ep * win) // 2)
    eps = []
    for i in range(n_ep):
        a = start0 + i * win
        eps.append(C_full[:, a : a + win])
    return np.stack(eps, axis=0).astype(np.float32), sfreq
