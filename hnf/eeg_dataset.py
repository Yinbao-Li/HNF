# -*- coding: utf-8 -*-
"""BIDS EEG dataset for OpenNeuro ds004504 (AD / FTD / HC)."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Paper taxonomy: HC / MCI / AD. ds004504 has CN / FTD / AD;
# by default FTD is mapped to the MCI slot (class 1).
LABEL_TO_ID = {"HC": 0, "MCI": 1, "AD": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Common Group spellings in participants.tsv
_GROUP_ALIASES = {
    "cn": "HC",
    "hc": "HC",
    "control": "HC",
    "healthy": "HC",
    "c": "HC",
    "ftd": "FTD",
    "f": "FTD",
    "frontotemporal": "FTD",
    "ad": "AD",
    "a": "AD",
    "alzheimers": "AD",
    "alzheimer": "AD",
    "mci": "MCI",
}

STANDARD_10_20 = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
)


@dataclass(frozen=True)
class SubjectRef:
    subject_id: str
    group_raw: str
    label: int
    set_path: Path


def _normalize_group(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in _GROUP_ALIASES:
        return _GROUP_ALIASES[key]
    upper = str(raw).strip().upper()
    if upper in {"HC", "MCI", "AD", "FTD", "CN"}:
        return "HC" if upper == "CN" else upper
    raise ValueError(f"Unrecognized Group value: {raw!r}")


def map_group_to_label(group: str, *, ftd_as_mci: bool = True) -> int:
    """Map clinical group to classifier label id.

    With ``ftd_as_mci=True`` (default), FTD occupies the MCI class slot so the
    head stays 3-way (HC/MCI/AD) as in the paper taxonomy.
    """
    g = _normalize_group(group)
    if g == "FTD":
        if not ftd_as_mci:
            raise ValueError("FTD requires ftd_as_mci=True for the 3-way HC/MCI/AD head")
        return LABEL_TO_ID["MCI"]
    if g not in LABEL_TO_ID:
        raise ValueError(f"Cannot map group {group!r} ({g}) to HC/MCI/AD")
    return LABEL_TO_ID[g]


def _find_set_file(data_dir: Path, subject_id: str) -> Optional[Path]:
    """Prefer preprocessed derivatives .set, then raw BIDS .set."""
    patterns = [
        data_dir / "derivatives" / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set",
        data_dir / "derivatives" / subject_id / f"{subject_id}_task-eyesclosed_eeg.set",
        data_dir / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set",
        data_dir / subject_id / f"{subject_id}_task-eyesclosed_eeg.set",
    ]
    for p in patterns:
        if p.is_file():
            return p
    # Fallback glob
    hits = sorted(data_dir.glob(f"**/{subject_id}*task-eyesclosed*eeg.set"))
    return hits[0] if hits else None


def _read_participants(data_dir: Path) -> list[tuple[str, str]]:
    tsv = data_dir / "participants.tsv"
    if not tsv.is_file():
        # Some downloads nest one level
        nested = list(data_dir.glob("**/participants.tsv"))
        if not nested:
            return []
        tsv = nested[0]
        data_dir = tsv.parent
    rows: list[tuple[str, str]] = []
    with tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = (row.get("participant_id") or row.get("subject") or "").strip()
            group = (row.get("Group") or row.get("group") or row.get("diagnosis") or "").strip()
            if not sid or not group:
                continue
            if not sid.startswith("sub-"):
                sid = f"sub-{sid}"
            rows.append((sid, group))
    return rows


def _subject_split(
    subject_ids: list[str],
    *,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Deterministic subject-level train/val/test assignment."""
    rng = np.random.default_rng(seed)
    ids = list(subject_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_ratio))) if n >= 5 else max(0, n // 5)
    n_val = max(1, int(round(n * val_ratio))) if n >= 5 else max(0, n // 5)
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    split_map: dict[str, str] = {}
    for sid in ids:
        if sid in test_ids:
            split_map[sid] = "test"
        elif sid in val_ids:
            split_map[sid] = "val"
        else:
            split_map[sid] = "train"
    return split_map


def _resample_channels(data: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """Resample (C, T) with linear interpolation along time."""
    if abs(src_hz - dst_hz) < 1e-6:
        return data.astype(np.float32, copy=False)
    c, t = data.shape
    duration = t / float(src_hz)
    new_t = max(1, int(round(duration * dst_hz)))
    src_x = np.linspace(0.0, 1.0, t, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, new_t, dtype=np.float64)
    out = np.empty((c, new_t), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(dst_x, src_x, data[i].astype(np.float64)).astype(np.float32)
    return out


def _pick_19_channels(raw) -> np.ndarray:
    """Return (19, T) array from an MNE Raw, preferring 10-20 names."""
    names = [ch.upper().replace("EEG", "").replace("-", "").strip() for ch in raw.ch_names]
    name_to_idx = {n: i for i, n in enumerate(names)}
    indices: list[int] = []
    for ch in STANDARD_10_20:
        key = ch.upper()
        if key in name_to_idx:
            indices.append(name_to_idx[key])
        else:
            # EEGLAB sometimes uses T7/T8 instead of T3/T4 etc.
            aliases = {
                "T3": "T7",
                "T4": "T8",
                "T5": "P7",
                "T6": "P8",
            }
            alt = aliases.get(ch, "").upper()
            if alt and alt in name_to_idx:
                indices.append(name_to_idx[alt])
    data = raw.get_data()
    if len(indices) >= 19:
        return data[indices[:19]]
    if data.shape[0] >= 19:
        return data[:19]
    # Pad missing channels with zeros
    out = np.zeros((19, data.shape[1]), dtype=np.float64)
    out[: data.shape[0]] = data
    return out


def _load_set(path: Path) -> tuple[np.ndarray, float]:
    """Load EEGLAB .set → (19, T) float64 and sample rate."""
    import mne

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    data = _pick_19_channels(raw)
    sfreq = float(raw.info["sfreq"])
    return data, sfreq


def make_synthetic_subjects(
    n_per_class: int = 4,
    duration_sec: float = 60.0,
    sfreq: float = 500.0,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, int]]:
    """CPU demo waveforms when OpenNeuro data is not downloaded yet."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sfreq), dtype=np.float64) / sfreq
    out: dict[str, tuple[np.ndarray, int]] = {}
    for label, tag in enumerate(("HC", "MCI", "AD")):
        base_f = 10.0 - 2.0 * label
        for i in range(n_per_class):
            sid = f"sub-synth{tag}{i:02d}"
            waves = []
            for c in range(19):
                phase = rng.uniform(0, 2 * np.pi)
                amp = 1.0 + 0.3 * label
                sig = amp * np.sin(2 * np.pi * (base_f + 0.2 * c) * t + phase)
                sig += 0.15 * rng.standard_normal(t.shape[0])
                waves.append(sig.astype(np.float32))
            out[sid] = (np.stack(waves, axis=0), label)
    return out


class EEGDataset(Dataset):
    """Subject-level split EEG epochs from ds004504 (or synthetic demo).

    Args:
        data_dir: Root of the OpenNeuro download (contains participants.tsv).
        split: One of ``train`` / ``val`` / ``test``.
        test_ratio / val_ratio: Subject fractions for hold-out splits.
        seed: Split RNG seed.
        sample_rate: Target Hz (paper uses 128).
        epoch_sec: Epoch length in seconds (paper uses 10).
        stride_sec: Sliding-window stride; ``None`` → non-overlap.
        ftd_as_mci: Map FTD → MCI class slot.
        use_derivatives: Prefer preprocessed derivatives .set files.
        synthetic_if_missing: Build synthetic subjects when data absent.
        cache_resampled: Keep resampled full recordings in memory.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        test_ratio: float = 0.2,
        val_ratio: float = 0.15,
        seed: int = 42,
        sample_rate: int = 128,
        epoch_sec: float = 10.0,
        stride_sec: Optional[float] = 5.0,
        n_channels: int = 19,
        ftd_as_mci: bool = True,
        use_derivatives: bool = True,
        synthetic_if_missing: bool = True,
        cache_resampled: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.sample_rate = int(sample_rate)
        self.epoch_sec = float(epoch_sec)
        self.stride_sec = float(stride_sec) if stride_sec is not None else float(epoch_sec)
        self.n_channels = int(n_channels)
        self.epoch_samples = int(round(self.epoch_sec * self.sample_rate))
        self.stride_samples = max(1, int(round(self.stride_sec * self.sample_rate)))
        self.ftd_as_mci = ftd_as_mci
        self.cache_resampled = cache_resampled
        self.data_dir = Path(data_dir)
        self._cache: dict[str, np.ndarray] = {}
        self._synth_native: dict[str, tuple[np.ndarray, float]] = {}

        subjects = self._discover_subjects(use_derivatives=use_derivatives)
        if not subjects and synthetic_if_missing:
            subjects = self._synthetic_subjects()
        if not subjects:
            raise FileNotFoundError(
                f"No EEG subjects under {self.data_dir}. "
                "Download ds004504 or enable synthetic_if_missing."
            )

        split_map = _subject_split(
            [s.subject_id for s in subjects],
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        self.subjects = [s for s in subjects if split_map[s.subject_id] == split]
        if not self.subjects:
            raise RuntimeError(f"Split {split!r} is empty after subject partitioning")

        self.epochs: list[tuple[SubjectRef, int]] = []
        for ref in self.subjects:
            n_samp = self._recording_length(ref)
            if n_samp < self.epoch_samples:
                continue
            starts = range(0, n_samp - self.epoch_samples + 1, self.stride_samples)
            for start in starts:
                self.epochs.append((ref, start))
        if not self.epochs:
            raise RuntimeError(f"No epochs constructed for split={split}")

    def _synthetic_subjects(self) -> list[SubjectRef]:
        synth = make_synthetic_subjects(seed=0)
        refs: list[SubjectRef] = []
        for sid, (arr, label) in synth.items():
            self._cache[sid] = arr  # already conceptually "500 Hz"; resample on read
            # Store as if native 500 Hz then resample in _get_recording
            refs.append(
                SubjectRef(
                    subject_id=sid,
                    group_raw=ID_TO_LABEL[label],
                    label=label,
                    set_path=Path(f"synthetic://{sid}"),
                )
            )
        # Mark synthetic native rate
        self._synth_native = {sid: (arr, 500.0) for sid, (arr, _) in synth.items()}
        return refs

    def _discover_subjects(self, *, use_derivatives: bool) -> list[SubjectRef]:
        del use_derivatives  # discovery prefers derivatives via _find_set_file order
        rows = _read_participants(self.data_dir)
        if not rows:
            # Infer from folder names only (no labels) → skip
            return []
        refs: list[SubjectRef] = []
        for sid, group in rows:
            path = _find_set_file(self.data_dir, sid)
            if path is None:
                continue
            try:
                label = map_group_to_label(group, ftd_as_mci=self.ftd_as_mci)
            except ValueError:
                continue
            refs.append(
                SubjectRef(
                    subject_id=sid,
                    group_raw=group,
                    label=label,
                    set_path=path,
                )
            )
        return refs

    def _get_recording(self, ref: SubjectRef) -> np.ndarray:
        if ref.subject_id in self._cache:
            return self._cache[ref.subject_id]
        if str(ref.set_path).startswith("synthetic://"):
            arr, hz = self._synth_native[ref.subject_id]
            rs = _resample_channels(arr, hz, float(self.sample_rate))
            if self.cache_resampled:
                self._cache[ref.subject_id] = rs
            return rs
        data, hz = _load_set(ref.set_path)
        rs = _resample_channels(data, hz, float(self.sample_rate))
        # z-score per channel
        mu = rs.mean(axis=1, keepdims=True)
        sd = rs.std(axis=1, keepdims=True) + 1e-6
        rs = ((rs - mu) / sd).astype(np.float32)
        if self.cache_resampled:
            self._cache[ref.subject_id] = rs
        return rs

    def _recording_length(self, ref: SubjectRef) -> int:
        return int(self._get_recording(ref).shape[1])

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        ref, start = self.epochs[idx]
        rec = self._get_recording(ref)
        epoch = rec[:, start : start + self.epoch_samples]
        if epoch.shape[1] < self.epoch_samples:
            pad = np.zeros((self.n_channels, self.epoch_samples), dtype=np.float32)
            pad[:, : epoch.shape[1]] = epoch
            epoch = pad
        x = torch.from_numpy(np.ascontiguousarray(epoch[: self.n_channels], dtype=np.float32))
        return {
            "x": x,
            "label": int(ref.label),
            "subject_id": ref.subject_id,
            "start": int(start),
        }

    def subject_ids(self) -> list[str]:
        return [s.subject_id for s in self.subjects]

    def fingerprint(self) -> str:
        blob = "|".join(sorted(self.subject_ids())) + f"|{self.split}|{self.sample_rate}"
        return hashlib.md5(blob.encode()).hexdigest()[:10]
