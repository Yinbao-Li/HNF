# -*- coding: utf-8 -*-
"""STEAD official-split dataset for detection and P/S picking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from hnf.robust_aug import (add_gap, add_lowfreq, add_real_noise, add_reverb, add_second_event, emergent_onset,
                            response_variation)
from hnf.shift_aug import shift_with_noise


STEAD_DIR = Path(__file__).resolve().parents[1] / "STEAD"
TEST_SPLIT = set(np.load(STEAD_DIR / "test.npy", allow_pickle=True).tolist())


def gaussian_pick_label(center_idx: int, seq_len: int, sigma_samples: float) -> torch.Tensor:
    """EQTransformer-style soft arrival label along the time axis."""
    if center_idx < 0:
        return torch.zeros(seq_len, dtype=torch.float32)
    t = torch.arange(seq_len, dtype=torch.float32)
    sigma = max(float(sigma_samples), 1.0)
    return torch.exp(-0.5 * ((t - float(center_idx)) / sigma) ** 2)


@dataclass
class SteadTraceRef:
    trace_name: str
    chunk: int
    is_event: int
    p_sample: Optional[int]
    s_sample: Optional[int]
    source_distance_km: Optional[float] = None
    source_depth_km: Optional[float] = None
    p_manual: int = 0


class _CompactRefs:
    """Read-only sequence of SteadTraceRef backed by numpy columns.

    A list of ~1M Python objects gets copied page-by-page into every forked
    DataLoader worker (refcount writes), costing ~1.5 GB per worker.
    """

    def __init__(self, refs: list[SteadTraceRef]):
        self.name = np.array([r.trace_name for r in refs], dtype="S")
        self.chunk = np.array([r.chunk for r in refs], dtype=np.int8)
        self.is_event = np.array([r.is_event for r in refs], dtype=np.int8)
        self.p = np.array([-1 if r.p_sample is None else r.p_sample for r in refs], dtype=np.int32)
        self.s = np.array([-1 if r.s_sample is None else r.s_sample for r in refs], dtype=np.int32)
        nan = float("nan")
        self.dist = np.array(
            [nan if r.source_distance_km is None else r.source_distance_km for r in refs], dtype=np.float32
        )
        self.depth = np.array(
            [nan if r.source_depth_km is None else r.source_depth_km for r in refs], dtype=np.float32
        )
        self.manual = np.array([r.p_manual for r in refs], dtype=np.int8)

    def __len__(self) -> int:
        return len(self.name)

    def __getitem__(self, idx: int) -> SteadTraceRef:
        p, s = int(self.p[idx]), int(self.s[idx])
        dist, depth = float(self.dist[idx]), float(self.depth[idx])
        return SteadTraceRef(
            self.name[idx].decode(),
            int(self.chunk[idx]),
            int(self.is_event[idx]),
            None if p < 0 else p,
            None if s < 0 else s,
            None if np.isnan(dist) else dist,
            None if np.isnan(depth) else depth,
            int(self.manual[idx]),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class STEADPickingDataset(Dataset):
    """STEAD dataset aligned to EQTransformer test split."""

    def __init__(
        self,
        split: str,
        seq_len: int = 300,
        max_event_traces: Optional[int] = None,
        max_noise_traces: Optional[int] = None,
        label_sigma_sec: float = 0.25,
        seed: int = 42,
        augment: bool = False,
        aug_amp_scale: tuple[float, float] = (0.5, 2.0),
        aug_noise_snr_db: tuple[float, float] = (5.0, 20.0),
        aug_time_shift_sec: float = 0.05,
        aug_channel_scale: tuple[float, float] = (0.8, 1.2),
        load_geometry: bool = True,
        fresh_augment: bool = False,
        aug_shift_prob: float = 0.0,
        aug_shift_p_range: tuple[float, float] = (1.0, 50.0),
        aug_shift_noise_max_sec: float = 30.0,
        eval_shift: bool = False,
        aug_channel_drop_prob: float = 0.0,
        aug_gauss_prob: float = 1.0,
        aug_second_event_prob: float = 0.0,
        aug_real_noise_prob: float = 0.0,
        aug_real_noise_snr_db: tuple[float, float] = (0.0, 20.0),
        aug_response_prob: float = 0.0,
        aug_gap_prob: float = 0.0,
        aug_phase_absent_prob: float = 0.0,
        aug_lowfreq_prob: float = 0.0,
        aug_reverb_prob: float = 0.0,
        aug_emergent_prob: float = 0.0,
        load_p_status: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.seq_len = seq_len
        self.seed = seed
        self.sample_rate = 100
        self.original_len = 6000
        self.label_sigma_samples = max(1.0, label_sigma_sec * seq_len / 60.0)
        self.augment = augment and split == "train"
        self.fresh_augment = fresh_augment
        self.aug_amp_scale = aug_amp_scale
        self.aug_noise_snr_db = aug_noise_snr_db
        self.aug_time_shift_sec = aug_time_shift_sec
        self.aug_channel_scale = aug_channel_scale
        self.aug_shift_prob = float(aug_shift_prob)
        self.aug_shift_p_range = aug_shift_p_range
        self.aug_shift_noise_max_sec = float(aug_shift_noise_max_sec)
        self._noise_idx: Optional[np.ndarray] = None
        self.eval_shift = bool(eval_shift)
        self.aug_channel_drop_prob = float(aug_channel_drop_prob)
        self.aug_gauss_prob = float(aug_gauss_prob)
        self.aug_second_event_prob = float(aug_second_event_prob)
        self.aug_real_noise_prob = float(aug_real_noise_prob)
        self.aug_real_noise_snr_db = aug_real_noise_snr_db
        self.aug_response_prob = float(aug_response_prob)
        self.aug_gap_prob = float(aug_gap_prob)
        self.aug_phase_absent_prob = float(aug_phase_absent_prob)
        self.aug_lowfreq_prob = float(aug_lowfreq_prob)
        self.aug_reverb_prob = float(aug_reverb_prob)
        self.aug_emergent_prob = float(aug_emergent_prob)
        self.load_p_status = bool(load_p_status)
        self._event_idx: Optional[np.ndarray] = None
        self.load_geometry = load_geometry
        self._handles: dict[int, h5py.File] = {}
        self._paths = {
            i: STEAD_DIR / f"chunk{i}_eofextract" / f"chunk{i}.hdf5"
            for i in range(1, 7)
        }

        refs = self._build_refs()
        events = [r for r in refs if r.is_event == 1]
        noise = [r for r in refs if r.is_event == 0]

        rng = np.random.default_rng(seed)
        if split != "test":
            rng.shuffle(events)
            rng.shuffle(noise)

        if max_event_traces is not None:
            events = events[:max_event_traces]
        if max_noise_traces is not None:
            noise = noise[:max_noise_traces]

        refs = events + noise
        if split != "test":
            rng.shuffle(refs)
        self.refs = _CompactRefs(refs)
        del refs, events, noise

    def _build_refs(self) -> list[SteadTraceRef]:
        all_train_val: list[SteadTraceRef] = []
        test_refs: list[SteadTraceRef] = []

        for chunk in range(1, 7):
            csv_path = STEAD_DIR / f"chunk{chunk}_eofextract" / f"chunk{chunk}.csv"
            cols = ["trace_name", "trace_category"]
            if chunk > 1:
                cols += ["p_arrival_sample", "s_arrival_sample"]
                if self.load_p_status:
                    cols += ["p_status"]
            if self.load_geometry:
                cols += ["source_distance_km", "source_depth_km"]
            df = pd.read_csv(csv_path, usecols=cols, low_memory=False)

            for row in df.itertuples(index=False):
                name = row.trace_name
                is_event = 0 if row.trace_category == "noise" else 1
                p_sample = int(row.p_arrival_sample) if is_event and not pd.isna(row.p_arrival_sample) else None
                s_sample = int(row.s_arrival_sample) if is_event and not pd.isna(row.s_arrival_sample) else None
                dist = None
                depth = None
                if self.load_geometry and is_event:
                    if hasattr(row, "source_distance_km") and not pd.isna(row.source_distance_km):
                        dist = float(row.source_distance_km)
                    if hasattr(row, "source_depth_km") and not pd.isna(row.source_depth_km):
                        depth = float(row.source_depth_km)
                manual = int(getattr(row, "p_status", "") == "manual")
                ref = SteadTraceRef(name, chunk, is_event, p_sample, s_sample, dist, depth, manual)
                if name in TEST_SPLIT:
                    test_refs.append(ref)
                else:
                    all_train_val.append(ref)

        if self.split == "test":
            return test_refs

        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_train_val)
        val_size = int(len(all_train_val) * 0.05)
        val_refs = all_train_val[:val_size]
        train_refs = all_train_val[val_size:]
        return train_refs if self.split == "train" else val_refs

    def __len__(self) -> int:
        return len(self.refs)

    def _get_handle(self, chunk: int) -> h5py.File:
        handle = self._handles.get(chunk)
        if handle is None:
            handle = h5py.File(self._paths[chunk], "r")
            self._handles[chunk] = handle
        return handle

    def _apply_augment(
        self,
        x: torch.Tensor,
        p_idx: int,
        s_idx: int,
        p_valid: float,
        s_valid: float,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, int, int, float, float]:
        """Waveform augmentations with synchronized pick indices."""
        if not self.augment:
            return x, p_idx, s_idx, p_valid, s_valid

        # Per-channel amplitude scaling (x is T×C)
        ch_scale = rng.uniform(*self.aug_channel_scale, size=(x.size(-1),)).astype(np.float32)
        x = x * torch.from_numpy(ch_scale)

        # Global amplitude scaling
        amp = float(rng.uniform(*self.aug_amp_scale))
        x = x * amp

        # Additive Gaussian noise at random SNR
        if self.aug_gauss_prob >= 1.0 or rng.random() < self.aug_gauss_prob:
            snr_db = float(rng.uniform(*self.aug_noise_snr_db))
            sig_pow = float((x**2).mean().item())
            noise_pow = sig_pow / max(10 ** (snr_db / 10.0), 1e-8)
            x = x + torch.randn_like(x) * (noise_pow ** 0.5)

        # Small temporal shift (seconds -> bins)
        max_shift_bins = int(round(self.aug_time_shift_sec * self.seq_len / 60.0))
        if max_shift_bins > 0:
            shift = int(rng.integers(-max_shift_bins, max_shift_bins + 1))
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)
                if p_valid > 0:
                    p_idx = p_idx + shift
                    if p_idx < 0 or p_idx >= self.seq_len:
                        p_valid = 0.0
                        p_idx = -1
                if s_valid > 0:
                    s_idx = s_idx + shift
                    if s_idx < 0 or s_idx >= self.seq_len:
                        s_valid = 0.0
                        s_idx = -1

        # Channel loss (channel order E, N, Z): horizontals, vertical, or one random channel
        if self.aug_channel_drop_prob > 0 and rng.random() < self.aug_channel_drop_prob:
            mode = int(rng.integers(3))
            drop = [0, 1] if mode == 0 else [2] if mode == 1 else [int(rng.integers(3))]
            x = x.clone()
            x[:, drop] = 0.0

        return x, p_idx, s_idx, p_valid, s_valid

    def _load_random(self, pool: str, rng: np.random.Generator):
        if self._noise_idx is None:
            self._noise_idx = np.flatnonzero(np.asarray(self.refs.is_event) == 0)
        if self._event_idx is None:
            self._event_idx = np.flatnonzero(np.asarray(self.refs.is_event) == 1)
        idx = self._noise_idx if pool == "noise" else self._event_idx
        if not len(idx):
            return None, None
        r = self.refs[int(rng.choice(idx))]
        return np.asarray(self._get_handle(r.chunk)["data"][r.trace_name][()], dtype=np.float32).T, r

    def _robust_raw(self, wave: np.ndarray, ref, p_s, s_s, rng: np.random.Generator):
        """Second event, real noise and instrument-response variation; returns (wave, p2, s2)."""
        p2 = s2 = None
        is_ev = ref.is_event == 1 and p_s is not None
        if is_ev and self.aug_second_event_prob > 0 and rng.random() < self.aug_second_event_prob:
            w2, r2 = self._load_random("event", rng)
            if w2 is not None and r2.p_sample is not None:
                res = add_second_event(wave, int(p_s), None if s_s is None else int(s_s), w2, int(r2.p_sample),
                                       r2.s_sample, rng)
                if res is not None:
                    wave, p2, s2 = res
        if is_ev and self.aug_emergent_prob > 0 and rng.random() < self.aug_emergent_prob:
            wave = emergent_onset(wave, int(p_s), rng)
        if self.aug_real_noise_prob > 0 and rng.random() < self.aug_real_noise_prob:
            nz, _ = self._load_random("noise", rng)
            if nz is not None:
                snr = float(rng.uniform(*self.aug_real_noise_snr_db))
                wave = add_real_noise(wave, int(p_s) if is_ev else None, nz, snr)
        if self.aug_response_prob > 0 and rng.random() < self.aug_response_prob:
            wave = response_variation(wave, rng)
        if self.aug_reverb_prob > 0 and rng.random() < self.aug_reverb_prob:
            wave = add_reverb(wave, rng)
        if self.aug_lowfreq_prob > 0 and rng.random() < self.aug_lowfreq_prob:
            wave = add_lowfreq(wave, rng)
        return wave, p2, s2

    def _random_shift(self, wave: np.ndarray, ref, rng: np.random.Generator):
        """Large random time shift with real-noise fill; returns (wave, p_sample, s_sample)."""
        p_s, s_s = ref.p_sample, ref.s_sample
        if ref.is_event == 1 and p_s is not None:
            lo, hi = self.aug_shift_p_range
            d = int(round(rng.uniform(lo, hi) * self.sample_rate)) - int(p_s)
            sp = int(s_s) - int(p_s) if s_s is not None else None
            if self.aug_phase_absent_prob > 0 and sp is not None and sp > 150:
                u = rng.random()
                if u < self.aug_phase_absent_prob / 2:
                    # P before the window start, S inside
                    d = int(rng.integers(50, min(sp - 50, 2000))) - int(s_s)
                elif u < self.aug_phase_absent_prob:
                    # S after the window end, P inside
                    lo_p, hi_p = max(self.original_len - sp + 50, 100), self.original_len - 100
                    if lo_p < hi_p:
                        d = int(rng.integers(lo_p, hi_p)) - int(p_s)
        else:
            m = int(self.aug_shift_noise_max_sec * self.sample_rate)
            d = int(rng.integers(-m, m + 1))
        if d == 0:
            return wave, p_s, s_s
        if self._noise_idx is None:
            self._noise_idx = np.flatnonzero(np.asarray(self.refs.is_event) == 0)
        noise = None
        if len(self._noise_idx):
            nref = self.refs[int(rng.choice(self._noise_idx))]
            noise = np.asarray(self._get_handle(nref.chunk)["data"][nref.trace_name][()], dtype=np.float32).T
        wave = shift_with_noise(wave, d, noise)
        t = wave.shape[-1]
        p_s = p_s + d if p_s is not None and 0 <= p_s + d < t else None
        s_s = s_s + d if s_s is not None and 0 <= s_s + d < t else None
        return wave, p_s, s_s

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.refs[idx]
        waveform = self._get_handle(ref.chunk)["data"][ref.trace_name][()]  # (6000, 3)
        if self.augment and self.fresh_augment:
            # draw from the (per-worker, per-epoch advancing) torch RNG so each
            # epoch sees a different augmentation of the same trace
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        else:
            rng = np.random.default_rng(self.seed + idx * 9973)
        p_sample, s_sample = ref.p_sample, ref.s_sample
        wave = np.asarray(waveform, dtype=np.float32).T  # (3, T)
        if self.eval_shift:
            wave, p_sample, s_sample = self._random_shift(wave, ref, rng)
        elif self.augment and self.aug_shift_prob > 0 and rng.random() < self.aug_shift_prob:
            wave, p_sample, s_sample = self._random_shift(wave, ref, rng)
        p2_sample = s2_sample = None
        if self.augment:
            wave, p2_sample, s2_sample = self._robust_raw(wave, ref, p_sample, s_sample, rng)
        x = torch.from_numpy(np.ascontiguousarray(wave))  # (3, T)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        x = F.interpolate(x.unsqueeze(0), size=self.seq_len, mode="linear", align_corners=False).squeeze(0)
        x = x.transpose(0, 1)  # (T, 3)

        t = torch.linspace(0.0, 60.0, self.seq_len, dtype=torch.float32).unsqueeze(-1)
        det = torch.tensor(float(ref.is_event), dtype=torch.float32)

        scale = self.seq_len / self.original_len
        p_idx = -1
        s_idx = -1
        p_valid = 0.0
        s_valid = 0.0
        if p_sample is not None:
            p_idx = min(self.seq_len - 1, max(0, int(round(p_sample * scale))))
            p_valid = 1.0
        if s_sample is not None:
            s_idx = min(self.seq_len - 1, max(0, int(round(s_sample * scale))))
            s_valid = 1.0
        # an event phase pushed out of the window is known to be absent: supervise with a zero target
        absent_ok = self.augment and self.aug_phase_absent_prob > 0 and ref.is_event == 1
        p_absent = absent_ok and ref.p_sample is not None and p_sample is None
        s_absent = absent_ok and ref.s_sample is not None and s_sample is None

        p_before = p_idx
        x, p_idx, s_idx, p_valid, s_valid = self._apply_augment(
            x, p_idx, s_idx, p_valid, s_valid, rng
        )
        jitter = p_idx - p_before if (p_valid > 0 and p_before >= 0) else 0
        p2_idx = s2_idx = -1
        if p2_sample is not None:
            p2_idx = int(round(p2_sample * scale)) + jitter
            p2_idx = p2_idx if 0 <= p2_idx < self.seq_len else -1
        if s2_sample is not None:
            s2_idx = int(round(s2_sample * scale)) + jitter
            s2_idx = s2_idx if 0 <= s2_idx < self.seq_len else -1
        if self.augment and self.aug_gap_prob > 0 and rng.random() < self.aug_gap_prob:
            x = add_gap(x, [p_idx, s_idx, p2_idx, s2_idx], rng)

        p_target = gaussian_pick_label(p_idx, self.seq_len, self.label_sigma_samples)
        s_target = gaussian_pick_label(s_idx, self.seq_len, self.label_sigma_samples)
        if p2_idx >= 0:
            p_target = torch.maximum(p_target, gaussian_pick_label(p2_idx, self.seq_len, self.label_sigma_samples))
        if s2_idx >= 0:
            s_target = torch.maximum(s_target, gaussian_pick_label(s2_idx, self.seq_len, self.label_sigma_samples))

        return {
            "x": x,
            "t": t,
            "det": det,
            "p_idx": torch.tensor(p_idx, dtype=torch.long),
            "s_idx": torch.tensor(s_idx, dtype=torch.long),
            "p_valid": torch.tensor(p_valid, dtype=torch.float32),
            "s_valid": torch.tensor(s_valid, dtype=torch.float32),
            "p_target": p_target,
            "s_target": s_target,
            "p2_idx": torch.tensor(p2_idx, dtype=torch.long),
            "s2_idx": torch.tensor(s2_idx, dtype=torch.long),
            "p_mask": torch.tensor(float(p_valid > 0 or p_absent), dtype=torch.float32),
            "s_mask": torch.tensor(float(s_valid > 0 or s_absent), dtype=torch.float32),
            "trace_name": ref.trace_name,
            "p_manual": torch.tensor(float(ref.p_manual), dtype=torch.float32),
            "source_distance_km": torch.tensor(
                float(ref.source_distance_km) if ref.source_distance_km is not None else float("nan"),
                dtype=torch.float32,
            ),
            "source_depth_km": torch.tensor(
                float(ref.source_depth_km) if ref.source_depth_km is not None else float("nan"),
                dtype=torch.float32,
            ),
        }

