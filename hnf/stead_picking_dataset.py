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
        self.aug_amp_scale = aug_amp_scale
        self.aug_noise_snr_db = aug_noise_snr_db
        self.aug_time_shift_sec = aug_time_shift_sec
        self.aug_channel_scale = aug_channel_scale
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

        self.refs = events + noise
        if split != "test":
            rng.shuffle(self.refs)

    def _build_refs(self) -> list[SteadTraceRef]:
        all_train_val: list[SteadTraceRef] = []
        test_refs: list[SteadTraceRef] = []

        for chunk in range(1, 7):
            csv_path = STEAD_DIR / f"chunk{chunk}_eofextract" / f"chunk{chunk}.csv"
            cols = ["trace_name", "trace_category"]
            if chunk > 1:
                cols += ["p_arrival_sample", "s_arrival_sample"]
            df = pd.read_csv(csv_path, usecols=cols, low_memory=False)

            for row in df.itertuples(index=False):
                name = row.trace_name
                is_event = 0 if row.trace_category == "noise" else 1
                p_sample = int(row.p_arrival_sample) if is_event and not pd.isna(row.p_arrival_sample) else None
                s_sample = int(row.s_arrival_sample) if is_event and not pd.isna(row.s_arrival_sample) else None
                ref = SteadTraceRef(name, chunk, is_event, p_sample, s_sample)
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

        return x, p_idx, s_idx, p_valid, s_valid

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.refs[idx]
        waveform = self._get_handle(ref.chunk)["data"][ref.trace_name][()]  # (6000, 3)
        x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).transpose(0, 1)  # (3, T)
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
        if ref.p_sample is not None:
            p_idx = min(self.seq_len - 1, max(0, int(round(ref.p_sample * scale))))
            p_valid = 1.0
        if ref.s_sample is not None:
            s_idx = min(self.seq_len - 1, max(0, int(round(ref.s_sample * scale))))
            s_valid = 1.0

        rng = np.random.default_rng(self.seed + idx * 9973)
        x, p_idx, s_idx, p_valid, s_valid = self._apply_augment(
            x, p_idx, s_idx, p_valid, s_valid, rng
        )

        p_target = gaussian_pick_label(p_idx, self.seq_len, self.label_sigma_samples)
        s_target = gaussian_pick_label(s_idx, self.seq_len, self.label_sigma_samples)

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
        }

