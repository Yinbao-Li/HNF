# -*- coding: utf-8 -*-
"""STEAD waveform dataset adapter for HNF classification."""

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
CHUNK1_DIR = STEAD_DIR / "chunk1_eofextract"
CHUNK2_DIR = STEAD_DIR / "chunk2_eofextract"


@dataclass
class SteadSampleRef:
    trace_name: str
    label: int
    source: str


class STEADDataset(Dataset):
    """Balanced STEAD binary dataset: noise vs earthquake_local."""

    def __init__(
        self,
        split: str,
        seq_len: int = 200,
        max_per_class: Optional[int] = 10000,
        seed: int = 42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.seq_len = seq_len
        self.seed = seed
        self._handles: dict[str, h5py.File] = {}
        self._paths = {
            "chunk1": CHUNK1_DIR / "chunk1.hdf5",
            "chunk2": CHUNK2_DIR / "chunk2.hdf5",
        }

        noise_df = pd.read_csv(CHUNK1_DIR / "chunk1.csv", usecols=["trace_name", "trace_category"])
        eq_df = pd.read_csv(CHUNK2_DIR / "chunk2.csv", usecols=["trace_name", "trace_category"])
        noise_df = noise_df[noise_df["trace_category"] == "noise"].copy()
        eq_df = eq_df[eq_df["trace_category"] == "earthquake_local"].copy()

        refs_noise = self._split_refs(noise_df["trace_name"].tolist(), 0, "chunk1", max_per_class)
        refs_eq = self._split_refs(eq_df["trace_name"].tolist(), 1, "chunk2", max_per_class)
        self.refs = refs_noise + refs_eq

        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])
        rng.shuffle(self.refs)

    def _split_refs(
        self,
        names: list[str],
        label: int,
        source: str,
        max_per_class: Optional[int],
    ) -> list[SteadSampleRef]:
        rng = np.random.default_rng(self.seed + label * 17)
        names = list(names)
        rng.shuffle(names)
        n = len(names)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        if self.split == "train":
            chunk = names[:train_end]
        elif self.split == "val":
            chunk = names[train_end:val_end]
        else:
            chunk = names[val_end:]
        if max_per_class is not None:
            chunk = chunk[:max_per_class]
        return [SteadSampleRef(trace_name=name, label=label, source=source) for name in chunk]

    def __len__(self) -> int:
        return len(self.refs)

    def _get_handle(self, source: str) -> h5py.File:
        handle = self._handles.get(source)
        if handle is None:
            handle = h5py.File(self._paths[source], "r")
            self._handles[source] = handle
        return handle

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.refs[idx]
        handle = self._get_handle(ref.source)
        waveform = handle["data"][ref.trace_name][()]  # (6000, 3)
        x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).transpose(0, 1)  # (3, T)

        # Per-channel standardization keeps the classifier focused on waveform shape.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std

        x = F.interpolate(x.unsqueeze(0), size=self.seq_len, mode="linear", align_corners=False).squeeze(0)
        x = x.transpose(0, 1)  # (T, 3)
        t = torch.linspace(0.0, 1.0, self.seq_len, dtype=torch.float32).unsqueeze(-1)

        return {
            "x": x,
            "t": t,
            "y": torch.tensor(ref.label, dtype=torch.long),
        }

