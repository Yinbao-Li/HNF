# -*- coding: utf-8 -*-
"""STEAD real-waveform dataset for Zhizi macro-head training (no true vp/vs)."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.utils.data import Dataset

from hnf.inversion_1d import default_synth_model
from hnf.picking_metrics import idx_to_sec
from hnf.stead_picking_dataset import STEADPickingDataset


def _scalar_tensor(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.reshape(-1)[0].item())
    return float(v)


def encode_neutral_geometry_tensor(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Canonical neutral geometry (50 km, 10 km depth) for non-STEAD inference."""
    return encode_geometry_tensor(50.0, 10.0, device=device, dtype=dtype)


def encode_geometry_tensor(
    distance_km: float,
    depth_km: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized epicentral distance + source depth for macro head conditioning."""
    d = math.log1p(max(float(distance_km), 1.0) / 50.0)
    z = max(float(depth_km), 1.0) / 30.0
    return torch.tensor([d, z], device=device, dtype=dtype)


class SteadZhiziInversionDataset(Dataset):
    """
    One STEAD event trace per item: waveform + GT P/S picks + geometry.

    Supervision at train time uses travel-time and waveform misfit (no true Earth).
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 600,
        max_traces: Optional[int] = None,
        min_distance_km: float = 1.0,
        max_distance_km: float = 200.0,
        min_depth_km: float = 1.0,
        seed: int = 42,
        augment: bool = False,
    ):
        if split not in {"train", "val"}:
            raise ValueError("STEAD zhizi inversion uses train/val only (test reserved for eval)")
        self.seq_len = seq_len
        self.base = STEADPickingDataset(
            split="train" if split == "train" else "val",
            seq_len=seq_len,
            max_event_traces=max_traces,
            max_noise_traces=0,
            label_sigma_sec=0.4,
            seed=seed,
            augment=augment and split == "train",
        )
        self.depths = default_synth_model("cpu").depths
        self.q = default_synth_model("cpu").q
        self.indices: list[int] = []
        for i in range(len(self.base)):
            item = self.base[i]
            if _scalar_tensor(item["det"]) <= 0.5:
                continue
            if _scalar_tensor(item["p_valid"]) <= 0 or _scalar_tensor(item["s_valid"]) <= 0:
                continue
            dist = _scalar_tensor(item["source_distance_km"])
            depth = _scalar_tensor(item["source_depth_km"])
            if not math.isfinite(dist) or not math.isfinite(depth):
                continue
            if dist < min_distance_km or dist > max_distance_km:
                continue
            if depth < min_depth_km:
                continue
            self.indices.append(i)

        if not self.indices:
            raise RuntimeError(f"No valid STEAD traces for split={split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.base[self.indices[idx]]
        dist = _scalar_tensor(item["source_distance_km"])
        depth = max(_scalar_tensor(item["source_depth_km"]), 1.0)
        p_idx = int(_scalar_tensor(item["p_idx"]))
        s_idx = int(_scalar_tensor(item["s_idx"]))
        obs_tp = idx_to_sec(p_idx, self.seq_len)
        obs_ts = idx_to_sec(s_idx, self.seq_len)
        geo = encode_geometry_tensor(dist, depth)

        return {
            "x": item["x"],  # (T, 3)
            "t": item["t"],  # (T, 1)
            "obs_tp": torch.tensor(obs_tp, dtype=torch.float32),
            "obs_ts": torch.tensor(obs_ts, dtype=torch.float32),
            "depths": self.depths.clone(),
            "true_q": self.q.clone(),
            "source_depth": torch.tensor(depth, dtype=torch.float32),
            "distances": torch.tensor([dist], dtype=torch.float32),
            "geo": geo,
            "trace_name": item["trace_name"],
            "p_idx": item["p_idx"],
            "s_idx": item["s_idx"],
        }
