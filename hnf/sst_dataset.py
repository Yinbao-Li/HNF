# -*- coding: utf-8 -*-
"""NOAA SST dataset adapter for Huygens Neural Field."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "field" / "data"


def _resolve_sst_dir() -> Path:
    env = os.environ.get("HNF_DATA_ROOT") or os.environ.get("RECFNO_DATA_ROOT")
    if env:
        return Path(env).expanduser() / "sst"
    candidates = [
        _DEFAULT_ROOT / "sst",
        Path("/media/bob/Work/TRELLIS/field/data/sst"),
        Path(__file__).resolve().parents[2] / "data" / "sst",
    ]
    for p in candidates:
        if (p / "sst_weekly.mat").is_file():
            return p
    return candidates[0]


SST_DIR = _resolve_sst_dir()
SST_H5 = SST_DIR / "sst_weekly.mat"
SST_SPLITS = SST_DIR / "splits.json"

_META = json.loads(SST_SPLITS.read_text())
SST_TRAIN = list(range(*_META["train"]))
SST_VAL = list(range(*_META["val"]))
SST_TEST = list(range(*_META["test"]))
SST_SENSOR_POS = np.array(_META["sensor_positions"], dtype=np.int64)
SST_MEAN = float(_META.get("mean", 0.0))
SST_STD = float(_META.get("std", 40.0))
FIELD_KEY = _META.get("field_key", "sst")
OUT_H, OUT_W = tuple(_META["out_size"])


def _load_sst_cube() -> np.ndarray:
    with h5py.File(SST_H5, "r") as f:
        sst = np.asarray(f[FIELD_KEY][:], dtype=np.float32)
    sst = np.nan_to_num(sst, nan=0.0)
    cube = sst.reshape(sst.shape[0], 1, 360, 180).transpose(0, 1, 3, 2)
    return np.flip(cube, axis=2).copy()


def _land_mask() -> np.ndarray:
    with h5py.File(SST_H5, "r") as f:
        raw = np.asarray(f[FIELD_KEY][0], dtype=np.float32)
    mask = np.isnan(raw).reshape(360, 180).T
    return np.flip(mask, axis=0).copy()


def build_grid_coords(h: int = OUT_H, w: int = OUT_W, device=None) -> torch.Tensor:
    """Normalized grid coordinates in [-1, 1]^2 as [H*W, 2]."""
    ys = torch.linspace(-1.0, 1.0, h)
    xs = torch.linspace(-1.0, 1.0, w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    return coords.to(device) if device else coords


def build_sensor_coords(positions: np.ndarray | None = None, h: int = OUT_H, w: int = OUT_W, device=None) -> torch.Tensor:
    """Sensor coordinates [N, 2] from integer (row, col) indices."""
    pos = SST_SENSOR_POS if positions is None else positions
    rows = torch.as_tensor(pos[:, 0], dtype=torch.float32)
    cols = torch.as_tensor(pos[:, 1], dtype=torch.float32)
    x = 2.0 * cols / (w - 1) - 1.0
    y = 2.0 * rows / (h - 1) - 1.0
    coords = torch.stack([x, y], dim=-1)
    return coords.to(device) if device else coords


class SSTDataset(Dataset):
    """HNF SST samples: sparse sensor values + dense ocean field."""

    def __init__(self, indices, mean: float = SST_MEAN, std: float = SST_STD):
        cube = _load_sst_cube()[indices]
        self.mean, self.std = mean, std
        self.fields = torch.from_numpy(cube).float()
        self.fields = (self.fields - mean) / std
        self.land_mask = torch.from_numpy(_land_mask()).bool()
        self.ocean_mask = (~self.land_mask).float()

        n_samples = self.fields.shape[0]
        n_sensors = len(SST_SENSOR_POS)
        obs = torch.zeros(n_samples, n_sensors, 1)
        for i, (r, c) in enumerate(SST_SENSOR_POS):
            obs[:, i, 0] = self.fields[:, 0, r, c]
        self.obs_values = obs

        self.target_coords = build_grid_coords()
        self.obs_coords = build_sensor_coords()

    def __len__(self) -> int:
        return self.fields.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        field = self.fields[idx, 0]  # [H, W]
        return {
            "obs_coords": self.obs_coords,
            "obs_values": self.obs_values[idx],  # [N, 1]
            "target_coords": self.target_coords,
            "target_field": field.reshape(-1, 1),
            "ocean_mask": self.ocean_mask.reshape(-1, 1),
            "field_2d": field,
        }
