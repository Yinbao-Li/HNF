# -*- coding: utf-8
"""RACLETTE 3D patch dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hnf.fluid_synth3d import sparsify_volume


class RacletteVolumeDataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path = "external_data/raclette_cache/gt_volumes.npz",
        split: str = "train",
        keep_frac: float = 0.1,
        seed: int = 42,
        augment: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.split = split
        self.keep_frac = float(keep_frac)
        self.seed = int(seed)
        self.augment = bool(augment) and split == "train"
        blob = np.load(Path(cache_path), allow_pickle=False)
        self.velocity = np.asarray(blob["velocity"], dtype=np.float32)  # (N,3,D,H,W)
        self.vessel_mask = np.asarray(blob["vessel_mask"], dtype=np.float32)
        n = self.velocity.shape[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train, n_val = int(0.7 * n), int(0.15 * n)
        if split == "train":
            self.indices = perm[:n_train].tolist()
        elif split == "val":
            self.indices = perm[n_train : n_train + n_val].tolist()
        else:
            self.indices = perm[n_train + n_val :].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        dense = self.velocity[i].copy()
        vmask = self.vessel_mask[i].copy()
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        if self.augment and rng.random() < 0.5:
            dense = dense[..., ::-1].copy()
            vmask = vmask[..., ::-1].copy()
        sparse, obs_mask = sparsify_volume(dense, self.keep_frac, rng)
        obs_mask = obs_mask * vmask[None]
        sparse = dense * obs_mask
        x = torch.from_numpy(np.concatenate([sparse, obs_mask], axis=0).astype(np.float32))
        return {
            "x": x,
            "dense": torch.from_numpy(dense.astype(np.float32)),
            "mask": torch.from_numpy(obs_mask.astype(np.float32)),
            "vessel_mask": torch.from_numpy(vmask.astype(np.float32)),
            "family": "raclette3d",
            "eta": 1.0,
            "index": int(i),
        }
