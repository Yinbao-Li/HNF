# -*- coding: utf-8 -*-
"""RACLETTE cached-slice dataset for Domain-III Stage-0b."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hnf.fluid_synth import sparsify


class RacletteSliceDataset(Dataset):
    """Sparse→dense on preprocessed RACLETTE GT in-plane velocity slices."""

    def __init__(
        self,
        cache_path: str | Path = "external_data/raclette_cache/gt_slices.npz",
        split: str = "train",
        keep_frac: float = 0.1,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.split = split
        self.keep_frac = float(keep_frac)
        self.seed = int(seed)
        self.augment = bool(augment) and split == "train"

        blob = np.load(Path(cache_path), allow_pickle=False)
        self.velocity = np.asarray(blob["velocity"], dtype=np.float32)  # (N,2,H,W)
        self.vessel_mask = np.asarray(blob["vessel_mask"], dtype=np.float32)
        self.meta = json.loads(str(blob["meta_json"]))
        n = self.velocity.shape[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        if split == "train":
            self.indices = perm[:n_train].tolist()
        elif split == "val":
            self.indices = perm[n_train : n_train + n_val].tolist()
        else:
            self.indices = perm[n_train + n_val :].tolist()
        if not self.indices:
            raise RuntimeError(f"Empty {split} split (n={n})")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        i = self.indices[idx]
        dense = self.velocity[i].copy()
        vmask = self.vessel_mask[i].copy()
        rng = np.random.default_rng(self.seed * 100_003 + idx * 17 + (0 if self.split != "train" else 1))
        if self.augment:
            if rng.random() < 0.5:
                dense = dense[:, :, ::-1].copy()
                vmask = vmask[:, ::-1].copy()
            if rng.random() < 0.5:
                dense = dense[:, ::-1, :].copy()
                vmask = vmask[::-1, :].copy()
            # Sign flip of velocity (time-reversal style)
            if rng.random() < 0.5:
                dense = -dense
        sparse, obs_mask = sparsify(dense, self.keep_frac, rng)
        # Only observe inside vessel when possible
        obs_mask = obs_mask * vmask[None]
        if float(obs_mask.sum()) < 1:
            obs_mask = sparsify(dense, max(self.keep_frac, 0.05), rng)[1]
        sparse = dense * obs_mask
        x = torch.from_numpy(np.concatenate([sparse, obs_mask], axis=0).astype(np.float32))
        return {
            "x": x,
            "dense": torch.from_numpy(dense.astype(np.float32)),
            "mask": torch.from_numpy(obs_mask.astype(np.float32)),
            "vessel_mask": torch.from_numpy(vmask.astype(np.float32)),
            "index": int(i),
        }
