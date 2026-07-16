# -*- coding: utf-8 -*-
"""Fluid datasets for Domain III Stage-0 (synthetic sparse→dense)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hnf.fluid_synth import make_sample

FAMILY_TO_ID = {"poiseuille": 0, "couette": 1, "vortex": 2}
ID_TO_FAMILY = {v: k for k, v in FAMILY_TO_ID.items()}


class SyntheticFluidDataset(Dataset):
    """On-the-fly 2D sparse velocity fields with dense GT (+ η label)."""

    def __init__(
        self,
        split: str = "train",
        n_samples: int = 2048,
        h: int = 32,
        w: int = 32,
        keep_frac: float = 0.1,
        seed: int = 42,
        families: Optional[list[str]] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.split = split
        self.n_samples = int(n_samples)
        self.h = int(h)
        self.w = int(w)
        self.keep_frac = float(keep_frac)
        self.families = families or list(FAMILY_TO_ID.keys())
        # Disjoint seed blocks per split
        base = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
        self._seeds = [base + seed * 10_000 + i for i in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float | str | int]:
        seed = self._seeds[idx]
        rng = np.random.default_rng(seed)
        family = str(rng.choice(self.families))
        sample = make_sample(
            h=self.h,
            w=self.w,
            keep_frac=self.keep_frac,
            family=family,
            seed=seed,
        )
        sparse = torch.from_numpy(sample["sparse"])  # (2,H,W)
        mask = torch.from_numpy(sample["mask"])  # (1,H,W)
        dense = torch.from_numpy(sample["dense"])
        x = torch.cat([sparse, mask], dim=0)  # (3,H,W)
        return {
            "x": x,
            "dense": dense,
            "mask": mask,
            "eta": float(sample["eta"]),
            "family": str(sample["family"]),
            "family_id": int(FAMILY_TO_ID[str(sample["family"])]),
            "seed": int(sample["seed"]),
        }
