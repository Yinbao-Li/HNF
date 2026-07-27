# -*- coding: utf-8
"""3D synthetic fluid dataset."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset

from hnf.fluid_synth3d import make_sample3d

FAMILY3D_TO_ID = {"pipe": 0, "shear3d": 1, "vortex_tube": 2}


class SyntheticFluid3DDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        n_samples: int = 1024,
        d: int = 12,
        h: int = 12,
        w: int = 12,
        keep_frac: float = 0.1,
        seed: int = 42,
        families: Optional[list[str]] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.split = split
        self.n_samples = int(n_samples)
        self.d, self.h, self.w = int(d), int(h), int(w)
        self.keep_frac = float(keep_frac)
        self.families = families or list(FAMILY3D_TO_ID.keys())
        base = {"train": 0, "val": 5_000_000, "test": 6_000_000}[split]
        self._seeds = [base + seed * 10_000 + i for i in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seed = self._seeds[idx]
        import numpy as np
        rng = np.random.default_rng(seed)
        family = str(rng.choice(self.families))
        s = make_sample3d(
            d=self.d, h=self.h, w=self.w,
            keep_frac=self.keep_frac, family=family, seed=seed,
        )
        sparse = torch.from_numpy(s["sparse"])
        mask = torch.from_numpy(s["mask"])
        x = torch.cat([sparse, mask], dim=0)
        return {
            "x": x,
            "dense": torch.from_numpy(s["dense"]),
            "mask": mask,
            "eta": float(s["eta"]),
            "family": str(s["family"]),
            "seed": int(s["seed"]),
        }
