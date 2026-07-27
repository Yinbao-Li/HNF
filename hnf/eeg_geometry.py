# -*- coding: utf-8 -*-
"""10–20 electrode geometry for EEG-native Huygens spatial kernels."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch

from hnf.eeg_dataset import STANDARD_10_20

# Approximate unit-sphere (θ, φ) for standard 10–20 names (degrees).
# Rough clinical montage; exact MRI coregistration is not required for the prior.
_SPH_DEG: dict[str, tuple[float, float]] = {
    "Fp1": (92.0, -72.0),
    "Fp2": (92.0, 72.0),
    "F7": (55.0, -54.0),
    "F3": (60.0, -45.0),
    "Fz": (60.0, 0.0),
    "F4": (60.0, 45.0),
    "F8": (55.0, 54.0),
    "T3": (0.0, -90.0),
    "C3": (0.0, -45.0),
    "Cz": (0.0, 0.0),
    "C4": (0.0, 45.0),
    "T4": (0.0, 90.0),
    "T5": (-55.0, -54.0),
    "P3": (-60.0, -45.0),
    "Pz": (-60.0, 0.0),
    "P4": (-60.0, 45.0),
    "T6": (-55.0, 54.0),
    "O1": (-92.0, -72.0),
    "O2": (-92.0, 72.0),
}

# Clinically motivated regions for AD vs FTD (FTD: frontotemporal; AD: more posterior/global).
REGION_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "central": ("C3", "Cz", "C4"),
    "posterior": ("P3", "Pz", "P4", "O1", "O2"),
}


def electrode_xyz(names: tuple[str, ...] = STANDARD_10_20) -> np.ndarray:
    """Return ``(C, 3)`` unit-sphere Cartesian positions."""
    pts = []
    for name in names:
        if name not in _SPH_DEG:
            raise KeyError(f"No geometry for electrode {name}")
        th, ph = _SPH_DEG[name]
        th_r = math.radians(th)
        ph_r = math.radians(ph)
        # θ=0 at Cz equator in this convention → x=cosθ cosφ, y=cosθ sinφ, z=sinθ
        x = math.cos(th_r) * math.cos(ph_r)
        y = math.cos(th_r) * math.sin(ph_r)
        z = math.sin(th_r)
        pts.append((x, y, z))
    xyz = np.asarray(pts, dtype=np.float64)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-12
    return xyz.astype(np.float32)


def pairwise_chord_distance(xyz: np.ndarray) -> np.ndarray:
    """Chordal distances on the unit sphere, shape ``(C, C)``."""
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)


def electrode_distance_tensor(
    names: tuple[str, ...] = STANDARD_10_20,
) -> torch.Tensor:
    xyz = electrode_xyz(names)
    return torch.from_numpy(pairwise_chord_distance(xyz))


def region_index_masks(
    names: tuple[str, ...] = STANDARD_10_20,
) -> Dict[str, torch.Tensor]:
    """Boolean masks ``(C,)`` for each clinical region."""
    name_to_i = {n: i for i, n in enumerate(names)}
    out: Dict[str, torch.Tensor] = {}
    for region, chans in REGION_CHANNELS.items():
        mask = torch.zeros(len(names), dtype=torch.bool)
        for ch in chans:
            if ch not in name_to_i:
                raise KeyError(f"Region {region} channel {ch} missing from montage")
            mask[name_to_i[ch]] = True
        out[region] = mask
    return out
