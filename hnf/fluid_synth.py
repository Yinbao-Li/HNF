# -*- coding: utf-8 -*-
"""Synthetic 2D flow fields for Domain-III Stage-0 / Stage-1 scaffolds.

RACLETTE .pv volumes need pyvista; until that loader lands, controlled 2D
Poiseuille / Couette / vortex fields provide sparse→dense + known-η labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FluidSampleMeta:
    family: str
    eta: float
    seed: int


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def poiseuille(h: int, w: int, eta: float, dp: float = 2.0) -> np.ndarray:
    """Channel flow: vx(y) ∝ (1-y^2)/η, vy=0. Returns (2, H, W)."""
    _, yy = _grid(h, w)
    vx = (dp / (2.0 * max(eta, 1e-4))) * (1.0 - yy * yy)
    vy = np.zeros_like(vx)
    return np.stack([vx, vy], axis=0).astype(np.float32)


def couette(h: int, w: int, eta: float, u_wall: float = 1.0) -> np.ndarray:
    """Linear shear; η unused in kinematics but stored as label."""
    del eta
    _, yy = _grid(h, w)
    vx = 0.5 * u_wall * (yy + 1.0)
    vy = np.zeros_like(vx)
    return np.stack([vx, vy], axis=0).astype(np.float32)


def vortex(h: int, w: int, eta: float, strength: float = 1.0) -> np.ndarray:
    """Soft Gaussian vortex; amplitude mildly damped by η."""
    xx, yy = _grid(h, w)
    r2 = xx * xx + yy * yy
    amp = strength / (1.0 + float(eta))
    vx = -amp * yy * np.exp(-3.0 * r2)
    vy = amp * xx * np.exp(-3.0 * r2)
    return np.stack([vx, vy], axis=0).astype(np.float32)


_GENERATORS = {
    "poiseuille": poiseuille,
    "couette": couette,
    "vortex": vortex,
}


def generate_dense(
    family: str,
    h: int,
    w: int,
    eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    gen = _GENERATORS[family]
    if family == "poiseuille":
        dp = float(rng.uniform(1.0, 3.0))
        return gen(h, w, eta, dp=dp)
    if family == "couette":
        u_wall = float(rng.uniform(0.5, 1.5))
        return gen(h, w, eta, u_wall=u_wall)
    strength = float(rng.uniform(0.5, 1.5))
    return gen(h, w, eta, strength=strength)


def sparsify(
    dense: np.ndarray,
    keep_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sparse_v, mask) with mask in {0,1}, shape (1,H,W)."""
    _, h, w = dense.shape
    n = h * w
    k = max(1, int(round(keep_frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    mask = np.zeros((1, h, w), dtype=np.float32)
    mask.reshape(-1)[idx] = 1.0
    sparse = dense * mask
    return sparse.astype(np.float32), mask


def make_sample(
    *,
    h: int = 32,
    w: int = 32,
    keep_frac: float = 0.1,
    family: Optional[str] = None,
    eta: Optional[float] = None,
    seed: int = 0,
) -> dict[str, np.ndarray | float | str | int]:
    rng = np.random.default_rng(seed)
    fam = family or str(rng.choice(list(_GENERATORS.keys())))
    eta_v = float(eta) if eta is not None else float(rng.uniform(0.2, 2.0))
    dense = generate_dense(fam, h, w, eta_v, rng)
    # Mild observation noise on kept pixels
    noise = 0.02 * float(np.std(dense) + 1e-6) * rng.standard_normal(dense.shape).astype(np.float32)
    sparse, mask = sparsify(dense + noise, keep_frac, rng)
    return {
        "sparse": sparse,
        "mask": mask,
        "dense": dense.astype(np.float32),
        "eta": eta_v,
        "family": fam,
        "seed": int(seed),
    }
