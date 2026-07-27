# -*- coding: utf-8 -*-
"""Synthetic 3D velocity fields for Domain-III (sparse voxel → dense volume)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from hnf.fluid_synth import sparsify


def sparsify_volume(
    dense: np.ndarray,
    keep_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``dense`` (C,D,H,W) → sparse + mask (1,D,H,W)."""
    c, d, h, w = dense.shape
    n = d * h * w
    k = max(1, int(round(keep_frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    mask = np.zeros((1, d, h, w), dtype=np.float32)
    mask.reshape(-1)[idx] = 1.0
    sparse = dense * mask
    return sparse.astype(np.float32), mask


def _grid3(d: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zs = np.linspace(-1.0, 1.0, d, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return zz, yy, xx


def pipe_flow(d: int, h: int, w: int, eta: float, vz0: float = 1.0) -> np.ndarray:
    """Parabolic axial flow in a cylindrical pipe (v_z dominant). Returns (3,D,H,W)."""
    _, yy, xx = _grid3(d, h, w)
    r2 = xx * xx + yy * yy
    profile = np.clip(1.0 - r2, 0.0, None) / max(eta, 1e-3)
    vz = vz0 * profile
    return np.stack([np.zeros_like(vz), np.zeros_like(vz), vz], axis=0).astype(np.float32)


def shear3d(d: int, h: int, w: int, eta: float, u0: float = 1.0) -> np.ndarray:
    """Linear vx(y) shear, weak vz drift."""
    del eta
    zz, yy, _ = _grid3(d, h, w)
    vx = 0.5 * u0 * (yy + 1.0)
    vz = 0.05 * u0 * zz
    return np.stack([vx, np.zeros_like(vx), vz], axis=0).astype(np.float32)


def vortex_tube(d: int, h: int, w: int, eta: float, strength: float = 1.0) -> np.ndarray:
    """Localized ω_z tube: azimuthal flow in xy, Gaussian envelope in z."""
    zz, yy, xx = _grid3(d, h, w)
    r2_xy = xx * xx + yy * yy
    amp = strength / (1.0 + float(eta))
    env = np.exp(-3.0 * r2_xy) * np.exp(-2.0 * zz * zz)
    vx = -amp * yy * env
    vy = amp * xx * env
    return np.stack([vx, vy, np.zeros_like(vx)], axis=0).astype(np.float32)


_GENERATORS_3D = {
    "pipe": pipe_flow,
    "shear3d": shear3d,
    "vortex_tube": vortex_tube,
}


def generate_dense3d(
    family: str,
    d: int,
    h: int,
    w: int,
    eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    gen = _GENERATORS_3D[family]
    if family == "pipe":
        return gen(d, h, w, eta, vz0=float(rng.uniform(0.6, 1.4)))
    if family == "shear3d":
        return gen(d, h, w, eta, u0=float(rng.uniform(0.5, 1.2)))
    return gen(d, h, w, eta, strength=float(rng.uniform(0.5, 1.5)))


def make_sample3d(
    *,
    d: int = 12,
    h: int = 12,
    w: int = 12,
    keep_frac: float = 0.1,
    family: Optional[str] = None,
    eta: Optional[float] = None,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    fam = family or str(rng.choice(list(_GENERATORS_3D.keys())))
    eta_v = float(eta) if eta is not None else float(rng.uniform(0.2, 2.0))
    dense = generate_dense3d(fam, d, h, w, eta_v, rng)
    noise = 0.02 * float(np.std(dense) + 1e-6) * rng.standard_normal(dense.shape).astype(np.float32)
    sparse, mask = sparsify_volume(dense + noise, keep_frac, rng)
    return {
        "sparse": sparse,
        "mask": mask,
        "dense": dense.astype(np.float32),
        "eta": eta_v,
        "family": fam,
        "seed": int(seed),
    }
