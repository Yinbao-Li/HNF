# -*- coding: utf-8
"""Synthetic 4D (3D + time) velocity fields — 4D Flow MRI analogue."""

from __future__ import annotations

from typing import Optional

import numpy as np

from hnf.fluid_sparsify4d import sparsify_spacetime
from hnf.fluid_synth3d import _grid3, pipe_flow


def pulsatile_pipe(
    d: int,
    h: int,
    w: int,
    t_steps: int,
    eta: float,
    vz0: float = 1.0,
    freq: float = 1.0,
) -> np.ndarray:
    """Axial pipe flow with sinusoidal time modulation. Returns (3,T,D,H,W)."""
    base = pipe_flow(d, h, w, eta, vz0=vz0)
    ts = np.linspace(0.0, 2.0 * np.pi, t_steps, dtype=np.float32)
    mod = (1.0 + 0.35 * np.sin(freq * ts)).astype(np.float32)
    out = np.stack([base * m for m in mod], axis=1)  # (3,T,D,H,W)
    return out.astype(np.float32)


def advecting_vortex(
    d: int,
    h: int,
    w: int,
    t_steps: int,
    eta: float,
    strength: float = 1.0,
) -> np.ndarray:
    """ω_z blob advected along +z over time."""
    zz, yy, xx = _grid3(d, h, w)
    frames = []
    for ti in range(t_steps):
        phase = (ti / max(t_steps - 1, 1)) * 0.8
        z_shift = zz - phase
        r2_xy = xx * xx + yy * yy
        amp = strength / (1.0 + float(eta))
        env = np.exp(-3.0 * r2_xy) * np.exp(-2.0 * z_shift * z_shift)
        vx = -amp * yy * env
        vy = amp * xx * env
        frames.append(np.stack([vx, vy, np.zeros_like(vx)], axis=0))
    return np.stack(frames, axis=1).astype(np.float32)


_GENERATORS_4D = {
    "pulsatile_pipe": pulsatile_pipe,
    "advecting_vortex": advecting_vortex,
}


def make_sample4d(
    *,
    t_steps: int = 4,
    d: int = 8,
    h: int = 12,
    w: int = 12,
    keep_frac: float = 0.1,
    family: Optional[str] = None,
    eta: Optional[float] = None,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    fam = family or str(rng.choice(list(_GENERATORS_4D.keys())))
    eta_v = float(eta) if eta is not None else float(rng.uniform(0.2, 2.0))
    if fam == "pulsatile_pipe":
        dense = pulsatile_pipe(
            d, h, w, t_steps, eta_v,
            vz0=float(rng.uniform(0.6, 1.2)),
            freq=float(rng.uniform(0.8, 1.5)),
        )
    else:
        dense = advecting_vortex(
            d, h, w, t_steps, eta_v,
            strength=float(rng.uniform(0.5, 1.5)),
        )
    noise = 0.02 * float(np.std(dense) + 1e-6) * rng.standard_normal(dense.shape).astype(np.float32)
    sparse, mask = sparsify_spacetime(dense + noise, keep_frac, rng)
    return {
        "sparse": sparse,
        "mask": mask,
        "dense": dense.astype(np.float32),
        "eta": eta_v,
        "family": fam,
        "seed": int(seed),
        "t_steps": int(t_steps),
    }
