# -*- coding: utf-8 -*-
"""Constitutive synthetic generators for Domain-III Stage-1.

Families (start set from DOMAIN_III doc):
  - Newtonian channel: θ = (η,)
  - Carreau / power-law channel: θ = (η₀, η∞, n, λ)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hnf.fluid_synth import _grid, sparsify

CONST_FAMILY_TO_ID = {"newtonian": 0, "carreau": 1}
CONST_ID_TO_FAMILY = {v: k for k, v in CONST_FAMILY_TO_ID.items()}
# Packed parameter vector: [eta0, eta_inf, n, lambda]
N_THETA = 4


def _newtonian_channel(h: int, w: int, eta: float, dp: float) -> np.ndarray:
    _, yy = _grid(h, w)
    vx = (dp / (2.0 * max(eta, 1e-4))) * (1.0 - yy * yy)
    vy = np.zeros_like(vx)
    return np.stack([vx, vy], axis=0).astype(np.float32)


def _carreau_channel(
    h: int,
    w: int,
    eta0: float,
    eta_inf: float,
    n: float,
    lam: float,
    dp: float,
) -> np.ndarray:
    """Approximate Carreau channel via local apparent viscosity + power-law shape.

    Shape exponent m=(n+1)/n; amplitude set by η_app(γ̇_char) so shear-thinning
    samples are distinguishable from Newtonian at the same η₀.
    """
    _, yy = _grid(h, w)
    m = (n + 1.0) / max(n, 1e-3)
    # Characteristic shear ~ wall gradient of unit channel
    gdot = abs(dp) / max(eta0, 1e-4)
    eta_app = eta_inf + (eta0 - eta_inf) * (1.0 + (lam * gdot) ** 2) ** ((n - 1.0) / 2.0)
    eta_app = float(np.clip(eta_app, 1e-4, 1e3))
    vmax = dp / (2.0 * eta_app)
    vx = vmax * (1.0 - np.abs(yy) ** m)
    vy = np.zeros_like(vx)
    return np.stack([vx, vy], axis=0).astype(np.float32)


def make_constitutive_sample(
    *,
    h: int = 32,
    w: int = 32,
    keep_frac: float = 0.1,
    family: Optional[str] = None,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    fam = family or str(rng.choice(list(CONST_FAMILY_TO_ID.keys())))
    dp = float(rng.uniform(1.0, 3.0))
    theta = np.zeros(N_THETA, dtype=np.float32)
    theta_mask = np.zeros(N_THETA, dtype=np.float32)

    if fam == "newtonian":
        eta = float(rng.uniform(0.3, 2.0))
        dense = _newtonian_channel(h, w, eta, dp)
        theta[0] = eta
        theta_mask[0] = 1.0
    elif fam == "carreau":
        eta0 = float(rng.uniform(0.8, 2.5))
        eta_inf = float(rng.uniform(0.05, 0.4))
        n = float(rng.uniform(0.35, 0.85))  # shear-thinning
        lam = float(rng.uniform(0.5, 3.0))
        dense = _carreau_channel(h, w, eta0, eta_inf, n, lam, dp)
        theta[:] = (eta0, eta_inf, n, lam)
        theta_mask[:] = 1.0
    else:
        raise ValueError(fam)

    noise = 0.02 * float(np.std(dense) + 1e-6) * rng.standard_normal(dense.shape).astype(np.float32)
    sparse, mask = sparsify(dense + noise, keep_frac, rng)
    return {
        "sparse": sparse,
        "mask": mask,
        "dense": dense.astype(np.float32),
        "theta": theta,
        "theta_mask": theta_mask,
        "family": fam,
        "family_id": int(CONST_FAMILY_TO_ID[fam]),
        "seed": int(seed),
        "dp": dp,
    }
