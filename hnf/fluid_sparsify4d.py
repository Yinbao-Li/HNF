# -*- coding: utf-8
"""4D sparsify helper."""

from __future__ import annotations

import numpy as np


def sparsify_spacetime(
    dense: np.ndarray,
    keep_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``dense`` (C,T,D,H,W) with shared spatial mask over time → mask (1,T,D,H,W)."""
    c, t, d, h, w = dense.shape
    n = d * h * w
    k = max(1, int(round(keep_frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    spatial = np.zeros((1, d, h, w), dtype=np.float32)
    spatial.reshape(-1)[idx] = 1.0
    mask = np.broadcast_to(spatial[:, None], (1, t, d, h, w)).copy()
    sparse = dense * mask
    return sparse.astype(np.float32), mask
