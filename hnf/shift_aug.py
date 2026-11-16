# -*- coding: utf-8 -*-
"""Shift a 3-component trace in time, filling the vacated span with real noise."""

from __future__ import annotations

from typing import Optional

import numpy as np

EDGE = 200
FADE = 50


def shift_with_noise(w: np.ndarray, d: int, noise: Optional[np.ndarray] = None) -> np.ndarray:
    """w: (C, T). d > 0 delays the content by d samples, d < 0 advances it.

    The vacated span is filled with ``noise`` (C, >=|d|) rescaled per channel to the std/mean of
    the EDGE samples it replaces (tiled edge if noise is None), cross-faded over FADE samples.
    """
    if d == 0:
        return w
    t = w.shape[-1]
    n = min(abs(d), t)
    edge = w[:, :EDGE] if d > 0 else w[:, -EDGE:]
    if noise is None:
        pad = np.tile(edge, (1, n // EDGE + 1))[:, :n]
    else:
        seg = noise[:, :n].astype(np.float32) - noise[:, :n].mean(-1, keepdims=True)
        pad = seg * (edge.std(-1, keepdims=True) / (seg.std(-1, keepdims=True) + 1e-8)) + edge.mean(-1, keepdims=True)
    f = min(FADE, n, t - n)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f, dtype=np.float32)) if f > 0 else None
    if d > 0:
        body = w[:, : t - n].copy()
        if f > 0:
            body[:, :f] = pad[:, -f:] * (1 - ramp) + body[:, :f] * ramp
        return np.concatenate([pad, body], axis=1).astype(np.float32)
    body = w[:, n:].copy()
    if f > 0:
        body[:, -f:] = body[:, -f:] * (1 - ramp) + pad[:, :f] * ramp
    return np.concatenate([body, pad], axis=1).astype(np.float32)
