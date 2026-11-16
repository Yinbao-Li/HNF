# -*- coding: utf-8 -*-
"""Robustness augmentations on raw 3-component traces (C, T) at 100 Hz.

Arrival times are preserved by every operation (zero-phase filters, sample-aligned superposition).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 100.0
P_WIN = 200  # samples after P used as the signal reference for SNR


def add_second_event(w: np.ndarray, p1: Optional[int], s1: Optional[int], w2: np.ndarray, p2: int,
                     s2: Optional[int], rng: np.random.Generator, amp_range=(0.3, 1.0), gap_sec=3.0,
                     lead=50, taper=50):
    """Superimpose event w2 after the primary S (or P) so its P lands in the free tail of the window.

    Returns (w, q_p, q_s) with the second event's P/S sample indices, or None if there is no room.
    """
    t = w.shape[-1]
    end1 = s1 if s1 is not None else p1
    if end1 is None:
        return None
    sp2 = (s2 - p2) if s2 is not None else 0
    lo = int(end1 + gap_sec * FS)
    hi = t - sp2 - int(1.0 * FS)
    if hi <= lo or p2 - lead < 0:
        return None
    q = int(rng.integers(lo, hi))
    seg = w2[:, p2 - lead:p2 - lead + (t - (q - lead))].astype(np.float32)
    seg = seg - seg[:, :lead].mean(-1, keepdims=True)
    ramp = np.ones(seg.shape[-1], dtype=np.float32)
    k = min(taper, seg.shape[-1])
    ramp[:k] = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k, dtype=np.float32))
    ref = np.abs(w[:, p1:p1 + P_WIN]).max() if p1 is not None else np.abs(w).max()
    peak2 = np.abs(seg).max() + 1e-8
    seg = seg * ramp * (float(rng.uniform(*amp_range)) * ref / peak2)
    out = w.copy()
    out[:, q - lead:q - lead + seg.shape[-1]] += seg
    q_s = q + sp2 if s2 is not None else None
    return out, q, q_s


def add_real_noise(w: np.ndarray, p: Optional[int], noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add a real noise trace scaled so that power(P window) / power(noise) = snr_db."""
    t = w.shape[-1]
    nz = noise[:, :t].astype(np.float32)
    nz = nz - nz.mean(-1, keepdims=True)
    if p is not None:
        sig = w[:, p:p + P_WIN]
        sig_pow = float(((sig - sig.mean(-1, keepdims=True)) ** 2).mean())
    else:
        sig_pow = float(((w - w.mean(-1, keepdims=True)) ** 2).mean())
    n_pow = float((nz ** 2).mean()) + 1e-12
    return w + nz * np.sqrt(sig_pow / (10 ** (snr_db / 10.0)) / n_pow)


def response_variation(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Instrument / site / path variation: velocity→acceleration (differentiate), or a random zero-phase
    band with a log-uniform low-pass corner down to 5 Hz (attenuated, distant-event spectra)."""
    if rng.random() < 0.4:
        return np.gradient(w, axis=-1).astype(np.float32) * FS
    lo = float(rng.uniform(0.5, 3.0))
    hi = float(np.exp(rng.uniform(np.log(5.0), np.log(45.0))))
    order = int(rng.integers(2, 5))
    sos = butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, w, axis=-1).astype(np.float32)


_MICRO_SOS = butter(2, [0.1, 1.0], btype="bandpass", fs=FS, output="sos")


def add_lowfreq(w: np.ndarray, rng: np.random.Generator, ratio_range=(0.3, 10.0)) -> np.ndarray:
    """Unfiltered-recording disturbances below the STEAD band: microseism (0.1–1 Hz, always), plus linear
    drift and DC offset (each with probability 1/2), at log-uniform amplitudes relative to the trace std."""
    c, t = w.shape
    std = w.std(-1, keepdims=True) + 1e-8
    lo, hi = np.log(ratio_range[0]), np.log(ratio_range[1])
    m = sosfiltfilt(_MICRO_SOS, rng.standard_normal((c, t)), axis=-1)
    m = m / (m.std(-1, keepdims=True) + 1e-8)
    out = w + (m * np.exp(rng.uniform(lo, hi, (c, 1))) * std).astype(np.float32)
    if rng.random() < 0.5:
        ramp = np.linspace(-1.0, 1.0, t, dtype=np.float32)[None]
        out = out + ramp * (rng.uniform(-1, 1, (c, 1)) * np.exp(rng.uniform(lo, hi, (c, 1))) * std)
    if rng.random() < 0.5:
        out = out + rng.uniform(-1, 1, (c, 1)) * np.exp(rng.uniform(lo, hi, (c, 1))) * std
    return out.astype(np.float32)


def add_reverb(w: np.ndarray, rng: np.random.Generator, depth_m=(50.0, 1500.0), refl=(0.3, 0.8),
               v_water=1500.0, max_order=8) -> np.ndarray:
    """Water-column multiples under an OBS: echoes every 2h/v_w with alternating sign (-r)^k.
    The first arrival is untouched, so pick labels are preserved."""
    t = w.shape[-1]
    lag = int(round(2.0 * rng.uniform(*depth_m) / v_water * FS))
    r = float(rng.uniform(*refl))
    out = w.copy()
    for k in range(1, max_order + 1):
        a = (-r) ** k
        if abs(a) < 0.05 or k * lag >= t:
            break
        out[:, k * lag:] += a * w[:, : t - k * lag]
    return out.astype(np.float32)


def emergent_onset(w: np.ndarray, p: Optional[int], rng: np.random.Generator, tau_sec=(0.03, 0.3)) -> np.ndarray:
    """Turn an impulsive onset emergent (attenuation / scattering): convolve the trace from P on with a causal
    unit-DC-gain kernel (t/τ)^k·exp(−t/τ), k ∈ {1, 2}. h(0) = 0, so energy still starts at P (label kept) but the
    rise is spread over ~kτ; samples before P are untouched."""
    if p is None:
        return w
    tau = float(np.exp(rng.uniform(np.log(tau_sec[0]), np.log(tau_sec[1])))) * FS
    k = int(rng.integers(1, 3))
    t = np.arange(int(np.ceil(8 * tau)) + 1, dtype=np.float64)
    h = (t / tau) ** k * np.exp(-t / tau)
    h /= h.sum()
    out = w.copy()
    for c in range(w.shape[0]):
        out[c, p:] = np.convolve(w[c], h)[p:w.shape[-1]]
    return out.astype(np.float32)


def add_gap(x, picks: list[int], rng: np.random.Generator, len_range=(50, 500), guard=100):
    """Zero one segment of a (T, C) tensor across all channels, keeping ±guard samples around picks."""
    t = x.shape[0]
    n = int(rng.integers(*len_range))
    for _ in range(10):
        a = int(rng.integers(0, t - n))
        if all(not (a - guard <= q <= a + n + guard) for q in picks if q is not None and q >= 0):
            x = x.clone()
            x[a:a + n] = 0.0
            return x
    return x
