# -*- coding: utf-8 -*-
"""Elliott et al. 2025-compatible Maxwell fit + WLF shift for PS melts.

Ports the non-interactive parts of ``PS_MWD_Prediction_UPDATED.py``
(DOI 10.5518/1689) so we can feed the published Keras models and compare
interpretable spectrum→MWD transfers on the same features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.integrate import simpson

# a-PS @ 180°C constants (Elliott script)
M_E = 12870.0
M_E_PE = 820.0
G0_180 = 220000.0
TAU_E_180 = 2.20e-4

# Maxwell τ library
LONG_TAU = 4
SMALL_TAU = -6
NUM_OF_TAU = 1 + (LONG_TAU - SMALL_TAU) * 8  # 81
TAU_VALUES = np.logspace(LONG_TAU, SMALL_TAU, NUM_OF_TAU, base=10)

# MWD lognormal mixture (Elliott)
NUM_MWD_PARAMS = 28
MWD_MEANS = np.linspace(np.log(0.1 * M_E_PE), np.log(10000 * M_E_PE), NUM_MWD_PARAMS)
_KNOWN_MEANS = np.linspace(np.log(10 * M_E_PE), np.log(1000 * M_E_PE), 7)
_MEANS_RATIO = np.exp(MWD_MEANS[1]) / np.exp(MWD_MEANS[0])
_KNOWN_RATIO = np.exp(_KNOWN_MEANS[1]) / np.exp(_KNOWN_MEANS[0])
MWD_SIGMA = 0.55 * (_MEANS_RATIO / _KNOWN_RATIO)
MWD_X = np.logspace(2, 7, 200)


@dataclass
class ElliottMaxwellFit:
    sample_id: str
    temperature_c: float
    omega_shifted: np.ndarray
    gp_shifted: np.ndarray
    gpp_shifted: np.ndarray
    tau: np.ndarray
    log_g: np.ndarray  # natural log of modal amplitudes
    g: np.ndarray
    nn_input: np.ndarray  # shape (1, 2, 2, K) for Keras


def wlf_shift_to_180(
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    temperature_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WLF + vertical shift used in Elliott script (T_r=180°C)."""
    T = float(temperature_c)
    B1, B2 = 651.9, -52.24
    alpha = 10 ** (-3.161)
    T_r = 180.0
    log10alpha_T = (-B1 * (T - T_r)) / ((B2 + T_r) * (B2 + T))
    alpha_T = 10 ** log10alpha_T
    b_T = ((1 + alpha * T) * (T_r + 273.15)) / ((1 + alpha * T_r) * (T + 273.15))
    order = np.argsort(omega)
    return omega[order] * alpha_T, gp[order] / b_T, gpp[order] / b_T


def _g_prime(w: float, g: np.ndarray, tau: np.ndarray) -> float:
    x2 = (w * tau) ** 2
    return float(np.sum(g * x2 / (1.0 + x2)))


def _g_dprime(w: float, g: np.ndarray, tau: np.ndarray) -> float:
    x = w * tau
    return float(np.sum(g * x / (1.0 + x * x)))


def fit_maxwell_elliott(
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    *,
    tau: np.ndarray = TAU_VALUES,
    lam_smooth: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Fit log G_k with curvature penalty (Elliott ``curve_fit`` residual)."""
    omega = np.asarray(omega, dtype=np.float64)
    gp = np.asarray(gp, dtype=np.float64)
    gpp = np.asarray(gpp, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    k = len(tau)
    n = len(omega)
    target_ln = np.log(np.concatenate([gp, gpp]))
    # smoothness weight
    p_w = n / (np.log10(omega.max()) - np.log10(omega.min()) + 1e-12)
    p_w_base = 95.0 / 14.0
    lam = 1.0 * (p_w / p_w_base) ** 0.5 if lam_smooth is None else float(lam_smooth)

    # init
    if k < n:
        splits = np.array_split(np.log(gp), k)
        splits2 = np.array_split(np.log(gpp), k)
        p0 = 0.5 * (np.array([a.mean() for a in splits]) + np.array([a.mean() for a in splits2])) - 2.0
    else:
        idx = np.linspace(0, n - 1, k)
        p0 = 0.5 * (
            np.interp(idx, np.arange(n), np.log(gp))
            + np.interp(idx, np.arange(n), np.log(gpp))
        ) - 2.0

    y_zeros = np.zeros(2 * n + k - 2)

    def residual(log_g: np.ndarray) -> np.ndarray:
        g = np.exp(log_g)
        gp_h = np.array([_g_prime(w, g, tau) for w in omega])
        gpp_h = np.array([_g_dprime(w, g, tau) for w in omega])
        pred_ln = np.log(np.concatenate([gp_h, gpp_h]) + 1e-30)
        data_res = pred_ln - target_ln
        second = log_g[2:] + log_g[:-2] - 2.0 * log_g[1:-1]
        return np.concatenate([data_res, lam * second])

    sol = optimize.least_squares(
        residual, p0, bounds=(-40.0, 50.0), method="trf", max_nfev=400
    )
    return sol.x, float(lam)


def nn_input_from_log_g(log_g: np.ndarray) -> np.ndarray:
    """Keras models expect ``(None, 2, 81, 1)`` = stacked log10(G_k)."""
    row = (np.asarray(log_g, dtype=np.float64) / np.log(10.0)).astype(np.float32)
    x = np.stack([row, row], axis=0)  # (2, K)
    return x.reshape(1, 2, len(row), 1)


def fit_sample_elliott(
    sample_id: str,
    temperature_c: float,
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
) -> ElliottMaxwellFit:
    w, gp_s, gpp_s = wlf_shift_to_180(omega, gp, gpp, temperature_c)
    log_g, _ = fit_maxwell_elliott(w, gp_s, gpp_s)
    return ElliottMaxwellFit(
        sample_id=sample_id,
        temperature_c=temperature_c,
        omega_shifted=w,
        gp_shifted=gp_s,
        gpp_shifted=gpp_s,
        tau=TAU_VALUES.copy(),
        log_g=log_g,
        g=np.exp(log_g),
        nn_input=nn_input_from_log_g(log_g),
    )


def lognormal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -((np.log(x) - mean) ** 2) / (2 * sigma ** 2)
    )


def mwd_from_weights(weights: np.ndarray, x: np.ndarray = MWD_X) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    assert weights.size == NUM_MWD_PARAMS
    y = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(weights):
        y += max(w, 0.0) * lognormal_pdf(x, MWD_MEANS[i], MWD_SIGMA)
    area = simpson(y, x=np.log(x))
    if area > 0:
        y = y / area
    return y


def normalize_mwd_on_x(M: np.ndarray, w: np.ndarray, x: np.ndarray = MWD_X) -> np.ndarray:
    y = np.interp(x, M, np.maximum(w, 0.0), left=0.0, right=0.0)
    area = simpson(y, x=np.log(x))
    if area > 0:
        y = y / area
    return y


def mwd_moments_on_grid(
    x: np.ndarray,
    y: np.ndarray,
    *,
    m_min: float = 5e3,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = x >= float(m_min)
    x, y = x[mask], y[mask]
    logx = np.log(x)
    area = simpson(y, x=logx)
    y = y / max(area, 1e-30)
    Mw = float(simpson(x * y, x=logx))
    inv_Mn = float(simpson(y / x, x=logx))
    Mn = 1.0 / max(inv_Mn, 1e-30)
    return {
        "Mn": Mn,
        "Mw": Mw,
        "D": Mw / max(Mn, 1e-30),
        "log10_Mw": float(np.log10(Mw)),
        "log10_Mn": float(np.log10(Mn)),
    }
