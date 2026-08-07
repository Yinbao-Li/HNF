# -*- coding: utf-8 -*-
"""Diffusion-style continuous relaxation spectra for SAOS (G', G'').

Prony/PNF uses discrete Maxwell modes. A diffusion-style memory instead places
mass on a continuous H(λ) (log-λ density), e.g. a stretched exponential /
power-law-exp form motivated by Rouse–diffusion continuous spectra:

  H(λ) = G0 · (λ/λc)^β · exp(−(λ/λc)^γ)

and recovers moduli by quadrature of Maxwell kernels:

  G'(ω)  = G∞ + ∫ H(λ) (ωλ)^2/(1+(ωλ)^2) d ln λ
  G''(ω) =       ∫ H(λ) (ωλ)/(1+(ωλ)^2)   d ln λ

Also provides a single-mode fractional Maxwell (Scott-Blair branch):

  G*(ω) = G / (1 + (i ω τ)^α)

Neither replaces PNF; they are alternate memory shapes for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from hnf.rheo_freq_fit import default_lambda_grid, score_freq_fit


@dataclass
class DiffFitResult:
    method: str
    params: dict
    lambda_: np.ndarray  # discrete projection grid
    g: np.ndarray  # mode mass on grid (for tube_corr)
    g_inf: float
    rel_l2: float
    rel_log: float
    rel_l2_gp: float
    rel_l2_gpp: float
    success: bool
    message: str = ""


def _H_diff(lam: np.ndarray, G0: float, lam_c: float, beta: float, gamma: float) -> np.ndarray:
    x = np.maximum(lam, 1e-30) / max(lam_c, 1e-30)
    return np.maximum(G0, 0.0) * np.power(x, beta) * np.exp(-np.power(x, max(gamma, 1e-3)))


def _moduli_from_H(
    omega: np.ndarray,
    lam_q: np.ndarray,
    H: np.ndarray,
    g_inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid quadrature in ln λ."""
    om = np.asarray(omega, dtype=np.float64).reshape(-1)
    lam = np.asarray(lam_q, dtype=np.float64).reshape(-1)
    H = np.asarray(H, dtype=np.float64).reshape(-1)
    ln = np.log(np.maximum(lam, 1e-30))
    dln = np.gradient(ln)
    x = om[:, None] * lam[None, :]
    den = 1.0 + x * x
    w = H[None, :] * dln[None, :]
    gp = float(g_inf) + (w * (x * x) / den).sum(axis=1)
    gpp = (w * x / den).sum(axis=1)
    return gp, gpp


def project_H_to_modes(lam_lib: np.ndarray, lam_q: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Bin continuous H onto a Prony library for tube_corr."""
    lam_lib = np.asarray(lam_lib, dtype=np.float64)
    lam_q = np.asarray(lam_q, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    ln_q = np.log(np.maximum(lam_q, 1e-30))
    dln = np.gradient(ln_q)
    mass = np.maximum(H * dln, 0.0)
    g = np.zeros(len(lam_lib), dtype=np.float64)
    log_lib = np.log(np.maximum(lam_lib, 1e-30))
    edges = np.empty(len(lam_lib) + 1)
    edges[0] = log_lib[0] - 0.5 * (log_lib[1] - log_lib[0])
    edges[-1] = log_lib[-1] + 0.5 * (log_lib[-1] - log_lib[-2])
    edges[1:-1] = 0.5 * (log_lib[:-1] + log_lib[1:])
    for i, lm in enumerate(ln_q):
        k = int(np.searchsorted(edges, lm) - 1)
        k = max(0, min(len(lam_lib) - 1, k))
        g[k] += mass[i]
    return g


def fit_diffusion_spectrum(
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    *,
    n_quad: int = 96,
    n_modes: int = 8,
    oscillatory: bool = False,
    osc_amp: float = 0.55,
    osc_k: float = 3.5,
    seed: int = 0,
) -> DiffFitResult:
    """Fit diffusion-style H(λ); optionally multiply by sin oscillation (ablation)."""
    omega = np.asarray(omega, dtype=np.float64)
    gp = np.asarray(gp, dtype=np.float64)
    gpp = np.asarray(gpp, dtype=np.float64)
    lam_q = default_lambda_grid(omega, int(n_quad))
    lam_lib = default_lambda_grid(omega, int(n_modes))
    scale = float(np.median(gp[gp > 0])) if np.any(gp > 0) else 1.0
    gp_n, gpp_n = gp / scale, gpp / scale
    y = np.concatenate(
        [
            np.log10(np.maximum(gp_n, 1e-12)),
            np.log10(np.maximum(gpp_n, 1e-12)),
        ]
    )

    # params: logG0, log_lam_c, beta, log_gamma, log_ginf
    lam_c0 = float(np.exp(np.median(np.log(lam_q))))
    x0 = np.array([0.0, np.log(lam_c0), 0.0, np.log(1.0), -8.0], dtype=np.float64)
    lo = np.array([-6.0, np.log(lam_q.min()) - 1.0, -2.5, np.log(0.25), -14.0])
    hi = np.array([6.0, np.log(lam_q.max()) + 1.0, 2.5, np.log(4.0), 0.0])

    def pack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
        G0 = float(np.exp(theta[0]))
        lam_c = float(np.exp(theta[1]))
        beta = float(theta[2])
        gamma = float(np.exp(theta[3]))
        ginf = float(np.exp(theta[4]))
        return G0, lam_c, beta, gamma, ginf

    def residual(theta: np.ndarray) -> np.ndarray:
        G0, lam_c, beta, gamma, ginf = pack(theta)
        H = _H_diff(lam_q, G0, lam_c, beta, gamma)
        if oscillatory:
            phase = osc_k * np.log(np.maximum(lam_q / lam_c, 1e-30))
            H = H * np.maximum(1.0 + osc_amp * np.sin(phase), 0.05)
        gp_h, gpp_h = _moduli_from_H(omega, lam_q, H, ginf)
        yhat = np.concatenate(
            [
                np.log10(np.maximum(gp_h, 1e-12)),
                np.log10(np.maximum(gpp_h, 1e-12)),
            ]
        )
        return yhat - y

    rng = np.random.default_rng(int(seed))
    best = None
    best_cost = np.inf
    starts = [x0]
    for _ in range(3):
        starts.append(x0 + rng.normal(0.0, 0.35, size=x0.size))
    for s0 in starts:
        s0 = np.clip(s0, lo, hi)
        res = least_squares(residual, s0, bounds=(lo, hi), max_nfev=300, ftol=1e-10)
        if res.cost < best_cost:
            best_cost = float(res.cost)
            best = res

    assert best is not None
    G0, lam_c, beta, gamma, ginf = pack(best.x)
    H = _H_diff(lam_q, G0, lam_c, beta, gamma)
    if oscillatory:
        phase = osc_k * np.log(np.maximum(lam_q / lam_c, 1e-30))
        H = H * np.maximum(1.0 + osc_amp * np.sin(phase), 0.05)
    g_proj = project_H_to_modes(lam_lib, lam_q, H) * scale
    ginf_s = ginf * scale
    sc = score_freq_fit(omega, gp, gpp, lam_lib, g_proj, ginf_s)
    # Prefer dense-H moduli score when reporting fit quality
    gp_h, gpp_h = _moduli_from_H(omega, lam_q, H * scale, ginf_s)
    y_true = np.concatenate([gp, gpp])
    y_hat = np.concatenate([gp_h, gpp_h])
    rel_l2 = float(np.linalg.norm(y_hat - y_true) / (np.linalg.norm(y_true) + 1e-12))
    rel_log = float(
        np.sqrt(
            np.mean(
                (
                    np.log10(np.maximum(y_hat, 1e-12))
                    - np.log10(np.maximum(y_true, 1e-12))
                )
                ** 2
            )
        )
    )
    method = "diffusion_H_osc" if oscillatory else "diffusion_H"
    return DiffFitResult(
        method=method,
        params={
            "G0": G0 * scale,
            "lam_c": lam_c,
            "beta": beta,
            "gamma": gamma,
            "g_inf": ginf_s,
            "oscillatory": bool(oscillatory),
        },
        lambda_=lam_lib,
        g=g_proj,
        g_inf=ginf_s,
        rel_l2=rel_l2,
        rel_log=rel_log,
        rel_l2_gp=float(np.linalg.norm(gp_h - gp) / (np.linalg.norm(gp) + 1e-12)),
        rel_l2_gpp=float(np.linalg.norm(gpp_h - gpp) / (np.linalg.norm(gpp) + 1e-12)),
        success=bool(best.success),
        message=str(best.message),
    )


def fit_fractional_maxwell(
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    *,
    n_modes: int = 8,
    seed: int = 0,
) -> DiffFitResult:
    """Single fractional Maxwell mode; project effective mass onto λ library for tube."""
    omega = np.asarray(omega, dtype=np.float64)
    gp = np.asarray(gp, dtype=np.float64)
    gpp = np.asarray(gpp, dtype=np.float64)
    lam_lib = default_lambda_grid(omega, int(n_modes))
    scale = float(np.median(gp[gp > 0])) if np.any(gp > 0) else 1.0
    gp_n, gpp_n = gp / scale, gpp / scale
    y = np.concatenate(
        [np.log10(np.maximum(gp_n, 1e-12)), np.log10(np.maximum(gpp_n, 1e-12))]
    )

    # params: logG, logτ, alpha in (0.05, 0.95), log_ginf
    tau0 = float(np.exp(np.median(np.log(1.0 / np.maximum(omega, 1e-30)))))
    x0 = np.array([0.0, np.log(tau0), 0.5, -8.0])
    lo = np.array([-4.0, np.log(lam_lib.min()) - 2, 0.08, -14.0])
    hi = np.array([4.0, np.log(lam_lib.max()) + 2, 0.95, 0.0])

    def residual(theta: np.ndarray) -> np.ndarray:
        G = np.exp(theta[0])
        tau = np.exp(theta[1])
        alpha = float(theta[2])
        ginf = np.exp(theta[3])
        # (iωτ)^α = (ωτ)^α * exp(i π α / 2)
        wt = omega * tau
        mag = np.power(np.maximum(wt, 1e-30), alpha)
        c = np.cos(0.5 * np.pi * alpha)
        s = np.sin(0.5 * np.pi * alpha)
        # 1 + (iωτ)^α
        re = 1.0 + mag * c
        im = mag * s
        den = re * re + im * im
        # G / (1+(iωτ)^α)
        gp_h = ginf + G * re / den
        gpp_h = G * im / den
        yhat = np.concatenate(
            [np.log10(np.maximum(gp_h, 1e-12)), np.log10(np.maximum(gpp_h, 1e-12))]
        )
        return yhat - y

    rng = np.random.default_rng(int(seed))
    best = None
    best_cost = np.inf
    for s0 in [x0] + [x0 + rng.normal(0, 0.3, size=4) for _ in range(3)]:
        s0 = np.clip(s0, lo, hi)
        res = least_squares(residual, s0, bounds=(lo, hi), max_nfev=250)
        if res.cost < best_cost:
            best_cost = float(res.cost)
            best = res
    assert best is not None
    G = float(np.exp(best.x[0]) * scale)
    tau = float(np.exp(best.x[1]))
    alpha = float(best.x[2])
    ginf = float(np.exp(best.x[3]) * scale)

    # Effective Prony-like mass near τ for tube_corr (soft peak on library)
    log_lib = np.log(np.maximum(lam_lib, 1e-30))
    log_tau = np.log(max(tau, 1e-30))
    w = np.exp(-0.5 * ((log_lib - log_tau) / 0.35) ** 2)
    g = G * w / max(w.sum(), 1e-30)

    wt = omega * tau
    mag = np.power(np.maximum(wt, 1e-30), alpha)
    c = np.cos(0.5 * np.pi * alpha)
    s = np.sin(0.5 * np.pi * alpha)
    re = 1.0 + mag * c
    im = mag * s
    den = re * re + im * im
    gp_h = ginf + G * re / den
    gpp_h = G * im / den
    y_true = np.concatenate([gp, gpp])
    y_hat = np.concatenate([gp_h, gpp_h])
    rel_l2 = float(np.linalg.norm(y_hat - y_true) / (np.linalg.norm(y_true) + 1e-12))
    rel_log = float(
        np.sqrt(
            np.mean(
                (np.log10(np.maximum(y_hat, 1e-12)) - np.log10(np.maximum(y_true, 1e-12)))
                ** 2
            )
        )
    )
    return DiffFitResult(
        method="fractional_maxwell",
        params={"G": G, "tau": tau, "alpha": alpha, "g_inf": ginf},
        lambda_=lam_lib,
        g=g,
        g_inf=ginf,
        rel_l2=rel_l2,
        rel_log=rel_log,
        rel_l2_gp=float(np.linalg.norm(gp_h - gp) / (np.linalg.norm(gp) + 1e-12)),
        rel_l2_gpp=float(np.linalg.norm(gpp_h - gpp) / (np.linalg.norm(gpp) + 1e-12)),
        success=bool(best.success),
        message=str(best.message),
    )
