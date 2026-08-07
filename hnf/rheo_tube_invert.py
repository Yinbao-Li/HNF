# -*- coding: utf-8 -*-
"""Nonlinear tube-heuristic inversion: Maxwell G_k → MWD mixture weights.

Inverts the same simplified entanglement map used in ``run_rheo_mwd_transfer``
(τ ≈ τ_e (M/Me)^α), so gains vs linear Ridge isolate **nonlinearity / inversion**,
not a secretly richer constitutive model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize
from sklearn.neural_network import MLPRegressor

from hnf.rheo_elliott_compat import (
    M_E,
    MWD_MEANS,
    MWD_X,
    NUM_MWD_PARAMS,
    TAU_E_180,
    TAU_VALUES,
    mwd_from_weights,
)


def tau_of_M(
    M: np.ndarray,
    *,
    Me: float = M_E,
    tau_e: float = TAU_E_180,
    alpha: float = 3.4,
) -> np.ndarray:
    return tau_e * (np.maximum(M, Me) / Me) ** alpha


def forward_G_from_weights(
    weights: np.ndarray,
    tau: np.ndarray = TAU_VALUES,
    *,
    Me: float = M_E,
    tau_e: float = TAU_E_180,
    alpha: float = 3.4,
    g_scale: float = 1e5,
) -> np.ndarray:
    """Approximate Maxwell amplitudes from MWD mixture weights via tube bins."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    x = MWD_X
    y = mwd_from_weights(weights, x)
    logx = np.log(x)
    taus = tau_of_M(x, Me=Me, tau_e=tau_e, alpha=alpha)
    g = np.zeros(len(tau), dtype=np.float64)
    dlog = np.empty_like(logx)
    dlog[1:-1] = 0.5 * (logx[2:] - logx[:-2])
    dlog[0] = logx[1] - logx[0]
    dlog[-1] = logx[-1] - logx[-2]
    log_tau = np.log(np.maximum(tau, 1e-30))
    for j in range(len(x)):
        if taus[j] < tau_e:
            continue
        k = int(np.argmin(np.abs(log_tau - np.log(max(taus[j], 1e-30)))))
        g[k] += y[j] * dlog[j]
    g = np.maximum(g, 0.0)
    if g.sum() <= 0:
        g[:] = 1e-8
    g = g / g.sum() * g_scale
    return g


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / max(e.sum(), 1e-30)


@dataclass
class TubeInvertResult:
    weights: np.ndarray
    mwd: np.ndarray
    g_fit: np.ndarray
    rel_log_g: float
    n_iter: int
    success: bool
    message: str = ""


def invert_tube_weights(
    g_obs: np.ndarray,
    *,
    w0: Optional[np.ndarray] = None,
    lam_entropy: float = 0.02,
    lam_smooth: float = 0.05,
    max_nfev: int = 2500,
) -> TubeInvertResult:
    """Fit mixture weights so forward_G(w) matches observed Maxwell amplitudes."""
    g_obs = np.maximum(np.asarray(g_obs, dtype=np.float64), 0.0)
    g_scale = float(g_obs.sum()) if g_obs.sum() > 0 else 1e5
    target = np.log10(np.maximum(g_obs, 1e-12))

    if w0 is None:
        # Broad prior peaked near Me-scale components
        z0 = -0.02 * (np.arange(NUM_MWD_PARAMS) - NUM_MWD_PARAMS / 2.0) ** 2
    else:
        w0 = np.maximum(np.asarray(w0, dtype=np.float64), 1e-12)
        w0 = w0 / w0.sum()
        z0 = np.log(w0)

    def pack_loss(z: np.ndarray) -> float:
        w = _softmax(z)
        g = forward_G_from_weights(w, g_scale=g_scale)
        pred = np.log10(np.maximum(g, 1e-12))
        data = float(np.mean((pred - target) ** 2))
        # Prefer fewer active components (entropy of w)
        ent = -float(np.sum(w * np.log(np.maximum(w, 1e-30))))
        # Smoothness on log-weight landscape along mixture index
        sm = float(np.sum((z[2:] - 2 * z[1:-1] + z[:-2]) ** 2))
        return data + lam_entropy * ent + lam_smooth * sm / NUM_MWD_PARAMS

    sol = optimize.minimize(
        pack_loss,
        z0,
        method="L-BFGS-B",
        options={"maxfun": max_nfev, "ftol": 1e-12},
    )
    w = _softmax(sol.x)
    g_fit = forward_G_from_weights(w, g_scale=g_scale)
    pred = np.log10(np.maximum(g_fit, 1e-12))
    rel = float(np.sqrt(np.mean((pred - target) ** 2)))
    return TubeInvertResult(
        weights=w,
        mwd=mwd_from_weights(w),
        g_fit=g_fit,
        rel_log_g=rel,
        n_iter=int(sol.nit),
        success=bool(sol.success),
        message=str(sol.message),
    )


def sample_synth_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    W = np.zeros((n, NUM_MWD_PARAMS), dtype=np.float64)
    for i in range(n):
        n_comp = int(rng.integers(2, 8))
        idx = rng.choice(NUM_MWD_PARAMS, size=n_comp, replace=False)
        amps = rng.lognormal(mean=0.0, sigma=0.8, size=n_comp)
        center = NUM_MWD_PARAMS // 2
        amps *= np.exp(-0.02 * (idx - center) ** 2)
        W[i, idx] = amps
        W[i] /= max(W[i].sum(), 1e-30)
    return W


def train_synth_mlp(
    n_synth: int,
    seed: int,
    *,
    noise_std: float = 0.15,
    hidden: tuple[int, ...] = (128, 64),
) -> MLPRegressor:
    """Nonlinear map log10 G → mixture weights, same synth prior as Ridge."""
    rng = np.random.default_rng(seed)
    W = sample_synth_weights(rng, n_synth)
    X = np.zeros((n_synth, len(TAU_VALUES)), dtype=np.float64)
    for i in range(n_synth):
        g = forward_G_from_weights(W[i])
        logg = np.log10(np.maximum(g, 1e-12))
        X[i] = logg + rng.normal(0.0, noise_std, size=logg.shape)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=400,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X, W)
    return model


def predict_weights_mlp(model: MLPRegressor, log_g_nat: np.ndarray) -> np.ndarray:
    """``log_g_nat`` is natural log of G_k (Elliott fit); convert to log10."""
    x = (np.asarray(log_g_nat, dtype=np.float64) / np.log(10.0)).reshape(1, -1)
    w = np.maximum(model.predict(x)[0], 0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def hybrid_invert(
    g_obs: np.ndarray,
    w_prior: Optional[np.ndarray] = None,
    *,
    prior_mix: float = 0.35,
) -> TubeInvertResult:
    """Warm-start inversion from a prior weight vector (e.g. synth MLP/Ridge)."""
    res = invert_tube_weights(g_obs, w0=w_prior)
    if w_prior is None or prior_mix <= 0:
        return res
    w_p = np.maximum(np.asarray(w_prior, dtype=np.float64), 0.0)
    w_p = w_p / max(w_p.sum(), 1e-30)
    w = (1.0 - prior_mix) * res.weights + prior_mix * w_p
    w = w / max(w.sum(), 1e-30)
    g_scale = float(np.maximum(g_obs, 0.0).sum()) if np.sum(g_obs) > 0 else 1e5
    g_fit = forward_G_from_weights(w, g_scale=g_scale)
    target = np.log10(np.maximum(g_obs, 1e-12))
    pred = np.log10(np.maximum(g_fit, 1e-12))
    return TubeInvertResult(
        weights=w,
        mwd=mwd_from_weights(w),
        g_fit=g_fit,
        rel_log_g=float(np.sqrt(np.mean((pred - target) ** 2))),
        n_iter=res.n_iter,
        success=res.success,
        message=f"hybrid prior_mix={prior_mix}; {res.message}",
    )
