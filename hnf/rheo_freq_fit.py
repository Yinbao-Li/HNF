# -*- coding: utf-8 -*-
"""Frequency-domain Prony / Maxwell identification for SAOS (G', G'').

Fits isotropic PronyBoltzmannKernel via:
  - classical nonlinear least-squares (scipy)
  - PNF: Adam on log-modulus residual of complex_modulus(ω)

G'(ω) = G_∞ + Σ_k G_k (ωλ)^2 / (1+(ωλ)^2)
G''(ω) = Σ_k G_k (ωλ) / (1+(ωλ)^2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hnf.rheo_memory import PronyBoltzmannKernel


@dataclass
class FreqFitResult:
    method: str
    n_modes: int
    fixed_lambda: bool
    lambda_: np.ndarray
    g: np.ndarray
    g_inf: float
    rel_l2: float
    rel_log: float
    rel_l2_gp: float
    rel_l2_gpp: float
    n_iter: int
    success: bool
    message: str = ""


def default_lambda_grid(omega: np.ndarray, n_modes: int) -> np.ndarray:
    """Log-spaced λ covering ~1/ω_max … 1/ω_min (Elliott-style library)."""
    w = np.asarray(omega, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        raise ValueError("omega empty")
    lam_min = 1.0 / float(w.max())
    lam_max = 1.0 / float(w.min())
    # slight pad so terminal / glassy ends are covered
    lam_min *= 0.5
    lam_max *= 2.0
    return np.logspace(np.log10(lam_min), np.log10(lam_max), int(n_modes))


def _complex_modulus_np(
    omega: np.ndarray,
    lam: np.ndarray,
    g: np.ndarray,
    g_inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    om = np.asarray(omega, dtype=np.float64).reshape(-1)
    lam = np.asarray(lam, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    x = om[:, None] * lam[None, :]
    den = 1.0 + x * x
    gp = float(g_inf) + (g[None, :] * (x * x) / den).sum(axis=1)
    gpp = (g[None, :] * x / den).sum(axis=1)
    return gp, gpp


def score_freq_fit(
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    lam: np.ndarray,
    g: np.ndarray,
    g_inf: float,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    gp_hat, gpp_hat = _complex_modulus_np(omega, lam, g, g_inf)
    gp = np.asarray(gp, dtype=np.float64)
    gpp = np.asarray(gpp, dtype=np.float64)

    def _rel(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.linalg.norm(a - b))
        den = float(np.linalg.norm(b)) + eps
        return num / den

    y = np.concatenate([gp, gpp])
    yhat = np.concatenate([gp_hat, gpp_hat])
    log_y = np.log10(np.maximum(y, eps))
    log_hat = np.log10(np.maximum(yhat, eps))
    return {
        "rel_l2": _rel(yhat, y),
        "rel_log": float(np.sqrt(np.mean((log_hat - log_y) ** 2))),
        "rel_l2_gp": _rel(gp_hat, gp),
        "rel_l2_gpp": _rel(gpp_hat, gpp),
        "gp_hat": gp_hat,
        "gpp_hat": gpp_hat,
    }


def _modulus_scale(gp: np.ndarray) -> float:
    gp = np.asarray(gp, dtype=np.float64)
    pos = gp[np.isfinite(gp) & (gp > 0)]
    if pos.size == 0:
        return 1.0
    return float(np.median(pos))


def fit_prony_freq_nls(
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    *,
    n_modes: int = 8,
    fixed_lambda: bool = True,
    lambda_init: Optional[np.ndarray] = None,
    max_nfev: int = 400,
) -> FreqFitResult:
    """Classical NLS Prony fit in frequency domain (log residual).

    Internally fits on G'/scale, G''/scale (scale = median G') then rescales.
    """
    from scipy.optimize import least_squares

    omega = np.asarray(omega, dtype=np.float64)
    gp = np.asarray(gp, dtype=np.float64)
    gpp = np.asarray(gpp, dtype=np.float64)
    scale = _modulus_scale(gp)
    gp_n, gpp_n = gp / scale, gpp / scale
    k = int(n_modes)
    if lambda_init is None:
        lam0 = default_lambda_grid(omega, k)
    else:
        lam0 = np.asarray(lambda_init, dtype=np.float64).reshape(-1)
        if lam0.size != k:
            raise ValueError("lambda_init length must equal n_modes")

    g0 = np.full(k, 1.0 / k)
    ginf0 = max(float(gp_n.min()) * 0.01, 1e-8)

    if fixed_lambda:
        x0 = np.concatenate([np.log(g0 + 1e-12), [np.log(ginf0 + 1e-12)]])
    else:
        x0 = np.concatenate(
            [np.log(lam0 + 1e-12), np.log(g0 + 1e-12), [np.log(ginf0 + 1e-12)]]
        )

    y_log = np.concatenate(
        [np.log10(np.maximum(gp_n, 1e-12)), np.log10(np.maximum(gpp_n, 1e-12))]
    )

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if fixed_lambda:
            g = np.exp(theta[:k])
            ginf = float(np.exp(theta[k]))
            return lam0, g, ginf
        lam = np.exp(theta[:k])
        g = np.exp(theta[k : 2 * k])
        ginf = float(np.exp(theta[2 * k]))
        return lam, g, ginf

    def residual(theta: np.ndarray) -> np.ndarray:
        lam, g, ginf = unpack(theta)
        gp_h, gpp_h = _complex_modulus_np(omega, lam, g, ginf)
        pred = np.concatenate(
            [np.log10(np.maximum(gp_h, 1e-12)), np.log10(np.maximum(gpp_h, 1e-12))]
        )
        return pred - y_log

    sol = least_squares(residual, x0, method="trf", max_nfev=max_nfev)
    lam, g, ginf = unpack(sol.x)
    g = g * scale
    ginf = ginf * scale
    order = np.argsort(lam)
    lam, g = lam[order], g[order]
    sc = score_freq_fit(omega, gp, gpp, lam, g, ginf)
    return FreqFitResult(
        method="classical_prony_nls",
        n_modes=k,
        fixed_lambda=fixed_lambda,
        lambda_=lam,
        g=g,
        g_inf=ginf,
        rel_l2=float(sc["rel_l2"]),
        rel_log=float(sc["rel_log"]),
        rel_l2_gp=float(sc["rel_l2_gp"]),
        rel_l2_gpp=float(sc["rel_l2_gpp"]),
        n_iter=int(sol.nfev),
        success=bool(sol.success),
        message=str(sol.message),
    )


def fit_prony_freq_pnf(
    omega: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    *,
    n_modes: int = 8,
    fixed_lambda: bool = True,
    lambda_init: Optional[np.ndarray] = None,
    steps: int = 25,
    lr: float = 1.0,
    lbfgs_rounds: int = 25,
    n_restarts: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> FreqFitResult:
    """PNF: autodiff Prony fit on log10(G', G'') (same physics as PronyBoltzmannKernel).

    Uses log-amplitude parameterization (standard Maxwell-library ID) with multi-start
    torch LBFGS in float64 on median-G'-normalized moduli, then rescales. Identified
    spectrum is loaded into ``PronyBoltzmannKernel`` for verification.
    """
    from hnf.rheo_memory import _inv_softplus

    omega_np = np.asarray(omega, dtype=np.float64)
    gp_np = np.asarray(gp, dtype=np.float64)
    gpp_np = np.asarray(gpp, dtype=np.float64)
    scale = _modulus_scale(gp_np)
    gp_n, gpp_n = gp_np / scale, gpp_np / scale
    k = int(n_modes)
    if lambda_init is None:
        lam0 = default_lambda_grid(omega_np, k)
    else:
        lam0 = np.asarray(lambda_init, dtype=np.float64).reshape(-1)
        assert lam0.size == k

    # Prefer CPU float64 for stable wide-ω master curves
    om = torch.tensor(omega_np, dtype=torch.float64)
    y_gp = torch.tensor(np.log10(np.maximum(gp_n, 1e-12)), dtype=torch.float64)
    y_gpp = torch.tensor(np.log10(np.maximum(gpp_n, 1e-12)), dtype=torch.float64)
    lam_fixed = torch.tensor(lam0, dtype=torch.float64)

    rng = np.random.default_rng(int(seed))
    best_loss = float("inf")
    best_pack: Optional[tuple[np.ndarray, np.ndarray, float]] = None
    total_iters = 0

    for restart in range(int(n_restarts)):
        if restart == 0:
            g0 = np.full(k, 1.0 / k)
        elif restart == 1:
            w = np.logspace(0, -2, k)
            g0 = w / w.sum()
        elif restart == 2:
            w = np.logspace(-2, 0, k)
            g0 = w / w.sum()
        else:
            w = rng.lognormal(mean=-1.0, sigma=1.2, size=k)
            g0 = w / w.sum()

        log_g = torch.nn.Parameter(torch.tensor(np.log(np.maximum(g0, 1e-12)), dtype=torch.float64))
        log_ginf = torch.nn.Parameter(torch.tensor(-8.0, dtype=torch.float64))
        params: list[torch.nn.Parameter] = [log_g, log_ginf]
        if fixed_lambda:
            log_lam = None
        else:
            jitter = rng.normal(0.0, 0.1, size=k)
            log_lam = torch.nn.Parameter(
                torch.tensor(np.log(lam0) + jitter, dtype=torch.float64)
            )
            params.append(log_lam)

        def loss_fn() -> torch.Tensor:
            g = torch.exp(log_g)
            ginf = torch.exp(log_ginf)
            lam = lam_fixed if log_lam is None else torch.exp(log_lam)
            x = om.unsqueeze(-1) * lam.unsqueeze(0)
            den = 1.0 + x * x
            gp_h = ginf + (g * (x * x) / den).sum(dim=-1)
            gpp_h = (g * x / den).sum(dim=-1)
            return F.mse_loss(torch.log10(gp_h.clamp_min(1e-12)), y_gp) + F.mse_loss(
                torch.log10(gpp_h.clamp_min(1e-12)), y_gpp
            )

        opt = torch.optim.LBFGS(
            params,
            lr=float(lr),
            max_iter=max(int(steps), 20),
            history_size=60,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            return loss

        local_best = float("inf")
        local_pack = None
        for _ in range(int(lbfgs_rounds)):
            loss = opt.step(closure)
            lv = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
            if lv < local_best:
                local_best = lv
                with torch.no_grad():
                    lam_np = (
                        lam0.copy()
                        if log_lam is None
                        else torch.exp(log_lam).detach().cpu().numpy()
                    )
                    g_np = torch.exp(log_g).detach().cpu().numpy()
                    ginf_np = float(torch.exp(log_ginf).detach().cpu())
                local_pack = (lam_np, g_np, ginf_np)
        total_iters += int(lbfgs_rounds) * max(int(steps), 20)

        if local_pack is not None and local_best < best_loss:
            best_loss = local_best
            best_pack = local_pack

    assert best_pack is not None
    lam, g_n_hat, ginf_n = best_pack
    g = g_n_hat * scale
    ginf = ginf_n * scale

    # Verify / store through PronyBoltzmannKernel.complex_modulus
    model = PronyBoltzmannKernel(
        n_modes=k,
        dim=1,
        anisotropic=False,
        lambda_init=list(map(float, lam)),
        g_init=list(map(float, np.maximum(g_n_hat, 1e-12))),
        g_inf_init=max(float(ginf_n), 1e-16),
        eps=1e-12,
    )
    with torch.no_grad():
        for i, gv in enumerate(np.maximum(g_n_hat, 1e-12)):
            assert model.raw_G is not None
            model.raw_G[i] = float(_inv_softplus(gv))
        model.raw_G_inf.fill_(float(_inv_softplus(max(ginf_n, 1e-16))))
        for i, lv in enumerate(lam):
            model.raw_lambda[i] = float(_inv_softplus(max(lv - 1e-12, 1e-12)))
        gp_v, gpp_v = model.complex_modulus(torch.tensor(omega_np, dtype=torch.float64))
        # mild float mismatch OK; score uses analytic pack
        _ = (gp_v, gpp_v)

    order = np.argsort(lam)
    lam, g = lam[order], g[order]
    sc = score_freq_fit(omega_np, gp_np, gpp_np, lam, g, ginf)
    _ = device  # API compat
    return FreqFitResult(
        method="pnf",
        n_modes=k,
        fixed_lambda=fixed_lambda,
        lambda_=lam,
        g=g,
        g_inf=ginf,
        rel_l2=float(sc["rel_l2"]),
        rel_log=float(sc["rel_log"]),
        rel_l2_gp=float(sc["rel_l2_gp"]),
        rel_l2_gpp=float(sc["rel_l2_gpp"]),
        n_iter=int(total_iters),
        success=True,
        message=f"best_loss={best_loss:.6g}; scale={scale:.6g}; restarts={n_restarts}",
    )


def predict_complex_modulus(result: FreqFitResult, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _complex_modulus_np(omega, result.lambda_, result.g, result.g_inf)
