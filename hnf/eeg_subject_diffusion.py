# -*- coding: utf-8 -*-
"""Subject-level anisotropic diffusion fit + residual / atrophy helpers.

Covariance-to-D: resting EEG channel correlation ≈ anisotropic diffusion
Green kernel on 10–20 geometry. Global checkpoint D is *not* a biomarker;
this amortized-free per-subject fit is.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from hnf.eeg_dataset import STANDARD_10_20
from hnf.eeg_geometry import REGION_CHANNELS, electrode_xyz


# Literature scalp-rank atrophy templates (unitless; not subject-matched MRI).
# Higher = more expected degeneration. AD: posterior/temporal; FTD: frontotemporal.
ATROPHY_TEMPLATES: dict[str, dict[str, float]] = {
    "AD": {"frontal": 0.40, "temporal": 0.85, "central": 0.45, "posterior": 0.95},
    "FTD": {"frontal": 0.95, "temporal": 0.90, "central": 0.35, "posterior": 0.25},
}
TEMPLATE_REGION_ORDER = ("frontal", "temporal", "central", "posterior")


def residualize(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """OLS residual of ``y`` on ``X`` (adds intercept). Returns resid, R², beta."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    resid = np.full(y.shape, np.nan, dtype=np.float64)
    if int(mask.sum()) < X.shape[1] + 3:
        return resid, float("nan"), np.full(X.shape[1] + 1, np.nan)
    y_m, X_m = y[mask], X[mask]
    A = np.column_stack([np.ones(len(y_m)), X_m])
    beta, *_ = np.linalg.lstsq(A, y_m, rcond=None)
    fit = A @ beta
    resid[mask] = y_m - fit
    denom = float(np.var(y_m))
    r2 = float(1.0 - np.var(y_m - fit) / denom) if denom > 1e-12 else float("nan")
    return resid, r2, beta


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 4:
        return float("nan")
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def spearman_r(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a[m], b[m])
    return float(r), float(p)


def icc_abs_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 5:
        return float("nan")
    x = np.stack([a[m], b[m]], axis=1)
    n = x.shape[0]
    grand = x.mean()
    ms_rows = ((x.mean(axis=1) - grand) ** 2).sum() * 2 / max(n - 1, 1)
    ms_err = (
        (x - x.mean(axis=1, keepdims=True) - x.mean(axis=0) + grand) ** 2
    ).sum() / max((n - 1), 1)
    return float((ms_rows - ms_err) / (ms_rows + ms_err + 1e-12))


def _pack_L(vec: np.ndarray) -> np.ndarray:
    L = np.zeros((3, 3), dtype=np.float64)
    L[0, 0] = np.exp(np.clip(vec[0], -4.0, 1.5))
    L[1, 0] = float(np.clip(vec[1], -2.0, 2.0))
    L[1, 1] = np.exp(np.clip(vec[2], -4.0, 1.5))
    L[2, 0] = float(np.clip(vec[3], -2.0, 2.0))
    L[2, 1] = float(np.clip(vec[4], -2.0, 2.0))
    L[2, 2] = np.exp(np.clip(vec[5], -4.0, 1.5))
    return L


def diffusion_kernel_from_vec(
    xyz: np.ndarray,
    vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(K, D, tau)`` from 7-vector ``[logL00, L10, logL11, L20, L21, logL22, log_tau]``."""
    L = _pack_L(vec[:6])
    D = L @ L.T + 1e-4 * np.eye(3)
    tau = float(np.exp(vec[6]))
    delta = xyz[:, None, :] - xyz[None, :, :]
    inv_d = np.linalg.inv(D)
    maha2 = np.einsum("ija,ab,ijb->ij", delta, inv_d, delta)
    K = np.exp(-maha2 / (4.0 * max(tau, 1e-6)))
    np.fill_diagonal(K, 0.0)
    return K, D, tau


def channel_correlation(epochs: np.ndarray) -> np.ndarray:
    """Mean channel correlation over epochs ``(N, C, T)``."""
    x = np.asarray(epochs, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    acc = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    n = 0
    for ep in x:
        c = np.corrcoef(ep)
        if not np.isfinite(c).all():
            continue
        acc += np.clip(c, -0.999, 0.999)
        n += 1
    if n == 0:
        return np.eye(x.shape[1])
    out = acc / n
    np.fill_diagonal(out, 1.0)
    return out


def fit_subject_diffusion(
    epochs: np.ndarray,
    *,
    xyz: Optional[np.ndarray] = None,
    maxiter: int = 120,
) -> dict[str, float | list[float]]:
    """Fit anisotropic D so K(D,τ) matches off-diagonal channel correlation."""
    from scipy.optimize import minimize

    xyz = electrode_xyz() if xyz is None else np.asarray(xyz, dtype=np.float64)
    C = channel_correlation(epochs)
    mask = ~np.eye(C.shape[0], dtype=bool)
    target = C[mask]
    target = target - target.mean()
    tnorm = float(np.linalg.norm(target)) + 1e-12

    def loss(vec: np.ndarray) -> float:
        K, _, _ = diffusion_kernel_from_vec(xyz, vec)
        k = K[mask]
        k = k - k.mean()
        # best scalar gain
        kn = float(np.linalg.norm(k)) + 1e-12
        cos = float((k * target).sum() / (kn * tnorm))
        return float(1.0 - cos)

    x0 = np.array([-1.2, 0.0, -1.2, 0.0, 0.0, -1.2, -1.4], dtype=np.float64)
    bounds = [(-4.0, 1.5), (-2.0, 2.0), (-4.0, 1.5), (-2.0, 2.0), (-2.0, 2.0), (-4.0, 1.5), (-4.0, 1.0)]
    res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
    vec = np.asarray(res.x, dtype=np.float64)
    K, D, tau = diffusion_kernel_from_vec(xyz, vec)
    k = K[mask]
    fit_r = pearson_r(k, C[mask])
    eig = np.sort(np.linalg.eigvalsh(D))
    trace = float(eig.sum())
    aniso = float(eig[-1] / max(eig[0], 1e-12))
    # fractional anisotropy of 3-vector eigenvalues
    mu = trace / 3.0
    fa = float(
        np.sqrt(1.5 * np.sum((eig - mu) ** 2) / max(np.sum(eig**2), 1e-12))
    )
    out: dict[str, float | list[float]] = {
        "D_eig0": float(eig[0]),
        "D_eig1": float(eig[1]),
        "D_eig2": float(eig[2]),
        "D_trace": trace,
        "D_aniso": aniso,
        "D_fa": fa,
        "tau": tau,
        "fit_r": float(fit_r),
        "fit_loss": float(res.fun),
        "fit_success": float(bool(res.success)),
        "D_flat": D.reshape(-1).tolist(),
    }
    out.update(region_coupling(K))
    return out


def region_index_arrays(
    names: tuple[str, ...] = STANDARD_10_20,
) -> dict[str, np.ndarray]:
    name_to_i = {n: i for i, n in enumerate(names)}
    out = {}
    for region, chans in REGION_CHANNELS.items():
        out[region] = np.asarray([name_to_i[c] for c in chans], dtype=np.int64)
    return out


def region_coupling(K: np.ndarray, names: tuple[str, ...] = STANDARD_10_20) -> dict[str, float]:
    idx = region_index_arrays(names)
    out: dict[str, float] = {}
    for region, ii in idx.items():
        sub = K[np.ix_(ii, ii)]
        n = len(ii)
        off = float(sub.sum() - np.trace(sub)) / max(n * (n - 1), 1)
        out[f"couple_{region}"] = off
    fi, ti, pi = idx["frontal"], idx["temporal"], idx["posterior"]
    out["couple_ft_cross"] = float(K[np.ix_(fi, ti)].mean())
    out["couple_pf_cross"] = float(K[np.ix_(pi, fi)].mean())
    out["couple_pf_contrast"] = out["couple_posterior"] - out["couple_frontal"]
    out["couple_ft_contrast"] = out["couple_frontal"] - out["couple_temporal"]
    return out


def atrophy_template_scores(couple: dict[str, float]) -> dict[str, float]:
    """Decoupling profile vs AD/FTD literature templates.

    ``profile = -coupling`` so weaker within-region coupling scores as more
    atrophy-like. Returns Pearson match to each template and FTD−AD delta.
    """
    profile = np.asarray(
        [-float(couple.get(f"couple_{r}", np.nan)) for r in TEMPLATE_REGION_ORDER],
        dtype=np.float64,
    )
    out: dict[str, float] = {}
    for name, tmpl in ATROPHY_TEMPLATES.items():
        t = np.asarray([tmpl[r] for r in TEMPLATE_REGION_ORDER], dtype=np.float64)
        out[f"tmpl_{name}"] = pearson_r(profile, t)
    out["tmpl_ftd_minus_ad"] = float(out["tmpl_FTD"] - out["tmpl_AD"])
    return out


def sex_to_float(gender: Iterable[object] | np.ndarray) -> np.ndarray:
    out = []
    for g in gender:
        s = str(g).strip().upper()[:1]
        if s == "M":
            out.append(1.0)
        elif s == "F":
            out.append(0.0)
        else:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)
