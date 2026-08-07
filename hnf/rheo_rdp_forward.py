# -*- coding: utf-8 -*-
"""Headless Rolie–Double–Poly (RDP) LVE forward: MWD → Maxwell G_k.

Ports the physics in RepTate ``TheoryRDPLVE`` + ``Dilution.relax_times_from_mwd``
(Likhtman–McLeish CLF, dynamic dilution) without Qt.

This is the G4b step-up from the α=3.4 mass-deposit heuristic: still not Elliott's
unpublished BoB/tube training engine, but a standard entangled-linear LVE theory
callable in pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hnf.rheo_elliott_compat import (
    G0_180,
    M_E,
    MWD_X,
    NUM_MWD_PARAMS,
    TAU_E_180,
    TAU_VALUES,
    mwd_from_weights,
)


def fZ_clf(z: float) -> float:
    """Likhtman–McLeish (2002) CLF factor for terminal time."""
    if z <= 1.0:
        return 1.0
    s = np.sqrt(z)
    return float(1.0 - 2 * 1.69 / s + 4.17 / z - 1.55 / (z * s))


def gZ_clf(z: float) -> float:
    """Likhtman–McLeish CLF factor for modulus."""
    if z <= 1.0:
        return 1.0
    s = np.sqrt(z)
    return float(1.0 - 1.69 / s + 2.0 / z - 1.24 / (z * s))


def _find_down_indx(tauseff: float, taud: np.ndarray) -> int:
    n = len(taud)
    down = n - 1
    while tauseff < taud[down]:
        if down == 0:
            return -1
        down -= 1
    return down


def _find_dilution(phi: np.ndarray, taud: np.ndarray, taus: float, interp: bool) -> float:
    n = len(phi)
    temp = -1
    phi_dil = 1.0
    tauseff = taus / phi_dil
    while True:
        down = _find_down_indx(tauseff, taud)
        if down == -1:
            return 1.0
        if down == n - 1:
            return float(phi[n - 1])
        if temp == down:
            return phi_dil
        temp = down
        phi_dil = 1.0 - float(np.sum(phi[:down]))
        if interp:
            x = (tauseff - taud[down]) / (taud[down + 1] - taud[down] + 1e-30)
            phi_dil = phi_dil - x * float(phi[down])
        else:
            phi_dil -= float(phi[down])
        tauseff = taus / max(phi_dil, 1e-12)


@dataclass
class RDPModes:
    m: np.ndarray
    phi: np.ndarray
    taud: np.ndarray
    zeff: np.ndarray
    ok: bool


def discretize_mwd_to_bins(
    y: np.ndarray,
    x: np.ndarray = MWD_X,
    *,
    n_bins: int = 40,
    m_min: float = 2.0 * M_E,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse continuous dW/dlnM onto log-spaced mass bins (volume ≈ weight fractions)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    logx = np.log(x)
    edges = np.linspace(np.log(x.min()), np.log(x.max()), n_bins + 1)
    m_c = []
    phi = []
    for i in range(n_bins):
        mask = (logx >= edges[i]) & (logx < edges[i + 1] if i < n_bins - 1 else logx <= edges[i + 1])
        if not np.any(mask):
            continue
        w = float(np.trapezoid(y[mask], logx[mask])) if hasattr(np, "trapezoid") else float(np.trapz(y[mask], logx[mask]))
        if w <= 0:
            continue
        m_avg = float(np.exp(np.average(logx[mask], weights=np.maximum(y[mask], 1e-30))))
        m_c.append(m_avg)
        phi.append(w)
    m_c = np.asarray(m_c, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if phi.size == 0:
        return np.array([10 * M_E]), np.array([1.0])
    phi = phi / max(phi.sum(), 1e-30)
    # drop ultra-light solvent bins later in dilution; keep all here
    return m_c, phi


def rdp_modes_from_mwd(
    m: np.ndarray,
    phi: np.ndarray,
    *,
    me: float = M_E,
    taue: float = TAU_E_180,
) -> RDPModes:
    """RepTate Dilution.relax_times_from_mwd (linear melts)."""
    m = np.asarray(m, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    order = np.argsort(m)
    m, phi = m[order].tolist(), phi[order].tolist()

    taus: list[float] = []
    phi_u = 0.0
    nshort = 0
    n = len(m)
    for i in range(n):
        z = m[i] / me
        ts = z * z * taue
        if m[i] < 2.0 * me:
            nshort += 1
            phi_u += phi[i]
        else:
            taus.append(ts)

    m = m[nshort:]
    phi = phi[nshort:]
    n = len(m)
    if n == 0:
        return RDPModes(m=np.array([]), phi=np.array([]), taud=np.array([]), zeff=np.array([]), ok=False)
    if n == 1:
        z = m[0] / me
        return RDPModes(
            m=np.array(m),
            phi=np.array([1.0]),
            taud=np.array([3.0 * z * taus[0]]),
            zeff=np.array([z]),
            ok=True,
        )

    me_eff = me / max(1.0 - phi_u, 1e-12)
    taue_eff = taue / max((1.0 - phi_u) ** 2, 1e-12)
    phi = [p / max(1.0 - phi_u, 1e-12) for p in phi]
    taud = [3.0 * (m[i] / me_eff) ** 3 * taue_eff for i in range(n)]
    phi_a = np.asarray(phi, dtype=np.float64)
    taud_a = np.asarray(taud, dtype=np.float64)
    taus_a = np.asarray(taus, dtype=np.float64)

    vphi = []
    interp = n > 2
    for i in range(n):
        if i == 0:
            vphi.append(1.0)
        else:
            vphi.append(_find_dilution(phi_a, taud_a, float(taus_a[i]), interp=interp))

    zeff = np.zeros(n)
    for i in range(n):
        z = m[i] / me_eff
        if z * vphi[i] < 1.0 and z > 1.0:
            sticky = taud_a[i - 1]
            for j in range(i, n):
                taud_a[j] = sticky
                zeff[j] = 1.0
            break
        fz = fZ_clf(z * vphi[i])
        if fz <= 0:
            fz = 1e-6
        taud_a[i] = taud_a[i] * fz
        zeff[i] = z * vphi[i]

    return RDPModes(
        m=np.asarray(m, dtype=np.float64),
        phi=phi_a,
        taud=taud_a,
        zeff=zeff,
        ok=True,
    )


def rdp_deposit_maxwell(
    modes: RDPModes,
    *,
    gn0: float = G0_180,
    tau_grid: np.ndarray = TAU_VALUES,
    with_gcorr: bool = True,
) -> np.ndarray:
    """Project RDP double-reptation spectrum onto fixed Elliott τ library."""
    g = np.zeros(len(tau_grid), dtype=np.float64)
    if not modes.ok or modes.phi.size == 0:
        g[:] = 1e-8
        return g
    log_tau = np.log(np.maximum(tau_grid, 1e-30))
    n = len(modes.phi)
    for i in range(n):
        Gi = gn0 * (gZ_clf(float(modes.zeff[i])) if with_gcorr else 1.0)
        for j in range(n):
            tau = 1.0 / (1.0 / modes.taud[i] + 1.0 / modes.taud[j])
            amp = Gi * modes.phi[i] * modes.phi[j]
            k = int(np.argmin(np.abs(log_tau - np.log(max(tau, 1e-30)))))
            g[k] += amp
    g = np.maximum(g, 0.0)
    if g.sum() <= 0:
        g[:] = 1e-8
    return g


def forward_G_from_mwd_weights(
    weights: np.ndarray,
    *,
    n_bins: int = 40,
    with_gcorr: bool = True,
) -> Optional[np.ndarray]:
    """28 lognormal weights → RDP Maxwell amplitudes on Elliott τ grid."""
    y = mwd_from_weights(weights)
    m, phi = discretize_mwd_to_bins(y, n_bins=n_bins)
    modes = rdp_modes_from_mwd(m, phi)
    if not modes.ok:
        return None
    return rdp_deposit_maxwell(modes, with_gcorr=with_gcorr)


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
