# -*- coding: utf-8 -*-
"""EEG temporal-chain characterization (early→late frame + θ→α propagation).

STEAD causal chain anchors at P (tau=0) and S (tau=1). EEG has no picks, so we use:

1. **Epoch frame** — tau ∈ [0, 1] over the 10 s epoch (early → late slowing drift).
2. **Rhythm propagation** — theta_env vs alpha_env cross-correlation lag and the
   theta/alpha ratio trajectory in tau (slowing / α-reactivity proxy).

Clustering uses *shape* (peak-normalised trajectories), not absolute amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# Normalised epoch time grid (0 = epoch start, 1 = epoch end).
TAU_LO = 0.0
TAU_HI = 1.0
TAU_N = 32

EEG_CHAIN_SHAPE_NAMES: tuple[str, ...] = tuple(
    [f"rho_tau_{i}" for i in range(TAU_N)]
    + [f"ta_ratio_tau_{i}" for i in range(TAU_N)]
    + [
        "theta_alpha_lag_norm",
        "rho_early_late_drift",
        "alpha_peak_tau",
        "theta_peak_tau",
        "rho_late_decay",
        "theta_alpha_coupling",
    ]
)


@dataclass
class EEGTemporalObservables:
    """Time-resolved chain for one EEG epoch."""

    rho: np.ndarray
    theta_env: np.ndarray
    alpha_env: np.ndarray
    delta_env: Optional[np.ndarray]
    wave_mean: np.ndarray
    epoch_sec: float
    clinical_group: str = ""
    subject_id: str = ""


def _smooth(series: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win) | 1)
    if win <= 1 or series.size < win:
        return series.astype(np.float64)
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(series.astype(np.float64), kernel, mode="same")


def _resample_epoch_tau(series: np.ndarray, epoch_sec: float) -> np.ndarray:
    """Sample physical-time series onto tau ∈ [0, 1]."""
    seq = series.size
    phys_t = np.linspace(0.0, epoch_sec, seq)
    tau_grid = np.linspace(TAU_LO, TAU_HI, TAU_N)
    query = tau_grid * epoch_sec
    return np.interp(query, phys_t, series, left=float(series[0]), right=float(series[-1]))


def _crosscorr_lag(a: np.ndarray, b: np.ndarray, epoch_sec: float) -> float:
    """Normalised lag (fraction of epoch) at max cross-correlation."""
    a = a - a.mean()
    b = b - b.mean()
    if a.size < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    corr = np.correlate(a, b, mode="full")
    lag_idx = int(np.argmax(corr)) - (a.size - 1)
    return float(lag_idx / max(a.size - 1, 1))


@torch.no_grad()
def extract_eeg_temporal_observables(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    epoch_sec: float = 10.0,
    clinical_group: str = "",
    subject_id: str = "",
) -> EEGTemporalObservables:
    """Dense forward; pull rho and rhythm envelopes."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    logits, aux = model(x, return_aux=True)
    del logits

    rho = aux["rho"][0, :, 0].detach().float().cpu().numpy()
    th = aux["theta_env"][0].detach().float().cpu().numpy().mean(axis=-1)
    al = aux["alpha_env"][0].detach().float().cpu().numpy().mean(axis=-1)
    de = aux.get("delta_env")
    de_np = de[0].detach().float().cpu().numpy().mean(axis=-1) if de is not None else None
    wave = x[0].detach().float().cpu().numpy().mean(axis=0)

    win = max(3, rho.size // 32)
    return EEGTemporalObservables(
        rho=_smooth(rho, win),
        theta_env=_smooth(th, win),
        alpha_env=_smooth(al, win),
        delta_env=_smooth(de_np, win) if de_np is not None else None,
        wave_mean=_smooth(wave, win),
        epoch_sec=float(epoch_sec),
        clinical_group=clinical_group,
        subject_id=subject_id,
    )


def eeg_temporal_chain_features(obs: EEGTemporalObservables) -> dict[str, float]:
    """Shape features in epoch tau frame + θ→α propagation scalars."""
    rho_tau = _resample_epoch_tau(obs.rho, obs.epoch_sec)
    th_tau = _resample_epoch_tau(obs.theta_env, obs.epoch_sec)
    al_tau = _resample_epoch_tau(obs.alpha_env, obs.epoch_sec)

    rho_shape = rho_tau / max(float(np.abs(rho_tau).max()), 1e-12)
    ta_ratio = np.log10((th_tau + 1e-8) / (al_tau + 1e-8))
    ta_shape = ta_ratio - float(ta_ratio.mean())

    lag_norm = _crosscorr_lag(obs.theta_env, obs.alpha_env, obs.epoch_sec)
    mid = TAU_N // 2
    rho_early = float(rho_tau[:mid].mean())
    rho_late = float(rho_tau[mid:].mean())
    rho_drift = rho_late - rho_early

    tau_grid = np.linspace(TAU_LO, TAU_HI, TAU_N)
    alpha_peak_tau = float(tau_grid[int(np.argmax(al_tau))])
    theta_peak_tau = float(tau_grid[int(np.argmax(th_tau))])

    late_mask = tau_grid >= 0.65
    rho_late_decay = 0.0
    if late_mask.sum() >= 3:
        y = np.log10(rho_tau[late_mask] + 1e-8)
        x = tau_grid[late_mask]
        rho_late_decay = float(np.polyfit(x, y, 1)[0])

    # coupling: correlation of theta/alpha trajectories in tau frame
    if np.std(th_tau) > 1e-9 and np.std(al_tau) > 1e-9:
        coupling = float(np.corrcoef(th_tau, al_tau)[0, 1])
    else:
        coupling = 0.0

    feat: dict[str, float] = {}
    for i, v in enumerate(rho_shape):
        feat[f"rho_tau_{i}"] = float(v)
    for i, v in enumerate(ta_shape):
        feat[f"ta_ratio_tau_{i}"] = float(v)
    feat["theta_alpha_lag_norm"] = lag_norm
    feat["rho_early_late_drift"] = rho_drift
    feat["alpha_peak_tau"] = alpha_peak_tau
    feat["theta_peak_tau"] = theta_peak_tau
    feat["rho_late_decay"] = rho_late_decay
    feat["theta_alpha_coupling"] = coupling
    return feat


def features_to_vector(feat: dict[str, float]) -> np.ndarray:
    vec = np.asarray([float(feat.get(n, 0.0)) for n in EEG_CHAIN_SHAPE_NAMES], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def mean_trajectory(obs_list: list[EEGTemporalObservables]) -> dict[str, np.ndarray]:
    """Average rho / theta / alpha trajectories in epoch tau frame."""
    rho, th, al = [], [], []
    for obs in obs_list:
        rho.append(_resample_epoch_tau(obs.rho, obs.epoch_sec))
        th.append(_resample_epoch_tau(obs.theta_env, obs.epoch_sec))
        al.append(_resample_epoch_tau(obs.alpha_env, obs.epoch_sec))
    return {
        "rho_tau": np.mean(np.stack(rho, axis=0), axis=0),
        "theta_tau": np.mean(np.stack(th, axis=0), axis=0),
        "alpha_tau": np.mean(np.stack(al, axis=0), axis=0),
        "tau_grid": np.linspace(TAU_LO, TAU_HI, TAU_N),
    }
