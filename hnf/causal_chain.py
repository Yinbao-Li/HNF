# -*- coding: utf-8 -*-
"""Causal-chain characterization for HNF picking traces.

The pattern library (``hnf/pattern_library.py``) clusters *scalar summaries* of a
forward pass (det, P-peak, P-S gap, a few rho moments). That mostly rediscovers
distance and SNR, which any onset picker gives you. It does not touch the causal
chain the Huygens field actually computes.

This module extracts the chain itself as time-resolved observables and clusters
its *shape* rather than its scalar footprint:

1. Observables per trace: rho(t) (medium response), P/S field envelopes (from the
   Huygens branches, before the pick heads), and the pick probabilities.
2. Causal reference frame: put the origin at the P onset and normalise time by the
   P->S gap, so P sits at tau=0 and S at tau=1 regardless of hypocentral distance.
   Two events at 50 km and 300 km with the *same physics* now overlap in tau.
3. Shape features in that frame: rho trajectory, S/P envelope-ratio curve, coda
   decay slope, secondary rho peaks (multipathing), onset sharpness, S/P amplitude.
   Distance (the gap) and absolute amplitude are deliberately *excluded* from the
   clustering vector, so modes that survive are mechanism modes, not distance bins.

The physical reading of each observable:
    ps_gap        -> hypocentral distance (via Vp/Vs)
    coda_slope    -> attenuation / scattering Q of the path
    n_rho_peaks   -> multipathing, reflectors, converted phases
    sp_amp_ratio  -> radiation pattern -> focal-mechanism hint
    onset_sharp   -> distance + source complexity (impulsive vs emergent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# tau grid for the causal frame: a little before P (pre-event noise), the P->S
# segment [0, 1], and coda out to two more gaps past S.
TAU_LO = -0.3
TAU_HI = 3.0
TAU_N = 48

CAUSAL_SHAPE_NAMES: tuple[str, ...] = tuple(
    [f"rho_tau_{i}" for i in range(TAU_N)]
    + [f"spr_tau_{i}" for i in range(TAU_N)]
    + [
        "coda_slope",
        "n_rho_peaks",
        "onset_sharp",
        "log_sp_amp_ratio",
        "pre_p_energy_frac",
    ]
)


@dataclass
class CausalObservables:
    """Time-resolved chain for one trace (all length-T numpy arrays)."""

    rho: np.ndarray
    p_env: np.ndarray
    s_env: np.ndarray
    p_prob: np.ndarray
    s_prob: np.ndarray
    wave_env: np.ndarray  # observed-waveform energy envelope (physical seismogram)
    det: float
    p_sec: float
    s_sec: float
    ps_gap_sec: float
    window_sec: float
    is_event: bool


def _smooth(series: np.ndarray, win: int) -> np.ndarray:
    """Odd-window moving average; kills single-sample spikes in rho/energy."""
    win = max(1, int(win) | 1)  # force odd
    if win <= 1 or series.size < win:
        return series
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(series, kernel, mode="same")


def _peak_sec(prob: np.ndarray, window_sec: float, thr: float) -> tuple[float, float]:
    peak = float(prob.max())
    if peak < thr:
        return peak, -1.0
    idx = int(prob.argmax())
    return peak, float(idx) / max(prob.size - 1, 1) * window_sec


@torch.no_grad()
def extract_causal_observables(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    window_sec: float = 60.0,
    pick_threshold: float = 0.3,
    is_event: Optional[bool] = None,
) -> CausalObservables:
    """Run a dense forward and pull the causal-chain time series."""
    t_in = t if t.dim() == 2 else t[0]
    out = model(x, t_in)

    det_p = torch.sigmoid(out["det"])
    if det_p.dim() > 1:
        det_p = det_p.amax(dim=-1)
    det_v = float(det_p.reshape(-1)[0].item())

    p_prob = torch.sigmoid(out["p"][0]).detach().float().cpu().numpy()
    s_prob = torch.sigmoid(out["s"][0]).detach().float().cpu().numpy()
    seq = x.size(1)
    smooth_win = max(3, seq // 100)
    rho = _smooth(out["rho"][0].detach().float().cpu().numpy(), smooth_win)
    p_env = _smooth(out["p_field_env"][0].detach().float().cpu().numpy(), smooth_win)
    s_env = _smooth(out["s_field_env"][0].detach().float().cpu().numpy(), smooth_win)
    # Physical seismogram energy envelope from the observed waveform (all channels).
    wave_env = _smooth(
        x[0].detach().float().cpu().pow(2).mean(dim=-1).numpy(), smooth_win
    )

    _, p_sec = _peak_sec(p_prob, window_sec, pick_threshold)
    _, s_sec = _peak_sec(s_prob, window_sec, pick_threshold)
    gap = s_sec - p_sec if (p_sec >= 0 and s_sec >= 0 and s_sec > p_sec) else -1.0

    return CausalObservables(
        rho=rho,
        p_env=p_env,
        s_env=s_env,
        p_prob=p_prob,
        s_prob=s_prob,
        wave_env=wave_env,
        det=det_v,
        p_sec=p_sec,
        s_sec=s_sec,
        ps_gap_sec=gap,
        window_sec=window_sec,
        is_event=bool(det_v > 0.5) if is_event is None else bool(is_event),
    )


def _resample_to_tau(
    series: np.ndarray, p_sec: float, gap: float, window_sec: float
) -> np.ndarray:
    """Sample a physical-time series onto the tau grid (P at 0, S at 1)."""
    seq = series.size
    phys_t = np.linspace(0.0, window_sec, seq)
    tau_grid = np.linspace(TAU_LO, TAU_HI, TAU_N)
    query_sec = p_sec + tau_grid * gap
    return np.interp(query_sec, phys_t, series, left=series[0], right=series[-1])


def _count_peaks(curve: np.ndarray, prominence: float) -> int:
    if curve.size < 3:
        return 0
    rng = float(curve.max() - curve.min())
    if rng <= 1e-9:
        return 0
    c = (curve - curve.min()) / rng
    n = 0
    for i in range(1, c.size - 1):
        if c[i] > c[i - 1] and c[i] >= c[i + 1] and c[i] > prominence:
            n += 1
    return n


def has_valid_chain(obs: CausalObservables) -> bool:
    """A causal chain needs an event with both P and S located."""
    return bool(obs.is_event and obs.p_sec >= 0 and obs.s_sec >= 0 and obs.ps_gap_sec > 0)


def _coda_decay_per_sec(obs: CausalObservables, coda_sec: float = 8.0) -> float:
    """log10(energy) slope per second over [S, S+coda_sec], clipped to the window."""
    seq = obs.wave_env.size
    phys_t = np.linspace(0.0, obs.window_sec, seq)
    lo = obs.s_sec
    hi = min(obs.window_sec, obs.s_sec + coda_sec)
    if hi - lo < 1.0:
        return 0.0
    mask = (phys_t >= lo) & (phys_t <= hi)
    if mask.sum() < 3:
        return 0.0
    y = np.log10(obs.wave_env[mask] + 1e-8)
    x = phys_t[mask]
    return float(np.polyfit(x, y, 1)[0])


def _onset_sharpness(obs: CausalObservables) -> float:
    """Normalised energy rise across the P onset on the observed waveform."""
    seq = obs.wave_env.size
    phys_t = np.linspace(0.0, obs.window_sec, seq)
    pre = (phys_t >= obs.p_sec - 0.6) & (phys_t < obs.p_sec)
    post = (phys_t >= obs.p_sec) & (phys_t <= obs.p_sec + 0.6)
    if pre.sum() < 1 or post.sum() < 1:
        return 0.0
    e_pre = float(obs.wave_env[pre].mean())
    e_post = float(obs.wave_env[post].max())
    return float((e_post - e_pre) / (e_post + 1e-8))


def causal_chain_features(obs: CausalObservables) -> dict[str, float]:
    """Distance-normalised shape descriptors of the chain.

    Raises if the trace has no valid P->S chain; callers gate with
    :func:`has_valid_chain`.
    """
    if not has_valid_chain(obs):
        raise ValueError("trace has no valid P->S causal chain")

    gap = obs.ps_gap_sec
    rho_tau = _resample_to_tau(obs.rho, obs.p_sec, gap, obs.window_sec)
    p_tau = _resample_to_tau(obs.p_env, obs.p_sec, gap, obs.window_sec)
    s_tau = _resample_to_tau(obs.s_env, obs.p_sec, gap, obs.window_sec)

    # Shape, not scale: normalise each trajectory by its own peak.
    rho_shape = rho_tau / max(float(np.abs(rho_tau).max()), 1e-12)
    total_tau = p_tau + s_tau
    spr_tau = np.log10((s_tau + 1e-8) / (p_tau + 1e-8))  # S/P envelope ratio curve

    tau_grid = np.linspace(TAU_LO, TAU_HI, TAU_N)

    # Coda decay is a *physical* quantity, so measure it on the observed waveform
    # energy (not the model field envelope) as a per-second log-decay rate over a
    # fixed window after S. Negative = decaying coda; the rate is intrinsic to the
    # path (attenuation / scattering Q), hence distance-independent.
    coda_slope = _coda_decay_per_sec(obs)

    # multipath: secondary rho peaks strictly between P and S
    between = (tau_grid > 0.05) & (tau_grid < 0.95)
    n_rho_peaks = float(_count_peaks(rho_shape[between], prominence=0.25))

    # onset sharpness on the observed waveform: normalised rise across P
    onset_sharp = _onset_sharpness(obs)

    # radiation-pattern hint: peak S vs peak P amplitude
    p_peak_amp = float(p_tau[(tau_grid >= -0.1) & (tau_grid <= 0.5)].max())
    s_peak_amp = float(s_tau[(tau_grid >= 0.6) & (tau_grid <= 1.5)].max())
    log_sp_amp_ratio = float(np.log10((s_peak_amp + 1e-8) / (p_peak_amp + 1e-8)))

    # how much observed energy sits before P (pre-event / emergent onset indicator)
    wave_tau = _resample_to_tau(obs.wave_env, obs.p_sec, gap, obs.window_sec)
    pre_mask = tau_grid < 0.0
    pre_p_energy_frac = float(
        wave_tau[pre_mask].sum() / (wave_tau.sum() + 1e-8)
    ) if pre_mask.any() else 0.0

    feat: dict[str, float] = {}
    for i, v in enumerate(rho_shape):
        feat[f"rho_tau_{i}"] = float(v)
    for i, v in enumerate(spr_tau):
        feat[f"spr_tau_{i}"] = float(v)
    feat["coda_slope"] = coda_slope
    feat["n_rho_peaks"] = n_rho_peaks
    feat["onset_sharp"] = onset_sharp
    feat["log_sp_amp_ratio"] = log_sp_amp_ratio
    feat["pre_p_energy_frac"] = pre_p_energy_frac
    return feat


def features_to_vector(feat: dict[str, float]) -> np.ndarray:
    vec = np.asarray([float(feat.get(n, 0.0)) for n in CAUSAL_SHAPE_NAMES], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def mean_trajectory(obs_list: list[CausalObservables]) -> dict[str, np.ndarray]:
    """Average rho / P-env / S-env trajectories in the causal frame (for plots)."""
    rho, pe, se = [], [], []
    for o in obs_list:
        if not has_valid_chain(o):
            continue
        rho.append(_resample_to_tau(o.rho, o.p_sec, o.ps_gap_sec, o.window_sec))
        pe.append(_resample_to_tau(o.p_env, o.p_sec, o.ps_gap_sec, o.window_sec))
        se.append(_resample_to_tau(o.s_env, o.p_sec, o.ps_gap_sec, o.window_sec))
    if not rho:
        return {}

    def _norm_stack(arrs: list[np.ndarray]) -> np.ndarray:
        m = np.stack(arrs)
        m = m / np.clip(np.abs(m).max(axis=1, keepdims=True), 1e-12, None)
        return m.mean(axis=0)

    return {
        "tau": np.linspace(TAU_LO, TAU_HI, TAU_N),
        "rho": _norm_stack(rho),
        "p_env": _norm_stack(pe),
        "s_env": _norm_stack(se),
    }


# ---------------------------------------------------------------------------
# Interpretable compact feature set for re-classification
# ---------------------------------------------------------------------------
# These are the knobs a seismologist can name. Clustering on them (not the
# 101-d trajectory) keeps modes readable and avoids overfitting magnitude.
INTERPRETABLE_NAMES: tuple[str, ...] = (
    # causal-chain shape (distance-independent)
    "coda_slope",
    "onset_sharp",
    "n_rho_peaks",
    "log_sp_amp_ratio",
    "pre_p_energy_frac",
    # source / radiation (need RAW amplitude; normalised waveforms erase these)
    "log_peak_amp",
    "log_p_rms",
    "log_s_rms",
    "log_coda_rms",
    "reduced_amp",          # log_peak_amp + log10(dist_km) — Richter-like
    # model confidence (interpretable outputs of HNF)
    "det",
    "p_peak",
    "s_peak",
)


def raw_amplitude_features(
    waveform: np.ndarray,
    *,
    p_sample: Optional[int],
    s_sample: Optional[int],
    dist_km: float,
    sample_rate: float = 100.0,
) -> dict[str, float]:
    """Peak / window RMS on the *unnormalised* STEAD waveform (T, C) or (C, T).

    Returns log10 amplitudes. ``reduced_amp = log_peak + log10(dist)`` is the
    single-station Richter proxy that the normalised training pipeline destroys.
    """
    w = np.asarray(waveform, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"expected 2-d waveform, got {w.shape}")
    if w.shape[0] < w.shape[1] and w.shape[0] <= 3:
        w = w.T  # (C, T) -> (T, C)
    tlen = w.shape[0]
    energy = np.sqrt((w ** 2).mean(axis=-1) + 1e-18)
    peak = float(np.max(np.abs(w)))
    log_peak = float(np.log10(peak + 1e-12))

    def _rms(i0: int, i1: int) -> float:
        i0 = max(0, min(tlen - 1, i0))
        i1 = max(i0 + 1, min(tlen, i1))
        return float(np.log10(energy[i0:i1].mean() + 1e-12))

    # windows in samples around picks (or proportional fallbacks)
    half = int(round(1.0 * sample_rate))
    if p_sample is None:
        p_sample = int(0.15 * tlen)
    if s_sample is None:
        s_sample = int(0.35 * tlen)
    log_p = _rms(p_sample - half // 2, p_sample + half)
    log_s = _rms(s_sample - half // 2, s_sample + half)
    coda0 = s_sample + int(1.0 * sample_rate)
    coda1 = s_sample + int(9.0 * sample_rate)
    log_coda = _rms(coda0, coda1)

    d = max(float(dist_km), 1.0) if np.isfinite(dist_km) else 1.0
    reduced = log_peak + float(np.log10(d))
    return {
        "log_peak_amp": log_peak,
        "log_p_rms": log_p,
        "log_s_rms": log_s,
        "log_coda_rms": log_coda,
        "reduced_amp": reduced,
    }


def interpretable_feature_dict(
    chain_feat: dict[str, float],
    amp_feat: dict[str, float],
    summary_feat: dict[str, float],
) -> dict[str, float]:
    """Merge chain shape + raw amplitude + model confidence into one dict."""
    out = {n: float("nan") for n in INTERPRETABLE_NAMES}
    for src in (chain_feat, amp_feat, summary_feat):
        for k, v in src.items():
            if k in out:
                out[k] = float(v)
    return out


def interpretable_to_vector(feat: dict[str, float]) -> np.ndarray:
    vec = np.asarray([float(feat.get(n, 0.0)) for n in INTERPRETABLE_NAMES], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def name_causal_mode(stats: dict[str, float], *, reduced_q33: float = 4.2, reduced_q66: float = 5.0) -> str:
    """Human-readable mode name from physical centroids.

    ``reduced_amp`` terciles default to STEAD count-unit scales seen on val;
    pass data-driven quantiles when available.
    """
    coda = stats.get("coda_slope", 0.0)
    onset = stats.get("onset_sharp", 0.0)
    peaks = stats.get("n_rho_peaks", 0.0)
    reduced = stats.get("reduced_amp", 0.0)
    gap = stats.get("ps_gap_sec", stats.get("ps_gap", 0.0))

    if peaks >= 0.7:
        shape = "multipath"
    elif onset >= 0.8 and coda <= -0.20:
        shape = "impulsive_fastQ"
    elif onset < 0.65:
        shape = "emergent"
    elif coda > -0.12:
        shape = "slow_coda"
    else:
        shape = "standard"

    if reduced >= reduced_q66:
        strength = "strong"
    elif reduced >= reduced_q33:
        strength = "mid"
    else:
        strength = "weak"

    if gap < 3.0:
        rng = "near"
    elif gap < 8.0:
        rng = "midR"
    else:
        rng = "far"

    return f"{shape}_{strength}_{rng}"
