# -*- coding: utf-8 -*-
"""Pick-time decoding beyond naive global argmax.

OBS: multi-peak curves are common; later taller peaks often steal argmax.
STEAD-style global argmax is not the right default readout.

Preferred OBS path: candidate local peaks on the pick curve, then judge with
**backbone-learned causal field** (Huygens P-branch envelope), not hand-tuned
time penalties on the pick curve.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def local_peak_mask(probs: torch.Tensor) -> torch.Tensor:
    """Strict local maxima along time. probs: (T,) or (B, T) → same shape bool."""
    if probs.dim() == 1:
        left = torch.roll(probs, 1, dims=0)
        right = torch.roll(probs, -1, dims=0)
        left[0] = -1.0
        right[-1] = -1.0
        return (probs >= left) & (probs >= right)
    left = torch.roll(probs, 1, dims=-1)
    right = torch.roll(probs, -1, dims=-1)
    left[..., 0] = -1.0
    right[..., -1] = -1.0
    return (probs >= left) & (probs >= right)


def backbone_field_cause_weight(field_env: torch.Tensor) -> torch.Tensor:
    """Cause-likeness from Huygens branch field envelope (backbone physics).

    High where the causal field is *rising* and little field energy has arrived
    yet — i.e. onset / cause, not coda / effect. Derived only from the learned
    wavefield, not from pick-curve clock penalties.
    """
    env = field_env.detach().float().reshape(-1).clamp_min(0.0)
    if env.numel() == 0:
        return env
    env_n = env / (env.max() + 1e-8)
    cum = torch.cumsum(env_n, dim=0)
    cum = cum / (cum[-1] + 1e-8)
    early = (1.0 - cum).clamp(0.0, 1.0)
    d = F.pad(env_n[1:] - env_n[:-1], (1, 0))
    rise = d.clamp(min=0.0)
    rise = rise / (rise.max() + 1e-8)
    w = rise * early
    if float(w.max().item()) <= 0:
        # Flat field: fall back to early-mass only.
        return early
    return w / (w.max() + 1e-8)


def _candidate_peaks(
    probs: torch.Tensor,
    *,
    pick_th: float,
    compete_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    probs = probs.detach().float().reshape(-1)
    tlen = int(probs.numel())
    gmax, gidx = float(probs.max().item()), int(probs.argmax().item())
    if tlen < 3:
        return (
            torch.tensor([gidx], device=probs.device, dtype=torch.long),
            torch.tensor([gmax], device=probs.device, dtype=torch.float32),
            gmax,
            gidx,
        )
    peaks = local_peak_mask(probs)
    peak_idx = torch.where(peaks)[0]
    if peak_idx.numel() == 0:
        return (
            torch.tensor([gidx], device=probs.device, dtype=torch.long),
            torch.tensor([gmax], device=probs.device, dtype=torch.float32),
            gmax,
            gidx,
        )
    scores = probs[peak_idx]
    keep = scores >= float(pick_th)
    if not bool(keep.any()):
        return (
            torch.tensor([gidx], device=probs.device, dtype=torch.long),
            torch.tensor([gmax], device=probs.device, dtype=torch.float32),
            gmax,
            gidx,
        )
    peak_idx = peak_idx[keep]
    scores = scores[keep]
    return peak_idx, scores, gmax, gidx


def decode_pick_index(
    probs: torch.Tensor,
    *,
    pick_th: float = 0.25,
    mode: str = "argmax",
    compete_ratio: float = 0.70,
    late_penalty: float = 0.0,
    field_env: Optional[torch.Tensor] = None,
    # Legacy hand-prior knobs (kept for ablation; prefer backbone_causal).
    echo_gap_lo_bins: int = 20,
    echo_gap_hi_bins: int = 300,
    echo_ratio: float = 0.80,
    echo_penalty: float = 0.0,
    onset_bonus: float = 0.0,
) -> tuple[int, float]:
    """Decode one (T,) probability curve → (pred_idx, peak_score).

    modes:
      - argmax: global max (STEAD-friendly legacy)
      - earliest_competitive: earliest local peak ≥ compete_ratio * gmax
      - score_minus_late: pick-curve hand prior (clock penalty)
      - backbone_causal: candidate pick peaks × Huygens field cause-weight
      - causal_peak_rerank: legacy hand echo prior (discouraged)
    """
    probs = probs.detach().float().reshape(-1)
    tlen = int(probs.numel())
    if tlen == 0:
        return 0, 0.0

    gmax, gidx = float(probs.max().item()), int(probs.argmax().item())
    mode = str(mode or "argmax")
    if mode == "argmax" or tlen < 3:
        return gidx, gmax

    peak_idx, scores, gmax, gidx = _candidate_peaks(
        probs, pick_th=pick_th, compete_ratio=compete_ratio
    )

    if mode == "earliest_competitive":
        floor = float(compete_ratio) * gmax
        ok = scores >= floor
        if bool(ok.any()):
            peak_idx = peak_idx[ok]
            scores = scores[ok]
        return int(peak_idx[0].item()), float(scores[0].item())

    if mode == "score_minus_late":
        t_norm = peak_idx.float() / float(max(tlen - 1, 1))
        utility = scores - float(late_penalty) * t_norm
        j = int(utility.argmax().item())
        return int(peak_idx[j].item()), float(scores[j].item())

    if mode == "backbone_causal":
        if field_env is None:
            # Without field, degrade to earliest competitive (OBS-safer than argmax).
            return decode_pick_index(
                probs,
                pick_th=pick_th,
                mode="earliest_competitive",
                compete_ratio=compete_ratio,
            )
        cause = backbone_field_cause_weight(field_env)
        if cause.numel() != tlen:
            cause = F.interpolate(
                cause.view(1, 1, -1), size=tlen, mode="linear", align_corners=False
            ).view(-1)
        # Fuse: pick confidence gated by backbone cause-likeness at that bin.
        # No clock penalty — judgment comes from learned causal field.
        floor = float(compete_ratio) * gmax
        competitive = scores >= floor
        if bool(competitive.any()):
            peak_idx = peak_idx[competitive]
            scores = scores[competitive]
        cause_at = cause[peak_idx]
        utility = scores * (0.05 + cause_at)
        j = int(utility.argmax().item())
        return int(peak_idx[j].item()), float(scores[j].item())

    if mode == "causal_peak_rerank":
        # Legacy hand prior kept for old ablations only.
        t_norm = peak_idx.float() / float(max(tlen - 1, 1))
        utility = scores - float(late_penalty) * t_norm
        n = int(peak_idx.numel())
        if n >= 2 and float(echo_penalty) > 0:
            t_j = peak_idx.unsqueeze(1)
            t_i = peak_idx.unsqueeze(0)
            gaps = t_j - t_i
            later = torch.triu(
                torch.ones(n, n, dtype=torch.bool, device=peak_idx.device), diagonal=1
            )
            in_window = (gaps >= int(echo_gap_lo_bins)) & (gaps <= int(echo_gap_hi_bins))
            strong_cause = scores.unsqueeze(0) >= (float(echo_ratio) * scores.unsqueeze(1))
            hit = later & in_window & strong_cause
            if bool(hit.any()):
                utility = utility - float(echo_penalty) * hit.any(dim=1).float() * scores
        if float(onset_bonus) > 0:
            competitive = scores >= (float(compete_ratio) * gmax)
            if bool(competitive.any()):
                earliest = int(torch.where(competitive)[0][0].item())
                utility = utility.clone()
                utility[earliest] = utility[earliest] + float(onset_bonus)
        j = int(utility.argmax().item())
        return int(peak_idx[j].item()), float(scores[j].item())

    return gidx, gmax


def decode_pick_indices_batch(
    probs: torch.Tensor,
    *,
    pick_th: float = 0.25,
    mode: str = "argmax",
    compete_ratio: float = 0.70,
    late_penalty: float = 0.0,
    field_env: Optional[torch.Tensor] = None,
    echo_gap_lo_bins: int = 20,
    echo_gap_hi_bins: int = 300,
    echo_ratio: float = 0.80,
    echo_penalty: float = 0.0,
    onset_bonus: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """probs (B,T) → pred_idx (B,), peak_score (B,). field_env optional (B,T)."""
    bsz = probs.size(0)
    idxs = []
    scores = []
    for i in range(bsz):
        fe = None if field_env is None else field_env[i]
        pi, sc = decode_pick_index(
            probs[i],
            pick_th=pick_th,
            mode=mode,
            compete_ratio=compete_ratio,
            late_penalty=late_penalty,
            field_env=fe,
            echo_gap_lo_bins=echo_gap_lo_bins,
            echo_gap_hi_bins=echo_gap_hi_bins,
            echo_ratio=echo_ratio,
            echo_penalty=echo_penalty,
            onset_bonus=onset_bonus,
        )
        idxs.append(pi)
        scores.append(sc)
    device = probs.device
    return (
        torch.tensor(idxs, device=device, dtype=torch.long),
        torch.tensor(scores, device=device, dtype=torch.float32),
    )


def apply_ps_order(
    p_idx: int,
    s_idx: int,
    p_score: float,
    s_score: float,
    *,
    p_th: float,
    s_th: float,
    min_gap_bins: int = 1,
    s_probs: Optional[torch.Tensor] = None,
) -> tuple[int, int, float, float]:
    """If both exist and S is not after P, drop S (or push S to first peak after P)."""
    if p_score < p_th or s_score < s_th:
        return p_idx, s_idx, p_score, s_score
    if s_idx > p_idx + min_gap_bins:
        return p_idx, s_idx, p_score, s_score
    if s_probs is None:
        return p_idx, s_idx, p_score, 0.0
    after = s_probs.clone()
    after[: p_idx + min_gap_bins + 1] = 0.0
    if float(after.max().item()) < s_th:
        return p_idx, s_idx, p_score, 0.0
    nj, ns = decode_pick_index(after, pick_th=s_th, mode="argmax")
    return p_idx, nj, p_score, ns
