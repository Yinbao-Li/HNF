# -*- coding: utf-8 -*-
"""Learnable temporal sampler: importance-weighted soft downsample to fixed length.

Predicts a density w(t) on a fine grid, then draws ``out_len`` points via a soft
inverse-CDF so high-mass regions claim more of the budget. Output time axis is a
uniform linspace (warped content coordinates) so sparse_band kernels stay valid.

Training signal: ``sampler_alignment_loss`` pushes mass onto P/S targets / energy.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTemporalSampler(nn.Module):
    """Content-aware soft resampling: (B, T_in, C) -> (B, out_len, C)."""

    def __init__(
        self,
        channels: int = 3,
        hidden: int = 32,
        out_len: int = 800,
        temperature: float = 0.05,
        duration_sec: float = 60.0,
    ):
        super().__init__()
        self.out_len = int(out_len)
        self.temperature = float(max(1e-4, temperature))
        self.duration_sec = float(duration_sec)
        self.score_net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def importance(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized density w on the input grid. x: (B, T, C) -> (B, T)."""
        score = self.score_net(x.transpose(1, 2)).squeeze(1)
        w = F.softplus(score) + 1e-4
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
          x: (B, out_len, C) resampled waveform
          t: (B, out_len, 1) uniform warped-time axis in [0, duration]
          w: (B, T_in) importance density
          attn: (B, T_in, out_len) soft assignment used for sampling / label remap
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B,T,C), got {tuple(x.shape)}")
        b, t_in, _c = x.shape
        w = self.importance(x)
        cdf = torch.cumsum(w, dim=-1)
        # Equal-mass query levels in (0,1).
        q = (
            torch.arange(self.out_len, device=x.device, dtype=x.dtype) + 0.5
        ) / float(self.out_len)
        # Soft inverse-CDF: concentrate attn near cdf^{-1}(q).
        dist = (cdf.unsqueeze(-1) - q.view(1, 1, -1)).abs()
        attn = F.softmax(-dist / self.temperature, dim=1)  # (B, T_in, out_len)
        x_out = torch.einsum("bti,btc->bic", attn, x)
        t_axis = torch.linspace(
            0.0, self.duration_sec, self.out_len, device=x.device, dtype=x.dtype
        )
        t_out = t_axis.view(1, self.out_len, 1).expand(b, -1, 1)
        return {"x": x_out, "t": t_out, "w": w, "attn": attn, "cdf": cdf}


def remap_sequence(attn: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
    """Map fine-grid sequence (B, T_in) -> coarse (B, out_len) with sampler attn."""
    return torch.einsum("bti,bt->bi", attn, fine)


def remap_index(attn: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Map discrete fine indices (B,) to coarse argmax bins; invalid (<0) kept."""
    b, t_in, out_len = attn.shape
    out = idx.new_full((b,), -1)
    valid = idx >= 0
    if not valid.any():
        return out
    # One-hot gather columns of attn at GT index, then argmax over out_len.
    ii = idx[valid].clamp(0, t_in - 1)
    # attn[valid, ii, :] -> (N, out_len)
    rows = attn[valid]
    gathered = rows[torch.arange(ii.numel(), device=attn.device), ii]
    out[valid] = gathered.argmax(dim=-1)
    return out


def sampler_alignment_loss(
    w: torch.Tensor,
    p_target: torch.Tensor,
    s_target: torch.Tensor,
    det: torch.Tensor,
    *,
    entropy_weight: float = 0.02,
    energy_x: Optional[torch.Tensor] = None,
    energy_weight: float = 0.05,
) -> torch.Tensor:
    """
    Push importance mass onto pick labels (and optional waveform energy).

    Also a light entropy penalty so w does not collapse to a single spike.
    """
    device = w.device
    event = det > 0.5
    loss = w.new_zeros(())
    if event.any():
        pick = (p_target[event] + s_target[event]).clamp(0.0, 1.0)
        pick_mass = pick.sum(dim=-1).clamp_min(1e-4)
        covered = (w[event] * pick).sum(dim=-1) / pick_mass
        # maximize covered mass → minimize 1 - covered
        loss = loss + (1.0 - covered).mean()
    else:
        loss = loss + w.new_zeros(())

    # Entropy regularizer: prefer smoother densities (higher entropy).
    ent = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=-1)
    # Normalize by log(T) so scale ~1.
    ent = ent / float(max(1.0, math_log_len(w.size(-1))))
    loss = loss + float(entropy_weight) * (1.0 - ent).clamp_min(0.0).mean()

    if energy_x is not None and energy_weight > 0:
        # Encourage mass on high |waveform| energy without GT.
        e = energy_x.pow(2).mean(dim=-1)
        e = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        loss = loss + float(energy_weight) * (1.0 - (w * e).sum(dim=-1)).mean()
    return loss


def math_log_len(n: int) -> float:
    import math

    return math.log(max(n, 2))
