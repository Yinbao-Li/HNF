# -*- coding: utf-8 -*-
"""Learnable / linear temporal sampler: fine grid -> fixed backbone length.

``linear`` mode: fixed F.interpolate (no trainable params) — stable 6000→800 path.
``learnable`` mode: importance-weighted soft inverse-CDF; last conv zero-inited so
the start is near-uniform (≈ linear), and softmax runs in fp32 for stability.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _linear_interp_attn(
    batch: int,
    t_in: int,
    out_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build (B, T_in, out_len) linear-interpolation assignment."""
    if out_len <= 1:
        attn = torch.zeros(batch, t_in, out_len, device=device, dtype=dtype)
        attn[:, 0, :] = 1.0
        return attn
    pos = torch.linspace(0, t_in - 1, out_len, device=device, dtype=dtype)
    left = pos.floor().long().clamp(0, t_in - 1)
    right = (left + 1).clamp(max=t_in - 1)
    frac = (pos - left.to(dtype)).clamp(0.0, 1.0)
    w_left = 1.0 - frac
    w_right = frac
    same = left == right
    w_left = torch.where(same, torch.ones_like(w_left), w_left)
    w_right = torch.where(same, torch.zeros_like(w_right), w_right)

    attn = torch.zeros(batch, t_in, out_len, device=device, dtype=dtype)
    b_ix = torch.arange(batch, device=device)[:, None].expand(batch, out_len)
    o_ix = torch.arange(out_len, device=device)[None, :].expand(batch, out_len)
    l_ix = left[None, :].expand(batch, out_len)
    r_ix = right[None, :].expand(batch, out_len)
    attn[b_ix, l_ix, o_ix] = w_left[None, :].expand(batch, out_len)
    attn[b_ix, r_ix, o_ix] = attn[b_ix, r_ix, o_ix] + w_right[None, :].expand(batch, out_len)
    return attn


class LearnableTemporalSampler(nn.Module):
    """Content-aware or fixed-linear soft resampling: (B, T_in, C) -> (B, out_len, C)."""

    def __init__(
        self,
        channels: int = 3,
        hidden: int = 32,
        out_len: int = 800,
        temperature: float = 0.25,
        duration_sec: float = 60.0,
        mode: str = "learnable",
    ):
        super().__init__()
        if mode not in {"learnable", "linear"}:
            raise ValueError(f"sampler mode must be learnable|linear, got {mode!r}")
        self.out_len = int(out_len)
        self.temperature = float(max(1e-4, temperature))
        self.duration_sec = float(duration_sec)
        self.mode = mode
        self.score_net: Optional[nn.Sequential] = None
        if mode == "learnable":
            self.score_net = nn.Sequential(
                nn.Conv1d(channels, hidden, kernel_size=9, padding=4),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )
            # Near-uniform importance at init → soft inverse-CDF ≈ linear downsample.
            nn.init.zeros_(self.score_net[-1].weight)
            nn.init.zeros_(self.score_net[-1].bias)

    def importance(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized density w on the input grid. x: (B, T, C) -> (B, T)."""
        if self.score_net is None:
            b, t_in, _ = x.shape
            return torch.full((b, t_in), 1.0 / float(t_in), device=x.device, dtype=x.dtype)
        score = self.score_net(x.transpose(1, 2)).squeeze(1)
        score = score.float()
        w = F.softplus(score) + 1e-4
        return (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)).to(dtype=x.dtype)

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
        t_axis = torch.linspace(
            0.0, self.duration_sec, self.out_len, device=x.device, dtype=x.dtype
        )
        t_out = t_axis.view(1, self.out_len, 1).expand(b, -1, 1)

        if self.mode == "linear" or t_in == self.out_len:
            if t_in == self.out_len:
                x_out = x
                w = torch.full((b, t_in), 1.0 / float(t_in), device=x.device, dtype=x.dtype)
                attn = torch.eye(t_in, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, -1, -1)
            else:
                x_out = F.interpolate(
                    x.transpose(1, 2),
                    size=self.out_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
                w = torch.full((b, t_in), 1.0 / float(t_in), device=x.device, dtype=x.dtype)
                attn = _linear_interp_attn(
                    b, t_in, self.out_len, device=x.device, dtype=x.dtype
                )
            return {"x": x_out, "t": t_out, "w": w, "attn": attn, "cdf": torch.cumsum(w, dim=-1)}

        w = self.importance(x)
        cdf = torch.cumsum(w.float(), dim=-1)
        cdf = cdf / cdf[:, -1:].clamp_min(1e-8)
        q = (
            torch.arange(self.out_len, device=x.device, dtype=torch.float32) + 0.5
        ) / float(self.out_len)
        # Floor temperature by ~1/T so long grids do not collapse to one-hots / NaNs.
        temp = max(float(self.temperature), 1.0 / float(max(t_in, 1)))
        dist = (cdf.unsqueeze(-1) - q.view(1, 1, -1)).abs()
        attn = F.softmax(-dist / temp, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        # Renormalize columns after nan scrub.
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attn = attn.to(dtype=x.dtype)
        x_out = torch.einsum("bti,btc->bic", attn, x)
        if not torch.isfinite(x_out).all():
            # Last-resort fallback: fixed linear resample (keeps train step alive).
            x_out = F.interpolate(
                x.transpose(1, 2),
                size=self.out_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            attn = _linear_interp_attn(
                b, t_in, self.out_len, device=x.device, dtype=x.dtype
            )
            w = torch.full((b, t_in), 1.0 / float(t_in), device=x.device, dtype=x.dtype)
            cdf = torch.cumsum(w, dim=-1)
        return {"x": x_out, "t": t_out, "w": w, "attn": attn, "cdf": cdf}


def remap_sequence(attn: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
    """Map fine-grid sequence (B, T_in) -> coarse (B, out_len) with sampler attn."""
    return torch.einsum("bti,bt->bi", attn, fine)


def remap_index(attn: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Map discrete fine indices (B,) to coarse bins; invalid (<0) kept.

    Soft-attn path: argmax over attn[b, fine_idx, :].
    Fallback: proportional index when that fine row has no mass (common for
    ``mode=linear`` interpolation knots that only touch left/right samples).
    """
    b, t_in, out_len = attn.shape
    out = idx.new_full((b,), -1)
    valid = idx >= 0
    if not valid.any():
        return out
    ii = idx[valid].clamp(0, t_in - 1)
    rows = attn[valid]
    gathered = rows[torch.arange(ii.numel(), device=attn.device), ii]
    row_mass = gathered.sum(dim=-1)
    argmax = gathered.argmax(dim=-1)
    prop = (
        (ii.float() * float(out_len - 1) / float(max(t_in - 1, 1)))
        .round()
        .long()
        .clamp(0, out_len - 1)
    )
    out[valid] = torch.where(row_mass > 1e-8, argmax, prop)
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
