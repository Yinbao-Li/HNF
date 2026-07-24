# -*- coding: utf-8 -*-
"""Learned causal peak ranking on Huygens backbone features.

OBS multi-peak: pick-curve argmax is unreliable. Rank local pick candidates
using crops of (pick_prob, p_field_env, rho) — i.e. backbone-learned causal
field cues — instead of hand-tuned clock penalties.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.pick_decode import local_peak_mask


def find_local_peak_candidates(
    probs: torch.Tensor,
    *,
    pick_th: float = 0.25,
    topk: int = 8,
    compete_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """probs (B,T) → peak_idx (B,K), peak_score (B,K), peak_mask (B,K).

    Peaks sorted by descending pick score within each sample; padded with 0/False.
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    bsz, tlen = probs.shape
    device = probs.device
    k = max(1, int(topk))
    peak_idx = torch.zeros(bsz, k, device=device, dtype=torch.long)
    peak_score = torch.zeros(bsz, k, device=device, dtype=torch.float32)
    peak_mask = torch.zeros(bsz, k, device=device, dtype=torch.bool)

    for b in range(bsz):
        p = probs[b]
        gmax = float(p.max().item())
        peaks = local_peak_mask(p)
        idx = torch.where(peaks)[0]
        if idx.numel() == 0:
            j = int(p.argmax().item())
            peak_idx[b, 0] = j
            peak_score[b, 0] = float(p[j].item())
            peak_mask[b, 0] = True
            continue
        sc = p[idx]
        keep = sc >= float(pick_th)
        if float(compete_ratio) > 0:
            keep = keep & (sc >= float(compete_ratio) * gmax)
        if not bool(keep.any()):
            j = int(p.argmax().item())
            peak_idx[b, 0] = j
            peak_score[b, 0] = float(p[j].item())
            peak_mask[b, 0] = True
            continue
        idx = idx[keep]
        sc = sc[keep]
        order = torch.argsort(sc, descending=True)
        idx = idx[order]
        sc = sc[order]
        n = min(k, int(idx.numel()))
        peak_idx[b, :n] = idx[:n]
        peak_score[b, :n] = sc[:n]
        peak_mask[b, :n] = True
    return peak_idx, peak_score, peak_mask


class CausalPeakRankHead(nn.Module):
    """Score each candidate peak from local backbone field crops.

    Channels per crop: pick_prob, p_field_env, rho (all normalized per-trace).
    """

    def __init__(
        self,
        crop_half_bins: int = 16,
        hidden: int = 48,
        topk: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.crop_half_bins = int(max(1, crop_half_bins))
        self.topk = int(max(1, topk))
        self.encoder = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def _norm_trace(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        mx = x.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return x / mx

    def _gather_crop(self, x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """x (B,T), center (B,K) → (B,K,L)."""
        bsz, tlen = x.shape
        k = center.size(1)
        half = self.crop_half_bins
        x_pad = F.pad(x, (half, half))  # (B, T+2h)
        idx = center.long().clamp(0, tlen - 1) + half  # (B,K)
        offs = torch.arange(-half, half + 1, device=x.device)
        gather_idx = idx.unsqueeze(-1) + offs.view(1, 1, -1)  # (B,K,L)
        batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(gather_idx)
        return x_pad[batch_idx, gather_idx]

    def forward(
        self,
        pick_probs: torch.Tensor,
        field_env: torch.Tensor,
        rho: torch.Tensor,
        peak_idx: torch.Tensor,
        peak_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return rank logits (B,K); invalid candidates → large negative."""
        pick_n = self._norm_trace(pick_probs)
        field_n = self._norm_trace(field_env)
        rho_n = self._norm_trace(rho)

        pick_c = self._gather_crop(pick_n, peak_idx)  # (B,K,L)
        field_c = self._gather_crop(field_n, peak_idx)
        rho_c = self._gather_crop(rho_n, peak_idx)
        bsz, k, length = pick_c.shape
        stacked = torch.stack([pick_c, field_c, rho_c], dim=2)  # (B,K,3,L)
        stacked = stacked.reshape(bsz * k, 3, length)
        h = self.encoder(stacked).squeeze(-1)  # (B*K, H)

        # Center scalars (already normalized traces).
        b_ix = torch.arange(bsz, device=peak_idx.device).unsqueeze(1).expand(bsz, k)
        c_ix = peak_idx.clamp(0, pick_n.size(1) - 1)
        center_feat = torch.stack(
            [
                pick_n[b_ix, c_ix],
                field_n[b_ix, c_ix],
                rho_n[b_ix, c_ix],
            ],
            dim=-1,
        ).reshape(bsz * k, 3)

        logits = self.score(torch.cat([h, center_feat], dim=-1)).view(bsz, k)
        if peak_mask is not None:
            logits = logits.masked_fill(~peak_mask, -1e4)
        return logits

    def decode(
        self,
        pick_probs: torch.Tensor,
        field_env: torch.Tensor,
        rho: torch.Tensor,
        *,
        pick_th: float = 0.25,
        compete_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """→ pred_idx (B,), peak_pick_score (B,) for gating."""
        peak_idx, peak_score, peak_mask = find_local_peak_candidates(
            pick_probs,
            pick_th=pick_th,
            topk=self.topk,
            compete_ratio=compete_ratio,
        )
        rank_logits = self.forward(pick_probs, field_env, rho, peak_idx, peak_mask)
        best = rank_logits.argmax(dim=-1)
        bsz = pick_probs.size(0)
        b_ix = torch.arange(bsz, device=pick_probs.device)
        pred = peak_idx[b_ix, best]
        gate_score = peak_score[b_ix, best]
        return pred, gate_score


def causal_peak_rank_loss(
    rank_logits: torch.Tensor,
    peak_idx: torch.Tensor,
    peak_mask: torch.Tensor,
    gt_idx: torch.Tensor,
    valid: torch.Tensor,
    *,
    radius_bins: int = 10,
) -> torch.Tensor:
    """Listwise CE: among candidates, prefer the one nearest GT (if within radius)."""
    bsz, k = rank_logits.shape
    device = rank_logits.device
    losses = []
    for b in range(bsz):
        if float(valid[b].item()) <= 0.5:
            continue
        if not bool(peak_mask[b].any()):
            continue
        gt = int(gt_idx[b].item())
        dist = (peak_idx[b].long() - gt).abs()
        dist = torch.where(peak_mask[b], dist, torch.full_like(dist, 10**9))
        best = int(dist.argmin().item())
        if int(dist[best].item()) > int(radius_bins):
            continue
        # CE over valid candidates only
        logits = rank_logits[b].clone()
        logits = torch.where(peak_mask[b], logits, torch.full_like(logits, -1e4))
        target = torch.tensor(best, device=device, dtype=torch.long)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
    if not losses:
        # Keep a grad path into the rank head so empty batches don't break AMP.
        return rank_logits.sum() * 0.0
    return torch.stack(losses).mean()
