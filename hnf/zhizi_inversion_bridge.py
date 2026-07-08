# -*- coding: utf-8 -*-
"""Zhizi inversion bridge: frozen backbone + trainable Physics Head."""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn as nn

from hnf.zhizi_physics_head import (
    ZhiziPhysicsHead,
    bucket_rho_to_layers,
    pool_wavefield_features,
    stack_pooled_features,
)


def pick_times_from_logits(
    p_logits: torch.Tensor,
    s_logits: torch.Tensor,
    window_sec: float = 60.0,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Normalized P/S peak times (B, 2) in [0, 1]."""
    p = torch.sigmoid(p_logits)
    s = torch.sigmoid(s_logits)
    if p.dim() > 1:
        p_idx = p.argmax(dim=-1).float()
        s_idx = s.argmax(dim=-1).float()
    else:
        p_idx = p.argmax().float().unsqueeze(0)
        s_idx = s.argmax().float().unsqueeze(0)
    t_len = float(seq_len or p.shape[-1])
    denom = max(t_len - 1, 1.0)
    return torch.stack([p_idx / denom, s_idx / denom], dim=-1)


def features_to_head_inputs(
    feat: dict[str, torch.Tensor],
    n_layers: int,
    window_sec: float = 60.0,
    pick_times: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    wave_stats = pool_wavefield_features(feat["h_real"], feat["h_imag"])
    rho_layers = bucket_rho_to_layers(feat["rho"], n_layers, window_sec=window_sec)
    v_latent = torch.stack([feat["kernel_vp"], feat["kernel_vs"]], dim=-1)
    if pick_times is None and "p_logits" in feat:
        pick_times = pick_times_from_logits(
            feat["p_logits"], feat["s_logits"], window_sec=window_sec, seq_len=feat["rho"].shape[-1]
        )
    return wave_stats, rho_layers, v_latent, pick_times


class ZhiziInversionBridge(nn.Module):
    """
    Frozen Zhizi backbone + trainable Physics Head.

    Does not modify picking heads or shared backbone when ``freeze_backbone=True``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_layers: int = 5,
        embed_dim: int = 64,
        hidden: int = 48,
        freeze_backbone: bool = True,
        infer_seq_len: int | None = 600,
        head_mode: str = "residual",
    ):
        super().__init__()
        self.backbone = backbone
        self.physics_head = ZhiziPhysicsHead(
            embed_dim=embed_dim, n_layers=n_layers, hidden=hidden, mode=head_mode
        )
        self.n_layers = n_layers
        self.infer_seq_len = infer_seq_len
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _maybe_downsample(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.infer_seq_len is None or x.shape[1] <= self.infer_seq_len:
            return x, t
        from hnf.picking_prior import downsample_traces

        return downsample_traces(x, t, self.infer_seq_len)

    @torch.no_grad()
    def extract_station_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        include_picks: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Single station (1, T, 3)."""
        self.backbone.eval()
        x, t = self._maybe_downsample(x, t)
        return self.backbone.forward_inversion_features(x, t, include_picks=include_picks)

    def extract_event_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        include_picks: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Multi-station event x: (N, T, 3) -> pooled head inputs + rho_layers for loss.

        Processes one station at a time to limit memory.
        """
        wave_list, rho_list, v_list, pick_list = [], [], [], []
        for i in range(x.shape[0]):
            xi = x[i : i + 1]
            feat = self.extract_station_features(xi, t, include_picks=include_picks)
            ws, rl, vl, pt = features_to_head_inputs(feat, self.n_layers)
            wave_list.append(ws[0])
            rho_list.append(rl[0])
            v_list.append(vl[0])
            if pt is not None:
                pick_list.append(pt[0])
            del feat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        pick_times = pick_list if pick_list else None
        ws, rl, vl, pt = stack_pooled_features(wave_list, rho_list, v_list, pick_times)
        return (
            ws.unsqueeze(0),
            rl.unsqueeze(0),
            vl.unsqueeze(0),
            pt.unsqueeze(0) if pt is not None else None,
            rl.unsqueeze(0),
        )

    def forward_event(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        include_picks: bool = True,
    ):
        """x: (N, T, 3) -> PhysicsHeadOutput with batch dim 1."""
        ws, rl, vl, pt, rho_layers = self.extract_event_features(x, t, include_picks)
        out = self.physics_head(ws, rl, vl, pt)
        return out, rho_layers

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.physics_head.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
