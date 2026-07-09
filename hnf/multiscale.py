# -*- coding: utf-8 -*-
"""Multi-scale DeepHuygens encoder for STEAD picking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.layers import HuygensWaveBlock


@dataclass(frozen=True)
class ScaleSpec:
    """One temporal scale in the multi-scale pyramid."""

    downsample: int = 1
    dim: int = 64
    local_window_sec: float = 15.0
    omega_scale: float = 1.0
    gamma_scale: float = 1.0
    num_layers: int = 3


def default_scale_specs(
    embed_dim: int = 64,
    local_window_sec: float = 15.0,
) -> list[ScaleSpec]:
    """Fine (P onset) + coarse (S envelope) scales aligned with run11 design."""
    return [
        ScaleSpec(
            downsample=1,
            dim=embed_dim,
            local_window_sec=min(8.0, local_window_sec),
            omega_scale=1.2,
            gamma_scale=1.1,
            num_layers=3,
        ),
        ScaleSpec(
            downsample=2,
            dim=max(32, embed_dim // 2),
            local_window_sec=max(20.0, local_window_sec),
            omega_scale=0.6,
            gamma_scale=0.9,
            num_layers=3,
        ),
    ]


class DeepHuygensStack(nn.Module):
    """Stack HuygensWaveBlocks with residual complex propagation."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        gamma: float,
        omega: float,
        wave_speed: float,
        local_window_sec: float,
        dropout: float,
        gamma_decay: float = 0.95,
        omega_growth: float = 1.05,
        sparse_band: bool = False,
        principle: str = "huygens",
        obliquity_scale: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HuygensWaveBlock(
                    dim=dim,
                    gamma=gamma * (gamma_decay ** i),
                    omega=omega * (omega_growth ** i),
                    wave_speed=wave_speed,
                    distance_mode="time",
                    local_window_sec=local_window_sec,
                    learnable_kernel_params=True,
                    dropout=dropout,
                    sparse_band=sparse_band,
                    principle=principle,
                    obliquity_scale=obliquity_scale,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_real, h_imag = layer(h_real, h_imag, t=t, rho=rho)
        return h_real, h_imag


class MultiScaleHuygensEncoder(nn.Module):
    """
    Parallel DeepHuygens stacks at multiple temporal scales, fused to full resolution.

    Each scale downsamples the shared wavefield, propagates with its own window/omega,
    then upsamples back before feature fusion.
    """

    def __init__(
        self,
        embed_dim: int,
        scale_specs: list[ScaleSpec],
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 6.0,
        dropout: float = 0.1,
        sparse_band: bool = False,
        principle: str = "huygens",
        obliquity_scale: float = 1.0,
    ):
        super().__init__()
        if not scale_specs:
            raise ValueError("scale_specs must not be empty")

        self.embed_dim = embed_dim
        self.scale_specs = list(scale_specs)
        self.branches = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        self.up_projs = nn.ModuleList()

        for spec in self.scale_specs:
            in_proj = (
                nn.Identity()
                if spec.dim == embed_dim
                else nn.Sequential(
                    nn.Linear(embed_dim, spec.dim, bias=False),
                    nn.LayerNorm(spec.dim),
                )
            )
            out_proj = (
                nn.Identity()
                if spec.dim == embed_dim
                else nn.Sequential(
                    nn.Linear(spec.dim, embed_dim, bias=False),
                    nn.LayerNorm(embed_dim),
                )
            )
            self.down_projs.append(in_proj)
            self.up_projs.append(out_proj)
            self.branches.append(
                DeepHuygensStack(
                    dim=spec.dim,
                    num_layers=spec.num_layers,
                    gamma=gamma * spec.gamma_scale,
                    omega=omega * spec.omega_scale,
                    wave_speed=wave_speed,
                    local_window_sec=spec.local_window_sec,
                    dropout=dropout,
                    sparse_band=sparse_band,
                    principle=principle,
                    obliquity_scale=obliquity_scale,
                )
            )

        fused_dim = embed_dim * len(self.scale_specs) * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, embed_dim * 2, bias=False),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _resample_time(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_real.size(1) == target_len:
            return h_real, h_imag, t, rho

        b, _, d = h_real.shape
        h_real_rs = F.interpolate(
            h_real.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        h_imag_rs = F.interpolate(
            h_imag.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        t_rs = F.interpolate(
            t.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        rho_rs = F.interpolate(
            rho.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return h_real_rs, h_imag_rs, t_rs, rho_rs

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_len = h_real.size(1)
        fused_parts: list[torch.Tensor] = []

        for spec, down_proj, up_proj, branch in zip(
            self.scale_specs,
            self.down_projs,
            self.up_projs,
            self.branches,
        ):
            target_len = max(2, full_len // spec.downsample)
            hr, hi, t_s, rho_s = self._resample_time(h_real, h_imag, t, rho, target_len)
            hr = down_proj(hr)
            hi = down_proj(hi)
            hr, hi = branch(hr, hi, t=t_s, rho=rho_s)
            hr = up_proj(hr)
            hi = up_proj(hi)
            if hr.size(1) != full_len:
                hr, hi, _, _ = self._resample_time(hr, hi, t, rho, full_len)
            fused_parts.append(hr)
            fused_parts.append(hi)

        fused = self.fusion(torch.cat(fused_parts, dim=-1))
        h_real, h_imag = fused.chunk(2, dim=-1)
        return h_real, h_imag
