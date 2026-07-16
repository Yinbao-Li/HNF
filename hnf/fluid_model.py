# -*- coding: utf-8 -*-
"""HNF fluid reconstructor: sparse 2D velocity → dense field (+ optional η)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hnf.multiscale import MultiScaleHuygensEncoder, ScaleSpec
from hnf.picking_model import TemporalMediumDensity


def fluid_scale_specs(embed_dim: int = 64) -> list[ScaleSpec]:
    return [
        ScaleSpec(
            downsample=1,
            dim=embed_dim,
            local_window_sec=8.0,
            omega_scale=1.0,
            gamma_scale=1.0,
            num_layers=2,
        ),
        ScaleSpec(
            downsample=2,
            dim=max(32, embed_dim // 2),
            local_window_sec=16.0,
            omega_scale=0.6,
            gamma_scale=0.9,
            num_layers=2,
        ),
    ]


class FluidHNFReconstructor(nn.Module):
    """Stage-0: (B, 3, H, W) sparse+mask → dense (B, 2, H, W), optional η.

    Spatial grid is rasterized to a length-(H*W) sequence for the multi-scale
    Huygens encoder (EEG/seismic temporal pattern reuse).
    """

    def __init__(
        self,
        h: int = 32,
        w: int = 32,
        in_channels: int = 3,
        out_channels: int = 2,
        embed_dim: int = 64,
        sample_rate: float = 32.0,
        dropout: float = 0.1,
        principle: str = "huygens_fresnel",
        predict_eta: bool = True,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.seq_len = h * w
        self.sample_rate = float(sample_rate)
        self.predict_eta = bool(predict_eta)
        self.embed_dim = embed_dim

        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.medium_net = TemporalMediumDensity(channels=embed_dim, hidden=32)
        self.encoder = MultiScaleHuygensEncoder(
            embed_dim=embed_dim,
            scale_specs=fluid_scale_specs(embed_dim),
            gamma=0.5,
            omega=0.3,
            wave_speed=1.0,
            dropout=dropout,
            sparse_band=False,
            principle=principle,
        )
        self.decode = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, out_channels),
        )
        if self.predict_eta:
            self.eta_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Softplus(),
            )

    def _time_axis(self, batch: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(length, device=device, dtype=dtype) / self.sample_rate
        return t.view(1, length, 1).expand(batch, -1, -1)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: (B, 3, H, W)
        b, _, h, w = x.shape
        if h != self.h or w != self.w:
            x = nn.functional.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=False)
            h, w = self.h, self.w
        feat = self.patch(x)  # (B, D, H, W)
        seq = feat.flatten(2).transpose(1, 2)  # (B, HW, D)
        rho = self.medium_net(seq)
        h_imag = torch.zeros_like(seq)
        t = self._time_axis(b, seq.size(1), seq.device, seq.dtype)
        h_real, h_imag = self.encoder(seq, h_imag, t=t, rho=rho)
        env = torch.sqrt(h_real.pow(2) + h_imag.pow(2) + 1e-8)
        dense = self.decode(env).transpose(1, 2).reshape(b, -1, h, w)  # (B, 2, H, W)
        aux: dict[str, torch.Tensor] = {"rho": rho, "env": env}
        if self.predict_eta:
            pooled = env.mean(dim=1)
            aux["eta"] = self.eta_head(pooled).squeeze(-1)
        if return_aux:
            return dense, aux
        return dense

    def collect_kernel_params(self) -> dict[str, dict[str, float]]:
        params: dict[str, dict[str, float]] = {}
        for si, branch in enumerate(self.encoder.branches):
            for li, layer in enumerate(branch.layers):
                k = layer.kernel
                params[f"scale{si}_layer{li}"] = {
                    "gamma": float(k.effective_gamma().detach().cpu()),
                    "omega": float(k.effective_omega().detach().cpu()),
                    "wave_speed": float(k.effective_wave_speed().detach().cpu()),
                }
        return params
