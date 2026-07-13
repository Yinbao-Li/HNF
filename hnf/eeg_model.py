# -*- coding: utf-8 -*-
"""EEG classifier built on multi-scale Huygens Neural Field blocks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hnf.multiscale import MultiScaleHuygensEncoder, ScaleSpec
from hnf.picking_model import TemporalMediumDensity


def eeg_scale_specs(embed_dim: int = 64) -> list[ScaleSpec]:
    """Paper Domain-II scales: fine 8s + coarse 20s."""
    return [
        ScaleSpec(
            downsample=1,
            dim=embed_dim,
            local_window_sec=8.0,
            omega_scale=1.2,
            gamma_scale=1.1,
            num_layers=3,
        ),
        ScaleSpec(
            downsample=2,
            dim=max(32, embed_dim // 2),
            local_window_sec=20.0,
            omega_scale=0.6,
            gamma_scale=0.9,
            num_layers=3,
        ),
    ]


class ChannelEmbed1x1(nn.Module):
    """1x1 Conv channel lift: C_in → embed_dim (EEG analogue of ComponentSecondarySources)."""

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, embed_dim)
        return self.proj(x).transpose(1, 2)


class EEGHNFClassifier(nn.Module):
    """HNF EEG classifier for HC / MCI / AD (3-way).

    Pipeline
    --------
    (B, 19, 1280)
      → 1x1 Conv embed (B, T, 64)
      → TemporalMediumDensity → rho (B, T, 1)
      → MultiScaleHuygensEncoder (principle=huygens_fresnel)
      → GAP over time
      → MLP → 3 logits
    """

    def __init__(
        self,
        n_channels: int = 19,
        seq_len: int = 1280,
        sample_rate: int = 128,
        embed_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
        multi_scale: bool = True,
        principle: str = "huygens_fresnel",
        obliquity_scale: float = 1.0,
        sparse_band: bool = False,
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 1.0,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        if not multi_scale:
            raise ValueError("EEGHNFClassifier currently requires multi_scale=True")
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.channel_embed = ChannelEmbed1x1(n_channels, embed_dim)
        self.medium_net = TemporalMediumDensity(channels=embed_dim, hidden=32)
        self.encoder = MultiScaleHuygensEncoder(
            embed_dim=embed_dim,
            scale_specs=eeg_scale_specs(embed_dim),
            gamma=gamma,
            omega=omega,
            wave_speed=wave_speed,
            dropout=dropout,
            sparse_band=sparse_band,
            principle=principle,
            obliquity_scale=obliquity_scale,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def _time_axis(self, batch: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(length, device=device, dtype=dtype) / float(self.sample_rate)
        return t.view(1, length, 1).expand(batch, -1, -1)

    def encode(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Encode waveforms to pooled features.

        Args:
            x: ``(B, C, T)`` or ``(B, T, C)`` float tensor.
            return_aux: If True, also return rho / wavefield tensors.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        if x.size(1) == self.n_channels:
            x_bt = x  # (B, C, T)
        elif x.size(2) == self.n_channels:
            x_bt = x.transpose(1, 2)
        else:
            raise ValueError(
                f"Cannot infer channel axis for shape {tuple(x.shape)} "
                f"with n_channels={self.n_channels}"
            )

        h = self.channel_embed(x_bt)  # (B, T, D)
        rho = self.medium_net(h)  # (B, T, 1)
        h_imag = torch.zeros_like(h)
        t = self._time_axis(h.size(0), h.size(1), h.device, h.dtype)
        h_real, h_imag = self.encoder(h, h_imag, t=t, rho=rho)
        # Envelope energy as classification features
        feat_t = torch.sqrt(h_real.pow(2) + h_imag.pow(2) + 1e-8)
        pooled = feat_t.mean(dim=1)
        if not return_aux:
            return pooled, None
        aux = {
            "rho": rho,
            "h_real": h_real,
            "h_imag": h_imag,
            "feat_t": feat_t,
        }
        return pooled, aux

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled, aux = self.encode(x, return_aux=return_aux)
        logits = self.head(pooled)
        if return_aux:
            assert aux is not None
            return logits, aux
        return logits

    def collect_kernel_params(self) -> dict[str, dict[str, float]]:
        """Export learned Huygens γ / ω / c for interpretability plots."""
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

    def freeze_huygens_kernels(self) -> int:
        """Freeze γ / ω / c (and related kernel tensors). Returns #frozen params."""
        n = 0
        for name, p in self.named_parameters():
            if ".kernel." in name or name.endswith(".kernel"):
                p.requires_grad = False
                n += p.numel()
        return n

    def load_seismic_encoder(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict_shapes: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Copy matching ``multi_scale_encoder.*`` weights from a picking checkpoint.

        Returns:
            (loaded_keys, skipped_keys)
        """
        dst = self.state_dict()
        loaded: list[str] = []
        skipped: list[str] = []
        remap_prefix = "multi_scale_encoder."
        for src_key, tensor in state_dict.items():
            if not src_key.startswith(remap_prefix):
                continue
            dst_key = "encoder." + src_key[len(remap_prefix) :]
            if dst_key not in dst:
                skipped.append(src_key)
                continue
            if strict_shapes and tuple(tensor.shape) != tuple(dst[dst_key].shape):
                skipped.append(src_key)
                continue
            dst[dst_key] = tensor
            loaded.append(dst_key)
        self.load_state_dict(dst, strict=False)
        return loaded, skipped
