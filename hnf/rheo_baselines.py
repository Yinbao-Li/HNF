# -*- coding: utf-8 -*-
"""Baselines for rheology stress prediction (anisotropic Boltzmann setting).

Compared against PronyBoltzmannKernel (PNF aniso memory):
  - IsotropicPronyBaseline: shared scalar G_k (misspecified under anisotropy)
  - DiagonalPronyBaseline: independent per-channel Maxwell (no cross-coupling)
  - LSTMStressBaseline: black-box sequence model (deep-learning SOTA proxy)
  - TCNStressBaseline: temporal convolutional network
  - LinearFIRBaseline: causal FIR / finite memory linear map
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.rheo_memory import PronyBoltzmannKernel


class IsotropicPronyBaseline(nn.Module):
    """Scalar Prony applied channel-wise with shared G_k (no A_k anisotropy)."""

    def __init__(self, n_modes: int = 2, dim: int = 2, **kwargs):
        super().__init__()
        self.dim = dim
        self.kernel = PronyBoltzmannKernel(
            n_modes=n_modes, dim=1, anisotropic=False, **kwargs
        )

    def forward(self, gammadot: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        # gammadot: (B,T,d) → apply same scalar kernel per channel
        if gammadot.dim() == 2:
            return self.kernel(gammadot, dt)
        outs = [self.kernel(gammadot[..., c], dt) for c in range(gammadot.size(-1))]
        return torch.stack(outs, dim=-1)

    def collect_params(self) -> dict:
        p = self.kernel.collect_params()
        p["baseline"] = "isotropic_prony"
        return p


class DiagonalPronyBaseline(nn.Module):
    """Per-channel independent Prony (diagonal anisotropy, literature default)."""

    def __init__(self, n_modes: int = 2, dim: int = 2, **kwargs):
        super().__init__()
        self.dim = dim
        self.channels = nn.ModuleList(
            [
                PronyBoltzmannKernel(n_modes=n_modes, dim=1, anisotropic=False, **kwargs)
                for _ in range(dim)
            ]
        )

    def forward(self, gammadot: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        if gammadot.dim() == 2:
            return self.channels[0](gammadot, dt)
        outs = [self.channels[c](gammadot[..., c], dt) for c in range(self.dim)]
        return torch.stack(outs, dim=-1)

    def collect_params(self) -> dict:
        return {
            "baseline": "diagonal_prony",
            "channels": [c.collect_params() for c in self.channels],
        }


class LSTMStressBaseline(nn.Module):
    """Bidirectional-capable LSTM mapping γ̇ history → σ (black-box SOTA proxy)."""

    def __init__(
        self,
        dim: int = 2,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, gammadot: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        if gammadot.dim() == 2:
            gammadot = gammadot.unsqueeze(-1)
        h, _ = self.lstm(gammadot)
        return self.head(h)

    def collect_params(self) -> dict:
        n = sum(p.numel() for p in self.parameters())
        return {"baseline": "lstm", "n_params": float(n)}


class TCNStressBaseline(nn.Module):
    """Causal dilated TCN (WaveNet-style) for hereditary response."""

    def __init__(
        self,
        dim: int = 2,
        channels: int = 48,
        n_blocks: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.dim = dim
        layers = []
        in_ch = dim
        dilation = 1
        for _ in range(n_blocks):
            pad = (kernel_size - 1) * dilation
            layers.append(
                _CausalConvBlock(in_ch, channels, kernel_size=kernel_size, dilation=dilation, pad=pad)
            )
            in_ch = channels
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, dim, kernel_size=1)

    def forward(self, gammadot: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        if gammadot.dim() == 2:
            gammadot = gammadot.unsqueeze(-1)
        # (B,T,d) → (B,d,T)
        x = gammadot.transpose(1, 2)
        y = self.head(self.net(x)).transpose(1, 2)
        return y

    def collect_params(self) -> dict:
        n = sum(p.numel() for p in self.parameters())
        return {"baseline": "tcn", "n_params": float(n)}


class LinearFIRBaseline(nn.Module):
    """Finite causal FIR memory (classical linear filter bank)."""

    def __init__(self, dim: int = 2, memory: int = 64):
        super().__init__()
        self.dim = dim
        self.memory = memory
        # Per output channel: mix of delayed input channels
        self.weight = nn.Parameter(0.01 * torch.randn(dim, dim, memory))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, gammadot: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        if gammadot.dim() == 2:
            gammadot = gammadot.unsqueeze(-1)
        # causal conv: pad left
        x = gammadot.transpose(1, 2)  # (B,d,T)
        x_pad = F.pad(x, (self.memory - 1, 0))
        y = F.conv1d(x_pad, self.weight)  # (B,d,T)
        return y.transpose(1, 2) + self.bias

    def collect_params(self) -> dict:
        return {"baseline": "linear_fir", "memory": float(self.memory)}


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, pad: int):
        super().__init__()
        self.pad = pad
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (self.pad, 0))
        y = self.act(self.norm(self.conv(y)))
        return y + self.res(x)


def build_baseline(name: str, *, dim: int = 2, n_modes: int = 2, **kwargs) -> nn.Module:
    name = name.lower()
    if name in {"isotropic", "isotropic_prony"}:
        return IsotropicPronyBaseline(n_modes=n_modes, dim=dim)
    if name in {"diagonal", "diagonal_prony"}:
        return DiagonalPronyBaseline(n_modes=n_modes, dim=dim)
    if name == "lstm":
        return LSTMStressBaseline(dim=dim, hidden=int(kwargs.get("hidden", 64)))
    if name == "tcn":
        return TCNStressBaseline(dim=dim, channels=int(kwargs.get("channels", 48)))
    if name in {"fir", "linear_fir"}:
        return LinearFIRBaseline(dim=dim, memory=int(kwargs.get("memory", 64)))
    raise ValueError(f"Unknown baseline: {name}")


__all__ = [
    "IsotropicPronyBaseline",
    "DiagonalPronyBaseline",
    "LSTMStressBaseline",
    "TCNStressBaseline",
    "LinearFIRBaseline",
    "build_baseline",
]
