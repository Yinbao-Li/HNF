# -*- coding: utf-8 -*-
"""Part 4: DensityNet — adaptive medium density rho(x)."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class DensityNet(nn.Module):
    """介质密度预测网络 rho(x)."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] | None = None,
        output_dim: int = 1,
        activation: str = "relu",
        use_fourier: bool = False,
        num_frequencies: int = 8,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        self.use_fourier = use_fourier

        if use_fourier:
            self.register_buffer(
                "freqs",
                2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies) * np.pi,
            )
            actual_input_dim = input_dim * 2 * num_frequencies
        else:
            actual_input_dim = input_dim

        layers: list[nn.Module] = []
        prev_dim = actual_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "swish":
                layers.append(nn.SiLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            x_freq = x.unsqueeze(-1) * self.freqs
            x_freq = x_freq.view(x.shape[0], x.shape[1], -1)
            x_encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
            rho = self.mlp(x_encoded)
        else:
            rho = self.mlp(x)
        return rho.squeeze(-1)
