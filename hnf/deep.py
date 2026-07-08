# -*- coding: utf-8 -*-
"""Part 5: DeepHuygensKernel — deep nested Huygens propagation."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from hnf.kernel import HuygensKernel


class DeepHuygensKernel(nn.Module):
    """深度嵌套惠更斯核."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        gamma: float = 1.0,
        omega: float = 1.0,
        causal: bool = True,
        wave_speed: float = 1.0,
        use_residual: bool = True,
        num_layers: int = 2,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.layer_norm = layer_norm

        dims = [input_dim] + hidden_dims
        self.kernels = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projections = nn.ModuleList()

        for i in range(num_layers):
            self.kernels.append(
                HuygensKernel(
                    gamma=gamma * (0.5 ** i),
                    omega=omega * (1.2 ** i),
                    causal=causal,
                    wave_speed=wave_speed * (0.8 ** i),
                    learnable_gamma=True,
                    learnable_omega=True,
                )
            )
            if layer_norm:
                self.norms.append(nn.LayerNorm(dims[min(i + 1, len(dims) - 1)]))
            in_dim = dims[i]
            out_dim = dims[i + 1] if i + 1 < len(dims) else dims[-1]
            self.projections.append(nn.Linear(in_dim, out_dim, bias=False))

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for i in range(self.num_layers):
            k = self.kernels[i](h, t, rho)
            h_next = torch.matmul(k, h.to(torch.complex64)).real
            h_next = self.projections[i](h_next)
            if self.use_residual and h.shape[-1] == h_next.shape[-1]:
                h_next = h + h_next
            if self.layer_norm:
                h_next = self.norms[i](h_next)
            h_next = self.activation(h_next)
            h = h_next
        return h
