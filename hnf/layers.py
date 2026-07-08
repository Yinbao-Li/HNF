# -*- coding: utf-8 -*-
"""Part 2: HuygensWaveLayer and HuygensAttention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.kernel import HuygensKernel


class HuygensWaveLayer(nn.Module):
    """惠更斯波传播层: X_out = K @ X_in"""

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        gamma: float = 1.0,
        omega: float = 1.0,
        causal: bool = True,
        wave_speed: float = 1.0,
        use_projection: bool = True,
        learnable_kernel_params: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features

        self.kernel = HuygensKernel(
            gamma=gamma,
            omega=omega,
            causal=causal,
            wave_speed=wave_speed,
            learnable_gamma=learnable_kernel_params,
            learnable_omega=learnable_kernel_params,
        )

        self.proj = (
            nn.Linear(in_features, self.out_features, bias=use_bias)
            if use_projection
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(self.out_features)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = self.kernel(x, t, rho)
        x_complex = x.to(torch.complex64)
        out = torch.matmul(k, x_complex).real
        out = self.proj(out)
        out = self.norm(out)
        return out


class HuygensAttention(nn.Module):
    """惠更斯启发的注意力机制 (H-Attention)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        gamma: float = 0.5,
        omega: float = 0.3,
        causal: bool = True,
        wave_speed: float = 0.5,
        dropout: float = 0.0,
        use_fmm: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_fmm = use_fmm

        self.kernel = HuygensKernel(
            gamma=gamma,
            omega=omega,
            causal=causal,
            wave_speed=wave_speed,
        )

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        k = self.kernel(x, t, rho)
        k = torch.abs(k)
        k = F.softmax(k, dim=-1)
        k = self.dropout(k)

        v = self.v_proj(x)
        k = k.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        out = torch.matmul(k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)
        return out


class HuygensWaveBlock(nn.Module):
    """
    惠更斯次波传播块: u_out = u + proj(Re(K @ u_complex))
    使用物理时间距离、复相位传播与可选非均匀介质 rho。
    """

    def __init__(
        self,
        dim: int,
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 6.0,
        causal: bool = True,
        distance_mode: str = "time",
        local_window_sec: Optional[float] = 15.0,
        learnable_kernel_params: bool = True,
        dropout: float = 0.0,
        sparse_band: bool = False,
    ):
        super().__init__()
        self.kernel = HuygensKernel(
            gamma=gamma,
            omega=omega,
            causal=causal,
            wave_speed=wave_speed,
            learnable_gamma=learnable_kernel_params,
            learnable_omega=learnable_kernel_params,
            learnable_wave_speed=learnable_kernel_params,
            distance_mode=distance_mode,
            local_window_sec=local_window_sec,
            sparse_band=sparse_band,
        )
        self.proj_real = nn.Linear(dim, dim, bias=False)
        self.proj_imag = nn.Linear(dim, dim, bias=False)
        self.norm_real = nn.LayerNorm(dim)
        self.norm_imag = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_c = torch.complex(h_real, h_imag)
        out_c = self.kernel.forward_apply(h_c, h_real, t=t, rho=rho)
        out_real = self.dropout(self.proj_real(out_c.real))
        out_imag = self.dropout(self.proj_imag(out_c.imag))
        h_real = self.norm_real(h_real + out_real)
        h_imag = self.norm_imag(h_imag + out_imag)
        return h_real, h_imag
