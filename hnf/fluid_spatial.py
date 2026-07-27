# -*- coding: utf-8 -*-
"""Domain-III spatial Huygens field with optional rotational secondary sources.

Unlike ``fluid_model.FluidHNFReconstructor`` (raster H*W → 1D temporal axis), this
module propagates on a **2D grid neighbourhood** with explicit:

- **momentum sources** (symmetric / divergent part): (s_x, s_y)
- **rotational sources** (antisymmetric / vorticity DOF): ω̂_z per grid point

The rotational term uses a 2D point-vortex kernel
  v_i += K(r_ij) · ω_j · (-Δy_ij, Δx_ij) / (r_ij² + ε)
which is the natural complement to translational secondary sources for vortex-dominated
flows (Stage-0 vortex vel_rel≈0.87 on the raster baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_coords(
    h: int,
    w: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized [-1, 1]² coordinates, shape ``(2, H, W)``."""
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)


@dataclass(frozen=True)
class SpatialStencil:
    """Precomputed local neighbourhood offsets for unfold-based propagation."""

    kernel_size: int
    dx: torch.Tensor  # (K*K,)
    dy: torch.Tensor
    r: torch.Tensor
    r2: torch.Tensor
    rot_x: torch.Tensor  # (-dy)/r2 for vortex-induced vx
    rot_y: torch.Tensor  # (dx)/r2 for vortex-induced vy


def build_spatial_stencil(kernel_size: int, cell_size: float = 2.0) -> SpatialStencil:
    """Offsets for ``nn.Unfold`` ordering (row-major over K×K patch)."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    half = kernel_size // 2
    offsets = []
    for oy in range(-half, half + 1):
        for ox in range(-half, half + 1):
            offsets.append((float(ox) * cell_size, float(oy) * cell_size))
    dx = torch.tensor([o[0] for o in offsets], dtype=torch.float32)
    dy = torch.tensor([o[1] for o in offsets], dtype=torch.float32)
    r2 = dx.pow(2) + dy.pow(2)
    r = torch.sqrt(r2 + 1e-8)
    # Center pixel: avoid singularity in rotational kernel
    r2_safe = r2.clone()
    r2_safe[r2_safe < 1e-8] = 1.0
    rot_x = -dy / r2_safe
    rot_y = dx / r2_safe
    return SpatialStencil(
        kernel_size=kernel_size,
        dx=dx,
        dy=dy,
        r=r,
        r2=r2,
        rot_x=rot_x,
        rot_y=rot_y,
    )


class SpatialMediumField(nn.Module):
    """Spatial ρ(x,y) medium modulation (replaces temporal ``TemporalMediumDensity``)."""

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat) + 1e-3


class SpatialRotatingHuygensBlock(nn.Module):
    """One spatial Huygens layer: momentum + optional vorticity secondary sources."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        gamma: float = 0.5,
        learnable_gamma: bool = True,
        use_rotation: bool = True,
        cell_size: float = 2.0 / 31.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.use_rotation = bool(use_rotation)
        self.stencil = build_spatial_stencil(kernel_size, cell_size=cell_size)
        self.pad = kernel_size // 2

        if learnable_gamma:
            self.log_gamma = nn.Parameter(torch.tensor(math.log(max(gamma, 1e-3)), dtype=torch.float32))
        else:
            self.register_buffer("log_gamma", torch.tensor(math.log(max(gamma, 1e-3))))

        self.source_mom = nn.Conv2d(channels, 2, kernel_size=1)
        self.source_vort = nn.Conv2d(channels, 1, kernel_size=1)
        out_ch = 3 if use_rotation else 2
        self.mix = nn.Sequential(
            nn.Conv2d(channels + out_ch, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def effective_gamma(self) -> torch.Tensor:
        return F.softplus(self.log_gamma) + 1e-3

    def _neighbor_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        g = self.effective_gamma().to(device=device, dtype=dtype)
        r = self.stencil.r.to(device=device, dtype=dtype)
        return torch.exp(-g * r.pow(2)) / (r + 1e-3)

    def propagate(
        self,
        mom: torch.Tensor,
        vort: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Local Huygens superposition on (B, *, H, W) momentum / vorticity sources."""
        b, _, h, w = mom.shape
        k = self.kernel_size
        n = k * k
        w_nb = self._neighbor_weights(mom.device, mom.dtype).view(1, 1, n, 1)

        mom_scaled = mom * rho
        mom_p = F.unfold(mom_scaled, k, padding=self.pad).view(b, 2, n, h * w)
        mom_agg = (mom_p * w_nb).sum(dim=2).view(b, 2, h, w)

        if not self.use_rotation:
            return mom_agg, torch.zeros_like(vort)

        vort_scaled = vort * rho
        vort_p = F.unfold(vort_scaled, k, padding=self.pad).view(b, n, h * w)
        rot_x = self.stencil.rot_x.to(device=mom.device, dtype=mom.dtype).view(1, n, 1)
        rot_y = self.stencil.rot_y.to(device=mom.device, dtype=mom.dtype).view(1, n, 1)
        vx_rot = (vort_p * w_nb.squeeze(1) * rot_x).sum(dim=1).view(b, 1, h, w)
        vy_rot = (vort_p * w_nb.squeeze(1) * rot_y).sum(dim=1).view(b, 1, h, w)
        mom_agg[:, 0:1] = mom_agg[:, 0:1] + vx_rot
        mom_agg[:, 1:2] = mom_agg[:, 1:2] + vy_rot
        vort_agg = (vort_p * w_nb.squeeze(1)).sum(dim=1).view(b, 1, h, w)
        return mom_agg, vort_agg

    def forward(self, feat: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        mom_src = self.source_mom(feat)
        vort_src = self.source_vort(feat)
        mom_out, vort_out = self.propagate(mom_src, vort_src, rho)
        if self.use_rotation:
            cat = torch.cat([feat, mom_out, vort_out], dim=1)
        else:
            cat = torch.cat([feat, mom_out], dim=1)
        return feat + self.mix(cat)


class SpatialRotatingHuygensStack(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        kernel_size: int = 7,
        gamma: float = 0.5,
        use_rotation: bool = True,
        dropout: float = 0.1,
        cell_size: float = 2.0 / 31.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpatialRotatingHuygensBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    gamma=gamma * (0.9 ** i),
                    use_rotation=use_rotation,
                    dropout=dropout,
                    cell_size=cell_size,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, feat: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        x = feat
        for layer in self.layers:
            x = layer(x, rho)
        return x


class SpatialFluidHNFReconstructor(nn.Module):
    """Stage-0b spatial HNF: sparse 2D velocity → dense field (+ optional η).

    Architecture: 2D patch embed → spatial ρ field → rotating Huygens stack → velocity head.
  **Not** a rasterized 1D temporal encoder.
    """

    def __init__(
        self,
        h: int = 32,
        w: int = 32,
        in_channels: int = 3,
        out_channels: int = 2,
        embed_dim: int = 64,
        kernel_size: int = 7,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_eta: bool = True,
        use_rotation: bool = True,
    ):
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        self.embed_dim = int(embed_dim)
        self.predict_eta = bool(predict_eta)
        self.use_rotation = bool(use_rotation)
        cell_size = 2.0 / max(h - 1, 1)

        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.medium = SpatialMediumField(embed_dim, hidden=32)
        self.encoder = SpatialRotatingHuygensStack(
            channels=embed_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            use_rotation=use_rotation,
            dropout=dropout,
            cell_size=cell_size,
        )
        self.velocity_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )
        if self.predict_eta:
            self.eta_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Softplus(),
            )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, _, h, w = x.shape
        if h != self.h or w != self.w:
            x = F.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=False)
        feat = self.patch(x)
        rho = self.medium(feat)
        feat = self.encoder(feat, rho)
        dense = self.velocity_head(feat)
        aux: dict[str, torch.Tensor] = {"rho": rho, "feat": feat}
        if self.predict_eta:
            aux["eta"] = self.eta_head(feat).squeeze(-1)
        if return_aux:
            return dense, aux
        return dense

    def collect_kernel_params(self) -> dict[str, dict[str, float]]:
        params: dict[str, dict[str, float]] = {}
        for i, layer in enumerate(self.encoder.layers):
            params[f"layer{i}"] = {
                "gamma": float(layer.effective_gamma().detach().cpu()),
                "use_rotation": float(layer.use_rotation),
            }
        return params


def curl_2d(v: torch.Tensor) -> torch.Tensor:
    """Scalar vorticity ω = ∂v_y/∂x - ∂v_x/∂y, shape ``(B, 1, H, W)``."""
    if v.dim() != 4 or v.size(1) != 2:
        raise ValueError("v must be (B, 2, H, W)")
    vx, vy = v[:, 0:1], v[:, 1:2]
    dvx_dy = vx[:, :, 1:, :] - vx[:, :, :-1, :]
    dvx_dy = F.pad(dvx_dy, (0, 0, 0, 1))
    dvy_dx = vy[:, :, :, 1:] - vy[:, :, :, :-1]
    dvy_dx = F.pad(dvy_dx, (0, 1, 0, 0))
    return dvy_dx - dvx_dy
