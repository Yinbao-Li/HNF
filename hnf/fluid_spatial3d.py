# -*- coding: utf-8
"""3D spatial Huygens with vector vorticity secondary sources."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold3d(x: torch.Tensor, kernel_size: int, padding: int) -> torch.Tensor:
    """Extract K³ neighborhoods: (B,C,D,H,W) → (B,C,K³,D,H,W)."""
    b, c, d, h, w = x.shape
    k = kernel_size
    x_p = F.pad(x, (padding, padding, padding, padding, padding, padding), mode="replicate")
    patches = []
    for iz in range(k):
        for iy in range(k):
            for ix in range(k):
                patches.append(x_p[:, :, iz : iz + d, iy : iy + h, ix : ix + w])
    return torch.stack(patches, dim=2)


@dataclass(frozen=True)
class SpatialStencil3D:
    kernel_size: int
    dx: torch.Tensor
    dy: torch.Tensor
    dz: torch.Tensor
    r: torch.Tensor
    r2_safe: torch.Tensor


def build_spatial_stencil3d(kernel_size: int, cell_size: float = 1.0) -> SpatialStencil3D:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    half = kernel_size // 2
    offsets = []
    for oz in range(-half, half + 1):
        for oy in range(-half, half + 1):
            for ox in range(-half, half + 1):
                offsets.append((ox * cell_size, oy * cell_size, oz * cell_size))
    dx = torch.tensor([o[0] for o in offsets], dtype=torch.float32)
    dy = torch.tensor([o[1] for o in offsets], dtype=torch.float32)
    dz = torch.tensor([o[2] for o in offsets], dtype=torch.float32)
    r2 = dx.pow(2) + dy.pow(2) + dz.pow(2)
    r = torch.sqrt(r2 + 1e-8)
    r2_safe = r2.clone()
    r2_safe[r2_safe < 1e-8] = 1.0
    return SpatialStencil3D(kernel_size=kernel_size, dx=dx, dy=dy, dz=dz, r=r, r2_safe=r2_safe)


def biot_savart_velocity(omega: torch.Tensor, stencil: SpatialStencil3D) -> torch.Tensor:
    """ω (B,3,K³,N) and offsets → induced Δv (B,3,K³,N)."""
    ox = stencil.dx.view(1, 1, -1, 1).to(omega.device, omega.dtype)
    oy = stencil.dy.view(1, 1, -1, 1).to(omega.device, omega.dtype)
    oz = stencil.dz.view(1, 1, -1, 1).to(omega.device, omega.dtype)
    r2 = stencil.r2_safe.view(1, 1, -1, 1).to(omega.device, omega.dtype)
    wx, wy, wz = omega[:, 0:1], omega[:, 1:2], omega[:, 2:3]
    vx = wy * oz - wz * oy
    vy = wz * ox - wx * oz
    vz = wx * oy - wy * ox
    return torch.cat([vx, vy, vz], dim=1) / r2


class SpatialMediumField3D(nn.Module):
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat) + 1e-3


class SpatialRotatingHuygensBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        gamma: float = 0.5,
        use_rotation: bool = True,
        cell_size: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.use_rotation = bool(use_rotation)
        self.pad = kernel_size // 2
        self.stencil = build_spatial_stencil3d(kernel_size, cell_size=cell_size)
        self.log_gamma = nn.Parameter(torch.tensor(math.log(max(gamma, 1e-3)), dtype=torch.float32))
        self.source_mom = nn.Conv3d(channels, 3, kernel_size=1)
        self.source_vort = nn.Conv3d(channels, 3, kernel_size=1)
        self.mix = nn.Sequential(
            nn.Conv3d(channels + 3, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def effective_gamma(self) -> torch.Tensor:
        return F.softplus(self.log_gamma) + 1e-3

    def _weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        g = self.effective_gamma().to(device=device, dtype=dtype)
        r = self.stencil.r.to(device=device, dtype=dtype)
        return torch.exp(-g * r.pow(2)) / (r + 1e-3)

    def propagate(self, mom: torch.Tensor, vort: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = mom.shape
        k3 = self.kernel_size ** 3
        n = d * h * w
        w_nb = self._weights(mom.device, mom.dtype).view(1, 1, k3, 1)

        mom_p = unfold3d(mom * rho, self.kernel_size, self.pad).view(b, 3, k3, n)
        out = (mom_p * w_nb).sum(dim=2).view(b, 3, d, h, w)

        if self.use_rotation:
            vort_p = unfold3d(vort * rho, self.kernel_size, self.pad).view(b, 3, k3, n)
            dv = biot_savart_velocity(vort_p, self.stencil)
            out = out + (dv * w_nb).sum(dim=2).view(b, 3, d, h, w)
        return out

    def forward(self, feat: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        mom = self.source_mom(feat)
        vort = self.source_vort(feat)
        prop = self.propagate(mom, vort, rho)
        return feat + self.mix(torch.cat([feat, prop], dim=1))


class SpatialRotatingHuygensStack3D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        kernel_size: int = 5,
        use_rotation: bool = True,
        dropout: float = 0.1,
        cell_size: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpatialRotatingHuygensBlock3D(
                    channels=channels,
                    kernel_size=kernel_size,
                    gamma=0.5 * (0.9 ** i),
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


class Spatial3DFluidHNFReconstructor(nn.Module):
    """Sparse 3D velocity (vx,vy,vz) + mask → dense (3,D,H,W)."""

    def __init__(
        self,
        d: int = 12,
        h: int = 12,
        w: int = 12,
        in_channels: int = 4,
        embed_dim: int = 48,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_eta: bool = True,
        use_rotation: bool = True,
    ):
        super().__init__()
        self.d, self.h, self.w = int(d), int(h), int(w)
        cell = 2.0 / max(min(d, h, w) - 1, 1)
        self.patch = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.medium = SpatialMediumField3D(embed_dim)
        self.encoder = SpatialRotatingHuygensStack3D(
            embed_dim, num_layers=num_layers, kernel_size=kernel_size,
            use_rotation=use_rotation, dropout=dropout, cell_size=cell,
        )
        self.velocity_head = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(embed_dim, 3, kernel_size=1),
        )
        self.predict_eta = bool(predict_eta)
        if predict_eta:
            self.eta_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Softplus(),
            )

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] == (self.d, self.h, self.w):
            return x
        return F.interpolate(x, size=(self.d, self.h, self.w), mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self._resize(x)
        feat = self.patch(x)
        rho = self.medium(feat)
        feat = self.encoder(feat, rho)
        dense = self.velocity_head(feat)
        aux = {"rho": rho, "feat": feat}
        if self.predict_eta:
            aux["eta"] = self.eta_head(feat).squeeze(-1)
        return (dense, aux) if return_aux else dense

    def collect_kernel_params(self) -> dict:
        return {
            f"layer{i}": {"gamma": float(layer.effective_gamma().detach().cpu())}
            for i, layer in enumerate(self.encoder.layers)
        }


def curl_3d(v: torch.Tensor) -> torch.Tensor:
    """∇×v for (B,3,D,H,W) → (B,3,D,H,W) vorticity components."""
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    dvz_dy = vz[:, :, 1:, :] - vz[:, :, :-1, :]
    dvz_dy = F.pad(dvz_dy, (0, 0, 0, 1))
    dvy_dz = vy[:, 1:, :, :] - vy[:, :-1, :, :]
    dvy_dz = F.pad(dvy_dz, (0, 0, 0, 0, 0, 1))
    wx = dvz_dy - dvy_dz

    dvx_dz = vx[:, 1:, :, :] - vx[:, :-1, :, :]
    dvx_dz = F.pad(dvx_dz, (0, 0, 0, 0, 0, 1))
    dvz_dx = vz[:, :, :, 1:] - vz[:, :, :, :-1]
    dvz_dx = F.pad(dvz_dx, (0, 1, 0, 0))
    wy = dvx_dz - dvz_dx

    dvy_dx = vy[:, :, :, 1:] - vy[:, :, :, :-1]
    dvy_dx = F.pad(dvy_dx, (0, 1, 0, 0))
    dvx_dy = vx[:, :, 1:, :] - vx[:, :, :-1, :]
    dvx_dy = F.pad(dvx_dy, (0, 0, 0, 1))
    wz = dvy_dx - dvx_dy
    return torch.stack([wx, wy, wz], dim=1)
