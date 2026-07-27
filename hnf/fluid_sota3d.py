# -*- coding: utf-8
"""Literature SOTA models for sparse 3D velocity reconstruction.

References
----------
RecFNO — Zhao et al., 2023, Int. J. Thermal Sciences / arXiv:2302.09808
    Mask-embedding Fourier neural operator for sparse observation → full field.

FlowMRI-Net — Wallerberger et al., 2025, JOCMR / arXiv:2410.08856
    Physics-driven unrolled optimization (simplified grid-domain variant here;
    original operates in k-space with complex-valued CRNN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """3D Fourier layer (Li et al. FNO extended to 3D)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / (in_channels * out_channels)
        m = self.modes
        self.weights_pos = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, m, m, m, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, m, m, m, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            b, self.out_channels, d, h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        md = min(self.modes, d)
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)
        w_pos = self.weights_pos[:, :, :md, :mh, :mw]
        out_ft[:, :, :md, :mh, :mw] = torch.einsum(
            "bixyz,ioxyz->boxyz", x_ft[:, :, :md, :mh, :mw], w_pos
        )
        if d > 1:
            w_neg = self.weights_neg[:, :, :md, :mh, :mw]
            out_ft[:, :, -md:, :mh, :mw] = torch.einsum(
                "bixyz,ioxyz->boxyz", x_ft[:, :, -md:, :mh, :mw], w_neg
            )
        return torch.fft.irfftn(out_ft, s=(d, h, w), dim=(-3, -2, -1))


class RecFNO3D(nn.Module):
    """RecFNO with **mask embedding** for grid sparse velocity (Zhao et al., 2023).

    Input: (B, 4, D, H, W) = sparse velocity (3) + observation mask (1).
    Output: (B, 3, D, H, W) reconstructed velocity.
    """

    def __init__(self, in_ch: int = 4, out_ch: int = 3, width: int = 24, modes: int = 4, depth: int = 4):
        super().__init__()
        self.width = int(width)
        self.lift = nn.Conv3d(in_ch, self.width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv3d(self.width, self.width, modes=modes) for _ in range(int(depth))]
        )
        self.local = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(int(depth))])
        self.head = nn.Sequential(
            nn.Conv3d(self.width, self.width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(self.width, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for spec, loc in zip(self.spectral, self.local):
            h = F.gelu(spec(h) + loc(h))
        return self.head(h)


class _ProxBlock3D(nn.Module):
    def __init__(self, ch: int, hidden: int):
        super().__init__()
        h = int(hidden)
        c = int(ch)
        self.net = nn.Sequential(
            nn.Conv3d(c, h, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(h, h, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(h, c, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FlowMRINetUnrolled3D(nn.Module):
    """Simplified FlowMRI-Net unrolled recon (Wallerberger et al., 2025).

    K stages of {data consistency on observed voxels → shared proximal CNN}.
    Matches the physics-driven unrolled spirit without k-space / complex CRNN.
    """

    def __init__(self, n_stages: int = 6, base: int = 24):
        super().__init__()
        self.n_stages = int(n_stages)
        self.prox = _ProxBlock3D(3, int(base))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sparse = x[:, :3]
        mask = x[:, 3:4].clamp(0.0, 1.0)
        v = sparse.clone()
        for _ in range(self.n_stages):
            v = mask * sparse + (1.0 - mask) * v
            v = self.prox(v)
        return v
