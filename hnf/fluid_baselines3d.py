# -*- coding: utf-8
"""3D/4D fluid reconstruction baselines (U-Net, CNN-AE) for comparison vs spatial HNF."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3DReconstructor(nn.Module):
    """Standard 3D U-Net baseline (4D Flow MRI literature)."""

    def __init__(self, in_ch: int = 4, out_ch: int = 3, base: int = 24):
        super().__init__()
        b = int(base)
        self.enc1 = _ConvBlock3D(in_ch, b)
        self.enc2 = _ConvBlock3D(b, b * 2)
        self.enc3 = _ConvBlock3D(b * 2, b * 4)
        self.pool = nn.MaxPool3d(2)
        self.bot = _ConvBlock3D(b * 4, b * 8)
        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = _ConvBlock3D(b * 8, b * 4)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = _ConvBlock3D(b * 4, b * 2)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = _ConvBlock3D(b * 2, b)
        self.head = nn.Conv3d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d0, h0, w0 = x.shape[-3:]
        pd = (8 - d0 % 8) % 8
        ph = (8 - h0 % 8) % 8
        pw = (8 - w0 % 8) % 8
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.head(d1)
        return out[..., :d0, :h0, :w0]


class Conv3DAutoencoder(nn.Module):
    """Lightweight 3D CNN encoder-decoder."""

    def __init__(self, in_ch: int = 4, out_ch: int = 3, width: int = 32):
        super().__init__()
        w = int(width)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, w, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(w, w * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(w * 2, w * 4, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(w * 4, w * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(w * 2, w, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(w, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class UNet4DReconstructor(nn.Module):
    """4D: process (B,4,T,D,H,W) with shared 3D U-Net per frame."""

    def __init__(self, t_steps: int = 4, base: int = 20):
        super().__init__()
        self.t_steps = int(t_steps)
        self.core = UNet3DReconstructor(in_ch=4, out_ch=3, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, d, h, w = x.shape
        outs = []
        for ti in range(t):
            outs.append(self.core(x[:, :, ti]))
        return torch.stack(outs, dim=2)
