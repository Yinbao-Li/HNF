# -*- coding: utf-8
"""4D (3D + time) spatial Huygens: per-frame 3D encoder + temporal mixing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.fluid_spatial3d import Spatial3DFluidHNFReconstructor, curl_3d


class Spatial4DFluidHNFReconstructor(nn.Module):
    """Input (B, 4, T, D, H, W) sparse+mask → (B, 3, T, D, H, W) dense velocity."""

    def __init__(
        self,
        t_steps: int = 4,
        d: int = 8,
        h: int = 12,
        w: int = 12,
        embed_dim: int = 40,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_rotation: bool = True,
        predict_eta: bool = False,
    ):
        super().__init__()
        self.t_steps = int(t_steps)
        self.d, self.h, self.w = int(d), int(h), int(w)
        self.embed_dim = int(embed_dim)
        self.core = Spatial3DFluidHNFReconstructor(
            d=d, h=h, w=w,
            in_channels=4,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
            predict_eta=False,
            use_rotation=use_rotation,
        )
        # Per-voxel temporal mixing (4D Flow cardiac phase axis)
        self.temporal = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.velocity_head = nn.Conv3d(embed_dim, 3, kernel_size=1)
        self.predict_eta = bool(predict_eta)
        if predict_eta:
            self.eta_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Softplus(),
            )

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,4,T,D,H,W) → feat (B,C,T,D,H,W)."""
        b, c, t, d, h, w = x.shape
        feats = []
        for ti in range(t):
            frame = x[:, :, ti]
            feat = self.core.patch(frame)
            rho = self.core.medium(feat)
            feat = self.core.encoder(feat, rho)
            feats.append(feat)
        return torch.stack(feats, dim=2)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        b, _, t, d, h, w = x.shape
        if t != self.t_steps:
            x = F.interpolate(
                x.view(b, -1, t, d * h, w),
                size=(self.t_steps, d * h, w),
                mode="trilinear",
                align_corners=False,
            )
            x = x.view(b, 4, self.t_steps, d, h, w)
        feat = self._encode_frames(x)
        b, c, t, d, h, w = feat.shape
        seq = feat.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, c, t)
        seq = self.temporal(seq)
        feat = seq.reshape(b, d, h, w, c, t).permute(0, 4, 5, 1, 2, 3)
        dense = self.velocity_head(feat.reshape(b * t, c, d, h, w)).reshape(b, 3, t, d, h, w)
        aux = {"feat": feat}
        if self.predict_eta:
            pooled = feat.mean(dim=(2, 3, 4, 5))
            aux["eta"] = self.eta_head(pooled).squeeze(-1)
        return (dense, aux) if return_aux else dense

    def collect_kernel_params(self) -> dict:
        return self.core.collect_kernel_params()


def curl_4d(v: torch.Tensor) -> torch.Tensor:
    """Spatial curl per time frame: (B,3,T,D,H,W) → (B,3,T,D,H,W)."""
    b, c, t, d, h, w = v.shape
    outs = []
    for ti in range(t):
        outs.append(curl_3d(v[:, :, ti]))
    return torch.stack(outs, dim=2)
