"""Matched-budget continuous-filter baseline (SchNet-style, no pyg)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.qm9_huygens import AtomSecondarySources


class CFConv(nn.Module):
    """Distance-filtered continuous convolution."""

    def __init__(self, dim: int, n_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = float(cutoff)
        self.n_rbf = int(n_rbf)
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = cutoff / n_rbf
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B,N,D) pos:(B,N,3) mask:(B,N)
        d = torch.cdist(pos, pos)  # (B,N,N)
        m = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        eye = torch.eye(d.size(-1), device=d.device, dtype=torch.bool).unsqueeze(0)
        m = m & ~eye & (d <= self.cutoff)
        # RBF
        rbf = torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (self.width ** 2 + 1e-8))
        W = self.filter_net(rbf)  # (B,N,N,D)
        W = W * m.unsqueeze(-1).float()
        msg = torch.einsum("bnmd,bmd->bnd", W, h)
        return self.norm(h + self.proj(msg))


class ContinuousFilterMolecule(nn.Module):
    """M1 baseline."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_layers: int = 3,
        max_z: int = 20,
        cutoff: float = 5.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.source = AtomSecondarySources(embed_dim, max_z=max_z)
        self.layers = nn.ModuleList([CFConv(embed_dim, cutoff=cutoff) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, positions, atomic_numbers, atom_mask=None, scramble_geometry: bool = False, **_):
        if atom_mask is None:
            atom_mask = atomic_numbers > 0
        pos = positions
        if scramble_geometry:
            pos = pos + torch.randn_like(pos) * 5.0
        h = self.source(atomic_numbers)
        for layer in self.layers:
            h = self.dropout(layer(h, pos, atom_mask))
        m = atom_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        y = self.head(pooled).squeeze(-1)
        return {"y": y}
