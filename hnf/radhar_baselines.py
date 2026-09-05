"""Non-wave baseline models for RadHAR 5-class ablation.

All models accept ``(B, C, T)`` and return ``{"logits": (B, n_classes)}``.
Same interface as ``RadHARHuygensClsModel`` so the training loop is shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.radhar_io import ACTIVITIES


class LinearBaseline(nn.Module):
    """Global average-pool → single linear layer."""

    def __init__(self, n_channels: int, n_classes: int = len(ACTIVITIES), **_):
        super().__init__()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x: torch.Tensor, t=None, **_) -> dict:
        logits = self.fc(x.mean(dim=-1))
        return {"logits": logits}


class MLPBaseline(nn.Module):
    """Global average-pool + 3-layer MLP."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = len(ACTIVITIES),
        hidden: int = 256,
        dropout: float = 0.3,
        **_,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, t=None, **_) -> dict:
        return {"logits": self.mlp(x.mean(dim=-1))}


class CNNBaseline(nn.Module):
    """1-D temporal CNN (no wave propagation)."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = len(ACTIVITIES),
        hidden: int = 128,
        n_layers: int = 4,
        dropout: float = 0.3,
        **_,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, t=None, **_) -> dict:
        h = self.backbone(x)           # (B, hidden, T)
        h_mean = h.mean(dim=-1)
        h_max = h.max(dim=-1).values
        h_peak = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        logits = self.head(torch.cat([h_mean, h_max, h_peak], dim=-1))
        return {"logits": logits}


class GRUBaseline(nn.Module):
    """Bidirectional GRU."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = len(ACTIVITIES),
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        **_,
    ):
        super().__init__()
        self.proj = nn.Linear(n_channels, hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, t=None, **_) -> dict:
        # x: (B, C, T) → (B, T, C)
        h = self.proj(x.transpose(1, 2))
        out, _ = self.gru(h)
        logits = self.head(out[:, -1, :])
        return {"logits": logits}
