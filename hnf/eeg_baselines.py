# -*- coding: utf-8 -*-
"""Standard EEG classifiers for Domain-II fair comparison (same (B,C,T) input)."""

from __future__ import annotations

import torch
import torch.nn as nn


class _Conv2dSameTime(nn.Module):
    """Temporal Conv2d that crops to preserve input time length."""

    def __init__(self, in_ch: int, out_ch: int, kern_length: int):
        super().__init__()
        self.pad = kern_length // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kern_length), padding=(0, self.pad), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # Odd kernels with pad=k//2 keep T; even can grow by 1 → crop.
        if y.size(-1) > x.size(-1):
            y = y[..., : x.size(-1)]
        return y


class EEGNet(nn.Module):
    """Compact EEGNet (Lawhern et al.) for (B, C, T) multi-class EEG.

    F1/D/F2 follow the original compact recipe; ``kern_length`` ≈ 0.5 s @ 128 Hz.
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 1280,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.temp_conv = _Conv2dSameTime(1, F1, kern_length)
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat = self._encode(dummy)
            flat = int(feat.numel())
        self.classifier = nn.Linear(flat, n_classes)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temp_conv(x)
        h = self.block1(h)
        h = self.sep_depth(h)
        if h.size(-1) % 2 == 1:
            h = h[..., :-1]
        return self.block2(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bct = _as_bct(x, self.n_channels)
        if x_bct.size(-1) != self.n_samples:
            x_bct = nn.functional.interpolate(
                x_bct, size=self.n_samples, mode="linear", align_corners=False
            )
        return self.classifier(self._encode(x_bct.unsqueeze(1)).flatten(1))


class Shallow1DCNN(nn.Module):
    """Light temporal CNN baseline: Conv1d stack → GAP → MLP."""

    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 1280,
        n_classes: int = 3,
        hidden: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        del n_samples  # length-agnostic via AdaptiveAvgPool
        self.n_channels = n_channels
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=25, padding=12),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(_as_bct(x, self.n_channels)))


def _as_bct(x: torch.Tensor, n_channels: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
    if x.size(1) == n_channels:
        return x
    if x.size(2) == n_channels:
        return x.transpose(1, 2)
    raise ValueError(f"Cannot infer channels for shape {tuple(x.shape)} (want C={n_channels})")


def build_eeg_baseline(
    name: str,
    *,
    n_channels: int = 19,
    n_samples: int = 1280,
    n_classes: int = 3,
    dropout: float = 0.25,
) -> nn.Module:
    key = name.strip().lower().replace("-", "").replace("_", "")
    if key == "eegnet":
        return EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout=dropout,
        )
    if key in {"shallow1d", "shallow", "cnn1d", "1dcnn"}:
        return Shallow1DCNN(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown EEG baseline model: {name!r}")
