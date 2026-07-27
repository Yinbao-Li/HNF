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
        if y.size(-1) > x.size(-1):
            y = y[..., : x.size(-1)]
        return y


class EEGNet(nn.Module):
    """Compact EEGNet (Lawhern et al.) for (B, C, T) multi-class EEG."""

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
        del n_samples
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


class EEGTransformer(nn.Module):
    """Conv stem → temporal tokens → Transformer encoder + CLS (ViT-style on EEG).

    Fair comparison baseline: same (B, C, T) input, no HNF inductive bias.
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 1280,
        n_classes: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.25,
        pool_second: int = 8,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model
        # 1280 → 320 → ~40 (or fewer) temporal tokens
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=25, padding=12),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=15, padding=7),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(pool_second),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bct = _as_bct(x, self.n_channels)
        if x_bct.size(-1) != self.n_samples:
            x_bct = nn.functional.interpolate(
                x_bct, size=self.n_samples, mode="linear", align_corners=False
            )
        h = self.stem(x_bct).transpose(1, 2)  # (B, T', D)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.transformer(h)
        return self.head(h[:, 0])


class _ConformerBlock(nn.Module):
    """Simplified Conformer block: FFN/2 → MHSA → Conv → FFN/2 (macaron)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, conv_kernel: int = 15):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        pad = conv_kernel // 2
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_pw1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.conv_dw = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel, padding=pad, groups=d_model
        )
        self.conv_bn = nn.BatchNorm1d(d_model)
        self.conv_pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_drop = nn.Dropout(dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + 0.5 * self.ff1(x)
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(a)
        h = self.conv_norm(x).transpose(1, 2)  # (B, D, T)
        h = self.conv_pw1(h)
        h = nn.functional.glu(h, dim=1)
        h = self.conv_dw(h)
        if h.size(-1) > x.size(1):
            h = h[..., : x.size(1)]
        h = nn.functional.silu(self.conv_bn(h))
        h = self.conv_drop(self.conv_pw2(h)).transpose(1, 2)
        x = x + h
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class EEGConformer(nn.Module):
    """EEG-oriented Conformer: spatial-temporal stem + Conformer blocks + GAP.

    Inspired by Song et al. EEG Conformer; compact for small clinical N.
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_samples: int = 1280,
        n_classes: int = 3,
        d_model: int = 40,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        # Patch embedding similar to EEG Conformer: temporal then spatial
        self.patch = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(1, 25), stride=(1, 5), padding=(0, 12), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.Conv2d(d_model, d_model, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5)),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [_ConformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bct = _as_bct(x, self.n_channels)
        if x_bct.size(-1) != self.n_samples:
            x_bct = nn.functional.interpolate(
                x_bct, size=self.n_samples, mode="linear", align_corners=False
            )
        h = self.patch(x_bct.unsqueeze(1))  # (B, D, 1, T')
        h = h.squeeze(2).transpose(1, 2)  # (B, T', D)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.transpose(1, 2))


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
    if key in {"transformer", "eegtransformer", "vit"}:
        return EEGTransformer(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout=dropout,
        )
    if key in {"tinytransformer", "tiny", "transformer_tiny"}:
        # Smaller + heavier dropout for small-N clinical EEG.
        return EEGTransformer(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dropout=max(dropout, 0.4),
            pool_second=10,
        )
    if key in {"conformer", "eegconformer"}:
        return EEGConformer(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout=max(dropout, 0.35),
        )
    raise ValueError(f"Unknown EEG baseline model: {name!r}")
