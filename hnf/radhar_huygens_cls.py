"""STEAD wave/detection path for RadHAR 5-class activity detection.

Backbone: fixed Huygens/wave (no regime ablation).

v3: ConvStem + EnergyPoolHead + residual_energy
v4 (default): ConvStem + mean-only head + fixed kernel (ablation-informed)
              + optional WaveGRU hybrid
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.layers import HuygensWaveBlock
from hnf.picking_model import ComponentSecondarySources, TemporalMediumDensity
from hnf.radhar_io import ACTIVITIES


def build_time_axis(seq_len: int, epoch_sec: float, device=None) -> torch.Tensor:
    return torch.linspace(0.0, float(epoch_sec), seq_len, device=device).unsqueeze(-1)


class ConvStem(nn.Module):
    """Reduce many input channels to stem_ch with local temporal context."""

    def __init__(self, in_ch: int, stem_ch: int, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch * 2, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(stem_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(stem_ch * 2, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, stem_ch, T)
        return self.net(x)


class EnergyPoolHead(nn.Module):
    """mean + max + temporal-peak → MLP → n_classes."""

    def __init__(self, embed_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, energy_t: torch.Tensor) -> torch.Tensor:
        # energy_t: (B, T, E)
        e_mean = energy_t.mean(dim=1)
        e_max = energy_t.max(dim=1).values
        e_peak = F.adaptive_max_pool1d(
            energy_t.transpose(1, 2), 1
        ).squeeze(-1)
        return self.mlp(torch.cat([e_mean, e_max, e_peak], dim=-1))


class RadHARHuygensClsModel(nn.Module):
    """Wave-fixed classifier: ``(B, C, T)`` → ``(B, 5)`` logits.

    v4 defaults (ablation-informed):
      - ConvStem kept  (+0.2pp)
      - mean-only head (EnergyPool removed, was -0.11pp)
      - fixed kernel   (learnable_kernel_params=False, was -0.09pp)
      - residual_energy removed (was part of EnergyPool path)
    """

    def __init__(
        self,
        n_channels: int = 312,
        n_classes: int = len(ACTIVITIES),
        stem_ch: int = 64,
        embed_dim: int = 128,
        num_shared_layers: int = 4,
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 1.0,
        epoch_sec: float = 2.0,
        local_window_sec: float = 0.5,
        dropout: float = 0.3,
        sparse_band: bool = True,
        principle: str = "huygens",
        rhythm_phase: bool = True,
        medium_hidden: int = 64,
        # v4 ablation-informed defaults
        learnable_kernel_params: bool = False,
        use_energy_pool: bool = False,
        residual_energy: bool = False,
    ):
        super().__init__()
        if local_window_sec >= epoch_sec:
            raise ValueError(f"local_window_sec={local_window_sec} >= epoch_sec={epoch_sec}")
        self.n_channels = int(n_channels)
        self.stem_ch = int(stem_ch)
        self.n_classes = int(n_classes)
        self.epoch_sec = float(epoch_sec)
        self.embed_dim = int(embed_dim)
        self.residual_energy = bool(residual_energy)
        self.use_energy_pool = bool(use_energy_pool)

        self._no_stem = False  # set to True externally for ablation
        self.stem = ConvStem(n_channels, stem_ch, kernel=5, dropout=0.1)
        self.source_embed = ComponentSecondarySources(embed_dim, channels=stem_ch)
        self.medium_net = TemporalMediumDensity(channels=stem_ch, hidden=medium_hidden)
        self.shared_layers = nn.ModuleList([
            HuygensWaveBlock(
                embed_dim,
                gamma=gamma * (0.95 ** i),
                omega=omega * (1.05 ** i),
                wave_speed=wave_speed,
                causal=True,
                distance_mode="time",
                local_window_sec=local_window_sec,
                dropout=dropout,
                sparse_band=sparse_band,
                principle=principle,
                rhythm_phase=rhythm_phase,
                learnable_kernel_params=learnable_kernel_params,
            )
            for i in range(int(num_shared_layers))
        ])
        if use_energy_pool:
            self.head = EnergyPoolHead(embed_dim, n_classes, dropout=dropout)
        else:
            # mean-only head (ablation winner)
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, n_classes),
            )
        if self.residual_energy:
            self.energy_proj = nn.Linear(1, n_classes, bias=True)

    def propagate(self, x: torch.Tensor, t: torch.Tensor):
        rho = self.medium_net(x)
        h_real = self.source_embed(x)
        h_imag = torch.zeros_like(h_real)
        for layer in self.shared_layers:
            h_real, h_imag = layer(h_real, h_imag, t=t, rho=rho)
        return h_real, h_imag, rho

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None, return_fields: bool = False
    ) -> dict[str, torch.Tensor]:
        b, c, seq_len = x.shape
        if c != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {c}")
        if t is None:
            t = build_time_axis(seq_len, self.epoch_sec, device=x.device)
            t = t.unsqueeze(0).expand(b, seq_len, 1)

        # stem: (B, C, T) → (B, stem_ch, T) ; skip for -ConvStem ablation
        xs = x if self._no_stem else self.stem(x)
        # propagate: (B, T, stem_ch)
        h_real, h_imag, rho = self.propagate(xs.transpose(1, 2), t)
        energy_t = h_real ** 2 + h_imag ** 2   # (B, T, E)

        if self.use_energy_pool:
            logits = self.head(energy_t)
        else:
            logits = self.head(energy_t.mean(dim=1))   # mean-only

        if self.residual_energy:
            total_log = torch.log(energy_t.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8))
            logits = logits + self.energy_proj(total_log.squeeze(-1))

        out = {"logits": logits, "rho_mean": rho.mean(dim=(1, 2))}
        if return_fields:
            out["energy_t"] = energy_t
            out["rho"] = rho
        return out


# ──────────────────────────────────────────────────────────────────────────────
# WaveGRU: Wave as feature extractor → BiGRU temporal reasoner
# ──────────────────────────────────────────────────────────────────────────────

class WaveGRUModel(nn.Module):
    """Wave backbone (sequence output) → BiGRU → classify.

    Wave blocks output energy_t: (B, T, embed_dim).
    BiGRU operates on this sequence, combining wave physics with
    recurrent temporal modeling.
    """

    def __init__(
        self,
        n_channels: int = 312,
        n_classes: int = len(ACTIVITIES),
        stem_ch: int = 64,
        embed_dim: int = 128,
        num_wave_layers: int = 4,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 1.0,
        epoch_sec: float = 2.0,
        local_window_sec: float = 0.5,
        dropout: float = 0.3,
        medium_hidden: int = 64,
        learnable_kernel_params: bool = False,
    ):
        super().__init__()
        if local_window_sec >= epoch_sec:
            raise ValueError(f"local_window_sec={local_window_sec} >= epoch_sec={epoch_sec}")
        self.n_channels = int(n_channels)
        self.epoch_sec = float(epoch_sec)
        self.embed_dim = int(embed_dim)

        self.stem = ConvStem(n_channels, stem_ch, kernel=5, dropout=0.1)
        self.source_embed = ComponentSecondarySources(embed_dim, channels=stem_ch)
        self.medium_net = TemporalMediumDensity(channels=stem_ch, hidden=medium_hidden)
        self.wave_layers = nn.ModuleList([
            HuygensWaveBlock(
                embed_dim,
                gamma=gamma * (0.95 ** i),
                omega=omega * (1.05 ** i),
                wave_speed=wave_speed,
                causal=True,
                distance_mode="time",
                local_window_sec=local_window_sec,
                dropout=dropout,
                learnable_kernel_params=learnable_kernel_params,
            )
            for i in range(int(num_wave_layers))
        ])
        # GRU takes energy sequence: (B, T, embed_dim)
        self.gru = nn.GRU(
            embed_dim, gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, n_classes),
        )

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None, **_
    ) -> dict[str, torch.Tensor]:
        b, c, seq_len = x.shape
        if t is None:
            t = build_time_axis(seq_len, self.epoch_sec, device=x.device)
            t = t.unsqueeze(0).expand(b, seq_len, 1)

        xs = self.stem(x)                                   # (B, stem_ch, T)
        rho = self.medium_net(xs.transpose(1, 2))           # (B, T, stem_ch)
        h_real = self.source_embed(xs.transpose(1, 2))      # (B, T, E)
        h_imag = torch.zeros_like(h_real)
        for layer in self.wave_layers:
            h_real, h_imag = layer(h_real, h_imag, t=t, rho=rho)

        energy_t = h_real ** 2 + h_imag ** 2               # (B, T, E)
        out_gru, _ = self.gru(energy_t)                     # (B, T, 2*H)
        logits = self.head(out_gru[:, -1, :])               # last step
        return {"logits": logits, "rho_mean": rho.mean(dim=(1, 2))}

