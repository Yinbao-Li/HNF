#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huygens physical-degradation diffusion (new paper; not TGRS).

Forward (Markov):
  h_t = H(h_{t-1}) + σ_t ε
with H = HuygensWaveBlock (causal wave propagation).

Reverse (learned):
  predict clean waveform x0 (or field h0) from (h_t, t).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.layers import HuygensWaveBlock


class WaveformToField(nn.Module):
    """(B, T, C) → coarse complex field (B, N, D); imag init 0."""

    def __init__(self, fine_len: int = 6000, coarse_len: int = 750, in_ch: int = 3, dim: int = 32):
        super().__init__()
        if fine_len % coarse_len != 0:
            raise ValueError("fine_len must be divisible by coarse_len")
        self.fine_len = fine_len
        self.coarse_len = coarse_len
        self.dim = dim
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, dim, kernel_size=9, stride=fine_len // coarse_len, padding=4),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,C)
        h = self.down(x.transpose(1, 2)).transpose(1, 2)  # (B,N,D)
        if h.size(1) != self.coarse_len:
            h = F.interpolate(h.transpose(1, 2), size=self.coarse_len, mode="linear", align_corners=False).transpose(1, 2)
        return h, torch.zeros_like(h)


class FieldToWaveform(nn.Module):
    """Coarse field → (B, T, C) waveform."""

    def __init__(self, fine_len: int = 6000, coarse_len: int = 750, out_ch: int = 3, dim: int = 32):
        super().__init__()
        self.fine_len = fine_len
        self.up = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, out_ch, 5, padding=2),
        )

    def forward(self, h_real: torch.Tensor, h_imag: Optional[torch.Tensor] = None) -> torch.Tensor:
        # use real (+ imag energy as optional extra later)
        feat = h_real if h_imag is None else h_real + 0.1 * h_imag
        y = F.interpolate(feat.transpose(1, 2), size=self.fine_len, mode="linear", align_corners=False)
        return self.up(y).transpose(1, 2).contiguous()


class HuygensDegradation(nn.Module):
    """Stack of Huygens blocks + noise schedule for physical forward diffusion."""

    def __init__(
        self,
        dim: int = 32,
        n_blocks: int = 2,
        timesteps: int = 8,
        sigma_min: float = 0.01,
        sigma_max: float = 0.35,
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 6.0,
        local_window_sec: float = 15.0,
        principle: str = "huygens_fresnel",
        duration_sec: float = 60.0,
        coarse_len: int = 750,
        freeze_huygens: bool = False,
    ):
        super().__init__()
        self.timesteps = int(timesteps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.coarse_len = int(coarse_len)
        self.duration_sec = float(duration_sec)
        self.blocks = nn.ModuleList(
            [
                HuygensWaveBlock(
                    dim=dim,
                    gamma=gamma * (0.95 ** i),
                    omega=omega * (1.05 ** i),
                    wave_speed=wave_speed,
                    causal=True,
                    distance_mode="time",
                    local_window_sec=local_window_sec,
                    learnable_kernel_params=not freeze_huygens,
                    sparse_band=True,
                    principle=principle,
                )
                for i in range(n_blocks)
            ]
        )
        if freeze_huygens:
            for p in self.blocks.parameters():
                p.requires_grad = False
        t = torch.linspace(0.0, duration_sec, coarse_len).view(1, coarse_len, 1)
        self.register_buffer("t_grid", t, persistent=False)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """t in {1..T} or float in [0,1]."""
        if t.dtype in (torch.int32, torch.int64, torch.long):
            u = t.float() / float(self.timesteps)
        else:
            u = t.float().clamp(0, 1)
        # linear schedule in amplitude std
        return self.sigma_min + (self.sigma_max - self.sigma_min) * u

    def propagate_once(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        rho: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = h_real.size(0)
        t = self.t_grid.expand(b, -1, -1)
        if rho is None:
            rho = torch.ones(b, h_real.size(1), 1, device=h_real.device, dtype=h_real.dtype)
        for blk in self.blocks:
            h_real, h_imag = blk(h_real, h_imag, t, rho)
        return h_real, h_imag

    def forward_degrade(
        self,
        h0_real: torch.Tensor,
        h0_imag: torch.Tensor,
        t_steps: torch.Tensor,
        rho: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply t_steps Markov Huygens+noise transitions from h0.

        t_steps: (B,) int in 1..T
        Returns h_t_real, h_t_imag, sigma (B,1,1)
        """
        h_r, h_i = h0_real, h0_imag
        max_t = int(t_steps.max().item())
        # For each k=1..max_t, propagate all, then add noise only to samples with t_steps>=k
        for k in range(1, max_t + 1):
            h_r, h_i = self.propagate_once(h_r, h_i, rho)
            active = (t_steps >= k).float().view(-1, 1, 1)
            sig = self.sigma(torch.full((h_r.size(0),), k, device=h_r.device, dtype=torch.long)).view(-1, 1, 1)
            eps_r = torch.randn(h_r.shape, device=h_r.device, dtype=h_r.dtype, generator=generator)
            eps_i = torch.randn(h_i.shape, device=h_i.device, dtype=h_i.dtype, generator=generator)
            h_r = h_r + active * sig * eps_r
            h_i = h_i + active * sig * eps_i
        sig_t = self.sigma(t_steps).view(-1, 1, 1)
        return h_r, h_i, sig_t


class ReverseDenoiser(nn.Module):
    """Predict clean waveform from degraded field + diffusion step."""

    def __init__(self, dim: int = 32, fine_len: int = 6000, hidden: int = 64, out_ch: int = 3):
        super().__init__()
        self.fine_len = fine_len
        self.t_emb = nn.Embedding(64, dim)  # support t up to 63
        self.mix = nn.Sequential(
            nn.Linear(dim * 2 + dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.read = FieldToWaveform(fine_len=fine_len, dim=dim, out_ch=out_ch)
        self.refine = nn.Sequential(
            nn.Conv1d(out_ch, hidden, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, out_ch, 7, padding=3),
        )

    def forward(self, h_r: torch.Tensor, h_i: torch.Tensor, t_steps: torch.Tensor) -> torch.Tensor:
        # (B,N,D)
        te = self.t_emb(t_steps.clamp(0, self.t_emb.num_embeddings - 1)).unsqueeze(1).expand(-1, h_r.size(1), -1)
        h = self.mix(torch.cat([h_r, h_i, te], dim=-1))
        x = self.read(h, h_i)
        return x + self.refine(x.transpose(1, 2)).transpose(1, 2)


class HuygensDegradationDiffusion(nn.Module):
    """Full encode → physical degrade → reverse → waveform."""

    def __init__(
        self,
        fine_len: int = 6000,
        coarse_len: int = 750,
        dim: int = 32,
        timesteps: int = 8,
        n_huygens_blocks: int = 2,
        freeze_huygens: bool = False,
        mode: str = "huygens",  # huygens | gaussian
        **deg_kw,
    ):
        super().__init__()
        if mode not in {"huygens", "gaussian"}:
            raise ValueError("mode must be huygens|gaussian")
        self.mode = mode
        self.timesteps = timesteps
        self.encode = WaveformToField(fine_len, coarse_len, dim=dim)
        self.degrade = HuygensDegradation(
            dim=dim,
            n_blocks=n_huygens_blocks,
            timesteps=timesteps,
            coarse_len=coarse_len,
            freeze_huygens=freeze_huygens,
            **{k: v for k, v in deg_kw.items() if k in {
                "sigma_min", "sigma_max", "gamma", "omega", "wave_speed",
                "local_window_sec", "principle", "duration_sec",
            }},
        )
        self.reverse = ReverseDenoiser(dim=dim, fine_len=fine_len)
        # autoencoder path for identity regularizer
        self.decode0 = FieldToWaveform(fine_len=fine_len, coarse_len=coarse_len, dim=dim)

    def encode_field(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(x)

    def forward_degrade_waveform(
        self, x0: torch.Tensor, t_steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h0_r, h0_i = self.encode(x0)
        if self.mode == "gaussian":
            # ablation: no Huygens, only noise on field
            u = t_steps.float() / float(self.timesteps)
            sig = (self.degrade.sigma_min + (self.degrade.sigma_max - self.degrade.sigma_min) * u).view(-1, 1, 1)
            eps_r = torch.randn_like(h0_r)
            eps_i = torch.randn_like(h0_i)
            h_r = h0_r + sig * eps_r
            h_i = h0_i + sig * eps_i
            return h_r, h_i, h0_r, h0_i
        h_r, h_i, _ = self.degrade.forward_degrade(h0_r, h0_i, t_steps)
        return h_r, h_i, h0_r, h0_i

    def forward(self, x0: torch.Tensor, t_steps: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        b = x0.size(0)
        device = x0.device
        if t_steps is None:
            t_steps = torch.randint(1, self.timesteps + 1, (b,), device=device)
        h_r, h_i, h0_r, h0_i = self.forward_degrade_waveform(x0, t_steps)
        x_hat = self.reverse(h_r, h_i, t_steps)
        x_id = self.decode0(h0_r, h0_i)
        return {
            "x_hat": x_hat,
            "x_id": x_id,
            "t_steps": t_steps,
            "h_t_real": h_r,
            "h0_real": h0_r,
        }

    def reconstruction_loss(self, x0: torch.Tensor, out: dict[str, torch.Tensor], id_weight: float = 0.1) -> torch.Tensor:
        rec = F.mse_loss(out["x_hat"], x0)
        # envelope term near energy (cheap onset-friendly)
        env0 = x0.abs().mean(-1)
        envh = out["x_hat"].abs().mean(-1)
        env = F.l1_loss(envh, env0)
        ident = F.mse_loss(out["x_id"], x0)
        return rec + 0.25 * env + id_weight * ident
