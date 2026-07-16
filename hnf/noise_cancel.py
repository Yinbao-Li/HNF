# -*- coding: utf-8 -*-
"""Huygens three-step physics-constrained noise cancellation for 3C waveforms."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.layers import build_huygens_kernel


class NoiseSourceInverter(nn.Module):
    """Step 1: map observed waveform to non-negative equivalent noise sources."""

    def __init__(self, channels: int = 3, hidden: int = 24, source_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, source_dim, kernel_size=3, padding=1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> S_noise: (B, T, source_dim), non-negative."""
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        return self.softplus(h)


class CoherentTriaxialEnhancer(nn.Module):
    """Step 3 ops 2-3: phase-aligned triaxial stack and backscatter to 3C."""

    def __init__(self, channels: int = 3, hidden: int = 16, ref_channel: int = 2):
        super().__init__()
        self.channels = channels
        self.ref_channel = ref_channel
        self.backscatter = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=3, padding=1),
        )
        self.mix = nn.Parameter(torch.tensor(0.5))

    def forward(self, u_denoised: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (u_enhanced_center, u_final) with shape (B,T,1) and (B,T,C)."""
        spec = torch.fft.rfft(u_denoised, dim=1)
        ref_phase = torch.angle(spec[..., self.ref_channel : self.ref_channel + 1])
        aligned = torch.abs(spec) * torch.exp(
            1j * (torch.angle(spec) - torch.angle(spec[..., self.ref_channel : self.ref_channel + 1]) + ref_phase)
        )
        center_spec = aligned.mean(dim=-1, keepdim=True)
        center = torch.fft.irfft(center_spec, n=u_denoised.size(1), dim=1)
        delta = self.backscatter(center.transpose(1, 2)).transpose(1, 2)
        mix = torch.sigmoid(self.mix)
        u_final = mix * (u_denoised + delta) + (1.0 - mix) * u_denoised
        return center, u_final


class HuygensNoiseCancelBranch(nn.Module):
    """
    Three-step HNF noise cancellation adapted to single-station 3C STEAD input.

    1) Invert equivalent temporal noise sources S_noise >= 0
    2) Propagate via causal Huygens kernel -> N_sim, with physics consistency hooks
    3) Subtract, coherent triaxial enhance, backscatter -> U_final
    """

    def __init__(
        self,
        channels: int = 3,
        source_dim: int = 16,
        hidden: int = 24,
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 6.0,
        local_window_sec: float = 15.0,
        learnable_kernel_params: bool = True,
        principle: str = "huygens",
        obliquity_scale: float = 1.0,
        obliquity_mix: float = 0.0,
        bayesian_mc: bool = False,
        n_samples: int = 32,
    ):
        super().__init__()
        self.source_dim = source_dim
        self.inverter = NoiseSourceInverter(channels=channels, hidden=hidden, source_dim=source_dim)
        self.rho_net = nn.Sequential(
            nn.Conv1d(channels, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
            nn.Softplus(),
        )
        self.prop_kernel = build_huygens_kernel(
            gamma=gamma,
            omega=omega,
            causal=True,
            wave_speed=wave_speed,
            learnable_kernel_params=learnable_kernel_params,
            learnable_wave_speed=learnable_kernel_params,
            distance_mode="time",
            local_window_sec=local_window_sec,
            sparse_band=True,
            principle=principle,
            obliquity_scale=obliquity_scale,
            obliquity_mix=obliquity_mix,
            bayesian_mc=bayesian_mc,
            n_samples=n_samples,
        )
        self.to_noise = nn.Conv1d(source_dim, channels, kernel_size=1, bias=False)
        self.enhancer = CoherentTriaxialEnhancer(channels=channels, hidden=hidden)

    def propagate_noise(
        self,
        s_noise: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """S_noise (B,T,D) -> N_sim (B,T,C) via causal Huygens propagation."""
        h_c = torch.complex(s_noise, torch.zeros_like(s_noise))
        n_field = self.prop_kernel.forward_apply(h_c, s_noise, t=t, rho=rho)
        return self.to_noise(n_field.real.transpose(1, 2)).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rho = self.rho_net(x.transpose(1, 2)).transpose(1, 2)
        s_noise = self.inverter(x)
        n_sim = self.propagate_noise(s_noise, t=t, rho=rho)
        u_denoised = x - n_sim
        u_center, u_final = self.enhancer(u_denoised)
        return {
            "s_noise": s_noise,
            "n_sim": n_sim,
            "u_denoised": u_denoised,
            "u_center": u_center,
            "u_final": u_final,
            "rho_noise": rho.squeeze(-1),
        }


def phase_smoothness_loss(field: torch.Tensor) -> torch.Tensor:
    """Penalize non-physical rapid phase jumps along time."""
    spec = torch.fft.rfft(field, dim=1)
    phase = torch.angle(spec)
    dphase = phase[:, 1:, :] - phase[:, :-1, :]
    dphase = torch.atan2(torch.sin(dphase), torch.cos(dphase))
    return (dphase**2).mean()


def noise_cancel_losses(
    outputs: dict[str, torch.Tensor],
    nc_out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    consistency_weight: float = 0.5,
    phase_weight: float = 0.1,
    preserve_weight: float = 0.3,
    energy_weight: float = 0.05,
    noise_suppress_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"]
    u_final = nc_out["u_final"]
    n_sim = nc_out["n_sim"]
    det = batch["det"]

    recon = F.mse_loss(u_final + n_sim, x)
    phase_loss = phase_smoothness_loss(n_sim)

    pick_weight = (batch["p_target"] + batch["s_target"]).clamp_max(1.0)
    event_mask = det > 0.5
    preserve = torch.tensor(0.0, device=x.device)
    if event_mask.any():
        diff = (u_final - x).pow(2).sum(dim=-1)
        preserve = (pick_weight[event_mask] * diff[event_mask]).mean()

    e_in = x.pow(2).mean()
    e_parts = u_final.pow(2).mean() + n_sim.pow(2).mean()
    energy = (e_in - e_parts).abs() / e_in.clamp_min(1e-6)

    noise_mask = det <= 0.5
    noise_suppress = torch.tensor(0.0, device=x.device)
    if noise_mask.any():
        noise_suppress = u_final[noise_mask].pow(2).mean()

    total = (
        consistency_weight * recon
        + phase_weight * phase_loss
        + preserve_weight * preserve
        + energy_weight * energy
        + noise_suppress_weight * noise_suppress
    )
    parts = {
        "nc_recon": float(recon.detach()),
        "nc_phase": float(phase_loss.detach()),
        "nc_preserve": float(preserve.detach()),
        "nc_energy": float(energy.detach()),
        "nc_noise_suppress": float(noise_suppress.detach()),
    }
    return total, parts
