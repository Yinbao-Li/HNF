# -*- coding: utf-8 -*-
"""EEG-native Huygens Neural Field — spatial secondary sources + rhythm priors.

v5 backbone upgrades (clinical AD/FTD oriented)
----------------------------------------------
1. **Regional spatial secondary sources** — frontal / temporal / posterior
   energy and frontotemporal contrast on top of global SpatialHuygensMix.
2. **Segmented temporal pooling** — early/late halves instead of pure GAP, so
   slowing drift is not washed out.
3. **δ + θ + α rhythm branches** — AD-relevant delta slowing joins θ/α priors.

Still not a biophysical EEG forward model — a domain-appropriate Huygens
inductive bias under the same research pattern.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.eeg_dataset import STANDARD_10_20
from hnf.eeg_geometry import electrode_distance_tensor, region_index_masks
from hnf.multiscale import DeepHuygensStack
from hnf.picking_model import TemporalMediumDensity


def _raw_for_softplus_target(target: float) -> float:
    """Invert softplus: softplus(x)≈target for target>0 → x = log(exp(t)-1)."""
    t = max(float(target), 1e-3)
    return float(math.log(math.expm1(t)))


class SpatialHuygensMix(nn.Module):
    """Mix electrode channels with a geometric Huygens-like spatial kernel.

    For each time sample:
        y = (α · K_geo(γ, ω) + (1-α) · I + β · W_learn) x
    where K_geo[i,j] = exp(-γ r_ij²) · cos(ω r_ij) / (r_ij²+ε), r from 10–20.
    """

    def __init__(
        self,
        n_channels: int = 19,
        gamma: float = 2.0,
        omega: float = 1.5,
        learnable: bool = True,
    ):
        super().__init__()
        if n_channels != len(STANDARD_10_20):
            raise ValueError("SpatialHuygensMix currently expects the 19-ch 10–20 set")
        dist = electrode_distance_tensor(STANDARD_10_20)
        self.register_buffer("dist", dist)
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(_raw_for_softplus_target(gamma)))
            self.omega = nn.Parameter(torch.tensor(_raw_for_softplus_target(omega)))
            self.mix = nn.Parameter(torch.tensor(0.7))
            self.res_scale = nn.Parameter(torch.tensor(0.2))
        else:
            self.register_buffer("gamma", torch.tensor(_raw_for_softplus_target(gamma)))
            self.register_buffer("omega", torch.tensor(_raw_for_softplus_target(omega)))
            self.register_buffer("mix", torch.tensor(0.7))
            self.register_buffer("res_scale", torch.tensor(0.0))
        self.W = nn.Parameter(0.01 * torch.randn(n_channels, n_channels))
        self.eps = 1e-4

    def geometric_kernel(self) -> torch.Tensor:
        g = F.softplus(self.gamma) + 1e-3
        w = F.softplus(self.omega) + 1e-3
        r = self.dist
        amp = 1.0 / (r * r + self.eps)
        env = torch.exp(-g * r * r)
        phase = torch.cos(w * r)
        K = amp * env * phase
        K = K / (K.abs().sum(dim=-1, keepdim=True) + 1e-6)
        return K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(B, C, T)`` → same shape."""
        K = self.geometric_kernel()
        eye = torch.eye(K.size(0), device=K.device, dtype=K.dtype)
        alpha = torch.sigmoid(self.mix)
        beta = F.softplus(self.res_scale)
        M = alpha * K + (1.0 - alpha) * eye + beta * self.W
        return torch.einsum("cd,bdt->bct", M, x)


class RegionalEnergy(nn.Module):
    """Scalp-region energies + frontotemporal contrast (interpretable)."""

    def __init__(self, n_channels: int = 19):
        super().__init__()
        if n_channels != len(STANDARD_10_20):
            raise ValueError("RegionalEnergy expects the 19-ch 10–20 montage")
        masks = region_index_masks(STANDARD_10_20)
        for name, mask in masks.items():
            self.register_buffer(f"mask_{name}", mask)
        self.region_names = ("frontal", "temporal", "central", "posterior")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x ``(B, C, T)`` → ``(B, 6)``: 4 region means + FT contrast + post/front."""
        # mean |amplitude| over time then channels in region
        amp = x.abs().mean(dim=-1)  # (B, C)
        vals = []
        for name in self.region_names:
            m = getattr(self, f"mask_{name}")
            vals.append(amp[:, m].mean(dim=-1))
        frontal, temporal, central, posterior = vals
        ft = frontal - temporal
        pf = posterior - frontal
        return torch.stack([frontal, temporal, central, posterior, ft, pf], dim=-1)


class SegmentPool(nn.Module):
    """Preserve early/late temporal structure instead of pure GAP."""

    def __init__(self, dim: int, n_segments: int = 2):
        super().__init__()
        self.n_segments = max(2, int(n_segments))
        # fuse [early, late, late-early] → dim
        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        """env ``(B, T, D)`` → ``(B, D)``."""
        b, t, d = env.shape
        # split into early / late halves (stable even if T odd)
        mid = max(1, t // 2)
        early = env[:, :mid].mean(dim=1)
        late = env[:, mid:].mean(dim=1) if mid < t else early
        drift = late - early
        return self.fuse(torch.cat([early, late, drift], dim=-1))


class RhythmBranch(nn.Module):
    """One temporal Huygens stack tuned to a target EEG rhythm (Hz)."""

    def __init__(
        self,
        embed_dim: int,
        target_hz: float,
        sample_rate: int,
        local_window_sec: float,
        num_layers: int = 2,
        dropout: float = 0.2,
        principle: str = "huygens_fresnel",
        dim: Optional[int] = None,
    ):
        super().__init__()
        dim = int(dim or embed_dim)
        self.embed_dim = embed_dim
        self.dim = dim
        self.target_hz = float(target_hz)
        omega0 = 2.0 * math.pi * float(target_hz)
        gamma0 = 1.5
        self.in_proj = (
            nn.Identity()
            if dim == embed_dim
            else nn.Sequential(nn.Linear(embed_dim, dim, bias=False), nn.LayerNorm(dim))
        )
        self.out_proj = (
            nn.Identity()
            if dim == embed_dim
            else nn.Sequential(nn.Linear(dim, embed_dim, bias=False), nn.LayerNorm(embed_dim))
        )
        self.stack = DeepHuygensStack(
            dim=dim,
            num_layers=num_layers,
            gamma=_raw_for_softplus_target(gamma0),
            omega=_raw_for_softplus_target(omega0),
            wave_speed=_raw_for_softplus_target(1.0),
            local_window_sec=local_window_sec,
            dropout=dropout,
            sparse_band=True,
            principle=principle,
            obliquity_scale=1.0,
        )

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hr = self.in_proj(h_real)
        hi = self.in_proj(h_imag)
        hr, hi = self.stack(hr, hi, t=t, rho=rho)
        return self.out_proj(hr), self.out_proj(hi)


class EEGHNFNativeClassifier(nn.Module):
    """EEG-native HNF v5 backbone.

    Pipeline
    --------
    (B, 19, T)
      → SpatialHuygensMix + RegionalEnergy
      → 1×1 channel embed → ρ(t)
      → temporal downsample ×2
      → RhythmBranch δ / θ / α
      → SegmentPool (early/late) per branch
      → concat region + ρ stats + band proxies → MLP
    """

    def __init__(
        self,
        n_channels: int = 19,
        seq_len: int = 1280,
        sample_rate: int = 128,
        embed_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.25,
        principle: str = "huygens_fresnel",
        mlp_hidden: int = 96,
        use_spatial: bool = True,
        temporal_downsample: int = 2,
        use_delta: bool = True,
        segment_pool: bool = True,
        include_region_in_head: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_spatial = use_spatial
        self.use_delta = use_delta
        self.segment_pool = segment_pool
        self.include_region_in_head = bool(include_region_in_head)
        self.temporal_downsample = max(1, int(temporal_downsample))
        self.branch_rate = max(1, sample_rate // self.temporal_downsample)

        self.spatial = SpatialHuygensMix(n_channels=n_channels) if use_spatial else nn.Identity()
        self.regional = RegionalEnergy(n_channels=n_channels)
        self.channel_embed = nn.Conv1d(n_channels, embed_dim, kernel_size=1)
        self.medium_net = TemporalMediumDensity(channels=embed_dim, hidden=32)

        self.theta = RhythmBranch(
            embed_dim, target_hz=6.0, sample_rate=self.branch_rate,
            local_window_sec=1.25, num_layers=1, dropout=dropout, principle=principle,
            dim=embed_dim,
        )
        self.alpha = RhythmBranch(
            embed_dim, target_hz=10.0, sample_rate=self.branch_rate,
            local_window_sec=0.75, num_layers=1, dropout=dropout, principle=principle,
            dim=max(32, embed_dim // 2),
        )
        self.delta = None
        if use_delta:
            self.delta = RhythmBranch(
                embed_dim, target_hz=2.5, sample_rate=self.branch_rate,
                local_window_sec=2.0, num_layers=1, dropout=dropout, principle=principle,
                dim=max(32, embed_dim // 2),
            )

        n_branches = 3 if use_delta else 2
        if segment_pool:
            self.pool_theta = SegmentPool(embed_dim)
            self.pool_alpha = SegmentPool(embed_dim)
            self.pool_delta = SegmentPool(embed_dim) if use_delta else None
        else:
            self.pool_theta = None
            self.pool_alpha = None
            self.pool_delta = None

        # branches + optional region(6) + rho(4) + band(4)
        # Pre-v5 native (v1–v3) heads used branches + rho + band only (dim 136 @ 64).
        region_dim = 6 if self.include_region_in_head else 0
        pooled_dim = embed_dim * n_branches + region_dim + 4 + 4
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )
        self.mmse_head = nn.Sequential(
            nn.Linear(pooled_dim, max(24, mlp_hidden // 2)),
            nn.GELU(),
            nn.Linear(max(24, mlp_hidden // 2), 1),
            nn.Sigmoid(),
        )
        self.band_proj = nn.Linear(embed_dim, 4)

    def _time_axis(self, batch: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(length, device=device, dtype=dtype) / float(self.branch_rate)
        return t.view(1, length, 1).expand(batch, -1, -1)

    @staticmethod
    def _env(hr: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(hr.pow(2) + hi.pow(2) + 1e-8)

    def _pool_env(self, env: torch.Tensor, pool: Optional[SegmentPool]) -> torch.Tensor:
        if pool is not None:
            return pool(env)
        return env.mean(dim=1)

    def encode(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        if x.size(1) == self.n_channels:
            x_bt = x
        elif x.size(2) == self.n_channels:
            x_bt = x.transpose(1, 2)
        else:
            raise ValueError(f"Cannot infer channels for {tuple(x.shape)}")

        x_mix = self.spatial(x_bt) if self.use_spatial else x_bt
        region = self.regional(x_mix)  # (B, 6)
        h = self.channel_embed(x_mix).transpose(1, 2)  # (B, T, D)
        rho = self.medium_net(h)

        if self.temporal_downsample > 1:
            h = F.avg_pool1d(h.transpose(1, 2), kernel_size=self.temporal_downsample).transpose(1, 2)
            rho = F.avg_pool1d(rho.transpose(1, 2), kernel_size=self.temporal_downsample).transpose(1, 2)

        h_imag0 = torch.zeros_like(h)
        t = self._time_axis(h.size(0), h.size(1), h.device, h.dtype)

        th_r, th_i = self.theta(h, h_imag0, t=t, rho=rho)
        al_r, al_i = self.alpha(h, h_imag0, t=t, rho=rho)
        th_env = self._env(th_r, th_i)
        al_env = self._env(al_r, al_i)
        p_th = self._pool_env(th_env, self.pool_theta)
        p_al = self._pool_env(al_env, self.pool_alpha)

        parts = [p_th, p_al]
        de_env = None
        if self.delta is not None:
            de_r, de_i = self.delta(h, h_imag0, t=t, rho=rho)
            de_env = self._env(de_r, de_i)
            parts.append(self._pool_env(de_env, self.pool_delta))

        rho_ = rho.squeeze(-1)
        rho_stats = torch.stack(
            [
                rho_.mean(dim=-1),
                rho_.std(dim=-1, unbiased=False),
                rho_.quantile(0.9, dim=-1),
                rho_.std(dim=-1, unbiased=False) / (rho_.mean(dim=-1).abs() + 1e-6),
            ],
            dim=-1,
        )
        band = torch.softmax(self.band_proj(h.mean(dim=1)), dim=-1)
        if self.include_region_in_head:
            pooled = torch.cat(parts + [region, rho_stats, band], dim=-1)
        else:
            pooled = torch.cat(parts + [rho_stats, band], dim=-1)

        if not return_aux:
            return pooled, None
        aux = {
            "rho": rho,
            "theta_env": th_env,
            "alpha_env": al_env,
            "band_proxy": band,
            "x_spatial": x_mix,
            "region_energy": region,
            "region_frontal": region[:, 0],
            "region_temporal": region[:, 1],
            "region_central": region[:, 2],
            "region_posterior": region[:, 3],
            "region_ft_contrast": region[:, 4],
            "region_pf_contrast": region[:, 5],
        }
        if de_env is not None:
            aux["delta_env"] = de_env
        return pooled, aux

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled, aux = self.encode(x, return_aux=return_aux)
        logits = self.head(pooled)
        if return_aux:
            assert aux is not None
            aux["mmse_pred_norm"] = self.mmse_head(pooled).squeeze(-1)
            return logits, aux
        return logits

    def collect_kernel_params(self) -> dict[str, dict[str, float]]:
        params: dict[str, dict[str, float]] = {}
        if self.use_spatial and isinstance(self.spatial, SpatialHuygensMix):
            params["spatial"] = {
                "gamma": float(F.softplus(self.spatial.gamma).detach().cpu() + 1e-3),
                "omega": float(F.softplus(self.spatial.omega).detach().cpu() + 1e-3),
                "mix": float(torch.sigmoid(self.spatial.mix).detach().cpu()),
            }
        branches = [("theta", self.theta), ("alpha", self.alpha)]
        if self.delta is not None:
            branches.append(("delta", self.delta))
        for name, branch in branches:
            for li, layer in enumerate(branch.stack.layers):
                k = layer.kernel
                params[f"{name}_layer{li}"] = {
                    "gamma": float(k.effective_gamma().detach().cpu()),
                    "omega": float(k.effective_omega().detach().cpu()),
                    "wave_speed": float(k.effective_wave_speed().detach().cpu()),
                    "target_hz": float(branch.target_hz),
                }
        return params
