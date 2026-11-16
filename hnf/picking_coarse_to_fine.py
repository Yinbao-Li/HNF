# -*- coding: utf-8 -*-
"""Coarse-to-fine Huygens picking v3: 6000 @ 100 Hz I/O, physics on T_c.

Pipeline
--------
x (B, 6000, 3)
  ├─ MultiScaleEncoder
  │     skip_hi  (B, 6000, Ds)   # timing
  │     skip_mid (B, 1500, Dm)   # /4
  │     z        (B,  750, D)    # /8 → Huygens
  ├─ HuygensWaveBlocks on z + t_coarse(sec)
  ├─ Up to mid → fuse(skip_mid) → Up to fine → fuse(skip_hi)
  └─ P/S logits @ 6000 + det
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.kalman import KalmanUpdate
from hnf.layers import FreeKernelWaveBlock, HuygensWaveBlock
from hnf.picking_model import WaveFieldPickingHead, ScalarDetHead


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, stride: int = 1):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleEncoder(nn.Module):
    """Encode waveform into hi / mid / coarse features (factor-8 coarse)."""

    def __init__(
        self,
        in_ch: int = 3,
        skip_dim: int = 64,
        mid_dim: int = 64,
        embed_dim: int = 96,
        factor: int = 8,
    ):
        super().__init__()
        if factor not in {8, 16}:
            raise ValueError(f"supported factors 8 or 16, got {factor}")
        self.factor = factor
        # full-rate stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, skip_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(skip_dim, skip_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(skip_dim, skip_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # /2 → /4 (mid = fine/4 = 1500)
        self.down1 = ConvBlock(skip_dim, skip_dim, k=5, stride=2)
        self.down2 = ConvBlock(skip_dim, mid_dim, k=5, stride=2)
        # /4 → /8 (coarse = 750); optional extra /2 for factor 16
        self.down3 = ConvBlock(mid_dim, embed_dim, k=5, stride=2)
        self.down4 = ConvBlock(embed_dim, embed_dim, k=5, stride=2) if factor == 16 else None
        self.skip_dim = skip_dim
        self.mid_dim = mid_dim
        self.out_ch = embed_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, C) → features as (B, T*, D)
        h = self.stem(x.transpose(1, 2))          # (B, Ds, T)
        skip_hi = h.transpose(1, 2)               # (B, T, Ds)
        h = self.down1(h)                         # /2
        h = self.down2(h)                         # /4
        skip_mid = h.transpose(1, 2)              # (B, T/4, Dm)
        h = self.down3(h)                         # /8
        if self.down4 is not None:
            h = self.down4(h)                     # /16
        z = h.transpose(1, 2)                     # (B, Tc, D)
        return skip_hi, skip_mid, z


class FuseUp(nn.Module):
    """Upsample field and fuse with skip features."""

    def __init__(self, field_dim: int, skip_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(field_dim + skip_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, field: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # field/skip: (B, T, *)
        if field.size(1) != skip.size(1):
            field = F.interpolate(
                field.transpose(1, 2), size=skip.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
        return self.proj(torch.cat([field, skip], dim=-1))


class TimingHead(nn.Module):
    """Full-rate CNN head from fused fine features → pick logits."""

    def __init__(self, in_ch: int, hidden: int = 48, kernel: int = 7, layers: int = 3):
        super().__init__()
        pad = kernel // 2
        mods: list[nn.Module] = []
        ch = in_ch
        for i in range(max(2, layers) - 1):
            dil = 1 if i == 0 else 2
            mods.extend(
                [
                    nn.Conv1d(ch, hidden, kernel_size=kernel, padding=pad * dil, dilation=dil),
                    nn.GELU(),
                ]
            )
            ch = hidden
        mods.append(nn.Conv1d(ch, 1, kernel_size=kernel, padding=pad))
        self.net = nn.Sequential(*mods)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, T)
        return self.net(feats.transpose(1, 2)).squeeze(1)


class CoarseToFineHuygensPicker(nn.Module):
    """Native 100 Hz picking with coarse Huygens + multi-scale timing path."""

    def __init__(
        self,
        *,
        fine_len: int = 6000,
        coarse_len: int = 750,
        channels: int = 3,
        embed_dim: int = 96,
        skip_dim: int = 64,
        mid_dim: int = 64,
        num_shared_layers: int = 3,
        num_branch_layers: int = 3,
        gamma: float = 0.5,
        omega: float = 0.3,
        vp: float = 8.0,
        vs: float = 4.5,
        local_window_sec: float = 15.0,
        dropout: float = 0.1,
        sparse_band: bool = True,
        principle: str = "huygens_fresnel",
        obliquity_scale: float = 1.0,
        pick_head_hidden: int = 48,
        pick_head_kernel: int = 7,
        pick_head_layers: int = 4,
        residual_pick_head: bool = True,
        duration_sec: float = 60.0,
        fusion: str = "sum",
        wave_input: str = "fused",
        refine_max: float = 4.0,
        time_head_drop: float = 0.0,
        skip_bottleneck: int = 0,
        refine_pool_sec: float = 0.0,
        propagation: str = "huygens",
        gate_floor: float = 0.0,
        kalman: str = "none",
        kalman_groups: int = 8,
    ):
        super().__init__()
        self.gate_floor = float(gate_floor)
        if kalman not in {"none", "kalman", "gate"}:
            raise ValueError(f"kalman must be none|kalman|gate, got {kalman}")
        self.kalman = KalmanUpdate(embed_dim, groups=kalman_groups, mode=kalman) if kalman != "none" else None
        if propagation not in {"huygens", "free"}:
            raise ValueError(f"propagation must be huygens|free, got {propagation}")
        self.propagation = propagation
        if fusion not in {"sum", "propose_refine"}:
            raise ValueError(f"fusion must be sum|propose_refine, got {fusion}")
        if wave_input not in {"fused", "field"}:
            raise ValueError(f"wave_input must be fused|field, got {wave_input}")
        self.fusion = fusion
        self.wave_input = wave_input
        self.refine_max = float(refine_max)
        self.time_head_drop = float(time_head_drop)
        self.refine_pool_bins = int(round(refine_pool_sec * fine_len / duration_sec))
        if fine_len % coarse_len != 0:
            raise ValueError(f"fine_len={fine_len} must be divisible by coarse_len={coarse_len}")
        factor = fine_len // coarse_len
        if factor not in {8, 16}:
            raise ValueError("coarse_len must be fine_len/8 or /16 (750 or 375 for 6000)")

        self.fine_len = int(fine_len)
        self.coarse_len = int(coarse_len)
        self.mid_len = self.fine_len // 4
        self.duration_sec = float(duration_sec)
        self.embed_dim = int(embed_dim)
        self.skip_dim = int(skip_dim)
        self.mid_dim = int(mid_dim)

        self.encoder = MultiScaleEncoder(
            in_ch=channels,
            skip_dim=skip_dim,
            mid_dim=mid_dim,
            embed_dim=embed_dim,
            factor=factor,
        )
        self.coarse_proj = nn.Linear(self.encoder.out_ch, embed_dim)
        self.medium_from_h = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softplus())

        window_bins = int(round(local_window_sec * coarse_len / duration_sec)) + 1

        def _block(speed: float, i: int) -> nn.Module:
            if propagation == "free":
                return FreeKernelWaveBlock(dim=embed_dim, window_bins=window_bins, dropout=dropout)
            return HuygensWaveBlock(
                dim=embed_dim,
                gamma=gamma * (0.95 ** i),
                omega=omega * (1.05 ** i),
                wave_speed=speed,
                causal=True,
                distance_mode="time",
                local_window_sec=local_window_sec,
                learnable_kernel_params=True,
                dropout=dropout,
                sparse_band=sparse_band,
                principle=principle,
                obliquity_scale=obliquity_scale,
            )

        self.shared = nn.ModuleList([_block(0.5 * (vp + vs), i) for i in range(num_shared_layers)])
        self.p_layers = nn.ModuleList([_block(vp, i) for i in range(num_branch_layers)])
        self.s_layers = nn.ModuleList([_block(vs, i) for i in range(num_branch_layers)])

        # progressive decode: coarse → mid → fine
        self.fuse_mid_p = FuseUp(embed_dim, mid_dim, embed_dim, dropout=dropout)
        self.fuse_mid_s = FuseUp(embed_dim, mid_dim, embed_dim, dropout=dropout)
        hi_dim = int(skip_bottleneck) if skip_bottleneck > 0 else skip_dim
        self.skip_bn = nn.Linear(skip_dim, hi_dim) if skip_bottleneck > 0 else None
        self.fuse_hi_p = FuseUp(embed_dim, hi_dim, embed_dim, dropout=dropout)
        self.fuse_hi_s = FuseUp(embed_dim, hi_dim, embed_dim, dropout=dropout)

        # physics envelope head (coarse-upsampled) + timing CNN head (hi-res fused)
        self.p_wave_head = WaveFieldPickingHead(
            channels=3,
            hidden=pick_head_hidden,
            kernel_size=pick_head_kernel,
            num_layers=max(2, pick_head_layers - 1),
            residual_envelope=residual_pick_head,
        )
        self.s_wave_head = WaveFieldPickingHead(
            channels=3,
            hidden=pick_head_hidden,
            kernel_size=pick_head_kernel,
            num_layers=max(2, pick_head_layers - 1),
            residual_envelope=residual_pick_head,
        )
        self.p_time_head = TimingHead(embed_dim, hidden=pick_head_hidden, kernel=pick_head_kernel, layers=pick_head_layers)
        self.s_time_head = TimingHead(embed_dim, hidden=pick_head_hidden, kernel=pick_head_kernel, layers=pick_head_layers)
        self.time_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.wave_logit_scale = nn.Parameter(torch.tensor(0.5))
        if fusion == "propose_refine":
            self.wave_bias = nn.Parameter(torch.tensor(-4.0))
            for head in (self.p_wave_head, self.s_wave_head):
                if head.residual_envelope:
                    nn.init.zeros_(head.env_scale)

        # direct hi-res onset residual (raw timing cue)
        self.p_skip_head = nn.Conv1d(hi_dim, 1, kernel_size=5, padding=2)
        self.s_skip_head = nn.Conv1d(hi_dim, 1, kernel_size=5, padding=2)
        self.skip_logit_scale = nn.Parameter(torch.tensor(0.4))

        self.det_head = ScalarDetHead(embed_dim, dropout=dropout, residual_energy=True)

        t_c = torch.linspace(0.0, self.duration_sec, self.coarse_len).view(1, self.coarse_len, 1)
        self.register_buffer("t_coarse", t_c, persistent=False)

    def _propagate(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        layers: nn.ModuleList,
        t: torch.Tensor,
        rho: torch.Tensor,
        update: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in layers:
            h_real, h_imag = layer(h_real, h_imag, t=t, rho=rho)
            if update and self.kalman is not None:
                h_real = self.kalman(h_real)
        return h_real, h_imag

    @staticmethod
    def _up(h: torch.Tensor, length: int) -> torch.Tensor:
        return F.interpolate(
            h.transpose(1, 2), size=length, mode="linear", align_corners=False
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        del t
        if x.dim() != 3 or x.size(1) != self.fine_len:
            raise ValueError(f"expected x (B,{self.fine_len},C), got {tuple(x.shape)}")

        b = x.size(0)
        skip_hi, skip_mid, z = self.encoder(x)
        if z.size(1) != self.coarse_len:
            z = self._up(z, self.coarse_len)
        if skip_mid.size(1) != self.mid_len:
            skip_mid = self._up(skip_mid, self.mid_len)

        h_real = self.coarse_proj(z)
        h_imag = torch.zeros_like(h_real)
        rho = self.medium_from_h(h_real)
        t_c = self.t_coarse.expand(b, -1, -1)
        if self.kalman is not None:
            self.kalman.prepare(h_real)

        h_real, h_imag = self._propagate(h_real, h_imag, self.shared, t_c, rho, update=True)
        p_real, p_imag = self._propagate(h_real, h_imag, self.p_layers, t_c, rho)
        s_real, s_imag = self._propagate(h_real, h_imag, self.s_layers, t_c, rho)

        if self.skip_bn is not None:
            skip_hi = self.skip_bn(skip_hi)
        # mid fuse
        p_mid = self.fuse_mid_p(p_real, skip_mid)
        s_mid = self.fuse_mid_s(s_real, skip_mid)
        # fine fuse
        p_hi = self.fuse_hi_p(p_mid, skip_hi)
        s_hi = self.fuse_hi_s(s_mid, skip_hi)

        p_imag_hi = self._up(p_imag, self.fine_len)
        s_imag_hi = self._up(s_imag, self.fine_len)
        if self.wave_input == "field":
            p_wave_in, s_wave_in = self._up(p_real, self.fine_len), self._up(s_real, self.fine_len)
        else:
            p_wave_in, s_wave_in = p_hi, s_hi

        wave_scale = F.softplus(self.wave_logit_scale)
        time_scale = F.softplus(self.time_logit_scale)
        skip_scale = F.softplus(self.skip_logit_scale)

        skip_t = skip_hi.transpose(1, 2)
        p_wave = self.p_wave_head(p_wave_in, p_imag_hi)
        s_wave = self.s_wave_head(s_wave_in, s_imag_hi)
        p_ref = time_scale * self.p_time_head(p_hi) + skip_scale * self.p_skip_head(skip_t).squeeze(1)
        s_ref = time_scale * self.s_time_head(s_hi) + skip_scale * self.s_skip_head(skip_t).squeeze(1)

        if self.fusion == "propose_refine":
            # physics proposal w gates a bounded CNN correction: logit = w + g(w)·R·tanh(r/R),
            # g(w) = ε + (1−ε)·σ(w); ε > 0 lets the CNN add evidence where the physics path proposes nothing
            p_wave = p_wave + self.wave_bias
            s_wave = s_wave + self.wave_bias
            p_use, s_use = p_ref, s_ref
            if self.training and self.time_head_drop > 0:
                keep = (torch.rand(b, 1, device=x.device) >= self.time_head_drop).to(p_ref.dtype)
                p_use, s_use = p_ref * keep, s_ref * keep
            r_max = self.refine_max
            p_base, s_base = p_wave, s_wave
            if self.refine_pool_bins > 0:
                # physics fixes whether/where within ±pool; the refine term fixes the instant
                k = self.refine_pool_bins
                p_base = F.max_pool1d(p_wave.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)
                s_base = F.max_pool1d(s_wave.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)
            eps = self.gate_floor
            p_logits = p_base + (eps + (1 - eps) * torch.sigmoid(p_base)) * r_max * torch.tanh(p_use / r_max)
            s_logits = s_base + (eps + (1 - eps) * torch.sigmoid(s_base)) * r_max * torch.tanh(s_use / r_max)
        else:
            p_logits = wave_scale * p_wave + p_ref
            s_logits = wave_scale * s_wave + s_ref

        wave_energy = (h_real.pow(2) + h_imag.pow(2)).mean(dim=1)
        total_energy = x.pow(2).mean(dim=(1, 2))
        det = self.det_head(wave_energy, total_energy)

        extra = self.kalman.readouts() if self.kalman is not None else {}
        return {
            **extra,
            "det": det,
            "p": p_logits,
            "s": s_logits,
            "p_wave": p_wave,
            "s_wave": s_wave,
            "p_ref": p_ref,
            "s_ref": s_ref,
            "rho": rho.squeeze(-1),
            "coarse_len": torch.tensor(self.coarse_len, device=x.device),
            "fine_len": torch.tensor(self.fine_len, device=x.device),
        }


def build_c2f_picker(
    *,
    fine_len: int = 6000,
    coarse_len: int = 750,
    embed_dim: int = 96,
    skip_dim: int = 64,
    mid_dim: int = 64,
    num_shared_layers: int = 3,
    num_branch_layers: int = 3,
    gamma: float = 0.5,
    omega: float = 0.3,
    vp: float = 8.0,
    vs: float = 4.5,
    local_window_sec: float = 15.0,
    dropout: float = 0.1,
    sparse_band: bool = True,
    principle: str = "huygens_fresnel",
    **kwargs,
) -> CoarseToFineHuygensPicker:
    return CoarseToFineHuygensPicker(
        fine_len=fine_len,
        coarse_len=coarse_len,
        embed_dim=embed_dim,
        skip_dim=skip_dim,
        mid_dim=mid_dim,
        num_shared_layers=num_shared_layers,
        num_branch_layers=num_branch_layers,
        gamma=gamma,
        omega=omega,
        vp=vp,
        vs=vs,
        local_window_sec=local_window_sec,
        dropout=dropout,
        sparse_band=sparse_band,
        principle=principle,
        **kwargs,
    )
