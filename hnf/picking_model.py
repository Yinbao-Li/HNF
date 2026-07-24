# -*- coding: utf-8 -*-
"""Physics-grounded Huygens picking model for STEAD."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.layers import HuygensWaveBlock
from hnf.multiscale import MultiScaleHuygensEncoder, ScaleSpec, default_scale_specs
from hnf.learnable_sampler import (
    LearnableTemporalSampler,
    remap_index,
    remap_sequence,
)
from hnf.noise_cancel import HuygensNoiseCancelBranch


def _obliquity_block_config(
    mode: str,
    target: str,
    mix: float,
) -> tuple[str, float]:
    """Per-block (principle, obliquity_mix) from global obliquity_mode."""
    mix = float(max(0.0, min(1.0, mix)))
    if mode == "full_fresnel":
        return "huygens_fresnel", 1.0
    if mode == "none":
        return "huygens", 0.0
    if mode == "soft_all":
        return "huygens", mix
    if mode == "soft_shared" and target == "shared":
        return "huygens", mix
    if mode == "det_soft" and target == "det_shared":
        return "huygens", mix
    if mode == "det_fresnel" and target == "det_shared":
        return "huygens_fresnel", 1.0
    if mode == "noise_soft" and target == "noise":
        return "huygens", mix
    if mode == "noise_fresnel" and target == "noise":
        return "huygens_fresnel", 1.0
    return "huygens", 0.0


def _needs_det_shared_layers(mode: str) -> bool:
    return mode in {"det_soft", "det_fresnel"}


class ComponentSecondarySources(nn.Module):
    """Multi-component coupled secondary sources (3C land or 4C OBS Z12H)."""

    def __init__(self, embed_dim: int, channels: int = 3):
        super().__init__()
        c = int(channels)
        self.channels = c
        self.cross_comp = nn.Parameter(torch.eye(c) * 0.6 + torch.ones(c, c) / float(max(c * 5, 1)))
        self.chan_proj = nn.Conv1d(c, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("btc,cd->btd", x, self.cross_comp)
        return self.chan_proj(x.transpose(1, 2)).transpose(1, 2)


class TemporalMediumDensity(nn.Module):
    """从局部波形估计非均匀介质密度 rho(t)，调制次波衰减."""

    def __init__(self, channels: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=7, padding=3),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class WaveFieldPickingHead(nn.Module):
    """从传播波场的包络、包络变化率与虚部能量生成拾取曲线."""

    def __init__(
        self,
        channels: int = 3,
        hidden: int = 24,
        kernel_size: int = 7,
        num_layers: int = 3,
        residual_envelope: bool = True,
    ):
        super().__init__()
        self.residual_envelope = residual_envelope
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        in_ch = channels
        depth = max(2, int(num_layers))
        for i in range(depth - 1):
            dilation = 1 if i == 0 else 2
            pad_d = pad * dilation
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        hidden,
                        kernel_size=kernel_size,
                        padding=pad_d,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                ]
            )
            in_ch = hidden
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=kernel_size, padding=pad))
        self.refine = nn.Sequential(*layers)
        if residual_envelope:
            self.env_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_real: torch.Tensor, h_imag: torch.Tensor) -> torch.Tensor:
        envelope = torch.sqrt((h_real**2 + h_imag**2).sum(dim=-1) + 1e-8)
        d_env = envelope[:, 1:] - envelope[:, :-1]
        d_env = F.pad(d_env, (0, 1))
        imag_mag = h_imag.norm(dim=-1)
        feats = torch.stack([envelope, d_env, imag_mag], dim=1)
        delta = self.refine(feats).squeeze(1)
        if self.residual_envelope:
            return self.env_scale * envelope + delta
        return delta


class ScalarDetHead(nn.Module):
    """Scalar detection with optional log-energy residual skip."""

    def __init__(self, embed_dim: int, dropout: float = 0.1, residual_energy: bool = True):
        super().__init__()
        self.residual_energy = residual_energy
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        if residual_energy:
            self.energy_weight = nn.Parameter(torch.zeros(1))

    def forward(self, wave_energy: torch.Tensor, total_energy: torch.Tensor) -> torch.Tensor:
        logit = self.mlp(wave_energy).squeeze(-1)
        if self.residual_energy:
            logit = logit + self.energy_weight * torch.log(total_energy + 1e-8)
        return logit


class RawOnsetEncoder(nn.Module):
    """Shallow high-pass style encoder on raw waveform for weak-event onset."""

    def __init__(self, channels: int = 3, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).squeeze(1)
        onset = F.relu(h[:, 1:] - h[:, :-1])
        peak = h.max(dim=-1).values
        onset_peak = F.pad(onset, (0, 1)).max(dim=-1).values
        return torch.log(peak + 1e-8), torch.log(onset_peak + 1e-8)


class OnsetAwareDetHead(nn.Module):
    """Scalar det using mean embed + temporal peak/onset cues (weak events)."""

    def __init__(self, embed_dim: int, dropout: float = 0.1, use_raw_onset: bool = True):
        super().__init__()
        self.use_raw_onset = use_raw_onset
        in_dim = embed_dim + 2 + (2 if use_raw_onset else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        wave_energy: torch.Tensor,
        energy_t: torch.Tensor,
        raw_onset_feats: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        max_e = energy_t.max(dim=1).values
        d_e = energy_t[:, 1:] - energy_t[:, :-1]
        max_onset = F.pad(d_e, (0, 1)).max(dim=1).values
        feats = [
            wave_energy,
            torch.log(max_e + 1e-8).unsqueeze(-1),
            torch.log(max_onset + 1e-8).unsqueeze(-1),
        ]
        if self.use_raw_onset and raw_onset_feats is not None:
            feats.extend([f.unsqueeze(-1) for f in raw_onset_feats])
        return self.mlp(torch.cat(feats, dim=-1)).squeeze(-1)


class NoiseCueAdapter(nn.Module):
    """Compress denoise outputs into lightweight cues for P/S refinement.

    Input channels must stay at ``input_dim * 3 + 1`` (=10 for 3C) to match
    run28 checkpoints. Do not append onset/preserve_gate here — those broke
    weight loading (shape 10→12) and zeroed out the trained adapter.
    """

    def __init__(self, input_dim: int = 3, hidden: int = 24, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim * 3 + 1, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, embed_dim, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        n_sim: torch.Tensor,
        u_denoised: torch.Tensor,
        s_noise: torch.Tensor,
        preserve_gate: Optional[torch.Tensor] = None,  # kept for call-site compat; unused
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del preserve_gate  # unused — preserves run28 cue adapter input layout
        noise_energy = torch.sqrt((s_noise**2).mean(dim=-1, keepdim=True) + 1e-8)
        feats = torch.cat([x_raw, n_sim, u_denoised, noise_energy], dim=-1)
        cue = self.net(feats.transpose(1, 2)).transpose(1, 2)
        gate = self.gate(cue.transpose(1, 2)).transpose(1, 2)
        return cue, gate


class POnsetRefineAdapter(nn.Module):
    """Lightweight P-only residual hint from raw waveform and preserve gate."""

    def __init__(self, input_dim: int = 3, hidden: int = 24, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim + 1, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, embed_dim, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        preserve_gate: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if preserve_gate is None:
            preserve_gate = torch.zeros(
                x_raw.size(0), x_raw.size(1), 1, device=x_raw.device, dtype=x_raw.dtype
            )
        elif preserve_gate.dim() == 2:
            preserve_gate = preserve_gate.unsqueeze(-1)
        feats = torch.cat([x_raw, preserve_gate], dim=-1)
        hint = self.net(feats.transpose(1, 2)).transpose(1, 2)
        gate = self.gate(hint.transpose(1, 2)).transpose(1, 2)
        return hint, gate


class PhaseExistHead(nn.Module):
    """Scalar phase-existence logit: is there a P/S in this window?

    Uses pooled branch energy + pick-curve peak/onset cues. Intended for OBS
    where S is frequently absent; gate picks at inference with exist_threshold.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        pick_logits: torch.Tensor,
    ) -> torch.Tensor:
        energy_t = (h_real**2 + h_imag**2).mean(dim=-1)
        pool = (h_real**2 + h_imag**2).mean(dim=1).sqrt()
        peak = pick_logits.amax(dim=-1)
        d = pick_logits[:, 1:] - pick_logits[:, :-1]
        onset = F.pad(d, (0, 1)).amax(dim=-1)
        cues = torch.stack(
            [
                torch.log(energy_t.amax(dim=-1) + 1e-8),
                peak,
                onset,
            ],
            dim=-1,
        )
        return self.mlp(torch.cat([pool, cues], dim=-1)).squeeze(-1)


class StrongPhaseExistHead(nn.Module):
    """Stronger S/P existence head: temporal attention over branch field + pick cues.

    Designed to close the gap to oracle-exist gated S F1 (~0.72) by using the
    full time axis instead of global pool only.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.temp = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.attn = nn.Conv1d(hidden, 1, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + 5, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        pick_logits: torch.Tensor,
    ) -> torch.Tensor:
        # (B, T, D)
        mag = torch.sqrt(h_real**2 + h_imag**2 + 1e-8)
        h = self.temp(mag.transpose(1, 2))  # (B, H, T)
        w = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * w).sum(dim=-1)  # (B, H)
        peak = pick_logits.amax(dim=-1)
        mean = pick_logits.mean(dim=-1)
        std = pick_logits.std(dim=-1)
        d = pick_logits[:, 1:] - pick_logits[:, :-1]
        onset = F.pad(d, (0, 1)).amax(dim=-1)
        prom = peak - mean
        cues = torch.stack([peak, mean, std, onset, prom], dim=-1)
        return self.mlp(torch.cat([pooled, cues], dim=-1)).squeeze(-1)


class ParallelPhaseExistHead(nn.Module):
    """Existence from branch field only — parallel to pick, no pick-logit cues.

    Reads the same post-backbone branch field as the pick head, pools with
    temporal attention, and emits a scalar present/absent logit.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.temp = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.attn = nn.Conv1d(hidden, 1, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Start "present" so soft_floor does not randomly kill L1200 picks at step 0.
        nn.init.constant_(self.mlp[-1].bias, 3.0)

    def forward(self, h_real: torch.Tensor, h_imag: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(h_real**2 + h_imag**2 + 1e-8)  # (B, T, D)
        energy_t = mag.mean(dim=-1)  # (B, T)
        h = self.temp(mag.transpose(1, 2))  # (B, H, T)
        w = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * w).sum(dim=-1)  # (B, H)
        cues = torch.stack(
            [
                torch.log(energy_t.amax(dim=-1) + 1e-8),
                energy_t.mean(dim=-1),
                energy_t.std(dim=-1),
            ],
            dim=-1,
        )
        return self.mlp(torch.cat([pooled, cues], dim=-1)).squeeze(-1)


class PhasePickExistFuse(nn.Module):
    """Fuse parallel exist logit with pick curve → final pick logits.

    Pick and exist stay independent until this MLP; the fused curve is what
    peak-picking / metrics should consume. Also emits a refined exist logit
    from (raw exist + pick summary) for optional gating/supervision.

    Starts as identity (pick passthrough / exist passthrough) via residual +
    zero-init deltas so L1200 pick curves are not destroyed at step 0.
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.curve = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.exist_refine = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, pick_logits: torch.Tensor, exist_logit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pick_logits: (B, T), exist_logit: (B,)
        e = exist_logit.unsqueeze(-1).expand_as(pick_logits)
        gate = torch.sigmoid(exist_logit).unsqueeze(-1).expand_as(pick_logits)
        feat = torch.stack([pick_logits, e, gate], dim=-1)  # (B, T, 3)
        fused = pick_logits + self.curve(feat).squeeze(-1)
        peak = pick_logits.amax(dim=-1)
        mean = pick_logits.mean(dim=-1)
        std = pick_logits.std(dim=-1)
        exist_ref = exist_logit + self.exist_refine(
            torch.stack([exist_logit, peak, mean, std], dim=-1)
        ).squeeze(-1)
        return fused, exist_ref


class PSGapHead(nn.Module):
    """Predict S-P interval (seconds) from shared Huygens latents.

    Outputs a positive gap via softplus. Used as a waveform-only prior for
    joint P/S pairing when catalog distance is unavailable.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # energy mean/max/std (3) + soft peak time (1) + rho mean/std/max (3) + wave pooled (embed)
        in_dim = embed_dim + 7
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # [gap_raw, log_sigma_raw]
        )

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy_t = (h_real**2 + h_imag**2).mean(dim=-1)  # (B, T)
        energy_mean = energy_t.mean(dim=-1)
        energy_max = energy_t.amax(dim=-1)
        energy_std = energy_t.std(dim=-1, unbiased=False)
        t_idx = torch.arange(energy_t.size(-1), device=energy_t.device, dtype=energy_t.dtype)
        soft = energy_t / energy_t.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        soft_peak = (soft * t_idx).sum(dim=-1) / max(energy_t.size(-1) - 1, 1)

        rho_ = rho.squeeze(-1) if rho.dim() == 3 else rho
        rho_mean = rho_.mean(dim=-1)
        rho_std = rho_.std(dim=-1, unbiased=False)
        rho_max = rho_.amax(dim=-1)

        wave_pool = (h_real**2 + h_imag**2).mean(dim=1).sqrt()  # (B, D)
        feats = torch.cat(
            [
                wave_pool,
                energy_mean.unsqueeze(-1),
                energy_max.unsqueeze(-1),
                energy_std.unsqueeze(-1),
                soft_peak.unsqueeze(-1),
                rho_mean.unsqueeze(-1),
                rho_std.unsqueeze(-1),
                rho_max.unsqueeze(-1),
            ],
            dim=-1,
        )
        raw = self.net(feats)
        gap_sec = F.softplus(raw[:, 0]) + 0.2
        gap_sec = gap_sec.clamp(max=30.0)
        log_sigma = raw[:, 1].clamp(-2.0, 2.5)
        return gap_sec, log_sigma


class LocalPeakRerankHead(nn.Module):
    """Learned local competition on pick logits (residual refine).

    Targets close-race wrong peaks: GT is often already a local max but loses
    to a slightly higher spurious peak. A small temporal conv reweights the
    logit curve without changing the Huygens backbone narrative.
    """

    def __init__(self, hidden: int = 16, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 5, padding=2),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, T)
        delta = self.net(logits.unsqueeze(1)).squeeze(1)
        return logits + delta


class PPeakResidualOffset(nn.Module):
    """Post-pick P residual Δt from a waveform crop around the coarse peak.

    Coarse peak t0 (argmax / decode) → crop ±crop_half_bins → CNN/MLP →
    tanh * max_delta_bins. Caps the correction so large wrong peaks stay
    the job of multi-peak / wrong-peak losses.
    """

    def __init__(
        self,
        input_dim: int = 4,
        crop_half_bins: int = 30,
        hidden: int = 32,
        max_delta_bins: float = 8.0,
    ):
        super().__init__()
        self.crop_half_bins = int(max(1, crop_half_bins))
        self.max_delta_bins = float(max_delta_bins)
        self.conv = nn.Sequential(
            nn.Conv1d(int(input_dim), hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start near identity so early training does not jitter coarse picks.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def gather_crop(self, x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C), center: (B,) → crop (B, C, 2*half+1)."""
        bsz, tlen, channels = x.shape
        half = self.crop_half_bins
        x_pad = F.pad(x.transpose(1, 2), (half, half))  # (B, C, T+2h)
        idx = center.long().clamp(0, tlen - 1) + half
        offs = torch.arange(-half, half + 1, device=x.device)
        gather_idx = (idx.unsqueeze(1) + offs.unsqueeze(0)).unsqueeze(1).expand(bsz, channels, -1)
        return torch.gather(x_pad, 2, gather_idx)

    def forward(self, x: torch.Tensor, coarse_idx: torch.Tensor) -> torch.Tensor:
        """Return residual offset in bins, shape (B,)."""
        crop = self.gather_crop(x, coarse_idx)
        h = self.conv(crop).squeeze(-1)
        raw = self.mlp(h).squeeze(-1)
        return torch.tanh(raw) * self.max_delta_bins


class STEADHNFPickingModel(nn.Module):
    """
    惠更斯物理拾取模型:
      1. 三分量次波源
      2. 介质密度 rho(t)
      3. 共享因果波传播 (次波叠加)
      4. P/S 双分支以不同波速与频率传播
      5. 波场包络/相位前沿 -> 拾取
    """

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 64,
        num_shared_layers: int = 2,
        num_branch_layers: int = 2,
        gamma: float = 0.5,
        omega: float = 0.3,
        vp: float = 8.0,
        vs: float = 4.5,
        omega_p: float = 1.2,
        omega_s: float = 0.6,
        local_window_sec: float = 15.0,
        dropout: float = 0.1,
        per_time_det: bool = False,
        pick_head_hidden: int = 24,
        pick_head_kernel: int = 7,
        pick_head_layers: int = 3,
        multi_scale: bool = False,
        scale_specs: Optional[list[ScaleSpec]] = None,
        sparse_band: bool = False,
        num_anchors: int = 0,
        residual_pick_head: bool = True,
        residual_det_head: bool = True,
        enhanced_det_head: bool = False,
        noise_cancel: bool = False,
        noise_source_dim: int = 16,
        noise_det_pick_split: bool = False,
        noise_pick_cues: bool = False,
        principle: str = "huygens",
        obliquity_scale: float = 1.0,
        obliquity_mode: str = "none",
        obliquity_mix: float = 0.0,
        predict_ps_gap: bool = False,
        ps_gap_hidden: int = 64,
        peak_rerank: bool = False,
        peak_rerank_hidden: int = 16,
        p_residual_offset: bool = False,
        p_residual_crop_half_bins: int = 30,
        p_residual_hidden: int = 32,
        p_residual_max_delta_bins: float = 8.0,
        causal_peak_rank: bool = False,
        causal_peak_rank_hidden: int = 48,
        causal_peak_rank_topk: int = 8,
        causal_peak_rank_crop_half: int = 16,
        phase_exist: bool = False,
        phase_exist_hidden: int = 64,
        strong_s_exist: bool = False,
        parallel_exist_fuse: bool = False,
        bayesian_mc: bool = False,
        n_samples: int = 32,
        learnable_sampler: bool = False,
        sampler_out_len: int = 800,
        sampler_hidden: int = 32,
        sampler_temperature: float = 0.05,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.per_time_det = per_time_det
        self.enhanced_det_head = enhanced_det_head
        self.noise_cancel = noise_cancel
        self.noise_det_pick_split = noise_det_pick_split
        self.noise_pick_cues = noise_pick_cues
        self.predict_ps_gap = predict_ps_gap
        self.peak_rerank = peak_rerank
        self.p_residual_offset_enabled = bool(p_residual_offset)
        self.causal_peak_rank_enabled = bool(causal_peak_rank)
        self.multi_scale = multi_scale
        self.num_anchors = max(0, int(num_anchors))
        self.bayesian_mc = bool(bayesian_mc)
        self.n_samples = max(1, int(n_samples))
        self.learnable_sampler = bool(learnable_sampler)
        self.sampler_out_len = int(sampler_out_len)
        # Legacy single principle; obliquity_mode overrides per-block routing.
        if obliquity_mode == "none" and principle == "huygens_fresnel":
            obliquity_mode = "full_fresnel"
        self.principle = principle
        self.obliquity_mode = obliquity_mode
        self.obliquity_mix = float(max(0.0, min(1.0, obliquity_mix)))
        self.obliquity_scale = obliquity_scale

        self.input_dim = int(input_dim)
        self.source_embed = ComponentSecondarySources(embed_dim, channels=self.input_dim)
        self.medium_net = TemporalMediumDensity(channels=self.input_dim)
        self.dropout = nn.Dropout(dropout)
        self.temporal_sampler: Optional[LearnableTemporalSampler] = None
        if self.learnable_sampler:
            self.temporal_sampler = LearnableTemporalSampler(
                channels=input_dim,
                hidden=sampler_hidden,
                out_len=self.sampler_out_len,
                temperature=sampler_temperature,
            )

        if multi_scale:
            specs = scale_specs or default_scale_specs(
                embed_dim=embed_dim,
                local_window_sec=local_window_sec,
            )
            self.multi_scale_encoder = MultiScaleHuygensEncoder(
                embed_dim=embed_dim,
                scale_specs=specs,
                gamma=gamma,
                omega=omega,
                wave_speed=6.0,
                dropout=dropout,
                sparse_band=sparse_band,
                principle=principle,
                obliquity_scale=obliquity_scale,
                bayesian_mc=self.bayesian_mc,
                n_samples=self.n_samples,
            )
            self.shared_layers = None
            self.shared_det_layers = None
        else:
            self.multi_scale_encoder = None
            sh_pr, sh_mix = _obliquity_block_config(obliquity_mode, "shared", self.obliquity_mix)
            self.shared_layers = nn.ModuleList(
                [
                    HuygensWaveBlock(
                        dim=embed_dim,
                        gamma=gamma * (0.95 ** i),
                        omega=omega * (1.05 ** i),
                        wave_speed=6.0,
                        distance_mode="time",
                        local_window_sec=local_window_sec,
                        learnable_kernel_params=True,
                        dropout=dropout,
                        sparse_band=sparse_band,
                        principle=sh_pr,
                        obliquity_scale=obliquity_scale,
                        obliquity_mix=sh_mix,
                        bayesian_mc=self.bayesian_mc,
                        n_samples=self.n_samples,
                    )
                    for i in range(num_shared_layers)
                ]
            )
            if _needs_det_shared_layers(obliquity_mode):
                det_pr, det_mix = _obliquity_block_config(
                    obliquity_mode, "det_shared", self.obliquity_mix
                )
                self.shared_det_layers = nn.ModuleList(
                    [
                        HuygensWaveBlock(
                            dim=embed_dim,
                            gamma=gamma * (0.95 ** i),
                            omega=omega * (1.05 ** i),
                            wave_speed=6.0,
                            distance_mode="time",
                            local_window_sec=local_window_sec,
                            learnable_kernel_params=True,
                            dropout=dropout,
                            sparse_band=sparse_band,
                            principle=det_pr,
                            obliquity_scale=obliquity_scale,
                            obliquity_mix=det_mix,
                            bayesian_mc=self.bayesian_mc,
                            n_samples=self.n_samples,
                        )
                        for i in range(num_shared_layers)
                    ]
                )
            else:
                self.shared_det_layers = None

        pk_pr, pk_mix = _obliquity_block_config(obliquity_mode, "pick", self.obliquity_mix)
        self.p_layers = nn.ModuleList(
            [
                HuygensWaveBlock(
                    dim=embed_dim,
                    gamma=gamma * 0.85,
                    omega=omega_p * (1.03 ** i),
                    wave_speed=vp,
                    distance_mode="time",
                    local_window_sec=local_window_sec,
                    learnable_kernel_params=True,
                    dropout=dropout,
                    sparse_band=sparse_band,
                    principle=pk_pr,
                    obliquity_scale=obliquity_scale,
                    obliquity_mix=pk_mix,
                    bayesian_mc=self.bayesian_mc,
                    n_samples=self.n_samples,
                )
                for i in range(num_branch_layers)
            ]
        )
        self.s_layers = nn.ModuleList(
            [
                HuygensWaveBlock(
                    dim=embed_dim,
                    gamma=gamma * (0.95 ** i),
                    omega=omega_s * (1.03 ** i),
                    wave_speed=vs,
                    distance_mode="time",
                    local_window_sec=local_window_sec,
                    learnable_kernel_params=True,
                    dropout=dropout,
                    sparse_band=sparse_band,
                    principle=pk_pr,
                    obliquity_scale=obliquity_scale,
                    obliquity_mix=pk_mix,
                    bayesian_mc=self.bayesian_mc,
                    n_samples=self.n_samples,
                )
                for i in range(num_branch_layers)
            ]
        )

        if per_time_det:
            self.det_head = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(16, 1, kernel_size=7, padding=3),
            )
            self.raw_onset_encoder = None
        elif enhanced_det_head:
            self.det_head = OnsetAwareDetHead(
                embed_dim=embed_dim,
                dropout=dropout,
                use_raw_onset=True,
            )
            self.raw_onset_encoder = RawOnsetEncoder(channels=input_dim)
        else:
            self.det_head = ScalarDetHead(
                embed_dim=embed_dim,
                dropout=dropout,
                residual_energy=residual_det_head,
            )
            self.raw_onset_encoder = None
        self.p_pick_head = WaveFieldPickingHead(
            hidden=pick_head_hidden,
            kernel_size=pick_head_kernel,
            num_layers=pick_head_layers,
            residual_envelope=residual_pick_head,
        )
        self.s_pick_head = WaveFieldPickingHead(
            hidden=pick_head_hidden,
            kernel_size=pick_head_kernel,
            num_layers=pick_head_layers,
            residual_envelope=residual_pick_head,
        )
        self.noise_cancel_branch: Optional[HuygensNoiseCancelBranch] = None
        self.noise_cue_adapter: Optional[NoiseCueAdapter] = None
        self.p_onset_refine: Optional[POnsetRefineAdapter] = None
        if noise_cancel:
            nc_pr, nc_mix = _obliquity_block_config(obliquity_mode, "noise", self.obliquity_mix)
            self.noise_cancel_branch = HuygensNoiseCancelBranch(
                channels=input_dim,
                source_dim=noise_source_dim,
                hidden=max(16, pick_head_hidden // 2),
                gamma=gamma,
                omega=omega,
                wave_speed=6.0,
                local_window_sec=local_window_sec,
                learnable_kernel_params=True,
                principle=nc_pr,
                obliquity_scale=obliquity_scale,
                obliquity_mix=nc_mix,
                bayesian_mc=self.bayesian_mc,
                n_samples=self.n_samples,
            )
            if noise_pick_cues:
                self.noise_cue_adapter = NoiseCueAdapter(
                    input_dim=input_dim,
                    hidden=max(16, pick_head_hidden // 2),
                    embed_dim=embed_dim,
                )
                # POnsetRefineAdapter is OBS-experimental; do NOT enable by default —
                # random init with no checkpoint weights collapses P/S quality.
        self.ps_gap_head: Optional[PSGapHead] = None
        if predict_ps_gap:
            self.ps_gap_head = PSGapHead(
                embed_dim=embed_dim,
                hidden=ps_gap_hidden,
                dropout=dropout,
            )
        self.p_peak_rerank: Optional[LocalPeakRerankHead] = None
        self.s_peak_rerank: Optional[LocalPeakRerankHead] = None
        if peak_rerank:
            self.p_peak_rerank = LocalPeakRerankHead(hidden=peak_rerank_hidden)
            self.s_peak_rerank = LocalPeakRerankHead(hidden=peak_rerank_hidden)
        self.p_residual_offset: Optional[PPeakResidualOffset] = None
        if self.p_residual_offset_enabled:
            self.p_residual_offset = PPeakResidualOffset(
                input_dim=self.input_dim,
                crop_half_bins=int(p_residual_crop_half_bins),
                hidden=int(p_residual_hidden),
                max_delta_bins=float(p_residual_max_delta_bins),
            )
        self.p_causal_peak_rank = None
        if self.causal_peak_rank_enabled:
            from hnf.causal_peak_rank import CausalPeakRankHead

            self.p_causal_peak_rank = CausalPeakRankHead(
                crop_half_bins=int(causal_peak_rank_crop_half),
                hidden=int(causal_peak_rank_hidden),
                topk=int(causal_peak_rank_topk),
                dropout=dropout,
            )
        self.phase_exist = bool(phase_exist)
        self.strong_s_exist = bool(strong_s_exist)
        self.parallel_exist_fuse = bool(parallel_exist_fuse)
        self.p_exist_head: Optional[nn.Module] = None
        self.s_exist_head: Optional[nn.Module] = None
        self.p_exist_fuse: Optional[PhasePickExistFuse] = None
        self.s_exist_fuse: Optional[PhasePickExistFuse] = None
        if self.parallel_exist_fuse:
            # Parallel field-only exist + fuse MLP (preferred OBS path).
            self.phase_exist = True
            hid = max(96, int(phase_exist_hidden))
            self.p_exist_head = ParallelPhaseExistHead(
                embed_dim=embed_dim, hidden=hid, dropout=dropout
            )
            self.s_exist_head = ParallelPhaseExistHead(
                embed_dim=embed_dim, hidden=hid, dropout=dropout
            )
            self.p_exist_fuse = PhasePickExistFuse(hidden=hid, dropout=dropout)
            self.s_exist_fuse = PhasePickExistFuse(hidden=hid, dropout=dropout)
        elif self.phase_exist:
            self.p_exist_head = PhaseExistHead(
                embed_dim=embed_dim, hidden=phase_exist_hidden, dropout=dropout
            )
            if self.strong_s_exist:
                self.s_exist_head = StrongPhaseExistHead(
                    embed_dim=embed_dim, hidden=max(96, phase_exist_hidden), dropout=dropout
                )
            else:
                self.s_exist_head = PhaseExistHead(
                    embed_dim=embed_dim, hidden=phase_exist_hidden, dropout=dropout
                )

    def _propagate(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        layers: nn.ModuleList,
        t: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in layers:
            h_real, h_imag = layer(h_real, h_imag, t=t, rho=rho)
        return h_real, h_imag

    def _encode_shared_wavefield(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
        layers: Optional[nn.ModuleList] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_n = h_real.size(1)
        if self.num_anchors > 0 and self.num_anchors < full_n:
            h_real, h_imag, t_shared, rho_shared = self._resample_field(
                h_real, h_imag, t, rho, self.num_anchors
            )
        else:
            t_shared, rho_shared = t, rho

        use_layers = layers
        if use_layers is None:
            use_layers = self.shared_layers
        h_real, h_imag = self._run_shared_propagation(
            h_real, h_imag, t=t_shared, rho=rho_shared, layers=use_layers
        )

        if self.num_anchors > 0 and self.num_anchors < full_n:
            h_real, h_imag, _, _ = self._resample_field(h_real, h_imag, t, rho, full_n)
        return h_real, h_imag

    def _det_logits(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.per_time_det:
            energy_t = (h_real**2 + h_imag**2).mean(dim=-1)
            return self.det_head(energy_t.unsqueeze(1)).squeeze(1)
        energy_t = (h_real**2 + h_imag**2).mean(dim=-1)
        wave_energy = (h_real**2 + h_imag**2).mean(dim=1)
        total_energy = (h_real**2 + h_imag**2).mean(dim=(1, 2))
        if isinstance(self.det_head, OnsetAwareDetHead):
            raw_feats = self.raw_onset_encoder(x) if self.raw_onset_encoder is not None and x is not None else None
            return self.det_head(wave_energy, energy_t, raw_feats)
        if isinstance(self.det_head, ScalarDetHead):
            return self.det_head(wave_energy, total_energy)
        return self.det_head(wave_energy).squeeze(-1)

    def _resample_field(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_real.size(1) == target_len:
            return h_real, h_imag, t, rho
        h_real_rs = F.interpolate(
            h_real.transpose(1, 2), size=target_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        h_imag_rs = F.interpolate(
            h_imag.transpose(1, 2), size=target_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        t_rs = F.interpolate(
            t.transpose(1, 2), size=target_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        rho_rs = F.interpolate(
            rho.transpose(1, 2), size=target_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        return h_real_rs, h_imag_rs, t_rs, rho_rs

    def _run_shared_propagation(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
        layers: Optional[nn.ModuleList] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.multi_scale_encoder is not None:
            return self.multi_scale_encoder(h_real, h_imag, t=t, rho=rho)
        use_layers = layers if layers is not None else self.shared_layers
        assert use_layers is not None
        for layer in use_layers:
            h_real, h_imag = layer(h_real, h_imag, t=t, rho=rho)
        return h_real, h_imag

    def copy_shared_det_from_pick(self) -> None:
        """Init det-only shared stack from pick shared weights after resume."""
        if getattr(self, "shared_det_layers", None) is None or self.shared_layers is None:
            return
        for det_layer, pick_layer in zip(self.shared_det_layers, self.shared_layers):
            det_layer.load_state_dict(pick_layer.state_dict(), strict=False)

    def collect_kernel_params(self) -> dict[str, dict[str, float]]:
        """Export learned Huygens kernel parameters for interpretability."""
        params: dict[str, dict[str, float]] = {}
        if self.multi_scale_encoder is not None:
            for si, branch in enumerate(self.multi_scale_encoder.branches):
                for li, layer in enumerate(branch.layers):
                    k = layer.kernel
                    params[f"scale{si}_layer{li}"] = {
                        "gamma": float(k.effective_gamma().detach().cpu()),
                        "omega": float(k.effective_omega().detach().cpu()),
                        "wave_speed": float(k.effective_wave_speed().detach().cpu()),
                    }
        elif self.shared_layers is not None:
            for i, layer in enumerate(self.shared_layers):
                k = layer.kernel
                params[f"shared_{i}"] = {
                    "gamma": float(k.effective_gamma().detach().cpu()),
                    "omega": float(k.effective_omega().detach().cpu()),
                    "wave_speed": float(k.effective_wave_speed().detach().cpu()),
                }
        for i, layer in enumerate(self.p_layers):
            k = layer.kernel
            params[f"p_branch_{i}"] = {
                "gamma": float(k.effective_gamma().detach().cpu()),
                "omega": float(k.effective_omega().detach().cpu()),
                "wave_speed": float(k.effective_wave_speed().detach().cpu()),
            }
        for i, layer in enumerate(self.s_layers):
            k = layer.kernel
            params[f"s_branch_{i}"] = {
                "gamma": float(k.effective_gamma().detach().cpu()),
                "omega": float(k.effective_omega().detach().cpu()),
                "wave_speed": float(k.effective_wave_speed().detach().cpu()),
            }
        return params

    def _apply_noise_cancel(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        if self.noise_cancel_branch is None or getattr(self, "bypass_noise_cancel", False):
            return x, x, None
        nc_out = self.noise_cancel_branch(x, t)
        x_det = nc_out["u_final"]
        x_pick = x if self.noise_det_pick_split else x_det
        return x_det, x_pick, nc_out

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        p_target: Optional[torch.Tensor] = None,
        s_target: Optional[torch.Tensor] = None,
        p_idx: Optional[torch.Tensor] = None,
        s_idx: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        sampler_w = None
        sampler_attn = None
        p_target_fine = p_target
        s_target_fine = s_target
        if self.temporal_sampler is not None:
            # Learnable equal-mass resample to sampler_out_len; keep sparse_band
            # on a uniform warped-time axis.
            samp = self.temporal_sampler(x)
            x = samp["x"]
            t = samp["t"]
            sampler_w = samp["w"]
            sampler_attn = samp["attn"]
            if p_target is not None:
                p_target = remap_sequence(sampler_attn, p_target)
            if s_target is not None:
                s_target = remap_sequence(sampler_attn, s_target)
            if p_idx is not None:
                p_idx = remap_index(sampler_attn, p_idx)
            if s_idx is not None:
                s_idx = remap_index(sampler_attn, s_idx)

        nc_out: Optional[dict[str, torch.Tensor]] = None
        x_det, x_pick, nc_out = self._apply_noise_cancel(x, t)

        rho_det = self.medium_net(x_det)
        h_det_real = self.source_embed(x_det)
        h_det_imag = torch.zeros_like(h_det_real)
        det_layers = self.shared_det_layers if getattr(self, "shared_det_layers", None) is not None else None
        h_det_real, h_det_imag = self._encode_shared_wavefield(
            h_det_real, h_det_imag, t=t, rho=rho_det, layers=det_layers
        )
        det = self._det_logits(h_det_real, h_det_imag, x=x_det)

        rho = self.medium_net(x_pick)
        h_real = self.source_embed(x_pick)
        h_imag = torch.zeros_like(h_real)
        h_real, h_imag = self._encode_shared_wavefield(h_real, h_imag, t=t, rho=rho)
        if nc_out is not None and self.noise_cue_adapter is not None:
            cue, gate = self.noise_cue_adapter(
                x,
                nc_out["n_sim"],
                nc_out["u_denoised"],
                nc_out["s_noise"],
                nc_out.get("preserve_gate"),
            )
            h_real = h_real + gate * cue

        p_seed_real = h_real
        if nc_out is not None and self.p_onset_refine is not None:
            p_hint, p_gate = self.p_onset_refine(x, nc_out.get("preserve_gate"))
            p_seed_real = p_seed_real + p_gate * p_hint

        p_real, p_imag = self._propagate(p_seed_real, h_imag, self.p_layers, t, rho)
        s_real, s_imag = self._propagate(h_real, h_imag, self.s_layers, t, rho)

        p = self.p_pick_head(p_real, p_imag)
        s = self.s_pick_head(s_real, s_imag)
        if self.p_peak_rerank is not None:
            p = self.p_peak_rerank(p)
        if self.s_peak_rerank is not None:
            s = self.s_peak_rerank(s)
        # Causal field envelopes from Huygens P/S branches (pre-pick-head physics).
        p_field_env = torch.sqrt((p_real**2 + p_imag**2).sum(dim=-1) + 1e-8)
        s_field_env = torch.sqrt((s_real**2 + s_imag**2).sum(dim=-1) + 1e-8)
        out: dict[str, torch.Tensor] = {
            "det": det,
            "p": p,
            "s": s,
            "rho": rho.squeeze(-1),
            "p_field_env": p_field_env,
            "s_field_env": s_field_env,
        }
        if self.p_exist_head is not None and self.s_exist_head is not None:
            if self.parallel_exist_fuse and self.p_exist_fuse is not None and self.s_exist_fuse is not None:
                p_exist = self.p_exist_head(p_real, p_imag)
                s_exist = self.s_exist_head(s_real, s_imag)
                out["p_pick"] = p
                out["s_pick"] = s
                p, p_exist_ref = self.p_exist_fuse(p, p_exist)
                s, s_exist_ref = self.s_exist_fuse(s, s_exist)
                out["p"] = p
                out["s"] = s
                out["p_exist"] = p_exist
                out["s_exist"] = s_exist
                out["p_exist_ref"] = p_exist_ref
                out["s_exist_ref"] = s_exist_ref
            else:
                p_exist = self.p_exist_head(p_real, p_imag, p)
                s_exist = self.s_exist_head(s_real, s_imag, s)
                out["p_exist"] = p_exist
                out["s_exist"] = s_exist
        if self.p_residual_offset is not None:
            coarse = torch.sigmoid(out["p"]).argmax(dim=-1).detach()
            out["p_coarse_idx"] = coarse
            out["p_delta_bins"] = self.p_residual_offset(x_pick, coarse)
        if sampler_w is not None:
            out["sampler_w"] = sampler_w
            out["sampler_attn"] = sampler_attn
            out["x_sampled"] = x
            out["t_sampled"] = t
            if p_target_fine is not None:
                out["p_target_fine"] = p_target_fine
            if s_target_fine is not None:
                out["s_target_fine"] = s_target_fine
        if p_target is not None:
            out["p_target"] = p_target
        if s_target is not None:
            out["s_target"] = s_target
        if p_idx is not None:
            out["p_idx"] = p_idx
        if s_idx is not None:
            out["s_idx"] = s_idx
        if self.ps_gap_head is not None:
            gap_sec, gap_log_sigma = self.ps_gap_head(h_real, h_imag, rho)
            out["ps_gap_sec"] = gap_sec
            out["ps_gap_log_sigma"] = gap_log_sigma
        if nc_out is not None:
            for key, value in nc_out.items():
                out[f"nc_{key}"] = value
        return out

    def refine_p_indices(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        coarse_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optional P residual offset. Uses pick-path waveform (post NC)."""
        if self.p_residual_offset is None:
            return coarse_idx.long()
        _x_det, x_pick, _nc = self._apply_noise_cancel(x, t)
        delta = self.p_residual_offset(x_pick, coarse_idx)
        tlen = x_pick.size(1)
        return (coarse_idx.float() + delta).round().long().clamp(0, tlen - 1)

    def forward_pick_only(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """P/S + rho only — skips detection branch to save memory at inference."""
        if self.temporal_sampler is not None:
            samp = self.temporal_sampler(x)
            x = samp["x"]
            t = samp["t"]
        _x_det, x_pick, nc_out = self._apply_noise_cancel(x, t)
        rho = self.medium_net(x_pick)
        h_real = self.source_embed(x_pick)
        h_imag = torch.zeros_like(h_real)
        h_real, h_imag = self._encode_shared_wavefield(h_real, h_imag, t=t, rho=rho)
        if nc_out is not None and self.noise_cue_adapter is not None:
            cue, gate = self.noise_cue_adapter(
                x,
                nc_out["n_sim"],
                nc_out["u_denoised"],
                nc_out["s_noise"],
                nc_out.get("preserve_gate"),
            )
            h_real = h_real + gate * cue
        p_seed_real = h_real
        if nc_out is not None and self.p_onset_refine is not None:
            p_hint, p_gate = self.p_onset_refine(x, nc_out.get("preserve_gate"))
            p_seed_real = p_seed_real + p_gate * p_hint
        p_real, p_imag = self._propagate(p_seed_real, h_imag, self.p_layers, t, rho)
        s_real, s_imag = self._propagate(h_real, h_imag, self.s_layers, t, rho)
        p = self.p_pick_head(p_real, p_imag)
        s = self.s_pick_head(s_real, s_imag)
        if self.p_peak_rerank is not None:
            p = self.p_peak_rerank(p)
        if self.s_peak_rerank is not None:
            s = self.s_peak_rerank(s)
        p_field_env = torch.sqrt((p_real**2 + p_imag**2).sum(dim=-1) + 1e-8)
        s_field_env = torch.sqrt((s_real**2 + s_imag**2).sum(dim=-1) + 1e-8)
        out: dict[str, torch.Tensor] = {
            "p": p,
            "s": s,
            "rho": rho.squeeze(-1),
            "p_field_env": p_field_env,
            "s_field_env": s_field_env,
        }
        if self.p_exist_head is not None and self.s_exist_head is not None:
            if self.parallel_exist_fuse and self.p_exist_fuse is not None and self.s_exist_fuse is not None:
                p_exist = self.p_exist_head(p_real, p_imag)
                s_exist = self.s_exist_head(s_real, s_imag)
                out["p_pick"] = p
                out["s_pick"] = s
                p, p_exist_ref = self.p_exist_fuse(p, p_exist)
                s, s_exist_ref = self.s_exist_fuse(s, s_exist)
                out["p"] = p
                out["s"] = s
                out["p_exist"] = p_exist
                out["s_exist"] = s_exist
                out["p_exist_ref"] = p_exist_ref
                out["s_exist_ref"] = s_exist_ref
            else:
                out["p_exist"] = self.p_exist_head(p_real, p_imag, p)
                out["s_exist"] = self.s_exist_head(s_real, s_imag, s)
        if self.p_residual_offset is not None:
            coarse = torch.sigmoid(out["p"]).argmax(dim=-1).detach()
            out["p_coarse_idx"] = coarse
            out["p_delta_bins"] = self.p_residual_offset(x_pick, coarse)
        if self.ps_gap_head is not None:
            gap_sec, gap_log_sigma = self.ps_gap_head(h_real, h_imag, rho)
            out["ps_gap_sec"] = gap_sec
            out["ps_gap_log_sigma"] = gap_log_sigma
        return out

    @torch.no_grad()
    def forward_inversion_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        include_picks: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Export latent features for the inversion Physics Head.

        rho and kernel wave_speed are uncalibrated latents — not physical units.
        """
        was_training = self.training
        self.eval()
        if t.dim() == 2:
            t = t.unsqueeze(0).expand(x.shape[0], -1, -1)
        _x_det, x_pick, nc_out = self._apply_noise_cancel(x, t)
        rho = self.medium_net(x_pick)
        h_real = self.source_embed(x_pick)
        h_imag = torch.zeros_like(h_real)
        h_real, h_imag = self._encode_shared_wavefield(h_real, h_imag, t=t, rho=rho)
        if nc_out is not None and self.noise_cue_adapter is not None:
            cue, gate = self.noise_cue_adapter(
                x,
                nc_out["n_sim"],
                nc_out["u_denoised"],
                nc_out["s_noise"],
                nc_out.get("preserve_gate"),
            )
            h_real = h_real + gate * cue
        p_seed_real = h_real
        if nc_out is not None and self.p_onset_refine is not None:
            p_hint, p_gate = self.p_onset_refine(x, nc_out.get("preserve_gate"))
            p_seed_real = p_seed_real + p_gate * p_hint

        kparams = self.collect_kernel_params()
        kernel_vp = torch.tensor(
            float(kparams.get("p_branch_0", {}).get("wave_speed", 8.0)),
            device=x.device,
            dtype=h_real.dtype,
        )
        kernel_vs = torch.tensor(
            float(kparams.get("s_branch_0", {}).get("wave_speed", 4.5)),
            device=x.device,
            dtype=h_real.dtype,
        )
        from hnf.zhizi_physics_head import kernel_summary_from_params

        ksum = kernel_summary_from_params(kparams, device=x.device, dtype=h_real.dtype)
        batch = x.shape[0]
        out: dict[str, torch.Tensor] = {
            "h_real": h_real,
            "h_imag": h_imag,
            "rho": rho.squeeze(-1),
            "kernel_vp": kernel_vp.expand(batch),
            "kernel_vs": kernel_vs.expand(batch),
            "kernel_summary": ksum.unsqueeze(0).expand(batch, -1),
        }
        if include_picks:
            p_real, p_imag = self._propagate(h_real, h_imag, self.p_layers, t, rho)
            s_real, s_imag = self._propagate(h_real, h_imag, self.s_layers, t, rho)
            p = self.p_pick_head(p_real, p_imag)
            s = self.s_pick_head(s_real, s_imag)
            out["p_logits"] = p
            out["s_logits"] = s
        if was_training:
            self.train()
        return out

    def forward_explain(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        include_kernel_row: bool = False,
        kernel_row_idx: Optional[int] = None,
        kernel_branch: str = "p",
    ) -> dict[str, torch.Tensor]:
        x_det, x_pick, nc_out = self._apply_noise_cancel(x, t)
        rho_det = self.medium_net(x_det)
        h_det_real = self.source_embed(x_det)
        h_det_imag = torch.zeros_like(h_det_real)
        h_det_real, h_det_imag = self._encode_shared_wavefield(h_det_real, h_det_imag, t=t, rho=rho_det)
        det = self._det_logits(h_det_real, h_det_imag, x=x_det)

        rho = self.medium_net(x_pick)
        h_real = self.source_embed(x_pick)
        h_imag = torch.zeros_like(h_real)
        h_real, h_imag = self._encode_shared_wavefield(h_real, h_imag, t=t, rho=rho)
        if nc_out is not None and self.noise_cue_adapter is not None:
            cue, gate = self.noise_cue_adapter(
                x,
                nc_out["n_sim"],
                nc_out["u_denoised"],
                nc_out["s_noise"],
            )
            h_real = h_real + gate * cue

        p_real, p_imag = self._propagate(p_seed_real, h_imag, self.p_layers, t, rho)
        s_real, s_imag = self._propagate(h_real, h_imag, self.s_layers, t, rho)

        p_env = torch.sqrt((p_real**2 + p_imag**2).sum(dim=-1) + 1e-8)
        s_env = torch.sqrt((s_real**2 + s_imag**2).sum(dim=-1) + 1e-8)

        p = self.p_pick_head(p_real, p_imag)
        s = self.s_pick_head(s_real, s_imag)

        out: dict[str, torch.Tensor] = {
            "det": det,
            "p": p,
            "s": s,
            "rho": rho.squeeze(-1),
            "wave_energy": (h_real**2 + h_imag**2).mean(dim=-1),
            "p_envelope": p_env,
            "s_envelope": s_env,
        }
        if nc_out is not None:
            out["nc_n_sim"] = nc_out["n_sim"]
            out["nc_u_final"] = nc_out["u_final"]
            out["nc_u_denoised"] = nc_out["u_denoised"]

        if include_kernel_row and kernel_row_idx is not None:
            branch_layers = self.p_layers if kernel_branch == "p" else self.s_layers
            h_br, h_bi = h_real, h_imag
            for layer in branch_layers[:-1]:
                h_br, h_bi = layer(h_br, h_bi, t=t, rho=rho)
            k_mat = branch_layers[-1].kernel(h_br, t=t, rho=rho, return_complex=True)
            out["kernel_contrib"] = torch.abs(k_mat[:, kernel_row_idx, :])

        return out


def build_picking_model(
    *,
    input_dim: int = 3,
    embed_dim: int = 64,
    num_shared_layers: int = 2,
    num_branch_layers: int = 2,
    gamma: float = 0.5,
    omega: float = 0.3,
    vp: float = 8.0,
    vs: float = 4.5,
    local_window_sec: float = 15.0,
    dropout: float = 0.1,
    per_time_det: bool = False,
    pick_head_hidden: int = 24,
    pick_head_kernel: int = 7,
    pick_head_layers: int = 3,
    multi_scale: bool = False,
    scale_specs: Optional[list[ScaleSpec]] = None,
    sparse_band: bool = False,
    num_anchors: int = 0,
    residual_pick_head: bool = True,
    residual_det_head: bool = True,
    enhanced_det_head: bool = False,
    noise_cancel: bool = False,
    noise_source_dim: int = 16,
    noise_det_pick_split: bool = False,
    noise_pick_cues: bool = False,
    principle: str = "huygens",
    obliquity_scale: float = 1.0,
    obliquity_mode: str = "none",
    obliquity_mix: float = 0.0,
    predict_ps_gap: bool = False,
    ps_gap_hidden: int = 64,
    peak_rerank: bool = False,
    peak_rerank_hidden: int = 16,
    p_residual_offset: bool = False,
    p_residual_crop_half_bins: int = 30,
    p_residual_hidden: int = 32,
    p_residual_max_delta_bins: float = 8.0,
    causal_peak_rank: bool = False,
    causal_peak_rank_hidden: int = 48,
    causal_peak_rank_topk: int = 8,
    causal_peak_rank_crop_half: int = 16,
    phase_exist: bool = False,
    phase_exist_hidden: int = 64,
    strong_s_exist: bool = False,
    parallel_exist_fuse: bool = False,
    bayesian_mc: bool = False,
    n_samples: int = 32,
    learnable_sampler: bool = False,
    sampler_out_len: int = 800,
    sampler_hidden: int = 32,
    sampler_temperature: float = 0.05,
) -> STEADHNFPickingModel:
    """Factory for STEAD/OBS HNF picking models."""
    return STEADHNFPickingModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_shared_layers=num_shared_layers,
        num_branch_layers=num_branch_layers,
        gamma=gamma,
        omega=omega,
        vp=vp,
        vs=vs,
        local_window_sec=local_window_sec,
        dropout=dropout,
        per_time_det=per_time_det,
        pick_head_hidden=pick_head_hidden,
        pick_head_kernel=pick_head_kernel,
        pick_head_layers=pick_head_layers,
        multi_scale=multi_scale,
        scale_specs=scale_specs,
        sparse_band=sparse_band,
        num_anchors=num_anchors,
        residual_pick_head=residual_pick_head,
        residual_det_head=residual_det_head,
        enhanced_det_head=enhanced_det_head,
        noise_cancel=noise_cancel,
        noise_source_dim=noise_source_dim,
        noise_det_pick_split=noise_det_pick_split,
        noise_pick_cues=noise_pick_cues,
        principle=principle,
        obliquity_scale=obliquity_scale,
        obliquity_mode=obliquity_mode,
        obliquity_mix=obliquity_mix,
        predict_ps_gap=predict_ps_gap,
        ps_gap_hidden=ps_gap_hidden,
        peak_rerank=peak_rerank,
        peak_rerank_hidden=peak_rerank_hidden,
        p_residual_offset=p_residual_offset,
        p_residual_crop_half_bins=p_residual_crop_half_bins,
        p_residual_hidden=p_residual_hidden,
        p_residual_max_delta_bins=p_residual_max_delta_bins,
        causal_peak_rank=causal_peak_rank,
        causal_peak_rank_hidden=causal_peak_rank_hidden,
        causal_peak_rank_topk=causal_peak_rank_topk,
        causal_peak_rank_crop_half=causal_peak_rank_crop_half,
        phase_exist=phase_exist,
        phase_exist_hidden=phase_exist_hidden,
        strong_s_exist=strong_s_exist,
        parallel_exist_fuse=parallel_exist_fuse,
        bayesian_mc=bayesian_mc,
        n_samples=n_samples,
        learnable_sampler=learnable_sampler,
        sampler_out_len=sampler_out_len,
        sampler_hidden=sampler_hidden,
        sampler_temperature=sampler_temperature,
    )


def remap_legacy_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map older checkpoints onto current module names."""
    remapped = dict(state_dict)
    det_pairs = [
        ("det_head.0.weight", "det_head.mlp.0.weight"),
        ("det_head.0.bias", "det_head.mlp.0.bias"),
        ("det_head.3.weight", "det_head.mlp.3.weight"),
        ("det_head.3.bias", "det_head.mlp.3.bias"),
    ]
    for old_key, new_key in det_pairs:
        if old_key in remapped and new_key not in remapped:
            remapped[new_key] = remapped.pop(old_key)
    return remapped


def load_picking_model_state(
    model: STEADHNFPickingModel,
    state_dict: dict[str, torch.Tensor],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load checkpoint with partial match when architecture differs."""
    state_dict = remap_legacy_checkpoint(state_dict)
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    missing = list(missing) + skipped
    return list(missing), list(unexpected)
