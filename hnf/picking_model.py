# -*- coding: utf-8 -*-
"""Physics-grounded Huygens picking model for STEAD."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.layers import HuygensWaveBlock
from hnf.multiscale import MultiScaleHuygensEncoder, ScaleSpec, default_scale_specs
from hnf.noise_cancel import HuygensNoiseCancelBranch


class ComponentSecondarySources(nn.Module):
    """E/N/Z 三分量作为耦合次波源."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.cross_comp = nn.Parameter(torch.eye(3) * 0.6 + torch.ones(3, 3) / 15.0)
        self.chan_proj = nn.Conv1d(3, embed_dim, kernel_size=1)

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
    """Compress denoise outputs into lightweight cues for P/S refinement."""

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_energy = torch.sqrt((s_noise**2).mean(dim=-1, keepdim=True) + 1e-8)
        feats = torch.cat([x_raw, n_sim, u_denoised, noise_energy], dim=-1)
        cue = self.net(feats.transpose(1, 2)).transpose(1, 2)
        gate = self.gate(cue.transpose(1, 2)).transpose(1, 2)
        return cue, gate


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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.per_time_det = per_time_det
        self.enhanced_det_head = enhanced_det_head
        self.noise_cancel = noise_cancel
        self.noise_det_pick_split = noise_det_pick_split
        self.noise_pick_cues = noise_pick_cues
        self.multi_scale = multi_scale
        self.num_anchors = max(0, int(num_anchors))
        self.principle = principle
        self.source_embed = ComponentSecondarySources(embed_dim)
        self.medium_net = TemporalMediumDensity(channels=input_dim)
        self.dropout = nn.Dropout(dropout)

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
            )
            self.shared_layers = None
        else:
            self.multi_scale_encoder = None
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
                        principle=principle,
                        obliquity_scale=obliquity_scale,
                    )
                    for i in range(num_shared_layers)
                ]
            )

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
                    principle=principle,
                    obliquity_scale=obliquity_scale,
                )
                for i in range(num_branch_layers)
            ]
        )
        self.s_layers = nn.ModuleList(
            [
                HuygensWaveBlock(
                    dim=embed_dim,
                    gamma=gamma,
                    omega=omega_s * (1.03 ** i),
                    wave_speed=vs,
                    distance_mode="time",
                    local_window_sec=local_window_sec,
                    learnable_kernel_params=True,
                    dropout=dropout,
                    sparse_band=sparse_band,
                    principle=principle,
                    obliquity_scale=obliquity_scale,
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
        if noise_cancel:
            self.noise_cancel_branch = HuygensNoiseCancelBranch(
                channels=input_dim,
                source_dim=noise_source_dim,
                hidden=max(16, pick_head_hidden // 2),
                gamma=gamma,
                omega=omega,
                wave_speed=6.0,
                local_window_sec=local_window_sec,
                learnable_kernel_params=True,
                principle=principle,
                obliquity_scale=obliquity_scale,
            )
            if noise_pick_cues:
                self.noise_cue_adapter = NoiseCueAdapter(
                    input_dim=input_dim,
                    hidden=max(16, pick_head_hidden // 2),
                    embed_dim=embed_dim,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_n = h_real.size(1)
        if self.num_anchors > 0 and self.num_anchors < full_n:
            h_real, h_imag, t_shared, rho_shared = self._resample_field(
                h_real, h_imag, t, rho, self.num_anchors
            )
        else:
            t_shared, rho_shared = t, rho

        h_real, h_imag = self._run_shared_propagation(
            h_real, h_imag, t=t_shared, rho=rho_shared
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.multi_scale_encoder is not None:
            return self.multi_scale_encoder(h_real, h_imag, t=t, rho=rho)
        assert self.shared_layers is not None
        for layer in self.shared_layers:
            h_real, h_imag = layer(h_real, h_imag, t=t, rho=rho)
        return h_real, h_imag

    def collect_kernel_params(self) -> dict[str, dict[str, float]]:
        """Export learned Huygens kernel parameters for interpretability."""
        params: dict[str, dict[str, float]] = {}
        if self.multi_scale_encoder is not None:
            for si, branch in enumerate(self.multi_scale_encoder.branches):
                for li, layer in enumerate(branch.layers):
                    k = layer.kernel
                    params[f"scale{si}_layer{li}"] = {
                        "gamma": float(k.effective_gamma().detach().cpu()),
                        "omega": float(k.omega.detach().cpu()),
                        "wave_speed": float(k.effective_wave_speed().detach().cpu()),
                    }
        elif self.shared_layers is not None:
            for i, layer in enumerate(self.shared_layers):
                k = layer.kernel
                params[f"shared_{i}"] = {
                    "gamma": float(k.effective_gamma().detach().cpu()),
                    "omega": float(k.omega.detach().cpu()),
                    "wave_speed": float(k.effective_wave_speed().detach().cpu()),
                }
        for i, layer in enumerate(self.p_layers):
            k = layer.kernel
            params[f"p_branch_{i}"] = {
                "gamma": float(k.effective_gamma().detach().cpu()),
                "omega": float(k.omega.detach().cpu()),
                "wave_speed": float(k.effective_wave_speed().detach().cpu()),
            }
        for i, layer in enumerate(self.s_layers):
            k = layer.kernel
            params[f"s_branch_{i}"] = {
                "gamma": float(k.effective_gamma().detach().cpu()),
                "omega": float(k.omega.detach().cpu()),
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        nc_out: Optional[dict[str, torch.Tensor]] = None
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

        p_real, p_imag = self._propagate(h_real, h_imag, self.p_layers, t, rho)
        s_real, s_imag = self._propagate(h_real, h_imag, self.s_layers, t, rho)

        p = self.p_pick_head(p_real, p_imag)
        s = self.s_pick_head(s_real, s_imag)
        out: dict[str, torch.Tensor] = {"det": det, "p": p, "s": s, "rho": rho.squeeze(-1)}
        if nc_out is not None:
            for key, value in nc_out.items():
                out[f"nc_{key}"] = value
        return out

    def forward_pick_only(self, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """P/S + rho only — skips detection branch to save memory at inference."""
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
            )
            h_real = h_real + gate * cue
        p_real, p_imag = self._propagate(h_real, h_imag, self.p_layers, t, rho)
        s_real, s_imag = self._propagate(h_real, h_imag, self.s_layers, t, rho)
        p = self.p_pick_head(p_real, p_imag)
        s = self.s_pick_head(s_real, s_imag)
        return {"p": p, "s": s, "rho": rho.squeeze(-1)}

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
            )
            h_real = h_real + gate * cue

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
        batch = x.shape[0]
        out: dict[str, torch.Tensor] = {
            "h_real": h_real,
            "h_imag": h_imag,
            "rho": rho.squeeze(-1),
            "kernel_vp": kernel_vp.expand(batch),
            "kernel_vs": kernel_vs.expand(batch),
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

        p_real, p_imag = self._propagate(h_real, h_imag, self.p_layers, t, rho)
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
) -> STEADHNFPickingModel:
    """Factory for STEAD HNF picking models."""
    return STEADHNFPickingModel(
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
