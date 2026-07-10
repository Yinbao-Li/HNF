# -*- coding: utf-8 -*-
"""
Zhizi (智子) inversion bridge: Physics Head maps latent picking features to 1D Earth.

Latent quantities (rho, kernel wave_speed) are NOT physical units — they condition
the head and regularization only. Calibration happens via travel-time physics loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.inversion_1d import LayeredEarth1D, default_synth_model, travel_time_phase


@dataclass
class PhysicsHeadOutput:
    vp: torch.Tensor
    vs: torch.Tensor
    vp_prior: torch.Tensor
    vs_prior: torch.Tensor
    logit_vs_ratio: torch.Tensor
    q: torch.Tensor | None = None


def bucket_rho_to_layers(
    rho: torch.Tensor,
    n_layers: int,
    window_sec: float = 60.0,
) -> torch.Tensor:
    """Map rho(t) (B, T) to per-layer summary (B, n_layers)."""
    b, n_time = rho.shape
    dev = rho.device
    edges = torch.linspace(0.0, window_sec, n_layers + 1, device=dev)
    out = []
    for i in range(n_layers):
        lo, hi = edges[i], edges[i + 1]
        n_time = int(rho.shape[-1])
        i0 = int(round(float(lo) / window_sec * (n_time - 1)))
        i1 = int(round(float(hi) / window_sec * (n_time - 1)))
        i0 = max(0, min(n_time - 1, i0))
        i1 = max(i0 + 1, min(n_time, i1))
        out.append(rho[:, i0:i1].mean(dim=1))
    return torch.stack(out, dim=1)


def pool_wavefield_features(
    h_real: torch.Tensor,
    h_imag: torch.Tensor,
) -> torch.Tensor:
    """Pool shared wavefield to (B, 2C): per-channel mean/std of envelope."""
    env = torch.sqrt((h_real ** 2 + h_imag ** 2) + 1e-8)
    mean_c = env.mean(dim=1)
    std_c = env.std(dim=1)
    return torch.cat([mean_c, std_c], dim=-1)


def latent_velocity_prior(
    kernel_vp: torch.Tensor,
    kernel_vs: torch.Tensor,
    n_layers: int,
    vp_ref: float = 4.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build uncalibrated vp/vs prior from kernel latents (B,).

    kernel values are dimensionless; only ratio and relative scale matter.
    """
    if kernel_vp.dim() == 0:
        kernel_vp = kernel_vp.unsqueeze(0)
    if kernel_vs.dim() == 0:
        kernel_vs = kernel_vs.unsqueeze(0)
    b = kernel_vp.shape[0]
    ratio = (kernel_vs / kernel_vp.clamp(min=1e-6)).clamp(0.35, 0.75)
    scale = (kernel_vp / 8.0).clamp(0.6, 1.4)
    vp0 = (vp_ref * scale).view(b, 1).expand(b, n_layers)
    vs0 = vp0 * ratio.view(b, 1)
    return vp0, vs0


def reference_layered_velocity(
    n_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical 1D reference model, interpolated if layer count differs."""
    ref = default_synth_model("cpu")
    vp_ref = ref.vp.to(device=device, dtype=dtype)
    vs_ref = ref.vs.to(device=device, dtype=dtype)
    if n_layers == ref.n_layers:
        return vp_ref, vs_ref
    vp = F.interpolate(vp_ref.view(1, 1, -1), size=n_layers, mode="linear", align_corners=True).view(-1)
    vs = F.interpolate(vs_ref.view(1, 1, -1), size=n_layers, mode="linear", align_corners=True).view(-1)
    return vp, vs


def monotonic_vp_from_raw(
    raw_base: torch.Tensor,
    raw_inc: torch.Tensor,
    vp_min: float = 1.5,
    inc_min: float = 0.05,
) -> torch.Tensor:
    """raw_base (B,), raw_inc (B, n-1) -> vp (B, n)."""
    base = F.softplus(raw_base) + vp_min
    inc = F.softplus(raw_inc) + inc_min
    vp = torch.cat([base.unsqueeze(1), base.unsqueeze(1) + torch.cumsum(inc, dim=1)], dim=1)
    return vp


class ZhiziPhysicsHead(nn.Module):
    """
    Map pooled Zhizi latent features to monotonic layered vp/vs.

    Modes:
      - residual: per-layer residual shifts around reference model
      - macro: low-dimensional controls (scale, depth-contrast, Vs ratio)
        relative to the canonical reference model; zero-init = reference Earth
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_layers: int = 5,
        hidden: int = 48,
        use_pick_times: bool = True,
        mode: str = "residual",
        geo_dim: int = 0,
        predict_q: bool = False,
    ):
        super().__init__()
        if mode not in {"residual", "macro"}:
            raise ValueError(f"Unknown physics head mode: {mode}")
        self.n_layers = n_layers
        self.use_pick_times = use_pick_times
        self.mode = mode
        self.geo_dim = int(geo_dim)
        self.predict_q = bool(predict_q)
        in_dim = 2 * embed_dim + n_layers + 2 + (2 if use_pick_times else 0) + self.geo_dim
        self.base_shift_max = 0.75
        self.inc_shift_max = 0.40
        self.ratio_shift_max = 0.10
        self.macro_scale_max = 0.20
        self.macro_contrast_max = 0.35
        self.macro_ratio_max = 0.08
        self.macro_q_max = 0.35

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        if mode == "macro":
            out_dim = 3 + (1 if self.predict_q else 0)  # scale, contrast, ratio [, q_scale]
        else:
            n_inc = max(1, n_layers - 1)
            out_dim = 1 + n_inc + n_layers + (1 if self.predict_q else 0)
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        wave_stats: torch.Tensor,
        rho_layers: torch.Tensor,
        v_latent: torch.Tensor,
        pick_times: torch.Tensor | None = None,
        geo: torch.Tensor | None = None,
    ) -> PhysicsHeadOutput:
        parts = [wave_stats, rho_layers, v_latent]
        if self.use_pick_times:
            if pick_times is None:
                pick_times = torch.zeros(wave_stats.shape[0], 2, device=wave_stats.device)
            parts.append(pick_times)
        if self.geo_dim > 0:
            if geo is None:
                geo = torch.zeros(wave_stats.shape[0], self.geo_dim, device=wave_stats.device)
            parts.append(geo)
        h = self.trunk(torch.cat(parts, dim=-1))
        raw = self.out(h)

        b = wave_stats.shape[0]
        ref_vp, ref_vs = reference_layered_velocity(self.n_layers, wave_stats.device, wave_stats.dtype)
        vp_base = ref_vp.unsqueeze(0).expand(b, -1)
        vs_base = ref_vs.unsqueeze(0).expand(b, -1)
        ref_ratio = (vs_base / vp_base).clamp(0.35, 0.75)

        kernel_scale = 1.0 + 0.08 * torch.tanh((v_latent[:, 0] - 8.0) / 2.0)
        scaled_vp_base = vp_base * kernel_scale.unsqueeze(1)
        scaled_vs_base = vs_base * kernel_scale.unsqueeze(1)

        q_out = None
        if self.mode == "macro":
            scale = 1.0 + self.macro_scale_max * torch.tanh(raw[:, 0:1])
            contrast = 1.0 + self.macro_contrast_max * torch.tanh(raw[:, 1:2])
            mean_vp = scaled_vp_base.mean(dim=1, keepdim=True)
            vp = (mean_vp * scale + (scaled_vp_base - mean_vp) * contrast).clamp(min=1.5)
            vp_parts = [vp[:, :1]]
            for i in range(1, vp.shape[1]):
                vp_parts.append(torch.maximum(vp[:, i : i + 1], vp_parts[-1] + 0.05))
            vp = torch.cat(vp_parts, dim=1)
            ratio_logits = raw[:, 2:3].expand(b, self.n_layers)
            ratio = (ref_ratio + self.macro_ratio_max * torch.tanh(ratio_logits)).clamp(0.35, 0.75)
            vs = vp * ratio
            if self.predict_q:
                # global Q scale around a canonical crustal Q (~120)
                q_scale = 1.0 + self.macro_q_max * torch.tanh(raw[:, 3:4])
                q_out = (120.0 * q_scale).expand(b, self.n_layers).clamp(40.0, 400.0)
            return PhysicsHeadOutput(
                vp=vp,
                vs=vs,
                vp_prior=scaled_vp_base,
                vs_prior=scaled_vs_base,
                logit_vs_ratio=ratio_logits,
                q=q_out,
            )

        n_inc = max(1, self.n_layers - 1)
        raw_base = raw[:, 0]
        raw_inc = raw[:, 1 : 1 + n_inc]
        ratio_logits = raw[:, 1 + n_inc : 1 + n_inc + self.n_layers]
        ref_inc = (ref_vp[1:] - ref_vp[:-1]).unsqueeze(0).expand(b, -1)
        scaled_ref_inc = ref_inc * kernel_scale.unsqueeze(1)

        base0 = scaled_vp_base[:, :1] + self.base_shift_max * torch.tanh(raw_base).unsqueeze(1)
        inc = (scaled_ref_inc + self.inc_shift_max * torch.tanh(raw_inc)).clamp(min=0.05)
        vp = torch.cat([base0, base0 + torch.cumsum(inc, dim=1)], dim=1).clamp(min=1.5)

        ratio = (ref_ratio + self.ratio_shift_max * torch.tanh(ratio_logits)).clamp(0.35, 0.75)
        vs = vp * ratio
        if self.predict_q:
            q_logit = raw[:, -1:]
            q_scale = 1.0 + self.macro_q_max * torch.tanh(q_logit)
            q_out = (120.0 * q_scale).expand(b, self.n_layers).clamp(40.0, 400.0)
        return PhysicsHeadOutput(
            vp=vp,
            vs=vs,
            vp_prior=scaled_vp_base,
            vs_prior=scaled_vs_base,
            logit_vs_ratio=ratio_logits,
            q=q_out,
        )

    def earth(
        self,
        output: PhysicsHeadOutput,
        depths: torch.Tensor,
        q: torch.Tensor,
    ) -> LayeredEarth1D:
        q_use = output.q if output.q is not None else q
        if output.q is not None and output.q.dim() > 1:
            q_use = output.q[0]
        return LayeredEarth1D(
            depths=depths,
            vp=output.vp[0] if output.vp.dim() > 1 else output.vp,
            vs=output.vs[0] if output.vs.dim() > 1 else output.vs,
            q=q_use,
        )


def stack_pooled_features(
    wave_stats_list: list[torch.Tensor],
    rho_layers_list: list[torch.Tensor],
    v_latent_list: list[torch.Tensor],
    pick_times_list: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Mean-pool features from multiple stations of one event."""
    wave_stats = torch.stack(wave_stats_list, dim=0).mean(dim=0)
    rho_layers = torch.stack(rho_layers_list, dim=0).mean(dim=0)
    v_latent = torch.stack(v_latent_list, dim=0).mean(dim=0)
    pick_times = None
    if pick_times_list is not None:
        pick_times = torch.stack(pick_times_list, dim=0).mean(dim=0)
    return wave_stats, rho_layers, v_latent, pick_times


def count_physics_head_params(head: ZhiziPhysicsHead) -> int:
    return sum(p.numel() for p in head.parameters() if p.requires_grad)
