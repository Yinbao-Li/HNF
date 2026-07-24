# -*- coding: utf-8 -*-
"""Central-fovea local processor: dense HNF on a cropped gaze window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.kernel import HuygensKernel


SUPPORTED_WINDOW_SIZES = (200, 400, 800, 1200, 1500)


@dataclass
class FoveaOutput:
    """One dense fovea evaluation on a cropped window."""

    focus_index: int
    window_start: int
    window_end: int
    window_size: int
    p_logits: torch.Tensor  # (B, W)
    s_logits: torch.Tensor  # (B, W)
    p_prob: torch.Tensor  # (B, W)
    s_prob: torch.Tensor  # (B, W)
    p_idx_local: torch.Tensor  # (B,)
    s_idx_local: torch.Tensor  # (B,)
    p_idx_global: torch.Tensor  # (B,)
    s_idx_global: torch.Tensor  # (B,)
    confidence: torch.Tensor  # (B,)
    uncertainty: torch.Tensor  # (B,)
    snr: torch.Tensor  # (B,)
    velocity_model: Optional[dict[str, torch.Tensor]] = None
    rho: Optional[torch.Tensor] = None  # (B, W)
    extras: Optional[dict[str, torch.Tensor]] = None


def _as_btc(waveform: torch.Tensor) -> torch.Tensor:
    """Accept (B,3,T) or (B,T,3) and return (B,T,3)."""
    if waveform.dim() != 3:
        raise ValueError(f"Expected 3D waveform, got {tuple(waveform.shape)}")
    if waveform.size(1) == 3:
        return waveform.transpose(1, 2).contiguous()
    if waveform.size(-1) == 3:
        return waveform.contiguous()
    raise ValueError(f"Expected channel dim=3, got shape {tuple(waveform.shape)}")


def crop_gaze_window(
    x_btc: torch.Tensor,
    focus_index: int | torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, int, int]:
    """Crop centered window with edge clamping. Returns (crop, start, end)."""
    _b, t, _c = x_btc.shape
    w = int(window_size)
    if w <= 0 or w > t:
        raise ValueError(f"window_size must be in [1, {t}], got {w}")

    if isinstance(focus_index, torch.Tensor):
        focus = int(focus_index.detach().reshape(-1)[0].item())
    else:
        focus = int(focus_index)
    focus = max(0, min(t - 1, focus))

    half = w // 2
    start = focus - half
    end = start + w
    if start < 0:
        start = 0
        end = w
    if end > t:
        end = t
        start = t - w
    return x_btc[:, start:end, :], start, end


def estimate_local_snr(x_btc: torch.Tensor) -> torch.Tensor:
    """Rough SNR from energy ratio of peak half vs quiet half. (B,)"""
    energy = (x_btc**2).mean(dim=-1)  # (B, W)
    mid = energy.size(-1) // 2
    a = energy[:, :mid].mean(dim=-1)
    b = energy[:, mid:].mean(dim=-1)
    signal = torch.maximum(a, b)
    noise = torch.minimum(a, b).clamp_min(1e-8)
    snr = 10.0 * torch.log10((signal + 1e-8) / noise)
    return snr.clamp(-20.0, 60.0)


def _soft_peak_logits(
    length: int,
    centers: torch.Tensor,
    peak_probs: torch.Tensor,
    *,
    sigma: float = 8.0,
) -> torch.Tensor:
    """Build (B, length) logits with a Gaussian bump at each center."""
    b = centers.shape[0]
    device = centers.device
    t = torch.arange(length, device=device, dtype=torch.float32).view(1, -1)
    c = centers.float().view(b, 1)
    gauss = torch.exp(-0.5 * ((t - c) / max(sigma, 1.0)) ** 2)
    amp = torch.logit(peak_probs.clamp(1e-4, 1.0 - 1e-4)).view(b, 1)
    # Background near -4 so uncovered→sigmoid≈0.02 (not 0.5).
    return -4.0 + (amp + 4.0).clamp_min(0.0) * gauss


class FoveaProcessor(nn.Module):
    """Dense local HNF processor compatible with existing HuygensKernel / picking model.

    Preferred path: wrap a pretrained ``STEADHNFPickingModel`` (and optional PhysicsDecoder).
    Fallback path: a lightweight HuygensKernel envelope head for unit tests / cold start.

    When ``align_mode="shift_downsample"`` (default with a picking model), each gaze
    recenters the full 60 s trace so the focus lands near the STEAD P-offset (~8 s),
    downsamples to ``backbone_seq_len`` (800), runs run28, then maps picks back.
    Native short crops (800 @ 100 Hz = 8 s) are **not** in the run28 training
    distribution and must not be used as the backbone input.
    """

    def __init__(
        self,
        picking_model: Optional[nn.Module] = None,
        physics_decoder: Optional[nn.Module] = None,
        *,
        seq_len: int = 6000,
        window_sec_full: float = 60.0,
        default_window_size: int = 800,
        supported_windows: Sequence[int] = SUPPORTED_WINDOW_SIZES,
        enable_inversion: bool = False,
        sample_rate_hz: Optional[float] = None,
        fallback_embed_dim: int = 16,
        backbone_seq_len: int = 800,
        align_mode: str = "auto",
        canonical_focus_sec: float = 8.0,
    ):
        super().__init__()
        self.picking_model = picking_model
        self.physics_decoder = physics_decoder
        self.seq_len = int(seq_len)
        self.window_sec_full = float(window_sec_full)
        self.default_window_size = int(default_window_size)
        self.supported_windows = tuple(int(w) for w in supported_windows)
        self.enable_inversion = bool(enable_inversion)
        self.backbone_seq_len = int(backbone_seq_len)
        self.canonical_focus_sec = float(canonical_focus_sec)
        if sample_rate_hz is None:
            sample_rate_hz = (self.seq_len - 1) / self.window_sec_full
        self.sample_rate_hz = float(sample_rate_hz)
        if align_mode == "auto":
            align_mode = (
                "shift_downsample"
                if picking_model is not None and self.seq_len != self.backbone_seq_len
                else "native_crop"
            )
        self.align_mode = align_mode

        self.fallback_kernel = HuygensKernel(
            gamma=0.5,
            omega=0.8,
            wave_speed=6.0,
            distance_mode="time",
            local_window_sec=8.0,
            sparse_band=True,
            use_complex=True,
            principle="huygens",
        )
        self.fallback_proj = nn.Conv1d(3, fallback_embed_dim, kernel_size=1)
        self.fallback_p_head = nn.Conv1d(fallback_embed_dim, 1, kernel_size=5, padding=2)
        self.fallback_s_head = nn.Conv1d(fallback_embed_dim, 1, kernel_size=5, padding=2)

    def validate_window_size(self, window_size: int) -> int:
        w = int(window_size)
        if w not in self.supported_windows:
            w = min(self.supported_windows, key=lambda s: abs(s - w))
        return w

    def _fallback_forward(
        self,
        crop: torch.Tensor,
        t_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.fallback_proj(crop.transpose(1, 2)).transpose(1, 2)
        h_c = torch.complex(h, torch.zeros_like(h))
        y = self.fallback_kernel.forward_apply(h_c, x=h, t=t_local)
        y_abs = torch.abs(y)
        p = self.fallback_p_head(y_abs.transpose(1, 2)).squeeze(1)
        s = self.fallback_s_head(y_abs.transpose(1, 2)).squeeze(1)
        rho = y_abs.mean(dim=-1)
        return p, s, rho

    def _picking_forward(
        self,
        crop: torch.Tensor,
        t_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict[str, torch.Tensor]]:
        assert self.picking_model is not None
        out = self.picking_model(crop, t_local)
        p = out["p"]
        s = out["s"]
        rho = out.get("rho")
        extras = {k: v for k, v in out.items() if k not in {"p", "s", "rho"}}
        return p, s, rho, extras

    def _shift_downsample_forward(
        self,
        x_btc: torch.Tensor,
        focus_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run backbone on a STEAD-aligned 800-pt view; return global P/S idx + peaks."""
        assert self.picking_model is not None
        b, t_native, _c = x_btc.shape
        device, dtype = x_btc.device, x_btc.dtype
        model_len = self.backbone_seq_len
        scale = t_native / float(model_len)
        target_focus = int(
            round(self.canonical_focus_sec / self.window_sec_full * (model_len - 1))
        )
        shift_native = target_focus * scale - float(focus_index)
        shift_i = int(round(shift_native))

        x_shift = torch.roll(x_btc, shifts=shift_i, dims=1)
        x800 = F.interpolate(
            x_shift.transpose(1, 2), size=model_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        t800 = (
            torch.linspace(0.0, self.window_sec_full, model_len, device=device, dtype=dtype)
            .view(1, model_len, 1)
            .expand(b, -1, -1)
        )

        out = self.picking_model(x800, t800)
        p_prob = torch.sigmoid(out["p"])
        s_prob = torch.sigmoid(out["s"])
        p_i8 = p_prob.argmax(dim=-1)
        s_i8 = s_prob.argmax(dim=-1)

        def map_back(idx800: torch.Tensor) -> torch.Tensor:
            idx_native_shift = idx800.float() * scale
            return torch.round(idx_native_shift - shift_i).long() % t_native

        p_global = map_back(p_i8)
        s_global = map_back(s_i8)
        extras = {k: v for k, v in out.items() if k not in {"p", "s", "rho"}}
        extras["p_peak"] = p_prob.amax(dim=-1)
        extras["s_peak"] = s_prob.amax(dim=-1)
        return p_global, s_global, extras["p_peak"], extras["s_peak"], extras

    def _maybe_invert(
        self,
        crop: torch.Tensor,
        t_local: torch.Tensor,
    ) -> Optional[dict[str, torch.Tensor]]:
        if not self.enable_inversion or self.physics_decoder is None:
            return None
        backbone = getattr(self.physics_decoder, "backbone", self.picking_model)
        if backbone is None or not hasattr(backbone, "forward_inversion_features"):
            return None
        feat = backbone.forward_inversion_features(crop, t_local, include_picks=True)
        from hnf.physics_decoder import features_to_head_inputs

        n_layers = int(getattr(self.physics_decoder, "n_layers", 5))
        ws, rl, vl, pt, ksum = features_to_head_inputs(
            feat,
            n_layers=n_layers,
            window_sec=float(crop.size(1) - 1) / self.sample_rate_hz,
        )
        head = self.physics_decoder.physics_head
        head_out = head(ws, rl, vl, pt, kernel_summary=ksum)
        return {
            "vp": head_out.vp,
            "vs": head_out.vs,
            "vp_prior": head_out.vp_prior,
            "vs_prior": head_out.vs_prior,
        }

    def forward(
        self,
        waveform: torch.Tensor,
        focus_index: int | torch.Tensor,
        window_size: Optional[int] = None,
    ) -> FoveaOutput:
        """
        Args:
            waveform: (B, 3, T) or (B, T, 3), T==seq_len (default 6000)
            focus_index: gaze center (sample index)
            window_size: coverage aperture one of {200,400,800,1200,1500}
        """
        x = _as_btc(waveform)
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected T={self.seq_len}, got T={x.size(1)}")
        w = self.validate_window_size(
            self.default_window_size if window_size is None else int(window_size)
        )
        focus = (
            int(focus_index)
            if not isinstance(focus_index, torch.Tensor)
            else int(focus_index.detach().reshape(-1)[0].item())
        )
        crop, start, end = crop_gaze_window(x, focus, w)
        b = crop.size(0)
        device, dtype = crop.device, crop.dtype

        extras: dict[str, torch.Tensor] = {}
        rho: Optional[torch.Tensor] = None
        if self.picking_model is not None and self.align_mode == "shift_downsample":
            p_global, s_global, p_peak, s_peak, extras = self._shift_downsample_forward(
                x, focus
            )
            p_local = (p_global - start).clamp(0, w - 1)
            s_local = (s_global - start).clamp(0, w - 1)
            p_logits = _soft_peak_logits(w, p_local, p_peak)
            s_logits = _soft_peak_logits(w, s_local, s_peak)
            p_idx_local = p_local
            s_idx_local = s_local
            p_idx_global = p_global
            s_idx_global = s_global
            peak = 0.5 * (p_peak + s_peak)
        elif self.picking_model is not None:
            t_full = torch.linspace(
                0.0, self.window_sec_full, self.seq_len, device=device, dtype=dtype
            )
            t_local = t_full[start:end].view(1, w, 1).expand(b, -1, -1)
            p_logits, s_logits, rho, extras = self._picking_forward(crop, t_local)
            p_prob_tmp = torch.sigmoid(p_logits)
            s_prob_tmp = torch.sigmoid(s_logits)
            p_idx_local = p_prob_tmp.argmax(dim=-1)
            s_idx_local = s_prob_tmp.argmax(dim=-1)
            p_idx_global = p_idx_local + start
            s_idx_global = s_idx_local + start
            peak = 0.5 * (p_prob_tmp.amax(dim=-1) + s_prob_tmp.amax(dim=-1))
        else:
            t_full = torch.linspace(
                0.0, self.window_sec_full, self.seq_len, device=device, dtype=dtype
            )
            t_local = t_full[start:end].view(1, w, 1).expand(b, -1, -1)
            p_logits, s_logits, rho = self._fallback_forward(crop, t_local)
            p_prob_tmp = torch.sigmoid(p_logits)
            s_prob_tmp = torch.sigmoid(s_logits)
            p_idx_local = p_prob_tmp.argmax(dim=-1)
            s_idx_local = s_prob_tmp.argmax(dim=-1)
            p_idx_global = p_idx_local + start
            s_idx_global = s_idx_local + start
            peak = 0.5 * (p_prob_tmp.amax(dim=-1) + s_prob_tmp.amax(dim=-1))

        p_prob = torch.sigmoid(p_logits)
        s_prob = torch.sigmoid(s_logits)
        snr = estimate_local_snr(crop)
        snr_term = torch.sigmoid((snr - 5.0) / 5.0)
        confidence = (0.7 * peak + 0.3 * snr_term).clamp(0.0, 1.0)
        p_ent = -(p_prob * (p_prob.clamp_min(1e-6).log())).mean(dim=-1)
        s_ent = -(s_prob * (s_prob.clamp_min(1e-6).log())).mean(dim=-1)
        uncertainty = (0.5 * (p_ent + s_ent) + (1.0 - snr_term)).clamp(0.0, 5.0)

        velocity = None
        if self.align_mode != "shift_downsample":
            t_full = torch.linspace(
                0.0, self.window_sec_full, self.seq_len, device=device, dtype=dtype
            )
            t_local = t_full[start:end].view(1, w, 1).expand(b, -1, -1)
            velocity = self._maybe_invert(crop, t_local)

        return FoveaOutput(
            focus_index=focus,
            window_start=start,
            window_end=end,
            window_size=w,
            p_logits=p_logits,
            s_logits=s_logits,
            p_prob=p_prob,
            s_prob=s_prob,
            p_idx_local=p_idx_local,
            s_idx_local=s_idx_local,
            p_idx_global=p_idx_global,
            s_idx_global=s_idx_global,
            confidence=confidence,
            uncertainty=uncertainty,
            snr=snr,
            velocity_model=velocity,
            rho=rho,
            extras=extras or None,
        )
