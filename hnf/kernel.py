# -*- coding: utf-8 -*-
"""Part 1: HuygensKernel — core Huygens kernel operator."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuygensKernel(nn.Module):
    """
    Huygens / Huygens–Fresnel kernel.

    Huygens (legacy):
      K = 1/(r^2+eps) * exp(-γ r^2) * exp(i ω r)

    Huygens–Fresnel:
      K = [i ω / (2π (r+eps))] * χ(θ) * exp(-γ r^2) * exp(i ω r)
      χ(θ) = (1 + cosθ) / 2   (Fresnel–Kirchhoff obliquity)
      cosθ ≈ (c·Δt) / sqrt((c·Δt)^2 + α||x_i-x_j||^2 + eps)
    """

    def __init__(
        self,
        gamma: float = 1.0,
        omega: float = 1.0,
        eps: float = 1e-6,
        causal: bool = True,
        wave_speed: float = 1.0,
        learnable_gamma: bool = False,
        learnable_omega: bool = False,
        learnable_wave_speed: bool = False,
        use_complex: bool = True,
        distance_mode: str = "feature",
        local_window_sec: Optional[float] = None,
        sparse_band: bool = False,
        principle: str = "huygens",
        obliquity_scale: float = 1.0,
        obliquity_mix: float = 0.0,
        learnable_obliquity: bool = False,
    ):
        super().__init__()
        if distance_mode not in {"feature", "time", "hybrid"}:
            raise ValueError(f"Unknown distance_mode: {distance_mode}")
        if principle not in {"huygens", "huygens_fresnel"}:
            raise ValueError(f"Unknown principle: {principle}")

        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

        if learnable_omega:
            self.omega = nn.Parameter(torch.tensor(omega, dtype=torch.float32))
        else:
            self.register_buffer("omega", torch.tensor(omega, dtype=torch.float32))

        if learnable_wave_speed:
            self.wave_speed = nn.Parameter(torch.tensor(wave_speed, dtype=torch.float32))
        else:
            self.register_buffer("wave_speed", torch.tensor(wave_speed, dtype=torch.float32))

        # Multiplicative scale on c (init=1). Stabilizes learning vs raw c gradients.
        self.c_log_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

        if learnable_obliquity:
            self.obliquity_scale = nn.Parameter(torch.tensor(obliquity_scale, dtype=torch.float32))
        else:
            self.register_buffer(
                "obliquity_scale", torch.tensor(obliquity_scale, dtype=torch.float32)
            )

        mix = float(max(0.0, min(1.0, obliquity_mix)))
        self.register_buffer("obliquity_mix", torch.tensor(mix, dtype=torch.float32))
        # Construction-time physical anchors for mild prior (resume-safe).
        self.register_buffer("gamma0", torch.tensor(float(gamma), dtype=torch.float32))
        self.register_buffer("omega0", torch.tensor(float(omega), dtype=torch.float32))
        self.register_buffer("wave_speed0", torch.tensor(float(wave_speed), dtype=torch.float32))

        self.eps = eps
        self.causal = causal
        self.use_complex = use_complex
        self.distance_mode = distance_mode
        self.local_window_sec = local_window_sec
        self.sparse_band = sparse_band
        self.principle = principle
        self._learnable_gamma = learnable_gamma
        self._learnable_omega = learnable_omega
        self._learnable_wave_speed = learnable_wave_speed
        self._learnable_obliquity = learnable_obliquity
        self._use_obliquity = principle == "huygens_fresnel" or mix > 0.0

    def effective_obliquity_mix(self) -> torch.Tensor:
        return self.obliquity_mix.clamp(0.0, 1.0)

    def _apply_obliquity_to_amplitude(
        self,
        amplitude: torch.Tensor,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full Fresnel χ or soft blend on Huygens amplitude."""
        if t is None:
            return amplitude
        chi = self._fresnel_obliquity(r, t=t, x=x)
        if self.principle == "huygens_fresnel":
            return amplitude * chi
        mix = self.effective_obliquity_mix()
        if float(mix.item()) <= 0.0:
            return amplitude
        return amplitude * (1.0 - mix + mix * chi)

    def effective_gamma(self) -> torch.Tensor:
        if self._learnable_gamma:
            return F.softplus(self.gamma) + 1e-3
        return self.gamma

    def effective_omega(self) -> torch.Tensor:
        """Positive angular frequency; softplus when learnable (near-id for ω≫0)."""
        if self._learnable_omega:
            return F.softplus(self.omega).clamp_min(1e-3)
        return self.omega.abs().clamp_min(1e-3)

    def effective_wave_speed(self) -> torch.Tensor:
        if self._learnable_wave_speed:
            base = F.softplus(self.wave_speed) + 1e-3
        else:
            base = self.wave_speed
        # Learnable scale layer: c = c_base * exp(clip(log_s)); init identity.
        scale = torch.exp(self.c_log_scale.clamp(-2.0, 2.0))
        return base * scale

    def effective_obliquity_scale(self) -> torch.Tensor:
        if self._learnable_obliquity:
            return F.softplus(self.obliquity_scale) + 1e-3
        return self.obliquity_scale

    def physics_prior_loss(self) -> torch.Tensor:
        """Pull effective γ/ω/c toward construction anchors (relative L2)."""
        g = self.effective_gamma()
        w = self.effective_omega()
        c = self.effective_wave_speed()
        g0 = self.gamma0.clamp_min(1e-3)
        w0 = self.omega0.clamp_min(1e-3)
        c0 = self.wave_speed0.clamp_min(1e-3)
        # When γ was learnable from init γ0, effective≈softplus(γ0); use softplus target.
        if self._learnable_gamma:
            g0 = F.softplus(self.gamma0) + 1e-3
        if self._learnable_omega:
            w0 = F.softplus(self.omega0).clamp_min(1e-3)
        if self._learnable_wave_speed:
            c0 = F.softplus(self.wave_speed0) + 1e-3
        return (
            ((g - g0) / g0).pow(2)
            + ((w - w0) / w0).pow(2)
            + ((c - c0) / c0).pow(2)
        )

    def compute_time_lag_matrix(self, t: torch.Tensor) -> torch.Tensor:
        """Temporal lag dt[i,j] = t_i - t_j (seconds). Positive => j is in the past."""
        if t.dim() == 3 and t.size(-1) == 1:
            t = t.squeeze(-1)
        return t.unsqueeze(-1) - t.unsqueeze(-2)

    def compute_distance_matrix(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pairwise distances. Self if y is None, else cross distances."""
        if y is None:
            return torch.cdist(x, x, p=2)
        return torch.cdist(x, y, p=2)

    def resolve_distance(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.distance_mode == "feature" or t is None:
            return self.compute_distance_matrix(x)

        dt = self.compute_time_lag_matrix(t)
        r_time = dt.clamp_min(0.0)
        if self.distance_mode == "time":
            r = torch.where(dt > 0, dt.clamp_min(self.eps), torch.zeros_like(dt))
            return r

        r_feat = self.compute_distance_matrix(x)
        feat_scale = r_feat.detach().mean().clamp_min(1e-3)
        time_scale = r_time.detach().mean().clamp_min(1e-3)
        return 0.5 * (r_time / time_scale + r_feat / feat_scale)

    def compute_causal_mask(
        self,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Light-cone mask: only past secondary sources can contribute."""
        if not self.causal or t is None:
            return torch.ones_like(r)

        dt = self.compute_time_lag_matrix(t)
        mask = (dt > 0).float()
        wave_speed = self.effective_wave_speed()
        if self.distance_mode in {"time", "hybrid"}:
            mask = mask * (r <= wave_speed * dt + 1e-6).float()
            if self.local_window_sec is not None:
                mask = mask * (dt <= self.local_window_sec + 1e-6).float()
        else:
            mask = mask * (r <= wave_speed * dt + 1e-6).float()
        return mask

    def _fresnel_obliquity(
        self,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        χ = (1 + cosθ)/2 with soft aperture geometry.
        cosθ ≈ axial / hypot(axial, lateral), axial = c·Δt.
        """
        if t is None:
            return torch.ones_like(r)

        dt = self.compute_time_lag_matrix(t).clamp_min(0.0)
        c = self.effective_wave_speed()
        axial = c * dt
        alpha = self.effective_obliquity_scale()
        if x is not None and self.distance_mode in {"feature", "hybrid"}:
            lateral2 = (alpha * self.compute_distance_matrix(x)) ** 2
        else:
            # Time-axis: soft lateral grows with lag (memory-light vs full feature cdist).
            lateral2 = (alpha ** 2) * (r.clamp_min(0.0) + self.eps)

        denom = torch.sqrt(axial ** 2 + lateral2 + self.eps)
        cos_theta = (axial / denom).clamp(-1.0, 1.0)
        return 0.5 * (1.0 + cos_theta)

    def _spherical_amplitude_mag(self, r: torch.Tensor) -> torch.Tensor:
        """Real non-negative spherical factor (Fresnel iω phase is applied separately)."""
        if self.principle == "huygens_fresnel":
            return (self.effective_omega() / (2.0 * math.pi)) / (r + self.eps)
        return 1.0 / (r ** 2 + self.eps)

    def _build_kernel(
        self,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        amplitude = self._spherical_amplitude_mag(r)

        if self.principle == "huygens_fresnel" or float(self.obliquity_mix.item()) > 0.0:
            amplitude = self._apply_obliquity_to_amplitude(amplitude, r, t=t, x=x)

        if rho is not None:
            if rho.dim() == 3 and rho.size(-1) == 1:
                rho = rho.squeeze(-1)
            rho_mean = (rho.unsqueeze(-1) + rho.unsqueeze(-2)) / 2.0
            amplitude = amplitude * torch.exp(-rho_mean * r)

        envelope = torch.exp(-self.effective_gamma() * r ** 2)
        amp = amplitude * envelope

        omega = self.effective_omega()
        if self.use_complex:
            # Fresnel–Kirchhoff carries an extra factor of i (= +π/2 phase)
            phase = torch.exp(1j * omega * r)
            if self.principle == "huygens_fresnel":
                phase = (1j) * phase
            k = amp.to(phase.dtype) * phase
        else:
            phase = torch.cos(omega * r)
            k = amp * phase

        if mask is None and t is not None:
            mask = self.compute_causal_mask(r, t)
        if mask is not None:
            k = k * mask
        return k

    def dt_step_sec(self, seq_len: int, t: Optional[torch.Tensor] = None) -> float:
        """Seconds between adjacent samples (matches torch.linspace(0, 60, N))."""
        if t is not None and t.size(1) >= 2:
            t0 = t[0, 0, 0] if t.dim() == 3 else t[0, 0]
            t1 = t[0, 1, 0] if t.dim() == 3 else t[0, 1]
            return float((t1 - t0).abs().item())
        return 60.0 / max(seq_len - 1, 1)

    def window_bins(self, seq_len: int, t: Optional[torch.Tensor] = None) -> int:
        """Max temporal lag (in samples) inside the causal local light cone."""
        if self.local_window_sec is None:
            return max(1, seq_len - 1)
        dt_step = self.dt_step_sec(seq_len, t)
        w = int(math.floor(self.local_window_sec / dt_step + 1e-9))
        return min(max(1, seq_len - 1), max(1, w))

    def _kernel_coeffs_for_lag(
        self,
        lag_sec: float,
        rho_i: Optional[torch.Tensor],
        rho_j: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Complex kernel K(i, i-d) as a function of lag only (uniform time grid)."""
        r = max(lag_sec, float(self.eps))
        amp = self._spherical_amplitude_mag(
            torch.tensor(r, device=self.gamma.device, dtype=torch.float32)
        )
        if self.principle == "huygens_fresnel":
            c = float(self.effective_wave_speed().detach())
            axial = max(c * r, 0.0)
            alpha = float(self.effective_obliquity_scale().detach())
            lateral = alpha * math.sqrt(r + float(self.eps))
            cos_theta = axial / math.sqrt(axial ** 2 + lateral ** 2 + float(self.eps))
            amp = amp * (0.5 * (1.0 + cos_theta))
        elif float(self.obliquity_mix.item()) > 0.0:
            c = float(self.effective_wave_speed().detach())
            axial = max(c * r, 0.0)
            alpha = float(self.effective_obliquity_scale().detach())
            lateral = alpha * math.sqrt(r + float(self.eps))
            cos_theta = axial / math.sqrt(axial ** 2 + lateral ** 2 + float(self.eps))
            chi = 0.5 * (1.0 + cos_theta)
            mix = float(self.obliquity_mix.item())
            amp = amp * (1.0 - mix + mix * chi)
        if rho_i is not None and rho_j is not None:
            rho_mean = (rho_i + rho_j) / 2.0
            amp = amp * torch.exp(-rho_mean * r)
        envelope = torch.exp(-self.effective_gamma() * r**2)
        amp = amp * envelope
        omega = self.effective_omega()
        if self.use_complex:
            phase = torch.exp(1j * omega * r)
            if self.principle == "huygens_fresnel":
                phase = (1j) * phase
            return amp.to(phase.dtype) * phase
        phase = torch.cos(omega * r)
        return amp * phase

    def _passes_light_cone(self, lag_sec: float) -> bool:
        if lag_sec <= 0:
            return False
        if self.local_window_sec is not None and lag_sec > self.local_window_sec + 1e-6:
            return False
        if self.causal and self.distance_mode in {"time", "hybrid"}:
            wave_speed = float(self.effective_wave_speed().detach().cpu())
            if wave_speed < 1.0 - 1e-6 and lag_sec > wave_speed * lag_sec + 1e-6:
                return False
        return True

    def forward_apply(
        self,
        h_c: torch.Tensor,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply K @ h without materializing full (N, N) when sparse_band is enabled.
        Falls back to dense matmul otherwise.
        """
        if not self.sparse_band or t is None or self.distance_mode not in {"time", "hybrid"}:
            k = self.forward(x, t=t, rho=rho, return_complex=True)
            return torch.matmul(k, h_c)

        n = h_c.size(1)
        dt_step = self.dt_step_sec(n, t)
        w_max = self.window_bins(n, t)
        b, _, d_feat = h_c.shape
        device = h_c.device
        dtype_r = torch.float32

        lags = (
            torch.arange(1, w_max + 1, device=device, dtype=dtype_r).view(w_max, 1, 1)
            * dt_step
        )
        rows = torch.arange(n, device=device)
        src = rows.view(1, n) - torch.arange(1, w_max + 1, device=device).view(w_max, 1)
        valid = (src >= 0) & (lags.squeeze(-1) <= (self.local_window_sec or 1e9) + 1e-6)
        src_clamped = src.clamp(min=0)

        amp = self._spherical_amplitude_mag(lags)
        if self.principle == "huygens_fresnel" or float(self.obliquity_mix.item()) > 0.0:
            c = self.effective_wave_speed()
            axial = (c * lags).clamp_min(0.0)
            alpha = self.effective_obliquity_scale()
            lateral2 = (alpha ** 2) * (lags.clamp_min(0.0) + self.eps)
            cos_theta = (axial / torch.sqrt(axial ** 2 + lateral2 + self.eps)).clamp(-1.0, 1.0)
            chi = 0.5 * (1.0 + cos_theta)
            if self.principle == "huygens_fresnel":
                amp = amp * chi
            else:
                mix = self.effective_obliquity_mix()
                amp = amp * (1.0 - mix + mix * chi)
        if rho is not None:
            rho_1d = rho.squeeze(-1) if rho.dim() == 3 and rho.size(-1) == 1 else rho
            rho_i = rho_1d.unsqueeze(1).expand(b, w_max, n)
            idx = src_clamped.unsqueeze(0).expand(b, w_max, n).long()
            rho_j = torch.gather(rho_1d.unsqueeze(1).expand(b, w_max, n), 2, idx)
            amp = amp.squeeze(-1).unsqueeze(0) * torch.exp(
                -(rho_i + rho_j) / 2.0 * lags.squeeze(-1).unsqueeze(0)
            )
        else:
            amp = amp.squeeze(-1).unsqueeze(0).expand(b, -1, -1)

        envelope = torch.exp(-self.effective_gamma() * lags**2).squeeze(-1)
        amp = amp * envelope.unsqueeze(0)
        omega = self.effective_omega()
        if self.use_complex:
            phase = torch.exp(1j * omega * lags).squeeze(-1)
            if self.principle == "huygens_fresnel":
                phase = (1j) * phase
            k_stack = (amp.to(phase.dtype) * phase.unsqueeze(0)) * valid.unsqueeze(0)
        else:
            phase = torch.cos(omega * lags).squeeze(-1)
            k_stack = (amp * phase.unsqueeze(0)) * valid.unsqueeze(0)

        idx_h = src_clamped.unsqueeze(0).unsqueeze(-1).expand(b, w_max, n, d_feat).long()
        h_stack = torch.gather(
            h_c.unsqueeze(1).expand(b, w_max, n, d_feat),
            2,
            idx_h,
        )
        h_stack = h_stack * valid.unsqueeze(0).unsqueeze(-1)

        return (k_stack.unsqueeze(-1) * h_stack).sum(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        return_complex: bool = True,
        regularization: float = 0.0,
    ) -> torch.Tensor:
        """K(x, x) with batch dim (B, N, D) -> (B, N, N)."""
        r = self.resolve_distance(x, t=t)
        k = self._build_kernel(r, t=t, rho=rho, x=x)

        if regularization > 0:
            eye = torch.eye(k.size(-1), device=k.device, dtype=k.dtype)
            k = k + regularization * eye

        if return_complex and self.use_complex:
            return k
        return torch.abs(k) if self.use_complex else k

    def forward_cross(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        rho_a: Optional[torch.Tensor] = None,
        rho_b: Optional[torch.Tensor] = None,
        return_complex: bool = True,
        t_a: Optional[torch.Tensor] = None,
        t_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-kernel K(x_a, x_b): (B, Na, D) x (B, Nb, D) -> (B, Na, Nb)."""
        r = self.compute_distance_matrix(x_a, x_b)
        amplitude = self._spherical_amplitude_mag(r)
        if self.principle == "huygens_fresnel":
            alpha = self.effective_obliquity_scale()
            if t_a is not None and t_b is not None:
                ta = t_a.squeeze(-1) if t_a.dim() == 3 else t_a
                tb = t_b.squeeze(-1) if t_b.dim() == 3 else t_b
                dt = (ta.unsqueeze(-1) - tb.unsqueeze(-2)).clamp_min(0.0)
                axial = (self.effective_wave_speed() * dt).clamp_min(0.0)
            else:
                axial = r
            lateral2 = (alpha * r) ** 2
            cos_theta = (axial / torch.sqrt(axial ** 2 + lateral2 + self.eps)).clamp(-1.0, 1.0)
            amplitude = amplitude * (0.5 * (1.0 + cos_theta))
        if rho_a is not None and rho_b is not None:
            rho_mean = (rho_a.unsqueeze(-1) + rho_b.unsqueeze(-2)) / 2.0
            amplitude = amplitude * torch.exp(-rho_mean * r)
        envelope = torch.exp(-self.effective_gamma() * r ** 2)
        amp = amplitude * envelope
        omega = self.effective_omega()
        if self.use_complex:
            phase = torch.exp(1j * omega * r)
            if self.principle == "huygens_fresnel":
                phase = (1j) * phase
            k = amp.to(phase.dtype) * phase
        else:
            k = amp * torch.cos(omega * r)
        if return_complex and self.use_complex:
            return k
        return torch.abs(k) if self.use_complex else k

    def to_real_positive_definite(self, K: torch.Tensor, method: str = "abs") -> torch.Tensor:
        if method == "real":
            return K.real
        if method == "abs":
            return torch.abs(K)
        if method == "real_plus_ridge":
            k_real = K.real
            min_eig = torch.linalg.eigvalsh(k_real).min()
            if min_eig < 0:
                k_real = k_real + (-min_eig + 1e-3) * torch.eye(K.size(-1), device=K.device)
            return k_real
        raise ValueError(f"Unknown method: {method}")


# Re-export for callers that import BayesianHuygensKernel from hnf.kernel
# (implementation lives in hnf.bayesian_kernel).
def __getattr__(name: str):
    if name == "BayesianHuygensKernel":
        from hnf.bayesian_kernel import BayesianHuygensKernel as _BK

        return _BK
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
