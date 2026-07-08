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
    惠更斯核函数 (Huygens Kernel)

    K(x_i, x_j) = 1/(r^2 + eps) * exp(-gamma * r^2) * exp(i * omega * r)
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
    ):
        super().__init__()
        if distance_mode not in {"feature", "time", "hybrid"}:
            raise ValueError(f"Unknown distance_mode: {distance_mode}")

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

        self.eps = eps
        self.causal = causal
        self.use_complex = use_complex
        self.distance_mode = distance_mode
        self.local_window_sec = local_window_sec
        self.sparse_band = sparse_band
        self._learnable_gamma = learnable_gamma
        self._learnable_wave_speed = learnable_wave_speed

    def effective_gamma(self) -> torch.Tensor:
        if self._learnable_gamma:
            return F.softplus(self.gamma) + 1e-3
        return self.gamma

    def effective_wave_speed(self) -> torch.Tensor:
        if self._learnable_wave_speed:
            return F.softplus(self.wave_speed) + 1e-3
        return self.wave_speed

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

    def _build_kernel(
        self,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        amplitude = 1.0 / (r ** 2 + self.eps)

        if rho is not None:
            if rho.dim() == 3 and rho.size(-1) == 1:
                rho = rho.squeeze(-1)
            rho_mean = (rho.unsqueeze(-1) + rho.unsqueeze(-2)) / 2.0
            amplitude = amplitude * torch.exp(-rho_mean * r)

        envelope = torch.exp(-self.effective_gamma() * r ** 2)

        if self.use_complex:
            phase = torch.exp(1j * self.omega * r)
        else:
            phase = torch.cos(self.omega * r)

        k = amplitude * envelope * phase

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
        amp = 1.0 / (r**2 + self.eps)
        if rho_i is not None and rho_j is not None:
            rho_mean = (rho_i + rho_j) / 2.0
            amp = amp * torch.exp(-rho_mean * r)
        envelope = torch.exp(-self.effective_gamma() * r**2)
        if self.use_complex:
            phase = torch.exp(1j * self.omega * r)
        else:
            phase = torch.cos(self.omega * r)
        return amp * envelope * phase

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

        amp = 1.0 / (lags**2 + self.eps)
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
        if self.use_complex:
            phase = torch.exp(1j * self.omega * lags).squeeze(-1)
        else:
            phase = torch.cos(self.omega * lags).squeeze(-1)
        k_stack = (amp * envelope.unsqueeze(0) * phase.unsqueeze(0)) * valid.unsqueeze(0)

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
        k = self._build_kernel(r, t=t, rho=rho)

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
    ) -> torch.Tensor:
        """Cross-kernel K(x_a, x_b): (B, Na, D) x (B, Nb, D) -> (B, Na, Nb)."""
        r = self.compute_distance_matrix(x_a, x_b)
        amplitude = 1.0 / (r ** 2 + self.eps)
        if rho_a is not None and rho_b is not None:
            rho_mean = (rho_a.unsqueeze(-1) + rho_b.unsqueeze(-2)) / 2.0
            amplitude = amplitude * torch.exp(-rho_mean * r)
        envelope = torch.exp(-self.effective_gamma() * r ** 2)
        if self.use_complex:
            phase = torch.exp(1j * self.omega * r)
        else:
            phase = torch.cos(self.omega * r)
        k = amplitude * envelope * phase
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
