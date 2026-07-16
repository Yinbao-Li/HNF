# -*- coding: utf-8 -*-
"""Bayesian–Monte Carlo Causal Kernel (BMCCK).

BayesianHuygensKernel:
  - γ, ω ~ LogNormal variational posteriors (torch.distributions + reparam)
  - Monte Carlo average over M causal mid-path samples inside the light cone
  - Drop-in API compatible with HuygensKernel (forward / forward_apply / priors)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal, kl_divergence

from hnf.kernel import HuygensKernel


def _positive_scale(log_scale: torch.Tensor, min_scale: float = 1e-3) -> torch.Tensor:
    return F.softplus(log_scale) + min_scale


class BayesianHuygensKernel(HuygensKernel):
    """Huygens kernel with LogNormal VI on (γ, ω) and MC causal path sampling."""

    def __init__(
        self,
        gamma: float = 1.0,
        omega: float = 1.0,
        eps: float = 1e-6,
        causal: bool = True,
        wave_speed: float = 1.0,
        learnable_wave_speed: bool = False,
        use_complex: bool = True,
        distance_mode: str = "feature",
        local_window_sec: Optional[float] = None,
        sparse_band: bool = False,
        principle: str = "huygens",
        obliquity_scale: float = 1.0,
        obliquity_mix: float = 0.0,
        learnable_obliquity: bool = False,
        n_samples: int = 32,
        prior_log_sigma: float = 0.35,
        init_log_sigma: float = 0.15,
        use_posterior_mean: bool = False,
        mc_path_temperature: float = 1.0,
        **kwargs,
    ):
        # Discard deterministic learnable_gamma/omega if passed by callers.
        kwargs.pop("learnable_gamma", None)
        kwargs.pop("learnable_omega", None)

        super().__init__(
            gamma=gamma,
            omega=omega,
            eps=eps,
            causal=causal,
            wave_speed=wave_speed,
            learnable_gamma=False,
            learnable_omega=False,
            learnable_wave_speed=learnable_wave_speed,
            use_complex=use_complex,
            distance_mode=distance_mode,
            local_window_sec=local_window_sec,
            sparse_band=sparse_band,
            principle=principle,
            obliquity_scale=obliquity_scale,
            obliquity_mix=obliquity_mix,
            learnable_obliquity=learnable_obliquity,
        )

        # Replace fixed γ/ω buffers with LogNormal variational parameters.
        # log(γ) ~ N(loc, scale); initialize so median matches construction gamma/omega.
        if "gamma" in self._buffers:
            del self._buffers["gamma"]
        if "omega" in self._buffers:
            del self._buffers["omega"]

        g0 = max(float(gamma), 1e-3)
        w0 = max(float(omega), 1e-3)
        self.gamma_loc = nn.Parameter(torch.tensor(math.log(g0), dtype=torch.float32))
        self.omega_loc = nn.Parameter(torch.tensor(math.log(w0), dtype=torch.float32))
        init_ls = math.log(math.expm1(max(float(init_log_sigma), 1e-4)))
        self.gamma_log_scale = nn.Parameter(torch.tensor(init_ls, dtype=torch.float32))
        self.omega_log_scale = nn.Parameter(torch.tensor(init_ls, dtype=torch.float32))

        # Keep legacy attribute names pointing at loc (Parameter) for tooling.
        self.gamma = self.gamma_loc
        self.omega = self.omega_loc
        self._learnable_gamma = True
        self._learnable_omega = True

        self.n_samples = max(1, int(n_samples))
        self.prior_log_sigma = float(prior_log_sigma)
        self.use_posterior_mean = bool(use_posterior_mean)
        self.mc_path_temperature = float(max(1e-3, mc_path_temperature))

        self.register_buffer("gamma_sample", torch.tensor(g0, dtype=torch.float32))
        self.register_buffer("omega_sample", torch.tensor(w0, dtype=torch.float32))

    # ------------------------------------------------------------------ VI --
    def gamma_posterior(self) -> LogNormal:
        return LogNormal(self.gamma_loc, _positive_scale(self.gamma_log_scale))

    def omega_posterior(self) -> LogNormal:
        return LogNormal(self.omega_loc, _positive_scale(self.omega_log_scale))

    def gamma_prior(self) -> LogNormal:
        loc = torch.log(self.gamma0.clamp_min(1e-3))
        scale = self.gamma0.new_tensor(self.prior_log_sigma)
        return LogNormal(loc, scale)

    def omega_prior(self) -> LogNormal:
        loc = torch.log(self.omega0.clamp_min(1e-3))
        scale = self.omega0.new_tensor(self.prior_log_sigma)
        return LogNormal(loc, scale)

    def sample_physical_params(self, deterministic: Optional[bool] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized draw (or posterior mean). Updates gamma_sample / omega_sample."""
        use_mean = self.use_posterior_mean if deterministic is None else deterministic
        if use_mean or (not self.training and deterministic is None and self.use_posterior_mean):
            g = self.gamma_posterior().mean
            w = self.omega_posterior().mean
        elif use_mean:
            g = self.gamma_posterior().mean
            w = self.omega_posterior().mean
        else:
            g = self.gamma_posterior().rsample()
            w = self.omega_posterior().rsample()
        g = g.clamp_min(1e-3)
        w = w.clamp_min(1e-3)
        self.gamma_sample = g.detach()
        self.omega_sample = w.detach()
        # Keep live (differentiable) values on the module for this forward.
        self._gamma_live = g
        self._omega_live = w
        return g, w

    def effective_gamma(self) -> torch.Tensor:
        if hasattr(self, "_gamma_live"):
            return self._gamma_live
        return self.gamma_sample.clamp_min(1e-3)

    def effective_omega(self) -> torch.Tensor:
        if hasattr(self, "_omega_live"):
            return self._omega_live
        return self.omega_sample.clamp_min(1e-3)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(γ,ω) || p(γ,ω)) under independent LogNormal factors."""
        return kl_divergence(self.gamma_posterior(), self.gamma_prior()) + kl_divergence(
            self.omega_posterior(), self.omega_prior()
        )

    def physics_prior_loss(self) -> torch.Tensor:
        """VI objective term: KL + weak relative pull of posterior medians to anchors."""
        kl = self.kl_divergence()
        g_med = torch.exp(self.gamma_loc)
        w_med = torch.exp(self.omega_loc)
        g0 = self.gamma0.clamp_min(1e-3)
        w0 = self.omega0.clamp_min(1e-3)
        c = self.effective_wave_speed()
        c0 = self.wave_speed0.clamp_min(1e-3)
        if self._learnable_wave_speed:
            c0 = F.softplus(self.wave_speed0) + 1e-3
        pull = ((g_med - g0) / g0).pow(2) + ((w_med - w0) / w0).pow(2) + ((c - c0) / c0).pow(2)
        return kl + 0.1 * pull

    # --------------------------------------------------- MC path sampling --
    def _sample_path_fractions(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Differentiable fractions α ∈ (0,1) along causal rays.

        Uses a continuous logistic reparameterization (smooth Uniform analogue).
        Temperature scales path stochasticity (Gumbel-Softmax-like sharpness).
        """
        # Logistic(0, T): sigmoid(eps * T) concentrates near {0,1} as T→0; near Uniform as T~1.
        eps = torch.randn(shape, device=device, dtype=dtype)
        return torch.sigmoid(eps / self.mc_path_temperature).clamp(1e-4, 1.0 - 1e-4)

    def _kernel_from_distance(
        self,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic kernel given current effective γ/ω (after sample_physical_params)."""
        return super()._build_kernel(r, t=t, rho=rho, mask=mask, x=x)

    def _build_kernel(
        self,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.sample_physical_params()
        if self.n_samples <= 1:
            return self._kernel_from_distance(r, t=t, rho=rho, mask=mask, x=x)

        # Monte Carlo over intermediate light-cone radii: r_m = α_m · r (ray midpoints).
        m = self.n_samples
        alphas = self._sample_path_fractions((m,), r.device, r.dtype)
        # Shape broadcast: (M, *r.shape)
        expand = (m,) + (1,) * r.dim()
        r_paths = alphas.view(expand) * r.unsqueeze(0)

        # Causal mask uses parent (full) lag r / dt — midpoints stay inside the cone.
        if mask is None and t is not None:
            mask = self.compute_causal_mask(r, t)

        acc = None
        for i in range(m):
            ki = self._kernel_from_distance(
                r_paths[i], t=t, rho=rho, mask=mask, x=x
            )
            acc = ki if acc is None else acc + ki
        return acc / float(m)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        return_complex: bool = True,
        regularization: float = 0.0,
    ) -> torch.Tensor:
        return super().forward(
            x, t=t, rho=rho, return_complex=return_complex, regularization=regularization
        )

    def forward_apply(
        self,
        h_c: torch.Tensor,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sparse-band matmul with MC mid-lag causal sampling (GPU vectorized)."""
        self.sample_physical_params()

        if not self.sparse_band or t is None or self.distance_mode not in {"time", "hybrid"}:
            k = self.forward(x, t=t, rho=rho, return_complex=True)
            return torch.matmul(k, h_c)

        n = h_c.size(1)
        dt_step = self.dt_step_sec(n, t)
        w_max = self.window_bins(n, t)
        b, _, d_feat = h_c.shape
        device = h_c.device
        dtype_r = torch.float32
        m = self.n_samples

        lags = (
            torch.arange(1, w_max + 1, device=device, dtype=dtype_r).view(w_max, 1, 1)
            * dt_step
        )  # (W,1,1)
        rows = torch.arange(n, device=device)
        src = rows.view(1, n) - torch.arange(1, w_max + 1, device=device).view(w_max, 1)
        valid = (src >= 0) & (lags.squeeze(-1) <= (self.local_window_sec or 1e9) + 1e-6)
        src_clamped = src.clamp(min=0)

        if m <= 1:
            path_lags = lags.unsqueeze(0)  # (1,W,1,1)
        else:
            alphas = self._sample_path_fractions((m,), device, dtype_r).view(m, 1, 1, 1)
            path_lags = alphas * lags.unsqueeze(0)  # (M,W,1,1)

        amp = self._spherical_amplitude_mag(path_lags)
        if self.principle == "huygens_fresnel" or float(self.obliquity_mix.item()) > 0.0:
            c = self.effective_wave_speed()
            axial = (c * path_lags).clamp_min(0.0)
            alpha_ob = self.effective_obliquity_scale()
            lateral2 = (alpha_ob ** 2) * (path_lags.clamp_min(0.0) + self.eps)
            cos_theta = (axial / torch.sqrt(axial ** 2 + lateral2 + self.eps)).clamp(-1.0, 1.0)
            chi = 0.5 * (1.0 + cos_theta)
            if self.principle == "huygens_fresnel":
                amp = amp * chi
            else:
                mix = self.effective_obliquity_mix()
                amp = amp * (1.0 - mix + mix * chi)

        # amp: (M,W,1,1) -> (M,W)
        amp = amp.squeeze(-1).squeeze(-1)

        if rho is not None:
            rho_1d = rho.squeeze(-1) if rho.dim() == 3 and rho.size(-1) == 1 else rho
            rho_i = rho_1d.unsqueeze(1).expand(b, w_max, n)
            idx = src_clamped.unsqueeze(0).expand(b, w_max, n).long()
            rho_j = torch.gather(rho_1d.unsqueeze(1).expand(b, w_max, n), 2, idx)
            # Use parent lag for rho coupling (same as deterministic sparse_band).
            parent_lag = lags.squeeze(-1).unsqueeze(0)  # (1,W,1)
            rho_fac = torch.exp(-(rho_i + rho_j) / 2.0 * parent_lag)  # (B,W,N)
            amp = amp.view(1, m, w_max, 1) * rho_fac.unsqueeze(1)  # (B,M,W,N)
        else:
            amp = amp.view(1, m, w_max, 1).expand(b, m, w_max, n)

        gamma = self.effective_gamma()
        omega = self.effective_omega()
        envelope = torch.exp(-gamma * path_lags.squeeze(-1).squeeze(-1) ** 2)  # (M,W)
        amp = amp * envelope.view(1, m, w_max, 1)

        if self.use_complex:
            phase = torch.exp(1j * omega * path_lags.squeeze(-1).squeeze(-1))  # (M,W)
            if self.principle == "huygens_fresnel":
                phase = (1j) * phase
            k_stack = (amp.to(phase.dtype) * phase.view(1, m, w_max, 1)) * valid.view(
                1, 1, w_max, n
            )
        else:
            phase = torch.cos(omega * path_lags.squeeze(-1).squeeze(-1))
            k_stack = (amp * phase.view(1, m, w_max, 1)) * valid.view(1, 1, w_max, n)

        # Monte Carlo mean over paths.
        k_mean = k_stack.mean(dim=1)  # (B,W,N)

        idx_h = src_clamped.unsqueeze(0).unsqueeze(-1).expand(b, w_max, n, d_feat).long()
        h_stack = torch.gather(
            h_c.unsqueeze(1).expand(b, w_max, n, d_feat),
            2,
            idx_h,
        )
        h_stack = h_stack * valid.unsqueeze(0).unsqueeze(-1)
        return (k_mean.unsqueeze(-1) * h_stack).sum(dim=1)

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
        self.sample_physical_params()
        return super().forward_cross(
            x_a, x_b, rho_a=rho_a, rho_b=rho_b, return_complex=return_complex, t_a=t_a, t_b=t_b
        )


def bayesian_kernel_kl_loss(model: nn.Module) -> torch.Tensor:
    """Mean KL over all BayesianHuygensKernel modules (0 if none)."""
    terms: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, BayesianHuygensKernel):
            terms.append(module.kl_divergence())
    if not terms:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(terms).mean()
