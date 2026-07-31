# -*- coding: utf-8 -*-
"""Anisotropic Prony / Boltzmann memory kernel for linear viscoelasticity.

Boltzmann superposition:
  σ(t) = ∫_{-∞}^t G(t-s) : γ̇(s) ds

Isotropic Prony series:
  G(τ) = Σ_k G_k exp(-τ/λ_k) + G_∞

Anisotropic (mode amplitudes SPD):
  G(τ) = Σ_k A_k exp(-τ/λ_k) + G_∞ I
  A_k = L_k L_k^T + ε I

Discrete Maxwell recurrence (γ̇ piecewise-constant on Δt):
  σ_k(t+Δt) = σ_k(t) e^{-Δt/λ_k}
            + A_k λ_k (1 - e^{-Δt/λ_k}) γ̇
  σ_∞ = G_∞ γ   (integrated strain from rest)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PronyBoltzmannKernel(nn.Module):
    """Causal Prony memory operator with optional anisotropic modal weights."""

    def __init__(
        self,
        n_modes: int = 2,
        dim: int = 1,
        anisotropic: bool = False,
        lambda_init: Optional[list[float]] = None,
        g_init: Optional[list[float]] = None,
        g_inf_init: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.n_modes = int(n_modes)
        self.dim = int(dim)
        self.anisotropic = bool(anisotropic)
        self.eps = float(eps)

        if lambda_init is None:
            lambda_init = [10.0 ** (i - (n_modes - 1) / 2.0) for i in range(n_modes)]
        if g_init is None:
            g_init = [1.0] * n_modes
        if len(lambda_init) != n_modes or len(g_init) != n_modes:
            raise ValueError("lambda_init / g_init length must equal n_modes")

        # λ_k > 0 via softplus(raw) + eps
        raw_lam = torch.tensor(
            [float(_inv_softplus(max(v, 1e-3))) for v in lambda_init],
            dtype=torch.float32,
        )
        self.raw_lambda = nn.Parameter(raw_lam)

        if anisotropic:
            # L_k such that A_k ≈ g_init[k] I at init
            L0 = torch.zeros(n_modes, dim, dim, dtype=torch.float32)
            for k, g in enumerate(g_init):
                L0[k].diagonal().copy_(torch.full((dim,), float(g) ** 0.5))
            self.diff_L = nn.Parameter(L0)
            self.raw_G = None
        else:
            raw_g = torch.tensor(
                [float(_inv_softplus(max(v, 1e-4))) for v in g_init],
                dtype=torch.float32,
            )
            self.raw_G = nn.Parameter(raw_g)
            self.register_parameter("diff_L", None)

        self.raw_G_inf = nn.Parameter(
            torch.tensor(_inv_softplus(max(g_inf_init, 0.0) + 1e-8), dtype=torch.float32)
            if g_inf_init > 0
            else torch.tensor(-8.0, dtype=torch.float32)
        )

    def relaxation_times(self) -> torch.Tensor:
        """λ_k, shape (K,)."""
        return F.softplus(self.raw_lambda) + self.eps

    def modal_weights(self) -> torch.Tensor:
        """Isotropic G_k (K,) or anisotropic A_k (K, d, d)."""
        if self.anisotropic:
            assert self.diff_L is not None
            L = torch.tril(self.diff_L)
            eye = torch.eye(self.dim, device=L.device, dtype=L.dtype)
            return L @ L.transpose(-1, -2) + self.eps * eye
        assert self.raw_G is not None
        return F.softplus(self.raw_G) + self.eps

    def g_inf(self) -> torch.Tensor:
        return F.softplus(self.raw_G_inf)

    def complex_modulus(
        self,
        omega: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Storage / loss moduli G'(ω), G''(ω) for isotropic Prony (scalar).

        G' = G_∞ + Σ_k G_k (ωλ)^2 / (1+(ωλ)^2)
        G'' = Σ_k G_k (ωλ) / (1+(ωλ)^2)
        """
        if self.anisotropic:
            raise NotImplementedError("complex_modulus is isotropic-only in R0/R1")
        lam = self.relaxation_times()  # (K,)
        g = self.modal_weights()  # (K,)
        om = omega.reshape(-1).clamp_min(1e-8)  # (W,)
        x = om.unsqueeze(-1) * lam.unsqueeze(0)  # (W, K)
        den = 1.0 + x * x
        gp = self.g_inf() + (g * (x * x) / den).sum(dim=-1)
        gpp = (g * x / den).sum(dim=-1)
        return gp, gpp

    def relaxation_modulus(self, tau: torch.Tensor) -> torch.Tensor:
        """G(τ) for τ>=0.

        Returns (...,) isotropic scalar or (..., d, d) anisotropic.
        """
        lam = self.relaxation_times()  # (K,)
        w = self.modal_weights()
        tau = tau.clamp_min(0.0)
        # exp: (..., K)
        decay = torch.exp(-tau.unsqueeze(-1) / lam)
        if self.anisotropic:
            # (..., K, d, d)
            return (decay.unsqueeze(-1).unsqueeze(-1) * w).sum(dim=-3) + self.g_inf() * torch.eye(
                self.dim, device=tau.device, dtype=tau.dtype
            )
        return (decay * w).sum(dim=-1) + self.g_inf()

    def forward(
        self,
        gammadot: torch.Tensor,
        dt: float | torch.Tensor,
        *,
        return_modes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply Boltzmann memory to strain-rate history.

        Args:
            gammadot: (B, T) or (B, T, d)
            dt: scalar step or (B,) / broadcastable
        Returns:
            stress: same shape as gammadot
        """
        if gammadot.dim() == 2:
            if self.dim != 1:
                raise ValueError(f"expected dim={self.dim}, got scalar series")
            gd = gammadot.unsqueeze(-1)  # (B, T, 1)
            squeeze = True
        elif gammadot.dim() == 3:
            if gammadot.size(-1) != self.dim:
                raise ValueError(f"last dim {gammadot.size(-1)} != {self.dim}")
            gd = gammadot
            squeeze = False
        else:
            raise ValueError("gammadot must be (B,T) or (B,T,d)")

        b, t, d = gd.shape
        device, dtype = gd.device, gd.dtype
        if not torch.is_tensor(dt):
            dt_t = torch.full((b,), float(dt), device=device, dtype=dtype)
        else:
            dt_t = dt.to(device=device, dtype=dtype).reshape(-1)
            if dt_t.numel() == 1:
                dt_t = dt_t.expand(b)

        lam = self.relaxation_times()  # (K,)
        w = self.modal_weights()
        ginf = self.g_inf()
        eye = torch.eye(d, device=device, dtype=dtype)

        # Integrated strain from rest
        strain = torch.cumsum(gd * dt_t.view(b, 1, 1), dim=1)
        stress_inf = ginf * strain

        # Maxwell modes via exact recurrence (vectorized over modes; loop over T)
        # σ_k[n] = α_k σ_k[n-1] + (A_k λ_k (1-α_k)) γ̇[n]
        alpha = torch.exp(-dt_t.unsqueeze(-1) / lam.unsqueeze(0))  # (B, K)
        one_m = 1.0 - alpha  # (B, K)
        if self.anisotropic:
            # A: (K,d,d); gain[b,k] = A_k * λ_k * (1-α_{b,k})
            gain = w.unsqueeze(0) * (lam * one_m).unsqueeze(-1).unsqueeze(-1)  # (B,K,d,d)
        else:
            # isotropic: gain scalar per mode → diagonal
            gain_scale = (w * lam).unsqueeze(0) * one_m  # (B, K)
            gain = gain_scale.unsqueeze(-1).unsqueeze(-1) * eye.view(1, 1, d, d)

        sig = torch.zeros(b, self.n_modes, d, device=device, dtype=dtype)
        outs = []
        for n in range(t):
            gd_n = gd[:, n, :]  # (B, d)
            # (B,K,d,d) @ (B,K,d,1)
            add = torch.matmul(gain, gd_n.unsqueeze(1).unsqueeze(-1).expand(b, self.n_modes, d, 1))
            add = add.squeeze(-1)  # (B, K, d)
            sig = alpha.unsqueeze(-1) * sig + add
            outs.append(sig.sum(dim=1))  # sum modes → (B, d)
        stress_modes_sum = torch.stack(outs, dim=1)  # (B, T, d)
        stress = stress_inf + stress_modes_sum

        if return_modes:
            # Recompute mode trajectories if requested (second pass, rare)
            sig = torch.zeros(b, self.n_modes, d, device=device, dtype=dtype)
            mode_outs = []
            for n in range(t):
                gd_n = gd[:, n, :]
                add = torch.matmul(
                    gain, gd_n.unsqueeze(1).unsqueeze(-1).expand(b, self.n_modes, d, 1)
                ).squeeze(-1)
                sig = alpha.unsqueeze(-1) * sig + add
                mode_outs.append(sig.clone())
            modes = torch.stack(mode_outs, dim=1)  # (B, T, K, d)
            modes = modes.permute(0, 1, 3, 2)  # (B, T, d, K)
            if squeeze:
                return stress.squeeze(-1), modes.squeeze(-2)
            return stress, modes

        if squeeze:
            return stress.squeeze(-1)
        return stress

    def collect_params(self) -> dict[str, float | list[float]]:
        lam = self.relaxation_times().detach().cpu().tolist()
        out: dict[str, float | list[float]] = {
            "lambda": [float(x) for x in lam],
            "g_inf": float(self.g_inf().detach().cpu()),
            "anisotropic": float(self.anisotropic),
            "n_modes": float(self.n_modes),
            "dim": float(self.dim),
        }
        w = self.modal_weights().detach().cpu()
        if self.anisotropic:
            out["A"] = w.reshape(self.n_modes, -1).tolist()
        else:
            out["G"] = [float(x) for x in w.tolist()]
        return out


def _inv_softplus(y: float) -> float:
    # softplus^{-1}(y) = log(exp(y) - 1); stable for small y
    y = max(float(y), 1e-8)
    if y > 20:
        return y
    return float(torch.log(torch.expm1(torch.tensor(y))).item())


def boltzmann_convolution_dense(
    gammadot: torch.Tensor,
    kernel: PronyBoltzmannKernel,
    dt: float,
) -> torch.Tensor:
    """Dense causal convolution reference (slow; for tests)."""
    if gammadot.dim() == 2:
        gd = gammadot.unsqueeze(-1)
        squeeze = True
    else:
        gd = gammadot
        squeeze = False
    b, t, d = gd.shape
    device, dtype = gd.device, gd.dtype
    lags = torch.arange(t, device=device, dtype=dtype) * dt  # (T,)
    # G_mat[i,j] = G((i-j)dt) for j<=i else 0
    # build lower-triangular Toeplitz via modulus
    stress = torch.zeros(b, t, d, device=device, dtype=dtype)
    for i in range(t):
        for j in range(i + 1):
            tau = lags[i - j]
            g = kernel.relaxation_modulus(tau)
            gd_j = gd[:, j, :]
            if kernel.anisotropic:
                # g: (d,d), σ += G @ γ̇ * dt
                stress[:, i, :] = stress[:, i, :] + torch.nn.functional.linear(gd_j, g) * dt
            else:
                stress[:, i, :] = stress[:, i, :] + g * gd_j * dt
    if squeeze:
        return stress.squeeze(-1)
    return stress
