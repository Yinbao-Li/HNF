# -*- coding: utf-8 -*-
"""Sparse-to-dense field reconstruction using the user's HuygensKernel."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hnf.density import DensityNet
from hnf.kernel import HuygensKernel


def solve_weights(kernel_matrix: torch.Tensor, obs_values: torch.Tensor, alpha: float) -> torch.Tensor:
    if obs_values.dim() == 1:
        obs_values = obs_values.unsqueeze(-1)
    n = kernel_matrix.shape[0]
    reg = alpha * torch.eye(n, device=kernel_matrix.device, dtype=kernel_matrix.dtype)
    return torch.linalg.solve(kernel_matrix + reg, obs_values)


class HuygensNeuralField(nn.Module):
    """
  物理场重建模型（SST 等稀疏观测任务）:
      1. K_obs = Re(kernel(obs, obs))
      2. weights = solve((K_obs + alpha I), y)
      3. field_pred = Re(kernel(target, obs)) @ weights
    """

    def __init__(
        self,
        gamma: float = 0.5,
        omega: float = 0.3,
        eps: float = 1e-6,
        alpha: float = 1e-3,
        causal: bool = False,
        wave_speed: float = 1.0,
        learnable_gamma: bool = True,
        learnable_omega: bool = True,
        use_density: bool = False,
        target_chunk_size: int = 4096,
    ):
        super().__init__()
        self.alpha = alpha
        self.use_density = use_density
        self.target_chunk_size = target_chunk_size
        self.kernel = HuygensKernel(
            gamma=gamma,
            omega=omega,
            eps=eps,
            causal=causal,
            wave_speed=wave_speed,
            learnable_gamma=learnable_gamma,
            learnable_omega=learnable_omega,
        )
        self.density_net = DensityNet(input_dim=2) if use_density else None

    @property
    def gamma(self) -> torch.Tensor:
        return self.kernel.gamma

    @property
    def omega(self) -> torch.Tensor:
        return self.kernel.omega

    def _rho(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        if self.density_net is None:
            return None
        return self.density_net(coords.unsqueeze(0)).squeeze(0)

    def compute_weights(self, obs_coords: torch.Tensor, obs_values: torch.Tensor) -> torch.Tensor:
        obs_b = obs_coords.unsqueeze(0)
        rho = self._rho(obs_coords)
        k_obs = self.kernel.forward_cross(obs_b, obs_b, rho, rho, return_complex=True).real.squeeze(0)
        return solve_weights(k_obs, obs_values, self.alpha)

    def fit_at_observations(self, obs_coords: torch.Tensor, obs_values: torch.Tensor) -> torch.Tensor:
        """Predict field values at observation coordinates (interpolation check)."""
        return self.forward(obs_coords, obs_values, obs_coords)

    def forward(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        target_coords: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.compute_weights(obs_coords, obs_values)
        obs_b = obs_coords.unsqueeze(0)
        tgt_b = target_coords.unsqueeze(0)
        rho_obs = self._rho(obs_coords)
        rho_tgt = self._rho(target_coords)

        m = target_coords.shape[0]
        if m <= self.target_chunk_size:
            k_cross = self.kernel.forward_cross(tgt_b, obs_b, rho_tgt, rho_obs, return_complex=True).real.squeeze(0)
            return k_cross @ weights

        chunks = []
        for start in range(0, m, self.target_chunk_size):
            end = min(start + self.target_chunk_size, m)
            tc = target_coords[start:end].unsqueeze(0)
            rho_chunk = self._rho(target_coords[start:end])
            k_chunk = self.kernel.forward_cross(tc, obs_b, rho_chunk, rho_obs, return_complex=True).real.squeeze(0)
            chunks.append(k_chunk @ weights)
        return torch.cat(chunks, dim=0)
