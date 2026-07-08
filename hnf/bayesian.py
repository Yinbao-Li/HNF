# -*- coding: utf-8 -*-
"""Part 6: BayesianHNF — Bayesian inference extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from hnf.deep import DeepHuygensKernel


@dataclass
class BayesianHNFConfig:
    input_dim: int = 2
    hidden_dim: int = 64
    num_layers: int = 2
    gamma: float = 0.5
    omega: float = 0.3
    causal: bool = True
    wave_speed: float = 0.5
    num_particles: int = 10
    learning_rate: float = 1e-2
    num_iterations: int = 1000
    num_samples: int = 100


class BayesianHNF(nn.Module):
    """贝叶斯惠更斯神经场."""

    def __init__(self, config: BayesianHNFConfig):
        super().__init__()
        self.config = config
        self.hnf = DeepHuygensKernel(
            input_dim=config.input_dim,
            hidden_dims=[config.hidden_dim] * config.num_layers,
            gamma=config.gamma,
            omega=config.omega,
            causal=config.causal,
            wave_speed=config.wave_speed,
            num_layers=config.num_layers,
        )
        self.mean_head = nn.Linear(config.hidden_dim, 1)
        self.var_head = nn.Linear(config.hidden_dim, 1)
        self.prior_mean = nn.Parameter(torch.zeros(config.input_dim))
        self.prior_log_var = nn.Parameter(torch.zeros(config.input_dim))
        self.likelihood_log_var = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        return_distribution: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.hnf(x, t=None, rho=None)
        mean = self.mean_head(h)
        log_var = self.var_head(h)
        var = torch.exp(log_var)

        if observations is not None:
            obs_indices = torch.randperm(x.shape[1])[: min(len(observations), x.shape[1] // 2)]
            rho = torch.ones(x.shape[0], x.shape[1], device=x.device)
            for idx in obs_indices[: min(len(observations), len(obs_indices))]:
                rho[:, idx] += 1.0
            h_mod = self.hnf(x, t=None, rho=rho)
            mean_mod = self.mean_head(h_mod)
            log_var_mod = self.var_head(h_mod)
            alpha = 0.5
            mean = alpha * mean_mod + (1 - alpha) * mean
            var = alpha * torch.exp(log_var_mod) + (1 - alpha) * var

        if return_distribution:
            return mean, var
        return mean

    def sample_posterior(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        mean, var = self.forward(x, return_distribution=True)
        return mean + torch.sqrt(var) * torch.randn(num_samples, *mean.shape, device=x.device)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        samples = self.sample_posterior(x, num_samples)
        mean = samples.mean(dim=0)
        var = samples.var(dim=0)
        aleatoric_var = torch.exp(self.likelihood_log_var)
        return mean, var, aleatoric_var
