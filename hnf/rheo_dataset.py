# -*- coding: utf-8 -*-
"""Dataset + model for Prony / Boltzmann rheology memory (Domain-III R0/R1)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from hnf.rheo_memory import PronyBoltzmannKernel
from hnf.rheo_synth import ID_TO_PROTOCOL, PROTOCOL_TO_ID, make_rheo_sample


class RheoMemoryDataset(Dataset):
    """On-the-fly strain-rate protocols with GT Prony stress.

    Default R0 mode: **fixed material** spectrum shared across samples;
    only the loading protocol varies (classic rheometric identification).
    """

    def __init__(
        self,
        split: str = "train",
        n_samples: int = 2048,
        n_steps: int = 256,
        dt: float = 0.05,
        n_modes: int = 2,
        dim: int = 1,
        anisotropic: bool = False,
        noise_std: float = 0.01,
        seed: int = 42,
        protocols: Optional[list[str]] = None,
        fixed_material: bool = True,
        material_lambdas: Optional[list[float]] = None,
        material_weights: Optional[list[float]] = None,
        material_g_inf: float = 0.0,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.split = split
        self.n_samples = int(n_samples)
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_modes = int(n_modes)
        self.dim = int(dim)
        self.anisotropic = bool(anisotropic)
        self.noise_std = float(noise_std)
        self.protocols = protocols or list(PROTOCOL_TO_ID.keys())
        self.fixed_material = bool(fixed_material)
        base = {"train": 20_000_000, "val": 21_000_000, "test": 22_000_000}[split]
        self._seeds = [base + seed * 10_000 + i for i in range(self.n_samples)]

        if material_lambdas is None:
            # Canonical R0 material: two Maxwell modes
            material_lambdas = [0.5, 5.0][:n_modes]
            while len(material_lambdas) < n_modes:
                material_lambdas.append(10.0 ** (len(material_lambdas) - 1))
        if material_weights is None:
            material_weights = [1.2, 0.6][:n_modes]
            while len(material_weights) < n_modes:
                material_weights.append(0.4)
        self.material_lambdas = [float(x) for x in material_lambdas]
        if anisotropic and dim > 1:
            # diagonal modal weights per channel
            self.material_weights = np.asarray(
                [[float(material_weights[k]) * (1.0 + 0.3 * c) for c in range(dim)] for k in range(n_modes)],
                dtype=np.float64,
            )
        else:
            self.material_weights = [float(x) for x in material_weights]
        self.material_g_inf = float(material_g_inf)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float | str | int]:
        seed = self._seeds[idx]
        rng = np.random.default_rng(seed)
        protocol = str(rng.choice(self.protocols))
        kw: dict = dict(
            n_steps=self.n_steps,
            dt=self.dt,
            n_modes=self.n_modes,
            dim=self.dim,
            anisotropic=self.anisotropic,
            protocol=protocol,
            seed=seed,
            noise_std=self.noise_std if self.split == "train" else self.noise_std * 0.5,
        )
        if self.fixed_material:
            kw.update(
                lambdas=self.material_lambdas,
                weights=self.material_weights,
                g_inf=self.material_g_inf,
            )
        s = make_rheo_sample(**kw)
        gd = torch.from_numpy(s["gammadot"])
        stress = torch.from_numpy(s["stress"])
        return {
            "gammadot": gd,
            "stress": stress,
            "dt": torch.tensor(s["dt"], dtype=torch.float32),
            "lambda": torch.from_numpy(s["lambda"]),
            "G": torch.from_numpy(s["G"]),
            "g_inf": torch.tensor(s["g_inf"], dtype=torch.float32),
            "protocol": str(s["protocol"]),
            "protocol_id": int(s["protocol_id"]),
            "seed": int(s["seed"]),
        }


class RheoMemoryModel(nn.Module):
    """Learn Prony params; forward predicts σ̂ = K * γ̇."""

    def __init__(
        self,
        n_modes: int = 2,
        dim: int = 1,
        anisotropic: bool = False,
        lambda_init: Optional[list[float]] = None,
        g_init: Optional[list[float]] = None,
    ):
        super().__init__()
        self.kernel = PronyBoltzmannKernel(
            n_modes=n_modes,
            dim=dim,
            anisotropic=anisotropic,
            lambda_init=lambda_init,
            g_init=g_init,
            g_inf_init=0.0,
        )

    def forward(self, gammadot: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        return self.kernel(gammadot, dt=dt)

    def collect_params(self) -> dict:
        return self.kernel.collect_params()


def rheo_memory_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    param_reg: float = 0.0,
    freq_weight: float = 0.0,
    param_weight: float = 0.0,
    kernel: Optional[PronyBoltzmannKernel] = None,
    gt_lambda: Optional[torch.Tensor] = None,
    gt_G: Optional[torch.Tensor] = None,
    gt_g_inf: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stress MSE + optional frequency G*/G** match + Prony param match."""
    mse = (pred - target).pow(2).mean()
    loss = mse
    stats: dict[str, float] = {"mse": float(mse.detach().item())}

    if kernel is not None and freq_weight > 0 and not kernel.anisotropic:
        # Match analytic G', G'' on a log-ω grid vs GT Prony (from batch labels).
        device = pred.device
        omega = torch.logspace(-1.0, 1.0, 16, device=device)
        gp_hat, gpp_hat = kernel.complex_modulus(omega)
        if gt_lambda is not None and gt_G is not None:
            # Use first sample's GT (fixed-material batches share spectrum)
            lam = gt_lambda[0].to(device).clamp_min(1e-6)
            g = gt_G[0].to(device)
            if g.dim() > 1:
                g = g.reshape(lam.numel(), -1).mean(dim=-1)
            ginf = (
                gt_g_inf[0].to(device)
                if gt_g_inf is not None
                else torch.zeros((), device=device)
            )
            x = omega.unsqueeze(-1) * lam.unsqueeze(0)
            den = 1.0 + x * x
            gp_gt = ginf + (g * (x * x) / den).sum(dim=-1)
            gpp_gt = (g * x / den).sum(dim=-1)
            freq = (gp_hat - gp_gt).pow(2).mean() + (gpp_hat - gpp_gt).pow(2).mean()
            loss = loss + freq_weight * freq
            stats["freq"] = float(freq.detach().item())

    if kernel is not None and param_weight > 0 and gt_lambda is not None and gt_G is not None:
        # Soft assignment: sort both by λ and match in log space
        lam = kernel.relaxation_times()
        gt_lam = gt_lambda[0].to(lam.device).clamp_min(1e-6)
        lam_s, _ = torch.sort(lam)
        gt_s, idx = torch.sort(gt_lam)
        w = kernel.modal_weights()
        if kernel.anisotropic:
            # compare mean diagonal
            w_iso = w.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            g_gt = gt_G[0].to(lam.device).reshape(gt_lam.numel(), -1).mean(dim=-1)
        else:
            w_iso = w
            g_gt = gt_G[0].to(lam.device)
            if g_gt.numel() != gt_lam.numel():
                g_gt = g_gt.reshape(gt_lam.numel(), -1).mean(dim=-1)
        g_s = w_iso[torch.argsort(lam)]
        g_gt_s = g_gt[idx]
        p_loss = (torch.log(lam_s) - torch.log(gt_s)).pow(2).mean()
        p_loss = p_loss + (torch.log(g_s.clamp_min(1e-6)) - torch.log(g_gt_s.clamp_min(1e-6))).pow(
            2
        ).mean()
        if gt_g_inf is not None:
            p_loss = p_loss + (kernel.g_inf() - gt_g_inf[0].to(lam.device)).pow(2)
        loss = loss + param_weight * p_loss
        stats["param"] = float(p_loss.detach().item())

    if param_reg > 0 and kernel is not None:
        lam = kernel.relaxation_times()
        if lam.numel() > 1:
            log_lam, _ = torch.sort(torch.log(lam))
            # Encourage *spread* (negative squared gap → soft hinge on min decade gap)
            gaps = log_lam[1:] - log_lam[:-1]
            # penalize collapsed modes (gap < log(3))
            reg = F.relu(math.log(3.0) - gaps).pow(2).mean()
            loss = loss + param_reg * reg
            stats["lam_spread_reg"] = float(reg.detach().item())

    stats["loss"] = float(loss.detach().item())
    return loss, stats


__all__ = [
    "RheoMemoryDataset",
    "RheoMemoryModel",
    "rheo_memory_loss",
    "PROTOCOL_TO_ID",
    "ID_TO_PROTOCOL",
]
