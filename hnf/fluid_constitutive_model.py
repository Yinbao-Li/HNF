# -*- coding: utf-8 -*-
"""Stage-1 constitutive dataset + HNF Physics Decoder head."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from hnf.fluid_constitutive import (
    CONST_FAMILY_TO_ID,
    CONST_ID_TO_FAMILY,
    N_THETA,
    make_constitutive_sample,
)
from hnf.fluid_model import fluid_scale_specs
from hnf.multiscale import MultiScaleHuygensEncoder
from hnf.picking_model import TemporalMediumDensity


class ConstitutiveFluidDataset(Dataset):
    """On-the-fly Newtonian / Carreau sparse fields with GT θ."""

    def __init__(
        self,
        split: str = "train",
        n_samples: int = 4096,
        h: int = 32,
        w: int = 32,
        keep_frac: float = 0.1,
        seed: int = 42,
        families: Optional[list[str]] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.split = split
        self.n_samples = int(n_samples)
        self.h = int(h)
        self.w = int(w)
        self.keep_frac = float(keep_frac)
        self.families = families or list(CONST_FAMILY_TO_ID.keys())
        base = {"train": 10_000_000, "val": 11_000_000, "test": 12_000_000}[split]
        self._seeds = [base + seed * 10_000 + i for i in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float | str | int]:
        seed = self._seeds[idx]
        rng = np.random.default_rng(seed)
        family = str(rng.choice(self.families))
        s = make_constitutive_sample(
            h=self.h, w=self.w, keep_frac=self.keep_frac, family=family, seed=seed
        )
        sparse = torch.from_numpy(s["sparse"])
        mask = torch.from_numpy(s["mask"])
        x = torch.cat([sparse, mask], dim=0)
        return {
            "x": x,
            "dense": torch.from_numpy(s["dense"]),
            "mask": mask,
            "theta": torch.from_numpy(s["theta"]),
            "theta_mask": torch.from_numpy(s["theta_mask"]),
            "family": str(s["family"]),
            "family_id": int(s["family_id"]),
            "seed": int(s["seed"]),
        }


class FluidConstitutiveModel(nn.Module):
    """Sparse v → dense v̂ + family logits + θ̂ (eta0, eta_inf, n, lambda)."""

    def __init__(
        self,
        h: int = 32,
        w: int = 32,
        embed_dim: int = 64,
        dropout: float = 0.1,
        principle: str = "huygens_fresnel",
        n_families: int = 2,
        n_theta: int = N_THETA,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.embed_dim = embed_dim
        self.sample_rate = float(h)
        self.n_theta = n_theta

        self.patch = nn.Conv2d(3, embed_dim, kernel_size=1)
        self.medium_net = TemporalMediumDensity(channels=embed_dim, hidden=32)
        self.encoder = MultiScaleHuygensEncoder(
            embed_dim=embed_dim,
            scale_specs=fluid_scale_specs(embed_dim),
            gamma=0.5,
            omega=0.3,
            wave_speed=1.0,
            dropout=dropout,
            sparse_band=False,
            principle=principle,
        )
        self.decode = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )
        self.family_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_families),
        )
        # Softplus ensures positivity; n also kept positive then optionally clamped in loss
        self.theta_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_theta),
            nn.Softplus(),
        )

    def _time_axis(self, batch: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(length, device=device, dtype=dtype) / self.sample_rate
        return t.view(1, length, 1).expand(batch, -1, -1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if h != self.h or w != self.w:
            x = nn.functional.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=False)
            h, w = self.h, self.w
        feat = self.patch(x)
        seq = feat.flatten(2).transpose(1, 2)
        rho = self.medium_net(seq)
        h_imag = torch.zeros_like(seq)
        t = self._time_axis(b, seq.size(1), seq.device, seq.dtype)
        h_real, h_imag = self.encoder(seq, h_imag, t=t, rho=rho)
        env = torch.sqrt(h_real.pow(2) + h_imag.pow(2) + 1e-8)
        return env, rho

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        env, rho = self.encode(x)
        b = x.size(0)
        dense = self.decode(env).transpose(1, 2).reshape(b, 2, self.h, self.w)
        pooled = env.mean(dim=1)
        family_logits = self.family_head(pooled)
        theta = self.theta_head(pooled)
        return {
            "dense": dense,
            "family_logits": family_logits,
            "theta": theta,
            "rho": rho,
            "pooled": pooled,
        }

    def collect_kernel_params(self) -> dict[str, dict[str, float]]:
        params: dict[str, dict[str, float]] = {}
        for si, branch in enumerate(self.encoder.branches):
            for li, layer in enumerate(branch.layers):
                k = layer.kernel
                params[f"scale{si}_layer{li}"] = {
                    "gamma": float(k.effective_gamma().detach().cpu()),
                    "omega": float(k.effective_omega().detach().cpu()),
                    "wave_speed": float(k.effective_wave_speed().detach().cpu()),
                }
        return params


def constitutive_loss(
    out: dict[str, torch.Tensor],
    batch: dict,
    *,
    vel_weight: float = 1.0,
    family_weight: float = 1.0,
    theta_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    dense = batch["dense"].to(out["dense"].device)
    loss_v = mse(out["dense"], dense)
    y_fam = torch.as_tensor(batch["family_id"], device=out["dense"].device, dtype=torch.long)
    loss_f = ce(out["family_logits"], y_fam)
    theta_t = batch["theta"].to(out["dense"].device)
    mask = batch["theta_mask"].to(out["dense"].device)
    # Relative-ish MSE on active params
    diff = (out["theta"] - theta_t) * mask
    denom = (theta_t.abs() * mask).sum().clamp_min(1e-6)
    loss_th = (diff.pow(2).sum() / denom)
    # Also absolute on masked entries for stability
    loss_th = loss_th + mse(out["theta"] * mask, theta_t * mask)
    loss = vel_weight * loss_v + family_weight * loss_f + theta_weight * loss_th
    stats = {
        "loss": float(loss.detach()),
        "loss_v": float(loss_v.detach()),
        "loss_f": float(loss_f.detach()),
        "loss_th": float(loss_th.detach()),
    }
    return loss, stats


# Re-exports for callers
__all__ = [
    "ConstitutiveFluidDataset",
    "FluidConstitutiveModel",
    "constitutive_loss",
    "CONST_FAMILY_TO_ID",
    "CONST_ID_TO_FAMILY",
    "N_THETA",
]
