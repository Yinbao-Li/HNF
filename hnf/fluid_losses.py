# -*- coding: utf-8 -*-
"""Shared reconstruction losses for Domain-III fluid models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hnf.fluid_spatial import curl_2d

try:
    from hnf.fluid_spatial3d import curl_3d
except ImportError:
    curl_3d = None  # type: ignore

try:
    from hnf.fluid_spatial4d import curl_4d
except ImportError:
    curl_4d = None  # type: ignore


def velocity_recon_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    obs_mask: torch.Tensor,
    *,
    region_mask: torch.Tensor | None = None,
    obs_weight: float = 0.25,
    recon_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Split MSE: sparse observed pixels vs missing (reconstruction) pixels.

    ``obs_mask`` and optional ``region_mask`` are (B, 1, H, W) in {0, 1}.
    """
    if obs_mask.dim() == 3:
        obs_mask = obs_mask.unsqueeze(1)
    obs = obs_mask.clamp(0.0, 1.0)
    if region_mask is not None:
        if region_mask.dim() == 3:
            region_mask = region_mask.unsqueeze(1)
        region = region_mask.clamp(0.0, 1.0)
        obs = obs * region
        recon_mask = (1.0 - obs_mask).clamp(0.0, 1.0) * region
    else:
        recon_mask = (1.0 - obs_mask).clamp(0.0, 1.0)

    err = (pred - gt).pow(2).sum(dim=1, keepdim=True)
    loss_obs = (err * obs).sum() / obs.sum().clamp_min(eps)
    loss_recon = (err * recon_mask).sum() / recon_mask.sum().clamp_min(eps)
    total = float(obs_weight) * loss_obs + float(recon_weight) * loss_recon
    stats = {
        "loss_obs": float(loss_obs.detach().item()),
        "loss_recon": float(loss_recon.detach().item()),
    }
    return total, stats


def normalized_curl_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-invariant curl matching; supports 2D/3D/4D velocity tensors."""
    if pred.dim() == 4:
        cp, cg = curl_2d(pred), curl_2d(gt)
    elif pred.dim() == 5 and curl_3d is not None:
        cp, cg = curl_3d(pred), curl_3d(gt)
    elif pred.dim() == 6 and curl_4d is not None:
        cp, cg = curl_4d(pred), curl_4d(gt)
    else:
        raise ValueError(f"unsupported velocity shape {tuple(pred.shape)}")
    scale = cg.pow(2).mean().detach().clamp_min(eps)
    return (cp - cg).pow(2).mean() / scale


def curl_weight_at_epoch(epoch: int, max_weight: float, warmup_epochs: int) -> float:
    if max_weight <= 0 or warmup_epochs <= 0:
        return float(max_weight) if epoch > 0 else 0.0
    if epoch <= warmup_epochs:
        return float(max_weight) * (epoch / warmup_epochs)
    return float(max_weight)


def lr_scale_at_epoch(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def rel_err_masked(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    if mask is None:
        num = (pred - gt).pow(2).sum().sqrt()
        den = gt.pow(2).sum().sqrt().clamp_min(1e-8)
        return float((num / den).item())
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    m = mask.clamp(0, 1)
    diff2 = ((pred - gt).pow(2) * m).sum()
    gt2 = (gt.pow(2) * m).sum().clamp_min(1e-8)
    return float((diff2.sqrt() / gt2.sqrt()).item())
