# -*- coding: utf-8 -*-
"""Physics losses for the Physics Decoder (travel-time + soft priors)."""

from __future__ import annotations

import torch

from hnf.inversion_1d import LayeredEarth1D, model_rmse, travel_time_phase
from hnf.zhizi_physics_head import PhysicsHeadOutput


def travel_time_loss(
    earth: LayeredEarth1D,
    source_depth: torch.Tensor | float,
    receiver_distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    src = (
        source_depth
        if isinstance(source_depth, torch.Tensor)
        else torch.tensor(source_depth, dtype=earth.vp.dtype, device=earth.vp.device)
    )
    tp = travel_time_phase(earth, "P", src, receiver_distances)
    ts = travel_time_phase(earth, "S", src, receiver_distances)
    loss_tp = torch.mean((tp - obs_tp) ** 2)
    loss_ts = torch.mean((ts - obs_ts) ** 2)
    loss = loss_tp + loss_ts
    return loss, {
        "loss_tt": float(loss.detach()),
        "loss_tp": float(loss_tp.detach()),
        "loss_ts": float(loss_ts.detach()),
    }


def soft_anchor_loss(
    output: PhysicsHeadOutput,
    anchor_weight_vp: float = 0.01,
    anchor_weight_vs: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Soft pull toward latent kernel-derived prior (uncalibrated)."""
    l_vp = torch.mean((output.vp - output.vp_prior) ** 2)
    l_vs = torch.mean((output.vs - output.vs_prior) ** 2)
    loss = anchor_weight_vp * l_vp + anchor_weight_vs * l_vs
    return loss, {
        "loss_anchor": float(loss.detach()),
        "loss_anchor_vp": float(l_vp.detach()),
        "loss_anchor_vs": float(l_vs.detach()),
    }


def rho_weighted_smoothness(
    vp: torch.Tensor,
    rho_layers: torch.Tensor,
    weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    rho layers high -> allow larger vp jumps (less smoothness penalty).

    rho is uncalibrated; used only as relative weights.
    """
    if vp.dim() == 1:
        vp = vp.unsqueeze(0)
    if rho_layers.dim() == 1:
        rho_layers = rho_layers.unsqueeze(0)
    dvp = vp[:, 1:] - vp[:, :-1]
    rho_norm = rho_layers / rho_layers.mean(dim=-1, keepdim=True).clamp(min=1e-4)
    w = 1.0 / rho_norm[:, 1:].clamp(min=0.25, max=4.0)
    smooth = torch.mean((dvp ** 2) * w)
    loss = weight * smooth
    return loss, {"loss_rho_smooth": float(loss.detach())}


def supervised_vp_loss(
    pred_vp: torch.Tensor,
    true_vp: torch.Tensor,
    weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred_vp.dim() == 1:
        pred_vp = pred_vp.unsqueeze(0)
    if true_vp.dim() == 1:
        true_vp = true_vp.unsqueeze(0)
    loss = weight * torch.mean((pred_vp - true_vp) ** 2)
    return loss, {"loss_vp_sup": float(loss.detach())}


def zhizi_inversion_loss(
    output: PhysicsHeadOutput,
    depths: torch.Tensor,
    q: torch.Tensor,
    source_depth: float,
    receiver_distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
    rho_layers: torch.Tensor,
    true_vp: torch.Tensor | None = None,
    true_vs: torch.Tensor | None = None,
    tt_weight: float = 1.0,
    anchor_weight: float = 0.01,
    rho_smooth_weight: float = 0.05,
    vp_sup_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    earth = LayeredEarth1D(
        depths=depths,
        vp=output.vp[0],
        vs=output.vs[0],
        q=q,
    )
    total = torch.tensor(0.0, device=output.vp.device)
    metrics: dict[str, float] = {}

    l_tt, m_tt = travel_time_loss(earth, source_depth, receiver_distances, obs_tp, obs_ts)
    total = total + tt_weight * l_tt
    metrics.update(m_tt)

    l_a, m_a = soft_anchor_loss(output, anchor_weight_vp=anchor_weight, anchor_weight_vs=anchor_weight)
    total = total + l_a
    metrics.update(m_a)

    l_r, m_r = rho_weighted_smoothness(output.vp, rho_layers, weight=rho_smooth_weight)
    total = total + l_r
    metrics.update(m_r)

    if vp_sup_weight > 0 and true_vp is not None:
        l_v, m_v = supervised_vp_loss(output.vp, true_vp, weight=vp_sup_weight)
        total = total + l_v
        metrics.update(m_v)

    if true_vp is not None and true_vs is not None:
        true_earth = LayeredEarth1D(depths=depths, vp=true_vp, vs=true_vs, q=q)
        rec_earth = earth
        rmse = model_rmse(true_earth, rec_earth)
        metrics.update({f"rmse_{k}": v for k, v in rmse.items()})

    metrics["loss"] = float(total.detach())
    return total, metrics
