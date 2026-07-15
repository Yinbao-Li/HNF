#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Huygens physics picking model on STEAD."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    det_pred_from_logits,
    finalize_metrics,
    picking_score,
    tolerance_bins,
    update_detection_counts,
    update_picking_counts,
)
from hnf.picking_model import build_picking_model, load_picking_model_state
from hnf.noise_cancel import noise_cancel_losses
from hnf.stead_picking_dataset import STEADPickingDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Huygens physics picking on STEAD")
    p.add_argument("--seq-len", type=int, default=400)
    p.add_argument("--max-event-train", type=int, default=240000)
    p.add_argument("--max-noise-train", type=int, default=120000)
    p.add_argument("--max-val", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-shared-layers", type=int, default=2)
    p.add_argument("--num-branch-layers", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=0.3)
    p.add_argument("--vp", type=float, default=8.0)
    p.add_argument("--vs", type=float, default=4.5)
    p.add_argument("--local-window-sec", type=float, default=15.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--label-sigma-sec", type=float, default=0.4)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--pick-tolerance-sec", type=float, default=0.5)
    p.add_argument("--pick-loss-weight", type=float, default=2.0)
    p.add_argument("--pick-pos-weight", type=float, default=25.0)
    p.add_argument("--noise-pick-penalty", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--output-dir", default="outputs/stead_hnf_picking")
    p.add_argument("--resume", default=None, help="Checkpoint path to fine-tune from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cudnn-benchmark", action="store_true", default=True)
    p.add_argument("--per-time-det", action="store_true")
    p.add_argument("--pick-head-hidden", type=int, default=24)
    p.add_argument("--pick-head-kernel", type=int, default=7)
    p.add_argument("--pick-head-layers", type=int, default=3)
    p.add_argument("--focal-gamma", type=float, default=0.0, help=">0 enables focal BCE for picking")
    p.add_argument("--det-loss-weight", type=float, default=1.0)
    p.add_argument("--det-event-weight", type=float, default=1.0, help="BCE weight on event traces")
    p.add_argument("--p-pick-loss-weight", type=float, default=1.0, help="Extra multiplier on P pick loss")
    p.add_argument("--s-pick-loss-weight", type=float, default=1.0, help="Extra multiplier on S pick loss")
    p.add_argument("--ps-order-loss-weight", type=float, default=0.0, help="Penalty when S precedes P")
    p.add_argument("--ps-min-gap-sec", type=float, default=0.1, help="Min P-S gap in seconds for order loss")
    p.add_argument(
        "--distance-gap-loss-weight",
        type=float,
        default=0.0,
        help="Consistency of soft P/S gap vs catalog distance prior (s_per_km*km)",
    )
    p.add_argument("--distance-gap-s-per-km", type=float, default=0.119)
    p.add_argument("--distance-gap-sigma-sec", type=float, default=1.2)
    p.add_argument("--predict-ps-gap", action="store_true", help="Enable dedicated S-P interval head")
    p.add_argument("--ps-gap-hidden", type=int, default=64)
    p.add_argument("--ps-gap-loss-weight", type=float, default=0.5, help="Supervised gap head loss weight")
    p.add_argument(
        "--ps-gap-consist-weight",
        type=float,
        default=0.1,
        help="Consistency between predicted gap and soft P/S expectation gap",
    )
    p.add_argument("--wrong-peak-loss-weight", type=float, default=0.0, help="Rank GT window above distant wrong peaks")
    p.add_argument("--wrong-peak-radius-sec", type=float, default=0.5)
    p.add_argument("--wrong-peak-margin", type=float, default=0.2)
    p.add_argument("--s-wrong-peak-scale", type=float, default=1.0)
    p.add_argument("--wrong-peak-listwise-weight", type=float, default=0.0, help="Listwise CE over top-k peaks vs GT")
    p.add_argument("--wrong-peak-listwise-topk", type=int, default=5)
    p.add_argument(
        "--rho-sparsity-weight",
        type=float,
        default=0.0,
        help="L1 on rho outside P/S event windows (event-driven medium)",
    )
    p.add_argument(
        "--rho-sparsity-radius-sec",
        type=float,
        default=1.5,
        help="Keep-window radius around GT P/S for rho sparsity",
    )
    p.add_argument(
        "--kernel-phys-prior-weight",
        type=float,
        default=0.0,
        help="Relative L2 prior on effective gamma/omega/c toward construction anchors",
    )
    p.add_argument("--peak-rerank", action="store_true", help="Learned local competition head on P/S logits")
    p.add_argument("--peak-rerank-hidden", type=int, default=16)
    p.add_argument("--post-process-p-before-s", action="store_true")
    p.add_argument("--score-mode", choices=["mean", "det_guard", "pick_focus"], default="mean")
    p.add_argument("--det-score-floor", type=float, default=0.985)
    p.add_argument("--reset-best-score", action="store_true", help="Ignore resume score when selecting best.pt")
    p.add_argument("--continue", action="store_true", dest="continue_train", help="Resume next epoch from checkpoint")
    p.add_argument("--freeze-backbone-epochs", type=int, default=0)
    p.add_argument("--freeze-det-epochs", type=int, default=0)
    p.add_argument("--freeze-all-but-det-epochs", type=int, default=0)
    p.add_argument("--enhanced-det-head", action="store_true")
    p.add_argument("--noise-cancel", action="store_true", help="Enable Huygens 3-step noise cancellation front-end")
    p.add_argument("--noise-source-dim", type=int, default=16)
    p.add_argument("--noise-det-pick-split", action="store_true", help="Use denoised path for det and raw path for P/S")
    p.add_argument("--noise-pick-cues", action="store_true", help="Fuse denoise cues into P/S path without replacing raw waveform")
    p.add_argument("--noise-cancel-weight", type=float, default=0.4)
    p.add_argument("--nc-consistency-weight", type=float, default=0.5)
    p.add_argument("--nc-phase-weight", type=float, default=0.1)
    p.add_argument("--nc-preserve-weight", type=float, default=0.3)
    p.add_argument("--nc-energy-weight", type=float, default=0.05)
    p.add_argument("--nc-noise-suppress-weight", type=float, default=0.2)
    p.add_argument("--freeze-all-but-noise-epochs", type=int, default=0)
    p.add_argument("--freeze-all-but-pick-epochs", type=int, default=0)
    p.add_argument("--freeze-all-but-gap-epochs", type=int, default=0, help="Train only ps_gap_head")
    p.add_argument("--multi-scale", action="store_true", help="Use multi-scale DeepHuygens encoder")
    p.add_argument("--sparse-band", action="store_true", help="Banded sparse light-cone matmul")
    p.add_argument("--num-anchors", type=int, default=0, help="Low-rank anchor count for shared propagation (0=off)")
    p.add_argument(
        "--principle",
        choices=["huygens", "huygens_fresnel"],
        default="huygens",
        help="Wave kernel principle (huygens baseline vs Huygens–Fresnel obliquity)",
    )
    p.add_argument("--obliquity-scale", type=float, default=1.0, help="Fresnel obliquity lateral scale")
    p.add_argument(
        "--obliquity-mode",
        default="none",
        choices=[
            "none",
            "soft_all",
            "soft_shared",
            "det_soft",
            "det_fresnel",
            "noise_soft",
            "noise_fresnel",
            "full_fresnel",
        ],
        help="Where to apply obliquity χ (soft blend or Fresnel)",
    )
    p.add_argument(
        "--obliquity-mix",
        type=float,
        default=0.25,
        help="Soft χ blend weight in [0,1] for huygens+χ modes",
    )
    p.set_defaults(residual_pick_head=True, residual_det_head=True)
    p.add_argument("--no-residual-pick-head", action="store_false", dest="residual_pick_head")
    p.add_argument("--no-residual-det-head", action="store_false", dest="residual_det_head")
    p.add_argument("--augment", action="store_true", help="Train-time waveform augmentation")
    p.add_argument("--aug-amp-min", type=float, default=0.5)
    p.add_argument("--aug-amp-max", type=float, default=2.0)
    p.add_argument("--aug-noise-snr-min", type=float, default=5.0)
    p.add_argument("--aug-noise-snr-max", type=float, default=20.0)
    p.add_argument("--aug-time-shift-sec", type=float, default=0.05)
    return p.parse_args()


def move_batch_to_device(batch: dict, device: torch.device, non_blocking: bool = False) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_pick_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    weight = torch.where(target > 0.05, pos_weight, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if focal_gamma > 0.0:
        prob = torch.sigmoid(logits)
        pt = torch.where(target > 0.05, prob, 1.0 - prob)
        bce = (1.0 - pt).pow(focal_gamma) * bce
    return (bce * weight).mean()


def ps_order_loss(
    p_logits: torch.Tensor,
    s_logits: torch.Tensor,
    event_mask: torch.Tensor,
    seq_len: int,
    min_gap_sec: float,
) -> torch.Tensor:
    """Soft constraint: S expected time should follow P by at least min_gap_sec."""
    if not event_mask.any():
        return torch.tensor(0.0, device=p_logits.device)
    p_prob = torch.sigmoid(p_logits[event_mask])
    s_prob = torch.sigmoid(s_logits[event_mask])
    t = torch.arange(p_prob.size(-1), device=p_prob.device, dtype=p_prob.dtype)
    p_exp = (p_prob * t).sum(dim=-1) / p_prob.sum(dim=-1).clamp_min(1e-6)
    s_exp = (s_prob * t).sum(dim=-1) / s_prob.sum(dim=-1).clamp_min(1e-6)
    min_gap_bins = max(1.0, min_gap_sec * seq_len / 60.0)
    return F.relu(p_exp + min_gap_bins - s_exp).mean()


def distance_gap_consistency_loss(
    p_logits: torch.Tensor,
    s_logits: torch.Tensor,
    distance_km: torch.Tensor,
    event_mask: torch.Tensor,
    p_valid: torch.Tensor,
    s_valid: torch.Tensor,
    seq_len: int,
    *,
    s_per_km: float = 0.119,
    sigma_sec: float = 1.2,
) -> torch.Tensor:
    """Pull soft P/S expectations toward catalog distance S−P prior.

    Uses STEAD source_distance_km: μ = s_per_km * distance. This injects the
    same geometric knowledge that helped at inference (fix_weak_distance)
    into training, without an external teacher model.
    """
    if distance_km is None:
        return torch.tensor(0.0, device=p_logits.device)
    mask = (
        event_mask
        & (p_valid > 0)
        & (s_valid > 0)
        & torch.isfinite(distance_km)
        & (distance_km > 0)
    )
    if not mask.any():
        return torch.tensor(0.0, device=p_logits.device)

    p_prob = torch.sigmoid(p_logits[mask])
    s_prob = torch.sigmoid(s_logits[mask])
    t = torch.arange(p_prob.size(-1), device=p_prob.device, dtype=p_prob.dtype)
    p_exp = (p_prob * t).sum(dim=-1) / p_prob.sum(dim=-1).clamp_min(1e-6)
    s_exp = (s_prob * t).sum(dim=-1) / s_prob.sum(dim=-1).clamp_min(1e-6)
    gap_sec = (s_exp - p_exp) * (60.0 / float(seq_len))
    mu = distance_km[mask] * float(s_per_km)
    # Softplus keeps gap positive preference; Huber on residual vs μ
    resid = gap_sec - mu
    return F.smooth_l1_loss(resid / max(sigma_sec, 1e-3), torch.zeros_like(resid))


def ps_gap_head_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    seq_len: int,
    consist_weight: float = 0.1,
) -> torch.Tensor:
    """Supervise predicted S-P gap; optional consistency with soft pick expectations."""
    if "ps_gap_sec" not in outputs:
        return torch.tensor(0.0, device=batch["det"].device)
    event_mask = (batch["det"] > 0.5) & (batch["p_valid"] > 0) & (batch["s_valid"] > 0)
    if not event_mask.any():
        return torch.tensor(0.0, device=batch["det"].device)

    pred = outputs["ps_gap_sec"][event_mask]
    log_sigma = outputs.get("ps_gap_log_sigma")
    if log_sigma is not None:
        log_sigma = log_sigma[event_mask]
    gt = (batch["s_idx"][event_mask].float() - batch["p_idx"][event_mask].float()) * (60.0 / float(seq_len))
    gt = gt.clamp(min=0.2, max=30.0)

    # Heteroscedastic Gaussian NLL on seconds (stable via softplus sigma).
    if log_sigma is not None:
        sigma = F.softplus(log_sigma) + 0.2
        nll = 0.5 * ((pred - gt) / sigma) ** 2 + torch.log(sigma)
        loss = nll.mean()
    else:
        loss = F.smooth_l1_loss(pred, gt)

    if consist_weight > 0:
        p_prob = torch.sigmoid(outputs["p"][event_mask])
        s_prob = torch.sigmoid(outputs["s"][event_mask])
        t = torch.arange(p_prob.size(-1), device=p_prob.device, dtype=p_prob.dtype)
        p_exp = (p_prob * t).sum(dim=-1) / p_prob.sum(dim=-1).clamp_min(1e-6)
        s_exp = (s_prob * t).sum(dim=-1) / s_prob.sum(dim=-1).clamp_min(1e-6)
        soft_gap = ((s_exp - p_exp) * (60.0 / float(seq_len))).clamp(min=0.0, max=30.0)
        loss = loss + consist_weight * F.smooth_l1_loss(pred, soft_gap.detach())
    return loss


def wrong_peak_rank_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    valid_mask: torch.Tensor,
    seq_len: int,
    radius_sec: float = 0.5,
    margin: float = 0.2,
) -> torch.Tensor:
    """Encourage the GT neighborhood peak to outrank distant spurious peaks."""
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)
    logits = logits[valid_mask]
    gt = target_idx[valid_mask]
    t = torch.arange(seq_len, device=logits.device).view(1, -1)
    radius_bins = max(1, int(round(radius_sec * seq_len / 60.0)))
    local_mask = (t - gt.unsqueeze(1)).abs() <= radius_bins
    neg_fill = torch.full_like(logits, -1e9)
    pos_max = torch.where(local_mask, logits, neg_fill).amax(dim=-1)
    neg_max = torch.where(~local_mask, logits, neg_fill).amax(dim=-1)
    return F.relu(margin - pos_max + neg_max).mean()


def wrong_peak_listwise_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    valid_mask: torch.Tensor,
    seq_len: int,
    radius_sec: float = 0.5,
    topk: int = 5,
    temperature: float = 0.75,
) -> torch.Tensor:
    """Listwise CE over top-k peaks: push mass onto the GT-neighborhood peak.

    Addresses close-race wrong peaks where GT is a local max but not global argmax.
    """
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)
    logits = logits[valid_mask]
    gt = target_idx[valid_mask]
    radius_bins = max(1, int(round(radius_sec * seq_len / 60.0)))
    # top-k global peaks as hard negatives + ensure GT window max is included
    k = min(topk, logits.size(-1))
    top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
    # GT window max location
    t = torch.arange(seq_len, device=logits.device).view(1, -1)
    local_mask = (t - gt.unsqueeze(1)).abs() <= radius_bins
    neg_fill = torch.full_like(logits, -1e9)
    pos_vals = torch.where(local_mask, logits, neg_fill)
    pos_idx = pos_vals.argmax(dim=-1)
    # Build candidate set: topk union pos_idx
    cand_idx = top_idx
    # Replace last slot with pos_idx when missing
    has_pos = (cand_idx == pos_idx.unsqueeze(1)).any(dim=-1)
    if (~has_pos).any():
        cand_idx = cand_idx.clone()
        cand_idx[~has_pos, -1] = pos_idx[~has_pos]
    cand_logits = torch.gather(logits, 1, cand_idx) / max(temperature, 1e-3)
    # Target: which candidate is closest to GT (prefer exact pos_idx)
    target = (cand_idx == pos_idx.unsqueeze(1)).float().argmax(dim=-1)
    return F.cross_entropy(cand_logits, target)


def rho_event_sparsity_loss(
    rho: torch.Tensor,
    batch: dict[str, torch.Tensor],
    seq_len: int,
    radius_sec: float = 1.5,
) -> torch.Tensor:
    """Penalize medium density outside P/S arrival windows (and on noise traces)."""
    if rho.dim() == 3 and rho.size(-1) == 1:
        rho = rho.squeeze(-1)
    b, t_len = rho.shape
    dt = 60.0 / max(seq_len - 1, 1)
    radius = max(1, int(round(radius_sec / dt)))
    t = torch.arange(t_len, device=rho.device).view(1, -1).expand(b, -1)
    keep = torch.zeros(b, t_len, device=rho.device, dtype=torch.bool)
    event = batch["det"] > 0.5
    for idx_key, valid_key in (("p_idx", "p_valid"), ("s_idx", "s_valid")):
        idx = batch[idx_key].long().clamp(0, t_len - 1).unsqueeze(1)
        valid = event & (batch[valid_key] > 0)
        keep = keep | (valid.unsqueeze(1) & ((t - idx).abs() <= radius))
    outside = rho * (~keep).float()
    loss = outside.mean()
    noise = ~event
    if noise.any():
        loss = loss + rho[noise].mean()
    return loss


def kernel_physics_prior_loss(model: nn.Module) -> torch.Tensor:
    """Average relative L2 prior over all HuygensKernel modules."""
    from hnf.kernel import HuygensKernel

    terms: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, HuygensKernel):
            terms.append(module.physics_prior_loss())
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).mean()


def remap_omega_after_legacy_resume(model: nn.Module, ckpt_state: dict) -> int:
    """Old checkpoints stored raw ω; new code uses softplus(ω). Preserve effective ω."""
    from hnf.kernel import HuygensKernel

    if any(k.endswith("c_log_scale") for k in ckpt_state):
        return 0
    n = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HuygensKernel) and module._learnable_omega:
                w = module.omega.data.clamp_min(1e-3)
                module.omega.data.copy_(torch.log(torch.expm1(w)))
                n += 1
    return n


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    pick_loss_weight: float = 2.0,
    pick_pos_weight: float = 25.0,
    noise_pick_penalty: float = 0.05,
    focal_gamma: float = 0.0,
    det_loss_weight: float = 1.0,
    det_event_weight: float = 1.0,
    p_pick_loss_weight: float = 1.0,
    s_pick_loss_weight: float = 1.0,
    ps_order_loss_weight: float = 0.0,
    ps_min_gap_sec: float = 0.1,
    distance_gap_loss_weight: float = 0.0,
    distance_gap_s_per_km: float = 0.119,
    distance_gap_sigma_sec: float = 1.2,
    ps_gap_loss_weight: float = 0.0,
    ps_gap_consist_weight: float = 0.1,
    wrong_peak_loss_weight: float = 0.0,
    wrong_peak_radius_sec: float = 0.5,
    wrong_peak_margin: float = 0.2,
    s_wrong_peak_scale: float = 1.0,
    wrong_peak_listwise_weight: float = 0.0,
    wrong_peak_listwise_topk: int = 5,
    rho_sparsity_weight: float = 0.0,
    rho_sparsity_radius_sec: float = 1.5,
    kernel_phys_prior_weight: float = 0.0,
    model: Optional[nn.Module] = None,
    seq_len: int = 400,
    noise_cancel_weight: float = 0.0,
    nc_consistency_weight: float = 0.5,
    nc_phase_weight: float = 0.1,
    nc_preserve_weight: float = 0.3,
    nc_energy_weight: float = 0.05,
    nc_noise_suppress_weight: float = 0.2,
) -> torch.Tensor:
    det_target = batch["det"]
    det_logits = outputs["det"]
    if det_logits.dim() == 1:
        det_w = torch.where(det_target > 0.5, det_event_weight, 1.0)
        det_loss = F.binary_cross_entropy_with_logits(
            det_logits, det_target, weight=det_w
        )
    else:
        det_t = det_target.unsqueeze(1).expand_as(det_logits)
        det_loss = F.binary_cross_entropy_with_logits(det_logits, det_t)

    event_mask = det_target > 0.5
    pick_loss = torch.tensor(0.0, device=det_target.device)
    if event_mask.any():
        for head_name, target_name, branch_w in [
            ("p", "p_target", p_pick_loss_weight),
            ("s", "s_target", s_pick_loss_weight),
        ]:
            head_loss = weighted_pick_loss(
                outputs[head_name][event_mask],
                batch[target_name][event_mask],
                pos_weight=pick_pos_weight,
                focal_gamma=focal_gamma,
            )
            pick_loss = pick_loss + branch_w * head_loss

    noise_mask = ~event_mask
    if noise_mask.any() and noise_pick_penalty > 0:
        for head_name in ("p", "s"):
            peak_logits = outputs[head_name][noise_mask].amax(dim=-1)
            pick_loss = pick_loss + noise_pick_penalty * F.softplus(peak_logits).mean()

    order_loss = torch.tensor(0.0, device=det_target.device)
    if ps_order_loss_weight > 0 and event_mask.any():
        order_loss = ps_order_loss(
            outputs["p"],
            outputs["s"],
            event_mask,
            seq_len=seq_len,
            min_gap_sec=ps_min_gap_sec,
        )

    dist_gap_loss = torch.tensor(0.0, device=det_target.device)
    if distance_gap_loss_weight > 0 and event_mask.any() and "source_distance_km" in batch:
        dist_gap_loss = distance_gap_consistency_loss(
            outputs["p"],
            outputs["s"],
            batch["source_distance_km"],
            event_mask,
            batch["p_valid"],
            batch["s_valid"],
            seq_len=seq_len,
            s_per_km=distance_gap_s_per_km,
            sigma_sec=distance_gap_sigma_sec,
        )

    gap_loss = torch.tensor(0.0, device=det_target.device)
    if ps_gap_loss_weight > 0 and "ps_gap_sec" in outputs:
        gap_loss = ps_gap_head_loss(
            outputs,
            batch,
            seq_len=seq_len,
            consist_weight=ps_gap_consist_weight,
        )

    wrong_peak_loss = torch.tensor(0.0, device=det_target.device)
    if wrong_peak_loss_weight > 0 and event_mask.any():
        p_valid = event_mask & (batch["p_valid"] > 0)
        s_valid = event_mask & (batch["s_valid"] > 0)
        wrong_peak_loss = wrong_peak_rank_loss(
            outputs["p"],
            batch["p_idx"],
            p_valid,
            seq_len=seq_len,
            radius_sec=wrong_peak_radius_sec,
            margin=wrong_peak_margin,
        )
        wrong_peak_loss = wrong_peak_loss + s_wrong_peak_scale * wrong_peak_rank_loss(
            outputs["s"],
            batch["s_idx"],
            s_valid,
            seq_len=seq_len,
            radius_sec=wrong_peak_radius_sec,
            margin=wrong_peak_margin,
        )
        if wrong_peak_listwise_weight > 0:
            lw = wrong_peak_listwise_loss(
                outputs["p"],
                batch["p_idx"],
                p_valid,
                seq_len=seq_len,
                radius_sec=wrong_peak_radius_sec,
                topk=wrong_peak_listwise_topk,
            )
            lw = lw + s_wrong_peak_scale * wrong_peak_listwise_loss(
                outputs["s"],
                batch["s_idx"],
                s_valid,
                seq_len=seq_len,
                radius_sec=wrong_peak_radius_sec,
                topk=wrong_peak_listwise_topk,
            )
            wrong_peak_loss = wrong_peak_loss + wrong_peak_listwise_weight * lw

    rho_loss = torch.tensor(0.0, device=det_target.device)
    if rho_sparsity_weight > 0 and "rho" in outputs:
        rho_loss = rho_event_sparsity_loss(
            outputs["rho"],
            batch,
            seq_len=seq_len,
            radius_sec=rho_sparsity_radius_sec,
        )

    prior_loss = torch.tensor(0.0, device=det_target.device)
    if kernel_phys_prior_weight > 0 and model is not None:
        prior_loss = kernel_physics_prior_loss(model)

    nc_loss = torch.tensor(0.0, device=det_target.device)
    if noise_cancel_weight > 0 and "nc_u_final" in outputs:
        nc_out = {
            "u_final": outputs["nc_u_final"],
            "n_sim": outputs["nc_n_sim"],
        }
        nc_loss, _ = noise_cancel_losses(
            outputs,
            nc_out,
            batch,
            consistency_weight=nc_consistency_weight,
            phase_weight=nc_phase_weight,
            preserve_weight=nc_preserve_weight,
            energy_weight=nc_energy_weight,
            noise_suppress_weight=nc_noise_suppress_weight,
        )

    return (
        det_loss_weight * det_loss
        + pick_loss_weight * pick_loss
        + ps_order_loss_weight * order_loss
        + distance_gap_loss_weight * dist_gap_loss
        + ps_gap_loss_weight * gap_loss
        + wrong_peak_loss_weight * wrong_peak_loss
        + rho_sparsity_weight * rho_loss
        + kernel_phys_prior_weight * prior_loss
        + noise_cancel_weight * nc_loss
    )


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seq_len: int,
    pick_threshold: float,
    pick_tolerance_sec: float,
    pick_loss_weight: float = 2.0,
    pick_pos_weight: float = 25.0,
    noise_pick_penalty: float = 0.05,
    focal_gamma: float = 0.0,
    det_loss_weight: float = 1.0,
    det_event_weight: float = 1.0,
    p_pick_loss_weight: float = 1.0,
    s_pick_loss_weight: float = 1.0,
    ps_order_loss_weight: float = 0.0,
    ps_min_gap_sec: float = 0.1,
    distance_gap_loss_weight: float = 0.0,
    distance_gap_s_per_km: float = 0.119,
    distance_gap_sigma_sec: float = 1.2,
    ps_gap_loss_weight: float = 0.0,
    ps_gap_consist_weight: float = 0.1,
    wrong_peak_loss_weight: float = 0.0,
    wrong_peak_radius_sec: float = 0.5,
    wrong_peak_margin: float = 0.2,
    s_wrong_peak_scale: float = 1.0,
    wrong_peak_listwise_weight: float = 0.0,
    wrong_peak_listwise_topk: int = 5,
    rho_sparsity_weight: float = 0.0,
    rho_sparsity_radius_sec: float = 1.5,
    kernel_phys_prior_weight: float = 0.0,
    post_process_p_before_s: bool = False,
    noise_cancel_weight: float = 0.0,
    nc_consistency_weight: float = 0.5,
    nc_phase_weight: float = 0.1,
    nc_preserve_weight: float = 0.3,
    nc_energy_weight: float = 0.05,
    nc_noise_suppress_weight: float = 0.2,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    skipped = 0
    acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, pick_tolerance_sec)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["x"], batch["t"])
        loss = compute_loss(
            outputs,
            batch,
            pick_loss_weight=pick_loss_weight,
            pick_pos_weight=pick_pos_weight,
            noise_pick_penalty=noise_pick_penalty,
            focal_gamma=focal_gamma,
            det_loss_weight=det_loss_weight,
            det_event_weight=det_event_weight,
            p_pick_loss_weight=p_pick_loss_weight,
            s_pick_loss_weight=s_pick_loss_weight,
            ps_order_loss_weight=ps_order_loss_weight,
            ps_min_gap_sec=ps_min_gap_sec,
            distance_gap_loss_weight=distance_gap_loss_weight,
            distance_gap_s_per_km=distance_gap_s_per_km,
            distance_gap_sigma_sec=distance_gap_sigma_sec,
            ps_gap_loss_weight=ps_gap_loss_weight,
            ps_gap_consist_weight=ps_gap_consist_weight,
            wrong_peak_loss_weight=wrong_peak_loss_weight,
            wrong_peak_radius_sec=wrong_peak_radius_sec,
            wrong_peak_margin=wrong_peak_margin,
            s_wrong_peak_scale=s_wrong_peak_scale,
            wrong_peak_listwise_weight=wrong_peak_listwise_weight,
            wrong_peak_listwise_topk=wrong_peak_listwise_topk,
            rho_sparsity_weight=rho_sparsity_weight,
            rho_sparsity_radius_sec=rho_sparsity_radius_sec,
            kernel_phys_prior_weight=kernel_phys_prior_weight,
            model=model,
            seq_len=seq_len,
            noise_cancel_weight=noise_cancel_weight,
            nc_consistency_weight=nc_consistency_weight,
            nc_phase_weight=nc_phase_weight,
            nc_preserve_weight=nc_preserve_weight,
            nc_energy_weight=nc_energy_weight,
            nc_noise_suppress_weight=nc_noise_suppress_weight,
        )
        if not torch.isfinite(loss):
            skipped += batch["x"].size(0)
            continue
        total_loss += loss.item() * batch["x"].size(0)
        total += batch["x"].size(0)

        det_pred = det_pred_from_logits(outputs["det"])
        det_true = batch["det"] > 0.5
        update_detection_counts(acc, det_pred, det_true)

        p_probs = torch.sigmoid(outputs["p"])
        s_probs = torch.sigmoid(outputs["s"])
        if post_process_p_before_s:
            p_probs, s_probs = apply_p_before_s_constraint(
                p_probs, s_probs, pick_threshold
            )

        for head_name, idx_name, valid_name, counts in [
            ("p", "p_idx", "p_valid", acc.p),
            ("s", "s_idx", "s_valid", acc.s),
        ]:
            probs = p_probs if head_name == "p" else s_probs
            update_picking_counts(
                counts,
                probs,
                det_pred,
                det_true,
                batch[valid_name] > 0,
                batch[idx_name],
                pick_threshold,
                tol,
                seq_len,
            )

    metrics = finalize_metrics(acc)
    metrics["loss"] = total_loss / max(total, 1) if total > 0 else float("nan")
    metrics["skipped_batches"] = skipped
    return metrics


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


BACKBONE_PREFIXES = (
    "source_embed.",
    "medium_net.",
    "shared_layers.",
    "multi_scale_encoder.",
)


PICK_TRAIN_PREFIXES = (
    "p_layers.",
    "s_layers.",
    "p_pick_head.",
    "s_pick_head.",
    "p_peak_rerank.",
    "s_peak_rerank.",
)


def apply_freeze_schedule(model: nn.Module, epoch: int, args: argparse.Namespace) -> None:
    if args.freeze_all_but_gap_epochs > 0 and epoch <= args.freeze_all_but_gap_epochs:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("ps_gap_head.")
        return
    if args.freeze_all_but_noise_epochs > 0 and epoch <= args.freeze_all_but_noise_epochs:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("noise_cancel_branch.")
        return
    if args.freeze_all_but_pick_epochs > 0 and epoch <= args.freeze_all_but_pick_epochs:
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(p) for p in PICK_TRAIN_PREFIXES)
        return
    if args.freeze_all_but_det_epochs > 0 and epoch <= args.freeze_all_but_det_epochs:
        for name, param in model.named_parameters():
            trainable = name.startswith("det_head.") or name.startswith("raw_onset_encoder.")
            param.requires_grad = trainable
        return
    backbone_frozen = epoch <= args.freeze_backbone_epochs
    det_frozen = epoch <= args.freeze_det_epochs
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in BACKBONE_PREFIXES):
            param.requires_grad = not backbone_frozen
        elif name.startswith("det_head."):
            param.requires_grad = not det_frozen
        else:
            param.requires_grad = True


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    val_metrics: dict[str, float],
    score: float,
    args: argparse.Namespace,
    n_params: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "score": score,
            "args": vars(args),
            "n_params": n_params,
        },
        path,
    )

def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = STEADPickingDataset(
        "train",
        seq_len=args.seq_len,
        max_event_traces=args.max_event_train,
        max_noise_traces=args.max_noise_train,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
        augment=args.augment,
        aug_amp_scale=(args.aug_amp_min, args.aug_amp_max),
        aug_noise_snr_db=(args.aug_noise_snr_min, args.aug_noise_snr_max),
        aug_time_shift_sec=args.aug_time_shift_sec,
    )
    val_ds = STEADPickingDataset(
        "val",
        seq_len=args.seq_len,
        max_event_traces=args.max_val // 2,
        max_noise_traces=args.max_val // 2,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
    )
    test_ds = STEADPickingDataset(
        "test",
        seq_len=args.seq_len,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = build_picking_model(
        embed_dim=args.embed_dim,
        num_shared_layers=args.num_shared_layers,
        num_branch_layers=args.num_branch_layers,
        gamma=args.gamma,
        omega=args.omega,
        vp=args.vp,
        vs=args.vs,
        local_window_sec=args.local_window_sec,
        dropout=args.dropout,
        per_time_det=args.per_time_det,
        pick_head_hidden=args.pick_head_hidden,
        pick_head_kernel=args.pick_head_kernel,
        pick_head_layers=args.pick_head_layers,
        multi_scale=args.multi_scale,
        sparse_band=args.sparse_band,
        num_anchors=args.num_anchors,
        residual_pick_head=args.residual_pick_head,
        residual_det_head=args.residual_det_head,
        enhanced_det_head=args.enhanced_det_head,
        noise_cancel=args.noise_cancel,
        noise_source_dim=args.noise_source_dim,
        noise_det_pick_split=args.noise_det_pick_split,
        noise_pick_cues=args.noise_pick_cues,
        principle=args.principle,
        obliquity_scale=args.obliquity_scale,
        obliquity_mode=args.obliquity_mode,
        obliquity_mix=args.obliquity_mix,
        predict_ps_gap=args.predict_ps_gap,
        ps_gap_hidden=args.ps_gap_hidden,
        peak_rerank=args.peak_rerank,
        peak_rerank_hidden=args.peak_rerank_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_score = -1.0
    resume_epoch = 0
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            missing, unexpected = load_picking_model_state(
                model, ckpt["state_dict"], strict=False
            )
            n_omega = remap_omega_after_legacy_resume(model, ckpt["state_dict"])
            if n_omega:
                print(
                    f"[STEAD-HNF-PHYS] remapped {n_omega} omega params for softplus",
                    flush=True,
                )
            model.copy_shared_det_from_pick()
            resume_epoch = int(ckpt.get("epoch", 0))
            src_score = float(ckpt.get("score", -1.0))
            arch_changed = bool(missing or unexpected)
            if args.continue_train:
                start_epoch = resume_epoch + 1
                best_score = src_score
                print(
                    f"[STEAD-HNF-PHYS] continue from epoch {start_epoch}/{args.epochs}",
                    flush=True,
                )
            elif args.reset_best_score or arch_changed:
                best_score = -1.0
                print(
                    f"[STEAD-HNF-PHYS] reset best_score (arch_changed={arch_changed})",
                    flush=True,
                )
            else:
                best_score = src_score
            print(
                f"[STEAD-HNF-PHYS] resumed from {args.resume}  src_epoch={resume_epoch}  "
                f"src_score={src_score:.4f}  missing={len(missing)}  unexpected={len(unexpected)}",
                flush=True,
            )
        except RuntimeError as exc:
            print(
                f"[STEAD-HNF-PHYS] resume failed: {exc}",
                flush=True,
            )
            raise SystemExit(1) from exc

    if start_epoch > args.epochs:
        print(
            f"[STEAD-HNF-PHYS] already at epoch {resume_epoch} >= {args.epochs}; skip training",
            flush=True,
        )
        return

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.continue_train and resume_epoch > 0:
        for _ in range(resume_epoch):
            sched.step()

    print(
        f"[STEAD-HNF-PHYS] device={device} params={n_params} train={len(train_ds)} "
        f"val={len(val_ds)} test={len(test_ds)} seq_len={args.seq_len} bs={args.batch_size} "
        f"accum={args.grad_accum_steps} amp={amp_enabled} vp={args.vp} vs={args.vs} "
        f"window={args.local_window_sec}s workers={args.num_workers} seed={args.seed} "
        f"lr={args.lr} per_time_det={args.per_time_det} focal={args.focal_gamma} "
        f"ps_order_w={args.ps_order_loss_weight} multi_scale={args.multi_scale} "
        f"sparse_band={args.sparse_band} num_anchors={args.num_anchors} "
        f"principle={args.principle} obliquity_mode={args.obliquity_mode} "
        f"obliquity_mix={args.obliquity_mix} obliquity_scale={args.obliquity_scale} "
        f"res_pick={args.residual_pick_head} res_det={args.residual_det_head} augment={args.augment} "
        f"pick_head={args.pick_head_hidden}x{args.pick_head_layers} score_mode={args.score_mode} "
        f"freeze_bb={args.freeze_backbone_epochs} freeze_det={args.freeze_det_epochs} "
        f"freeze_det_only={args.freeze_all_but_det_epochs} freeze_noise_only={args.freeze_all_but_noise_epochs} "
        f"freeze_pick_only={args.freeze_all_but_pick_epochs} enhanced_det={args.enhanced_det_head} "
        f"noise_cancel={args.noise_cancel} det_pick_split={args.noise_det_pick_split} "
        f"noise_pick_cues={args.noise_pick_cues}",
        flush=True,
    )

    run_t0 = time.time()

    history_path = out_dir / "history.csv"
    if args.continue_train and history_path.is_file():
        pass
    else:
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "det_f1",
                    "p_f1",
                    "s_f1",
                    "p_mae_sec",
                    "s_mae_sec",
                    "score",
                    "lr",
                    "ep_time_sec",
                ]
            )

    for epoch in range(start_epoch, args.epochs + 1):
        apply_freeze_schedule(model, epoch, args)
        model.train()
        total_loss = 0.0
        total = 0
        skipped = 0
        opt.zero_grad(set_to_none=True)
        epoch_t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Ep {epoch:02d}/{args.epochs:02d}",
            leave=False,
            ncols=100,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device, non_blocking=True)
            gap_only = (
                args.freeze_all_but_gap_epochs > 0
                and epoch <= args.freeze_all_but_gap_epochs
                and args.predict_ps_gap
            )
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(batch["x"], batch["t"])
                loss = compute_loss(
                    outputs,
                    batch,
                    pick_loss_weight=0.0 if gap_only else args.pick_loss_weight,
                    pick_pos_weight=args.pick_pos_weight,
                    noise_pick_penalty=0.0 if gap_only else args.noise_pick_penalty,
                    focal_gamma=args.focal_gamma,
                    det_loss_weight=0.0 if gap_only else args.det_loss_weight,
                    det_event_weight=args.det_event_weight,
                    p_pick_loss_weight=args.p_pick_loss_weight,
                    s_pick_loss_weight=args.s_pick_loss_weight,
                    ps_order_loss_weight=0.0 if gap_only else args.ps_order_loss_weight,
                    ps_min_gap_sec=args.ps_min_gap_sec,
                    distance_gap_loss_weight=0.0 if gap_only else args.distance_gap_loss_weight,
                    distance_gap_s_per_km=args.distance_gap_s_per_km,
                    distance_gap_sigma_sec=args.distance_gap_sigma_sec,
                    ps_gap_loss_weight=args.ps_gap_loss_weight if args.predict_ps_gap else 0.0,
                    ps_gap_consist_weight=0.0 if gap_only else args.ps_gap_consist_weight,
                    wrong_peak_loss_weight=0.0 if gap_only else args.wrong_peak_loss_weight,
                    wrong_peak_radius_sec=args.wrong_peak_radius_sec,
                    wrong_peak_margin=args.wrong_peak_margin,
                    s_wrong_peak_scale=args.s_wrong_peak_scale,
                    wrong_peak_listwise_weight=0.0 if gap_only else args.wrong_peak_listwise_weight,
                    wrong_peak_listwise_topk=args.wrong_peak_listwise_topk,
                    rho_sparsity_weight=0.0 if gap_only else args.rho_sparsity_weight,
                    rho_sparsity_radius_sec=args.rho_sparsity_radius_sec,
                    kernel_phys_prior_weight=0.0 if gap_only else args.kernel_phys_prior_weight,
                    model=model,
                    seq_len=args.seq_len,
                    noise_cancel_weight=0.0 if gap_only else (args.noise_cancel_weight if args.noise_cancel else 0.0),
                    nc_consistency_weight=args.nc_consistency_weight,
                    nc_phase_weight=args.nc_phase_weight,
                    nc_preserve_weight=args.nc_preserve_weight,
                    nc_energy_weight=args.nc_energy_weight,
                    nc_noise_suppress_weight=args.nc_noise_suppress_weight,
                )
                scaled_loss = loss / args.grad_accum_steps

            if not torch.isfinite(loss):
                skipped += batch["x"].size(0)
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(scaled_loss).backward()

            if step % args.grad_accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss += loss.item() * batch["x"].size(0)
            total += batch["x"].size(0)

            if step % 20 == 0 or step == len(train_loader):
                pbar.set_postfix_str(
                    f"loss={loss.item():.3f} {datetime.now().strftime('%H:%M:%S')} "
                    f"run={_fmt_duration(time.time() - run_t0)}"
                )

        pbar.close()

        if step % args.grad_accum_steps != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        sched.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_loss = total_loss / max(total, 1) if total > 0 else float("nan")
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            seq_len=args.seq_len,
            pick_threshold=args.pick_threshold,
            pick_tolerance_sec=args.pick_tolerance_sec,
            pick_loss_weight=args.pick_loss_weight,
            pick_pos_weight=args.pick_pos_weight,
            noise_pick_penalty=args.noise_pick_penalty,
            focal_gamma=args.focal_gamma,
            det_loss_weight=args.det_loss_weight,
            det_event_weight=args.det_event_weight,
            p_pick_loss_weight=args.p_pick_loss_weight,
            s_pick_loss_weight=args.s_pick_loss_weight,
            ps_order_loss_weight=args.ps_order_loss_weight,
            ps_min_gap_sec=args.ps_min_gap_sec,
            distance_gap_loss_weight=args.distance_gap_loss_weight,
            distance_gap_s_per_km=args.distance_gap_s_per_km,
            distance_gap_sigma_sec=args.distance_gap_sigma_sec,
            ps_gap_loss_weight=args.ps_gap_loss_weight if args.predict_ps_gap else 0.0,
            ps_gap_consist_weight=args.ps_gap_consist_weight,
            wrong_peak_loss_weight=args.wrong_peak_loss_weight,
            wrong_peak_radius_sec=args.wrong_peak_radius_sec,
            wrong_peak_margin=args.wrong_peak_margin,
            s_wrong_peak_scale=args.s_wrong_peak_scale,
            wrong_peak_listwise_weight=args.wrong_peak_listwise_weight,
            wrong_peak_listwise_topk=args.wrong_peak_listwise_topk,
            rho_sparsity_weight=args.rho_sparsity_weight,
            rho_sparsity_radius_sec=args.rho_sparsity_radius_sec,
            kernel_phys_prior_weight=args.kernel_phys_prior_weight,
            post_process_p_before_s=args.post_process_p_before_s,
            noise_cancel_weight=args.noise_cancel_weight if args.noise_cancel else 0.0,
            nc_consistency_weight=args.nc_consistency_weight,
            nc_phase_weight=args.nc_phase_weight,
            nc_preserve_weight=args.nc_preserve_weight,
            nc_energy_weight=args.nc_energy_weight,
            nc_noise_suppress_weight=args.nc_noise_suppress_weight,
        )
        score = picking_score(
            val_metrics,
            mode=args.score_mode,
            det_floor=args.det_score_floor,
        )
        ep_sec = time.time() - epoch_t0

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    train_loss,
                    val_metrics["loss"],
                    val_metrics["det_f1"],
                    val_metrics["p_f1"],
                    val_metrics["s_f1"],
                    val_metrics["p_mae_sec"],
                    val_metrics["s_mae_sec"],
                    score,
                    opt.param_groups[0]["lr"],
                    round(ep_sec, 1),
                ]
            )

        print(
            f"ep {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"det_f1={val_metrics['det_f1']:.4f}  p_f1={val_metrics['p_f1']:.4f}  "
            f"s_f1={val_metrics['s_f1']:.4f}  p_mae={val_metrics['p_mae_sec']:.3f}s  "
            f"s_mae={val_metrics['s_mae_sec']:.3f}s  score={score:.4f}  "
            f"ep_time={_fmt_duration(ep_sec)}  total={_fmt_duration(time.time() - run_t0)}"
            + (f"  skipped={skipped}" if skipped else ""),
            flush=True,
        )

        if score > best_score and torch.isfinite(torch.tensor(score)):
            best_score = score
            _save_checkpoint(
                out_dir / "best.pt",
                epoch=epoch,
                model=model,
                val_metrics=val_metrics,
                score=score,
                args=args,
                n_params=n_params,
            )

        _save_checkpoint(
            out_dir / "last.pt",
            epoch=epoch,
            model=model,
            val_metrics=val_metrics,
            score=score,
            args=args,
            n_params=n_params,
        )

    eval_path = out_dir / "best.pt"
    if not eval_path.is_file():
        eval_path = out_dir / "last.pt"
    if not eval_path.is_file():
        print("[STEAD-HNF-PHYS] no checkpoint saved; skip test eval", flush=True)
        return

    ckpt = torch.load(eval_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        seq_len=args.seq_len,
        pick_threshold=args.pick_threshold,
        pick_tolerance_sec=args.pick_tolerance_sec,
        pick_loss_weight=args.pick_loss_weight,
        pick_pos_weight=args.pick_pos_weight,
        noise_pick_penalty=args.noise_pick_penalty,
        focal_gamma=args.focal_gamma,
        det_loss_weight=args.det_loss_weight,
        det_event_weight=args.det_event_weight,
        p_pick_loss_weight=args.p_pick_loss_weight,
        s_pick_loss_weight=args.s_pick_loss_weight,
        ps_order_loss_weight=args.ps_order_loss_weight,
        ps_min_gap_sec=args.ps_min_gap_sec,
        distance_gap_loss_weight=args.distance_gap_loss_weight,
        distance_gap_s_per_km=args.distance_gap_s_per_km,
        distance_gap_sigma_sec=args.distance_gap_sigma_sec,
        ps_gap_loss_weight=args.ps_gap_loss_weight if args.predict_ps_gap else 0.0,
        ps_gap_consist_weight=args.ps_gap_consist_weight,
        wrong_peak_loss_weight=args.wrong_peak_loss_weight,
        wrong_peak_radius_sec=args.wrong_peak_radius_sec,
        wrong_peak_margin=args.wrong_peak_margin,
        s_wrong_peak_scale=args.s_wrong_peak_scale,
        wrong_peak_listwise_weight=args.wrong_peak_listwise_weight,
        wrong_peak_listwise_topk=args.wrong_peak_listwise_topk,
        rho_sparsity_weight=args.rho_sparsity_weight,
        rho_sparsity_radius_sec=args.rho_sparsity_radius_sec,
        kernel_phys_prior_weight=args.kernel_phys_prior_weight,
        post_process_p_before_s=args.post_process_p_before_s,
        noise_cancel_weight=args.noise_cancel_weight if args.noise_cancel else 0.0,
        nc_consistency_weight=args.nc_consistency_weight,
        nc_phase_weight=args.nc_phase_weight,
        nc_preserve_weight=args.nc_preserve_weight,
        nc_energy_weight=args.nc_energy_weight,
        nc_noise_suppress_weight=args.nc_noise_suppress_weight,
    )
    test_metrics["best_epoch"] = ckpt["epoch"]
    test_metrics["val_score_at_best"] = ckpt["score"]
    test_metrics["n_params"] = ckpt.get("n_params", n_params)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(
        f"[STEAD-HNF-PHYS] done  det_f1={test_metrics['det_f1']:.4f}  "
        f"p_f1={test_metrics['p_f1']:.4f}  s_f1={test_metrics['s_f1']:.4f}  "
        f"p_mae={test_metrics['p_mae_sec']:.3f}s  s_mae={test_metrics['s_mae_sec']:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    train()
