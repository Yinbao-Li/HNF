#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Huygens physics picking model on STEAD."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

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
    p.add_argument("--wrong-peak-loss-weight", type=float, default=0.0, help="Rank GT window above distant wrong peaks")
    p.add_argument("--wrong-peak-radius-sec", type=float, default=0.5)
    p.add_argument("--wrong-peak-margin", type=float, default=0.2)
    p.add_argument("--s-wrong-peak-scale", type=float, default=1.0)
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
    wrong_peak_loss_weight: float = 0.0,
    wrong_peak_radius_sec: float = 0.5,
    wrong_peak_margin: float = 0.2,
    s_wrong_peak_scale: float = 1.0,
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
        + wrong_peak_loss_weight * wrong_peak_loss
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
    wrong_peak_loss_weight: float = 0.0,
    wrong_peak_radius_sec: float = 0.5,
    wrong_peak_margin: float = 0.2,
    s_wrong_peak_scale: float = 1.0,
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
            wrong_peak_loss_weight=wrong_peak_loss_weight,
            wrong_peak_radius_sec=wrong_peak_radius_sec,
            wrong_peak_margin=wrong_peak_margin,
            s_wrong_peak_scale=s_wrong_peak_scale,
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
)


def apply_freeze_schedule(model: nn.Module, epoch: int, args: argparse.Namespace) -> None:
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
        f"principle={args.principle} obliquity_scale={args.obliquity_scale} "
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
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(batch["x"], batch["t"])
                loss = compute_loss(
                    outputs,
                    batch,
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
                    wrong_peak_loss_weight=args.wrong_peak_loss_weight,
                    wrong_peak_radius_sec=args.wrong_peak_radius_sec,
                    wrong_peak_margin=args.wrong_peak_margin,
                    s_wrong_peak_scale=args.s_wrong_peak_scale,
                    seq_len=args.seq_len,
                    noise_cancel_weight=args.noise_cancel_weight if args.noise_cancel else 0.0,
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
            wrong_peak_loss_weight=args.wrong_peak_loss_weight,
            wrong_peak_radius_sec=args.wrong_peak_radius_sec,
            wrong_peak_margin=args.wrong_peak_margin,
            s_wrong_peak_scale=args.s_wrong_peak_scale,
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
    model.load_state_dict(ckpt["state_dict"])
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
        wrong_peak_loss_weight=args.wrong_peak_loss_weight,
        wrong_peak_radius_sec=args.wrong_peak_radius_sec,
        wrong_peak_margin=args.wrong_peak_margin,
        s_wrong_peak_scale=args.s_wrong_peak_scale,
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
