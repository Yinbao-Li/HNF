#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train coarse-to-fine Huygens picker (6000 @ 100 Hz I/O, physics at T_c).

  python tools/train_stead_c2f_picking.py --device cuda --coarse-len 750 --no-amp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.picking_coarse_to_fine import build_c2f_picker
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
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.train_stead_picking import move_batch_to_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/stead_c2f_750_v3"))
    p.add_argument("--resume", type=Path, default=None, help="load state_dict from ckpt")
    p.add_argument("--expand-resume", action="store_true",
                   help="resume into a wider skip bottleneck; new input columns start at zero")
    p.add_argument("--continue-run", action="store_true",
                   help="continue this output dir from last.pt (epoch, optimizer, history, best)")
    p.add_argument("--fine-len", type=int, default=6000)
    p.add_argument("--coarse-len", type=int, default=750, choices=[750, 375])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--lr-schedule",
        default="cosine",
        choices=["cosine", "constant"],
        help="constant: hold lr (good for short finetunes)",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--skip-dim", type=int, default=64)
    p.add_argument("--mid-dim", type=int, default=64)
    p.add_argument("--num-shared-layers", type=int, default=3)
    p.add_argument("--num-branch-layers", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=0.3)
    p.add_argument("--vp", type=float, default=8.0)
    p.add_argument("--vs", type=float, default=4.5)
    p.add_argument("--local-window-sec", type=float, default=15.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--principle", default="huygens_fresnel",
                   choices=["huygens", "huygens_fresnel"])
    p.add_argument("--sparse-band", action="store_true", default=True)
    p.add_argument("--no-sparse-band", action="store_false", dest="sparse_band")
    p.add_argument("--max-event-train", type=int, default=40000)
    p.add_argument("--max-noise-train", type=int, default=20000)
    p.add_argument("--max-val", type=int, default=8000)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--pick-threshold", type=float, default=0.4)
    p.add_argument("--pick-tolerance-sec", type=float, default=0.5)
    p.add_argument("--pick-loss-weight", type=float, default=2.0)
    p.add_argument("--pick-pos-weight", type=float, default=8.0)
    p.add_argument("--det-loss-weight", type=float, default=1.0)
    p.add_argument("--noise-loss-weight", type=float, default=0.15)
    p.add_argument("--fusion", default="sum", choices=["sum", "propose_refine"])
    p.add_argument("--wave-input", default="fused", choices=["fused", "field"])
    p.add_argument("--refine-max", type=float, default=4.0)
    p.add_argument("--time-head-drop", type=float, default=0.0)
    p.add_argument("--skip-bottleneck", type=int, default=0)
    p.add_argument("--wave-sup-weight", type=float, default=0.0,
                   help="deep supervision of physics wave-head logits")
    p.add_argument("--refine-sup-weight", type=float, default=0.0,
                   help="deep supervision of refine-branch logits (p_ref/s_ref)")
    p.add_argument("--refine-label-sigma-sec", type=float, default=0.0,
                   help="narrower Gaussian target for the refine supervision (0 = same as main labels)")
    p.add_argument("--refine-pool-sec", type=float, default=0.0,
                   help="max-pool half-width of the physics proposal before gating the refine term")
    p.add_argument("--propagation", choices=["huygens", "free"], default="huygens",
                   help="free = control: unconstrained learned causal kernel instead of the Huygens kernel")
    p.add_argument("--noise-max-weight", type=float, default=0.0,
                   help="softplus(max_t logit) penalty on noise traces")
    p.add_argument("--order-loss-weight", type=float, default=0.0,
                   help="softplus(max S logit before P+gap) on event traces")
    p.add_argument("--order-gap-sec", type=float, default=0.5)
    p.add_argument("--physics-prior-weight", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # AMP off by default: Huygens kernels + fp16 caused NaN in v1
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"],
                   help="bf16 needs no GradScaler; Huygens blocks stay fp32 internally")
    p.add_argument("--samples-per-epoch", type=int, default=0,
                   help="draw this many random train traces per epoch (0 = full pass)")
    p.add_argument("--fresh-augment", action="store_true", default=False,
                   help="re-draw augmentation every epoch instead of fixing it per trace")
    p.add_argument("--aug-shift-prob", type=float, default=0.0,
                   help="probability of a large random time shift with real-noise fill")
    p.add_argument("--val-shift", action="store_true",
                   help="also validate on a deterministically shifted val set; select on the mean")
    p.add_argument("--aug-shift-p-min", type=float, default=1.0)
    p.add_argument("--aug-shift-p-max", type=float, default=50.0)
    p.add_argument("--aug-channel-drop-prob", type=float, default=0.0,
                   help="probability of zeroing the horizontals, the vertical, or one random channel")
    p.add_argument("--aug-gauss-prob", type=float, default=1.0, help="probability of additive Gaussian noise")
    p.add_argument("--aug-gauss-snr-min", type=float, default=5.0)
    p.add_argument("--aug-gauss-snr-max", type=float, default=20.0)
    p.add_argument("--aug-second-event-prob", type=float, default=0.0,
                   help="probability of superimposing a second (labelled) event after the primary S")
    p.add_argument("--aug-real-noise-prob", type=float, default=0.0,
                   help="probability of adding a real STEAD noise trace at a random P-window SNR")
    p.add_argument("--aug-real-noise-snr-min", type=float, default=0.0)
    p.add_argument("--aug-real-noise-snr-max", type=float, default=20.0)
    p.add_argument("--aug-response-prob", type=float, default=0.0,
                   help="probability of differentiating (acceleration) or random zero-phase band-limiting")
    p.add_argument("--aug-gap-prob", type=float, default=0.0,
                   help="probability of zeroing one 0.5–5 s segment away from the picks")
    p.add_argument("--aug-phase-absent-prob", type=float, default=0.0,
                   help="within a large shift: probability of pushing P before the window or S past it "
                        "(half each); absent phases get a zero target")
    p.add_argument("--aug-lowfreq-prob", type=float, default=0.0,
                   help="add microseism (0.1-1 Hz) + drift/offset below the STEAD band")
    p.add_argument("--aug-reverb-prob", type=float, default=0.0, help="OBS water-column multiples")
    p.add_argument("--aug-emergent-prob", type=float, default=0.0,
                   help="causal smoothing from P on (emergent onsets; hnf.robust_aug.emergent_onset)")
    p.add_argument("--gate-floor", type=float, default=0.0,
                   help="propose_refine: minimum gate ε on the CNN correction, g(w) = ε + (1−ε)σ(w)")
    p.add_argument("--late-sigma-ratio", type=float, default=1.0,
                   help="<1: P targets (main and refine) use sigma*ratio after the onset (penalise late picks)")
    p.add_argument("--manual-weight", type=float, default=1.0,
                   help="P-loss weight for analyst-picked (p_status=manual) traces")
    p.add_argument("--kalman", choices=["none", "kalman", "gate"], default="none",
                   help="measurement update after each shared Huygens layer (gate = same budget, no Riccati)")
    p.add_argument("--kalman-groups", type=int, default=8)
    p.add_argument("--new-lr-mult", type=float, default=10.0, help="lr multiplier for freshly added kalman params")
    p.add_argument("--wave-pos-weight", type=float, default=0.0,
                   help="positive weight of the physics-proposal loss (0 = pick-pos-weight); >1 favours recall")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", action="store_false", dest="augment")
    p.add_argument("--skip-final-test", action="store_true")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--nan-patience", type=int, default=1,
                   help="stop after this many NaN epochs (0=disable)")
    return p.parse_args()


def load_expanded(model, state_dict):
    """Copy the overlapping slice of every tensor; extra entries are zero so the
    widened model starts function-identical (skip_bn keeps fresh init for new rows)."""
    own = model.state_dict()
    for k, v in state_dict.items():
        tgt = own[k]
        if tgt.shape == v.shape:
            tgt.copy_(v)
            continue
        if not k.startswith("skip_bn."):
            tgt.zero_()
        tgt[tuple(slice(0, n) for n in v.shape)].copy_(v)
        print(f"[c2f] expanded {k}: {tuple(v.shape)} -> {tuple(tgt.shape)}", flush=True)
    model.load_state_dict(own, strict=True)


def schedule_lr(epoch, total, base_lr, warmup=5, kind: str = "cosine"):
    if kind == "constant":
        if warmup > 0 and epoch <= warmup:
            return base_lr * epoch / max(warmup, 1)
        return base_lr
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup + 1, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def cosine_lr(epoch, total, base_lr, warmup=5):
    return schedule_lr(epoch, total, base_lr, warmup=warmup, kind="cosine")


def asym_gaussian_target(idx, seq_len, sigma_bins, ratio, device):
    t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    j = idx.to(device=device, dtype=torch.float32).unsqueeze(-1)
    sig = torch.where(t > j, sigma_bins * ratio, torch.full_like(t, float(sigma_bins)))
    return torch.exp(-0.5 * ((t - j) / sig) ** 2) * (j >= 0).float()


def asym_p_target(batch, seq_len, sigma_bins, ratio, device):
    tgt = asym_gaussian_target(batch["p_idx"], seq_len, sigma_bins, ratio, device)
    if "p2_idx" in batch:
        tgt = torch.maximum(tgt, asym_gaussian_target(batch["p2_idx"], seq_len, sigma_bins, ratio, device))
    return tgt


def gaussian_target(idx, seq_len, sigma_bins, device):
    t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    j = idx.to(device=device, dtype=torch.float32).unsqueeze(-1)
    target = torch.exp(-0.5 * ((t - j) / sigma_bins) ** 2) * (j >= 0).float()
    return target


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


@torch.no_grad()
def evaluate(model, loader, device, *, seq_len, pick_th, tol_sec):
    model.eval()
    acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, tol_sec)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        if not (_finite(out["det"]) and _finite(out["p"]) and _finite(out["s"])):
            continue
        det_pred = det_pred_from_logits(out["det"])
        det_true = batch["det"] > 0.5
        update_detection_counts(acc, det_pred, det_true)
        p_probs = torch.sigmoid(out["p"])
        s_probs = torch.sigmoid(out["s"])
        p_probs, s_probs = apply_p_before_s_constraint(p_probs, s_probs, pick_th)
        for probs, idx_key, valid_key, counts in [
            (p_probs, "p_idx", "p_valid", acc.p),
            (s_probs, "s_idx", "s_valid", acc.s),
        ]:
            update_picking_counts(
                counts, probs, det_pred, det_true,
                batch[valid_key] > 0, batch[idx_key], pick_th, tol, seq_len,
            )
    metrics = finalize_metrics(acc)
    metrics["pick_focus"] = picking_score(metrics, mode="pick_focus")
    metrics["score"] = picking_score(metrics, mode="mean")
    return metrics


def train_one_epoch(model, loader, opt, scaler, device, args, seq_len, epoch, total_epochs):
    model.train()
    lr = schedule_lr(epoch, total_epochs, args.lr, warmup=args.warmup, kind=args.lr_schedule)
    for pg in opt.param_groups:
        pg["lr"] = lr * pg.get("lr_mult", 1.0)
    run_loss, n = 0.0, 0
    skipped = 0
    sigma_bins = max(1.0, args.label_sigma_sec * seq_len / 60.0)
    pos_w = torch.tensor(args.pick_pos_weight, device=device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    run_loss_t = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(batch["x"], batch["t"])
            det_loss = F.binary_cross_entropy_with_logits(
                out["det"].float(), batch["det"].float()
            )
            if "p_target" in batch:
                p_tgt = batch["p_target"].float()
                s_tgt = batch["s_target"].float()
            else:
                p_tgt = gaussian_target(batch["p_idx"], seq_len, sigma_bins, device)
                s_tgt = gaussian_target(batch["s_idx"], seq_len, sigma_bins, device)
            if args.late_sigma_ratio < 1.0:
                p_tgt = asym_p_target(batch, seq_len, sigma_bins, args.late_sigma_ratio, device)
            p_mask = (batch.get("p_mask", batch["p_valid"]) > 0).float().unsqueeze(-1)
            if args.manual_weight != 1.0 and "p_manual" in batch:
                p_mask = p_mask * (1.0 + (args.manual_weight - 1.0) * batch["p_manual"].float()).unsqueeze(-1)
            s_mask = (batch.get("s_mask", batch["s_valid"]) > 0).float().unsqueeze(-1)
            p_loss = F.binary_cross_entropy_with_logits(
                out["p"], p_tgt, reduction="none"
            )
            s_loss = F.binary_cross_entropy_with_logits(
                out["s"], s_tgt, reduction="none"
            )
            p_loss = (p_loss * (1.0 + (pos_w - 1.0) * p_tgt) * p_mask).mean()
            s_loss = (s_loss * (1.0 + (pos_w - 1.0) * s_tgt) * s_mask).mean()
            noise = (batch["det"] < 0.5).float().unsqueeze(-1)
            p_noise = (F.softplus(out["p"]) * noise).mean()
            s_noise = (F.softplus(out["s"]) * noise).mean()
            loss = (
                args.det_loss_weight * det_loss
                + args.pick_loss_weight * (p_loss + s_loss)
                + args.noise_loss_weight * (p_noise + s_noise)
            )
            if args.wave_sup_weight > 0:
                pw = F.binary_cross_entropy_with_logits(out["p_wave"].float(), p_tgt, reduction="none")
                sw = F.binary_cross_entropy_with_logits(out["s_wave"].float(), s_tgt, reduction="none")
                wpos = args.wave_pos_weight if args.wave_pos_weight > 0 else args.pick_pos_weight
                pw = (pw * (1.0 + (wpos - 1.0) * p_tgt) * p_mask).mean()
                sw = (sw * (1.0 + (wpos - 1.0) * s_tgt) * s_mask).mean()
                loss = loss + args.wave_sup_weight * (pw + sw)
            if args.refine_sup_weight > 0:
                if args.refine_label_sigma_sec > 0:
                    rs = max(1.0, args.refine_label_sigma_sec * seq_len / 60.0)
                    p_rt = gaussian_target(batch["p_idx"], seq_len, rs, device)
                    s_rt = gaussian_target(batch["s_idx"], seq_len, rs, device)
                    if "p2_idx" in batch:
                        p_rt = torch.maximum(p_rt, gaussian_target(batch["p2_idx"], seq_len, rs, device))
                        s_rt = torch.maximum(s_rt, gaussian_target(batch["s2_idx"], seq_len, rs, device))
                    if args.late_sigma_ratio < 1.0:
                        p_rt = asym_p_target(batch, seq_len, rs, args.late_sigma_ratio, device)
                else:
                    p_rt, s_rt = p_tgt, s_tgt
                pr = F.binary_cross_entropy_with_logits(out["p_ref"].float(), p_rt, reduction="none")
                sr = F.binary_cross_entropy_with_logits(out["s_ref"].float(), s_rt, reduction="none")
                pr = (pr * (1.0 + (pos_w - 1.0) * p_rt) * p_mask).mean()
                sr = (sr * (1.0 + (pos_w - 1.0) * s_rt) * s_mask).mean()
                loss = loss + args.refine_sup_weight * (pr + sr)
            if args.noise_max_weight > 0:
                nz = noise.squeeze(-1)
                pm = F.softplus(out["p"].float().amax(-1))
                sm = F.softplus(out["s"].float().amax(-1))
                loss = loss + args.noise_max_weight * ((pm + sm) * nz).sum() / nz.sum().clamp_min(1.0)
            if args.order_loss_weight > 0:
                gap = int(round(args.order_gap_sec * seq_len / 60.0))
                t_ax = torch.arange(seq_len, device=device).unsqueeze(0)
                ev = ((batch["p_valid"] > 0) & (batch["s_valid"] > 0)).float()
                before = t_ax < (batch["p_idx"].long().unsqueeze(-1) + gap)
                s_before = out["s"].float().masked_fill(~before, -1e4).amax(-1)
                loss = loss + args.order_loss_weight * (F.softplus(s_before) * ev).sum() / ev.sum().clamp_min(1.0)
        if args.physics_prior_weight > 0:
            prior = sum(m.physics_prior_loss().sum() for m in model.modules()
                        if hasattr(m, "physics_prior_loss") and m is not model)
            loss = loss + args.physics_prior_weight * prior

        if not _finite(loss):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        bs = batch["x"].size(0)
        run_loss_t += loss.detach().double() * bs
        n += bs

    run_loss = float(run_loss_t)
    mean_loss = run_loss / max(n, 1)
    nan_epoch = n == 0 or (not math.isfinite(mean_loss))
    return {
        "loss": mean_loss if math.isfinite(mean_loss) else float("nan"),
        "lr": lr,
        "skipped": skipped,
        "nan_epoch": nan_epoch,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    seq_len = args.fine_len
    print(
        f"[c2f] fine={args.fine_len} coarse={args.coarse_len} "
        f"device={device} amp={args.amp} lr={args.lr} schedule={args.lr_schedule} "
        f"pos_w={args.pick_pos_weight} th={args.pick_threshold} "
        f"sparse_band={args.sparse_band} principle={args.principle}",
        flush=True,
    )

    train_ds = STEADPickingDataset(
        "train",
        seq_len=seq_len,
        max_event_traces=args.max_event_train,
        max_noise_traces=args.max_noise_train,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
        augment=args.augment,
        fresh_augment=args.fresh_augment,
        aug_shift_prob=args.aug_shift_prob,
        aug_shift_p_range=(args.aug_shift_p_min, args.aug_shift_p_max),
        aug_channel_drop_prob=args.aug_channel_drop_prob,
        aug_noise_snr_db=(args.aug_gauss_snr_min, args.aug_gauss_snr_max),
        aug_gauss_prob=args.aug_gauss_prob,
        aug_second_event_prob=args.aug_second_event_prob,
        aug_real_noise_prob=args.aug_real_noise_prob,
        aug_real_noise_snr_db=(args.aug_real_noise_snr_min, args.aug_real_noise_snr_max),
        aug_response_prob=args.aug_response_prob,
        aug_gap_prob=args.aug_gap_prob,
        aug_phase_absent_prob=args.aug_phase_absent_prob,
        aug_lowfreq_prob=args.aug_lowfreq_prob,
        aug_reverb_prob=args.aug_reverb_prob,
        aug_emergent_prob=args.aug_emergent_prob,
        load_p_status=args.manual_weight != 1.0,
    )
    val_ds = STEADPickingDataset(
        "val",
        seq_len=seq_len,
        max_event_traces=args.max_val // 2,
        max_noise_traces=args.max_val // 2,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
    )
    test_ds = STEADPickingDataset(
        "test",
        seq_len=seq_len,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
    )
    print(
        f"[c2f] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"(native labels @ 100 Hz → seq_len={seq_len})",
        flush=True,
    )

    if args.samples_per_epoch > 0:
        sampler = RandomSampler(
            train_ds, replacement=False,
            num_samples=min(args.samples_per_epoch, len(train_ds)),
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_shift_loader = None
    if args.val_shift:
        val_shift_ds = STEADPickingDataset(
            "val", seq_len=seq_len, max_event_traces=args.max_val // 2, max_noise_traces=args.max_val // 2,
            label_sigma_sec=args.label_sigma_sec, seed=args.seed, eval_shift=True,
            aug_shift_p_range=(args.aug_shift_p_min, args.aug_shift_p_max),
        )
        val_shift_loader = DataLoader(
            val_shift_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = build_c2f_picker(
        fine_len=args.fine_len,
        coarse_len=args.coarse_len,
        embed_dim=args.embed_dim,
        skip_dim=args.skip_dim,
        mid_dim=args.mid_dim,
        num_shared_layers=args.num_shared_layers,
        num_branch_layers=args.num_branch_layers,
        gamma=args.gamma,
        omega=args.omega,
        vp=args.vp,
        vs=args.vs,
        local_window_sec=args.local_window_sec,
        dropout=args.dropout,
        sparse_band=args.sparse_band,
        principle=args.principle,
        fusion=args.fusion,
        wave_input=args.wave_input,
        refine_max=args.refine_max,
        time_head_drop=args.time_head_drop,
        skip_bottleneck=args.skip_bottleneck,
        refine_pool_sec=args.refine_pool_sec,
        propagation=args.propagation,
        gate_floor=args.gate_floor,
        kalman=args.kalman,
        kalman_groups=args.kalman_groups,
    ).to(device)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if args.expand_resume:
            load_expanded(model, ckpt["state_dict"])
        elif args.kalman != "none" and not any(k.startswith("kalman.") for k in ckpt["state_dict"]):
            res = model.load_state_dict(ckpt["state_dict"], strict=False)
            if res.unexpected_keys or any(not k.startswith("kalman.") for k in res.missing_keys):
                raise RuntimeError(f"resume mismatch: {res}")
            print(f"[c2f] fresh kalman params: {len(res.missing_keys)} tensors", flush=True)
        else:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[c2f] resumed weights from {args.resume}", flush=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[c2f] params={n_params:,}", flush=True)

    new_p = [p for n, p in model.named_parameters() if n.startswith("kalman.")]
    old_p = [p for n, p in model.named_parameters() if not n.startswith("kalman.")]
    groups = [{"params": old_p, "lr_mult": 1.0}]
    if new_p:
        groups.append({"params": new_p, "lr_mult": args.new_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    need_scaler = use_amp and args.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if need_scaler else None

    best = {"pick_focus": -1.0}
    history = []
    nan_streak = 0
    start_epoch = 1
    if args.continue_run:
        last = torch.load(args.output_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(last["state_dict"], strict=True)
        if "optimizer" in last:
            opt.load_state_dict(last["optimizer"])
        start_epoch = int(last["epoch"]) + 1
        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False).get("best", best)
        hist_path = args.output_dir / "history.json"
        if hist_path.exists():
            history = [r for r in json.loads(hist_path.read_text()) if r["epoch"] < start_epoch]
        print(f"[c2f] continue from ep{start_epoch - 1} (optimizer={'optimizer' in last}) "
              f"best ep{best.get('epoch', '?')} pick_focus={best['pick_focus']:.4f}", flush=True)
    else:
        (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # Seed best from resumed weights so finetune cannot regress the checkpoint silently
    if args.resume is not None:
        va0 = evaluate(
            model, val_loader, device,
            seq_len=seq_len, pick_th=args.pick_threshold, tol_sec=args.pick_tolerance_sec,
        )
        best = {"epoch": 0, **va0}
        if val_shift_loader is not None:
            vs0 = evaluate(
                model, val_shift_loader, device,
                seq_len=seq_len, pick_th=args.pick_threshold, tol_sec=args.pick_tolerance_sec,
            )
            best["select_score"] = 0.5 * (va0["pick_focus"] + vs0["pick_focus"])
            print(f"[c2f] resume baseline shifted P={vs0['p_f1']:.4f} S={vs0['s_f1']:.4f} "
                  f"select={best['select_score']:.4f}", flush=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "best": best,
                "epoch": 0,
            },
            args.output_dir / "best.pt",
        )
        print(
            f"[c2f] resume baseline val det={va0['det_f1']:.4f} P={va0['p_f1']:.4f} "
            f"S={va0['s_f1']:.4f} pick_focus={va0['pick_focus']:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, opt, scaler, device, args, seq_len, epoch, args.epochs
        )
        row = {"epoch": epoch, **tr, "sec": time.time() - t0}

        if tr["nan_epoch"]:
            nan_streak += 1
            print(
                f"ep{epoch:02d} lr={tr['lr']:.2e} loss=nan skipped={tr['skipped']} "
                f"nan_streak={nan_streak} ({row['sec']:.0f}s)",
                flush=True,
            )
            history.append(row)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
            # restore last good weights if available
            best_path = args.output_dir / "best.pt"
            if best_path.exists() and best.get("pick_focus", -1) > 0:
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["state_dict"])
                print(f"[c2f] restored best.pt after NaN (ep{best.get('epoch', '?')})", flush=True)
            if args.nan_patience > 0 and nan_streak >= args.nan_patience:
                print(f"[c2f] NaN patience reached ({nan_streak}); stop training.", flush=True)
                break
            continue

        nan_streak = 0
        if epoch % args.val_every == 0:
            va = evaluate(
                model, val_loader, device,
                seq_len=seq_len, pick_th=args.pick_threshold, tol_sec=args.pick_tolerance_sec,
            )
            row["val"] = va
            score = va["pick_focus"]
            shift_msg = ""
            if val_shift_loader is not None:
                vs = evaluate(
                    model, val_shift_loader, device,
                    seq_len=seq_len, pick_th=args.pick_threshold, tol_sec=args.pick_tolerance_sec,
                )
                row["val_shift"] = vs
                score = 0.5 * (va["pick_focus"] + vs["pick_focus"])
                shift_msg = f" | shifted P={vs['p_f1']:.4f} S={vs['s_f1']:.4f} select={score:.4f}"
            print(
                f"ep{epoch:02d} lr={tr['lr']:.2e} loss={tr['loss']:.4f} skipped={tr['skipped']} "
                f"val det={va['det_f1']:.4f} P={va['p_f1']:.4f} S={va['s_f1']:.4f} "
                f"pick_focus={va['pick_focus']:.4f}{shift_msg} ({row['sec']:.0f}s)",
                flush=True,
            )
            # IMPORTANT: compare pick_focus only; do not let va["score"] overwrite the key
            if score > best.get("select_score", best["pick_focus"]):
                best = {"epoch": epoch, **va, "select_score": score}
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "best": best,
                        "epoch": epoch,
                    },
                    args.output_dir / "best.pt",
                )
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val": va,
                 "optimizer": opt.state_dict(), "args": vars(args)},
                args.output_dir / "last.pt",
            )
        else:
            print(
                f"ep{epoch:02d} lr={tr['lr']:.2e} loss={tr['loss']:.4f} "
                f"skipped={tr['skipped']} ({row['sec']:.0f}s)",
                flush=True,
            )
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_path = args.output_dir / "best.pt"
    if not best_path.exists():
        print("[c2f] no best.pt saved; aborting test.", flush=True)
        return
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if "best" in ckpt and best.get("pick_focus", -1) < 0:
        best = ckpt["best"]
    summary = {
        "best_val": best,
        "n_params": n_params,
        "fine_len": args.fine_len,
        "coarse_len": args.coarse_len,
        "stopped_early_nan": bool(args.nan_patience > 0 and nan_streak >= args.nan_patience),
    }
    if not args.skip_final_test:
        print("[c2f] full test eval…", flush=True)
        te = evaluate(
            model, test_loader, device,
            seq_len=seq_len, pick_th=args.pick_threshold, tol_sec=args.pick_tolerance_sec,
        )
        summary["test"] = te
        print(
            f"[test] det={te['det_f1']:.4f} P={te['p_f1']:.4f} S={te['s_f1']:.4f} "
            f"pick_focus={te['pick_focus']:.4f} "
            f"P_MAE={te['p_mae_sec']:.4f}s S_MAE={te['s_mae_sec']:.4f}s",
            flush=True,
        )
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
