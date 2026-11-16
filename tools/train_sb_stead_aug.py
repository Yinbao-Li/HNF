#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Augmentation control: PhaseNet / EQTransformer (SeisBench architectures, random init) trained on STEAD
with exactly the c2f data pipeline, augmentation recipe and input normalisation (ENZ, per-channel demean/std).

Evaluate the resulting last.pt with --sbc name=path in the cross-dataset tools (same preprocessing as c2f).

  python tools/train_sb_stead_aug.py --model eqt --output-dir outputs/sb_aug_eqt --epochs 20 ...
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

from hnf.picking_metrics import (  # noqa: E402
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402

T = 6000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["phasenet", "eqt"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--samples-per-epoch", type=int, default=200000)
    p.add_argument("--max-val", type=int, default=8000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--aug-shift-prob", type=float, default=0.5)
    p.add_argument("--aug-second-event-prob", type=float, default=0.3)
    p.add_argument("--aug-real-noise-prob", type=float, default=0.5)
    p.add_argument("--aug-gauss-prob", type=float, default=0.3)
    p.add_argument("--aug-response-prob", type=float, default=0.3)
    p.add_argument("--aug-gap-prob", type=float, default=0.2)
    p.add_argument("--aug-channel-drop-prob", type=float, default=0.1)
    p.add_argument("--aug-phase-absent-prob", type=float, default=0.3)
    return p.parse_args()


def build(name):
    import seisbench.models as sbm

    if name == "phasenet":
        return sbm.PhaseNet(in_channels=3, classes=3, phases="NPS", sampling_rate=100)
    return sbm.EQTransformer(in_channels=3, in_samples=T, sampling_rate=100)


def forward_probs(name, model, x):
    """x (B,T,3) normalised → det (B,) or None, p (B,T), s (B,T) probabilities."""
    out = model(x.transpose(1, 2).contiguous())
    if name == "phasenet":
        return None, out[:, 1], out[:, 2]
    det, p, s = (t.squeeze(1) if t.dim() == 3 else t for t in out)
    return det, p, s


def det_span(batch, device):
    """EQT detection target: 1 from P to S + 1.4 (S - P), per event in the window (SeisBench convention)."""
    b = batch["p_idx"].size(0)
    tgt = torch.zeros(b, T, device=device)
    is_ev = batch["det"].view(-1) > 0.5
    for i in range(b):
        if not is_ev[i]:
            continue
        for pk, sk in (("p_idx", "s_idx"), ("p2_idx", "s2_idx")):
            p, s = int(batch[pk][i]), int(batch[sk][i])
            if pk == "p_idx":
                p = p if batch["p_valid"][i] > 0 else -1
                s = s if batch["s_valid"][i] > 0 else -1
            if p < 0 and s < 0:
                continue
            a = max(p, 0)
            e = T if s < 0 else (min(T, int(s + 1.4 * (s - p))) if p >= 0 else min(T, s + 100))
            tgt[i, a:e] = 1.0
    return tgt


def loss_fn(name, model, batch, device):
    x = batch["x"].to(device)
    p_t, s_t = batch["p_target"].to(device).float(), batch["s_target"].to(device).float()
    is_ev = (batch["det"].to(device).view(-1) > 0.5).float()
    wp = torch.where(is_ev > 0, batch["p_mask"].to(device), torch.ones_like(is_ev)).unsqueeze(-1)
    ws = torch.where(is_ev > 0, batch["s_mask"].to(device), torch.ones_like(is_ev)).unsqueeze(-1)
    p_t, s_t = p_t * is_ev.unsqueeze(-1), s_t * is_ev.unsqueeze(-1)
    if name == "phasenet":
        probs = model(x.transpose(1, 2).contiguous())  # (B, 3, T) softmax over N, P, S
        tgt = torch.stack([(1.0 - p_t - s_t).clamp_min(0.0), p_t, s_t], 1)
        tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-6)
        ce = -(tgt * torch.log(probs.clamp_min(1e-6))).sum(1)
        return (ce * torch.minimum(wp, ws)).mean()
    det, p, s = forward_probs(name, model, x)
    bce = lambda a, b: F.binary_cross_entropy(a.float().clamp(1e-6, 1 - 1e-6), b, reduction="none")  # noqa: E731
    return (0.05 * bce(det, det_span(batch, device)).mean()
            + 0.40 * (bce(p, p_t) * wp).mean() + 0.55 * (bce(s, s_t) * ws).mean())


@torch.no_grad()
def evaluate(name, model, loader, device, th, det_th):
    model.eval()
    acc = EvalAccumulator()
    for batch in loader:
        x = batch["x"].to(device)
        det, p, s = forward_probs(name, model, x)
        det_pred = (torch.maximum(p.amax(-1), s.amax(-1)) >= th) if det is None else det.amax(-1) >= det_th
        p, s = apply_p_before_s_constraint(p.float(), s.float(), th)
        is_ev = batch["det"].to(device).view(-1) > 0.5
        pv = (batch["p_valid"].to(device) > 0) & is_ev
        sv = (batch["s_valid"].to(device) > 0) & is_ev
        update_detection_counts(acc, det_pred, is_ev)
        update_picking_counts(acc.p, p, det_pred, is_ev, pv, batch["p_idx"].to(device), th, 50, T)
        update_picking_counts(acc.s, s, det_pred, is_ev, sv, batch["s_idx"].to(device), th, 50, T)
    model.train()
    return finalize_metrics(acc)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    aug = not args.no_augment
    train_ds = STEADPickingDataset(
        "train", seq_len=T, max_event_traces=10_000_000, max_noise_traces=10_000_000,
        label_sigma_sec=args.label_sigma_sec, seed=args.seed, augment=aug, fresh_augment=True,
        load_geometry=False,
        aug_shift_prob=args.aug_shift_prob if aug else 0.0,
        aug_channel_drop_prob=args.aug_channel_drop_prob if aug else 0.0,
        aug_noise_snr_db=(5.0, 20.0),
        aug_gauss_prob=args.aug_gauss_prob if aug else 0.0,
        aug_second_event_prob=args.aug_second_event_prob if aug else 0.0,
        aug_real_noise_prob=args.aug_real_noise_prob if aug else 0.0,
        aug_real_noise_snr_db=(0.0, 20.0),
        aug_response_prob=args.aug_response_prob if aug else 0.0,
        aug_gap_prob=args.aug_gap_prob if aug else 0.0,
        aug_phase_absent_prob=args.aug_phase_absent_prob if aug else 0.0,
    )
    val_ds = STEADPickingDataset("val", seq_len=T, max_event_traces=args.max_val, max_noise_traces=args.max_val,
                                 label_sigma_sec=args.label_sigma_sec, seed=args.seed, load_geometry=False)
    sampler = RandomSampler(train_ds, replacement=False, num_samples=min(args.samples_per_epoch, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                              drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=args.num_workers)
    model = build(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_loader)

    def lr_at(step):
        e = step / steps
        if e < args.warmup:
            return (e + 1e-3) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (e - args.warmup) / max(args.epochs - args.warmup, 1e-6)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"[sb-aug] {args.model} params={n_params} train={len(train_ds)} val={len(val_ds)} aug={aug}", flush=True)
    history = []
    for ep in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(args.model, model, batch, device)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        m = evaluate(args.model, model, val_loader, device, args.pick_threshold, args.det_threshold)
        row = {"epoch": ep, "loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
               "det_f1": m["det_f1"], "p_f1": m["p_f1"], "s_f1": m["s_f1"]}
        history.append(row)
        print(f"ep{ep:02d} lr={row['lr']:.2e} loss={row['loss']:.4f} val det={m['det_f1']:.4f} "
              f"P={m['p_f1']:.4f} S={m['s_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save({"model_name": args.model, "state_dict": model.state_dict(), "epoch": ep, "val": m},
                   args.output_dir / "last.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
