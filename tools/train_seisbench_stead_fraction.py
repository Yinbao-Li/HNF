#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train PhaseNet / EQTransformer from scratch on a STEAD data fraction.

Uses the same STEADPickingDataset splits/seed as HNF for fair sample-efficiency.
Native length 6000 @ 100 Hz. Random init (not STEAD-pretrained).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.picking_metrics import (  # noqa: E402
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.train_stead_picking import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["phasenet", "eqtransformer"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-event-train", type=int, default=10000)
    p.add_argument("--max-noise-train", type=int, default=5000)
    p.add_argument("--max-val", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="")
    p.add_argument("--eval-max-events", type=int, default=8000)
    p.add_argument("--eval-max-noise", type=int, default=2000)
    return p.parse_args()


def build_model(name: str):
    import seisbench.models as sbm

    if name == "phasenet":
        return sbm.PhaseNet(in_channels=3, sampling_rate=100)
    return sbm.EQTransformer(in_channels=3, sampling_rate=100)


def loss_batch(model_name: str, model, batch, device) -> torch.Tensor:
    # Dataset returns x as (B, T, 3) → model wants (B, 3, T)
    wave = batch["x"].to(device).transpose(1, 2).contiguous()
    p_lab = batch["p_target"].to(device)
    s_lab = batch["s_target"].to(device)
    is_event = batch["det"].to(device).float().view(-1, 1)

    if model_name == "phasenet":
        pred = model(wave)  # (B, 3, T) NPS softmax
        n_lab = (1.0 - torch.maximum(p_lab, s_lab)).clamp(0, 1)
        n_lab = torch.where(is_event > 0.5, n_lab, torch.ones_like(n_lab))
        p_t = torch.where(is_event > 0.5, p_lab, torch.zeros_like(p_lab))
        s_t = torch.where(is_event > 0.5, s_lab, torch.zeros_like(s_lab))
        tgt = torch.stack([n_lab, p_t, s_t], dim=1)
        tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logp = torch.log(pred.clamp_min(1e-6))
        return -(tgt * logp).sum(dim=1).mean()

    det, p_pred, s_pred = model(wave)
    if det.dim() == 3:
        det = det.squeeze(1)
    if p_pred.dim() == 3:
        p_pred = p_pred.squeeze(1)
    if s_pred.dim() == 3:
        s_pred = s_pred.squeeze(1)
    det_t = torch.maximum(p_lab, s_lab) * is_event
    p_t = p_lab * is_event
    s_t = s_lab * is_event
    return (
        F.binary_cross_entropy(det.clamp(1e-6, 1 - 1e-6), det_t)
        + F.binary_cross_entropy(p_pred.clamp(1e-6, 1 - 1e-6), p_t)
        + F.binary_cross_entropy(s_pred.clamp(1e-6, 1 - 1e-6), s_t)
    )


@torch.no_grad()
def evaluate_model(model_name, model, loader, device, pick_th, det_th, tol_sec, seq_len=6000):
    model.eval()
    acc = EvalAccumulator()
    tol_bins = max(1, int(round(tol_sec * 100.0)))
    for batch in loader:
        wave = batch["x"].to(device).transpose(1, 2).contiguous()
        is_event = batch["det"].to(device).bool()
        p_true = batch["p_idx"].to(device)
        s_true = batch["s_idx"].to(device)
        p_valid = batch["p_valid"].to(device).bool() & is_event
        s_valid = batch["s_valid"].to(device).bool() & is_event
        if model_name == "phasenet":
            out = model(wave)
            p_prob, s_prob = out[:, 1], out[:, 2]
            det_pred = torch.maximum(p_prob.amax(-1), s_prob.amax(-1)) >= pick_th
        else:
            det, p_prob, s_prob = model(wave)
            if det.dim() == 3:
                det = det.squeeze(1)
            if p_prob.dim() == 3:
                p_prob = p_prob.squeeze(1)
            if s_prob.dim() == 3:
                s_prob = s_prob.squeeze(1)
            det_pred = det.amax(dim=-1) >= det_th
        p_prob, s_prob = apply_p_before_s_constraint(p_prob, s_prob, pick_th)
        update_detection_counts(acc, det_pred, is_event)
        update_picking_counts(
            acc.p, p_prob, det_pred, is_event, p_valid, p_true, pick_th, tol_bins, seq_len
        )
        update_picking_counts(
            acc.s, s_prob, det_pred, is_event, s_valid, s_true, pick_th, tol_bins, seq_len
        )
    return finalize_metrics(acc)


def subsample_test(ds, max_events, max_noise, seed):
    from torch.utils.data import Subset

    ev = [i for i, r in enumerate(ds.refs) if r.is_event == 1]
    nz = [i for i, r in enumerate(ds.refs) if r.is_event == 0]
    rng = np.random.default_rng(seed)
    if len(ev) > max_events:
        ev = sorted(rng.choice(ev, size=max_events, replace=False).tolist())
    if len(nz) > max_noise:
        nz = sorted(rng.choice(nz, size=max_noise, replace=False).tolist())
    return Subset(ds, ev + nz)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = STEADPickingDataset(
        "train",
        seq_len=6000,
        max_event_traces=args.max_event_train,
        max_noise_traces=args.max_noise_train,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
        augment=True,
        load_geometry=False,
    )
    val_ds = STEADPickingDataset(
        "val",
        seq_len=6000,
        max_event_traces=args.max_val,
        max_noise_traces=max(1, args.max_val // 4),
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
        load_geometry=False,
    )
    test_full = STEADPickingDataset(
        "test", seq_len=6000, label_sigma_sec=args.label_sigma_sec, seed=args.seed, load_geometry=False
    )
    test_ds = subsample_test(test_full, args.eval_max_events, args.eval_max_noise, seed=11)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    history = []
    best_val = -1.0
    best_path = out / "best.pt"
    t0 = time.time()
    print(
        f"[{args.model}] device={device} params={n_params} "
        f"train={len(train_ds)} val={len(val_ds)} test_eval={len(test_ds)}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_batch(args.model, model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        val_m = evaluate_model(
            args.model, model, val_loader, device, args.pick_threshold, args.det_threshold, args.tol_sec
        )
        score = 0.5 * (val_m.get("p_f1", 0.0) + val_m.get("s_f1", 0.0))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_p_f1": val_m.get("p_f1"),
            "val_s_f1": val_m.get("s_f1"),
            "val_det_f1": val_m.get("det_f1"),
            "val_score": score,
        }
        history.append(row)
        print(
            f"  ep{epoch:02d} loss={row['train_loss']:.4f} "
            f"P/S F1={val_m.get('p_f1', 0):.3f}/{val_m.get('s_f1', 0):.3f}",
            flush=True,
        )
        ckpt = {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "n_params": n_params,
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_m,
        }
        torch.save(ckpt, out / "last.pt")
        if score >= best_val:
            best_val = score
            torch.save(ckpt, best_path)

    # test with best
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["state_dict"])
    test_m = evaluate_model(
        args.model, model, test_loader, device, args.pick_threshold, args.det_threshold, args.tol_sec
    )
    test_m["n_params"] = n_params
    test_m["n_train"] = len(train_ds)
    test_m["n_event_train"] = int(args.max_event_train)
    test_m["n_noise_train"] = int(args.max_noise_train)
    test_m["model"] = args.model
    test_m["seconds"] = time.time() - t0
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "test_metrics.json").write_text(json.dumps(test_m, indent=2))
    print("TEST", json.dumps({k: test_m[k] for k in ["det_f1", "p_f1", "s_f1", "p_mae_sec", "s_mae_sec", "n_params"]}, indent=2))


if __name__ == "__main__":
    main()
