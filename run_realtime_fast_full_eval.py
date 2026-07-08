#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast full-test picking eval: default vs bypass (no threshold sweep)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.picking_metrics import (
    apply_p_before_s_constraint,
    det_pred_from_logits,
    finalize_metrics,
    tolerance_bins,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset
from train_stead_picking import EvalAccumulator


def eval_full(model, loader, device, seq_len, pick_th=0.3):
    model.eval()
    acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, 0.5)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["x"], batch["t"])
            det_pred = det_pred_from_logits(outputs["det"])
            det_true = batch["det"] > 0.5
            update_detection_counts(acc, det_pred, det_true)
            p_probs, s_probs = apply_p_before_s_constraint(
                torch.sigmoid(outputs["p"]), torch.sigmoid(outputs["s"]), pick_th
            )
            for head, probs, idx_k, valid_k in [
                ("p", p_probs, "p_idx", "p_valid"),
                ("s", s_probs, "s_idx", "s_valid"),
            ]:
                update_picking_counts(
                    getattr(acc, head), probs, det_pred, det_true,
                    batch[valid_k] > 0, batch[idx_k], pick_th, tol, seq_len,
                )
    return finalize_metrics(acc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output", default="outputs/realtime_full_compare/fast_report.json")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    ckpt = Path(args.checkpoint)
    ds = STEADPickingDataset("test", seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    report = {"checkpoint": str(ckpt), "n_test": len(ds), "modes": {}}
    for name, bypass in [("default_denoise_split", False), ("bypass_no_denoise", True)]:
        model, ckpt_args = load_model(ckpt, device, bypass_noise_cancel=bypass)
        pick_th = float(ckpt_args.get("pick_threshold", 0.3))
        m = eval_full(model, loader, device, args.seq_len, pick_th)
        report["modes"][name] = {"bypass_noise_cancel": bypass, **m}
        print(name, m, flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[fast_full] -> {out}")


if __name__ == "__main__":
    main()
