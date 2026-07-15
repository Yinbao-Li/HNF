#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose whether wrong-peak FNs still have a local max near GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tools.analyze_stead_picking import _noise_ratio_from_outputs, _subsample_indices, load_model
from hnf.picking_metrics import apply_p_before_s_constraint, idx_to_sec, tolerance_bins
from hnf.stead_picking_dataset import STEADPickingDataset
from run_adaptive_pick_eval import _local_max_indices
from tools.train_stead_picking import move_batch_to_device


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output", default="outputs/run20/20_wrongpeak_sharp/wrongpeak_diag.json")
    p.add_argument("--max-events", type=int, default=800)
    p.add_argument("--max-noise", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--subset-seed", type=int, default=11)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.checkpoint), device)
    ds = STEADPickingDataset("test", seq_len=args.seq_len, load_geometry=False)
    indices = _subsample_indices(ds, args.max_events, args.max_noise, args.subset_seed)
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False)
    tol = tolerance_bins(args.seq_len, 0.5)

    stats = {
        "p": {"wrong": 0, "gt_local_max": 0, "gt_in_topk3": 0, "gt_in_topk5": 0, "gt_prob": [], "pred_prob": [], "noise": []},
        "s": {"wrong": 0, "gt_local_max": 0, "gt_in_topk3": 0, "gt_in_topk5": 0, "gt_prob": [], "pred_prob": [], "noise": []},
    }

    for bi, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["x"], batch["t"])
        det = torch.sigmoid(outputs["det"])
        if det.dim() > 1:
            det = det.amax(dim=-1)
        p_probs, s_probs = apply_p_before_s_constraint(
            torch.sigmoid(outputs["p"]), torch.sigmoid(outputs["s"]), args.pick_threshold
        )
        noise = _noise_ratio_from_outputs(outputs)
        for i in range(batch["x"].size(0)):
            if batch["det"][i] <= 0.5 or det[i] < args.det_threshold:
                continue
            nr = None if noise is None else float(noise[i].item())
            for head, probs, idx_k, valid_k in [
                ("p", p_probs[i], "p_idx", "p_valid"),
                ("s", s_probs[i], "s_idx", "s_valid"),
            ]:
                if batch[valid_k][i] <= 0:
                    continue
                gt = int(batch[idx_k][i].item())
                peak = float(probs.max().item())
                pred = int(probs.argmax().item())
                if peak < args.pick_threshold or abs(pred - gt) <= tol:
                    continue
                st = stats[head]
                st["wrong"] += 1
                locals_ = _local_max_indices(probs)
                if any(abs(j - gt) <= tol for j in locals_):
                    st["gt_local_max"] += 1
                vals, idxs = torch.topk(probs, k=min(5, probs.numel()))
                top = idxs.tolist()
                if any(abs(j - gt) <= tol for j in top[:3]):
                    st["gt_in_topk3"] += 1
                if any(abs(j - gt) <= tol for j in top):
                    st["gt_in_topk5"] += 1
                st["gt_prob"].append(float(probs[gt].item()))
                st["pred_prob"].append(peak)
                if nr is not None:
                    st["noise"].append(nr)
        if (bi + 1) % 25 == 0:
            print(f"[diag] batch {bi+1}/{len(loader)}", flush=True)

    out = {"n_events": args.max_events, "heads": {}}
    for head, st in stats.items():
        n = max(st["wrong"], 1)
        out["heads"][head] = {
            "wrong_peak_n": st["wrong"],
            "frac_gt_is_local_max": st["gt_local_max"] / n,
            "frac_gt_in_topk3": st["gt_in_topk3"] / n,
            "frac_gt_in_topk5": st["gt_in_topk5"] / n,
            "mean_gt_prob": float(np.mean(st["gt_prob"])) if st["gt_prob"] else None,
            "mean_pred_prob": float(np.mean(st["pred_prob"])) if st["pred_prob"] else None,
            "mean_noise_ratio": float(np.mean(st["noise"])) if st["noise"] else None,
            "median_gt_prob": float(np.median(st["gt_prob"])) if st["gt_prob"] else None,
        }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
