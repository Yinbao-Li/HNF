#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a STEAD HNF picking checkpoint on the test split."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hnf.picking_model import build_picking_model, load_picking_model_state
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.train_stead_picking import evaluate


def evaluate_checkpoint(
    checkpoint: str,
    cfg: dict,
    post_process_p_before_s: bool = False,
    device: str | None = None,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    merged = {**ckpt_args, **cfg}

    model = build_picking_model(
        embed_dim=merged.get("embed_dim", 64),
        num_shared_layers=merged.get("num_shared_layers", 2),
        num_branch_layers=merged.get("num_branch_layers", 2),
        gamma=merged.get("gamma", 0.5),
        omega=merged.get("omega", 0.3),
        vp=merged.get("vp", 8.0),
        vs=merged.get("vs", 4.5),
        local_window_sec=merged.get("local_window_sec", 15.0),
        dropout=merged.get("dropout", 0.1),
        per_time_det=merged.get("per_time_det", False),
        pick_head_hidden=merged.get("pick_head_hidden", 24),
        pick_head_kernel=merged.get("pick_head_kernel", 7),
        pick_head_layers=merged.get("pick_head_layers", 3),
        multi_scale=merged.get("multi_scale", False),
        sparse_band=merged.get("sparse_band", False),
        num_anchors=int(merged.get("num_anchors", 0)),
        residual_pick_head=merged.get("residual_pick_head", True),
        residual_det_head=merged.get("residual_det_head", True),
        enhanced_det_head=merged.get("enhanced_det_head", False),
        noise_cancel=merged.get("noise_cancel", False),
        noise_source_dim=merged.get("noise_source_dim", 16),
        noise_det_pick_split=merged.get("noise_det_pick_split", False),
        noise_pick_cues=merged.get("noise_pick_cues", False),
        principle=merged.get("principle", "huygens"),
        obliquity_scale=float(merged.get("obliquity_scale", 1.0)),
    ).to(dev)
    load_picking_model_state(model, ckpt["state_dict"], strict=False)

    seq_len = int(merged.get("seq_len", 400))
    test_ds = STEADPickingDataset(
        "test",
        seq_len=seq_len,
        label_sigma_sec=merged.get("label_sigma_sec", 0.4),
    )
    loader = DataLoader(test_ds, batch_size=merged.get("batch_size", 16), shuffle=False)

    return evaluate(
        model,
        loader,
        dev,
        seq_len=seq_len,
        pick_threshold=merged.get("pick_threshold", 0.3),
        pick_tolerance_sec=merged.get("pick_tolerance_sec", 0.5),
        pick_loss_weight=merged.get("pick_loss_weight", 2.0),
        pick_pos_weight=merged.get("pick_pos_weight", 25.0),
        noise_pick_penalty=merged.get("noise_pick_penalty", 0.05),
        focal_gamma=merged.get("focal_gamma", 0.0),
        s_pick_loss_weight=merged.get("s_pick_loss_weight", 1.0),
        post_process_p_before_s=post_process_p_before_s,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--pick-threshold", type=float, default=None)
    p.add_argument("--post-process-p-before-s", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = {}
    if args.seq_len is not None:
        cfg["seq_len"] = args.seq_len
    if args.pick_threshold is not None:
        cfg["pick_threshold"] = args.pick_threshold
    metrics = evaluate_checkpoint(
        args.checkpoint,
        cfg,
        post_process_p_before_s=args.post_process_p_before_s,
    )
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
