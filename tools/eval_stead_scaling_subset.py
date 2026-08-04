#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate an HNF STEAD checkpoint on the shared scaling-law test subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.analyze_stead_picking import load_model  # noqa: E402
from tools.train_stead_picking import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--max-events", type=int, default=8000)
    p.add_argument("--max-noise", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--tol-sec", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.checkpoint, device)
    seq_len = int(meta.get("seq_len", 800))
    ds = STEADPickingDataset("test", seq_len=seq_len, load_geometry=False)
    ev = [i for i, r in enumerate(ds.refs) if r.is_event == 1]
    nz = [i for i, r in enumerate(ds.refs) if r.is_event == 0]
    rng = np.random.default_rng(args.seed)
    if len(ev) > args.max_events:
        ev = sorted(rng.choice(ev, size=args.max_events, replace=False).tolist())
    if len(nz) > args.max_noise:
        nz = sorted(rng.choice(nz, size=args.max_noise, replace=False).tolist())
    loader = DataLoader(Subset(ds, ev + nz), batch_size=args.batch_size, shuffle=False, num_workers=2)
    metrics = evaluate(
        model,
        loader,
        device,
        seq_len,
        args.pick_threshold,
        args.tol_sec,
        post_process_p_before_s=True,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cargs = ckpt.get("args", {})
    if hasattr(cargs, "__dict__"):
        cargs = vars(cargs)
    metrics["n_params"] = int(ckpt.get("n_params") or sum(p.numel() for p in model.parameters()))
    metrics["n_event_train"] = int(cargs.get("max_event_train", -1))
    metrics["n_noise_train"] = int(cargs.get("max_noise_train", -1))
    metrics["model"] = "hnf"
    metrics["n_eval"] = len(ev) + len(nz)
    Path(args.output_json).write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ["det_f1", "p_f1", "s_f1", "p_mae_sec", "s_mae_sec", "n_params"]}, indent=2))


if __name__ == "__main__":
    main()
