#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an interpretable picking pattern library from dense run28 forwards.

Example:
  PYTHONPATH=. python tools/build_pattern_library.py \\
    --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt \\
    --max-event 2000 --max-noise 1000 --k 6 \\
    --output-dir outputs/pattern_library_run28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.pattern_library import (
    FEATURE_NAMES,
    PatternLibrary,
    downsample_trace,
    extract_pattern_features,
    features_to_vector,
)
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model
from tools.train_stead_picking import move_batch_to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HNF picking pattern library")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/pattern_library_run28")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--max-event", type=int, default=2000)
    p.add_argument("--max-noise", type=int, default=1000)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument(
        "--coarse-len",
        type=int,
        default=200,
        help="grid used for routing features; must match eval's coarse pass",
    )
    p.add_argument("--coarse-bypass-nc", action="store_true", default=True)
    p.add_argument("--no-coarse-bypass-nc", action="store_false", dest="coarse_bypass_nc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-batches", type=int, default=0, help="0=all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    ds = STEADPickingDataset(
        args.split,
        seq_len=args.seq_len,
        max_event_traces=args.max_event,
        max_noise_traces=args.max_noise,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows = []
    feat_mat = []
    is_event = []
    p_mae = []
    s_mae = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            x = batch["x"]
            t = batch["t"][0] if batch["t"].dim() == 3 else batch["t"]
            x_c, t_c = downsample_trace(x, t, args.coarse_len, window_sec=60.0)
            feat = extract_pattern_features(
                model,
                x_c,
                t_c,
                window_sec=60.0,
                pick_threshold=args.pick_threshold,
                bypass_noise_cancel=bool(args.coarse_bypass_nc),
            )
            ev = bool(float(batch["det"][0].item()) > 0.5)
            # timing error vs labels when event
            mae_p = mae_s = 0.0
            if ev and feat["p_sec"] >= 0:
                gt_p = float(batch["p_idx"][0].item()) / max(args.seq_len - 1, 1) * 60.0
                mae_p = abs(feat["p_sec"] - gt_p)
            if ev and feat["s_sec"] >= 0 and float(batch["s_valid"][0].item()) > 0.5:
                gt_s = float(batch["s_idx"][0].item()) / max(args.seq_len - 1, 1) * 60.0
                mae_s = abs(feat["s_sec"] - gt_s)

            feat_mat.append(features_to_vector(feat))
            is_event.append(1 if ev else 0)
            p_mae.append(mae_p)
            s_mae.append(mae_s)
            rows.append({"feat": feat, "is_event": ev, "p_mae": mae_p, "s_mae": mae_s})
            if (bi + 1) % 200 == 0:
                print(f"[pattern-lib] extracted {bi+1}/{len(ds)}", flush=True)

    feat_mat = np.stack(feat_mat, axis=0)
    lib = PatternLibrary.build_from_feature_matrix(
        feat_mat,
        is_event=np.asarray(is_event),
        p_mae=np.asarray(p_mae),
        s_mae=np.asarray(s_mae),
        k=args.k,
        seed=args.seed,
        feature_names=FEATURE_NAMES,
        window_sec=60.0,
        seq_len=args.seq_len,
        coarse_len=args.coarse_len,
    )
    lib_path = out / "pattern_library.json"
    lib.save(lib_path)
    np.savez(
        out / "features.npz",
        feats=feat_mat,
        is_event=np.asarray(is_event),
        p_mae=np.asarray(p_mae),
        s_mae=np.asarray(s_mae),
        names=np.asarray(list(FEATURE_NAMES)),
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "n_samples": int(feat_mat.shape[0]),
        "k": args.k,
        "coarse_len": args.coarse_len,
        "coarse_bypass_nc": bool(args.coarse_bypass_nc),
        "elapsed_sec": time.time() - t0,
        "prototypes": lib.summary(),
        "feature_names": list(FEATURE_NAMES),
    }
    (out / "build_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[pattern-lib] wrote {lib_path}", flush=True)


if __name__ == "__main__":
    main()
