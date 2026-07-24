#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBS-native 4C bottleneck-fix training.

Priority recipe vs prior 3C from-scratch:
  1) input_dim=4 (Z12H, keep hydrophone)
  2) noise_cancel_weight=0.05 (actually supervise NC branch)
  3) stronger P: p_pick_loss_weight=2.0, wrong_peak=0.35
  4) enable preserve_gate for weak OBS onsets
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ALL_CHUNKS = (
    "201805,201806,201807,201808,201809,201810,201811,201812,"
    "201901,201902,201903,201904,201905,201906,201907,201908,000000"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS 4C native bottleneck-fix train")
    p.add_argument("--output-dir", default="outputs/run_obs_native/obs_4c_nc_pboost_20ep")
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument("--chunks", default=ALL_CHUNKS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "train_obs_picking.py"),
        "--output-dir",
        str(out),
        "--from-scratch",
        "--split-json",
        args.split_json,
        "--chunks",
        args.chunks,
        "--input-dim",
        "4",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--lr",
        str(args.lr),
        "--seq-len",
        str(args.seq_len),
        "--device",
        args.device,
        "--augment",
        "--multi-scale",
        "--sparse-band",
        "--principle",
        "huygens_fresnel",
        "--noise-cancel",
        "--noise-pick-cues",
        "--noise-det-pick-split",
        "--enable-preserve-gate",
        "--noise-cancel-weight",
        "0.05",
        "--enhanced-det-head",
        "--wrong-peak-loss-weight",
        "0.35",
        "--wrong-peak-margin",
        "0.30",
        "--ps-order-loss-weight",
        "0.12",
        "--pick-loss-weight",
        "2.8",
        "--pick-pos-weight",
        "32",
        "--p-pick-loss-weight",
        "2.0",
        "--s-pick-loss-weight",
        "1.4",
        "--label-sigma-sec",
        "0.35",
        "--local-window-sec",
        "15.0",
    ]
    print("[obs-4c] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
