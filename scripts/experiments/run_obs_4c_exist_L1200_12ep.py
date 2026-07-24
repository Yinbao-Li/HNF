#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune OBS 4C+exist from-scratch ckpt at seq_len=1200 (P-side context boost).

Resume weights; keep phase_exist + NC; modest LR; 12 epochs.
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
    p = argparse.ArgumentParser(description="OBS 4C+exist L1200 fine-tune")
    p.add_argument(
        "--resume",
        default="outputs/run_obs_native/obs_4c_exist_fromscratch_30ep/best.pt",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=1200)
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
        "--resume",
        args.resume,
        "--split-json",
        "outputs/obs_full_native_split/split.json",
        "--chunks",
        ALL_CHUNKS,
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
        "--phase-exist",
        "--exist-loss-weight",
        "1.0",
        "--s-exist-pos-weight",
        "1.35",
        "--s-exist-neg-weight",
        "1.0",
        "--exist-th",
        "0.60",
        "--enhanced-det-head",
        "--wrong-peak-loss-weight",
        "0.45",
        "--wrong-peak-margin",
        "0.30",
        "--ps-order-loss-weight",
        "0.12",
        "--pick-loss-weight",
        "2.8",
        "--pick-pos-weight",
        "32",
        "--p-pick-loss-weight",
        "2.4",
        "--s-pick-loss-weight",
        "1.2",
        "--label-sigma-sec",
        "0.30",
        "--local-window-sec",
        "12.0",
    ]
    print("[obs-L1200] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
