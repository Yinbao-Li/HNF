#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBS-native from-scratch training (STEAD run28 recipe on full OBS corpus).

Mirrors scripts/experiments/run28_stead_ms_fresnel_phys.py, but:
  - data = SeisBench OBS (Z12 → 3C), full downloaded chunks
  - split = outputs/obs_full_native_split/split.json
  - init = random (--from-scratch), not STEAD transfer
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
    p = argparse.ArgumentParser(description="OBS-native from-scratch (run28 recipe)")
    p.add_argument("--output-dir", default="outputs/run_obs_native/obs_ms_fresnel_phys_20ep")
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument("--chunks", default=ALL_CHUNKS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print("[obs-native] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_split:
        run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "build_obs_full_split.py"),
                "--chunks",
                args.chunks,
                "--output",
                args.split_json,
            ],
            args.dry_run,
        )

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
        # run28 architecture / loss (defaults already match; set explicitly)
        "--multi-scale",
        "--sparse-band",
        "--principle",
        "huygens_fresnel",
        "--noise-cancel",
        "--noise-pick-cues",
        "--noise-det-pick-split",
        "--enhanced-det-head",
        "--wrong-peak-loss-weight",
        "0.15",
        "--ps-order-loss-weight",
        "0.12",
        "--pick-loss-weight",
        "2.8",
        "--pick-pos-weight",
        "28",
        "--p-pick-loss-weight",
        "1.3",
        "--s-pick-loss-weight",
        "1.6",
        "--label-sigma-sec",
        "0.35",
        "--local-window-sec",
        "15.0",
    ]
    run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
