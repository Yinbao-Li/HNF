#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train learned causal peak-rank head on L1200 board (backbone frozen).

Uses Huygens P-field + rho crops at pick-curve local peaks; listwise CE to
the candidate nearest GT. Inference: learned_causal_rank decode.
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_causal_peak_rank_6ep",
    )
    p.add_argument(
        "--resume",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument("--epochs", type=int, default=6)
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
        "4",
        "--grad-accum-steps",
        "12",
        "--lr",
        "3e-4",
        "--seq-len",
        "1200",
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
        "2.0",
        "--exist-th",
        "0.60",
        "--soft-gate-s-weight",
        "0.35",
        "--channel-mask-mode",
        "soft_h",
        "--enhanced-det-head",
        "--causal-peak-rank",
        "--causal-peak-rank-hidden",
        "48",
        "--causal-peak-rank-topk",
        "8",
        "--causal-peak-rank-crop-half",
        "16",
        "--causal-peak-rank-loss-weight",
        "1.0",
        "--causal-peak-rank-radius-sec",
        "0.5",
        "--train-causal-rank-only",
        "--p-decode-mode",
        "learned_causal_rank",
        "--decode-late-penalty",
        "0.60",
        "--wrong-peak-loss-weight",
        "0.15",
        "--ps-order-loss-weight",
        "0.12",
        "--pick-loss-weight",
        "2.8",
        "--pick-pos-weight",
        "32",
        "--label-sigma-sec",
        "0.30",
        "--local-window-sec",
        "12.0",
        "--gate-mode",
        "soft_floor",
        "--soft-th",
        "0.25",
    ]
    print("[obs-causal-rank] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
