#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune L1200 board with peak-rerank head + late-aware P decode (8ep).

FNO / spectral-sideband path removed; start from L1200 HNF board.
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
        default="outputs/run_obs_native/obs_4c_peak_rerank_decode_8ep",
    )
    p.add_argument(
        "--resume",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument("--epochs", type=int, default=8)
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
        "2",
        "--grad-accum-steps",
        "24",
        "--lr",
        "1e-4",
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
        "--peak-rerank",
        "--peak-rerank-hidden",
        "16",
        "--p-decode-mode",
        "score_minus_late",
        "--decode-late-penalty",
        "0.60",
        "--wrong-peak-loss-weight",
        "0.55",
        "--wrong-peak-margin",
        "0.30",
        "--wrong-peak-radius-sec",
        "0.45",
        "--p-late-wrong-peak-weight",
        "1.0",
        "--wrong-peak-listwise-weight",
        "0.25",
        "--wrong-peak-listwise-topk",
        "5",
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
    print("[obs-peak-rerank] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
