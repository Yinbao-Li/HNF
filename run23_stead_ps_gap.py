#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run23: fine-tune a dedicated S-P gap head on top of run20.

Architecture flags MUST match run20/20_wrongpeak_sharp, otherwise pick/noise
weights are shape-skipped on resume and validation F1 collapses.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune PS gap head from run20")
    p.add_argument(
        "--resume",
        default="outputs/run20/20_wrongpeak_sharp/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/run23/23_ps_gap_head")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--freeze-gap-epochs", type=int, default=2, help="Train only gap head first")
    p.add_argument("--ps-gap-loss-weight", type=float, default=1.0)
    p.add_argument("--ps-gap-consist-weight", type=float, default=0.15)
    p.add_argument("--device", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "train_stead_picking.py",
        "--resume",
        args.resume,
        "--reset-best-score",
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seq-len",
        "800",
        # --- match run20 architecture (critical for resume) ---
        "--pick-head-hidden",
        "48",
        "--pick-head-layers",
        "4",
        "--pick-head-kernel",
        "7",
        "--noise-source-dim",
        "16",
        "--no-residual-det-head",  # run20: residual_det_head=False; pick residual stays default True
        # --- gap head ---
        "--predict-ps-gap",
        "--ps-gap-hidden",
        "64",
        "--ps-gap-loss-weight",
        str(args.ps_gap_loss_weight),
        "--ps-gap-consist-weight",
        str(args.ps_gap_consist_weight),
        "--freeze-all-but-gap-epochs",
        str(args.freeze_gap_epochs),
        # --- run20 picking / noise recipe ---
        "--wrong-peak-loss-weight",
        "0.15",
        "--wrong-peak-radius-sec",
        "0.45",
        "--wrong-peak-margin",
        "0.25",
        "--s-wrong-peak-scale",
        "1.35",
        "--ps-order-loss-weight",
        "0.12",
        "--ps-min-gap-sec",
        "0.1",
        "--post-process-p-before-s",
        "--noise-cancel",
        "--noise-pick-cues",
        "--noise-det-pick-split",
        "--enhanced-det-head",
        "--noise-cancel-weight",
        "0.05",
        "--pick-loss-weight",
        "2.8",
        "--pick-pos-weight",
        "28",
        "--p-pick-loss-weight",
        "1.3",
        "--s-pick-loss-weight",
        "1.6",
        "--det-event-weight",
        "2.0",
        "--score-mode",
        "det_guard",
        "--det-score-floor",
        "0.988",
        # ComplexHalf matmul unsupported; keep FP32 unless --amp is forced later.
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("[run23]", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
