#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run24: wrong-peak / listwise / peak-rerank / low-SNR aug fine-tune from run20.

No distillation. Architecture matches run20 pick/noise heads.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run24 wrong-peak + peak-rerank fine-tune")
    p.add_argument("--resume", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/run24/24_wrongpeak_rerank")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-pick-epochs", type=int, default=1, help="Train pick(+rerank) only first N epochs")
    p.add_argument("--device", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "tools/train_stead_picking.py",
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
        # match run20 architecture
        "--pick-head-hidden",
        "48",
        "--pick-head-layers",
        "4",
        "--pick-head-kernel",
        "7",
        "--noise-source-dim",
        "16",
        "--no-residual-det-head",
        # new: local peak rerank head
        "--peak-rerank",
        "--peak-rerank-hidden",
        "16",
        # wrong-peak emphasis + listwise
        "--wrong-peak-loss-weight",
        "0.35",
        "--wrong-peak-radius-sec",
        "0.45",
        "--wrong-peak-margin",
        "0.30",
        "--s-wrong-peak-scale",
        "2.0",
        "--wrong-peak-listwise-weight",
        "0.5",
        "--wrong-peak-listwise-topk",
        "5",
        "--freeze-all-but-pick-epochs",
        str(args.freeze_pick_epochs),
        # low-SNR augmentation
        "--augment",
        "--aug-noise-snr-min",
        "0.0",
        "--aug-noise-snr-max",
        "12.0",
        "--aug-time-shift-sec",
        "0.05",
        # run20 recipe
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
        "1.8",
        "--det-event-weight",
        "2.0",
        "--score-mode",
        "det_guard",
        "--det-score-floor",
        "0.988",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("[run24]", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
