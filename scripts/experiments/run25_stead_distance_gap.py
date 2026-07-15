#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run25: inject catalog distance S−P prior into training (no distillation).

Resumes run20 with matching architecture. Adds differentiable
distance_gap_consistency_loss so soft P/S expectations track μ=0.119·km —
the same geometric knowledge that helped at inference.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run25 distance-gap knowledge fine-tune")
    p.add_argument("--resume", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/run25/25_distance_gap")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=5e-5)
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
        # match run20 architecture (critical)
        "--pick-head-hidden",
        "48",
        "--pick-head-layers",
        "4",
        "--pick-head-kernel",
        "7",
        "--noise-source-dim",
        "16",
        "--no-residual-det-head",
        # geometric knowledge: soft P/S gap vs distance prior
        "--distance-gap-loss-weight",
        "0.4",
        "--distance-gap-s-per-km",
        "0.119",
        "--distance-gap-sigma-sec",
        "1.2",
        # keep run20 wrong-peak recipe (no peak_rerank / no heavy listwise)
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
        # light freeze of backbone first epoch to protect run20 picks
        "--freeze-backbone-epochs",
        "1",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("[run25]", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
