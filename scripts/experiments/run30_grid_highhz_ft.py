#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run29 best -> high-Hz grid FT (incl. 2000 / 6000).

run29 unlocked 400–1000 pts. Probe showed 2000/6000 still collapse. This FT
mixes denser grids; Huygens bands are clamped via --grid-max-band-bins so a
12 GB card can train at T=6000 (physical light-cone shrinks on dense grids).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run29 -> high-Hz grid invariance FT")
    p.add_argument(
        "--resume",
        default="outputs/run29/29_grid_invariance_ft_v2/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/run30/30_grid_highhz_ft")
    p.add_argument("--epochs", type=int, default=8)
    # Peak memory is T=6000 + band clamp; bs=1 + AMP fits ~8 GB activations.
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=48)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--local-window-sec", type=float, default=15.0)
    p.add_argument(
        "--grid-aug-lens",
        default="400,800,1000,2000,4000,6000",
    )
    p.add_argument("--grid-aug-prob", type=float, default=0.75)
    p.add_argument("--val-grid-lens", default="400,2000,6000")
    p.add_argument("--grid-max-band-bins", type=int, default=140)
    p.add_argument("--max-event-train", type=int, default=40000)
    p.add_argument("--max-noise-train", type=int, default=20000)
    p.add_argument("--max-val", type=int, default=1500)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--device", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    cmd = [
        sys.executable,
        "tools/train_stead_picking.py",
        "--resume",
        str(args.resume),
        "--output-dir",
        str(out),
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
        "--grid-aug-lens",
        str(args.grid_aug_lens),
        "--grid-aug-prob",
        str(args.grid_aug_prob),
        "--val-grid-lens",
        str(args.val_grid_lens),
        "--grid-max-band-bins",
        str(args.grid_max_band_bins),
        "--local-window-sec",
        str(args.local_window_sec),
        "--sparse-band",
        "--max-event-train",
        str(args.max_event_train),
        "--max-noise-train",
        str(args.max_noise_train),
        "--max-val",
        str(args.max_val),
        "--reset-best-score",
    ]
    if args.amp:
        cmd.append("--amp")
    cmd.extend([
        "--embed-dim", "64",
        "--num-shared-layers", "2",
        "--num-branch-layers", "2",
        "--seed", "42",
        "--multi-scale",
        "--principle", "huygens_fresnel",
        "--obliquity-scale", "1.0",
        "--rho-sparsity-weight", "0.02",
        "--rho-sparsity-radius-sec", "1.5",
        "--kernel-phys-prior-weight", "0.005",
        "--pick-head-hidden", "48",
        "--pick-head-layers", "4",
        "--pick-head-kernel", "7",
        "--noise-source-dim", "16",
        "--no-residual-det-head",
        "--enhanced-det-head",
        "--noise-cancel",
        "--noise-pick-cues",
        "--noise-det-pick-split",
        "--noise-cancel-weight", "0.05",
        "--wrong-peak-loss-weight", "0.15",
        "--wrong-peak-radius-sec", "0.45",
        "--wrong-peak-margin", "0.25",
        "--s-wrong-peak-scale", "1.35",
        "--ps-order-loss-weight", "0.12",
        "--ps-min-gap-sec", "0.1",
        "--post-process-p-before-s",
        "--pick-loss-weight", "2.8",
        "--pick-pos-weight", "28",
        "--p-pick-loss-weight", "1.3",
        "--s-pick-loss-weight", "1.6",
        "--det-event-weight", "2.0",
        "--label-sigma-sec", "0.35",
        "--score-mode", "det_guard",
        "--det-score-floor", "0.988",
    ])
    if args.device:
        cmd.extend(["--device", args.device])

    banner = "[run30-highhz] " + " ".join(cmd)
    print(banner, flush=True)
    print(f"[run30-highhz] logging to {log_path}", flush=True)
    if args.dry_run:
        return
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        raise SystemExit(subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
