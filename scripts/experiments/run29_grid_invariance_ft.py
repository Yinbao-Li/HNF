#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run28 best -> grid-invariant FT: same 60 s window, mixed sample rates.

The run28 checkpoint only works at 800 points / 60 s. Feeding the same window
at 400 or 200 points drops det AUC from 0.997 to ~0.85 (event det median
0.96 -> 0.000), and re-gridding a crop to 800 points loses 71 of 79 P picks.
That single limitation blocks the 100 Hz / 6000-sample fine-tune and every
coarse-to-fine inference speedup, since each cheap tier needs another grid.

Here each step draws a grid length from --grid-aug-lens and resamples the batch
(waveform + every pick label) onto it. Physics kernels run on real time, so
they can generalise; the conv heads are bin-indexed and have to adapt.

Best-checkpoint selection stays on the native 800 grid so the FT cannot trade
away current accuracy; the extra grids are logged each epoch for monitoring.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run28 -> grid-invariance FT")
    p.add_argument(
        "--resume",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/run29/29_grid_invariance_ft_v2")
    p.add_argument("--epochs", type=int, default=8)
    # Peak memory is set by the longest grid (not the native 800). On an 11 GB
    # card, T=1200 + bs=8 + fp32 OOMs in the Huygens kernel; keep bs=4 + AMP
    # and cap training grids at 1000. Effective batch stays 48 via accum.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--local-window-sec", type=float, default=15.0)
    p.add_argument("--grid-aug-lens", default="400,600,800,1000")
    p.add_argument("--grid-aug-prob", type=float, default=0.6)
    p.add_argument("--val-grid-lens", default="400,1000")
    p.add_argument("--max-event-train", type=int, default=120000)
    p.add_argument("--max-noise-train", type=int, default=60000)
    p.add_argument("--max-val", type=int, default=4000)
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
        # architecture below must match the run28 checkpoint or resume silently
        # drops weights
        "--embed-dim",
        "64",
        "--num-shared-layers",
        "2",
        "--num-branch-layers",
        "2",
        "--seed",
        "42",
        "--multi-scale",
        "--principle",
        "huygens_fresnel",
        "--obliquity-scale",
        "1.0",
        "--rho-sparsity-weight",
        "0.02",
        "--rho-sparsity-radius-sec",
        "1.5",
        "--kernel-phys-prior-weight",
        "0.005",
        "--pick-head-hidden",
        "48",
        "--pick-head-layers",
        "4",
        "--pick-head-kernel",
        "7",
        "--noise-source-dim",
        "16",
        "--no-residual-det-head",
        "--enhanced-det-head",
        "--noise-cancel",
        "--noise-pick-cues",
        "--noise-det-pick-split",
        "--noise-cancel-weight",
        "0.05",
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
        "--label-sigma-sec",
        "0.35",
        "--score-mode",
        "det_guard",
        "--det-score-floor",
        "0.988",
    ])
    if args.device:
        cmd.extend(["--device", args.device])

    banner = "[run29-gridinv] " + " ".join(cmd)
    print(banner, flush=True)
    print(f"[run29-gridinv] logging to {log_path}", flush=True)
    if args.dry_run:
        return
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        raise SystemExit(subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
