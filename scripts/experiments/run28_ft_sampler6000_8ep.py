#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run28 best → 100 Hz / 6000-sample input, grid-aligned FT via learnable sampler.

Rationale (vs the failed native-6000 head-only probe):
  - Feed native 6000@100Hz, but the LearnableTemporalSampler warps it to the
    run28 backbone grid (seq_len=800). backbone + noise_cancel + Huygens layers
    all run at 800, so local_window=15s stays ~200 bins == run28 (no cone/grid
    mismatch, no OOM).
  - Staged unfreeze: ep1-2 freeze backbone (train pick heads + sampler + det),
    ep3+ unfreeze everything at a small LR. Avoids the cold-start collapse of
    freeze-all-but-pick.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run28 → sampler 6000->800 grid-aligned FT")
    p.add_argument(
        "--resume",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/run28/28_ft_sampler6000_8ep")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--input-seq-len", type=int, default=6000)
    p.add_argument("--local-window-sec", type=float, default=15.0)
    p.add_argument("--max-event-train", type=int, default=120000)
    p.add_argument("--max-noise-train", type=int, default=60000)
    p.add_argument("--max-val", type=int, default=8000)
    p.add_argument("--freeze-backbone-epochs", type=int, default=2)
    p.add_argument("--sampler-align-weight", type=float, default=0.2)
    p.add_argument("--sampler-temperature", type=float, default=0.05)
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
        "--input-seq-len",
        str(args.input_seq_len),
        "--learnable-sampler",
        "--sampler-temperature",
        str(args.sampler_temperature),
        "--sampler-align-weight",
        str(args.sampler_align_weight),
        "--local-window-sec",
        str(args.local_window_sec),
        "--sparse-band",
        "--max-event-train",
        str(args.max_event_train),
        "--max-noise-train",
        str(args.max_noise_train),
        "--max-val",
        str(args.max_val),
        "--freeze-backbone-epochs",
        str(args.freeze_backbone_epochs),
        "--reset-best-score",
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
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    banner = "[run28-ftsampler] " + " ".join(cmd)
    print(banner, flush=True)
    print(f"[run28-ftsampler] logging to {log_path}", flush=True)
    if args.dry_run:
        return
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        raise SystemExit(subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
