#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run29c: Bayesian–MC + multi-scale + learnable temporal sampler.

Same speed knobs as run29b, plus:
  - learnable importance sampler (input_seq_len -> seq_len)
  - sampler alignment loss
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run29c Bayes-MC + learnable sampler @800")
    p.add_argument("--output-dir", default="outputs/run29/29c_bayes_mc_sampler_ms_30ep")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--input-seq-len", type=int, default=1600)
    p.add_argument("--mc-n-samples", type=int, default=4)
    p.add_argument("--sampler-align-weight", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
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
        "--num-workers",
        str(args.num_workers),
        "--amp",
        "--embed-dim",
        "64",
        "--num-shared-layers",
        "2",
        "--num-branch-layers",
        "2",
        "--local-window-sec",
        "15.0",
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
        "--sparse-band",
        "--bayesian-mc-kernel",
        "--mc-n-samples",
        str(args.mc_n_samples),
        "--learnable-sampler",
        "--sampler-align-weight",
        str(args.sampler_align_weight),
        "--pick-head-hidden",
        "48",
        "--pick-head-layers",
        "4",
        "--pick-head-kernel",
        "7",
        "--no-residual-det-head",
        "--enhanced-det-head",
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

    banner = "[run29c] " + " ".join(cmd)
    print(banner, flush=True)
    print(f"[run29c] logging to {log_path}", flush=True)
    if args.dry_run:
        return
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        raise SystemExit(subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
