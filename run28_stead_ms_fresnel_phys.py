#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run28: from-scratch kitchen-sink physics train (20 epochs, single cosine).

Enabled by request:
  - causal kernel (default True)
  - multi-scale HNF
  - gamma/omega physical prior (weak)
  - rho event sparsity (weak)
  - c learnable scale (in HuygensKernel)
  - Kirchhoff / Huygens–Fresnel (full)

Also keeps run20-style pick recipe (wrong-peak, noise split+cues).
Logs to train.log (not .out).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run28 multi-scale Fresnel phys from-scratch 20ep")
    p.add_argument("--output-dir", default="outputs/run28/28_ms_fresnel_phys_20ep")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
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
        "train_stead_picking.py",
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
        "800",
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
        # --- requested physics / architecture ---
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
        # --- run20-like pick recipe ---
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

    banner = "[run28] " + " ".join(cmd)
    print(banner, flush=True)
    print(f"[run28] logging to {log_path}", flush=True)
    if args.dry_run:
        return
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        raise SystemExit(subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
