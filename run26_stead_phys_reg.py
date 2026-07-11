#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run26: physics regularizers on run20 (resume, same architecture).

Decision: continue from run20, do NOT train from scratch.
- Same graph → pick heads preserved
- New losses only: rho event sparsity + gamma/omega/c prior
- Kernel: omega softplus (value-preserving remap) + c_log_scale (init identity)
- Multi-scale deferred (architecture change → separate from-scratch run)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run26 physics-reg fine-tune from run20")
    p.add_argument("--resume", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/run26/26_phys_reg")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-5)
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
        # physics regularizers (this run's point)
        "--rho-sparsity-weight",
        "0.08",
        "--rho-sparsity-radius-sec",
        "1.5",
        "--kernel-phys-prior-weight",
        "0.02",
        # keep run20 wrong-peak recipe
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
        # do NOT freeze backbone: medium_net must learn sparse rho
        "--freeze-backbone-epochs",
        "0",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("[run26]", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    log_path = out / "train.log"
    print(f"[run26] logging to {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write("[run26] " + " ".join(cmd) + "\n")
        logf.flush()
        raise SystemExit(subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
