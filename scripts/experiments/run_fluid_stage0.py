#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7 / Domain III Stage-0: synthetic sparse→dense fluid recon + eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fluid Stage-0 launcher")
    p.add_argument("--output-dir", default="outputs/fluid/stage0_synth")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--keep-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    train_cmd = [
        sys.executable,
        "tools/train_fluid.py",
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--keep-frac",
        str(args.keep_frac),
        "--device",
        args.device,
    ]
    eval_cmd = [
        sys.executable,
        "tools/eval_fluid.py",
        "--checkpoint",
        str(out / "best.pt"),
        "--output",
        str(out / "test_metrics.json"),
        "--device",
        args.device,
    ]

    banner = "[fluid-stage0] " + " ".join(train_cmd)
    print(banner, flush=True)
    if args.dry_run:
        print("[fluid-stage0] eval:", " ".join(eval_cmd))
        return

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        rc = subprocess.call(train_cmd, stdout=logf, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(rc)
        logf.write("\n===== EVAL =====\n")
        logf.flush()
        raise SystemExit(subprocess.call(eval_cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
