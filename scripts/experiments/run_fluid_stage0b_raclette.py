#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-0b launcher: preprocess RACLETTE GT slices → train → eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RACLETTE Stage-0b launcher")
    p.add_argument("--output-dir", default="outputs/fluid/stage0b_raclette")
    p.add_argument("--cache", default="external_data/raclette_cache/gt_slices.npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--keep-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    # Preprocess needs CPython 3.10 + pyvista_zstd
    py310 = "/usr/bin/python3"
    preprocess_cmd = [
        py310,
        "tools/preprocess_raclette_slices.py",
        "--out",
        args.cache,
    ]
    train_cmd = [
        sys.executable,
        "tools/train_raclette_stage0b.py",
        "--cache",
        args.cache,
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--keep-frac",
        str(args.keep_frac),
        "--device",
        args.device,
    ]
    eval_cmd = [
        sys.executable,
        "tools/eval_raclette_stage0b.py",
        "--checkpoint",
        str(out / "best.pt"),
        "--cache",
        args.cache,
        "--output",
        str(out / "test_metrics.json"),
        "--device",
        args.device,
    ]

    if args.dry_run:
        print(preprocess_cmd)
        print(train_cmd)
        print(eval_cmd)
        return

    with log_path.open("w", encoding="utf-8") as logf:
        if not args.skip_preprocess and not Path(args.cache).is_file():
            logf.write("$ " + " ".join(preprocess_cmd) + "\n")
            logf.flush()
            rc = subprocess.call(preprocess_cmd, stdout=logf, stderr=subprocess.STDOUT)
            if rc != 0:
                raise SystemExit(rc)
        logf.write("$ " + " ".join(train_cmd) + "\n")
        logf.flush()
        rc = subprocess.call(train_cmd, stdout=logf, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(rc)
        logf.write("\n===== EVAL =====\n")
        logf.flush()
        raise SystemExit(subprocess.call(eval_cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
