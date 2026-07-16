#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6 / Domain II Stage-1: train + eval HNF EEG on ds004504."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG Stage-1 train+eval launcher")
    p.add_argument("--output-dir", default="outputs/eeg/adftd_hnf_stage1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
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
        "tools/train_eeg.py",
        "--data-dir",
        "external_data/eeg_adftd",
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--multi-scale",
        "--principle",
        "huygens_fresnel",
        "--no-synthetic",
        "--num-workers",
        "2",
    ]
    eval_cmd = [
        sys.executable,
        "tools/eval_eeg.py",
        "--data-dir",
        "external_data/eeg_adftd",
        "--checkpoint",
        str(out / "best.pt"),
        "--output",
        str(out / "test_metrics.json"),
        "--device",
        args.device,
        "--no-synthetic",
    ]
    analysis_cmd = [
        sys.executable,
        "scripts/domain/run_eeg_analysis.py",
        "--data-dir",
        "external_data/eeg_adftd",
        "--checkpoint",
        str(out / "best.pt"),
        "--metrics-json",
        str(out / "test_metrics.json"),
        "--fig-dir",
        "docs/figures/eeg",
        "--device",
        args.device,
        "--no-synthetic",
    ]

    banner = "[eeg-stage1] " + " ".join(train_cmd)
    print(banner, flush=True)
    print(f"[eeg-stage1] logging to {log_path}", flush=True)
    if args.dry_run:
        print("[eeg-stage1] eval:", " ".join(eval_cmd))
        print("[eeg-stage1] analysis:", " ".join(analysis_cmd))
        return

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        rc = subprocess.call(train_cmd, stdout=logf, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(rc)
        logf.write("\n===== EVAL =====\n")
        logf.flush()
        rc = subprocess.call(eval_cmd, stdout=logf, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(rc)
        logf.write("\n===== ANALYSIS =====\n")
        logf.flush()
        raise SystemExit(subprocess.call(analysis_cmd, stdout=logf, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
