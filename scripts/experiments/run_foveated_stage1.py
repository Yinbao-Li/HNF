#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch foveated engine Stage-1 (BC) + Stage-2 (joint) on STEAD 60 s windows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Foveated STEAD experiment launcher")
    p.add_argument("--output-dir", default="outputs/foveated/stage1_run28")
    p.add_argument("--checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--stage2-epochs", type=int, default=8)
    p.add_argument("--max-event-train", type=int, default=4000)
    p.add_argument("--max-val", type=int, default=400)
    p.add_argument("--max-gazes", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
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
        "tools/train_foveated.py",
        "--output-dir",
        str(out),
        "--checkpoint",
        args.checkpoint,
        "--stage",
        args.stage,
        "--stage1-epochs",
        str(args.stage1_epochs),
        "--stage2-epochs",
        str(args.stage2_epochs),
        "--max-event-train",
        str(args.max_event_train),
        "--max-val",
        str(args.max_val),
        "--max-gazes",
        str(args.max_gazes),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--freeze-backbone",
    ]
    eval_cmd = [
        sys.executable,
        "tools/eval_foveated.py",
        "--checkpoint",
        str(out / "best.pt"),
        "--backbone-checkpoint",
        args.checkpoint,
        "--output",
        str(out / "test_metrics.json"),
        "--max-val",
        "800",
        "--device",
        args.device,
    ]
    viz_cmd = [
        sys.executable,
        "tools/eval_foveated.py",
        "--checkpoint",
        str(out / "best.pt"),
        "--backbone-checkpoint",
        args.checkpoint,
        "--output",
        str(out / "viz_metrics.json"),
        "--max-val",
        "24",
        "--save-trajectory-fig",
        str(out / "gaze_trajectory_sample.png"),
        "--device",
        args.device,
    ]

    banner = "[foveated] " + " ".join(train_cmd)
    print(banner, flush=True)
    print(f"[foveated] logging to {log_path}", flush=True)
    if args.dry_run:
        print("[foveated] eval:", " ".join(eval_cmd))
        print("[foveated] viz:", " ".join(viz_cmd))
        return

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(banner + "\n")
        logf.flush()
        rc = subprocess.call(train_cmd, stdout=logf, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(rc)
        best = out / "best.pt"
        if best.exists():
            logf.write("\n===== EVAL =====\n")
            logf.flush()
            subprocess.call(eval_cmd, stdout=logf, stderr=subprocess.STDOUT)
            logf.write("\n===== VIZ =====\n")
            logf.flush()
            subprocess.call(viz_cmd, stdout=logf, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
