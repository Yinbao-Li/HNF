#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end runner for run28 retraining on OBS and cross-domain comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run28 full OBS retrain + compare")
    p.add_argument("--output-dir", default="outputs/run28_obs_full_800")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="")
    p.add_argument("--chunks", default="201805,201806,201807,201808")
    p.add_argument("--split-json", default="outputs/obs_matched_adapt_split_randoffset/split.json")
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--compare-output-dir", default="outputs/obs_step4_run28_obsfull")
    p.add_argument("--explain-output-dir", default="")
    p.add_argument("--paper-output-dir", default="outputs/paper_stead_obs_interpret_compare")
    p.add_argument("--stead-explain", default="outputs/stead_hnf_picking_run7/explain/explain_summary.json")
    p.add_argument("--geo-confirm", default="outputs/paper_geo_confirm/geo_confirm_report.json")
    p.add_argument("--eqt-adapt-checkpoint", default="outputs/obs_light_adapt_eqt_randoff/best.pt")
    p.add_argument("--phasenet-adapt-checkpoint", default="outputs/obs_light_adapt_phasenet_offset8_dj/best.pt")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-compare", action="store_true")
    p.add_argument("--skip-explain", action="store_true")
    p.add_argument("--skip-paper", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    banner = "[run28-obs] " + " ".join(cmd)
    print(banner, flush=True)
    if dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ckpt = out_dir / "best.pt"

    if not args.skip_train:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "train_obs_picking.py"),
            "--output-dir",
            str(out_dir),
            "--seq-len",
            str(args.seq_len),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--lr",
            str(args.lr),
            "--chunks",
            args.chunks,
            "--max-events",
            str(args.max_events),
            "--augment",
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.split_json.strip():
            cmd += ["--split-json", args.split_json.strip()]
        run(cmd, args.dry_run)

    if not args.skip_compare:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"),
            "--checkpoint",
            str(train_ckpt),
            "--hnf-label",
            "HNF(run28/OBS-full)",
            "--output-dir",
            args.compare_output_dir,
            "--seq-len",
            str(args.seq_len),
            "--chunks",
            args.chunks,
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.split_json.strip():
            cmd += ["--split-json", args.split_json.strip()]
        if args.eqt_adapt_checkpoint.strip():
            cmd += ["--eqt-adapt-checkpoint", args.eqt_adapt_checkpoint.strip()]
        if args.phasenet_adapt_checkpoint.strip():
            cmd += ["--phasenet-adapt-checkpoint", args.phasenet_adapt_checkpoint.strip()]
        run(cmd, args.dry_run)

    explain_out = args.explain_output_dir or str(out_dir / "explain_obs")
    if not args.skip_explain:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "explain_obs_picking.py"),
            "--checkpoint",
            str(train_ckpt),
            "--output-dir",
            explain_out,
            "--seq-len",
            str(args.seq_len),
            "--chunks",
            args.chunks,
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.split_json.strip():
            cmd += ["--split-json", args.split_json.strip()]
        run(cmd, args.dry_run)

    if not args.skip_paper:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "run_paper_stead_obs_interpret_compare.py"),
            "--stead-explain",
            args.stead_explain,
            "--obs-explain",
            str(Path(explain_out) / "explain_summary.json"),
            "--obs-compare",
            str(Path(args.compare_output_dir) / "obs_picking_compare_report.json"),
            "--geo-confirm",
            args.geo_confirm,
            "--output-dir",
            args.paper_output_dir,
        ]
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
