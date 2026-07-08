#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run11 ablation: data aug / multi-scale DeepHuygens on ablation-01 base."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "run11"
STATE_PATH = OUT_ROOT / "state.json"

# Ablation 01 strict test baseline
BASELINE = {
    "det_f1": 0.9889249729369639,
    "p_f1": 0.904525849450614,
    "s_f1": 0.9042715072500761,
    "ps_sum": 1.80879735670069,
}

BASE_RESUME = ROOT / "outputs" / "ablation" / "01_seq800" / "best.pt"

COMMON = [
    sys.executable,
    str(ROOT / "train_stead_picking.py"),
    "--seq-len",
    "800",
    "--batch-size",
    "12",
    "--grad-accum-steps",
    "4",
    "--epochs",
    "8",
    "--num-workers",
    "0",
    "--embed-dim",
    "64",
    "--num-shared-layers",
    "2",
    "--num-branch-layers",
    "2",
    "--lr",
    "5e-4",
    "--pick-pos-weight",
    "25",
    "--label-sigma-sec",
    "0.4",
    "--local-window-sec",
    "15.0",
    "--seed",
    "42",
]

RUNS = [
    (
        "11a_aug",
        {"augment": True},
        str(BASE_RESUME),
    ),
    (
        "11b_multiscale",
        {"multi_scale": True},
        str(BASE_RESUME),
    ),
    (
        "11c_aug_multiscale",
        {"augment": True, "multi_scale": True},
        str(BASE_RESUME),
    ),
]


def build_cmd(name: str, flags: dict[str, bool], resume: str) -> list[str]:
    cmd = COMMON + ["--output-dir", str(OUT_ROOT / name)]
    if resume:
        cmd += ["--resume", resume]
    if flags.get("augment"):
        cmd.append("--augment")
    if flags.get("multi_scale"):
        cmd.append("--multi-scale")
    return cmd


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"completed": [], "results": []}


def save_state(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run11 STEAD picking experiments")
    p.add_argument("--only", default=None, help="Run single experiment name, e.g. 11a_aug")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    state = load_state()
    runs = RUNS
    if args.only:
        runs = [r for r in RUNS if r[0] == args.only]
        if not runs:
            raise SystemExit(f"Unknown run: {args.only}")

    for name, flags, resume in runs:
        if name in state.get("completed", []):
            print(f"[run11] skip completed {name}", flush=True)
            continue
        cmd = build_cmd(name, flags, resume)
        print(f"[run11] >>> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"[run11] FAILED {name} code={proc.returncode}", flush=True)
            raise SystemExit(proc.returncode)

        metrics_path = OUT_ROOT / name / "test_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            entry = {
                "name": name,
                "flags": flags,
                "det_f1": metrics.get("det_f1"),
                "p_f1": metrics.get("p_f1"),
                "s_f1": metrics.get("s_f1"),
                "ps_sum": metrics.get("p_f1", 0) + metrics.get("s_f1", 0),
            }
            state.setdefault("results", []).append(entry)
        state.setdefault("completed", []).append(name)
        save_state(state)

    print("[run11] done", flush=True)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
