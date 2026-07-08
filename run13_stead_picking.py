#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run13: fixed run12 compute strategy (anchors) + run11 ablation on top.

Base (no ablation flags):
  - seq800, 01 architecture, scalar det
  - --num-anchors 128  (run12, adopted without ablation)
  - sparse-band omitted (GPU slower than dense at seq800)

Ablations (run11-style, sequential from 01 checkpoint):
  - 13_base              (no residual pick/det heads)
  - 13_aug               (+ residual heads, default on)
  - 13_multiscale
  - 13_aug_multiscale
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "run13"
STATE_PATH = OUT_ROOT / "state.json"
BASE_RESUME = ROOT / "outputs" / "ablation" / "01_seq800" / "best.pt"

COMMON = [
    sys.executable,
    str(ROOT / "train_stead_picking.py"),
    "--seq-len",
    "800",
    "--batch-size",
    "24",
    "--grad-accum-steps",
    "2",
    "--epochs",
    "8",
    "--num-workers",
    "2",
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
    "--num-anchors",
    "128",
]

RUNS = [
    ("13_base", {}),
    ("13_aug", {"augment": True}),
    ("13_multiscale", {"multi_scale": True}),
    ("13_aug_multiscale", {"augment": True, "multi_scale": True}),
]


def build_cmd(name: str, flags: dict[str, bool]) -> list[str]:
    cmd = COMMON + [
        "--output-dir",
        str(OUT_ROOT / name),
        "--resume",
        str(BASE_RESUME),
    ]
    if name == "13_base":
        cmd.extend(["--no-residual-pick-head", "--no-residual-det-head"])
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

    p = argparse.ArgumentParser(description="Run13: anchors base + run11 ablations")
    p.add_argument("--only", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    state = load_state()
    runs = RUNS
    if args.only:
        runs = [r for r in RUNS if r[0] == args.only]
        if not runs:
            raise SystemExit(f"Unknown run: {args.only}")

    for name, flags in runs:
        if name in state.get("completed", []):
            print(f"[run13] skip completed {name}", flush=True)
            continue
        cmd = build_cmd(name, flags)
        print(f"[run13] >>> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"[run13] FAILED {name} code={proc.returncode}", flush=True)
            raise SystemExit(proc.returncode)

        metrics_path = OUT_ROOT / name / "test_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            state.setdefault("results", []).append(
                {
                    "name": name,
                    "flags": flags,
                    "det_f1": metrics.get("det_f1"),
                    "p_f1": metrics.get("p_f1"),
                    "s_f1": metrics.get("s_f1"),
                    "ps_sum": metrics.get("p_f1", 0) + metrics.get("s_f1", 0),
                }
            )
        state.setdefault("completed", []).append(name)
        save_state(state)

    print("[run13] done", flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
