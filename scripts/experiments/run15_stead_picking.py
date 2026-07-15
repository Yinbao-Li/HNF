#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run15: refine P/S toward 0.95+ from 14_main, keep det>=0.99.

Strategy:
  - Resume 14_main/best.pt (same 48x4 pick heads, no anchors)
  - Freeze shared backbone + det head for all epochs
  - Train only P/S branches + pick heads
  - Sharper labels, stronger pick loss, extra P weight
  - det_guard checkpoint selection
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "run15"
STATE_PATH = OUT_ROOT / "state.json"
BASE_RESUME = ROOT / "outputs" / "run14" / "14_main" / "best.pt"

# Freeze through epoch 999 => entire run
FREEZE_ALL = "999"

COMMON = [
    sys.executable,
    str(ROOT / "train_stead_picking.py"),
    "--seq-len",
    "800",
    "--batch-size",
    "24",
    "--grad-accum-steps",
    "2",
    "--num-workers",
    "2",
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
    "--pick-head-hidden",
    "48",
    "--pick-head-layers",
    "4",
    "--pick-head-kernel",
    "7",
    "--score-mode",
    "det_guard",
    "--det-score-floor",
    "0.985",
    "--post-process-p-before-s",
    "--no-residual-det-head",
    "--freeze-backbone-epochs",
    FREEZE_ALL,
    "--freeze-det-epochs",
    FREEZE_ALL,
]

RUNS = [
    (
        "15_pick_refine",
        {
            "resume": BASE_RESUME,
            "epochs": 6,
            "lr": "1.5e-4",
            "label_sigma_sec": "0.30",
            "pick_pos_weight": "35",
            "pick_loss_weight": "4.0",
            "p_pick_loss_weight": "1.8",
            "s_pick_loss_weight": "1.0",
            "det_loss_weight": "1.5",
            "ps_order_loss_weight": "0.25",
        },
    ),
]


def build_cmd(name: str, cfg: dict) -> list[str]:
    return COMMON + [
        "--output-dir",
        str(OUT_ROOT / name),
        "--resume",
        str(cfg["resume"]),
        "--epochs",
        str(cfg["epochs"]),
        "--lr",
        str(cfg["lr"]),
        "--label-sigma-sec",
        str(cfg["label_sigma_sec"]),
        "--pick-pos-weight",
        str(cfg["pick_pos_weight"]),
        "--pick-loss-weight",
        str(cfg["pick_loss_weight"]),
        "--p-pick-loss-weight",
        str(cfg["p_pick_loss_weight"]),
        "--s-pick-loss-weight",
        str(cfg["s_pick_loss_weight"]),
        "--det-loss-weight",
        str(cfg["det_loss_weight"]),
        "--ps-order-loss-weight",
        str(cfg["ps_order_loss_weight"]),
    ]


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"completed": [], "results": []}


def save_state(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run15: branch-only P/S refine toward 0.95+")
    p.add_argument("--only", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    state = load_state()
    runs = RUNS
    if args.only:
        runs = [r for r in RUNS if r[0] == args.only]
        if not runs:
            raise SystemExit(f"Unknown run: {args.only}")

    for name, cfg in runs:
        if name in state.get("completed", []):
            print(f"[run15] skip completed {name}", flush=True)
            continue
        resume_path = Path(cfg["resume"])
        if not resume_path.is_file():
            print(f"[run15] missing resume for {name}: {resume_path}", flush=True)
            raise SystemExit(1)
        cmd = build_cmd(name, cfg)
        print(f"[run15] >>> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"[run15] FAILED {name} code={proc.returncode}", flush=True)
            raise SystemExit(proc.returncode)

        metrics_path = OUT_ROOT / name / "test_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            state.setdefault("results", []).append(
                {
                    "name": name,
                    "det_f1": metrics.get("det_f1"),
                    "p_f1": metrics.get("p_f1"),
                    "s_f1": metrics.get("s_f1"),
                    "ps_sum": metrics.get("p_f1", 0) + metrics.get("s_f1", 0),
                }
            )
        state.setdefault("completed", []).append(name)
        save_state(state)

    print("[run15] done", flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
