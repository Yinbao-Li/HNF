#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run16: analysis-driven two-phase push toward det 0.995+ / P/S 0.95+.

Phase A (16_det_onset): onset-aware det + raw HF branch, train det only
Phase B (16_pick_refine): freeze shared+det, refine P/S branches + pick heads
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "run16"
STATE_PATH = OUT_ROOT / "state.json"
BASE_RESUME = ROOT / "outputs" / "run14" / "14_main" / "best.pt"

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
    "0.988",
    "--post-process-p-before-s",
    "--no-residual-det-head",
    "--enhanced-det-head",
]

RUNS = [
    (
        "16_det_onset",
        {
            "resume": BASE_RESUME,
            "epochs": 4,
            "lr": "2e-4",
            "det_loss_weight": "2.5",
            "det_event_weight": "3.0",
            "pick_loss_weight": "1.0",
            "freeze_all_but_det_epochs": "4",
            "freeze_backbone_epochs": "0",
            "freeze_det_epochs": "0",
        },
    ),
    (
        "16_pick_refine",
        {
            "resume": OUT_ROOT / "16_det_onset" / "best.pt",
            "epochs": 6,
            "lr": "1.2e-4",
            "label_sigma_sec": "0.32",
            "pick_pos_weight": "35",
            "pick_loss_weight": "4.0",
            "p_pick_loss_weight": "1.5",
            "s_pick_loss_weight": "1.3",
            "det_loss_weight": "1.5",
            "det_event_weight": "2.0",
            "ps_order_loss_weight": "0.2",
            "freeze_backbone_epochs": "999",
            "freeze_det_epochs": "999",
            "freeze_all_but_det_epochs": "0",
        },
    ),
]


def build_cmd(name: str, cfg: dict) -> list[str]:
    cmd = COMMON + [
        "--output-dir",
        str(OUT_ROOT / name),
        "--resume",
        str(cfg["resume"]),
        "--epochs",
        str(cfg["epochs"]),
        "--lr",
        str(cfg["lr"]),
        "--det-loss-weight",
        str(cfg["det_loss_weight"]),
        "--det-event-weight",
        str(cfg["det_event_weight"]),
        "--freeze-backbone-epochs",
        str(cfg["freeze_backbone_epochs"]),
        "--freeze-det-epochs",
        str(cfg["freeze_det_epochs"]),
        "--freeze-all-but-det-epochs",
        str(cfg["freeze_all_but_det_epochs"]),
        "--pick-loss-weight",
        str(cfg["pick_loss_weight"]),
    ]
    if "label_sigma_sec" in cfg:
        cmd.extend(["--label-sigma-sec", str(cfg["label_sigma_sec"])])
    if "pick_pos_weight" in cfg:
        cmd.extend(["--pick-pos-weight", str(cfg["pick_pos_weight"])])
    if "p_pick_loss_weight" in cfg:
        cmd.extend(["--p-pick-loss-weight", str(cfg["p_pick_loss_weight"])])
    if "s_pick_loss_weight" in cfg:
        cmd.extend(["--s-pick-loss-weight", str(cfg["s_pick_loss_weight"])])
    if "ps_order_loss_weight" in cfg:
        cmd.extend(["--ps-order-loss-weight", str(cfg["ps_order_loss_weight"])])
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

    p = argparse.ArgumentParser(description="Run16: onset det + P/S refine")
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
            print(f"[run16] skip completed {name}", flush=True)
            continue
        resume_path = Path(cfg["resume"])
        if not resume_path.is_file():
            print(f"[run16] missing resume for {name}: {resume_path}", flush=True)
            raise SystemExit(1)
        cmd = build_cmd(name, cfg)
        print(f"[run16] >>> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"[run16] FAILED {name} code={proc.returncode}", flush=True)
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

    print("[run16] done", flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
