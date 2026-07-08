#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run14: keep det>=0.99 while pushing P/S F1 toward 0.95+.

Strategy (vs run13):
  - No num_anchors (full temporal resolution, preserve det)
  - Same 01_seq800 backbone (no multi_scale — it hurt det in run13)
  - Deeper dilated pick heads + envelope residual
  - Stronger pick loss / P emphasis / P-S order constraint
  - det_guard checkpoint scoring + freeze backbone/det early
  - 12-epoch fine-tune, then optional 6-epoch sharp pass
  - (updated) 8-epoch main + 4-epoch sharp
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "run14"
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
    "--reset-best-score",
    "--no-residual-det-head",
]

RUNS = [
    (
        "14_main",
        {
            "resume": BASE_RESUME,
            "epochs": 8,
            "lr": "3e-4",
            "label_sigma_sec": "0.35",
            "pick_pos_weight": "30",
            "pick_loss_weight": "3.5",
            "p_pick_loss_weight": "1.5",
            "s_pick_loss_weight": "1.0",
            "det_loss_weight": "1.5",
            "ps_order_loss_weight": "0.2",
            "freeze_backbone_epochs": "2",
            "freeze_det_epochs": "8",
            "continue_train": False,
        },
    ),
    (
        "14_sharp",
        {
            "resume": OUT_ROOT / "14_main" / "best.pt",
            "epochs": 4,
            "lr": "1e-4",
            "label_sigma_sec": "0.32",
            "pick_pos_weight": "35",
            "pick_loss_weight": "4.0",
            "p_pick_loss_weight": "1.6",
            "s_pick_loss_weight": "1.0",
            "det_loss_weight": "1.5",
            "ps_order_loss_weight": "0.25",
            "freeze_backbone_epochs": "0",
            "freeze_det_epochs": "4",
            "continue_train": False,
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
        "--freeze-backbone-epochs",
        str(cfg["freeze_backbone_epochs"]),
        "--freeze-det-epochs",
        str(cfg["freeze_det_epochs"]),
    ]
    if cfg.get("continue_train"):
        cmd.append("--continue")
    return cmd


def maybe_continue_main(name: str, cfg: dict) -> dict:
    """If 14_main stopped mid-run, resume from last.pt for remaining epochs."""
    if name != "14_main" or cfg.get("continue_train"):
        return cfg
    last_ckpt = OUT_ROOT / "14_main" / "last.pt"
    if not last_ckpt.is_file():
        return cfg
    import torch

    done_epoch = int(torch.load(last_ckpt, map_location="cpu", weights_only=False).get("epoch", 0))
    target_epochs = int(cfg["epochs"])
    if done_epoch <= 0 or done_epoch >= target_epochs:
        return cfg
    updated = dict(cfg)
    updated["resume"] = last_ckpt
    updated["continue_train"] = True
    print(
        f"[run14] continue {name} from epoch {done_epoch + 1}/{target_epochs}",
        flush=True,
    )
    return updated


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"completed": [], "results": []}


def save_state(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run14: det-safe P/S boost toward 0.95+")
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
            print(f"[run14] skip completed {name}", flush=True)
            continue
        cfg = maybe_continue_main(name, cfg)
        resume_path = Path(cfg["resume"])
        if not resume_path.is_file():
            print(f"[run14] missing resume for {name}: {resume_path}", flush=True)
            raise SystemExit(1)
        cmd = build_cmd(name, cfg)
        print(f"[run14] >>> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"[run14] FAILED {name} code={proc.returncode}", flush=True)
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

    print("[run14] done", flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
