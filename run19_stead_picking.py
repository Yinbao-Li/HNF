#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run19: det uses denoised path; P/S uses raw waveform plus denoise cues.

Motivation:
  - run17/18 showed denoise branch helps det
  - direct denoised input hurts P/S via wrong_peak inflation
  - this run keeps raw waveform for P/S while injecting noise cues as guidance
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "run19"
STATE_PATH = OUT_ROOT / "state.json"
BASE_RESUME = ROOT / "outputs" / "run17" / "17_noise_warmup" / "best.pt"

COMMON = [
    sys.executable,
    str(ROOT / "train_stead_picking.py"),
    "--seq-len", "800",
    "--batch-size", "16",
    "--grad-accum-steps", "3",
    "--num-workers", "1",
    "--embed-dim", "64",
    "--num-shared-layers", "2",
    "--num-branch-layers", "2",
    "--local-window-sec", "15.0",
    "--seed", "42",
    "--pick-head-hidden", "48",
    "--pick-head-layers", "4",
    "--pick-head-kernel", "7",
    "--score-mode", "det_guard",
    "--det-score-floor", "0.988",
    "--post-process-p-before-s",
    "--no-residual-det-head",
    "--enhanced-det-head",
    "--noise-cancel",
    "--noise-source-dim", "16",
    "--noise-det-pick-split",
    "--noise-pick-cues",
    "--reset-best-score",
]

RUNS = [
    (
        "19_detpick_split",
        {
            "resume": BASE_RESUME,
            "epochs": 6,
            "lr": "8e-5",
            "label_sigma_sec": "0.35",
            "pick_pos_weight": "28",
            "pick_loss_weight": "2.8",
            "p_pick_loss_weight": "1.3",
            "s_pick_loss_weight": "1.45",
            "det_loss_weight": "1.2",
            "det_event_weight": "2.0",
            "ps_order_loss_weight": "0.12",
            "noise_cancel_weight": "0.08",
            "freeze_all_but_noise_epochs": "0",
            "freeze_all_but_pick_epochs": "0",
            "freeze_backbone_epochs": "1",
            "freeze_det_epochs": "0",
            "freeze_all_but_det_epochs": "0",
        },
    ),
]


def build_cmd(name: str, cfg: dict) -> list[str]:
    cmd = COMMON + [
        "--output-dir", str(OUT_ROOT / name),
        "--resume", str(cfg["resume"]),
        "--epochs", str(cfg["epochs"]),
        "--lr", str(cfg["lr"]),
        "--noise-cancel-weight", str(cfg["noise_cancel_weight"]),
        "--pick-loss-weight", str(cfg["pick_loss_weight"]),
        "--det-loss-weight", str(cfg["det_loss_weight"]),
        "--freeze-all-but-noise-epochs", str(cfg["freeze_all_but_noise_epochs"]),
        "--freeze-all-but-pick-epochs", str(cfg["freeze_all_but_pick_epochs"]),
        "--freeze-backbone-epochs", str(cfg["freeze_backbone_epochs"]),
        "--freeze-det-epochs", str(cfg["freeze_det_epochs"]),
        "--freeze-all-but-det-epochs", str(cfg["freeze_all_but_det_epochs"]),
    ]
    for key, flag in [
        ("label_sigma_sec", "--label-sigma-sec"),
        ("pick_pos_weight", "--pick-pos-weight"),
        ("p_pick_loss_weight", "--p-pick-loss-weight"),
        ("s_pick_loss_weight", "--s-pick-loss-weight"),
        ("det_event_weight", "--det-event-weight"),
        ("ps_order_loss_weight", "--ps-order-loss-weight"),
    ]:
        if key in cfg:
            cmd.extend([flag, str(cfg[key])])
    if cfg.get("continue_train"):
        cmd.append("--continue")
    return cmd


def maybe_continue(name: str, cfg: dict) -> dict:
    last_ckpt = OUT_ROOT / name / "last.pt"
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
    print(f"[run19] continue {name} from epoch {done_epoch + 1}/{target_epochs}", flush=True)
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

    p = argparse.ArgumentParser(description="Run19: denoise-for-det, cues-for-pick")
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
            print(f"[run19] skip completed {name}", flush=True)
            continue
        cfg = maybe_continue(name, cfg)
        resume_path = Path(cfg["resume"])
        if not resume_path.is_file():
            print(f"[run19] missing resume for {name}: {resume_path}", flush=True)
            raise SystemExit(1)
        cmd = build_cmd(name, cfg)
        print(f"[run19] >>> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if proc.returncode != 0:
            print(f"[run19] FAILED {name} code={proc.returncode}", flush=True)
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
                    "n_params": metrics.get("n_params"),
                }
            )
        state.setdefault("completed", []).append(name)
        save_state(state)

    print("[run19] done", flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
