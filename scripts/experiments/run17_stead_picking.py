#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run17: Huygens three-step noise cancellation + joint picking fine-tune.

Phase A (17_noise_warmup): train noise-cancel branch only (3 ep)
Phase B (17_joint): joint denoise + det/P/S from warmup checkpoint (6 ep)

Resume base: 16_det_onset/best.pt (best current det/P/S before run16 B regression)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "run17"
STATE_PATH = OUT_ROOT / "state.json"
BASE_RESUME = ROOT / "outputs" / "run16" / "16_det_onset" / "best.pt"

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
    "--noise-cancel",
    "--noise-source-dim",
    "16",
    "--reset-best-score",
]

RUNS = [
    (
        "17_noise_warmup",
        {
            "resume": BASE_RESUME,
            "epochs": 3,
            "lr": "2e-4",
            "noise_cancel_weight": "1.0",
            "pick_loss_weight": "0.5",
            "det_loss_weight": "0.5",
            "freeze_all_but_noise_epochs": "3",
            "freeze_backbone_epochs": "0",
            "freeze_det_epochs": "0",
            "freeze_all_but_det_epochs": "0",
        },
    ),
    (
        "17_joint",
        {
            "resume": OUT_ROOT / "17_noise_warmup" / "best.pt",
            "epochs": 6,
            "lr": "1e-4",
            "batch_size": "16",
            "grad_accum_steps": "3",
            "num_workers": "1",
            "label_sigma_sec": "0.35",
            "pick_pos_weight": "30",
            "pick_loss_weight": "3.0",
            "p_pick_loss_weight": "1.4",
            "s_pick_loss_weight": "1.2",
            "det_loss_weight": "1.5",
            "det_event_weight": "2.0",
            "ps_order_loss_weight": "0.15",
            "noise_cancel_weight": "0.4",
            "freeze_all_but_noise_epochs": "0",
            "freeze_backbone_epochs": "1",
            "freeze_det_epochs": "0",
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
        "--noise-cancel-weight",
        str(cfg["noise_cancel_weight"]),
        "--pick-loss-weight",
        str(cfg["pick_loss_weight"]),
        "--det-loss-weight",
        str(cfg["det_loss_weight"]),
        "--freeze-all-but-noise-epochs",
        str(cfg["freeze_all_but_noise_epochs"]),
        "--freeze-backbone-epochs",
        str(cfg["freeze_backbone_epochs"]),
        "--freeze-det-epochs",
        str(cfg["freeze_det_epochs"]),
        "--freeze-all-but-det-epochs",
        str(cfg["freeze_all_but_det_epochs"]),
    ]
    if "label_sigma_sec" in cfg:
        cmd.extend(["--label-sigma-sec", str(cfg["label_sigma_sec"])])
    if "pick_pos_weight" in cfg:
        cmd.extend(["--pick-pos-weight", str(cfg["pick_pos_weight"])])
    if "p_pick_loss_weight" in cfg:
        cmd.extend(["--p-pick-loss-weight", str(cfg["p_pick_loss_weight"])])
    if "s_pick_loss_weight" in cfg:
        cmd.extend(["--s-pick-loss-weight", str(cfg["s_pick_loss_weight"])])
    if "det_event_weight" in cfg:
        cmd.extend(["--det-event-weight", str(cfg["det_event_weight"])])
    if "ps_order_loss_weight" in cfg:
        cmd.extend(["--ps-order-loss-weight", str(cfg["ps_order_loss_weight"])])
    if cfg.get("batch_size"):
        idx = cmd.index("--batch-size")
        cmd[idx + 1] = str(cfg["batch_size"])
    if cfg.get("grad_accum_steps"):
        idx = cmd.index("--grad-accum-steps")
        cmd[idx + 1] = str(cfg["grad_accum_steps"])
    if cfg.get("num_workers") is not None:
        idx = cmd.index("--num-workers")
        cmd[idx + 1] = str(cfg["num_workers"])
    if cfg.get("continue_train"):
        cmd.append("--continue")
    return cmd


def maybe_continue_joint(name: str, cfg: dict) -> dict:
    """Resume 17_joint from last.pt after OOM or interruption."""
    if name != "17_joint":
        return cfg
    last_ckpt = OUT_ROOT / "17_joint" / "last.pt"
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
        f"[run17] continue {name} from epoch {done_epoch + 1}/{target_epochs}",
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

    p = argparse.ArgumentParser(description="Run17: HNF noise cancel + joint picking")
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
            print(f"[run17] skip completed {name}", flush=True)
            continue
        resume_path = Path(cfg["resume"])
        if not resume_path.is_file():
            print(f"[run17] missing resume for {name}: {resume_path}", flush=True)
            raise SystemExit(1)
        cfg = maybe_continue_joint(name, cfg)
        cmd = build_cmd(name, cfg)
        print(f"[run17] >>> {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        env = dict(**__import__("os").environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if proc.returncode != 0:
            print(f"[run17] FAILED {name} code={proc.returncode}", flush=True)
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

    print("[run17] done", flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
