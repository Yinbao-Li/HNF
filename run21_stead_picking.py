#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run21: S-focused pick refine from frozen run20 backbone + det.

Only P/S branch heads train (wrong-peak emphasis on S).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "run21"
STATE_PATH = OUT_ROOT / "state.json"
BASE_RESUME = ROOT / "outputs" / "run20" / "20_wrongpeak_sharp" / "best.pt"

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
        "21_s_only_refine",
        {
            "resume": BASE_RESUME,
            "epochs": "4",
            "lr": "2e-5",
            "label_sigma_sec": "0.35",
            "pick_pos_weight": "28",
            "pick_loss_weight": "2.6",
            "p_pick_loss_weight": "1.0",
            "s_pick_loss_weight": "2.0",
            "det_loss_weight": "0.0",
            "det_event_weight": "0.0",
            "ps_order_loss_weight": "0.15",
            "wrong_peak_loss_weight": "0.22",
            "wrong_peak_radius_sec": "0.45",
            "wrong_peak_margin": "0.28",
            "s_wrong_peak_scale": "1.6",
            "noise_cancel_weight": "0.0",
            "freeze_all_but_noise_epochs": "0",
            "freeze_all_but_pick_epochs": "4",
            "freeze_backbone_epochs": "99",
            "freeze_det_epochs": "99",
            "freeze_all_but_det_epochs": "0",
        },
    ),
]


def build_cmd(run_name: str, overrides: dict) -> list[str]:
    out_dir = OUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = COMMON + ["--output-dir", str(out_dir)]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def main() -> None:
    if not BASE_RESUME.exists():
        raise FileNotFoundError(f"run20 checkpoint missing: {BASE_RESUME}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    state: dict = {"runs": []}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())

    for run_name, overrides in RUNS:
        out_dir = OUT_ROOT / run_name
        best = out_dir / "best.pt"
        if best.exists():
            print(f"[run21] skip existing {best}", flush=True)
            continue
        cmd = build_cmd(run_name, overrides)
        print(f"[run21] {run_name}: {' '.join(cmd)}", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
        state["runs"].append({"name": run_name, "best": str(best)})
        STATE_PATH.write_text(json.dumps(state, indent=2))

    print(f"[run21] done -> {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
