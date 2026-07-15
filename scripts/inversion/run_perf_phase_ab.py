#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase A/B performance iteration:
  A) run21 S-only picking refine (optional)
  B) STEAD real-waveform macro head + geo conditioning
  C) proof + interpret suites vs frozen baselines
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A/B: run21 pick + STEAD macro train + eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-run21", action="store_true")
    p.add_argument("--skip-macro-train", action="store_true")
    p.add_argument("--skip-proof", action="store_true")
    p.add_argument("--skip-interpret", action="store_true")
    p.add_argument("--pick-checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--macro-output", default="outputs/zhizi_inversion_stead_macro")
    p.add_argument("--macro-epochs", type=int, default=10)
    p.add_argument("--stead-max-train", type=int, default=400)
    p.add_argument("--stead-max-val", type=int, default=80)
    p.add_argument("--proof-output", default="outputs/proof_suite_phase_ab")
    p.add_argument("--interpret-output", default="outputs/interpret_suite_phase_ab")
    return p.parse_args()


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"[phase-ab] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    args = parse_args()
    py = sys.executable
    report: dict = {"pick_checkpoint": args.pick_checkpoint}

    run21_best = ROOT / "outputs" / "run21" / "21_s_only_refine" / "best.pt"
    if not args.skip_run21:
        run([py, str(ROOT / "scripts" / "experiments" / "run21_stead_picking.py")])
        if run21_best.exists():
            report["run21_checkpoint"] = str(run21_best)
            args.pick_checkpoint = str(run21_best)
    elif run21_best.exists():
        report["run21_checkpoint"] = str(run21_best)
        args.pick_checkpoint = str(run21_best)

    macro_head = Path(args.macro_output) / "best_physics_head.pt"
    if not args.skip_macro_train:
        run(
            [
                py,
                str(ROOT / "tools" / "train_zhizi_inversion.py"),
                "--checkpoint",
                args.pick_checkpoint,
                "--output-dir",
                args.macro_output,
                "--dataset",
                "stead",
                "--geo-condition",
                "--head-mode",
                "macro",
                "--epochs",
                str(args.macro_epochs),
                "--stead-max-train",
                str(args.stead_max_train),
                "--stead-max-val",
                str(args.stead_max_val),
                "--waveform-weight",
                "0.8",
                "--unrolled-weight",
                "0.4",
                "--unrolled-steps",
                "3",
                "--lr",
                "2e-3",
                "--resume-physics-head",
                "outputs/zhizi_inversion_bridge_macro/best_physics_head.pt",
                "--device",
                args.device,
            ]
        )
    report["macro_head"] = str(macro_head)

    if not args.skip_proof and macro_head.exists():
        run(
            [
                py,
                str(ROOT / "scripts" / "inversion" / "run_proof_suite.py"),
                "--checkpoint",
                args.pick_checkpoint,
                "--physics-head",
                str(macro_head),
                "--head-mode",
                "macro",
                "--output-dir",
                args.proof_output,
                "--device",
                args.device,
            ]
        )
        proof_json = Path(args.proof_output) / "proof_report.json"
        if proof_json.exists():
            report["proof"] = json.loads(proof_json.read_text())

    if not args.skip_interpret and macro_head.exists():
        run(
            [
                py,
                str(ROOT / "scripts" / "interpret" / "run_interpret_suite.py"),
                "--checkpoint",
                args.pick_checkpoint,
                "--physics-head",
                str(macro_head),
                "--output-dir",
                args.interpret_output,
                "--device",
                args.device,
            ]
        )

    out = ROOT / "outputs" / "phase_ab_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[phase-ab] -> {out}")


if __name__ == "__main__":
    main()
