#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D: dual-path proof — STEAD geo head + synthetic macro baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D dual-path proof + interpret")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pick-checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--stead-head", default="outputs/zhizi_inversion_mixed_geo/best_physics_head.pt")
    p.add_argument("--synth-head", default="outputs/zhizi_inversion_bridge_macro/best_physics_head.pt")
    p.add_argument("--proof-output", default="outputs/proof_suite_dual_path")
    p.add_argument("--interpret-output", default="outputs/interpret_suite_dual_path")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"[phase-d] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    py = sys.executable
    (ROOT / "outputs" / "phase_d").mkdir(parents=True, exist_ok=True)
    report: dict = {
        "phase": "D_dual_path",
        "pick_checkpoint": args.pick_checkpoint,
        "stead_head": args.stead_head,
        "synth_head": args.synth_head,
    }

    run(
        [
            py,
            str(ROOT / "run_proof_suite.py"),
            "--checkpoint",
            args.pick_checkpoint,
            "--dual-path",
            "--physics-head-stead",
            args.stead_head,
            "--physics-head-synth",
            args.synth_head,
            "--output-dir",
            args.proof_output,
            "--device",
            args.device,
        ]
    )
    proof_json = Path(args.proof_output) / "proof_report.json"
    if proof_json.exists():
        report["proof"] = json.loads(proof_json.read_text())

    run(
        [
            py,
            str(ROOT / "run_interpret_suite.py"),
            "--checkpoint",
            args.pick_checkpoint,
            "--physics-head",
            args.stead_head,
            "--output-dir",
            args.interpret_output,
            "--device",
            args.device,
        ]
    )

    out = ROOT / "outputs" / "phase_d_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report.get("proof", report), indent=2))
    print(f"[phase-d] -> {out}")


if __name__ == "__main__":
    main()
