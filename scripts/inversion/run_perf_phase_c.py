#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase C: mixed synthetic+STEAD macro head with geo-only fine-tune.

Preserves synth Route A2 while adapting distance/depth conditioning on STEAD.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C: mixed geo-only macro + eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-proof", action="store_true")
    p.add_argument("--skip-interpret", action="store_true")
    p.add_argument("--pick-checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--macro-output", default="outputs/zhizi_inversion_mixed_geo")
    p.add_argument("--macro-epochs", type=int, default=16)
    p.add_argument("--proof-output", default="outputs/proof_suite_phase_c")
    p.add_argument("--interpret-output", default="outputs/interpret_suite_phase_c")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"[phase-c] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    py = sys.executable
    macro_head = Path(args.macro_output) / "best_physics_head.pt"
    report: dict = {"pick_checkpoint": args.pick_checkpoint, "phase": "C_mixed_geo_only"}

    if not args.skip_train:
        run(
            [
                py,
                str(ROOT / "train_zhizi_inversion.py"),
                "--checkpoint",
                args.pick_checkpoint,
                "--output-dir",
                args.macro_output,
                "--dataset",
                "mixed",
                "--geo-condition",
                "--train-geo-only",
                "--head-mode",
                "macro",
                "--epochs",
                str(args.macro_epochs),
                "--n-train",
                "96",
                "--n-val",
                "24",
                "--stead-max-train",
                "200",
                "--stead-max-val",
                "40",
                "--lr",
                "5e-4",
                "--anchor-weight",
                "0.05",
                "--vp-sup-weight",
                "0.12",
                "--stead-waveform-weight",
                "0.2",
                "--stead-unrolled-weight",
                "0.12",
                "--synth-unrolled-weight",
                "0.35",
                "--unrolled-steps",
                "3",
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
                str(ROOT / "run_proof_suite.py"),
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
                str(ROOT / "run_interpret_suite.py"),
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

    out = ROOT / "outputs" / "phase_c_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[phase-c] -> {out}")


if __name__ == "__main__":
    main()
