#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation: raster Stage-0 vs spatial HNF (+/- rotation) on vortex / all families."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ablation] {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=_REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): see {log_path}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fluid spatial HNF ablation suite")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--families", default="vortex", help="vortex | all (empty) | comma list")
    p.add_argument("--skip-raster", action="store_true")
    p.add_argument("--output-root", default="outputs/fluid/spatial_ablation")
    args = p.parse_args()

    fam_arg = "" if args.families in {"", "all"} else f"--families {args.families}"
    fam_tag = args.families.replace(",", "_") if args.families else "all"
    root = Path(args.output_root) / fam_tag
    root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    common = f"--epochs {args.epochs} --device {args.device} --n-train 1024 --n-val 128 --n-test 128"
    if fam_arg:
        common += f" {fam_arg}"

    jobs: list[tuple[str, list[str], Path]] = []

    if not args.skip_raster:
        jobs.append(
            (
                "raster",
                [py, "tools/train_fluid.py", "--output-dir", str(root / "raster"), *common.split()],
                root / "raster.log",
            )
        )

    jobs.extend(
        [
            (
                "spatial_rot",
                [
                    py,
                    "tools/train_fluid_spatial.py",
                    "--output-dir",
                    str(root / "spatial_rot"),
                    "--curl-weight",
                    "0.01",
                    "--curl-warmup-epochs",
                    "10",
                    "--kernel-size",
                    "9",
                    *common.split(),
                ],
                root / "spatial_rot.log",
            ),
            (
                "spatial_no_rot",
                [
                    py,
                    "tools/train_fluid_spatial.py",
                    "--output-dir",
                    str(root / "spatial_no_rot"),
                    "--no-rotation",
                    *common.split(),
                ],
                root / "spatial_no_rot.log",
            ),
        ]
    )

    board: dict[str, dict] = {}
    for name, cmd, log in jobs:
        run(cmd, log)
        summary_path = Path(cmd[cmd.index("--output-dir") + 1]) / "summary.json"
        if not summary_path.exists():
            hist_path = Path(cmd[cmd.index("--output-dir") + 1]) / "history.json"
            if hist_path.exists():
                with hist_path.open(encoding="utf-8") as f:
                    board[name] = json.load(f)
            continue
        with summary_path.open(encoding="utf-8") as f:
            board[name] = json.load(f)

    md_lines = [
        f"# Fluid spatial ablation ({fam_tag})",
        "",
        f"- epochs={args.epochs}, keep=10%, families={fam_tag}",
        "",
        "| Model | test vel_rel | val vel_rel | notes |",
        "|-------|-------------:|------------:|-------|",
    ]
    for name in [j[0] for j in jobs]:
        rec = board.get(name, {})
        test = rec.get("test_best") or rec.get("best_val") or {}
        val = rec.get("best_val") or {}
        md_lines.append(
            f"| {name} | {test.get('vel_rel', float('nan')):.4f} | "
            f"{val.get('vel_rel', float('nan')):.4f} | spatial HNF ablation |"
        )
    report = root / "ABLATION.md"
    report.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with (root / "ablation_board.json").open("w", encoding="utf-8") as f:
        json.dump(board, f, indent=2)
    print(f"[ablation] report → {report}", flush=True)


if __name__ == "__main__":
    main()
