#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable spatial HNF suite: vortex 50ep → all families → RACLETTE spatial."""

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
    print(f"[suite] {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, cwd=_REPO, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"Failed ({rc}): {log_path}")


def load_summary(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Spatial HNF stable training suite")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs-vortex", type=int, default=50)
    p.add_argument("--epochs-all", type=int, default=50)
    p.add_argument("--epochs-raclette", type=int, default=40)
    p.add_argument("--skip-raclette", action="store_true")
    p.add_argument("--output-root", default="outputs/fluid/spatial_suite")
    args = p.parse_args()

    py = sys.executable
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    stable = [
        "--kernel-size", "9",
        "--obs-weight", "0.25",
        "--recon-weight", "1.0",
        "--curl-weight", "0.01",
        "--curl-warmup-epochs", "10",
        "--lr-warmup-epochs", "3",
        "--device", args.device,
    ]

    jobs = [
        (
            "vortex50",
            [
                py, "tools/train_fluid_spatial.py",
                "--output-dir", str(root / "vortex50"),
                "--epochs", str(args.epochs_vortex),
                "--families", "vortex",
                "--n-train", "2048", "--n-val", "256", "--n-test", "256",
                *stable,
            ],
            root / "vortex50.log",
        ),
        (
            "all50",
            [
                py, "tools/train_fluid_spatial.py",
                "--output-dir", str(root / "all50"),
                "--epochs", str(args.epochs_all),
                "--n-train", "2048", "--n-val", "256", "--n-test", "256",
                *stable,
            ],
            root / "all50.log",
        ),
    ]

    board: dict[str, dict] = {}
    for name, cmd, log in jobs:
        run(cmd, log)
        board[name] = load_summary(Path(cmd[cmd.index("--output-dir") + 1]) / "summary.json")

    if not args.skip_raclette:
        cache = _REPO / "external_data/raclette_cache/gt_slices.npz"
        if cache.is_file():
            raclette_out = root / "raclette_spatial"
            run(
                [
                    py, "tools/train_raclette_spatial.py",
                    "--output-dir", str(raclette_out),
                    "--epochs", str(args.epochs_raclette),
                    "--kernel-size", "9",
                    "--curl-weight", "0.005",
                    "--curl-warmup-epochs", "8",
                    "--device", args.device,
                ],
                root / "raclette_spatial.log",
            )
            run(
                [
                    py, "tools/eval_raclette_spatial.py",
                    "--checkpoint", str(raclette_out / "best.pt"),
                    "--output", str(raclette_out / "test_metrics.json"),
                    "--device", args.device,
                ],
                root / "raclette_spatial_eval.log",
            )
            board["raclette_spatial"] = json.loads((raclette_out / "test_metrics.json").read_text(encoding="utf-8"))
        else:
            print(f"[suite] skip RACLETTE — missing {cache}", flush=True)

    lines = [
        "# Spatial HNF stable suite",
        "",
        "## Synthetic (spatial+rot, k=9, mask-aware, curl warmup)",
        "",
        "| Run | test vel_rel | per-family |",
        "|-----|-------------:|------------|",
    ]
    for name in ("vortex50", "all50"):
        rec = board.get(name, {})
        test = rec.get("test_best", {})
        fam = ", ".join(f"{k.replace('vel_rel_', '')}={v:.3f}" for k, v in test.items() if k.startswith("vel_rel_"))
        lines.append(f"| {name} | {test.get('vel_rel', float('nan')):.4f} | {fam or '—'} |")

    if "raclette_spatial" in board:
        r = board["raclette_spatial"]
        lines.extend([
            "",
            "## RACLETTE spatial",
            "",
            f"- full vel_rel: **{r.get('vel_rel', float('nan')):.4f}**",
            f"- inside vessel: **{r.get('vel_rel_inside_vessel', float('nan')):.4f}** (raster baseline ~0.793)",
        ])

    report = root / "SUITE.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (root / "suite_board.json").open("w", encoding="utf-8") as f:
        json.dump(board, f, indent=2)
    print(f"[suite] report → {report}", flush=True)


if __name__ == "__main__":
    main()
