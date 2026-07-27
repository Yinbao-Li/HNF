#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D/4D spatial HNF experiment suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
PY310 = "/usr/bin/python3"


def run(cmd, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[3d4d]", " ".join(cmd), flush=True)
    with log.open("w") as f:
        rc = subprocess.call(cmd, cwd=_REPO, stdout=f, stderr=subprocess.STDOUT)
    if rc:
        raise RuntimeError(log)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()
    root = Path("outputs/fluid/spatial_3d4d_suite")
    root.mkdir(parents=True, exist_ok=True)
    stable = ["--epochs", str(args.epochs), "--device", args.device, "--batch-size", "4",
              "--kernel-size", "5", "--curl-weight", "0.01", "--curl-warmup-epochs", "8"]
    board = {}

    jobs = [
        ("synth3d_all", [PY, "tools/train_fluid_spatial3d4d.py", "--mode", "3d", "--output-dir", str(root / "synth3d_all"), *stable]),
        ("synth3d_vortex", [PY, "tools/train_fluid_spatial3d4d.py", "--mode", "3d", "--families", "vortex_tube", "--output-dir", str(root / "synth3d_vortex"), *stable]),
        ("synth4d_all", [PY, "tools/train_fluid_spatial3d4d.py", "--mode", "4d", "--output-dir", str(root / "synth4d_all"), "--d", "8", "--h", "12", "--w", "12", "--t-steps", "4", *stable]),
    ]
    for name, cmd in jobs:
        run(cmd, root / f"{name}.log")
        board[name] = json.loads((Path(cmd[cmd.index("--output-dir") + 1]) / "summary.json").read_text())

    vol_cache = _REPO / "external_data/raclette_cache/gt_volumes.npz"
    slice_cache = _REPO / "external_data/raclette_cache/gt_slices.npz"
    if not vol_cache.is_file() and slice_cache.is_file():
        run([PY310, "tools/preprocess_raclette_volumes.py"], root / "raclette_vol_preprocess.log")
    if vol_cache.is_file():
        run([PY, "tools/train_raclette_spatial3d.py", "--output-dir", str(root / "raclette3d"), "--epochs", str(args.epochs), "--device", args.device], root / "raclette3d.log")
        ckpt = root / "raclette3d" / "best.pt"
        if ckpt.is_file():
            import torch
            board["raclette3d"] = torch.load(ckpt, map_location="cpu", weights_only=False).get("val_metrics", {})

    lines = ["# 3D/4D Spatial HNF Suite", "", "| Run | test vel_rel | notes |", "|-----|-------------:|-------|"]
    for name in board:
        tb = board[name].get("test_best", board[name])
        vr = tb.get("vel_rel", tb.get("vel_rel_inside_vessel", float("nan")))
        extra = ""
        if "vel_rel_inside_vessel" in tb:
            extra = f" inside={tb['vel_rel_inside_vessel']:.3f}"
        fam = ", ".join(f"{k.replace('vel_rel_','')}={v:.3f}" for k, v in tb.items() if k.startswith("vel_rel_"))
        lines.append(f"| {name} | {vr:.4f} | {fam or extra or '—'} |")
    (root / "SUITE.md").write_text("\n".join(lines) + "\n")
    (root / "board.json").write_text(json.dumps(board, indent=2))
    print(f"[3d4d] → {root / 'SUITE.md'}", flush=True)


if __name__ == "__main__":
    main()
