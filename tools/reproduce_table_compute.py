#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-reproduce manuscript Table V (compute) parameter count for c2f / tgrs-v14.

  python tools/reproduce_table_compute.py \
    --checkpoint outputs/stead_c2f_imp_RSP/last.pt \
    --expected paper/tgrs/expected/compute.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--expected",
        type=Path,
        default=_REPO / "paper/tgrs/expected/compute.json",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        print(f"FAIL: checkpoint missing: {args.checkpoint}", file=sys.stderr)
        print("Download the tgrs-v14 Release asset to this path (see REPRODUCE.md).", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    model, cfg, _ckpt = load_model(args.checkpoint, device)
    n_params = int(sum(q.numel() for q in model.parameters()))

    expected = json.loads(args.expected.read_text())
    exp_params = int(expected["results"]["ours"]["params"])
    params_ok = n_params == exp_params == 600916

    tex_line = f"Ours & {n_params / 1e6:.3f} & ..."
    report = {
        "table": "Table V (compute) — Params column",
        "manuscript_tex": "paper/tgrs/tables/compute.tex / expected/compute.tex",
        "checkpoint": str(args.checkpoint),
        "fusion": cfg.get("fusion"),
        "fine_len": cfg.get("fine_len"),
        "n_params": n_params,
        "expected_params": exp_params,
        "params_ok": params_ok,
        "tex_params_cell": f"{n_params / 1e6:.3f}",
        "note": "Throughput/MiB columns require CUDA and are hardware-dependent; not asserted here.",
    }
    print(json.dumps(report, indent=2))
    print(f"[reproduce] {tex_line} params_ok={params_ok}")
    return 0 if params_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
