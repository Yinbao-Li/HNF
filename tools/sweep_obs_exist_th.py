#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep exist_th on OBS exist-gated checkpoints (holdout, score_absent=True).

Compares from-scratch vs gate-supervised ckpts without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.obs_matched_split import load_split_samples
from tools.train_obs_exist_gate import (
    build_model_from_ckpt,
    eval_full,
    sweep_exist_th,
)
from tools.train_obs_picking import filter_alive_channels, _load_obs_compare_module
from tools.train_stead_picking import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS exist_th holdout sweep")
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "outputs/run_obs_native/obs_4c_exist_fromscratch_30ep/best.pt",
            "outputs/run_obs_native/obs_4c_exist_gate_sup_8ep/best.pt",
        ],
    )
    p.add_argument("--names", nargs="+", default=["fromscratch_30ep", "gate_sup_8ep"])
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_exist_th_sweep",
    )
    p.add_argument("--pick-threshold", type=float, default=0.25)
    p.add_argument("--th-min", type=float, default=0.20)
    p.add_argument("--th-max", type=float, default=0.70)
    p.add_argument("--th-step", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = __import__("torch").device(args.device)
    obs_mod = _load_obs_compare_module()

    holdout, _, _ = load_split_samples(args.split_json, "holdout")
    names = list(args.names)
    while len(names) < len(args.checkpoints):
        names.append(Path(args.checkpoints[len(names)]).parent.name)

    thresholds = [
        round(float(x), 2)
        for x in np.arange(args.th_min, args.th_max + 1e-9, args.th_step)
    ]
    board: dict = {"thresholds": thresholds, "models": {}, "note": "score_absent=True"}

    md = [
        "# OBS exist_th holdout sweep",
        "",
        f"- pick_th={args.pick_threshold}, score_absent=True",
        f"- thresholds: {thresholds}",
        "",
    ]

    for ckpt_s, name in zip(args.checkpoints, names):
        ckpt = Path(ckpt_s)
        print(f"[th-sweep] loading {ckpt}", flush=True)
        model, cfg, _ = build_model_from_ckpt(ckpt, device)
        dim = int(cfg["input_dim"])
        samples = filter_alive_channels(holdout, dim)
        print(f"[th-sweep] {name}: n_holdout={len(samples)}", flush=True)
        rows = sweep_exist_th(
            model,
            samples,
            device,
            cfg,
            obs_mod,
            args.pick_threshold,
            thresholds=thresholds,
        )
        best = max(rows, key=lambda r: r["score"])
        # also report @0.50 for protocol continuity
        at50 = next((r for r in rows if abs(r["exist_th"] - 0.5) < 1e-6), None)
        board["models"][name] = {
            "checkpoint": str(ckpt),
            "n_holdout": len(samples),
            "sweep": rows,
            "best": best,
            "at_0.50": at50,
        }
        print(
            f"[th-sweep] {name} best th={best['exist_th']:.2f} "
            f"P={best['p_f1']:.3f} S={best['s_f1']:.3f} existAccS={best['exist_acc_s']:.3f}",
            flush=True,
        )
        md += [
            f"## {name}",
            "",
            f"- ckpt: `{ckpt}`",
            f"- @0.50: P={at50['p_f1']:.3f} S={at50['s_f1']:.3f} existAccS={at50['exist_acc_s']:.3f}"
            if at50
            else "- @0.50: n/a",
            f"- **best th={best['exist_th']:.2f}**: P={best['p_f1']:.3f} S={best['s_f1']:.3f} "
            f"existAccS={best['exist_acc_s']:.3f}",
            "",
            "| exist_th | P F1 | S F1 | existAccS | score |",
            "|---------:|-----:|-----:|----------:|------:|",
        ]
        for r in rows:
            md.append(
                f"| {r['exist_th']:.2f} | {r['p_f1']:.3f} | {r['s_f1']:.3f} | "
                f"{r['exist_acc_s']:.3f} | {r['score']:.3f} |"
            )
        md.append("")

    # Side-by-side summary
    md += ["## Summary", ""]
    for name, block in board["models"].items():
        b, a = block["best"], block.get("at_0.50")
        line = f"- **{name}**: best th={b['exist_th']:.2f} → P={b['p_f1']:.3f}/S={b['s_f1']:.3f}"
        if a:
            line += f"  (vs @0.50 P={a['p_f1']:.3f}/S={a['s_f1']:.3f})"
        md.append(line)
    md.append("")

    (out_dir / "exist_th_sweep.json").write_text(json.dumps(board, indent=2))
    (out_dir / "exist_th_sweep.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
