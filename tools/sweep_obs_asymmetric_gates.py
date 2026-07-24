#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asymmetric P/S gate-rule sweep on a cached forward (L1200).

P and S use independent thresholds — the free lunch we under-used when tying them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.picking_metrics import tolerance_bins
from tools.obs_matched_split import load_split_samples
from tools.sweep_obs_gate_rules import cache_forward, score_phase
from tools.train_obs_exist_gate import build_model_from_ckpt
from tools.train_obs_picking import filter_alive_channels, _load_obs_compare_module
from tools.train_stead_picking import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asymmetric P/S gate sweep")
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/asymmetric_gate_sweep",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    obs = _load_obs_compare_module()
    model, cfg, _ = build_model_from_ckpt(Path(args.checkpoint), device)
    holdout, _, _ = load_split_samples(args.split_json, "holdout")
    holdout = filter_alive_channels(holdout, int(cfg["input_dim"]), mode="strict")
    print(f"[asym-sweep] cache n={len(holdout)}", flush=True)
    cache = cache_forward(model, holdout, device, cfg, obs, 4)
    tol = tolerance_bins(cache["seq_len"], 0.5)

    p_pick_ths = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    s_pick_ths = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    s_exist_ths = [0.35, 0.45, 0.55, 0.60, 0.65]
    s_soft_ths = [0.20, 0.26, 0.30, 0.34, 0.38]

    rows = []
    # hard P × hard S
    for ppt in p_pick_ths:
        p_m = score_phase(
            cache["p_peak"], cache["p_pred"], cache["p_exist"],
            cache["p_valid"], cache["p_gt"],
            mode="hard", pick_th=ppt, exist_th=0.5, soft_th=0.0, tol=tol, score_absent=False,
        )
        for spt in s_pick_ths:
            for set_ in s_exist_ths:
                s_m = score_phase(
                    cache["s_peak"], cache["s_pred"], cache["s_exist"],
                    cache["s_valid"], cache["s_gt"],
                    mode="hard", pick_th=spt, exist_th=set_, soft_th=0.0, tol=tol, score_absent=True,
                )
                score = 0.55 * p_m["f1"] + 0.45 * s_m["f1"]
                rows.append(
                    {
                        "p_mode": "hard",
                        "s_mode": "hard",
                        "p_pick_th": ppt,
                        "s_pick_th": spt,
                        "s_exist_th": set_,
                        "s_soft_th": None,
                        "p_f1": p_m["f1"],
                        "s_f1": s_m["f1"],
                        "score": score,
                    }
                )
        for sst in s_soft_ths:
            s_m = score_phase(
                cache["s_peak"], cache["s_pred"], cache["s_exist"],
                cache["s_valid"], cache["s_gt"],
                mode="soft", pick_th=0.0, exist_th=0.0, soft_th=sst, tol=tol, score_absent=True,
            )
            score = 0.55 * p_m["f1"] + 0.45 * s_m["f1"]
            rows.append(
                {
                    "p_mode": "hard",
                    "s_mode": "soft",
                    "p_pick_th": ppt,
                    "s_pick_th": None,
                    "s_exist_th": None,
                    "s_soft_th": sst,
                    "p_f1": p_m["f1"],
                    "s_f1": s_m["f1"],
                    "score": score,
                }
            )

    best = max(rows, key=lambda r: r["score"])
    best_bal = max(rows, key=lambda r: min(r["p_f1"], r["s_f1"]))
    base = next(
        r
        for r in rows
        if r["p_pick_th"] == 0.25 and r["s_mode"] == "hard" and r["s_pick_th"] == 0.25 and r["s_exist_th"] == 0.60
    )
    rec045 = next(
        r
        for r in rows
        if r["p_pick_th"] == 0.45 and r["s_mode"] == "hard" and r["s_pick_th"] == 0.45 and r["s_exist_th"] == 0.60
    )

    report = {
        "checkpoint": args.checkpoint,
        "n": len(holdout),
        "baseline_tied_025_060": base,
        "tied_045_060": rec045,
        "best_score": best,
        "best_minPS": best_bal,
        "top15": sorted(rows, key=lambda r: -r["score"])[:15],
        "n_configs": len(rows),
    }
    (out / "asymmetric_gate_sweep.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Asymmetric P/S gate sweep",
        "",
        f"- ckpt: `{args.checkpoint}` n={len(holdout)}",
        "",
        f"- tied 0.25/0.60: P={base['p_f1']:.3f} S={base['s_f1']:.3f}",
        f"- tied 0.45/0.60: P={rec045['p_f1']:.3f} S={rec045['s_f1']:.3f}",
        f"- **best score**: P={best['p_f1']:.3f} S={best['s_f1']:.3f}  {best}",
        f"- **best min(P,S)**: P={best_bal['p_f1']:.3f} S={best_bal['s_f1']:.3f}  {best_bal}",
        "",
        "## Top 10 by score",
        "",
        "| P mode/th | S mode | S params | P | S | score |",
        "|-----------|--------|----------|--:|--:|------:|",
    ]
    for r in report["top15"][:10]:
        sparam = (
            f"pick={r['s_pick_th']} exist={r['s_exist_th']}"
            if r["s_mode"] == "hard"
            else f"soft={r['s_soft_th']}"
        )
        md.append(
            f"| hard/{r['p_pick_th']} | {r['s_mode']} | {sparam} | "
            f"{r['p_f1']:.3f} | {r['s_f1']:.3f} | {r['score']:.3f} |"
        )
    md.append("")
    (out / "asymmetric_gate_sweep.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
