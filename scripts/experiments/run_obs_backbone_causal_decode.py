#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode-only ablation: hand priors vs backbone-causal field gating."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "paper"))

from tools.analyze_stead_picking import load_model  # noqa: E402
from tools.obs_matched_split import load_split_samples  # noqa: E402
from tools.train_obs_picking import filter_alive_channels, shrink_sample_waves  # noqa: E402
import run_paper_obs_picking_compare as obs_cmp  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument("--split", default="holdout", choices=["holdout", "val"])
    p.add_argument("--max-n", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/backbone_causal_decode.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(Path(args.checkpoint), device)
    model.eval()
    samples, info, _ = load_split_samples(args.split_json, args.split)
    samples = filter_alive_channels(samples, 4, mode="strict")
    shrink_sample_waves(samples, 4)
    if args.max_n > 0:
        samples = samples[: args.max_n]
    print(f"[bb-causal] n={len(samples)} split={args.split} {info}", flush=True)

    seq_len = int(ckpt_args.get("seq_len", 1200))
    window_sec = float(ckpt_args.get("window_sec", 60.0))
    recipes = [
        {"name": "argmax", "p_decode_mode": "argmax"},
        {
            "name": "score_minus_late_0.60",
            "p_decode_mode": "score_minus_late",
            "decode_late_penalty": 0.60,
        },
        {
            "name": "backbone_causal",
            "p_decode_mode": "backbone_causal",
            "decode_compete_ratio": 0.70,
        },
        {
            "name": "earliest_competitive_0.70",
            "p_decode_mode": "earliest_competitive",
            "decode_compete_ratio": 0.70,
        },
    ]
    rows = []
    for r in recipes:
        print(f"[bb-causal] eval {r['name']} …", flush=True)
        m = obs_cmp.eval_hnf(
            model,
            samples,
            device,
            seq_len,
            window_sec,
            pick_th=0.25,
            det_th=0.5,
            tol_sec=0.5,
            batch_size=args.batch_size,
            n_channels=4,
            exist_th=0.60,
            score_absent=True,
            gate_mode="soft_floor",
            soft_th=0.25,
            p_decode_mode=r["p_decode_mode"],
            s_decode_mode="argmax",
            decode_compete_ratio=float(r.get("decode_compete_ratio", 0.70)),
            decode_late_penalty=float(r.get("decode_late_penalty", 0.0)),
            apply_p_residual=False,
        )
        po = m["pick_only"]
        row = {
            "name": r["name"],
            "p_f1": po.get("p_f1"),
            "s_f1": po.get("s_f1"),
            "p_precision": po.get("p_precision"),
            "p_recall": po.get("p_recall"),
        }
        rows.append(row)
        print(f"  → P={row['p_f1']:.4f} S={row['s_f1']:.4f}", flush=True)

    out = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n": len(samples),
        "note": "backbone_causal = pick candidates × Huygens P-field cause weight",
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[bb-causal] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
