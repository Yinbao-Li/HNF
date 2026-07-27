#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Induce EEG pattern library from a frozen ckpt, then route-eval on val/test.

Tight-policy recipe (v2):
  - stricter AD↔FTD second-look induction + runtime confusion gate
  - val-calibrated OOD distance abstain
  - online confirm/reject counters on val→test (centres frozen)

Usage:
  PYTHONPATH=. python tools/build_eeg_pattern_library.py \\
    --checkpoint outputs/eeg/adftd_hnf_native_v3/best.pt \\
    --output-dir outputs/eeg/pattern_library_native_v3_tight \\
    --device cuda --no-synthetic --online-update
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from typing import Any

import numpy as np
import torch

from hnf.eeg_pattern_library import EEGPatternLibrary, evaluate_routed_subjects
from tools.run_eeg_clinical_suite import (
    _aggregate_subjects,
    _collect_split,
    _load_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build + eval EEG clinical pattern library")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--checkpoint", default="outputs/eeg/adftd_hnf_native_v3/best.pt")
    p.add_argument("--output-dir", default="outputs/eeg/pattern_library_native_v3_tight")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-synthetic", action="store_true")
    p.add_argument("--min-coverage", type=float, default=0.70)
    p.add_argument(
        "--online-update",
        action="store_true",
        default=True,
        help="Update confirm/reject counters on val then test (centres frozen)",
    )
    p.add_argument("--no-online-update", action="store_false", dest="online_update")
    return p.parse_args()


def _mean_omega(model: torch.nn.Module) -> float:
    if not hasattr(model, "collect_kernel_params"):
        return 0.0
    vals = []
    for _n, d in model.collect_kernel_params().items():
        if "omega" in d:
            vals.append(float(d["omega"]))
    return float(np.mean(vals)) if vals else 0.0


def _collect_subjects(
    model: torch.nn.Module,
    split: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
    sample_rate: int,
    epoch_sec: float,
    mean_omega: float,
) -> list[dict[str, Any]]:
    packed = _collect_split(
        model,
        split,
        data_dir=args.data_dir,
        seed=args.seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        batch_size=args.batch_size,
        device=device,
        synthetic_if_missing=not args.no_synthetic,
        max_epochs=0,
        mean_omega=mean_omega,
    )
    return _aggregate_subjects(packed["epochs"])


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lib_sum = payload["library_summary"]
    cal = payload.get("calibration") or {}
    lines = [
        "# EEG pattern library report (tight policies)",
        "",
        f"Checkpoint: `{payload['checkpoint']}`",
        f"k = **{payload['k']}**, seed = {payload['seed']}",
        f"max_route_distance = `{cal.get('chosen')}` ({cal.get('reason')})",
        "",
        "## Induced prototypes (train)",
        "",
        "| id | name | n | HC/FTD/AD | train_acc | policy | confirm/reject |",
        "|---:|------|--:|-----------|----------:|--------|---------------:|",
    ]
    for r in lib_sum:
        lines.append(
            f"| {r['id']} | `{r['name']}` | {r['count']} | "
            f"{r['HC']}/{r['FTD']}/{r['AD']} | {r['train_acc']:.3f} | {r['policy']} | "
            f"{r['confirm']}/{r['reject']} |"
        )
    lines += ["", "## Split metrics", ""]
    for split, m in payload["metrics"].items():
        lines += [
            f"### {split}",
            f"- n = {m['n_subjects']}, coverage = {m['coverage']:.3f}, abstain = {m['n_abstain']}",
            f"- baseline subject_acc = **{m['baseline_subject_acc']:.3f}**, "
            f"AD↔FTD = **{m['baseline_ad_ftd_acc']:.3f}**",
            f"- routed (abstain→baseline fill) subject_acc = **{m['routed_fill_subject_acc']:.3f}**, "
            f"AD↔FTD = **{m['routed_fill_ad_ftd_acc']:.3f}**",
            f"- routed **kept-only** subject_acc = **{m['routed_kept_subject_acc']}**, "
            f"AD↔FTD = **{m['routed_kept_ad_ftd_acc']}**",
            f"- policy counts: `{m['policy_counts']}`",
            f"- decision reasons: `{m.get('reason_counts')}`",
            "",
        ]
    lines += [
        "## Reading",
        "",
        "- Tight second-look only fires when the head is *confused* (small AD–FTD gap).",
        "- Abstain = low confidence **or** OOD route distance (val-calibrated).",
        "- kept-only metrics are the clinically honest operating point (coverage trade-off).",
        "- Online updates touch confirm/reject counters only — centres stay frozen.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, a, arch = _load_model(Path(args.checkpoint), device)
    sample_rate = int(a.get("sample_rate", 128))
    epoch_sec = float(a.get("epoch_sec", 10.0))
    mean_omega = _mean_omega(model)
    print(
        f"[eeg-pattern-lib] device={device} arch={arch} ckpt={args.checkpoint} "
        f"k={args.k} mean_omega={mean_omega:.3f} online={args.online_update}",
        flush=True,
    )

    print("[eeg-pattern-lib] collecting train/val/test …", flush=True)
    train_subj = _collect_subjects(
        model, "train", args=args, device=device,
        sample_rate=sample_rate, epoch_sec=epoch_sec, mean_omega=mean_omega,
    )
    val_subj = _collect_subjects(
        model, "val", args=args, device=device,
        sample_rate=sample_rate, epoch_sec=epoch_sec, mean_omega=mean_omega,
    )
    test_subj = _collect_subjects(
        model, "test", args=args, device=device,
        sample_rate=sample_rate, epoch_sec=epoch_sec, mean_omega=mean_omega,
    )
    print(
        f"[eeg-pattern-lib] n train/val/test = "
        f"{len(train_subj)}/{len(val_subj)}/{len(test_subj)}",
        flush=True,
    )

    lib = EEGPatternLibrary.build_from_subjects(
        train_subj,
        k=args.k,
        seed=args.seed,
        checkpoint=str(args.checkpoint),
    )
    print("[eeg-pattern-lib] calibrate distance gate on val …", flush=True)
    calibration = lib.calibrate_distance_gate(
        val_subj, min_coverage=float(args.min_coverage)
    )
    print(json.dumps(calibration, indent=2, default=float), flush=True)

    lib_path = out / "pattern_library.json"
    lib.save(lib_path)
    print(json.dumps(lib.summary(), indent=2), flush=True)

    # Single library instance so val→test counters accumulate when online.
    metrics: dict[str, Any] = {}
    for split, subj in (("train", train_subj), ("val", val_subj), ("test", test_subj)):
        online = bool(args.online_update and split in {"val", "test"})
        print(f"[eeg-pattern-lib] evaluate {split} (online={online}) …", flush=True)
        m = evaluate_routed_subjects(
            lib,
            subj,
            online_update=online,
            update_center=False,
        )
        metrics[split] = m
        print(
            f"[eeg-pattern-lib] {split}: base_acc={m['baseline_subject_acc']:.3f} "
            f"fill={m['routed_fill_subject_acc']:.3f} "
            f"kept={m['routed_kept_subject_acc']} "
            f"adftd_fill={m['routed_fill_ad_ftd_acc']:.3f} "
            f"adftd_kept={m['routed_kept_ad_ftd_acc']} "
            f"coverage={m['coverage']:.3f} abstain={m['n_abstain']}",
            flush=True,
        )

    lib.save(out / "pattern_library_after_online.json")
    payload = {
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "k": args.k,
        "seed": args.seed,
        "n_train_subjects": len(train_subj),
        "calibration": calibration,
        "library_summary": lib.summary(),
        "max_route_distance": lib.max_route_distance,
        "metrics": {
            split: {k: v for k, v in m.items() if k != "rows"}
            for split, m in metrics.items()
        },
        "test_rows": metrics["test"]["rows"],
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out / "REPORT.md", payload)
    print(f"[eeg-pattern-lib] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
