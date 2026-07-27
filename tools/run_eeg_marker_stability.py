#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap FDR stability + cross-checkpoint marker agreement for EEG clinical mining."""

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

from tools.run_eeg_clinical_suite import (
    _aggregate_subjects,
    _collect_split,
    _fdr_mine,
    _load_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints",
        default="outputs/eeg/adftd_hnf_native_v1/best.pt,"
        "outputs/eeg/adftd_hnf_native_v3/best.pt,"
        "outputs/eeg/adftd_hnf_native_v5/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/eeg/marker_stability_native")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


def _bootstrap_stability(
    subjects: list[dict[str, Any]],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(subjects)
    if n == 0:
        return []
    keys = set()
    counts: dict[tuple[str, str], int] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [subjects[i] for i in idx]
        mine = _fdr_mine(sample, alpha)
        for r in mine:
            if r.get("reject_fdr"):
                k = (r["feature"], r["contrast"])
                keys.add(k)
                counts[k] = counts.get(k, 0) + 1
    out = []
    for feat, contrast in sorted(keys):
        c = counts[(feat, contrast)]
        out.append(
            {
                "feature": feat,
                "contrast": contrast,
                "bootstrap_hits": c,
                "bootstrap_freq": float(c / n_boot),
            }
        )
    out.sort(key=lambda r: -r["bootstrap_freq"])
    return out


def _collect_train_subjects(ckpt: Path, device: torch.device, args: argparse.Namespace) -> tuple[str, list[dict]]:
    model, ckpt_args, arch = _load_model(ckpt, device)
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    kparams = model.collect_kernel_params()
    mean_omega = float(np.mean([v["omega"] for v in kparams.values() if "omega" in v])) if kparams else 0.0
    pack = _collect_split(
        model,
        "train",
        data_dir=args.data_dir,
        seed=args.seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        batch_size=16,
        device=device,
        synthetic_if_missing=not args.no_synthetic,
        max_epochs=0,
        mean_omega=mean_omega,
    )
    subj = _aggregate_subjects(pack["epochs"])
    return str(ckpt), subj


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ckpts = [Path(c.strip()) for c in args.checkpoints.split(",") if c.strip()]

    per_ckpt: dict[str, Any] = {}
    train_hits: dict[tuple[str, str], list[str]] = {}

    for ckpt in ckpts:
        if not ckpt.is_file():
            print(f"[marker-stab] skip missing {ckpt}", flush=True)
            continue
        label, subj = _collect_train_subjects(ckpt, device, args)
        mine = _fdr_mine(subj, args.fdr_alpha)
        boot = _bootstrap_stability(subj, n_boot=args.bootstrap, alpha=args.fdr_alpha, seed=args.seed)
        per_ckpt[label] = {
            "n_train_subjects": len(subj),
            "n_fdr_train": sum(1 for r in mine if r.get("reject_fdr")),
            "fdr_train_top": [r for r in mine if r.get("reject_fdr")][:15],
            "bootstrap_stable": [r for r in boot if r["bootstrap_freq"] >= 0.5],
            "bootstrap_all": boot,
        }
        for r in mine:
            if r.get("reject_fdr"):
                k = (r["feature"], r["contrast"])
                train_hits.setdefault(k, []).append(label)

    cross = []
    for (feat, contrast), ckpt_list in sorted(train_hits.items()):
        cross.append(
            {
                "feature": feat,
                "contrast": contrast,
                "n_checkpoints": len(ckpt_list),
                "checkpoints": ckpt_list,
            }
        )
    cross.sort(key=lambda r: (-r["n_checkpoints"], r["feature"]))

    report = {
        "protocol": {
            "bootstrap_iters": args.bootstrap,
            "fdr_alpha": args.fdr_alpha,
            "stable_threshold": 0.5,
            "checkpoints": [str(c) for c in ckpts if c.is_file()],
        },
        "per_checkpoint": per_ckpt,
        "cross_checkpoint_agreement": cross,
    }
    (out / "marker_stability.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# EEG marker stability (bootstrap + cross-checkpoint)",
        "",
        f"- bootstrap={args.bootstrap}, FDR α={args.fdr_alpha}, stable if freq≥0.5",
        "",
        "## Cross-checkpoint FDR hits",
        "",
        "| feature | contrast | # ckpts | checkpoints |",
        "|---------|----------|--------:|-------------|",
    ]
    for r in cross[:20]:
        lines.append(
            f"| {r['feature']} | {r['contrast']} | {r['n_checkpoints']} | "
            f"{', '.join(Path(c).parent.name for c in r['checkpoints'])} |"
        )
    md = "\n".join(lines)
    (out / "marker_stability.md").write_text(md)
    print(md, flush=True)
    print(f"[marker-stab] wrote {out / 'marker_stability.json'}", flush=True)


if __name__ == "__main__":
    main()
