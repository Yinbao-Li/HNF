#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train EEGNet + Shallow1D on the same protocol as HNF Stage-1 and write a compare board."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG baseline compare launcher")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--hnf-metrics", default="outputs/eeg/adftd_hnf_stage1/test_metrics.json")
    p.add_argument("--output-dir", default="outputs/eeg/adftd_baseline_compare")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", default="eegnet,shallow1d")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], log_path: Path) -> None:
    print("[eeg-compare]", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write("\n$ " + " ".join(cmd) + "\n")
        logf.flush()
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(rc)


def _row(name: str, metrics: dict, n_params: int | None = None) -> dict:
    return {
        "model": name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "subject_accuracy": metrics.get("subject_accuracy"),
        "auc_macro": metrics.get("auc_macro"),
        "n_params": n_params if n_params is not None else metrics.get("n_params"),
        "n_subjects": metrics.get("n_subjects"),
        "source": metrics.get("checkpoint") or metrics.get("source"),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "compare.log"
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.dry_run:
        print("models:", models)
        return

    rows = []
    hnf_path = Path(args.hnf_metrics)
    if hnf_path.is_file():
        with hnf_path.open(encoding="utf-8") as f:
            hnf = json.load(f)
        # HNF stage1 params from known run (~89k)
        rows.append(
            _row(
                "HNF(stage1)",
                {**hnf, "source": str(hnf_path), "checkpoint": hnf.get("checkpoint")},
                n_params=89442,
            )
        )
    else:
        print(f"[eeg-compare] WARN missing HNF metrics: {hnf_path}", flush=True)

    for name in models:
        model_out = out / name
        model_out.mkdir(parents=True, exist_ok=True)
        ckpt = model_out / "best.pt"
        metrics_path = model_out / "test_metrics.json"
        if not args.skip_train:
            _run(
                [
                    sys.executable,
                    "tools/train_eeg_baseline.py",
                    "--model",
                    name,
                    "--data-dir",
                    args.data_dir,
                    "--output-dir",
                    str(model_out),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--no-synthetic",
                ],
                log_path,
            )
        _run(
            [
                sys.executable,
                "tools/eval_eeg_baseline.py",
                "--checkpoint",
                str(ckpt),
                "--output",
                str(metrics_path),
                "--data-dir",
                args.data_dir,
                "--device",
                args.device,
                "--no-synthetic",
            ],
            log_path,
        )
        with metrics_path.open(encoding="utf-8") as f:
            m = json.load(f)
        rows.append(_row(name, m))

    summary = {
        "protocol": {
            "dataset": "OpenNeuro ds004504",
            "split_seed": 42,
            "input": "19ch x 10s @ 128Hz (1280)",
            "train_stride_sec": 5.0,
            "test_stride": "non-overlap",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "selection": "best val macro-AUC",
        },
        "rows": rows,
    }
    summary_path = out / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# EEG Stage-1 vs baselines (same protocol)",
        "",
        "| Model | subject_acc | macro-AUC | epoch_acc | macro-F1 | params |",
        "|-------|------------:|----------:|----------:|---------:|-------:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['subject_accuracy']:.3f} | {r['auc_macro']:.3f} | "
            f"{r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r.get('n_params', '—')} |"
        )
    md_lines.append("")
    md_path = out / "compare_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines), flush=True)
    print(f"[eeg-compare] wrote {summary_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
