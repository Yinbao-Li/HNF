#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train EEGNet / Shallow1D / Transformer on same protocol as HNF; write compare board."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG baseline + HNF compare launcher")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--output-dir", default="outputs/eeg/adftd_baseline_compare")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", default="eegnet,shallow1d,transformer")
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


def _ad_ftd_from_per_subject(per_subj: dict) -> float:
    yt, yp = [], []
    for _sid, row in per_subj.items():
        lab = int(row["label"])
        if lab not in (1, 2):
            continue
        yt.append(lab)
        yp.append(int(row["pred"]))
    if not yt:
        return float("nan")
    import numpy as np

    yt = np.asarray(yt)
    yp = np.asarray(yp)
    return float((yt == yp).mean())


def _hnf_row(label: str, path: Path) -> dict | None:
    if not path.is_file():
        print(f"[eeg-compare] WARN missing {label}: {path}", flush=True)
        return None
    with path.open(encoding="utf-8") as f:
        m = json.load(f)
    subj_acc = m.get("subject_accuracy", m.get("test_subject_accuracy"))
    auc = m.get("auc_macro", m.get("test_epoch_auc"))
    ad_ftd = m.get("ad_ftd_subject_accuracy")
    if ad_ftd is None and "per_subject" in m:
        ad_ftd = _ad_ftd_from_per_subject(m["per_subject"])
    return {
        "model": label,
        "accuracy": m.get("accuracy", m.get("test_epoch_acc")),
        "macro_f1": m.get("macro_f1"),
        "subject_accuracy": subj_acc,
        "ad_ftd_subject_accuracy": ad_ftd,
        "auc_macro": auc,
        "n_params": m.get("n_params"),
        "n_subjects": m.get("n_subjects", m.get("n_test_subjects")),
        "source": str(path),
    }


def _row(name: str, metrics: dict) -> dict:
    return {
        "model": name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "subject_accuracy": metrics.get("subject_accuracy"),
        "ad_ftd_subject_accuracy": metrics.get("ad_ftd_subject_accuracy"),
        "auc_macro": metrics.get("auc_macro"),
        "n_params": metrics.get("n_params"),
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

    rows: list[dict] = []
    for hnf_label, hnf_path in [
        ("HNF Stage-1", Path("outputs/eeg/adftd_hnf_stage1/test_metrics.json")),
        ("HNF native v3", Path("outputs/eeg/adftd_hnf_native_v3/test_metrics.json")),
    ]:
        r = _hnf_row(hnf_label, hnf_path)
        if r is not None:
            rows.append(r)

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
            "note": "Transformer = Conv stem + CLS + 3-layer encoder (d=64, h=4)",
        },
        "rows": rows,
    }
    summary_path = out / "compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# EEG classification compare (same split, n_test=18 subjects)",
        "",
        "| Model | subject acc | AD↔FTD acc | macro-AUC | epoch acc | macro-F1 | params |",
        "|-------|------------:|-----------:|----------:|----------:|---------:|-------:|",
    ]
    for r in rows:
        ad = r.get("ad_ftd_subject_accuracy")
        ad_s = f"{ad:.3f}" if ad is not None and ad == ad else "—"
        mf1 = r.get("macro_f1")
        mf1_s = f"{mf1:.3f}" if mf1 is not None and mf1 == mf1 else "—"
        md_lines.append(
            f"| {r['model']} | {r['subject_accuracy']:.3f} | {ad_s} | "
            f"{r['auc_macro']:.3f} | {r['accuracy']:.3f} | {mf1_s} | {r.get('n_params', '—')} |"
        )
    md_lines += [
        "",
        "Protocol: subject-level = mean epoch softmax then argmax; AD↔FTD on true AD∪FTD only.",
        "",
    ]
    md_path = out / "compare_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines), flush=True)
    print(f"[eeg-compare] wrote {summary_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
