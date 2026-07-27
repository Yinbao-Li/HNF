#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperparameter sweep for Braindecode SOTA EEG models on ds004504."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

SWEEP_GRIDS: dict[str, list[dict]] = {
    "eegnetv4": [
        {"lr": 1e-3, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 3e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-3, "dropout": 0.5, "weight_decay": 1e-4},
        {"lr": 3e-4, "dropout": 0.5, "weight_decay": 5e-4},
    ],
    "shallowfbcsp": [
        {"lr": 1e-3, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 3e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.5, "weight_decay": 1e-4},
        {"lr": 3e-4, "dropout": 0.5, "weight_decay": 5e-4},
        {"lr": 1e-3, "dropout": 0.5, "weight_decay": 5e-4},
    ],
    "deep4net": [
        {"lr": 3e-4, "dropout": 0.5, "weight_decay": 5e-4},
        {"lr": 1e-4, "dropout": 0.5, "weight_decay": 5e-4},
        {"lr": 3e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-3, "dropout": 0.5, "weight_decay": 1e-3},
    ],
    "eegconformer": [
        {"lr": 3e-4, "dropout": 0.35, "weight_decay": 5e-4, "att_dropout": 0.35},
        {"lr": 2e-4, "dropout": 0.35, "weight_decay": 5e-4, "att_dropout": 0.35},
        {"lr": 1e-4, "dropout": 0.5, "weight_decay": 1e-3, "att_dropout": 0.5},
        {"lr": 3e-4, "dropout": 0.5, "weight_decay": 1e-3, "att_dropout": 0.5},
        {"lr": 2e-4, "dropout": 0.45, "weight_decay": 5e-4, "att_dropout": 0.4},
        {"lr": 1e-4, "dropout": 0.35, "weight_decay": 5e-4, "att_dropout": 0.3},
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/eeg/adftd_braindecode_sota")
    p.add_argument(
        "--models",
        default="eegnetv4,shallowfbcsp,deep4net,eegconformer",
    )
    p.add_argument("--sweep-epochs", type=int, default=30)
    p.add_argument("--final-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    return p.parse_args()


def _run_train(model: str, out: Path, epochs: int, hp: dict, args: argparse.Namespace) -> float:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_REPO / "tools/train_eeg_braindecode.py"),
        "--model",
        model,
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(out),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(hp.get("batch_size", args.batch_size)),
        "--lr",
        str(hp["lr"]),
        "--weight-decay",
        str(hp["weight_decay"]),
        "--dropout",
        str(hp["dropout"]),
        "--device",
        args.device,
        "--no-synthetic",
    ]
    if hp.get("att_dropout") is not None:
        cmd.extend(["--att-dropout", str(hp["att_dropout"])])
    print("[bd-sweep]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    hist = json.loads((out / "history.json").read_text())
    return float(hist["best_auc"])


def _eval(model: str, ckpt: Path, out: Path, args: argparse.Namespace) -> dict:
    metrics_path = out / "test_metrics.json"
    cmd = [
        sys.executable,
        str(_REPO / "tools/eval_eeg_baseline.py"),
        "--checkpoint",
        str(ckpt),
        "--output",
        str(metrics_path),
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--no-synthetic",
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    return json.loads(metrics_path.read_text())


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report: dict = {"models": {}, "protocol": vars(args)}

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        grid = SWEEP_GRIDS[model]
        best_hp, best_auc, trials = grid[0], -1.0, []
        for i, hp in enumerate(grid):
            trial_out = out_root / model / "sweep" / f"trial_{i:02d}"
            auc = _run_train(model, trial_out, args.sweep_epochs, hp, args)
            trials.append({"hp": hp, "val_auc": auc})
            print(f"[bd-sweep] {model} trial_{i:02d} val_auc={auc:.4f} hp={hp}", flush=True)
            if auc >= best_auc:
                best_auc, best_hp = auc, hp

        final_out = out_root / model / "best_tuned"
        final_auc = _run_train(model, final_out, args.final_epochs, best_hp, args)
        metrics = _eval(model, final_out / "best.pt", final_out, args)
        report["models"][model] = {
            "best_hp": best_hp,
            "sweep_trials": trials,
            "sweep_best_val_auc": best_auc,
            "final_val_auc": final_auc,
            "test_metrics": metrics,
        }
        (out_root / "sweep_report.json").write_text(json.dumps(report, indent=2))

    (out_root / "sweep_report.json").write_text(json.dumps(report, indent=2))
    print(f"[bd-sweep] wrote {out_root / 'sweep_report.json'}", flush=True)


if __name__ == "__main__":
    main()
