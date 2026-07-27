#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Val-set hyperparameter sweep for classic EEG SOTA baselines, then full retrain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Curated grids for small-N clinical EEG (ds004504).
SWEEP_GRIDS: dict[str, list[dict]] = {
    "eegnet": [
        {"lr": 3e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-3, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 3e-4, "dropout": 0.40, "weight_decay": 1e-4},
        {"lr": 3e-4, "dropout": 0.50, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.50, "weight_decay": 5e-4},
        {"lr": 3e-4, "dropout": 0.40, "weight_decay": 5e-4},
    ],
    "shallow1d": [
        {"lr": 3e-4, "dropout": 0.25, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.35, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.50, "weight_decay": 5e-4},
        {"lr": 3e-4, "dropout": 0.50, "weight_decay": 5e-4},
        {"lr": 5e-4, "dropout": 0.40, "weight_decay": 1e-4},
        {"lr": 1e-4, "dropout": 0.25, "weight_decay": 5e-4},
        {"lr": 3e-4, "dropout": 0.35, "weight_decay": 5e-4, "batch_size": 8},
    ],
    "conformer": [
        {"lr": 3e-4, "dropout": 0.35, "weight_decay": 5e-4},
        {"lr": 2e-4, "dropout": 0.35, "weight_decay": 5e-4},
        {"lr": 3e-4, "dropout": 0.45, "weight_decay": 5e-4},
        {"lr": 2e-4, "dropout": 0.50, "weight_decay": 1e-3},
        {"lr": 1e-4, "dropout": 0.40, "weight_decay": 1e-3},
        {"lr": 3e-4, "dropout": 0.50, "weight_decay": 1e-3},
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/eeg/adftd_sota_tuned")
    p.add_argument("--models", default="eegnet,shallow1d,conformer")
    p.add_argument("--sweep-epochs", type=int, default=30)
    p.add_argument("--final-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--skip-sweep", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[sota-sweep]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def _train(
    model: str,
    out: Path,
    *,
    epochs: int,
    device: str,
    data_dir: str,
    batch_size: int,
    hp: dict,
) -> float:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_REPO / "tools/train_eeg_baseline.py"),
        "--model",
        model,
        "--data-dir",
        data_dir,
        "--output-dir",
        str(out),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(hp.get("batch_size", batch_size)),
        "--lr",
        str(hp["lr"]),
        "--weight-decay",
        str(hp["weight_decay"]),
        "--dropout",
        str(hp["dropout"]),
        "--device",
        device,
        "--no-synthetic",
    ]
    _run(cmd)
    hist = json.loads((out / "history.json").read_text())
    return float(hist["best_auc"])


def _eval(model: str, ckpt: Path, out: Path, device: str, data_dir: str) -> dict:
    metrics_path = out / "test_metrics.json"
    _run(
        [
            sys.executable,
            str(_REPO / "tools/eval_eeg_baseline.py"),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(metrics_path),
            "--data-dir",
            data_dir,
            "--device",
            device,
            "--no-synthetic",
        ]
    )
    return json.loads(metrics_path.read_text())


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sweep_report: dict = {"models": {}, "protocol": vars(args)}

    for model in models:
        grid = SWEEP_GRIDS.get(model)
        if not grid:
            raise ValueError(f"No sweep grid for {model}")
        best_hp = grid[0]
        best_auc = -1.0
        trials = []
        if not args.skip_sweep:
            for i, hp in enumerate(grid):
                tag = f"trial_{i:02d}"
                trial_out = out_root / model / "sweep" / tag
                auc = _train(
                    model,
                    trial_out,
                    epochs=args.sweep_epochs,
                    device=args.device,
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    hp=hp,
                )
                trials.append({"hp": hp, "val_auc": auc, "dir": str(trial_out)})
                print(f"[sota-sweep] {model} {tag} val_auc={auc:.4f} hp={hp}", flush=True)
                if auc >= best_auc:
                    best_auc = auc
                    best_hp = hp
        else:
            prev = out_root / "sweep_report.json"
            if prev.is_file():
                best_hp = json.loads(prev.read_text())["models"][model]["best_hp"]

        final_out = out_root / model / "best_tuned"
        final_auc = _train(
            model,
            final_out,
            epochs=args.final_epochs,
            device=args.device,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            hp=best_hp,
        )
        metrics = _eval(model, final_out / "best.pt", final_out, args.device, args.data_dir)
        sweep_report["models"][model] = {
            "best_hp": best_hp,
            "sweep_trials": trials,
            "sweep_best_val_auc": best_auc,
            "final_val_auc": final_auc,
            "test_metrics": metrics,
        }

    (out_root / "sweep_report.json").write_text(json.dumps(sweep_report, indent=2))
    print(f"[sota-sweep] wrote {out_root / 'sweep_report.json'}", flush=True)


if __name__ == "__main__":
    main()
