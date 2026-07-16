#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate EEG baseline checkpoint with the same protocol as eval_eeg.py."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hnf.eeg_baselines import build_eeg_baseline
from hnf.eeg_dataset import EEGDataset, ID_TO_LABEL, LABEL_TO_ID
from tools.eval_eeg import _macro_f1, _safe_auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval EEG baseline classifier")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-if-missing", action="store_true", default=True)
    p.add_argument("--no-synthetic", action="store_false", dest="synthetic_if_missing")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model_name = str(ckpt.get("model_name") or ckpt_args.get("model") or "eegnet")
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    dropout = float(ckpt_args.get("dropout", 0.25))
    n_samples = int(round(epoch_sec * sample_rate))

    ds = EEGDataset(
        data_dir=args.data_dir,
        split="test",
        seed=args.seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,
        synthetic_if_missing=args.synthetic_if_missing,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_eeg_baseline(
        model_name,
        n_channels=19,
        n_samples=n_samples,
        n_classes=len(LABEL_TO_ID),
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    all_logits: list[np.ndarray] = []
    all_y: list[int] = []
    all_sid: list[str] = []
    for batch in loader:
        logits = model(batch["x"].to(device)).cpu().numpy()
        all_logits.append(logits)
        all_y.extend([int(v) for v in batch["label"]])
        all_sid.extend(list(batch["subject_id"]))

    logits = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    y = np.asarray(all_y, dtype=np.int64)
    pred = probs.argmax(axis=1)
    num_classes = probs.shape[1]

    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    labels_by_sid: dict[str, int] = {}
    for sid, pr, yi in zip(all_sid, probs, y):
        buckets[sid].append(pr)
        labels_by_sid[sid] = int(yi)
    sub_pred = []
    sub_true = []
    for sid, plist in buckets.items():
        mean_p = np.mean(np.stack(plist, axis=0), axis=0)
        sub_pred.append(int(mean_p.argmax()))
        sub_true.append(labels_by_sid[sid])

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1

    result = {
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "n_epochs_eval": int(len(y)),
        "n_subjects": int(len(buckets)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": _macro_f1(y, pred, num_classes),
        "subject_accuracy": float((np.asarray(sub_pred) == np.asarray(sub_true)).mean()),
        "confusion_matrix": cm.tolist(),
        "id_to_label": ID_TO_LABEL,
        **_safe_auc(y, probs, num_classes),
        "n_params": int(ckpt.get("n_params", -1)),
        "best_val_auc": float(ckpt.get("val_metrics", {}).get("auc", float("nan"))),
    }
    out = Path(args.output or Path(args.checkpoint).with_name("test_metrics.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {k: result[k] for k in ("model", "accuracy", "macro_f1", "subject_accuracy", "auc_macro")},
            indent=2,
        )
    )
    print(f"[EEG-base-eval] wrote {out}")


if __name__ == "__main__":
    main()
