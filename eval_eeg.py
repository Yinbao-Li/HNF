# -*- coding: utf-8 -*-
"""Evaluate EEG HNF classifier on the held-out test split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hnf.eeg_dataset import EEGDataset, ID_TO_LABEL, LABEL_TO_ID
from hnf.eeg_model import EEGHNFClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval EEG HNF classifier")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--checkpoint", default="outputs/eeg/adftd_hnf/best.pt")
    p.add_argument("--output", default="outputs/eeg/adftd_hnf/test_metrics.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-if-missing", action="store_true", default=True)
    p.add_argument("--no-synthetic", action="store_false", dest="synthetic_if_missing")
    return p.parse_args()


def _safe_auc(y: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        from sklearn.metrics import roc_auc_score

        for c in range(num_classes):
            yt = (y == c).astype(np.int64)
            if yt.min() == yt.max():
                out[f"auc_class_{c}"] = float("nan")
                continue
            out[f"auc_class_{c}"] = float(roc_auc_score(yt, probs[:, c]))
        present = len(set(y.tolist()))
        if present >= 2:
            out["auc_macro"] = float(
                roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(num_classes)))
            )
        else:
            out["auc_macro"] = float("nan")
    except Exception:
        out["auc_macro"] = float("nan")
        for c in range(num_classes):
            out[f"auc_class_{c}"] = float("nan")
    return out


def _macro_f1(y: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(float(2 * prec * rec / (prec + rec + 1e-8)))
    return float(np.mean(f1s))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    principle = str(ckpt_args.get("principle", "huygens_fresnel"))
    dropout = float(ckpt_args.get("dropout", 0.2))

    ds = EEGDataset(
        data_dir=args.data_dir,
        split="test",
        seed=args.seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,  # non-overlap for eval
        synthetic_if_missing=args.synthetic_if_missing,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = EEGHNFClassifier(
        n_channels=19,
        seq_len=int(round(epoch_sec * sample_rate)),
        sample_rate=sample_rate,
        embed_dim=embed_dim,
        num_classes=len(LABEL_TO_ID),
        dropout=dropout,
        principle=principle,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    all_logits: list[np.ndarray] = []
    all_y: list[int] = []
    all_sid: list[str] = []
    for batch in loader:
        x = batch["x"].to(device)
        logits = model(x).cpu().numpy()
        all_logits.append(logits)
        all_y.extend([int(v) for v in batch["label"]])
        all_sid.extend(list(batch["subject_id"]))

    logits = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    y = np.asarray(all_y, dtype=np.int64)
    pred = probs.argmax(axis=1)
    num_classes = probs.shape[1]

    epoch_acc = float((pred == y).mean())
    epoch_f1 = _macro_f1(y, pred, num_classes)
    aucs = _safe_auc(y, probs, num_classes)

    # Subject-level: mean softmax then argmax
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
    sub_pred_a = np.asarray(sub_pred)
    sub_true_a = np.asarray(sub_true)
    subject_acc = float((sub_pred_a == sub_true_a).mean()) if len(sub_true) else float("nan")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1

    result = {
        "checkpoint": str(args.checkpoint),
        "n_epochs_eval": int(len(y)),
        "n_subjects": int(len(buckets)),
        "accuracy": epoch_acc,
        "macro_f1": epoch_f1,
        "subject_accuracy": subject_acc,
        "confusion_matrix": cm.tolist(),
        "id_to_label": ID_TO_LABEL,
        **aucs,
        "per_subject": {
            sid: {
                "label": labels_by_sid[sid],
                "pred": int(np.mean(np.stack(buckets[sid], 0), 0).argmax()),
            }
            for sid in sorted(buckets)
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("accuracy", "macro_f1", "subject_accuracy", "auc_macro")}, indent=2))
    print(f"[EEG-eval] wrote {out}")


if __name__ == "__main__":
    main()
