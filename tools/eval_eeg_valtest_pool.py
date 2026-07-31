#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-evaluate EEG models on pooled val+test subjects (no train leakage).

Caveat: checkpoints were typically selected on val metrics, so pooling val into
the evaluation set is mildly optimistic vs a pure held-out test. Still useful as
a larger-N (≈31 subject) ranking under the same seed=42 subject split.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from hnf.eeg_baselines import build_eeg_baseline
from hnf.eeg_braindecode_models import build_braindecode_model, display_name
from hnf.eeg_dataset import CLINICAL_ID_TO_LABEL, EEGDataset, LABEL_TO_ID
from tools.eval_eeg import _macro_f1, _safe_auc
from tools.run_eeg_clinical_suite import _load_model
from tools.train_eeg import _macro_auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval EEG models on val+test pool")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--output-dir", default="outputs/eeg/valtest_pool_eval")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


def _pool_loaders(
    *,
    data_dir: str,
    seed: int,
    sample_rate: int,
    epoch_sec: float,
    batch_size: int,
    synthetic_if_missing: bool,
    overlap: bool,
):
    stride = float(epoch_sec) * (0.5 if overlap else 1.0)
    common = dict(
        data_dir=data_dir,
        seed=seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=stride,
        synthetic_if_missing=synthetic_if_missing,
        test_ratio=0.2,
        val_ratio=0.15,
    )
    val_ds = EEGDataset(split="val", **common)
    test_ds = EEGDataset(split="test", **common)
    pool = ConcatDataset([val_ds, test_ds])
    loader = DataLoader(pool, batch_size=batch_size, shuffle=False)
    meta = {
        "n_val_subjects": len(val_ds.subjects),
        "n_test_subjects": len(test_ds.subjects),
        "n_pool_subjects": len(val_ds.subjects) + len(test_ds.subjects),
        "n_val_epochs": len(val_ds),
        "n_test_epochs": len(test_ds),
        "n_pool_epochs": len(pool),
        "stride_sec": stride,
        "overlap": overlap,
        "val_label_counts": dict(Counter(s.label for s in val_ds.subjects)),
        "test_label_counts": dict(Counter(s.label for s in test_ds.subjects)),
        "pool_label_counts": dict(
            Counter([s.label for s in val_ds.subjects] + [s.label for s in test_ds.subjects])
        ),
    }
    return loader, meta


@torch.no_grad()
def _eval_model(model, loader, device) -> dict:
    model.eval()
    by: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, int] = {}
    all_probs: list[np.ndarray] = []
    all_y: list[int] = []
    for batch in loader:
        x = batch["x"].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        y = [int(v) for v in batch["label"]]
        for i, sid in enumerate(batch["subject_id"]):
            sid = str(sid)
            by[sid].append(probs[i])
            labels[sid] = y[i]
            all_probs.append(probs[i])
            all_y.append(y[i])
    ep_probs = np.stack(all_probs, 0)
    ep_y = np.asarray(all_y, dtype=np.int64)
    ep_pred = ep_probs.argmax(axis=-1)
    sids = sorted(by)
    sub_probs = np.stack([np.mean(np.stack(by[s], 0), 0) for s in sids])
    sub_y = np.asarray([labels[s] for s in sids], dtype=np.int64)
    sub_pred = sub_probs.argmax(axis=-1)
    af = (sub_y == 1) | (sub_y == 2)
    recalls = {}
    for c, name in CLINICAL_ID_TO_LABEL.items():
        m = sub_y == c
        recalls[name] = float((sub_pred[m] == sub_y[m]).mean()) if m.any() else float("nan")
    conf = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(sub_y, sub_pred):
        conf[int(t), int(p)] += 1
    return {
        "n_subjects": len(sids),
        "n_epochs": int(len(ep_y)),
        "subject_accuracy": float((sub_pred == sub_y).mean()),
        "subject_auc_macro": float(_macro_auc(sub_y, sub_probs, 3)),
        "ad_ftd_subject_accuracy": float((sub_pred[af] == sub_y[af]).mean()) if af.any() else float("nan"),
        "n_ad_ftd_subjects": int(af.sum()),
        "epoch_accuracy": float((ep_pred == ep_y).mean()),
        "epoch_auc_macro": float(_safe_auc(ep_y, ep_probs, 3).get("auc_macro", float("nan"))),
        "epoch_macro_f1": float(_macro_f1(ep_y, ep_pred, 3)),
        "recalls": recalls,
        "confusion_hc_ftd_ad": conf.tolist(),
        "subject_label_counts": {
            CLINICAL_ID_TO_LABEL[c]: int((sub_y == c).sum()) for c in range(3)
        },
    }


def _load_any(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    arch = str(ckpt.get("arch") or a.get("arch") or "")
    backend = str(ckpt.get("backend") or a.get("backend") or "")
    model_name = str(ckpt.get("model_name") or a.get("model") or "")
    sd = ckpt.get("state_dict", {})

    # Native / aniso HNF
    if (
        "native" in arch
        or "aniso" in arch
        or "eeg_hnf" in arch
        or any(k.startswith("theta.stack.") for k in sd)
        or any(k.startswith("spatial.") for k in sd)
    ):
        model, args, name = _load_model(ckpt_path, device)
        return model, args, name or arch or "HNF-native"

    # Stage-1 temporal HNF (EEGHNFClassifier)
    if any(k.startswith("encoder.branches.") for k in sd) or any(
        k.startswith("channel_embed.proj.") for k in sd
    ):
        from hnf.eeg_model import EEGHNFClassifier

        sample_rate = int(a.get("sample_rate", 128))
        epoch_sec = float(a.get("epoch_sec", 10.0))
        embed_dim = int(a.get("embed_dim", 64))
        principle = str(a.get("principle", "huygens_fresnel"))
        dropout = float(a.get("dropout", 0.2))
        model = EEGHNFClassifier(
            n_channels=19,
            seq_len=int(round(epoch_sec * sample_rate)),
            sample_rate=sample_rate,
            embed_dim=embed_dim,
            num_classes=len(LABEL_TO_ID),
            dropout=dropout,
            principle=principle,
        ).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, a, "HNF Stage-1"

    sample_rate = int(a.get("sample_rate", 128))
    epoch_sec = float(a.get("epoch_sec", 10.0))
    dropout = float(a.get("dropout", 0.25))
    n_samples = int(round(epoch_sec * sample_rate))

    if backend == "braindecode" or model_name.lower().replace("_", "").replace("-", "") in {
        "eegnetv4",
        "shallowfbcsp",
        "deep4net",
        "eegconformer",
    }:
        extra = {}
        if a.get("att_dropout") is not None:
            extra["att_drop_prob"] = float(a["att_dropout"])
        model = build_braindecode_model(
            model_name or "eegnetv4",
            n_channels=19,
            n_samples=n_samples,
            n_classes=len(LABEL_TO_ID),
            dropout=dropout,
            extra=extra,
        ).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, a, display_name(model_name)

    # in-house baselines
    name = model_name or "eegnet"
    model = build_eeg_baseline(
        name,
        n_channels=19,
        n_samples=n_samples,
        n_classes=len(LABEL_TO_ID),
        dropout=dropout,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, a, name


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("HNF aniso phase_off", "outputs/eeg/aniso_diffusion_ablation/phase_off/best.pt"),
        ("HNF aniso phase_on", "outputs/eeg/aniso_diffusion_ablation/phase_on/best.pt"),
        ("HNF native v3", "outputs/eeg/adftd_hnf_native_v3/best.pt"),
        ("HNF native v5", "outputs/eeg/adftd_hnf_native_v5/best.pt"),
        ("HNF Stage-1", "outputs/eeg/adftd_hnf_stage1/best.pt"),
        ("EEGNetv4 (Braindecode)", "outputs/eeg/adftd_braindecode_sota/eegnetv4/best_tuned/best.pt"),
        ("ShallowFBCSPNet", "outputs/eeg/adftd_braindecode_sota/shallowfbcsp/best_tuned/best.pt"),
        ("Deep4Net", "outputs/eeg/adftd_braindecode_sota/deep4net/best_tuned/best.pt"),
        ("EEG Conformer", "outputs/eeg/adftd_braindecode_sota/eegconformer/best_tuned/best.pt"),
        ("EEGNet (in-house tuned)", "outputs/eeg/adftd_sota_tuned/eegnet/best_tuned/best.pt"),
        ("Conformer (in-house tuned)", "outputs/eeg/adftd_sota_tuned/conformer/best_tuned/best.pt"),
    ]

    # Shared pool geometry from first existing ckpt args
    sample_rate, epoch_sec = 128, 10.0
    rows = []
    pool_meta = None
    for display, ckpt in jobs:
        path = Path(ckpt)
        if not path.exists():
            print(f"[skip] missing {ckpt}", flush=True)
            continue
        print(f"[eval] {display} ← {ckpt}", flush=True)
        model, ckpt_args, resolved = _load_any(path, device)
        sr = int(ckpt_args.get("sample_rate", sample_rate))
        es = float(ckpt_args.get("epoch_sec", epoch_sec))
        # Match prior classification boards: overlapping 10s / stride 5s
        loader, meta = _pool_loaders(
            data_dir=args.data_dir,
            seed=args.seed,
            sample_rate=sr,
            epoch_sec=es,
            batch_size=args.batch_size,
            synthetic_if_missing=not args.no_synthetic,
            overlap=True,
        )
        if pool_meta is None:
            pool_meta = meta
            print(f"[pool] subjects={meta['n_pool_subjects']} "
                  f"(val {meta['n_val_subjects']}+test {meta['n_test_subjects']}) "
                  f"epochs={meta['n_pool_epochs']} labels={meta['pool_label_counts']}",
                  flush=True)
        metrics = _eval_model(model, loader, device)
        row = {
            "model": display,
            "resolved_name": resolved,
            "checkpoint": ckpt,
            **metrics,
        }
        rows.append(row)
        print(
            f"  subject_acc={metrics['subject_accuracy']:.3f} "
            f"adftd={metrics['ad_ftd_subject_accuracy']:.3f} "
            f"subj_auc={metrics['subject_auc_macro']:.3f} "
            f"epoch_auc={metrics['epoch_auc_macro']:.3f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows_sorted = sorted(rows, key=lambda r: (-r["subject_accuracy"], -r["subject_auc_macro"]))
    board = {
        "protocol": {
            "dataset": "OpenNeuro ds004504",
            "eval_set": "val + test (subject-level pool)",
            "train_excluded": True,
            "seed": args.seed,
            "caveat": (
                "Models were often selected on val; pooling val+test is larger-N "
                "but mildly optimistic vs pure test-only."
            ),
            "pool": pool_meta,
            "label_names": {str(k): v for k, v in CLINICAL_ID_TO_LABEL.items()},
        },
        "rows": rows_sorted,
    }
    (out / "VALTEST_POOL_BOARD.json").write_text(json.dumps(board, indent=2), encoding="utf-8")

    md = [
        "# EEG re-eval: val + test as holdout pool",
        "",
        f"- Subjects: **{pool_meta['n_pool_subjects']}** "
        f"(val {pool_meta['n_val_subjects']} + test {pool_meta['n_test_subjects']})",
        f"- Epochs (10s, stride 5s): **{pool_meta['n_pool_epochs']}**",
        f"- Pool label counts (HC/FTD/AD ids 0/1/2): `{pool_meta['pool_label_counts']}`",
        "- Train subjects excluded.",
        "- **Caveat:** ckpts often selected on val → mildly optimistic vs test-only.",
        "",
        "| Model | subject acc | AD↔FTD | subject AUC | epoch acc | epoch AUC | n_subj |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        md.append(
            f"| {r['model']} | {r['subject_accuracy']:.3f} | "
            f"{r['ad_ftd_subject_accuracy']:.3f} | {r['subject_auc_macro']:.3f} | "
            f"{r['epoch_accuracy']:.3f} | {r['epoch_auc_macro']:.3f} | {r['n_subjects']} |"
        )
    md.append("")
    (out / "VALTEST_POOL_BOARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"top": rows_sorted[0] if rows_sorted else None, "n_models": len(rows_sorted)}, indent=2))
    print(f"[done] → {out / 'VALTEST_POOL_BOARD.md'}", flush=True)


if __name__ == "__main__":
    main()
