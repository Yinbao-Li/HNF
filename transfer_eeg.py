# -*- coding: utf-8 -*-
"""Few-shot seismic→EEG transfer vs from-scratch baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from hnf.eeg_dataset import EEGDataset, LABEL_TO_ID
from hnf.eeg_model import EEGHNFClassifier
from train_eeg import evaluate, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seismic→EEG transfer few-shot study")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument(
        "--seismic-ckpt",
        default="outputs/run20/20_wrongpeak_sharp/best.pt",
        help="STEAD picking checkpoint for encoder init",
    )
    p.add_argument("--output-dir", default="outputs/eeg/transfer_fewshot")
    p.add_argument("--shots", default="5,10,20", help="Subjects per class")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--synthetic-if-missing", action="store_true", default=True)
    return p.parse_args()


def _indices_by_subject_budget(
    ds: EEGDataset,
    shots_per_class: int,
    seed: int,
) -> list[int]:
    """Pick up to ``shots_per_class`` subjects per label, keep all their epochs."""
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for ref in ds.subjects:
        by_label[ref.label].append(ref.subject_id)
    chosen: set[str] = set()
    for lab, sids in by_label.items():
        sids = list(sids)
        rng.shuffle(sids)
        chosen.update(sids[:shots_per_class])
    idx = [i for i, (ref, _) in enumerate(ds.epochs) if ref.subject_id in chosen]
    return idx


def _train_one(
    model: EEGHNFClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    best_auc = -1.0
    best_metrics: dict[str, float] = {}
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        metrics = evaluate(model, val_loader, device)
        score = metrics["auc"] if metrics["auc"] == metrics["auc"] else metrics["acc"]
        if score >= best_auc:
            best_auc = score
            best_metrics = metrics
    return best_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    shots_list = [int(x) for x in args.shots.split(",") if x.strip()]

    common = dict(
        data_dir=args.data_dir,
        seed=args.seed,
        synthetic_if_missing=args.synthetic_if_missing,
        stride_sec=5.0,
    )
    train_ds = EEGDataset(split="train", **common)
    val_ds = EEGDataset(split="val", **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    seismic_state = None
    seismic_path = Path(args.seismic_ckpt)
    if seismic_path.is_file():
        seismic_state = torch.load(seismic_path, map_location="cpu", weights_only=False)["state_dict"]
        print(f"[transfer] loaded seismic ckpt {seismic_path}", flush=True)
    else:
        print(f"[transfer] WARN missing seismic ckpt {seismic_path}; transfer=encoder-random", flush=True)

    results = []
    for shots in shots_list:
        idx = _indices_by_subject_budget(train_ds, shots, seed=args.seed + shots)
        subset = Subset(train_ds, idx)
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

        # from-scratch
        scratch = EEGHNFClassifier().to(device)
        m_scratch = _train_one(scratch, train_loader, val_loader, device, args.epochs, args.lr)

        # transfer: load encoder, freeze kernels, train embed+head (+ optionally density)
        transfer = EEGHNFClassifier().to(device)
        if seismic_state is not None:
            loaded, skipped = transfer.load_seismic_encoder(seismic_state)
            print(f"[transfer] shots={shots} loaded={len(loaded)} skipped={len(skipped)}", flush=True)
        n_frozen = transfer.freeze_huygens_kernels()
        # Also freeze encoder non-kernel weights for true linear-probe-ish transfer
        for name, p in transfer.named_parameters():
            if name.startswith("encoder."):
                p.requires_grad = False
        # Re-enable only channel_embed + medium_net + head
        for name, p in transfer.named_parameters():
            if name.startswith("channel_embed.") or name.startswith("medium_net.") or name.startswith("head."):
                p.requires_grad = True
        m_transfer = _train_one(transfer, train_loader, val_loader, device, args.epochs, args.lr)

        row = {
            "shots_per_class": shots,
            "n_train_epochs": len(idx),
            "scratch_val_acc": m_scratch.get("acc"),
            "scratch_val_auc": m_scratch.get("auc"),
            "transfer_val_acc": m_transfer.get("acc"),
            "transfer_val_auc": m_transfer.get("auc"),
            "n_frozen_kernel_params": n_frozen,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    out_json = out / "fewshot_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"[transfer] wrote {out_json}")


if __name__ == "__main__":
    main()
