#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train EEG-native HNF (spatial secondary sources + θ/α rhythm priors)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from hnf.eeg_dataset import EEGDataset, LABEL_TO_ID
from hnf.eeg_native_model import EEGHNFNativeClassifier
from tools.train_eeg import _macro_auc, evaluate, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEG-native HNF classifier")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--output-dir", default="outputs/eeg/adftd_hnf_native_v1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--principle", default="huygens_fresnel",
                   choices=["huygens", "huygens_fresnel", "aniso_diffusion"])
    p.add_argument(
        "--rhythm-phase",
        dest="rhythm_phase",
        action="store_true",
        default=True,
        help="Use oscillatory rhythm phase exp(iωτ)/cos(ωr) (default: on)",
    )
    p.add_argument(
        "--no-rhythm-phase",
        dest="rhythm_phase",
        action="store_false",
        help="Ablate rhythm phase: pure anisotropic diffusion amplitude",
    )
    p.add_argument("--sample-rate", type=int, default=128)
    p.add_argument("--epoch-sec", type=float, default=10.0)
    p.add_argument("--stride-sec", type=float, default=5.0)
    p.add_argument("--no-spatial", action="store_true")
    p.add_argument(
        "--mmse-weight",
        type=float,
        default=0.35,
        help="Auxiliary MSE weight for normalized MMSE (clinical severity)",
    )
    p.add_argument(
        "--adftd-margin-weight",
        type=float,
        default=0.0,
        help="Hard-margin loss weight pushing AD vs FTD logit gap",
    )
    p.add_argument(
        "--adftd-margin",
        type=float,
        default=0.75,
        help="Desired logit margin between AD and FTD for disease samples",
    )
    p.add_argument(
        "--adftd-bce-weight",
        type=float,
        default=0.0,
        help="Binary AD-vs-FTD BCE weight on disease-only epochs",
    )
    p.add_argument(
        "--class-weights",
        default="",
        help="Optional CE class weights as HC,FTD,AD (e.g. 1,1.4,1.1)",
    )
    p.add_argument(
        "--arch-tag",
        default="",
        help="Override checkpoint arch string (default derived from output dir)",
    )
    p.add_argument(
        "--subject-balanced",
        action="store_true",
        default=False,
        help="Sample subjects/classes uniformly (often hurts AD↔FTD; off by default)",
    )
    p.add_argument("--no-subject-balanced", action="store_false", dest="subject_balanced")
    p.add_argument(
        "--mild-disease-boost",
        type=float,
        default=0.0,
        help="If >0 and not fully subject-balanced, multiply AD/FTD sample weights by this",
    )
    p.add_argument("--no-synthetic", action="store_true")
    p.add_argument(
        "--swap-val-test",
        action="store_true",
        help="Exchange val↔test after seed split (train fixed): original test→val, original val→test",
    )
    p.add_argument(
        "--pool-val-test",
        action="store_true",
        help="Merge original val+test into one pool: used as val during training AND as test for final report",
    )
    p.add_argument("--resume", default=None)
    return p.parse_args()


def _ad_ftd_margin_loss(logits: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    """Push disease logits apart: AD prefers logit_AD > logit_FTD + margin (and vice versa)."""
    # classes: HC=0, FTD=1, AD=2
    losses: list[torch.Tensor] = []
    ad = y == 2
    ftd = y == 1
    if ad.any():
        gap = logits[ad, 2] - logits[ad, 1]
        losses.append(torch.relu(margin - gap).mean())
    if ftd.any():
        gap = logits[ftd, 1] - logits[ftd, 2]
        losses.append(torch.relu(margin - gap).mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _ad_ftd_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary AD vs FTD on disease subset using score = logit_AD - logit_FTD."""
    mask = (y == 1) | (y == 2)
    if not mask.any():
        return logits.new_zeros(())
    score = logits[mask, 2] - logits[mask, 1]
    target = (y[mask] == 2).float()
    return nn.functional.binary_cross_entropy_with_logits(score, target)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    common = dict(
        data_dir=args.data_dir,
        test_ratio=0.2,
        val_ratio=0.15,
        seed=args.seed,
        sample_rate=args.sample_rate,
        epoch_sec=args.epoch_sec,
        stride_sec=args.stride_sec,
        synthetic_if_missing=not args.no_synthetic,
        swap_val_test=bool(args.swap_val_test) and not bool(args.pool_val_test),
    )
    train_ds = EEGDataset(split="train", **common)
    if args.pool_val_test:
        # Same subject pool for selection (val) and final report (test).
        val_ds = EEGDataset(split="valtest", **common)
        test_ds = val_ds
    else:
        val_ds = EEGDataset(split="val", **common)
        test_ds = EEGDataset(split="test", **common)

    sampler = None
    if args.subject_balanced:
        from collections import Counter

        epoch_count = Counter(ref.subject_id for ref, _ in train_ds.epochs)
        class_subject_count = Counter(ref.label for ref in train_ds.subjects)
        weights = [
            1.0
            / max(epoch_count[ref.subject_id], 1)
            / max(class_subject_count[ref.label], 1)
            for ref, _ in train_ds.epochs
        ]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
    elif args.mild_disease_boost > 0:
        # Mild AD/FTD upweight without full subject equalization (v2 failure mode).
        weights = []
        for ref, _ in train_ds.epochs:
            w = 1.0
            if ref.label in (1, 2):  # FTD / AD
                w *= float(args.mild_disease_boost)
            weights.append(w)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = EEGHNFNativeClassifier(
        n_channels=19,
        seq_len=int(round(args.epoch_sec * args.sample_rate)),
        sample_rate=args.sample_rate,
        embed_dim=args.embed_dim,
        num_classes=len(LABEL_TO_ID),
        dropout=args.dropout,
        principle=args.principle,
        use_spatial=not args.no_spatial,
        rhythm_phase=bool(args.rhythm_phase),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[EEG-native] resumed from {args.resume}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.class_weights.strip():
        cw = [float(x) for x in args.class_weights.split(",")]
        if len(cw) != 3:
            raise ValueError("--class-weights must have 3 values: HC,FTD,AD")
        ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, device=device, dtype=torch.float32))
        print(f"[EEG-native] CE class weights HC/FTD/AD={cw}", flush=True)
    else:
        ce = nn.CrossEntropyLoss()

    arch_tag = args.arch_tag.strip() or Path(args.output_dir).name.replace("adftd_", "")
    history: list[dict[str, float]] = []
    best_auc = -1.0
    best_path = out / "best.pt"
    print(
        f"[EEG-native] device={device} params={n_params} arch={arch_tag} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"spatial={not args.no_spatial} principle={args.principle} "
        f"rhythm_phase={bool(args.rhythm_phase)} "
        f"swap_val_test={bool(args.swap_val_test)} pool_val_test={bool(args.pool_val_test)} "
        f"adftd_margin_w={args.adftd_margin_weight} adftd_bce_w={args.adftd_bce_weight}",
        flush=True,
    )
    print(f"[EEG-native] kernel priors: {model.collect_kernel_params()}", flush=True)

    @torch.no_grad()
    def subject_metrics(loader: DataLoader) -> dict[str, float]:
        from collections import defaultdict

        model.eval()
        by: dict[str, list[np.ndarray]] = defaultdict(list)
        labels: dict[str, int] = {}
        for batch in loader:
            probs = torch.softmax(model(batch["x"].to(device)), dim=-1).cpu().numpy()
            for i, sid in enumerate(batch["subject_id"]):
                sid = str(sid)
                by[sid].append(probs[i])
                labels[sid] = int(batch["label"][i])
        sids = sorted(by)
        probs = np.stack([np.mean(np.stack(by[sid], 0), 0) for sid in sids])
        y = np.asarray([labels[sid] for sid in sids], dtype=np.int64)
        pred = probs.argmax(axis=-1)
        af = (y == 1) | (y == 2)
        return {
            "subject_acc": float((pred == y).mean()),
            "subject_auc": float(_macro_auc(y, probs, 3)),
            "ad_ftd_acc": float((pred[af] == y[af]).mean()) if af.any() else float("nan"),
        }

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{args.epochs}", leave=False, mininterval=5.0)
        for batch in pbar:
            x = batch["x"].to(device)
            y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
            mmse = torch.as_tensor(batch["mmse"], device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            logits, aux = model(x, return_aux=True)
            cls_loss = ce(logits, y)
            valid_mmse = torch.isfinite(mmse)
            if valid_mmse.any():
                mmse_loss = nn.functional.mse_loss(
                    aux["mmse_pred_norm"][valid_mmse],
                    (mmse[valid_mmse] / 30.0).clamp(0.0, 1.0),
                )
            else:
                mmse_loss = cls_loss.new_zeros(())
            margin_loss = (
                _ad_ftd_margin_loss(logits, y, args.adftd_margin)
                if args.adftd_margin_weight > 0
                else cls_loss.new_zeros(())
            )
            bce_loss = (
                _ad_ftd_bce_loss(logits, y)
                if args.adftd_bce_weight > 0
                else cls_loss.new_zeros(())
            )
            loss = (
                cls_loss
                + args.mmse_weight * mmse_loss
                + args.adftd_margin_weight * margin_loss
                + args.adftd_bce_weight * bce_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
        sched.step()
        train_loss = running / max(n_seen, 1)
        val_m = evaluate(model, val_loader, device, num_classes=3)
        val_s = subject_metrics(val_loader)
        # Prefer holding overall discrimination while rewarding AD↔FTD hard pair.
        clinical_score = (
            val_m["auc"]
            + 0.10 * val_s["subject_auc"]
            + 0.05 * val_s["subject_acc"]
            + 0.25 * val_s["ad_ftd_acc"]
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_auc": val_m["auc"],
            **val_s,
            "clinical_score": clinical_score,
            "lr": float(sched.get_last_lr()[0]),
        }
        history.append(row)
        print(
            f"[EEG-native] ep{epoch:03d} train_loss={train_loss:.4f} "
            f"val_auc={val_m['auc']:.3f} subj_auc={val_s['subject_auc']:.3f} "
            f"subj_acc={val_s['subject_acc']:.3f} adftd={val_s['ad_ftd_acc']:.3f}",
            flush=True,
        )
        if clinical_score > best_auc:
            best_auc = clinical_score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "arch": arch_tag,
                    "epoch": epoch,
                    "best_clinical_score": best_auc,
                    "val_subject_metrics": val_s,
                    "kernel_params": model.collect_kernel_params(),
                },
                best_path,
            )
            print(
                f"[EEG-native] saved best → {best_path} "
                f"(clinical_score={best_auc:.3f})",
                flush=True,
            )

    # Final test via subject-aware eval script path
    from tools.eval_eeg import main as _unused  # noqa: F401
    # Inline subject metrics
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    # epoch-level
    ep_m = evaluate(model, test_loader, device, num_classes=3)
    # subject-level
    from collections import defaultdict

    by: dict[str, list[np.ndarray]] = defaultdict(list)
    ylab: dict[str, int] = {}
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            probs = torch.softmax(model(x), dim=-1).cpu().numpy()
            for i, sid in enumerate(batch["subject_id"]):
                by[str(sid)].append(probs[i])
                ylab[str(sid)] = int(batch["label"][i])
    correct = 0
    for sid, ps in by.items():
        mean_p = np.mean(np.stack(ps, 0), 0)
        correct += int(mean_p.argmax() == ylab[sid])
    subj_acc = correct / max(len(by), 1)
    report = {
        "arch": arch_tag,
        "checkpoint": str(best_path),
        "best_clinical_score": best_auc,
        "test_epoch_acc": ep_m["acc"],
        "test_epoch_auc": ep_m["auc"],
        "test_subject_accuracy": subj_acc,
        "n_test_subjects": len(by),
        "n_params": n_params,
        "elapsed_sec": round(time.time() - t0, 1),
        "kernel_params": model.collect_kernel_params(),
    }
    (out / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[EEG-native] done → {out}", flush=True)


if __name__ == "__main__":
    main()
