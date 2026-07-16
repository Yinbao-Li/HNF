#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched OBS light-adapt for SeisBench EQT / PhaseNet (STEAD init).

Same split/budget as tools/train_obs_light_adapt.py:
  seed, chunks, max-events, val-frac, epochs, p_offset, pick-only eval.

Treatment: freeze trunk, tune pick/decoder heads only (det branch frozen / unused in loss).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import importlib.util
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hnf.stead_picking_dataset import gaussian_pick_label
from tools.train_stead_picking import set_seed
from tools.obs_matched_split import load_split_samples


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Light-adapt EQT/PhaseNet on OBS")
    p.add_argument("--model", choices=["eqt", "phasenet"], required=True)
    p.add_argument("--weights", default="stead", help="SeisBench pretrained tag")
    p.add_argument("--output-dir", default="")
    p.add_argument("--chunks", default="201805,201806,201807,201808")
    p.add_argument("--max-events", type=int, default=2400)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--p-offset-sec", type=float, default=8.0)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--pick-pos-weight", type=float, default=28.0)
    p.add_argument("--p-loss-weight", type=float, default=1.5)
    p.add_argument("--s-loss-weight", type=float, default=1.2)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument(
        "--tune",
        default="heads",
        choices=["heads", "all"],
        help="heads: EQT pick_*; PhaseNet last up_branch + out",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument(
        "--split-json",
        default="",
        help="Disjoint train/holdout keys from tools/obs_matched_split.py",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    return p.parse_args()


def load_sb_model(kind: str, weights: str):
    import seisbench.models as sbm

    if kind == "eqt":
        m = sbm.EQTransformer.from_pretrained(weights)
        return m, "eqt", 3, getattr(m, "norm", None) or "peak"
    m = sbm.PhaseNet.from_pretrained(weights)
    return m, "phasenet", 3, getattr(m, "norm", None) or "peak"


def freeze_for_adapt(model: torch.nn.Module, kind: str, tune: str) -> list[str]:
    for p in model.parameters():
        p.requires_grad = False
    if tune == "all":
        for p in model.parameters():
            p.requires_grad = True
        return ["*"]
    trained = []
    if kind == "eqt":
        prefixes = ("pick_lstms.", "pick_attentions.", "pick_decoders.", "pick_convs.")
    else:
        # PhaseNet has no separate pick head; last upsample + classifier ≈ heads.
        prefixes = ("up_branch.3.", "out.")
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad = True
            trained.append(name)
    return trained


class ItemDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _pad_or_crop(x: torch.Tensor, target_len: int) -> torch.Tensor:
    # x: (C, T)
    t = x.shape[-1]
    if t == target_len:
        return x
    if t > target_len:
        start = (t - target_len) // 2
        return x[:, start : start + target_len]
    return F.pad(x, (0, target_len - t))


def build_items(
    samples: list[dict],
    kind: str,
    in_samples: int,
    window_sec: float,
    label_sigma_sec: float,
    normalize_wave,
    norm_mode: str,
) -> list[dict]:
    sigma = max(1.0, label_sigma_sec * in_samples / window_sec)
    items = []
    for s in samples:
        raw = s["wave_3_raw"]
        x = torch.from_numpy(normalize_wave(raw, norm_mode)).float()
        # Resample to model sampling grid length (EQT: 6000 @ 100 Hz for 60 s).
        if x.shape[-1] != in_samples:
            x = F.interpolate(
                x.unsqueeze(0), size=in_samples, mode="linear", align_corners=False
            ).squeeze(0)
        x = _pad_or_crop(x, in_samples)
        scale = in_samples / float(raw.shape[-1])
        p_idx = int(max(0, min(in_samples - 1, round(s["p_idx_native"] * scale))))
        p_target = gaussian_pick_label(p_idx, in_samples, sigma)
        if s["s_valid"]:
            s_idx = int(max(0, min(in_samples - 1, round(s["s_idx_native"] * scale))))
            s_target = gaussian_pick_label(s_idx, in_samples, sigma)
            s_valid = 1.0
        else:
            s_idx = 0
            s_target = torch.zeros(in_samples, dtype=torch.float32)
            s_valid = 0.0
        items.append(
            {
                "x": x,
                "p_target": p_target,
                "s_target": s_target,
                "p_idx": p_idx,
                "s_idx": s_idx,
                "p_valid": 1.0,
                "s_valid": s_valid,
            }
        )
    return items


def collate(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "p_target": torch.stack([b["p_target"] for b in batch]),
        "s_target": torch.stack([b["s_target"] for b in batch]),
        "p_idx": torch.tensor([b["p_idx"] for b in batch], dtype=torch.long),
        "s_idx": torch.tensor([b["s_idx"] for b in batch], dtype=torch.long),
        "p_valid": torch.tensor([b["p_valid"] for b in batch], dtype=torch.float32),
        "s_valid": torch.tensor([b["s_valid"] for b in batch], dtype=torch.float32),
    }


def _weighted_bce_prob(prob: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    weight = torch.where(target > 0.05, pos_weight, 1.0)
    bce = F.binary_cross_entropy(prob, target, reduction="none")
    return (bce * weight).mean()


def _weighted_bce_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = torch.where(target > 0.05, pos_weight, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weight).mean()


def adapt_loss(p_pred, s_pred, batch, args, kind: str) -> tuple[torch.Tensor, dict]:
    if kind == "eqt":
        loss_p = _weighted_bce_prob(p_pred, batch["p_target"], args.pick_pos_weight)
        s_mask = batch["s_valid"] > 0.5
        if s_mask.any():
            loss_s = _weighted_bce_prob(
                s_pred[s_mask], batch["s_target"][s_mask], args.pick_pos_weight
            )
        else:
            loss_s = p_pred.new_zeros(())
    else:
        # PhaseNet logits (P, S channels); Noise unsupervised aside from softmax coupling.
        loss_p = _weighted_bce_logits(p_pred, batch["p_target"], args.pick_pos_weight)
        s_mask = batch["s_valid"] > 0.5
        if s_mask.any():
            loss_s = _weighted_bce_logits(
                s_pred[s_mask], batch["s_target"][s_mask], args.pick_pos_weight
            )
        else:
            loss_s = p_pred.new_zeros(())
    loss = args.p_loss_weight * loss_p + args.s_loss_weight * loss_s
    return loss, {
        "loss": float(loss.detach()),
        "loss_p": float(loss_p.detach()),
        "loss_s": float(loss_s.detach()),
    }


def forward_picks(model, x, kind: str):
    if kind == "eqt":
        _det, p, s = model(x)
        if p.dim() == 3:
            p = p.squeeze(1)
        if s.dim() == 3:
            s = s.squeeze(1)
        return p, s
    # PhaseNet: train on logits for BCE-with-logits; eval uses softmax probs.
    out = model(x, logits=True)
    labels = "".join(getattr(model, "labels", "PSN") or "PSN").upper()
    if labels.startswith("PS"):
        return out[:, 0], out[:, 1]
    if labels.startswith("NP"):
        return out[:, 1], out[:, 2]
    raise RuntimeError(f"Unexpected PhaseNet labels {labels}")


@torch.no_grad()
def eval_pick_only(model, samples, device, args, obs_mod, kind: str, n_channels: int, norm_mode: str):
    model.eval()
    return obs_mod.eval_seisbench(
        model,
        samples,
        device,
        args.pick_threshold,
        args.det_threshold,
        args.tol_sec,
        args.batch_size,
        kind=kind,
        n_channels=n_channels,
        norm_mode=norm_mode,
    )["pick_only"]


def main() -> None:
    args = parse_args()
    if not args.output_dir:
        args.output_dir = f"outputs/obs_light_adapt_{args.model}_offset8"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    obs_mod = _load_obs_compare_module()
    chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
    print(f"[sb-adapt] model={args.model} chunks={chunks} device={device}", flush=True)
    holdout_samples = []
    if args.split_json.strip():
        samples, load_info, split_meta = load_split_samples(args.split_json.strip(), "train")
        holdout_samples, hold_info, _ = load_split_samples(args.split_json.strip(), "holdout")
        load_info = {
            **load_info,
            "holdout_info": hold_info,
            "split_json": args.split_json.strip(),
            "p_offset_min": split_meta.get("p_offset_min"),
            "p_offset_max": split_meta.get("p_offset_max"),
            "protocol": split_meta.get("protocol"),
        }
        print(
            f"[sb-adapt] disjoint split {args.split_json}: "
            f"train_pool={len(samples)} holdout={len(holdout_samples)} "
            f"offset=[{split_meta.get('p_offset_min')},{split_meta.get('p_offset_max')}]",
            flush=True,
        )
    else:
        samples, load_info = obs_mod.load_obs_windows(
            chunks,
            args.max_events,
            args.window_sec,
            args.p_offset_sec,
            args.seed,
            require_full_3c=True,
        )
    print(
        f"[sb-adapt] n={len(samples)} with_S={sum(1 for s in samples if s['s_valid'])} "
        f"info={load_info}",
        flush=True,
    )
    if len(samples) < 50:
        raise RuntimeError("Too few OBS samples for adapt")

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    n_val = max(32, int(round(len(samples) * args.val_frac)))
    val_idx = set(idxs[:n_val].tolist())
    train_samples = [samples[i] for i in idxs if i not in val_idx]
    val_samples = [samples[i] for i in sorted(val_idx)]

    model, kind, n_channels, norm_mode = load_sb_model(args.model, args.weights)
    model = model.to(device)
    native_len = int(samples[0]["wave_3_raw"].shape[-1])
    if kind == "eqt":
        # EQT asserts exact in_samples (6000 for 60s@100Hz).
        in_samples = int(model.in_samples)
    else:
        # PhaseNet accepts variable length; match eval_seisbench native windows.
        in_samples = native_len

    train_items = build_items(
        train_samples,
        kind,
        in_samples,
        args.window_sec,
        args.label_sigma_sec,
        obs_mod.normalize_wave,
        norm_mode,
    )
    train_loader = DataLoader(
        ItemDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    trained_names = freeze_for_adapt(model, kind, args.tune)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"[sb-adapt] tune={args.tune} trainable={n_train}/{n_all} "
        f"groups={len(trained_names)} in_samples={in_samples} norm={norm_mode}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    base_metrics = eval_pick_only(
        model, val_samples, device, args, obs_mod, kind, n_channels, norm_mode
    )
    print(f"[sb-adapt] val zero-shot pick_only={base_metrics}", flush=True)

    history = []
    best = {"val_score": -1.0, "epoch": -1, "metrics": None}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.eval()  # freeze BN/dropout in trunk
        for name, mod in model.named_modules():
            if any(name.startswith(pref.rstrip(".")) for pref in (
                ("pick_lstms", "pick_attentions", "pick_decoders", "pick_convs")
                if kind == "eqt"
                else ("up_branch.3", "out")
            )):
                mod.train()
        run = {"loss": 0.0, "loss_p": 0.0, "loss_s": 0.0, "n": 0}
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            p_pred, s_pred = forward_picks(model, batch["x"], kind)
            loss, stats = adapt_loss(p_pred, s_pred, batch, args, kind)
            if not torch.isfinite(loss):
                print(f"[sb-adapt] skip non-finite loss ep{ep} stats={stats}", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            bs = batch["x"].size(0)
            for k in ("loss", "loss_p", "loss_s"):
                run[k] += stats[k] * bs
            run["n"] += bs
        sched.step()
        train_stats = {k: run[k] / max(1, run["n"]) for k in ("loss", "loss_p", "loss_s")}
        row = {"epoch": ep, "train": train_stats, "lr": opt.param_groups[0]["lr"]}
        if ep % args.eval_every == 0 or ep == args.epochs:
            metrics = eval_pick_only(
                model, val_samples, device, args, obs_mod, kind, n_channels, norm_mode
            )
            score = 0.65 * metrics["p_f1"] + 0.35 * metrics["s_f1"]
            row["val_pick_only"] = metrics
            row["val_score"] = score
            print(
                f"[sb-adapt] ep{ep} train_loss={train_stats['loss']:.4f} "
                f"val P={metrics['p_f1']:.3f} S={metrics['s_f1']:.3f} score={score:.3f}",
                flush=True,
            )
            if score > best["val_score"]:
                best = {"val_score": score, "epoch": ep, "metrics": metrics}
                ckpt = {
                    "state_dict": model.state_dict(),
                    "model": args.model,
                    "weights_init": args.weights,
                    "kind": kind,
                    "n_channels": n_channels,
                    "norm_mode": norm_mode,
                    "in_samples": in_samples,
                    "adapt_args": vars(args),
                    "base_val_pick_only": base_metrics,
                    "best_val_pick_only": metrics,
                    "trained_param_names": trained_names,
                    "trainable_params": n_train,
                    "epoch": ep,
                }
                torch.save(ckpt, out_dir / "best.pt")
                torch.save(ckpt, out_dir / "last.pt")
        else:
            print(f"[sb-adapt] ep{ep} train_loss={train_stats['loss']:.4f}", flush=True)
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_path = out_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    eval_samples = holdout_samples if holdout_samples else samples
    eval_name = "holdout_adapted" if holdout_samples else "full_pool_adapted"
    full_metrics = eval_pick_only(
        model, eval_samples, device, args, obs_mod, kind, n_channels, norm_mode
    )
    report = {
        "model": args.model,
        "weights_init": args.weights,
        "best_ckpt": str(best_path),
        "chunks": chunks,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_eval": len(eval_samples),
        "n_full": len(samples),
        "split_json": args.split_json or None,
        "tune": args.tune,
        "trainable_params": n_train,
        "val_zero_shot": base_metrics,
        "val_best": best["metrics"],
        "val_best_epoch": best["epoch"],
        eval_name: full_metrics,
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "load_info": load_info,
        "protocol": {
            "pick_threshold": args.pick_threshold,
            "tol_sec": args.tol_sec,
            "require_full_3c": True,
            "metric": "pick_only",
            "matched_to": "tools/train_obs_light_adapt.py",
            "disjoint_holdout": bool(holdout_samples),
        },
    }
    (out_dir / "adapt_report.json").write_text(json.dumps(report, indent=2))
    md = [
        f"# OBS light-adapt report ({args.model})",
        "",
        f"- init: `{args.weights}`",
        f"- best: `{best_path}` (ep {best['epoch']})",
        f"- trainable: {n_train}",
        f"- val ZS P/S: {base_metrics['p_f1']:.3f} / {base_metrics['s_f1']:.3f}",
        f"- val best P/S: {best['metrics']['p_f1']:.3f} / {best['metrics']['s_f1']:.3f}",
        f"- full-pool adapt P/S: {full_metrics['p_f1']:.3f} / {full_metrics['s_f1']:.3f}",
    ]
    (out_dir / "adapt_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
