#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBS light-adapt: freeze STEAD backbone, fine-tune pick/det heads on OBS.

Step 4 follow-up (EXPERIMENT_PLAN). Train on multi-chunk SeisBench OBS windows,
evaluate pick-only F1 with the same protocol as run_paper_obs_picking_compare.
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
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hnf.stead_picking_dataset import gaussian_pick_label
from tools.analyze_stead_picking import load_model
from tools.train_stead_picking import move_batch_to_device, set_seed, weighted_pick_loss
from tools.obs_matched_split import load_split_samples


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Light-adapt HNF picking on OBS")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/obs_light_adapt_run28")
    p.add_argument("--chunks", default="201805,201806,201807,201808")
    p.add_argument("--max-events", type=int, default=2400)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument(
        "--force-sparse-band",
        action="store_true",
        help="Force HuygensKernel.sparse_band=True (needed for long seq_len)",
    )
    p.add_argument(
        "--cap-local-window-sec",
        type=float,
        default=0.0,
        help="If >0, cap every kernel local_window_sec (e.g. 3.0 for seq_len=6000)",
    )
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument(
        "--p-offset-sec",
        type=float,
        default=8.0,
        help="Match STEAD P prior (~6–8s into 60s window); avoid 15s absolute-time mismatch.",
    )
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--pick-pos-weight", type=float, default=28.0)
    p.add_argument("--p-loss-weight", type=float, default=1.5)
    p.add_argument("--s-loss-weight", type=float, default=1.2)
    p.add_argument("--det-loss-weight", type=float, default=0.0)
    p.add_argument("--pick-threshold", type=float, default=0.25)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument(
        "--tune",
        default="heads",
        choices=["heads", "heads+noise", "heads+onset", "trunk-tail", "all"],
        help="heads / heads+noise / heads+onset / trunk-tail(all p/s layers) / all",
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


def freeze_for_adapt(model: torch.nn.Module, tune: str) -> list[str]:
    for p in model.parameters():
        p.requires_grad = False
    if tune == "all":
        for p in model.parameters():
            p.requires_grad = True
        return ["*"]
    allow_prefixes = {
        "heads": ("p_pick_head.", "s_pick_head."),
        "heads+noise": (
            "p_pick_head.",
            "s_pick_head.",
            "det_head.",
            "noise_cancel_branch.",
            "noise_cue_adapter.",
        ),
        "heads+onset": (
            "p_pick_head.",
            "s_pick_head.",
            "det_head.",
            "raw_onset_encoder.",
            "noise_cancel_branch.",
            "noise_cue_adapter.",
            "p_layers.1.",
            "s_layers.1.",
        ),
        "trunk-tail": (
            "p_pick_head.",
            "s_pick_head.",
            "det_head.",
            "raw_onset_encoder.",
            "noise_cancel_branch.",
            "noise_cue_adapter.",
            "p_layers.",
            "s_layers.",
            "source_embed.",
        ),
    }[tune]
    trained = []
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in allow_prefixes):
            p.requires_grad = True
            trained.append(name)
    return trained


def configure_long_sequence(model: torch.nn.Module, force_sparse: bool, cap_local: float) -> dict:
    """Make seq_len≈6000 feasible on ~12GB GPUs via sparse band + shorter light-cone."""
    n_sparse = 0
    n_capped = 0
    for mod in model.modules():
        if force_sparse and hasattr(mod, "sparse_band"):
            mod.sparse_band = True
            n_sparse += 1
        if cap_local > 0 and hasattr(mod, "local_window_sec") and mod.local_window_sec is not None:
            before = float(mod.local_window_sec)
            mod.local_window_sec = min(before, float(cap_local))
            if mod.local_window_sec < before:
                n_capped += 1
    return {"n_sparse": n_sparse, "n_capped": n_capped, "cap_local_window_sec": cap_local}


class ItemDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def build_items(
    samples: list[dict],
    seq_len: int,
    window_sec: float,
    label_sigma_sec: float,
    normalize_wave,
) -> list[dict]:
    sigma = max(1.0, label_sigma_sec * seq_len / window_sec)
    items = []
    for s in samples:
        x = torch.from_numpy(normalize_wave(s["wave_3_raw"], "std")).float()  # (3, Tn)
        x = F.interpolate(x.unsqueeze(0), size=seq_len, mode="linear", align_corners=False).squeeze(0)
        x = x.transpose(0, 1)  # (T, 3)
        t = torch.linspace(0.0, window_sec, seq_len).unsqueeze(-1)
        scale = seq_len / float(s["wave_3_raw"].shape[-1])
        p_idx = int(round(s["p_idx_native"] * scale))
        p_idx = int(max(0, min(seq_len - 1, p_idx)))
        p_target = gaussian_pick_label(p_idx, seq_len, sigma)
        if s["s_valid"]:
            s_idx = int(round(s["s_idx_native"] * scale))
            s_idx = int(max(0, min(seq_len - 1, s_idx)))
            s_target = gaussian_pick_label(s_idx, seq_len, sigma)
            s_valid = 1.0
        else:
            s_idx = 0
            s_target = torch.zeros(seq_len, dtype=torch.float32)
            s_valid = 0.0
        items.append(
            {
                "x": x,
                "t": t,
                "p_idx": p_idx,
                "s_idx": s_idx,
                "p_target": p_target,
                "s_target": s_target,
                "p_valid": 1.0,
                "s_valid": s_valid,
                "is_event": 1.0,
                "chunk": s.get("chunk", ""),
            }
        )
    return items


def collate(batch: list[dict]) -> dict:
    out = {
        "x": torch.stack([b["x"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
        "p_target": torch.stack([b["p_target"] for b in batch]),
        "s_target": torch.stack([b["s_target"] for b in batch]),
        "p_idx": torch.tensor([b["p_idx"] for b in batch], dtype=torch.long),
        "s_idx": torch.tensor([b["s_idx"] for b in batch], dtype=torch.long),
        "p_valid": torch.tensor([b["p_valid"] for b in batch], dtype=torch.float32),
        "s_valid": torch.tensor([b["s_valid"] for b in batch], dtype=torch.float32),
        "is_event": torch.tensor([b["is_event"] for b in batch], dtype=torch.float32),
    }
    return out


def adapt_loss(out: dict, batch: dict, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    p_logits = out.get("p_logits", out["p"])
    s_logits = out.get("s_logits", out["s"])
    det_logits = out.get("det_logits", out["det"])

    p_logits = torch.nan_to_num(p_logits, nan=-50.0, posinf=50.0, neginf=-50.0)
    s_logits = torch.nan_to_num(s_logits, nan=-50.0, posinf=50.0, neginf=-50.0)

    loss_p = weighted_pick_loss(p_logits, batch["p_target"], args.pick_pos_weight)
    # Only supervise S where labeled.
    s_mask = batch["s_valid"] > 0.5
    if s_mask.any():
        loss_s = weighted_pick_loss(
            s_logits[s_mask], batch["s_target"][s_mask], args.pick_pos_weight
        )
    else:
        loss_s = p_logits.new_zeros(())

    # OBS windows are already events; run28 det often NaNs on OBS — do not let
    # that poison pick-head gradients.
    if args.det_loss_weight > 0 and torch.isfinite(det_logits).all():
        if det_logits.dim() == 1:
            det_t = batch["is_event"]
            loss_det = F.binary_cross_entropy_with_logits(det_logits, det_t)
        else:
            det_prob = torch.sigmoid(det_logits).amax(dim=-1)
            loss_det = F.binary_cross_entropy(det_prob.clamp(1e-6, 1 - 1e-6), batch["is_event"])
    else:
        loss_det = p_logits.new_zeros(())

    loss = (
        args.p_loss_weight * loss_p
        + args.s_loss_weight * loss_s
        + args.det_loss_weight * loss_det
    )
    return loss, {
        "loss": float(loss.detach()),
        "loss_p": float(loss_p.detach()),
        "loss_s": float(loss_s.detach()),
        "loss_det": float(loss_det.detach()),
    }


@torch.no_grad()
def eval_pick_only(model, samples, device, args, obs_mod) -> dict:
    model.eval()
    return obs_mod.eval_hnf(
        model,
        samples,
        device,
        args.seq_len,
        args.window_sec,
        args.pick_threshold,
        args.det_threshold,
        args.tol_sec,
        args.batch_size,
    )["pick_only"]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    obs_mod = _load_obs_compare_module()
    chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
    print(f"[obs-adapt] chunks={chunks} device={device}", flush=True)
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
            f"[obs-adapt] disjoint split {args.split_json}: "
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
        f"[obs-adapt] n={len(samples)} with_S={sum(1 for s in samples if s['s_valid'])} "
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

    train_items = build_items(
        train_samples, args.seq_len, args.window_sec, args.label_sigma_sec, obs_mod.normalize_wave
    )
    train_loader = DataLoader(
        ItemDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    model, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=False)
    long_cfg = configure_long_sequence(
        model, bool(args.force_sparse_band), float(args.cap_local_window_sec)
    )
    trained_names = freeze_for_adapt(model, args.tune)
    if args.seq_len >= 3000:
        # Noise-cancel banded kernels dominate VRAM at EQT-grid resolution.
        for name, p in model.named_parameters():
            if name.startswith("noise_"):
                p.requires_grad = False
        trained_names = [n for n in trained_names if not n.startswith("noise_")]
        print("[obs-adapt] seq_len>=3000: froze noise_* for VRAM", flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"[obs-adapt] tune={args.tune} trainable={n_train}/{n_all} "
        f"groups={len(trained_names)} seq_len={args.seq_len} long_cfg={long_cfg}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # Baseline zero-shot on the val pool
    base_metrics = eval_pick_only(model, val_samples, device, args, obs_mod)
    print(f"[obs-adapt] val zero-shot pick_only={base_metrics}", flush=True)

    history = []
    best = {"val_score": -1.0, "epoch": -1, "metrics": None}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        # Keep dropout/noise off in frozen STEAD trunk; only update adapted modules.
        model.eval()
        train_mod_prefixes = {
            "heads": ("p_pick_head", "s_pick_head"),
            "heads+noise": (
                "p_pick_head", "s_pick_head", "det_head",
                "noise_cancel_branch", "noise_cue_adapter",
            ),
            "heads+onset": (
                "p_pick_head", "s_pick_head", "det_head", "raw_onset_encoder",
                "noise_cancel_branch", "noise_cue_adapter",
            ),
            "trunk-tail": (
                "p_pick_head", "s_pick_head", "det_head", "raw_onset_encoder",
                "noise_cancel_branch", "noise_cue_adapter", "source_embed",
                "p_layers", "s_layers",
            ),
            "all": (),
        }[args.tune]
        if args.tune == "all":
            model.train()
        else:
            for name, mod in model.named_modules():
                if name in train_mod_prefixes or any(
                    name.startswith(p + ".") for p in train_mod_prefixes
                ):
                    mod.train()
            if args.tune == "heads+onset":
                for layer_list_name in ("p_layers", "s_layers"):
                    layers = getattr(model, layer_list_name, None)
                    if layers is not None and len(layers) > 1:
                        layers[-1].train()
            if args.tune == "trunk-tail":
                for layer_list_name in ("p_layers", "s_layers"):
                    layers = getattr(model, layer_list_name, None)
                    if layers is not None:
                        for layer in layers:
                            layer.train()
        run = {"loss": 0.0, "loss_p": 0.0, "loss_s": 0.0, "loss_det": 0.0, "n": 0}
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch["x"], batch["t"])
            loss, stats = adapt_loss(out, batch, args)
            if not torch.isfinite(loss):
                print(f"[obs-adapt] skip non-finite loss ep{ep} stats={stats}", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            bs = batch["x"].size(0)
            for k in ("loss", "loss_p", "loss_s", "loss_det"):
                run[k] += stats[k] * bs
            run["n"] += bs
        sched.step()
        train_stats = {k: run[k] / max(1, run["n"]) for k in ("loss", "loss_p", "loss_s", "loss_det")}

        row = {"epoch": ep, "train": train_stats, "lr": opt.param_groups[0]["lr"]}
        if ep % args.eval_every == 0 or ep == args.epochs:
            metrics = eval_pick_only(model, val_samples, device, args, obs_mod)
            # Prefer lifting P; secondary S.
            score = 0.65 * metrics["p_f1"] + 0.35 * metrics["s_f1"]
            row["val_pick_only"] = metrics
            row["val_score"] = score
            print(
                f"[obs-adapt] ep{ep} train_loss={train_stats['loss']:.4f} "
                f"val P={metrics['p_f1']:.3f} S={metrics['s_f1']:.3f} score={score:.3f}",
                flush=True,
            )
            if score > best["val_score"]:
                best = {"val_score": score, "epoch": ep, "metrics": metrics}
                ckpt = {
                    "state_dict": model.state_dict(),
                    "args": ckpt_args,
                    "adapt_args": vars(args),
                    "long_sequence_cfg": long_cfg,
                    "base_val_pick_only": base_metrics,
                    "best_val_pick_only": metrics,
                    "trained_param_names": trained_names,
                    "epoch": ep,
                }
                torch.save(ckpt, out_dir / "best.pt")
                torch.save(ckpt, out_dir / "last.pt")
        else:
            print(f"[obs-adapt] ep{ep} train_loss={train_stats['loss']:.4f}", flush=True)
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Final eval: prefer disjoint holdout when available
    best_path = out_dir / "best.pt"
    model, _ = load_model(best_path, device, bypass_noise_cancel=False)
    configure_long_sequence(
        model, bool(args.force_sparse_band), float(args.cap_local_window_sec)
    )
    eval_samples = holdout_samples if holdout_samples else samples
    eval_name = "holdout_adapted" if holdout_samples else "full_pool_adapted"
    full_metrics = eval_pick_only(model, eval_samples, device, args, obs_mod)
    report = {
        "checkpoint_init": args.checkpoint,
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
            "disjoint_holdout": bool(holdout_samples),
        },
    }
    (out_dir / "adapt_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# OBS light-adapt report",
        "",
        f"- init: `{args.checkpoint}`",
        f"- tune: `{args.tune}` ({n_train} params)",
        f"- chunks: {chunks}",
        f"- n_train/val/full: {len(train_samples)}/{len(val_samples)}/{len(samples)}",
        f"- device: `{device}`",
        "",
        "## Val pick-only",
        f"- zero-shot: P={base_metrics['p_f1']:.3f} S={base_metrics['s_f1']:.3f}",
        f"- best (ep{best['epoch']}): P={best['metrics']['p_f1']:.3f} S={best['metrics']['s_f1']:.3f}",
        "",
        "## Full-pool adapted (same n as train+val)",
        f"- P={full_metrics['p_f1']:.3f} S={full_metrics['s_f1']:.3f}",
        "",
        f"best.pt → `{best_path}`",
    ]
    (out_dir / "adapt_report.md").write_text("\n".join(md))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
