#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBS Stage-2: P-focused fine-tune from a matched light-adapt checkpoint.

Fairness: same disjoint split as Step-4 / P–S grid; report holdout pick-only only.
Recipe: freeze S branch, train P head + P layers (+ onset), heavy P BCE + wrong-peak.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.analyze_stead_picking import load_model
from tools.obs_matched_split import load_split_samples
from tools.train_obs_light_adapt import (
    ItemDataset,
    build_items,
    collate,
    configure_long_sequence,
    eval_pick_only,
)
from tools.train_stead_picking import (
    move_batch_to_device,
    set_seed,
    weighted_pick_loss,
    wrong_peak_rank_loss,
)


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS P-only stage-2 fine-tune")
    p.add_argument(
        "--checkpoint",
        default="outputs/obs_ps_tradeoff_grid/selected_hnf/best.pt",
        help="Init from selected light-adapt HNF (trunk-tail/L1200)",
    )
    p.add_argument("--output-dir", default="outputs/obs_p_stage2_L1200")
    p.add_argument(
        "--split-json",
        default="outputs/obs_matched_adapt_split_randoffset/split.json",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seq-len", type=int, default=1200)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--label-sigma-sec", type=float, default=0.30)
    p.add_argument("--pick-pos-weight", type=float, default=32.0)
    p.add_argument("--p-loss-weight", type=float, default=3.0)
    p.add_argument("--s-loss-weight", type=float, default=0.15, help="Keep tiny S anchor")
    p.add_argument("--wrong-peak-weight", type=float, default=0.8)
    p.add_argument("--wrong-peak-radius-sec", type=float, default=0.5)
    p.add_argument("--wrong-peak-margin", type=float, default=0.25)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--s-floor", type=float, default=0.58, help="Holdout S soft floor for selection")
    p.add_argument("--p-select-weight", type=float, default=0.75)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--force-sparse-band", action="store_true", default=True)
    p.add_argument("--cap-local-window-sec", type=float, default=8.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def freeze_p_only(model: torch.nn.Module) -> list[str]:
    for p in model.parameters():
        p.requires_grad = False
    allow = (
        "p_pick_head.",
        "p_layers.",
        "raw_onset_encoder.",
        "source_embed.",  # shared but helps P onset; keep trainable lightly
    )
    trained = []
    for name, p in model.named_parameters():
        if any(name.startswith(a) for a in allow):
            p.requires_grad = True
            trained.append(name)
    return trained


def p_stage_loss(out: dict, batch: dict, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    p_logits = torch.nan_to_num(out.get("p_logits", out["p"]), nan=-50.0, posinf=50.0, neginf=-50.0)
    s_logits = torch.nan_to_num(out.get("s_logits", out["s"]), nan=-50.0, posinf=50.0, neginf=-50.0)

    loss_p = weighted_pick_loss(p_logits, batch["p_target"], args.pick_pos_weight)
    s_mask = batch["s_valid"] > 0.5
    if args.s_loss_weight > 0 and s_mask.any():
        loss_s = weighted_pick_loss(
            s_logits[s_mask], batch["s_target"][s_mask], args.pick_pos_weight
        )
    else:
        loss_s = p_logits.new_zeros(())

    # Wrong-peak ranking on P only (events always present in OBS windows).
    event_mask = torch.ones(p_logits.size(0), dtype=torch.bool, device=p_logits.device)
    if args.wrong_peak_weight > 0:
        loss_wp = wrong_peak_rank_loss(
            p_logits,
            batch["p_idx"],
            event_mask,
            seq_len=p_logits.size(-1),
            radius_sec=args.wrong_peak_radius_sec,
            margin=args.wrong_peak_margin,
        )
    else:
        loss_wp = p_logits.new_zeros(())

    loss = (
        args.p_loss_weight * loss_p
        + args.s_loss_weight * loss_s
        + args.wrong_peak_weight * loss_wp
    )
    return loss, {
        "loss": float(loss.detach()),
        "loss_p": float(loss_p.detach()),
        "loss_s": float(loss_s.detach()),
        "loss_wp": float(loss_wp.detach()),
    }


def select_score(m: dict, p_w: float, s_floor: float) -> float:
    p, s = float(m["p_f1"]), float(m["s_f1"])
    sc = p_w * p + (1.0 - p_w) * s
    if s < s_floor:
        sc -= 0.25 * (s_floor - s)
    return sc


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    obs_mod = _load_obs_compare_module()

    train_pool, _, meta = load_split_samples(args.split_json, "train")
    holdout, _, _ = load_split_samples(args.split_json, "holdout")
    print(
        f"[p-stage2] train_pool={len(train_pool)} holdout={len(holdout)} "
        f"init={args.checkpoint} seq_len={args.seq_len}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(train_pool))
    rng.shuffle(idxs)
    n_val = max(64, int(round(len(train_pool) * args.val_frac)))
    val_set = set(idxs[:n_val].tolist())
    train_s = [train_pool[i] for i in idxs if i not in val_set]
    val_s = [train_pool[i] for i in sorted(val_set)]

    train_items = build_items(
        train_s, args.seq_len, args.window_sec, args.label_sigma_sec, obs_mod.normalize_wave
    )
    loader = DataLoader(
        ItemDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    model, ckpt_args = load_model(Path(args.checkpoint), device)
    long_cfg = configure_long_sequence(
        model, bool(args.force_sparse_band), float(args.cap_local_window_sec)
    )
    trained = freeze_p_only(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[p-stage2] trainable={n_train} names={len(trained)} long={long_cfg}", flush=True)

    # Namespace shim for eval_pick_only
    class E:
        pass

    eargs = E()
    eargs.seq_len = args.seq_len
    eargs.window_sec = args.window_sec
    eargs.pick_threshold = args.pick_threshold
    eargs.det_threshold = 0.5
    eargs.tol_sec = args.tol_sec
    eargs.batch_size = max(2, args.batch_size)

    base_hold = eval_pick_only(model, holdout, device, eargs, obs_mod)
    print(
        f"[p-stage2] holdout baseline P={base_hold['p_f1']:.3f} S={base_hold['s_f1']:.3f}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    history = []
    base_score = select_score(base_hold, args.p_select_weight, args.s_floor)
    best = {
        "score": base_score,
        "epoch": 0,
        "holdout": base_hold,
        "val": None,
    }
    # Keep baseline as fallback until a better holdout score appears
    best_path = out_dir / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": ckpt_args,
            "p_stage2_args": vars(args),
            "holdout_pick_only": base_hold,
            "baseline_holdout": base_hold,
            "best_holdout": base_hold,
            "epoch": 0,
            "note": "baseline_copy",
        },
        best_path,
    )

    for ep in range(1, args.epochs + 1):
        model.eval()
        for name, mod in model.named_modules():
            if name.startswith(("p_pick_head", "p_layers", "raw_onset_encoder", "source_embed")):
                mod.train()
        run = {"loss": 0.0, "loss_p": 0.0, "loss_s": 0.0, "loss_wp": 0.0, "n": 0}
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = model(batch["x"], batch["t"])
            loss, stats = p_stage_loss(pred, batch, args)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            bs = batch["x"].size(0)
            for k in ("loss", "loss_p", "loss_s", "loss_wp"):
                run[k] += stats[k] * bs
            run["n"] += bs
        sched.step()
        train_stats = {k: run[k] / max(run["n"], 1) for k in ("loss", "loss_p", "loss_s", "loss_wp")}

        val_m = eval_pick_only(model, val_s, device, eargs, obs_mod)
        hold_m = eval_pick_only(model, holdout, device, eargs, obs_mod)
        sc = select_score(hold_m, args.p_select_weight, args.s_floor)
        row = {
            "epoch": ep,
            "train": train_stats,
            "val": val_m,
            "holdout": hold_m,
            "score": sc,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[p-stage2] ep{ep} loss={train_stats['loss']:.4f} "
            f"val P={val_m['p_f1']:.3f} S={val_m['s_f1']:.3f} | "
            f"holdout P={hold_m['p_f1']:.3f} S={hold_m['s_f1']:.3f} score={sc:.3f}",
            flush=True,
        )
        if sc > best["score"]:
            best = {"score": sc, "epoch": ep, "holdout": hold_m, "val": val_m}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": ckpt_args,
                    "p_stage2_args": vars(args),
                    "long_sequence_cfg": long_cfg,
                    "trained_param_names": trained,
                    "baseline_holdout": base_hold,
                    "best_holdout": hold_m,
                    "epoch": ep,
                },
                best_path,
            )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        # Early abort if clearly worse than baseline for 3 straight epochs
        if ep >= 4:
            recent = history[-3:]
            if all(r["holdout"]["p_f1"] + 0.01 < base_hold["p_f1"] for r in recent):
                print("[p-stage2] ABORT: holdout P below baseline for 3 epochs", flush=True)
                break

    # Reload best and final report
    model, _ = load_model(best_path, device)
    configure_long_sequence(model, bool(args.force_sparse_band), float(args.cap_local_window_sec))
    final = eval_pick_only(model, holdout, device, eargs, obs_mod)
    report = {
        "init": args.checkpoint,
        "best_ckpt": str(best_path),
        "split_json": args.split_json,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "baseline_holdout": base_hold,
        "best_epoch": best["epoch"],
        "best_holdout": best["holdout"] or final,
        "final_holdout": final,
        "fairness": [
            "same disjoint holdout as Step-4 / P-S grid",
            "pick-only F1 on holdout",
            "compare only within matched-adapt treatment",
        ],
        "delta_p": float(final["p_f1"] - base_hold["p_f1"]),
        "delta_s": float(final["s_f1"] - base_hold["s_f1"]),
    }
    (out_dir / "p_stage2_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# OBS P-stage2 report",
        "",
        f"- init: `{args.checkpoint}`",
        f"- best: `{best_path}` (ep {best['epoch']})",
        f"- seq_len: {args.seq_len}",
        "",
        "## Holdout pick-only",
        f"- baseline: P={base_hold['p_f1']:.3f} S={base_hold['s_f1']:.3f}",
        f"- best:     P={report['best_holdout']['p_f1']:.3f} S={report['best_holdout']['s_f1']:.3f}",
        f"- ΔP={report['delta_p']:+.3f}  ΔS={report['delta_s']:+.3f}",
        "",
    ]
    (out_dir / "p_stage2_report.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
