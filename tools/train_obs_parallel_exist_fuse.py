#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FT parallel field-only P/S exist heads + fuse MLP on frozen L1200 backbone.

Architecture (user-requested):
  branch field ──► pick_head  ──┐
                 ──► exist_head ─┴► fuse_MLP ► final pick / exist_ref

Exist does NOT consume pick logits; fuse decides the emitted curve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.picking_model import build_picking_model, load_picking_model_state
from tools.obs_matched_split import load_split_samples
from tools.train_obs_picking import (
    ItemDataset,
    build_items,
    collate,
    filter_alive_channels,
    weighted_phase_exist_loss,
    _load_obs_compare_module,
)
from tools.train_stead_picking import move_batch_to_device, set_seed, weighted_pick_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_parallel_exist_fuse_12ep",
    )
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--absent-oversample", type=float, default=2.5)
    p.add_argument("--pick-loss-weight", type=float, default=1.0)
    p.add_argument("--exist-loss-weight", type=float, default=1.5)
    p.add_argument("--s-exist-neg-weight", type=float, default=2.0)
    p.add_argument("--pick-th", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-pick-heads",
        action="store_true",
        help="Also unfreeze p/s pick heads (default: only exist+fuse)",
    )
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device, train_pick: bool):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    input_dim = int(a.get("input_dim") or 4)
    model = build_picking_model(
        input_dim=input_dim,
        embed_dim=a.get("embed_dim", 64),
        num_shared_layers=a.get("num_shared_layers", 2),
        num_branch_layers=a.get("num_branch_layers", 2),
        local_window_sec=a.get("local_window_sec", 12.0),
        dropout=a.get("dropout", 0.1),
        pick_head_hidden=a.get("pick_head_hidden", 48),
        pick_head_kernel=a.get("pick_head_kernel", 7),
        pick_head_layers=a.get("pick_head_layers", 4),
        multi_scale=bool(a.get("multi_scale", True)),
        sparse_band=bool(a.get("sparse_band", True)),
        residual_pick_head=bool(a.get("residual_pick_head", True)),
        residual_det_head=bool(a.get("residual_det_head", False)),
        enhanced_det_head=bool(a.get("enhanced_det_head", True)),
        noise_cancel=bool(a.get("noise_cancel", True)),
        noise_source_dim=a.get("noise_source_dim", 16),
        noise_det_pick_split=bool(a.get("noise_det_pick_split", True)),
        noise_pick_cues=bool(a.get("noise_pick_cues", True)),
        principle=a.get("principle", "huygens_fresnel"),
        phase_exist=True,
        phase_exist_hidden=int(a.get("phase_exist_hidden", 64)),
        parallel_exist_fuse=True,
    ).to(device)
    if bool(a.get("enable_preserve_gate", False)) and getattr(model, "noise_cancel_branch", None) is not None:
        model.noise_cancel_branch.enable_preserve_gate = True
    missing, unexpected = load_picking_model_state(model, ckpt["state_dict"], strict=False)
    train_prefixes = (
        "p_exist_head.",
        "s_exist_head.",
        "p_exist_fuse.",
        "s_exist_fuse.",
    )
    if train_pick:
        train_prefixes = train_prefixes + ("p_pick_head.", "s_pick_head.")
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(train_prefixes)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(
        f"[par-fuse] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)} "
        f"trainable={len(trainable)} e.g. {trainable[:6]}",
        flush=True,
    )
    cfg = dict(a)
    cfg["input_dim"] = input_dim
    cfg["seq_len"] = int(a.get("seq_len", 1200))
    cfg["window_sec"] = float(a.get("window_sec", 60.0))
    cfg["label_sigma_sec"] = float(a.get("label_sigma_sec", 0.30))
    cfg["phase_exist"] = True
    cfg["parallel_exist_fuse"] = True
    return model, cfg


@torch.no_grad()
def eval_gated(
    model,
    samples,
    device,
    cfg,
    obs_mod,
    pick_th: float,
    exist_th: float,
    *,
    gate_mode: str = "soft_floor",
    soft_th: float = 0.25,
    p_decode_mode: str = "score_minus_late",
    decode_late_penalty: float = 0.60,
):
    """Board-aligned eval: soft_floor + P score_minus_late."""
    model.eval()
    ev = obs_mod.eval_hnf(
        model,
        samples,
        device,
        int(cfg["seq_len"]),
        float(cfg["window_sec"]),
        pick_th,
        0.5,
        0.5,
        4,
        n_channels=int(cfg["input_dim"]),
        exist_th=exist_th,
        score_absent=True,
        gate_mode=gate_mode,
        soft_th=soft_th,
        p_decode_mode=p_decode_mode,
        s_decode_mode="argmax",
        decode_late_penalty=decode_late_penalty,
    )
    return ev["pick_only"], ev["exist_acc"]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    obs = _load_obs_compare_module()
    model, cfg = load_model(Path(args.checkpoint), device, args.train_pick_heads)

    train_pool, _, meta = load_split_samples(args.split_json, "train")
    holdout, _, _ = load_split_samples(args.split_json, "holdout")
    val, _ = obs.load_obs_windows_from_entries(
        meta["val_entries"], float(meta["window_sec"]), require_full_3c=True
    )
    dim = int(cfg["input_dim"])
    train = filter_alive_channels(train_pool, dim, mode="strict")
    val = filter_alive_channels(val, dim, mode="strict")
    holdout = filter_alive_channels(holdout, dim, mode="strict")

    items = build_items(
        train,
        int(cfg["seq_len"]),
        float(cfg["window_sec"]),
        float(cfg["label_sigma_sec"]),
        obs.normalize_wave,
        augment=True,
        seed=args.seed,
        input_dim=dim,
    )
    weights = [
        1.0 if float(it["s_valid"].item()) > 0.5 else float(args.absent_oversample)
        for it in items
    ]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(items),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        ItemDataset(items),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    print(
        f"[par-fuse] train={len(train)} val={len(val)} holdout={len(holdout)}",
        flush=True,
    )

    history = []
    best = {"score": -1.0, "exist_th": 0.5, "metrics": None, "epoch": 0}
    for ep in range(1, args.epochs + 1):
        model.eval()
        for m in (model.p_exist_head, model.s_exist_head, model.p_exist_fuse, model.s_exist_fuse):
            if m is not None:
                m.train()
        if args.train_pick_heads:
            model.p_pick_head.train()
            model.s_pick_head.train()
        run_loss = 0.0
        n = 0
        t0 = time.time()
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            fwd = model(batch["x"], batch["t"])
            p = torch.nan_to_num(fwd["p"], nan=-50.0)
            s = torch.nan_to_num(fwd["s"], nan=-50.0)
            loss_p = weighted_pick_loss(p, batch["p_target"], pos_weight=32.0)
            s_mask = batch["s_valid"] > 0.5
            if s_mask.any():
                loss_s = weighted_pick_loss(s[s_mask], batch["s_target"][s_mask], pos_weight=32.0)
            else:
                loss_s = p.new_zeros(())
            loss_ep = F.binary_cross_entropy_with_logits(fwd["p_exist"], batch["p_valid"])
            loss_es = weighted_phase_exist_loss(
                fwd["s_exist"],
                batch["s_valid"],
                pos_weight=1.35,
                neg_weight=args.s_exist_neg_weight,
            )
            loss_ref = p.new_zeros(())
            if "p_exist_ref" in fwd:
                loss_ref = 0.5 * (
                    F.binary_cross_entropy_with_logits(fwd["p_exist_ref"], batch["p_valid"])
                    + weighted_phase_exist_loss(
                        fwd["s_exist_ref"],
                        batch["s_valid"],
                        pos_weight=1.35,
                        neg_weight=args.s_exist_neg_weight,
                    )
                )
            loss = (
                args.pick_loss_weight * (loss_p + loss_s)
                + args.exist_loss_weight * (loss_ep + loss_es + loss_ref)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * batch["p_valid"].size(0)
            n += int(batch["p_valid"].size(0))

        best_ep = None
        for eth in np.arange(0.20, 0.81, 0.05):
            m, ea = eval_gated(model, val, device, cfg, obs, args.pick_th, float(eth))
            score = 0.5 * (m["p_f1"] + m["s_f1"])
            if best_ep is None or score > best_ep["score"]:
                best_ep = {**m, "exist_th": float(eth), "exist_acc_s": ea["s"], "score": score}
        soft_m, soft_ea = eval_gated(model, val, device, cfg, obs, args.pick_th, 0.05)
        row = {
            "epoch": ep,
            "loss": run_loss / max(n, 1),
            "sec": time.time() - t0,
            "val_best": best_ep,
            "val_fuse_soft": {**soft_m, "exist_acc_s": soft_ea["s"]},
        }
        history.append(row)
        print(
            f"[par-fuse] ep{ep} loss={row['loss']:.4f} "
            f"val P={best_ep['p_f1']:.3f} S={best_ep['s_f1']:.3f} "
            f"@exist={best_ep['exist_th']:.2f} "
            f"fuseSoft P={soft_m['p_f1']:.3f} S={soft_m['s_f1']:.3f}",
            flush=True,
        )
        if best_ep["score"] > best["score"]:
            best = {**best_ep, "epoch": ep}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": {
                        **cfg,
                        "parallel_exist_fuse": True,
                        "exist_th": best_ep["exist_th"],
                    },
                    "phase_exist": True,
                    "parallel_exist_fuse": True,
                    "input_dim": dim,
                    "init": str(args.checkpoint),
                    "best_val": best_ep,
                    "epoch": ep,
                },
                out_dir / "best.pt",
            )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    load_picking_model_state(model, ck["state_dict"], strict=False)
    eth = float(best["exist_th"])
    hold_m, hold_ea = eval_gated(model, holdout, device, cfg, obs, args.pick_th, eth)
    hold_soft, _ = eval_gated(model, holdout, device, cfg, obs, args.pick_th, 0.05)
    hold_rows = []
    for e in np.arange(0.20, 0.81, 0.05):
        m, ea = eval_gated(model, holdout, device, cfg, obs, args.pick_th, float(e))
        hold_rows.append({**m, "exist_th": float(e), "exist_acc_s": ea["s"]})
    hold_best = max(hold_rows, key=lambda r: 0.5 * (r["p_f1"] + r["s_f1"]))
    report = {
        "init": args.checkpoint,
        "best_epoch": best["epoch"],
        "val_best": {k: best[k] for k in best if k != "epoch"},
        "holdout_at_val_th": {**hold_m, "exist_th": eth, "exist_acc": hold_ea},
        "holdout_fuse_soft": hold_soft,
        "holdout_best": hold_best,
    }
    (out_dir / "parallel_exist_fuse_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Parallel exist + fuse MLP",
        "",
        f"- init: `{args.checkpoint}`",
        f"- best ep{best['epoch']} val P={best['p_f1']:.3f} S={best['s_f1']:.3f} @exist={eth:.2f}",
        f"- holdout @val_th: P={hold_m['p_f1']:.3f} S={hold_m['s_f1']:.3f}",
        f"- holdout fuseSoft(exist=0.05): P={hold_soft['p_f1']:.3f} S={hold_soft['s_f1']:.3f}",
        f"- holdout best: P={hold_best['p_f1']:.3f} S={hold_best['s_f1']:.3f} "
        f"@exist={hold_best['exist_th']:.2f}",
        "",
    ]
    (out_dir / "parallel_exist_fuse_report.md").write_text("\n".join(md))
    print("[par-fuse] done", json.dumps(report["holdout_at_val_th"]), flush=True)


if __name__ == "__main__":
    main()
