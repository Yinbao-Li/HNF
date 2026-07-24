#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train StrongPhaseExistHead on frozen L1200 backbone to push gated S F1 → 0.7+.

Oracle ceiling ≈ 0.72–0.75. Post-hoc MLP calibrator stalled at ~0.59.
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

from hnf.picking_model import StrongPhaseExistHead, build_picking_model, load_picking_model_state
from tools.obs_matched_split import load_split_samples
from tools.train_obs_picking import (
    ItemDataset,
    build_items,
    collate,
    filter_alive_channels,
    _load_obs_compare_module,
)
from tools.train_stead_picking import move_batch_to_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_strong_sexist_12ep",
    )
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--absent-oversample", type=float, default=2.5)
    p.add_argument("--hard-absent-boost", type=float, default=3.0)
    p.add_argument("--neg-weight", type=float, default=2.0)
    p.add_argument("--pick-th", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_backbone(ckpt_path: Path, device: torch.device):
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
        obliquity_scale=float(a.get("obliquity_scale", 1.0)),
        phase_exist=True,
        phase_exist_hidden=int(a.get("phase_exist_hidden", 64)),
        strong_s_exist=True,
    ).to(device)
    # Load backbone; s_exist_head is new StrongPhaseExistHead → expected missing
    missing, unexpected = load_picking_model_state(model, ckpt["state_dict"], strict=False)
    if bool(a.get("enable_preserve_gate")) and model.noise_cancel_branch is not None:
        model.noise_cancel_branch.enable_preserve_gate = True
    # Freeze all but strong s exist head
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("s_exist_head.")
    print(
        f"[strong-s] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)} "
        f"trainable={[n for n,_ in model.named_parameters() if _.requires_grad][:6]}...",
        flush=True,
    )
    cfg = dict(a)
    cfg["input_dim"] = input_dim
    cfg["seq_len"] = int(a.get("seq_len", 1200))
    cfg["window_sec"] = float(a.get("window_sec", 60.0))
    cfg["label_sigma_sec"] = float(a.get("label_sigma_sec", 0.30))
    cfg["strong_s_exist"] = True
    return model, cfg


@torch.no_grad()
def eval_gated_s(
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
    """Board-aligned eval: soft_floor + P score_minus_late (keeps P ≈ L1200)."""
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
    device = torch.device(args.device)
    obs = _load_obs_compare_module()
    model, cfg = load_backbone(Path(args.checkpoint), device)

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
    # Oversample no-S; extra weight if we can peek peak later — use label only here
    weights = []
    for it in items:
        if float(it["s_valid"].item()) > 0.5:
            weights.append(1.0)
        else:
            weights.append(float(args.absent_oversample))
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
        f"[strong-s] train={len(train)} val={len(val)} holdout={len(holdout)} "
        f"absent_os={args.absent_oversample}",
        flush=True,
    )

    history = []
    best = {"score": -1.0, "exist_th": 0.5, "metrics": None}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        # keep BN/dropout of backbone in eval since frozen
        model.eval()
        model.s_exist_head.train()
        run_loss = 0.0
        n = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            # Full forward; backbone params frozen (grads unused / not stepped)
            fwd = model(batch["x"], batch["t"])
            s_logits = torch.nan_to_num(fwd.get("s_logits", fwd["s"]), nan=-50.0)
            s_peak = torch.sigmoid(s_logits).amax(dim=-1).detach()
            y = batch["s_valid"]
            logits = fwd["s_exist"]
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            # hard absent: no-S but high peak → boost weight
            hard = (y <= 0.5) & (s_peak >= args.pick_th)
            w = torch.ones_like(y)
            w = torch.where(y <= 0.5, w * args.neg_weight, w)
            w = torch.where(hard, w * args.hard_absent_boost, w)
            loss = (bce * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * y.size(0)
            n += y.size(0)

        # sweep exist_th on val
        best_ep = None
        for eth in np.arange(0.25, 0.81, 0.05):
            m, ea = eval_gated_s(model, val, device, cfg, obs, args.pick_th, float(eth))
            score = m["s_f1"]
            if best_ep is None or score > best_ep["s_f1"]:
                best_ep = {**m, "exist_th": float(eth), "exist_acc_s": ea["s"]}
        row = {
            "epoch": ep,
            "loss": run_loss / max(n, 1),
            "val": best_ep,
        }
        history.append(row)
        print(
            f"[strong-s] ep{ep} loss={row['loss']:.4f} "
            f"valS={best_ep['s_f1']:.3f} P={best_ep['p_f1']:.3f} "
            f"@exist_th={best_ep['exist_th']:.2f} existAccS={best_ep['exist_acc_s']:.3f}",
            flush=True,
        )
        if best_ep["s_f1"] > best["score"]:
            best = {
                "score": best_ep["s_f1"],
                "exist_th": best_ep["exist_th"],
                "metrics": best_ep,
                "epoch": ep,
            }
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": {**cfg, "strong_s_exist": True, "exist_th": best_ep["exist_th"]},
                    "phase_exist": True,
                    "strong_s_exist": True,
                    "input_dim": dim,
                    "init": str(args.checkpoint),
                    "best_val": best_ep,
                    "epoch": ep,
                },
                out_dir / "best.pt",
            )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Holdout
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    load_picking_model_state(model, ck["state_dict"], strict=False)
    eth = float(best["exist_th"])
    hold_m, hold_ea = eval_gated_s(model, holdout, device, cfg, obs, args.pick_th, eth)
    # also sweep holdout
    hold_rows = []
    for e in np.arange(0.25, 0.81, 0.05):
        m, ea = eval_gated_s(model, holdout, device, cfg, obs, args.pick_th, float(e))
        hold_rows.append({**m, "exist_th": float(e), "exist_acc_s": ea["s"]})
    hold_best = max(hold_rows, key=lambda r: r["s_f1"])
    report = {
        "init": args.checkpoint,
        "best_epoch": best["epoch"],
        "val_best": best["metrics"],
        "holdout_at_val_th": {**hold_m, "exist_th": eth, "exist_acc": hold_ea},
        "holdout_best": hold_best,
        "elapsed_sec": time.time() - t0,
        "target": "gated S F1 >= 0.70",
    }
    (out_dir / "strong_sexist_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Strong S-exist report",
        "",
        f"- init: `{args.checkpoint}`",
        f"- best ep{best['epoch']} val S={best['score']:.3f} @exist_th={eth:.2f}",
        f"- holdout @val_th: P={hold_m['p_f1']:.3f} S={hold_m['s_f1']:.3f}",
        f"- holdout best: P={hold_best['p_f1']:.3f} S={hold_best['s_f1']:.3f} "
        f"@exist_th={hold_best['exist_th']:.2f}",
        "",
    ]
    (out_dir / "strong_sexist_report.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
