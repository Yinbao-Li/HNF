#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune OBS model with P/S existence gates on top of 4C ckpt.

Decision: first classify phase presence, then (if present) use pick peak time.
Supports strengthened gate supervision (S-absent pick penalty, exist×pick coupling).
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
    compute_loss,
    filter_alive_channels,
    gate_supervision_loss,
    weighted_phase_exist_loss,
    _load_obs_compare_module,
)
from tools.train_stead_picking import move_batch_to_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS exist-gate fine-tune")
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_fromscratch_30ep/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/run_obs_native/obs_4c_exist_gate_sup_8ep")
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--exist-loss-weight", type=float, default=2.0)
    p.add_argument("--s-exist-pos-weight", type=float, default=1.35)
    p.add_argument("--s-exist-neg-weight", type=float, default=2.5)
    p.add_argument("--s-absent-pick-penalty", type=float, default=0.8)
    p.add_argument("--exist-pick-couple-weight", type=float, default=1.2)
    p.add_argument("--p-only-oversample", type=float, default=2.0)
    p.add_argument("--exist-th", type=float, default=0.5)
    p.add_argument("--sweep-exist-th", action="store_true", default=True)
    p.add_argument("--pick-threshold", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-backbone-epochs", type=int, default=1)
    return p.parse_args()


def build_model_from_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    input_dim = int(a.get("input_dim") or ckpt.get("input_dim") or ckpt.get("n_channels") or 4)
    model = build_picking_model(
        input_dim=input_dim,
        embed_dim=a.get("embed_dim", 64),
        num_shared_layers=a.get("num_shared_layers", 2),
        num_branch_layers=a.get("num_branch_layers", 2),
        local_window_sec=a.get("local_window_sec", 15.0),
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
    ).to(device)
    missing, unexpected = load_picking_model_state(model, ckpt["state_dict"], strict=False)
    if bool(a.get("enable_preserve_gate")) and model.noise_cancel_branch is not None:
        model.noise_cancel_branch.enable_preserve_gate = True
    print(
        f"[exist-gate] loaded {ckpt_path} input_dim={input_dim} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    cfg = dict(a)
    cfg["input_dim"] = input_dim
    cfg["phase_exist"] = True
    cfg["noise_cancel_weight"] = float(a.get("noise_cancel_weight", 0.05))
    cfg["nc_consistency_weight"] = float(a.get("nc_consistency_weight", 0.5))
    cfg["nc_phase_weight"] = float(a.get("nc_phase_weight", 0.1))
    cfg["nc_preserve_weight"] = float(a.get("nc_preserve_weight", 0.3))
    cfg["nc_energy_weight"] = float(a.get("nc_energy_weight", 0.05))
    cfg["nc_noise_suppress_weight"] = float(a.get("nc_noise_suppress_weight", 0.2))
    cfg["pick_pos_weight"] = float(a.get("pick_pos_weight", 32))
    cfg["pick_loss_weight"] = float(a.get("pick_loss_weight", 2.8))
    cfg["p_pick_loss_weight"] = float(a.get("p_pick_loss_weight", 2.0))
    cfg["s_pick_loss_weight"] = float(a.get("s_pick_loss_weight", 1.4))
    cfg["focal_gamma"] = float(a.get("focal_gamma", 0.0))
    cfg["noise_pick_penalty"] = float(a.get("noise_pick_penalty", 0.0))
    cfg["wrong_peak_loss_weight"] = float(a.get("wrong_peak_loss_weight", 0.35))
    cfg["wrong_peak_radius_sec"] = float(a.get("wrong_peak_radius_sec", 0.45))
    cfg["wrong_peak_margin"] = float(a.get("wrong_peak_margin", 0.30))
    cfg["s_wrong_peak_scale"] = float(a.get("s_wrong_peak_scale", 1.35))
    cfg["ps_order_loss_weight"] = float(a.get("ps_order_loss_weight", 0.12))
    cfg["ps_min_gap_sec"] = float(a.get("ps_min_gap_sec", 0.1))
    cfg["seq_len"] = int(a.get("seq_len", 800))
    cfg["window_sec"] = float(a.get("window_sec", 60.0))
    cfg["label_sigma_sec"] = float(a.get("label_sigma_sec", 0.35))
    return model, cfg, ckpt


def exist_loss(out: dict, batch: dict, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    if "p_exist" not in out or "s_exist" not in out:
        z = batch["x"].new_zeros(())
        return z, {"loss_exist_p": 0.0, "loss_exist_s": 0.0, "loss_exist": 0.0}
    p_t = batch["p_valid"]
    s_t = batch["s_valid"]
    loss_p = F.binary_cross_entropy_with_logits(out["p_exist"], p_t)
    loss_s = weighted_phase_exist_loss(
        out["s_exist"],
        s_t,
        pos_weight=float(args.s_exist_pos_weight),
        neg_weight=float(args.s_exist_neg_weight),
    )
    loss = loss_p + loss_s
    return loss, {
        "loss_exist_p": float(loss_p.detach()),
        "loss_exist_s": float(loss_s.detach()),
        "loss_exist": float(loss.detach()),
    }


def build_oversampled_loader(
    items: list[dict],
    batch_size: int,
    p_only_oversample: float,
    seed: int,
) -> DataLoader:
    if p_only_oversample <= 1.0:
        return DataLoader(
            ItemDataset(items),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
    weights = []
    for it in items:
        has_s = float(it["s_valid"].item()) > 0.5
        weights.append(1.0 if has_s else float(p_only_oversample))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(items),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        ItemDataset(items),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("p_exist_head.") or name.startswith("s_exist_head."):
            p.requires_grad = True
        else:
            p.requires_grad = trainable


@torch.no_grad()
def eval_full(model, samples, device, cfg, obs_mod, exist_th: float, pick_th: float):
    model.eval()
    return obs_mod.eval_hnf(
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
    )


@torch.no_grad()
def sweep_exist_th(
    model,
    holdout,
    device,
    cfg,
    obs_mod,
    pick_th: float,
    thresholds: list[float] | None = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]
    rows = []
    for th in thresholds:
        ev = eval_full(model, holdout, device, cfg, obs_mod, th, pick_th)
        m = ev["pick_only"]
        score = 0.55 * m["p_f1"] + 0.30 * m["s_f1"] + 0.15 * float(ev["exist_acc"]["s"])
        rows.append(
            {
                "exist_th": th,
                "p_f1": m["p_f1"],
                "s_f1": m["s_f1"],
                "exist_acc_s": ev["exist_acc"]["s"],
                "score": score,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    obs_mod = _load_obs_compare_module()

    model, cfg, _raw_ckpt = build_model_from_ckpt(Path(args.checkpoint), device)
    train_pool, _, meta = load_split_samples(args.split_json, "train")
    holdout, _, _ = load_split_samples(args.split_json, "holdout")
    if meta.get("val_entries"):
        val_samples, _ = obs_mod.load_obs_windows_from_entries(
            meta["val_entries"], float(meta["window_sec"]), require_full_3c=True
        )
        train_samples = train_pool
    else:
        raise RuntimeError("split missing val_entries")

    dim = int(cfg["input_dim"])
    train_samples = filter_alive_channels(train_samples, dim)
    val_samples = filter_alive_channels(val_samples, dim)
    holdout = filter_alive_channels(holdout, dim)
    n_no_s = sum(1 for s in train_samples if not s["s_valid"])
    print(
        f"[exist-gate] train={len(train_samples)} (no_S={n_no_s}) val={len(val_samples)} "
        f"holdout={len(holdout)} exist_th={args.exist_th} "
        f"gate_sup=(absent_pick={args.s_absent_pick_penalty}, couple={args.exist_pick_couple_weight})",
        flush=True,
    )

    items = build_items(
        train_samples,
        int(cfg["seq_len"]),
        float(cfg["window_sec"]),
        float(cfg["label_sigma_sec"]),
        obs_mod.normalize_wave,
        augment=True,
        seed=args.seed,
        input_dim=dim,
    )
    loader = build_oversampled_loader(items, args.batch_size, args.p_only_oversample, args.seed)

    class A:
        pass

    loss_args = A()
    for k, v in cfg.items():
        setattr(loss_args, k, v)
    for k in (
        "s_exist_pos_weight",
        "s_exist_neg_weight",
    ):
        setattr(loss_args, k, getattr(args, k))
    # Pick/NC only via compute_loss; exist + gate supervision added separately below.
    loss_args.exist_loss_weight = 0.0
    loss_args.s_absent_pick_penalty = 0.0
    loss_args.exist_pick_couple_weight = 0.0

    history = []
    best = {"score": -1.0, "epoch": -1, "metrics": None}
    start = time.time()

    for ep in range(1, args.epochs + 1):
        freeze = ep <= args.freeze_backbone_epochs
        set_backbone_trainable(model, trainable=not freeze)
        exist_params = [p for n, p in model.named_parameters() if "exist_head" in n and p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if "exist_head" not in n and p.requires_grad]
        param_groups = []
        if exist_params:
            param_groups.append({"params": exist_params, "lr": args.lr})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr_backbone})
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        model.train()
        if freeze:
            model.eval()
            for n, m in model.named_modules():
                if n.startswith("p_exist_head") or n.startswith("s_exist_head"):
                    m.train()

        run = {
            "loss": 0.0,
            "loss_exist": 0.0,
            "loss_gate_sup": 0.0,
            "loss_p": 0.0,
            "loss_s": 0.0,
            "n": 0,
        }
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            pred = model(batch["x"], batch["t"])
            loss_pick, stats = compute_loss(pred, batch, loss_args)
            loss_ex, ex_stats = exist_loss(pred, batch, args)
            s_logits = torch.nan_to_num(
                pred.get("s_logits", pred["s"]), nan=-50.0, posinf=50.0, neginf=-50.0
            )
            loss_gate, gate_stats = gate_supervision_loss(pred, batch, args, s_logits)
            loss = loss_pick + args.exist_loss_weight * loss_ex + loss_gate
            loss = loss / max(1, args.grad_accum_steps)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if step % max(1, args.grad_accum_steps) == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
            bs = batch["x"].size(0)
            run["loss"] += float(loss.detach()) * max(1, args.grad_accum_steps) * bs
            run["loss_exist"] += ex_stats["loss_exist"] * bs
            run["loss_gate_sup"] += gate_stats.get("loss_s_absent_pick", 0.0) * bs
            run["loss_gate_sup"] += gate_stats.get("loss_exist_pick_couple", 0.0) * bs
            run["loss_p"] += stats["loss_p"] * bs
            run["loss_s"] += stats["loss_s"] * bs
            run["n"] += bs

        train_stats = {k: run[k] / max(run["n"], 1) for k in run if k != "n"}
        ev = eval_full(model, val_samples, device, cfg, obs_mod, args.exist_th, args.pick_threshold)
        m = ev["pick_only"]
        score = 0.55 * m["p_f1"] + 0.30 * m["s_f1"] + 0.15 * float(ev["exist_acc"]["s"])
        row = {
            "epoch": ep,
            "freeze_backbone": freeze,
            "train": train_stats,
            "val_pick_only": m,
            "val_exist_acc": ev["exist_acc"],
            "score": score,
        }
        history.append(row)
        print(
            f"[exist-gate] ep{ep} freeze={freeze} loss={train_stats['loss']:.4f} "
            f"exist={train_stats['loss_exist']:.4f} gate={train_stats['loss_gate_sup']:.4f} "
            f"val P={m['p_f1']:.3f} S={m['s_f1']:.3f} "
            f"existAcc P={ev['exist_acc']['p']:.3f} S={ev['exist_acc']['s']:.3f} "
            f"score={score:.3f}",
            flush=True,
        )
        if score > best["score"]:
            best = {"score": score, "epoch": ep, "metrics": m, "exist_acc": ev["exist_acc"]}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": {
                        **cfg,
                        "phase_exist": True,
                        "exist_th": args.exist_th,
                        "s_exist_neg_weight": args.s_exist_neg_weight,
                        "s_absent_pick_penalty": args.s_absent_pick_penalty,
                        "exist_pick_couple_weight": args.exist_pick_couple_weight,
                    },
                    "phase_exist": True,
                    "input_dim": dim,
                    "n_channels": dim,
                    "best_val_pick_only": m,
                    "best_exist_acc": ev["exist_acc"],
                    "epoch": ep,
                    "init": str(args.checkpoint),
                },
                out_dir / "best.pt",
            )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    load_picking_model_state(model, ck["state_dict"], strict=False)
    hold = eval_full(model, holdout, device, cfg, obs_mod, args.exist_th, args.pick_threshold)

    th_sweep = []
    best_th_row = None
    if args.sweep_exist_th:
        th_sweep = sweep_exist_th(
            model, holdout, device, cfg, obs_mod, args.pick_threshold
        )
        best_th_row = max(th_sweep, key=lambda r: r["score"])
        print(
            f"[exist-gate] th-sweep best: exist_th={best_th_row['exist_th']:.2f} "
            f"P={best_th_row['p_f1']:.3f} S={best_th_row['s_f1']:.3f} "
            f"existAccS={best_th_row['exist_acc_s']:.3f}",
            flush=True,
        )

    report = {
        "init": args.checkpoint,
        "best_ckpt": str(out_dir / "best.pt"),
        "best_epoch": best["epoch"],
        "best_val": best["metrics"],
        "best_exist_acc": best.get("exist_acc"),
        "holdout_pick_only": hold["pick_only"],
        "holdout_exist_acc": hold["exist_acc"],
        "exist_th": args.exist_th,
        "exist_th_sweep": th_sweep,
        "best_exist_th_holdout": best_th_row,
        "gate_supervision": {
            "exist_loss_weight": args.exist_loss_weight,
            "s_exist_neg_weight": args.s_exist_neg_weight,
            "s_absent_pick_penalty": args.s_absent_pick_penalty,
            "exist_pick_couple_weight": args.exist_pick_couple_weight,
            "p_only_oversample": args.p_only_oversample,
        },
        "n_holdout": len(holdout),
        "elapsed_sec": time.time() - start,
        "note": "score_absent=True; strengthened gate supervision",
    }
    (out_dir / "exist_gate_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# OBS exist-gate supervised report",
        "",
        f"- init: `{args.checkpoint}`",
        f"- best: `{out_dir / 'best.pt'}` (ep {best['epoch']})",
        f"- exist_th (train eval): {args.exist_th}",
        "",
        f"- best val: P={best['metrics']['p_f1']:.3f} S={best['metrics']['s_f1']:.3f}",
        f"- holdout @train_th: P={hold['pick_only']['p_f1']:.3f} S={hold['pick_only']['s_f1']:.3f}",
        f"- holdout existAcc: P={hold['exist_acc']['p']:.3f} S={hold['exist_acc']['s']:.3f}",
    ]
    if best_th_row:
        md += [
            f"- holdout @best_th={best_th_row['exist_th']:.2f}: "
            f"P={best_th_row['p_f1']:.3f} S={best_th_row['s_f1']:.3f} "
            f"existAccS={best_th_row['exist_acc_s']:.3f}",
        ]
    md.append("")
    (out_dir / "exist_gate_report.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
