#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast baseline vs adaptive pick comparison (no threshold sweeps)."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analyze_stead_picking import (
    _adaptive_pick_threshold,
    _noise_ratio_from_outputs,
    _rerank_peak,
    _subsample_indices,
    load_model,
    pick_prf,
)
from hnf.picking_metrics import apply_p_before_s_constraint, idx_to_sec, tolerance_bins
from hnf.stead_picking_dataset import STEADPickingDataset
from train_stead_picking import move_batch_to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline vs adaptive pick eval")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output", default="outputs/run20/20_wrongpeak_sharp/adaptive_pick_eval.json")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--max-events", type=int, default=2000)
    p.add_argument("--max-noise", type=int, default=500)
    p.add_argument("--subset-seed", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--noise-ref", type=float, default=0.239, help="noise_ratio pivot (≈ upper tertile)")
    p.add_argument("--noise-slope", type=float, default=0.25, help="Lower threshold when noise_ratio > ref")
    p.add_argument("--min-pick-threshold", type=float, default=0.22)
    p.add_argument("--max-pick-threshold", type=float, default=0.35)
    p.add_argument("--rho-topk", type=int, default=3)
    p.add_argument("--rho-weight", type=float, default=0.10)
    p.add_argument("--rho-peak-penalty", type=float, default=0.08)
    return p.parse_args()


def _score_pick(
    *,
    counts: Counter,
    breakdown: Counter,
    head: str,
    peak: float,
    pred: int,
    gt: int,
    threshold: float,
    det_ok: bool,
    tol: int,
    seq_len: int,
    err_sum: dict,
    tp_n: dict,
) -> None:
    has_peak = peak >= threshold
    if not det_ok:
        counts["fn"] += 1
        breakdown[f"{head}_fn_missed_det"] += 1
        return
    if has_peak and abs(pred - gt) <= tol:
        counts["tp"] += 1
        err_sum[head] += abs(idx_to_sec(pred, seq_len) - idx_to_sec(gt, seq_len))
        tp_n[head] += 1
        return
    counts["fn"] += 1
    if has_peak:
        breakdown[f"{head}_fn_wrong_peak"] += 1
    else:
        breakdown[f"{head}_fn_no_peak"] += 1


def _clean_boost_threshold(
    base_threshold: float,
    noise_ratio: Optional[float],
    *,
    noise_ref: float,
    boost: float,
    min_threshold: float,
) -> float:
    """Only lower threshold on cleaner traces (low noise_ratio)."""
    if noise_ratio is None:
        return base_threshold
    if noise_ratio >= noise_ref:
        return base_threshold
    return max(min_threshold, base_threshold - boost * (noise_ref - noise_ratio))


def _local_max_indices(probs: torch.Tensor) -> list[int]:
    vals = probs.detach().cpu().numpy()
    idxs = []
    n = len(vals)
    for i in range(n):
        left = vals[i - 1] if i > 0 else -1.0
        right = vals[i + 1] if i + 1 < n else -1.0
        if vals[i] >= left and vals[i] >= right:
            idxs.append(i)
    if not idxs:
        idxs = [int(np.argmax(vals))]
    return idxs


def _rho_local_pick(
    probs: torch.Tensor,
    rho: torch.Tensor,
    threshold: float,
    *,
    rho_weight: float,
    min_frac: float = 0.55,
) -> tuple[float, int]:
    """Among local maxima above a soft floor, maximize p + w*rho."""
    peak = float(probs.max().item())
    pred = int(probs.argmax().item())
    if peak < threshold:
        return peak, pred
    floor = max(threshold * min_frac, threshold - 0.15)
    rho_norm = rho / rho.amax().clamp_min(1e-8)
    best_score = float("-inf")
    best_peak = peak
    best_idx = pred
    for idx in _local_max_indices(probs):
        val = float(probs[idx].item())
        if val < floor:
            continue
        score = val + rho_weight * float(rho_norm[idx].item())
        if score > best_score:
            best_score = score
            best_peak = val
            best_idx = idx
    return best_peak, best_idx


def _disagree_rerank(
    probs: torch.Tensor,
    rho: torch.Tensor,
    threshold: float,
    *,
    gap_bins: int,
    rho_weight: float,
    topk: int,
    rho_peak_penalty: float,
) -> tuple[float, int]:
    peak, pred = float(probs.max().item()), int(probs.argmax().item())
    if peak < threshold:
        return peak, pred
    rho_peak = int(rho.argmax().item())
    if abs(pred - rho_peak) <= gap_bins:
        return peak, pred
    return _rerank_peak(
        probs,
        rho,
        threshold,
        topk=topk,
        rho_weight=rho_weight,
        rho_peak_penalty=rho_peak_penalty,
    )


def _empty_strategy() -> dict:
    return {
        "counts": {"p": Counter(), "s": Counter()},
        "breakdown": Counter(),
        "err_sum": {"p": 0.0, "s": 0.0},
        "tp_n": {"p": 0, "s": 0},
    }


@torch.no_grad()
def evaluate_adaptive(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[adaptive-eval] device={device}", flush=True)
    model, _ckpt_args = load_model(Path(args.checkpoint), device)
    ds = STEADPickingDataset("test", seq_len=args.seq_len, load_geometry=False)
    indices = _subsample_indices(ds, args.max_events, args.max_noise, args.subset_seed)
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    tol = tolerance_bins(args.seq_len, args.tol_sec)
    n_ev = sum(1 for i in indices if ds.refs[i].is_event == 1)
    n_nz = len(indices) - n_ev
    print(
        f"[adaptive-eval] n={len(indices)} events={n_ev} noise={n_nz} "
        f"batches={len(loader)} batch_size={args.batch_size}",
        flush=True,
    )

    strategy_names = [
        "baseline",
        "fixed_th_025",
        "noise_th_linear",
        "clean_boost_th",
        "rho_topk_rerank",
        "rho_local_max",
        "disagree_rerank",
        "clean_boost_plus_local_rho",
    ]
    strategies = {name: _empty_strategy() for name in strategy_names}

    t0 = time.time()
    for bi, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["x"], batch["t"])
        det_probs = torch.sigmoid(outputs["det"])
        if det_probs.dim() > 1:
            det_probs = det_probs.amax(dim=-1)
        p_probs = torch.sigmoid(outputs["p"])
        s_probs = torch.sigmoid(outputs["s"])
        rho = outputs["rho"]
        noise_ratio = _noise_ratio_from_outputs(outputs)
        det_true = batch["det"] > 0.5
        bsz = batch["x"].size(0)

        for i in range(bsz):
            is_event = bool(det_true[i].item())
            det_ok = bool((det_probs[i] >= args.det_threshold).item())
            trace_noise: Optional[float] = None if noise_ratio is None else float(noise_ratio[i].item())

            th_map = {
                "baseline": args.pick_threshold,
                "fixed_th_025": 0.25,
                "noise_th_linear": _adaptive_pick_threshold(
                    args.pick_threshold,
                    trace_noise,
                    noise_ref=args.noise_ref,
                    noise_slope=args.noise_slope,
                    min_threshold=args.min_pick_threshold,
                    max_threshold=args.max_pick_threshold,
                ),
                "clean_boost_th": _clean_boost_threshold(
                    args.pick_threshold,
                    trace_noise,
                    noise_ref=args.noise_ref,
                    boost=0.25,
                    min_threshold=args.min_pick_threshold,
                ),
                "rho_topk_rerank": args.pick_threshold,
                "rho_local_max": args.pick_threshold,
                "disagree_rerank": args.pick_threshold,
                "clean_boost_plus_local_rho": _clean_boost_threshold(
                    args.pick_threshold,
                    trace_noise,
                    noise_ref=args.noise_ref,
                    boost=0.25,
                    min_threshold=args.min_pick_threshold,
                ),
            }

            for name, st in strategies.items():
                th = th_map[name]
                p_use, s_use = apply_p_before_s_constraint(
                    p_probs[i : i + 1], s_probs[i : i + 1], th
                )
                p_use, s_use = p_use[0], s_use[0]

                for head, probs, idx_k, valid_k in [
                    ("p", p_use, "p_idx", "p_valid"),
                    ("s", s_use, "s_idx", "s_valid"),
                ]:
                    rho_w = args.rho_weight if head == "p" else args.rho_weight * 0.6
                    if name in {"rho_topk_rerank"}:
                        peak, pred = _rerank_peak(
                            probs,
                            rho[i],
                            th,
                            topk=args.rho_topk,
                            rho_weight=rho_w,
                            rho_peak_penalty=args.rho_peak_penalty,
                        )
                    elif name in {"rho_local_max", "clean_boost_plus_local_rho"}:
                        peak, pred = _rho_local_pick(
                            probs, rho[i], th, rho_weight=rho_w, min_frac=0.55
                        )
                    elif name == "disagree_rerank":
                        peak, pred = _disagree_rerank(
                            probs,
                            rho[i],
                            th,
                            gap_bins=max(8, tol),
                            rho_weight=rho_w,
                            topk=max(args.rho_topk, 5),
                            rho_peak_penalty=args.rho_peak_penalty,
                        )
                    else:
                        peak = float(probs.max().item())
                        pred = int(probs.argmax().item())

                    if is_event:
                        if batch[valid_k][i] <= 0:
                            continue
                        gt = int(batch[idx_k][i].item())
                        _score_pick(
                            counts=st["counts"][head],
                            breakdown=st["breakdown"],
                            head=head,
                            peak=peak,
                            pred=pred,
                            gt=gt,
                            threshold=th,
                            det_ok=det_ok,
                            tol=tol,
                            seq_len=args.seq_len,
                            err_sum=st["err_sum"],
                            tp_n=st["tp_n"],
                        )
                    else:
                        if peak >= th:
                            st["counts"][head]["fp"] += 1

        if (bi + 1) % 20 == 0 or (bi + 1) == len(loader):
            elapsed = time.time() - t0
            rate = (bi + 1) / max(elapsed, 1e-6)
            eta = (len(loader) - bi - 1) / max(rate, 1e-6)
            print(
                f"[adaptive-eval] batch {bi+1}/{len(loader)} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "n_eval": len(indices),
        "n_events": n_ev,
        "n_noise": n_nz,
        "config": {
            "seq_len": args.seq_len,
            "det_threshold": args.det_threshold,
            "pick_threshold": args.pick_threshold,
            "tol_sec": args.tol_sec,
            "noise_ref": args.noise_ref,
            "noise_slope": args.noise_slope,
            "min_pick_threshold": args.min_pick_threshold,
            "max_pick_threshold": args.max_pick_threshold,
            "rho_topk": args.rho_topk,
            "rho_weight": args.rho_weight,
            "rho_peak_penalty": args.rho_peak_penalty,
            "max_events": args.max_events,
            "max_noise": args.max_noise,
            "subset_seed": args.subset_seed,
        },
        "strategies": {},
    }
    base_p = None
    base_s = None
    for name, st in strategies.items():
        p_m = pick_prf(st["counts"]["p"]["tp"], st["counts"]["p"]["fp"], st["counts"]["p"]["fn"])
        s_m = pick_prf(st["counts"]["s"]["tp"], st["counts"]["s"]["fp"], st["counts"]["s"]["fn"])
        entry = {
            "p": p_m,
            "s": s_m,
            "p_mae_sec": st["err_sum"]["p"] / max(st["tp_n"]["p"], 1),
            "s_mae_sec": st["err_sum"]["s"] / max(st["tp_n"]["s"], 1),
            "error_breakdown": dict(st["breakdown"]),
        }
        if name == "baseline":
            base_p, base_s = p_m, s_m
        else:
            entry["delta_vs_baseline"] = {
                "p_f1": p_m["f1"] - base_p["f1"],
                "s_f1": s_m["f1"] - base_s["f1"],
                "p_recall": p_m["recall"] - base_p["recall"],
                "s_recall": s_m["recall"] - base_s["recall"],
                "p_precision": p_m["precision"] - base_p["precision"],
                "s_precision": s_m["precision"] - base_s["precision"],
            }
        report["strategies"][name] = entry
    return report


def main() -> None:
    args = parse_args()
    report = evaluate_adaptive(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[adaptive-eval] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
