#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate pattern-routed picking vs dense baseline (accuracy + latency).

Coarse pass extracts features (optionally NC-bypassed) → nearest prototype →
policy (skip / crop / bypass NC) → fine forward. High-confidence fines update
the library (EMA) when --feedback is set.

Example:
  PYTHONPATH=. python tools/eval_pattern_routed_picking.py \\
    --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt \\
    --library outputs/pattern_library_run28/pattern_library.json \\
    --max-event 500 --max-noise 200 --feedback
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.pattern_library import (
    PatternLibrary,
    PatternPolicy,
    RouteDecision,
    apply_route_crop,
    downsample_trace,
    extract_pattern_features,
)
from hnf.picking_metrics import apply_p_before_s_constraint
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model
from tools.train_stead_picking import move_batch_to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pattern-routed vs dense picking eval")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument(
        "--library",
        default="outputs/pattern_library_run28/pattern_library.json",
    )
    p.add_argument("--output-dir", default="outputs/pattern_routed_eval")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--max-event", type=int, default=500)
    p.add_argument("--max-noise", type=int, default=200)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--coarse-bypass-nc", action="store_true", default=True)
    p.add_argument("--no-coarse-bypass-nc", action="store_false", dest="coarse_bypass_nc")
    p.add_argument(
        "--coarse-len",
        type=int,
        default=0,
        help="0 = use the value stored in the library",
    )
    p.add_argument(
        "--det-gate",
        action="store_true",
        help="cheap det-only pass first; only run the full forward when it fires",
    )
    p.add_argument("--gate-threshold", type=float, default=0.5)
    p.add_argument("--gate-bypass-nc", action="store_true", default=True)
    p.add_argument("--no-gate-bypass-nc", action="store_false", dest="gate_bypass_nc")
    p.add_argument("--feedback", action="store_true")
    p.add_argument("--confirm-peak", type=float, default=0.45)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


def _peak(probs: torch.Tensor, thr: float, seq_len: int) -> tuple[int | None, float]:
    p = probs.reshape(-1)
    m = float(p.max().item())
    if m < thr:
        return None, m
    return int(p.argmax().item()), m


def _hit(pred_idx: int | None, gt_idx: int, tol_bins: int) -> bool:
    if pred_idx is None or gt_idx < 0:
        return False
    return abs(pred_idx - gt_idx) <= tol_bins


@torch.no_grad()
def run_dense(model, x, t, pick_th: float):
    out = model(x, t if t.dim() == 2 else t[0])
    p = torch.sigmoid(out["p"])
    s = torch.sigmoid(out["s"])
    p, s = apply_p_before_s_constraint(p, s, pick_th)
    det = torch.sigmoid(out["det"])
    if det.dim() > 1:
        det = det.amax(dim=-1)
    return p, s, float(det[0].item()), out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    lib = PatternLibrary.load(args.library)
    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    ds = STEADPickingDataset(
        args.split,
        seq_len=args.seq_len,
        max_event_traces=args.max_event,
        max_noise_traces=args.max_noise,
        seed=1,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    tol_bins = max(1, int(round(args.tol_sec / 60.0 * (args.seq_len - 1))))
    coarse_len = args.coarse_len if args.coarse_len > 0 else lib.coarse_len

    stats = {
        "dense": {"p_tp": 0, "p_fp": 0, "p_fn": 0, "s_tp": 0, "s_fp": 0, "s_fn": 0, "ms": 0.0, "n": 0},
        "routed": {
            "p_tp": 0,
            "p_fp": 0,
            "p_fn": 0,
            "s_tp": 0,
            "s_fp": 0,
            "s_fn": 0,
            "ms": 0.0,
            "coarse_ms": 0.0,
            "fine_ms": 0.0,
            "n": 0,
            "skipped": 0,
            "cropped": 0,
            "bypass_nc": 0,
            "by_pattern": {},
        },
    }

    def add_prf(bucket, phase, hit, pred_on, gt_on):
        if gt_on and hit:
            bucket[f"{phase}_tp"] += 1
        elif pred_on and not gt_on:
            bucket[f"{phase}_fp"] += 1
        elif gt_on and not hit:
            bucket[f"{phase}_fn"] += 1
        elif pred_on and gt_on and not hit:
            bucket[f"{phase}_fp"] += 1
            bucket[f"{phase}_fn"] += 1

    for bi, batch in enumerate(loader):
        if args.max_batches > 0 and bi >= args.max_batches:
            break
        batch = move_batch_to_device(batch, device)
        x = batch["x"]
        t = batch["t"][0] if batch["t"].dim() == 3 else batch["t"]
        is_event = float(batch["det"][0].item()) > 0.5
        p_gt = int(batch["p_idx"][0].item()) if is_event else -1
        s_gt = int(batch["s_idx"][0].item()) if is_event and float(batch["s_valid"][0].item()) > 0.5 else -1

        # --- dense baseline ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.bypass_noise_cancel = False
        p_d, s_d, _det_d, _ = run_dense(model, x, t, args.pick_threshold)
        if device.type == "cuda":
            torch.cuda.synchronize()
        stats["dense"]["ms"] += (time.perf_counter() - t0) * 1000.0
        stats["dense"]["n"] += 1
        pi, _ = _peak(p_d[0], args.pick_threshold, args.seq_len)
        si, _ = _peak(s_d[0], args.pick_threshold, args.seq_len)
        add_prf(stats["dense"], "p", _hit(pi, p_gt, tol_bins), pi is not None, p_gt >= 0)
        add_prf(stats["dense"], "s", _hit(si, s_gt, tol_bins), si is not None, s_gt >= 0)

        # --- routed ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if args.det_gate:
            # Native-grid det-only pass: the model does not survive resampling,
            # so the cheap tier drops the P/S branches instead of the grid.
            det_logit = model.forward_det_only(
                x, t, bypass_noise_cancel=bool(args.gate_bypass_nc)
            )
            gate_p = float(torch.sigmoid(det_logit).reshape(-1)[0].item())
            gate_open = gate_p >= args.gate_threshold
            coarse = {"det": gate_p, "p_sec": -1.0, "s_sec": -1.0}
            pol = PatternPolicy(name="det_gate_skip" if not gate_open else "det_gate_full")
            pol.skip_pick = not gate_open
            decision = RouteDecision(-1, pol.name, 0.0, pol, coarse)
        else:
            x_c, t_c = downsample_trace(x, t, coarse_len, window_sec=60.0)
            coarse = extract_pattern_features(
                model,
                x_c,
                t_c,
                pick_threshold=args.pick_threshold,
                bypass_noise_cancel=bool(args.coarse_bypass_nc),
            )
            decision = lib.route(coarse)
            pol = decision.policy
            # Safety net: never skip when coarse det says "event".
            if pol.skip_pick and float(coarse.get("det", 0.0)) >= args.det_threshold:
                pol = PatternPolicy(
                    name=f"{pol.name}_det_keep",
                    skip_pick=False,
                    bypass_noise_cancel=False,
                    crop_around="none",
                )
                decision = RouteDecision(
                    decision.pattern_id, decision.name, decision.distance, pol, coarse
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        coarse_ms = (time.perf_counter() - t1) * 1000.0
        stats["routed"]["coarse_ms"] += coarse_ms

        stats["routed"]["by_pattern"][decision.name] = (
            stats["routed"]["by_pattern"].get(decision.name, 0) + 1
        )

        t2 = time.perf_counter()
        if pol.skip_pick:
            stats["routed"]["skipped"] += 1
            # predict absent
            add_prf(stats["routed"], "p", False, False, p_gt >= 0)
            add_prf(stats["routed"], "s", False, False, s_gt >= 0)
            confirmed = (p_gt < 0) and (s_gt < 0)
            ppeak = speak = 0.0
        else:
            fine_len = int(pol.crop_len) if (pol.crop_around != "none" and pol.crop_len) else args.seq_len
            x_f, t_f, shift = apply_route_crop(
                x, t, coarse, pol, window_sec=60.0, out_len=fine_len
            )
            if pol.crop_around != "none":
                stats["routed"]["cropped"] += 1
            if pol.bypass_noise_cancel:
                stats["routed"]["bypass_nc"] += 1
            model.bypass_noise_cancel = bool(pol.bypass_noise_cancel)
            p_r, s_r, _det_r, _ = run_dense(model, x_f, t_f, args.pick_threshold)
            model.bypass_noise_cancel = False
            pi_r, ppeak = _peak(p_r[0], args.pick_threshold, fine_len)
            si_r, speak = _peak(s_r[0], args.pick_threshold, fine_len)
            # crop-local index (fine_len grid over `dur` seconds) -> full-window bin
            dur = float(t_f[-1, 0].item()) if t_f.numel() else 60.0

            def remap_to_full(idx: int | None) -> int | None:
                if idx is None:
                    return None
                sec_global = shift + float(idx) / max(fine_len - 1, 1) * dur
                return int(round(sec_global / 60.0 * (args.seq_len - 1)))

            pi_r = remap_to_full(pi_r)
            si_r = remap_to_full(si_r)

            add_prf(stats["routed"], "p", _hit(pi_r, p_gt, tol_bins), pi_r is not None, p_gt >= 0)
            add_prf(stats["routed"], "s", _hit(si_r, s_gt, tol_bins), si_r is not None, s_gt >= 0)
            confirmed = (ppeak >= args.confirm_peak) or (speak >= args.confirm_peak)

        if device.type == "cuda":
            torch.cuda.synchronize()
        stats["routed"]["fine_ms"] += (time.perf_counter() - t2) * 1000.0
        stats["routed"]["ms"] += coarse_ms + (time.perf_counter() - t2) * 1000.0
        stats["routed"]["n"] += 1

        # Feedback is an offline bookkeeping step, so it stays out of the timing.
        # Only bump confirm/reject counts — do NOT EMA-update centres with fine
        # features (that drifts prototypes off the coarse routing manifold).
        if args.feedback:
            lib.update_from_fine(
                decision.pattern_id,
                coarse,
                confirmed=confirmed,
                update_center=False,
            )

        if (bi + 1) % 100 == 0:
            print(f"[pattern-route] {bi+1}/{len(ds)}", flush=True)

    def f1(b, phase):
        tp, fp, fn = b[f"{phase}_tp"], b[f"{phase}_fp"], b[f"{phase}_fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-8),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    report = {
        "dense": {
            "p": f1(stats["dense"], "p"),
            "s": f1(stats["dense"], "s"),
            "ms_per_trace": stats["dense"]["ms"] / max(stats["dense"]["n"], 1),
            "n": stats["dense"]["n"],
        },
        "routed": {
            "p": f1(stats["routed"], "p"),
            "s": f1(stats["routed"], "s"),
            "ms_per_trace": stats["routed"]["ms"] / max(stats["routed"]["n"], 1),
            "coarse_ms_per_trace": stats["routed"]["coarse_ms"] / max(stats["routed"]["n"], 1),
            "fine_ms_per_trace": stats["routed"]["fine_ms"] / max(stats["routed"]["n"], 1),
            "speedup_vs_dense": (
                stats["dense"]["ms"] / max(stats["routed"]["ms"], 1e-9)
            ),
            "coarse_len": coarse_len,
            "n": stats["routed"]["n"],
            "skipped": stats["routed"]["skipped"],
            "cropped": stats["routed"]["cropped"],
            "bypass_nc": stats["routed"]["bypass_nc"],
            "by_pattern": stats["routed"]["by_pattern"],
        },
        "library_summary": lib.summary(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.feedback:
        lib.save(out_dir / "pattern_library_updated.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
