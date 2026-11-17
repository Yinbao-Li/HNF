#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A: same-subset ablation of dump path vs manuscript eval_ethz_cross decoder.

  python tools/phase2_diagnose_dump_vs_manuscript.py --dataset ethz --max-traces 800 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

_REPO = Path(__file__).resolve().parents[1]
for sub in ("", "scripts/paper"):
    if str(_REPO / sub) not in sys.path:
        sys.path.insert(0, str(_REPO / sub))

from hnf.picking_metrics import apply_p_before_s_constraint  # noqa: E402
from tools.eval_ethz_cross import (  # noqa: E402
    P_COLS,
    S_COLS,
    T,
    cut_window,
    first_arrival,
    load_dataset,
)
from tools.eval_shift_invariance import run_c2f  # noqa: E402
from tools.sweep_seisbench_threshold import metrics_at  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--dataset", choices=["ethz", "obs"], default="ethz")
    p.add_argument("--max-traces", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("results/phase2/2_7_ci/diagnose_dump_vs_ms.json"))
    return p.parse_args()


def build_windows(dataset_name, max_traces, seed):
    data = load_dataset(dataset_name, "test")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))[: max_traces if max_traces > 0 else len(data)]
    waves, y = [], {"p_idx": [], "s_idx": [], "p_valid": [], "s_valid": [], "is_event": []}
    for i in order:
        w, meta = data.get_sample(int(i))
        p = first_arrival(meta, P_COLS)
        if p is None or w.shape[0] != 3:
            continue
        w = np.nan_to_num(np.asarray(w, dtype=np.float32))
        if dataset_name == "obs" and not (w.std(axis=1) > 0).all():
            continue
        s = first_arrival(meta, S_COLS)
        p = int(round(p))
        tp = int(round(rng.uniform(5.0, 30.0) * 100))
        win, _, _ = cut_window(w, p, tp)
        s_rel = None if s is None else int(round(s)) - p + tp
        s_ok = s_rel is not None and 50 <= s_rel <= T - 50 and s_rel > tp
        waves.append(win)
        y["p_idx"].append(tp)
        y["s_idx"].append(s_rel if s_ok else 0)
        y["p_valid"].append(True)
        y["s_valid"].append(bool(s_ok))
        y["is_event"].append(True)
    waves = np.stack(waves)
    sos = butter(4, [1.0, 45.0], btype="bandpass", fs=100.0, output="sos")
    waves = sosfiltfilt(sos, waves, axis=-1).astype(np.float32)
    y = {k: torch.tensor(v) for k, v in y.items()}
    return waves, y


def dump_style_f1(det, p_curve, s_curve, y, pick_th, det_th, tol):
    """Broken Phase-2 dump F1: wrong-pick counted as FP and FN."""
    p_max, p_idx = p_curve.max(-1)
    s_max, s_idx = s_curve.max(-1)
    p_pred = (det >= det_th) & (p_max >= pick_th)
    s_pred = (det >= det_th) & (s_max >= pick_th)
    out = {}
    for phase, pred, idx, gt, valid in (
        ("p", p_pred, p_idx, y["p_idx"], y["p_valid"]),
        ("s", s_pred, s_idx, y["s_idx"], y["s_valid"]),
    ):
        pred_b = pred.cpu().numpy().astype(bool)
        idx_np = idx.cpu().numpy()
        gt_np = gt.cpu().numpy()
        valid_b = valid.cpu().numpy().astype(bool)
        hit = pred_b & valid_b & (np.abs(idx_np - gt_np) <= tol)
        tp = (hit & valid_b & pred_b).sum()
        fp = (pred_b & ~hit).sum()
        fn = (valid_b & ~hit).sum()
        out[phase] = float(2 * tp / max(1, 2 * tp + fp + fn))
    return out


@torch.no_grad()
def infer_variants(model, waves, device, bs, det_th, pick_th):
    """Return dict of (det, p, s) tensors for decoder ablations."""
    variants = {}

    # A: manuscript run_c2f
    det, p, s = run_c2f(model, waves, device, bs, det_th)
    variants["ms_run_c2f"] = (det, p.float(), s.float())

    # B: dump-like: NO normalize, t as (B,T) flat arange scale
    dets, ps, ss = [], [], []
    for a in range(0, len(waves), bs):
        chunk = waves[a : a + bs]
        x = torch.from_numpy(chunk).to(device).transpose(1, 2)  # no normalize
        t = torch.arange(x.shape[1], device=device).float().unsqueeze(0).expand(x.shape[0], -1) * (
            60.0 / x.shape[1]
        )
        out = model(x, t)
        det_prob = torch.sigmoid(out["det"])
        gate = (det_prob >= det_th).float().unsqueeze(-1)
        pp = torch.sigmoid(out["p"]) * gate
        ss_ = torch.sigmoid(out["s"]) * gate
        pp, ss_ = apply_p_before_s_constraint(pp, ss_, pick_th)
        dets.append(det_prob.cpu())
        ps.append(pp.cpu())
        ss.append(ss_.cpu())
    variants["dump_raw_no_norm"] = (torch.cat(dets), torch.cat(ps), torch.cat(ss))

    # C: dump + normalize (same as run_c2f), keep flat t
    dets, ps, ss = [], [], []
    for a in range(0, len(waves), bs):
        chunk = waves[a : a + bs]
        x = torch.from_numpy(chunk).to(device)
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        x = x.transpose(1, 2).contiguous()
        t = torch.arange(x.shape[1], device=device).float().unsqueeze(0).expand(x.shape[0], -1) * (
            60.0 / x.shape[1]
        )
        out = model(x, t)
        det_prob = torch.sigmoid(out["det"])
        gate = (det_prob >= det_th).float().unsqueeze(-1)
        pp = torch.sigmoid(out["p"]) * gate
        ss_ = torch.sigmoid(out["s"]) * gate
        pp, ss_ = apply_p_before_s_constraint(pp, ss_, pick_th)
        dets.append(det_prob.cpu())
        ps.append(pp.cpu())
        ss.append(ss_.cpu())
    variants["dump_plus_norm_flat_t"] = (torch.cat(dets), torch.cat(ps), torch.cat(ss))

    # D: normalize + manuscript linspace t (B,T,1)
    dets, ps, ss = [], [], []
    for a in range(0, len(waves), bs):
        chunk = waves[a : a + bs]
        x = torch.from_numpy(chunk).to(device)
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        x = x.transpose(1, 2).contiguous()
        t = torch.linspace(0.0, 60.0, T, device=device).view(1, T, 1).expand(x.size(0), -1, -1)
        out = model(x, t)
        det_prob = torch.sigmoid(out["det"])
        gate = (det_prob >= det_th).float().unsqueeze(-1)
        pp = torch.sigmoid(out["p"]) * gate
        ss_ = torch.sigmoid(out["s"]) * gate
        # manuscript metrics_at applies p_before_s inside; run_c2f does NOT
        dets.append(det_prob.cpu())
        ps.append(pp.cpu())
        ss.append(ss_.cpu())
    variants["norm_linspace_t_no_pbs"] = (torch.cat(dets), torch.cat(ps), torch.cat(ss))

    return variants


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, _, _ = load_model(args.ours, device)
    waves, y = build_windows(args.dataset, args.max_traces, args.seed)
    print(f"[{args.dataset}] n={len(waves)} S-valid={int(y['s_valid'].sum())}", flush=True)

    variants = infer_variants(model, waves, device, args.batch_size, args.det_threshold, args.pick_threshold)
    report = {
        "checkpoint": str(args.ours),
        "dataset": args.dataset,
        "n": len(waves),
        "n_with_s": int(y["s_valid"].sum()),
        "seed": args.seed,
        "variants": {},
        "per_sample_ms_vs_dump_raw": {},
    }

    for name, (det, p, s) in variants.items():
        row = {"metrics_at": {}, "dump_style_f1": {}}
        for tol_sec, tol in ((0.1, 10), (0.5, 50)):
            m = metrics_at(det, p, s, y, "eqt", args.pick_threshold, args.det_threshold, tol, device)
            row["metrics_at"][f"{tol_sec:g}"] = {"p_f1": m["p_f1"], "s_f1": m["s_f1"], "det_f1": m["det_f1"]}
            ds = dump_style_f1(det, p, s, y, args.pick_threshold, args.det_threshold, tol)
            row["dump_style_f1"][f"{tol_sec:g}"] = ds
            print(
                f"  {name:28s} metrics_at@{tol_sec:g} P={m['p_f1']:.4f} S={m['s_f1']:.4f} | "
                f"dumpF1 P={ds['p']:.4f} S={ds['s']:.4f}",
                flush=True,
            )
        report["variants"][name] = row

    # Per-sample: ms_run_c2f vs dump_raw on same windows (tol 0.5 P)
    det_m, p_m, s_m = variants["ms_run_c2f"]
    det_d, p_d, s_d = variants["dump_raw_no_norm"]
    _, pi_m = p_m.max(-1)
    _, pi_d = p_d.max(-1)
    pm_m = p_m.max(-1).values
    pm_d = p_d.max(-1).values
    pred_m = (det_m >= args.det_threshold) & (pm_m >= args.pick_threshold)
    pred_d = (det_d >= args.det_threshold) & (pm_d >= args.pick_threshold)
    gt = y["p_idx"]
    hit_m = pred_m & (torch.abs(pi_m - gt) <= 50)
    hit_d = pred_d & (torch.abs(pi_d - gt) <= 50)
    report["per_sample_ms_vs_dump_raw"] = {
        "n": len(waves),
        "ms_p_hit_rate_tol0.5": float(hit_m.float().mean()),
        "dump_raw_p_hit_rate_tol0.5": float(hit_d.float().mean()),
        "agree_both_hit": float((hit_m & hit_d).float().mean()),
        "ms_hit_dump_miss": float((hit_m & ~hit_d).float().mean()),
        "dump_hit_ms_miss": float((hit_d & ~hit_m).float().mean()),
        "mean_abs_idx_diff_when_both_pred": float(
            torch.abs(pi_m - pi_d)[pred_m & pred_d].float().mean().item()
        )
        if bool((pred_m & pred_d).any())
        else None,
        "frac_pred_ms": float(pred_m.float().mean()),
        "frac_pred_dump_raw": float(pred_d.float().mean()),
    }
    print("[per-sample]", json.dumps(report["per_sample_ms_vs_dump_raw"], indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output}", flush=True)


if __name__ == "__main__":
    main()
