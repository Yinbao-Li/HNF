#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Holdout FN/FP breakdown for OBS exist-gated picking models.

Categories (pick-only, score_absent for S):
  TP / FP_wrong_peak / FN_no_peak / FN_wrong_peak / FN_exist_gate
  + for S absent windows: TN_absent / FP_absent (false S on no-S)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.picking_metrics import apply_p_before_s_constraint, tolerance_bins
from tools.obs_matched_split import load_split_samples
from tools.train_obs_exist_gate import build_model_from_ckpt
from tools.train_obs_picking import filter_alive_channels, _load_obs_compare_module
from tools.train_stead_picking import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS holdout FN/FP analysis")
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_fromscratch_30ep/best.pt",
    )
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_exist_fromscratch_30ep/fnfp",
    )
    p.add_argument("--exist-th", type=float, default=0.60)
    p.add_argument("--pick-threshold", type=float, default=0.25)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _classify_phase(
    probs: torch.Tensor,
    valid: torch.Tensor,
    gt_idx: torch.Tensor,
    exist_prob: torch.Tensor | None,
    pick_th: float,
    exist_th: float,
    tol: int,
    score_absent: bool,
) -> Counter:
    c: Counter = Counter()
    peak, pred_idx = probs.max(dim=-1)
    for i in range(probs.size(0)):
        peak_ok = float(peak[i]) >= pick_th
        exist_ok = True
        if exist_prob is not None:
            exist_ok = float(exist_prob[i]) >= exist_th
        pred_exists = peak_ok and exist_ok
        has_gt = bool(valid[i].item())
        pred_i = int(pred_idx[i])
        if not has_gt:
            if score_absent:
                if pred_exists:
                    c["FP_absent"] += 1
                else:
                    c["TN_absent"] += 1
                    if peak_ok and not exist_ok:
                        c["TN_absent_exist_blocked"] += 1
                    elif not peak_ok:
                        c["TN_absent_no_peak"] += 1
            continue
        gt_i = int(gt_idx[i])
        within = abs(pred_i - gt_i) <= tol
        c["n_present"] += 1
        if pred_exists and within:
            c["TP"] += 1
        elif pred_exists and not within:
            c["FP_wrong_peak"] += 1
            c["FN_wrong_peak"] += 1  # miss GT + emit wrong
        elif not pred_exists:
            c["FN"] += 1
            if peak_ok and not exist_ok:
                c["FN_exist_gate"] += 1
            elif not peak_ok:
                c["FN_no_peak"] += 1
            else:
                c["FN_other"] += 1
            # peak location if any (for timing diagnostics)
            if peak_ok and not within:
                c["FN_peak_off_tol"] += 1
            elif peak_ok and within:
                c["FN_peak_ok_but_gated"] += 1
    return c


def _prf(c: Counter, *, score_absent: bool) -> dict:
    tp = c["TP"]
    fp = c["FP_wrong_peak"] + (c["FP_absent"] if score_absent else 0)
    fn = c["FN"] + c["FN_wrong_peak"]
    # FN_wrong_peak already counted with FP_wrong_peak as joint miss;
    # for F1: wrong-peak present windows contribute 1 FP + 1 FN (same as eval protocol)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "breakdown": dict(c),
    }


@torch.no_grad()
def analyze(model, samples, device, cfg, obs_mod, args) -> dict:
    model.eval()
    seq_len = int(cfg["seq_len"])
    window_sec = float(cfg["window_sec"])
    dim = int(cfg["input_dim"])
    tol = tolerance_bins(seq_len, args.tol_sec)
    p_c: Counter = Counter()
    s_c: Counter = Counter()
    err_bins_p: list[int] = []
    err_bins_s: list[int] = []

    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        x, t, p_idx, s_idx, p_valid, s_valid = obs_mod.to_hnf_batch(
            chunk, seq_len, window_sec, device, n_channels=dim
        )
        out = model(x, t)
        p_logits = torch.nan_to_num(out.get("p_logits", out["p"]), nan=-50.0)
        s_logits = torch.nan_to_num(out.get("s_logits", out["s"]), nan=-50.0)
        p_probs = torch.sigmoid(p_logits)
        s_probs = torch.sigmoid(s_logits)
        p_probs, s_probs = apply_p_before_s_constraint(p_probs, s_probs, args.pick_threshold)
        p_exist = s_exist = None
        if "p_exist" in out and "s_exist" in out:
            p_exist = torch.sigmoid(torch.nan_to_num(out["p_exist"], nan=-50.0))
            s_exist = torch.sigmoid(torch.nan_to_num(out["s_exist"], nan=-50.0))

        p_c.update(
            _classify_phase(
                p_probs, p_valid, p_idx, p_exist, args.pick_threshold, args.exist_th, tol, False
            )
        )
        s_c.update(
            _classify_phase(
                s_probs, s_valid, s_idx, s_exist, args.pick_threshold, args.exist_th, tol, True
            )
        )
        # timing errors on present windows (argmax vs GT, ungated)
        for i in range(len(chunk)):
            if float(p_valid[i]) > 0.5:
                err_bins_p.append(abs(int(p_probs[i].argmax()) - int(p_idx[i])))
            if float(s_valid[i]) > 0.5:
                err_bins_s.append(abs(int(s_probs[i].argmax()) - int(s_idx[i])))

    def _timing(errs: list[int]) -> dict:
        if not errs:
            return {}
        import numpy as np

        a = np.asarray(errs, dtype=np.float32)
        return {
            "mean_bins": float(a.mean()),
            "median_bins": float(np.median(a)),
            "within_tol": float((a <= tol).mean()),
            "n": len(errs),
        }

    return {
        "n_holdout": len(samples),
        "exist_th": args.exist_th,
        "pick_th": args.pick_threshold,
        "tol_bins": tol,
        "p": _prf(p_c, score_absent=False),
        "s": _prf(s_c, score_absent=True),
        "p_timing_ungated": _timing(err_bins_p),
        "s_timing_ungated": _timing(err_bins_s),
    }


def to_md(report: dict, ckpt: str) -> str:
    p, s = report["p"], report["s"]
    pb, sb = p["breakdown"], s["breakdown"]
    lines = [
        "# OBS FN/FP breakdown",
        "",
        f"- ckpt: `{ckpt}`",
        f"- n={report['n_holdout']} exist_th={report['exist_th']} pick_th={report['pick_th']}",
        "",
        "## P (present-only)",
        f"- F1={p['f1']:.3f}  P={p['precision']:.3f}  R={p['recall']:.3f}  "
        f"TP={p['tp']} FP={p['fp']} FN={p['fn']}",
        f"- FN_no_peak={pb.get('FN_no_peak', 0)}  FN_exist_gate={pb.get('FN_exist_gate', 0)}  "
        f"FP/FN_wrong_peak={pb.get('FP_wrong_peak', 0)}",
        f"- ungated timing within_tol={report['p_timing_ungated'].get('within_tol', 0):.3f} "
        f"mean_bins={report['p_timing_ungated'].get('mean_bins', 0):.1f}",
        "",
        "## S (score_absent)",
        f"- F1={s['f1']:.3f}  P={s['precision']:.3f}  R={s['recall']:.3f}  "
        f"TP={s['tp']} FP={s['fp']} FN={s['fn']}",
        f"- present: FN_no_peak={sb.get('FN_no_peak', 0)}  FN_exist_gate={sb.get('FN_exist_gate', 0)}  "
        f"wrong_peak={sb.get('FP_wrong_peak', 0)}  FN_peak_ok_but_gated={sb.get('FN_peak_ok_but_gated', 0)}",
        f"- absent: FP_absent={sb.get('FP_absent', 0)}  TN_absent={sb.get('TN_absent', 0)}  "
        f"(exist_blocked={sb.get('TN_absent_exist_blocked', 0)})",
        f"- ungated timing within_tol={report['s_timing_ungated'].get('within_tol', 0):.3f} "
        f"mean_bins={report['s_timing_ungated'].get('mean_bins', 0):.1f}",
        "",
    ]
    # quick diagnosis
    s_fp_abs = sb.get("FP_absent", 0)
    s_fp_wp = sb.get("FP_wrong_peak", 0)
    s_fn_np = sb.get("FN_no_peak", 0)
    s_fn_eg = sb.get("FN_exist_gate", 0)
    p_fn_np = pb.get("FN_no_peak", 0)
    p_wp = pb.get("FP_wrong_peak", 0)
    lines += ["## Diagnosis", ""]
    if p_wp + p_fn_np > 0:
        lines.append(
            f"- **P bottleneck**: wrong_peak={p_wp} vs no_peak={p_fn_np} "
            f"→ {'timing/localization' if p_wp >= p_fn_np else 'missed detection'} dominant"
        )
    lines.append(
        f"- **S FP mix**: absent_FP={s_fp_abs} vs wrong_peak={s_fp_wp} "
        f"→ {'false alarms on no-S' if s_fp_abs >= s_fp_wp else 'wrong timing on with-S'} dominant"
    )
    lines.append(
        f"- **S FN mix**: no_peak={s_fn_np} vs exist_gate={s_fn_eg} "
        f"→ {'missed S peak' if s_fn_np >= s_fn_eg else 'exist over-gating'} dominant"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    obs_mod = _load_obs_compare_module()
    model, cfg, _ = build_model_from_ckpt(Path(args.checkpoint), device)
    holdout, _, _ = load_split_samples(args.split_json, "holdout")
    holdout = filter_alive_channels(holdout, int(cfg["input_dim"]))
    print(
        f"[fnfp] n={len(holdout)} exist_th={args.exist_th} ckpt={args.checkpoint}",
        flush=True,
    )
    report = analyze(model, holdout, device, cfg, obs_mod, args)
    report["checkpoint"] = args.checkpoint
    (out_dir / "fnfp_report.json").write_text(json.dumps(report, indent=2))
    md = to_md(report, args.checkpoint)
    (out_dir / "fnfp_report.md").write_text(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
