#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc conditional S exist loosen on frozen L1200 (no retrain).

Board keeps P = soft_floor + score_minus_late.
S: soft_floor with optional loosen when peak is strong but exist is borderline
(targets FN_peak_ok_but_gated without fully opening absent windows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.pick_decode import decode_pick_index
from hnf.picking_metrics import apply_p_before_s_constraint, tolerance_bins
from tools.obs_matched_split import load_split_samples
from tools.sweep_obs_gate_rules import score_phase
from tools.train_obs_exist_gate import build_model_from_ckpt
from tools.train_obs_picking import filter_alive_channels, _load_obs_compare_module
from tools.train_stead_picking import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional S-gate loosen sweep")
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/conditional_s_gate_sweep",
    )
    p.add_argument("--pick-th", type=float, default=0.25)
    p.add_argument("--exist-th", type=float, default=0.60)
    p.add_argument("--soft-th", type=float, default=0.25)
    p.add_argument("--decode-late-penalty", type=float, default=0.60)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def cache_board_forward(model, samples, device, cfg, obs_mod, batch_size: int, late_pen: float) -> dict:
    model.eval()
    seq_len = int(cfg["seq_len"])
    window_sec = float(cfg["window_sec"])
    dim = int(cfg["input_dim"])
    p_peaks, s_peaks, p_idx, s_idx = [], [], [], []
    p_exist, s_exist = [], []
    p_gt, s_gt, p_valid, s_valid = [], [], [], []
    s_second = []  # 2nd local-peak height (0 if none)

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        x, t, p_i, s_i, pv, sv = obs_mod.to_hnf_batch(
            chunk, seq_len, window_sec, device, n_channels=dim
        )
        out = model(x, t)
        p_logits = torch.nan_to_num(out.get("p_logits", out["p"]), nan=-50.0)
        s_logits = torch.nan_to_num(out.get("s_logits", out["s"]), nan=-50.0)
        pp = torch.sigmoid(p_logits)
        sp = torch.sigmoid(s_logits)
        pp, sp = apply_p_before_s_constraint(pp, sp, 0.25)
        pe = torch.sigmoid(torch.nan_to_num(out["p_exist"], nan=-50.0))
        se = torch.sigmoid(torch.nan_to_num(out["s_exist"], nan=-50.0))

        b = pp.size(0)
        p_pk = torch.empty(b)
        p_ix = torch.empty(b, dtype=torch.long)
        s_pk = torch.empty(b)
        s_ix = torch.empty(b, dtype=torch.long)
        s2 = torch.zeros(b)
        for i in range(b):
            pi, pk = decode_pick_index(
                pp[i], pick_th=0.25, mode="score_minus_late", late_penalty=late_pen
            )
            si, sk = decode_pick_index(sp[i], pick_th=0.25, mode="argmax")
            p_pk[i], p_ix[i] = pk, pi
            s_pk[i], s_ix[i] = sk, si
            # second local peak (crude sharpness / multi-peak cue)
            probs = sp[i].detach().float().cpu()
            left = torch.roll(probs, 1, 0)
            right = torch.roll(probs, -1, 0)
            left[0] = -1.0
            right[-1] = -1.0
            peaks = (probs >= left) & (probs >= right)
            vals = probs[peaks]
            if vals.numel() >= 2:
                top2 = torch.topk(vals, k=2).values
                s2[i] = float(top2[1])

        p_peaks.append(p_pk)
        s_peaks.append(s_pk)
        p_idx.append(p_ix)
        s_idx.append(s_ix)
        p_exist.append(pe.cpu())
        s_exist.append(se.cpu())
        p_gt.append(p_i.cpu())
        s_gt.append(s_i.cpu())
        p_valid.append(pv.cpu())
        s_valid.append(sv.cpu())
        s_second.append(s2)

    return {
        "p_peak": torch.cat(p_peaks),
        "s_peak": torch.cat(s_peaks),
        "p_pred": torch.cat(p_idx),
        "s_pred": torch.cat(s_idx),
        "p_exist": torch.cat(p_exist),
        "s_exist": torch.cat(s_exist),
        "p_gt": torch.cat(p_gt),
        "s_gt": torch.cat(s_gt),
        "p_valid": torch.cat(p_valid),
        "s_valid": torch.cat(s_valid),
        "s_second": torch.cat(s_second),
        "seq_len": seq_len,
    }


def score_s_conditional(
    peak,
    pred_idx,
    exist,
    second,
    valid,
    gt_idx,
    *,
    soft_th: float,
    exist_th: float,
    exist_loose: float,
    peak_hi: float,
    peak_ratio_hi: float,
    tol: int,
) -> dict:
    """soft_floor, plus loosen when strong peak + borderline exist."""
    tp = fp = fn = 0
    n_loose = 0
    for i in range(peak.numel()):
        pk = float(peak[i])
        ex = float(exist[i])
        sec = float(second[i])
        soft_ok = (pk * ex) >= soft_th
        base = soft_ok and (ex >= exist_th)
        ratio = pk / max(sec, 1e-6) if sec > 0 else 99.0
        loose = (
            (not base)
            and soft_ok
            and (ex >= exist_loose)
            and (ex < exist_th)
            and (pk >= peak_hi)
            and (ratio >= peak_ratio_hi)
        )
        if loose:
            n_loose += 1
        pred_exists = base or loose
        has_gt = bool(valid[i].item() > 0.5)
        if not has_gt:
            if pred_exists:
                fp += 1
            continue
        within = abs(int(pred_idx[i]) - int(gt_idx[i])) <= tol
        if pred_exists and within:
            tp += 1
        elif pred_exists:
            fp += 1
            fn += 1
        else:
            fn += 1
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
        "n_loose_triggers": n_loose,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    obs = _load_obs_compare_module()
    model, cfg, _ = build_model_from_ckpt(Path(args.checkpoint), device)
    holdout, _, _ = load_split_samples(args.split_json, "holdout")
    holdout = filter_alive_channels(holdout, int(cfg["input_dim"]), mode="strict")
    print(f"[cond-s] cache n={len(holdout)}", flush=True)
    cache = cache_board_forward(
        model, holdout, device, cfg, obs, args.batch_size, args.decode_late_penalty
    )
    tol = tolerance_bins(cache["seq_len"], args.tol_sec)

    # P fixed at board recipe
    p_m = score_phase(
        cache["p_peak"],
        cache["p_pred"],
        cache["p_exist"],
        cache["p_valid"],
        cache["p_gt"],
        mode="soft_floor",
        pick_th=args.pick_th,
        exist_th=args.exist_th,
        soft_th=args.soft_th,
        tol=tol,
        score_absent=False,
    )
    s_base = score_phase(
        cache["s_peak"],
        cache["s_pred"],
        cache["s_exist"],
        cache["s_valid"],
        cache["s_gt"],
        mode="soft_floor",
        pick_th=args.pick_th,
        exist_th=args.exist_th,
        soft_th=args.soft_th,
        tol=tol,
        score_absent=True,
    )

    exist_looses = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    peak_his = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    ratios = [1.0, 1.2, 1.5, 2.0, 3.0]

    rows = []
    for el in exist_looses:
        for ph in peak_his:
            for pr in ratios:
                s_m = score_s_conditional(
                    cache["s_peak"],
                    cache["s_pred"],
                    cache["s_exist"],
                    cache["s_second"],
                    cache["s_valid"],
                    cache["s_gt"],
                    soft_th=args.soft_th,
                    exist_th=args.exist_th,
                    exist_loose=el,
                    peak_hi=ph,
                    peak_ratio_hi=pr,
                    tol=tol,
                )
                rows.append(
                    {
                        "exist_loose": el,
                        "peak_hi": ph,
                        "peak_ratio_hi": pr,
                        "p_f1": p_m["f1"],
                        "s_f1": s_m["f1"],
                        "s_precision": s_m["precision"],
                        "s_recall": s_m["recall"],
                        "s_tp": s_m["tp"],
                        "s_fp": s_m["fp"],
                        "s_fn": s_m["fn"],
                        "n_loose": s_m["n_loose_triggers"],
                        "delta_s": s_m["f1"] - s_base["f1"],
                    }
                )

    best = max(rows, key=lambda r: r["s_f1"])
    # prefer configs that beat baseline without collapsing P (P is fixed here)
    beat = [r for r in rows if r["s_f1"] > s_base["f1"] + 1e-6]
    best_safe = max(beat, key=lambda r: r["s_f1"]) if beat else best

    report = {
        "checkpoint": args.checkpoint,
        "n": len(holdout),
        "board": {
            "gate_mode": "soft_floor",
            "pick_th": args.pick_th,
            "exist_th": args.exist_th,
            "soft_th": args.soft_th,
            "p_decode": f"score_minus_late({args.decode_late_penalty})",
            "p_f1": p_m["f1"],
            "s_f1_base": s_base["f1"],
            "s_base": s_base,
        },
        "best_s": best,
        "best_s_above_base": best_safe if beat else None,
        "n_configs": len(rows),
        "n_beat_base": len(beat),
        "top15": sorted(rows, key=lambda r: -r["s_f1"])[:15],
    }
    (out / "conditional_s_gate_sweep.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Conditional S-gate loosen sweep",
        "",
        f"- ckpt: `{args.checkpoint}` n={len(holdout)}",
        f"- board P (soft_floor+late): **{p_m['f1']:.3f}**",
        f"- board S soft_floor base: **{s_base['f1']:.3f}**",
        f"- configs beating base: {len(beat)}/{len(rows)}",
        "",
        f"- **best S**: {best['s_f1']:.3f} (Δ={best['delta_s']:+.3f}) "
        f"exist_loose={best['exist_loose']} peak_hi={best['peak_hi']} "
        f"ratio≥{best['peak_ratio_hi']} n_loose={best['n_loose']}",
        "",
        "## Top 10 by S-F1",
        "",
        "| exist_loose | peak_hi | ratio | S | ΔS | P | n_loose |",
        "|------------:|--------:|------:|--:|---:|--:|--------:|",
    ]
    for r in sorted(rows, key=lambda x: -x["s_f1"])[:10]:
        md.append(
            f"| {r['exist_loose']:.2f} | {r['peak_hi']:.2f} | {r['peak_ratio_hi']:.1f} | "
            f"{r['s_f1']:.3f} | {r['delta_s']:+.3f} | {r['p_f1']:.3f} | {r['n_loose']} |"
        )
    md.append("")
    (out / "conditional_s_gate_sweep.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
