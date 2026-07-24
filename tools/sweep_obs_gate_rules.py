#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline holdout sweep: hard exist gate vs soft (peak×exist) vs dual thresholds.

Caches one forward pass, then sweeps decision rules — no retraining.
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

from hnf.picking_metrics import apply_p_before_s_constraint, tolerance_bins
from tools.obs_matched_split import load_split_samples
from tools.train_obs_exist_gate import build_model_from_ckpt
from tools.train_obs_picking import filter_alive_channels, _load_obs_compare_module
from tools.train_stead_picking import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS soft/dual gate sweep")
    p.add_argument(
        "--checkpoint",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/best.pt",
    )
    p.add_argument("--split-json", default="outputs/obs_full_native_split/split.json")
    p.add_argument(
        "--output-dir",
        default="outputs/run_obs_native/obs_4c_exist_L1200_12ep/gate_rule_sweep",
    )
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def cache_forward(model, samples, device, cfg, obs_mod, batch_size: int) -> dict:
    model.eval()
    seq_len = int(cfg["seq_len"])
    window_sec = float(cfg["window_sec"])
    dim = int(cfg["input_dim"])
    p_peaks, s_peaks, p_idx_pred, s_idx_pred = [], [], [], []
    p_exist, s_exist = [], []
    p_gt, s_gt, p_valid, s_valid = [], [], [], []

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
        p_pk, p_ix = pp.max(dim=-1)
        s_pk, s_ix = sp.max(dim=-1)
        p_peaks.append(p_pk.cpu())
        s_peaks.append(s_pk.cpu())
        p_idx_pred.append(p_ix.cpu())
        s_idx_pred.append(s_ix.cpu())
        p_exist.append(pe.cpu())
        s_exist.append(se.cpu())
        p_gt.append(p_i.cpu())
        s_gt.append(s_i.cpu())
        p_valid.append(pv.cpu())
        s_valid.append(sv.cpu())

    return {
        "p_peak": torch.cat(p_peaks),
        "s_peak": torch.cat(s_peaks),
        "p_pred": torch.cat(p_idx_pred),
        "s_pred": torch.cat(s_idx_pred),
        "p_exist": torch.cat(p_exist),
        "s_exist": torch.cat(s_exist),
        "p_gt": torch.cat(p_gt),
        "s_gt": torch.cat(s_gt),
        "p_valid": torch.cat(p_valid),
        "s_valid": torch.cat(s_valid),
        "seq_len": seq_len,
    }


def score_phase(
    peak,
    pred_idx,
    exist,
    valid,
    gt_idx,
    *,
    mode: str,
    pick_th: float,
    exist_th: float,
    soft_th: float,
    tol: int,
    score_absent: bool,
) -> dict:
    tp = fp = fn = 0
    for i in range(peak.numel()):
        pk = float(peak[i])
        ex = float(exist[i])
        if mode == "hard":
            pred_exists = (pk >= pick_th) and (ex >= exist_th)
        elif mode == "soft":
            pred_exists = (pk * ex) >= soft_th
        elif mode == "dual":
            # dual: independent thresholds (same as hard but named for sweeps)
            pred_exists = (pk >= pick_th) and (ex >= exist_th)
        elif mode == "soft_floor":
            # soft score but require exist above a floor
            pred_exists = (pk * ex) >= soft_th and (ex >= exist_th)
        else:
            raise ValueError(mode)
        has_gt = bool(valid[i].item() > 0.5)
        if not has_gt:
            if score_absent and pred_exists:
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
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


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
    print(f"[gate-sweep] cache forward n={len(holdout)}", flush=True)
    cache = cache_forward(model, holdout, device, cfg, obs_mod, args.batch_size)
    tol = tolerance_bins(cache["seq_len"], args.tol_sec)

    pick_ths = [round(x, 2) for x in np.arange(0.15, 0.46, 0.05)]
    exist_ths = [round(x, 2) for x in np.arange(0.25, 0.71, 0.05)]
    soft_ths = [round(x, 2) for x in np.arange(0.08, 0.36, 0.02)]

    rows: list[dict] = []

    # hard / dual (same rule, grid pick×exist)
    for pth in pick_ths:
        for eth in exist_ths:
            p = score_phase(
                cache["p_peak"], cache["p_pred"], cache["p_exist"],
                cache["p_valid"], cache["p_gt"],
                mode="hard", pick_th=pth, exist_th=eth, soft_th=0.0, tol=tol, score_absent=False,
            )
            s = score_phase(
                cache["s_peak"], cache["s_pred"], cache["s_exist"],
                cache["s_valid"], cache["s_gt"],
                mode="hard", pick_th=pth, exist_th=eth, soft_th=0.0, tol=tol, score_absent=True,
            )
            score = 0.55 * p["f1"] + 0.45 * s["f1"]
            rows.append(
                {
                    "mode": "hard",
                    "pick_th": pth,
                    "exist_th": eth,
                    "soft_th": None,
                    "p_f1": p["f1"],
                    "s_f1": s["f1"],
                    "p_prec": p["precision"],
                    "p_rec": p["recall"],
                    "s_prec": s["precision"],
                    "s_rec": s["recall"],
                    "score": score,
                }
            )

    # soft: peak×exist ≥ soft_th (P uses same soft; P-exist usually ~1)
    for sth in soft_ths:
        for pth_ref in [0.25]:  # unused for soft except logging
            p = score_phase(
                cache["p_peak"], cache["p_pred"], cache["p_exist"],
                cache["p_valid"], cache["p_gt"],
                mode="soft", pick_th=pth_ref, exist_th=0.0, soft_th=sth, tol=tol, score_absent=False,
            )
            s = score_phase(
                cache["s_peak"], cache["s_pred"], cache["s_exist"],
                cache["s_valid"], cache["s_gt"],
                mode="soft", pick_th=pth_ref, exist_th=0.0, soft_th=sth, tol=tol, score_absent=True,
            )
            score = 0.55 * p["f1"] + 0.45 * s["f1"]
            rows.append(
                {
                    "mode": "soft",
                    "pick_th": None,
                    "exist_th": None,
                    "soft_th": sth,
                    "p_f1": p["f1"],
                    "s_f1": s["f1"],
                    "p_prec": p["precision"],
                    "p_rec": p["recall"],
                    "s_prec": s["precision"],
                    "s_rec": s["recall"],
                    "score": score,
                }
            )

    # soft_floor: soft_th + exist floor
    for sth in soft_ths:
        for eth in [0.20, 0.30, 0.40, 0.50]:
            p = score_phase(
                cache["p_peak"], cache["p_pred"], cache["p_exist"],
                cache["p_valid"], cache["p_gt"],
                mode="soft_floor", pick_th=0.0, exist_th=eth, soft_th=sth, tol=tol, score_absent=False,
            )
            s = score_phase(
                cache["s_peak"], cache["s_pred"], cache["s_exist"],
                cache["s_valid"], cache["s_gt"],
                mode="soft_floor", pick_th=0.0, exist_th=eth, soft_th=sth, tol=tol, score_absent=True,
            )
            score = 0.55 * p["f1"] + 0.45 * s["f1"]
            rows.append(
                {
                    "mode": "soft_floor",
                    "pick_th": None,
                    "exist_th": eth,
                    "soft_th": sth,
                    "p_f1": p["f1"],
                    "s_f1": s["f1"],
                    "p_prec": p["precision"],
                    "p_rec": p["recall"],
                    "s_prec": s["precision"],
                    "s_rec": s["recall"],
                    "score": score,
                }
            )

    # baselines
    base_hard = next(
        r for r in rows if r["mode"] == "hard" and r["pick_th"] == 0.25 and r["exist_th"] == 0.60
    )
    best_overall = max(rows, key=lambda r: r["score"])
    best_soft = max((r for r in rows if r["mode"] == "soft"), key=lambda r: r["score"])
    best_hard = max((r for r in rows if r["mode"] == "hard"), key=lambda r: r["score"])
    best_sf = max((r for r in rows if r["mode"] == "soft_floor"), key=lambda r: r["score"])

    report = {
        "checkpoint": args.checkpoint,
        "n_holdout": len(holdout),
        "baseline_hard_025_060": base_hard,
        "best_overall": best_overall,
        "best_hard": best_hard,
        "best_soft": best_soft,
        "best_soft_floor": best_sf,
        "top20": sorted(rows, key=lambda r: -r["score"])[:20],
        "n_configs": len(rows),
    }
    (out_dir / "gate_rule_sweep.json").write_text(json.dumps(report, indent=2))

    md = [
        "# OBS gate-rule sweep (L1200)",
        "",
        f"- ckpt: `{args.checkpoint}`",
        f"- n={len(holdout)} configs={len(rows)}",
        "",
        "## Baseline",
        f"- hard pick=0.25 exist=0.60 → P={base_hard['p_f1']:.3f} S={base_hard['s_f1']:.3f}",
        "",
        "## Best by mode",
        f"- **overall** `{best_overall['mode']}` → P={best_overall['p_f1']:.3f} S={best_overall['s_f1']:.3f} "
        f"(pick={best_overall['pick_th']} exist={best_overall['exist_th']} soft={best_overall['soft_th']})",
        f"- hard → P={best_hard['p_f1']:.3f} S={best_hard['s_f1']:.3f} "
        f"(pick={best_hard['pick_th']} exist={best_hard['exist_th']})",
        f"- soft → P={best_soft['p_f1']:.3f} S={best_soft['s_f1']:.3f} (soft_th={best_soft['soft_th']})",
        f"- soft_floor → P={best_sf['p_f1']:.3f} S={best_sf['s_f1']:.3f} "
        f"(soft={best_sf['soft_th']} exist_floor={best_sf['exist_th']})",
        "",
        "## Top 10",
        "",
        "| mode | pick | exist | soft | P | S | score |",
        "|------|-----:|------:|-----:|--:|--:|------:|",
    ]
    for r in report["top20"][:10]:
        md.append(
            f"| {r['mode']} | {r['pick_th']} | {r['exist_th']} | {r['soft_th']} | "
            f"{r['p_f1']:.3f} | {r['s_f1']:.3f} | {r['score']:.3f} |"
        )
    md.append("")
    (out_dir / "gate_rule_sweep.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
