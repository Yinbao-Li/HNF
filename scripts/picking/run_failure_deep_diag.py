#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep failure analysis: FN taxonomy + recoverable wrong-peak ranking."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analyze_stead_picking import _noise_ratio_from_outputs, _subsample_indices, load_model
from hnf.picking_metrics import apply_p_before_s_constraint, idx_to_sec, tolerance_bins
from hnf.stead_picking_dataset import STEADPickingDataset
from run_adaptive_pick_eval import _local_max_indices
from train_stead_picking import move_batch_to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/run23/23_ps_gap_head/best.pt")
    p.add_argument("--output", default="outputs/run23/23_ps_gap_head/failure_deep_2k.json")
    p.add_argument("--max-events", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--subset-seed", type=int, default=11)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(Path(args.checkpoint), device)
    ds = STEADPickingDataset("test", seq_len=args.seq_len, load_geometry=True)
    indices = _subsample_indices(ds, args.max_events, 0, args.subset_seed)
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False)
    tol = tolerance_bins(args.seq_len, 0.5)
    th = args.pick_threshold

    stats: dict[str, Counter] = defaultdict(Counter)
    margins: list[dict] = []
    nr_by_fn: dict[str, list] = defaultdict(list)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        det = torch.sigmoid(out["det"])
        if det.dim() > 1:
            det = det.amax(dim=-1)
        p_probs, s_probs = apply_p_before_s_constraint(
            torch.sigmoid(out["p"]), torch.sigmoid(out["s"]), th
        )
        noise = _noise_ratio_from_outputs(out)
        for i in range(batch["x"].size(0)):
            if batch["det"][i] <= 0.5:
                continue
            det_ok = bool(det[i] >= args.det_threshold)
            nri = float(noise[i].item()) if noise is not None else float("nan")
            dist = float(batch["source_distance_km"][i].item())
            dist = dist if np.isfinite(dist) else None
            for head, probs, valid_k, idx_k in [
                ("p", p_probs[i], "p_valid", "p_idx"),
                ("s", s_probs[i], "s_valid", "s_idx"),
            ]:
                if batch[valid_k][i] <= 0:
                    continue
                gt = int(batch[idx_k][i].item())
                peak = float(probs.max().item())
                pred = int(probs.argmax().item())
                if not det_ok:
                    stats[head]["fn_missed_det"] += 1
                    nr_by_fn[f"{head}_missed_det"].append(nri)
                    continue
                if peak < th:
                    stats[head]["fn_no_peak"] += 1
                    nr_by_fn[f"{head}_no_peak"].append(nri)
                    locs = _local_max_indices(probs)
                    near = [j for j in locs if abs(j - gt) <= tol]
                    if near:
                        stats[head]["no_peak_but_local_near_gt"] += 1
                        jbest = max(near, key=lambda j: float(probs[j]))
                        if float(probs[jbest]) >= th * 0.7:
                            stats[head]["no_peak_near_gt_ge_0p7th"] += 1
                    continue
                if abs(pred - gt) <= tol:
                    stats[head]["tp"] += 1
                    continue

                stats[head]["fn_wrong_peak"] += 1
                nr_by_fn[f"{head}_wrong"].append(nri)
                locs = _local_max_indices(probs)
                win = probs[max(0, gt - tol) : gt + tol + 1]
                gt_best = float(win.max().item()) if win.numel() else 0.0
                loc_vals = sorted((float(probs[j]) for j in locs), reverse=True)
                rank = 1 + sum(1 for v in loc_vals if v > gt_best + 1e-12)
                gt_local = any(abs(j - gt) <= tol for j in locs)
                if gt_local:
                    stats[head]["wrong_gt_is_local"] += 1
                if rank <= 3:
                    stats[head]["wrong_gt_local_rank_le3"] += 1
                if rank <= 5:
                    stats[head]["wrong_gt_local_rank_le5"] += 1
                if gt_local and gt_best >= th:
                    stats[head]["wrong_recoverable_if_pick_gt_local"] += 1
                # close race: top barely beats GT
                if gt_best >= th and (peak - gt_best) < 0.08:
                    stats[head]["wrong_close_race_margin_lt_0p08"] += 1
                margins.append(
                    {
                        "head": head,
                        "margin": peak - gt_best,
                        "gt_prob": gt_best,
                        "top_prob": peak,
                        "rank": rank,
                        "nr": nri,
                        "dist": dist,
                    }
                )
                if head == "s" and dist is not None and batch["p_valid"][i] > 0:
                    p_peak = float(p_probs[i].max().item())
                    p_pred = int(p_probs[i].argmax().item())
                    if p_peak >= th:
                        gt_gap = idx_to_sec(gt, args.seq_len) - idx_to_sec(
                            int(batch["p_idx"][i].item()), args.seq_len
                        )
                        pred_gap = idx_to_sec(pred, args.seq_len) - idx_to_sec(p_pred, args.seq_len)
                        mu = 0.119 * dist
                        if abs(gt_gap - mu) < abs(pred_gap - mu):
                            stats[head]["wrong_dist_prefers_gt"] += 1

    report: dict = {"checkpoint": args.checkpoint, "n_events": args.max_events}
    for head in ("p", "s"):
        st = stats[head]
        fn = st["fn_wrong_peak"] + st["fn_no_peak"] + st["fn_missed_det"]
        tp = st["tp"]
        entry = {
            "tp": int(tp),
            "fn_total": int(fn),
            "recall": float(tp / max(tp + fn, 1)),
            "fn_wrong_peak": int(st["fn_wrong_peak"]),
            "fn_no_peak": int(st["fn_no_peak"]),
            "fn_missed_det": int(st["fn_missed_det"]),
            "wrong_frac_of_fn": float(st["fn_wrong_peak"] / max(fn, 1)),
            "wrong_gt_is_local": int(st["wrong_gt_is_local"]),
            "wrong_gt_local_rank_le3": int(st["wrong_gt_local_rank_le3"]),
            "wrong_gt_local_rank_le5": int(st["wrong_gt_local_rank_le5"]),
            "wrong_recoverable_if_pick_gt_local": int(st["wrong_recoverable_if_pick_gt_local"]),
            "wrong_close_race_margin_lt_0p08": int(st["wrong_close_race_margin_lt_0p08"]),
            "wrong_dist_prefers_gt": int(st.get("wrong_dist_prefers_gt", 0)),
            "no_peak_but_local_near_gt": int(st["no_peak_but_local_near_gt"]),
            "no_peak_near_gt_ge_0p7th": int(st["no_peak_near_gt_ge_0p7th"]),
        }
        m = [x for x in margins if x["head"] == head]
        if m:
            entry["margin_median"] = float(np.median([x["margin"] for x in m]))
            entry["gt_prob_median"] = float(np.median([x["gt_prob"] for x in m]))
            entry["top_prob_median"] = float(np.median([x["top_prob"] for x in m]))
            entry["frac_margin_lt_0.05"] = float(np.mean([x["margin"] < 0.05 for x in m]))
            entry["frac_margin_lt_0.10"] = float(np.mean([x["margin"] < 0.10 for x in m]))
            entry["frac_gt_prob_ge_th"] = float(np.mean([x["gt_prob"] >= th for x in m]))
            # oracle recall if all recoverable wrong peaks fixed
            rec = entry["wrong_recoverable_if_pick_gt_local"]
            entry["oracle_recall_if_fix_recoverable_wrong"] = float((tp + rec) / max(tp + fn, 1))
        report[head] = entry

    report["noise_ratio"] = {}
    for k, vals in nr_by_fn.items():
        vals = [v for v in vals if np.isfinite(v)]
        report["noise_ratio"][k] = {
            "n": len(vals),
            "median": float(np.median(vals)) if vals else None,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[failure-deep] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
