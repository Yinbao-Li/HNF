#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference-time P/S joint pairing using S-P interval priors."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tools.analyze_stead_picking import (
    _noise_ratio_from_outputs,
    _subsample_indices,
    load_model,
    pick_prf,
)
from hnf.picking_metrics import apply_p_before_s_constraint, idx_to_sec, tolerance_bins
from hnf.stead_picking_dataset import STEADPickingDataset
from run_adaptive_pick_eval import _local_max_indices
from tools.train_stead_picking import move_batch_to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P/S joint pairing eval")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output", default="outputs/run20/20_wrongpeak_sharp/ps_pair_eval.json")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--max-events", type=int, default=2000)
    p.add_argument("--max-noise", type=int, default=500)
    p.add_argument("--subset-seed", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--min-cand-frac", type=float, default=0.45)
    p.add_argument("--min-gap-sec", type=float, default=0.4)
    p.add_argument("--max-gap-sec", type=float, default=25.0)
    # Empirical STEAD prior from earlier mining (approx median/std on subset)
    p.add_argument("--prior-mu-sec", type=float, default=5.2)
    p.add_argument("--prior-sigma-sec", type=float, default=4.5)
    p.add_argument("--gap-weight", type=float, default=0.35)
    p.add_argument("--homog-s_per_km", type=float, default=0.119)
    p.add_argument("--homog-sigma-sec", type=float, default=1.2)
    # Only apply predicted Δt prior when Huygens noise_ratio is high (harder traces).
    # Default 0.239 ≈ upper tertile on the run23 2k STEAD subset.
    p.add_argument(
        "--pred-gap-noise-ratio-min",
        type=float,
        default=0.239,
        help="Gate fix_weak_pred_gap_nr: enable Δt prior only if noise_ratio >= this",
    )
    return p.parse_args()


def _empty() -> dict:
    return {
        "counts": {"p": Counter(), "s": Counter()},
        "breakdown": Counter(),
        "err_sum": {"p": 0.0, "s": 0.0},
        "tp_n": {"p": 0, "s": 0},
        "mode_cases": Counter(),
        "rescued": Counter(),
    }


def _score_one(
    st: dict,
    head: str,
    peak: float,
    pred: int,
    gt: Optional[int],
    threshold: float,
    det_ok: bool,
    tol: int,
    seq_len: int,
    is_event: bool,
) -> None:
    if not is_event:
        if peak >= threshold:
            st["counts"][head]["fp"] += 1
        return
    if gt is None:
        return
    has_peak = peak >= threshold
    if not det_ok:
        st["counts"][head]["fn"] += 1
        st["breakdown"][f"{head}_fn_missed_det"] += 1
        return
    if has_peak and abs(pred - gt) <= tol:
        st["counts"][head]["tp"] += 1
        st["err_sum"][head] += abs(idx_to_sec(pred, seq_len) - idx_to_sec(gt, seq_len))
        st["tp_n"][head] += 1
        return
    st["counts"][head]["fn"] += 1
    if has_peak:
        st["breakdown"][f"{head}_fn_wrong_peak"] += 1
    else:
        st["breakdown"][f"{head}_fn_no_peak"] += 1


def _candidate_peaks(
    probs: torch.Tensor,
    threshold: float,
    topk: int,
    min_frac: float,
) -> list[tuple[int, float]]:
    peak = float(probs.max().item())
    floor = max(threshold * min_frac, threshold - 0.18)
    cands: list[tuple[int, float]] = []
    for idx in _local_max_indices(probs):
        val = float(probs[idx].item())
        if val >= floor:
            cands.append((idx, val))
    if not cands and peak >= threshold:
        cands = [(int(probs.argmax().item()), peak)]
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[: max(1, topk)]


def _log_gauss(x: float, mu: float, sigma: float) -> float:
    s = max(sigma, 1e-3)
    return -0.5 * ((x - mu) / s) ** 2 - math.log(s * math.sqrt(2.0 * math.pi))


def _prior_params(
    *,
    prior_mu: float,
    prior_sigma: float,
    distance_km: Optional[float],
    homog_s_per_km: float,
    homog_sigma: float,
    use_distance: bool,
    pred_gap_sec: Optional[float] = None,
    pred_gap_sigma: Optional[float] = None,
    use_pred_gap: bool = False,
) -> tuple[float, float]:
    if use_pred_gap and pred_gap_sec is not None and np.isfinite(pred_gap_sec):
        sigma = pred_gap_sigma if pred_gap_sigma is not None else max(0.8, 0.25 * float(pred_gap_sec))
        return float(pred_gap_sec), float(max(sigma, 0.5))
    if use_distance and distance_km is not None and np.isfinite(distance_km) and distance_km > 0:
        return float(distance_km) * homog_s_per_km, homog_sigma
    return prior_mu, prior_sigma


def _gap_plausible(gap: float, mu: float, sigma: float, zmax: float = 2.5) -> bool:
    return abs(gap - mu) <= zmax * max(sigma, 1e-3)


def _pair_picks(
    p_probs: torch.Tensor,
    s_probs: torch.Tensor,
    *,
    threshold: float,
    seq_len: int,
    topk: int,
    min_frac: float,
    min_gap_sec: float,
    max_gap_sec: float,
    gap_weight: float,
    prior_mu: float,
    prior_sigma: float,
    distance_km: Optional[float] = None,
    homog_s_per_km: float = 0.119,
    homog_sigma: float = 1.2,
    use_distance: bool = False,
    pred_gap_sec: Optional[float] = None,
    pred_gap_sigma: Optional[float] = None,
    use_pred_gap: bool = False,
    mode: str = "full",
    anchor_margin: float = 0.12,
    zmax: float = 2.5,
) -> tuple[float, int, float, int]:
    """Return (p_peak, p_idx, s_peak, s_idx) after joint pairing.

    mode:
      - full: always re-pair over candidate grid (aggressive)
      - repair: keep independent picks unless gap is implausible / one side weak
      - anchor: freeze the stronger phase, only re-search the weaker one
      - fix_weak: only when gap is clearly off; freeze strong phase and repair weak
    """
    p_peak0 = float(p_probs.max().item())
    s_peak0 = float(s_probs.max().item())
    p_idx0 = int(p_probs.argmax().item())
    s_idx0 = int(s_probs.argmax().item())

    # Soft order fix identical to baseline post-process.
    if p_peak0 >= threshold and s_peak0 >= threshold and s_idx0 <= p_idx0:
        s_peak0 = 0.0

    mu, sigma = _prior_params(
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        distance_km=distance_km,
        homog_s_per_km=homog_s_per_km,
        homog_sigma=homog_sigma,
        use_distance=use_distance,
        pred_gap_sec=pred_gap_sec,
        pred_gap_sigma=pred_gap_sigma,
        use_pred_gap=use_pred_gap,
    )

    gap0 = idx_to_sec(s_idx0, seq_len) - idx_to_sec(p_idx0, seq_len)
    both_ok = p_peak0 >= threshold and s_peak0 >= threshold
    if mode == "repair" and both_ok and _gap_plausible(gap0, mu, sigma, zmax=zmax):
        return p_peak0, p_idx0, s_peak0, s_idx0

    p_cands = _candidate_peaks(p_probs, threshold, topk, min_frac)
    s_cands = _candidate_peaks(s_probs, threshold, topk, min_frac)
    if not p_cands:
        p_cands = [(p_idx0, p_peak0)] if p_peak0 >= threshold * min_frac else []
    if not s_cands:
        s_cands = [(s_idx0, s_peak0)] if s_peak0 >= threshold * min_frac else []
    if not p_cands and not s_cands:
        return p_peak0, p_idx0, s_peak0, s_idx0
    if not p_cands:
        return 0.0, p_idx0, s_cands[0][1], s_cands[0][0]
    if not s_cands:
        return p_cands[0][1], p_cands[0][0], 0.0, s_idx0

    # Anchor the stronger phase when requested / in repair with one strong side.
    freeze_p = False
    freeze_s = False
    if mode in {"anchor", "repair", "fix_weak"}:
        if p_peak0 >= threshold and (p_peak0 - s_peak0) >= anchor_margin:
            freeze_p = True
            p_cands = [(p_idx0, p_peak0)]
        elif s_peak0 >= threshold and (s_peak0 - p_peak0) >= anchor_margin:
            freeze_s = True
            s_cands = [(s_idx0, s_peak0)]
        elif mode == "fix_weak" and p_peak0 >= threshold and s_peak0 >= threshold:
            # Equal-ish confidence: still freeze the slightly stronger side.
            if p_peak0 >= s_peak0:
                freeze_p = True
                p_cands = [(p_idx0, p_peak0)]
            else:
                freeze_s = True
                s_cands = [(s_idx0, s_peak0)]

    # fix_weak: only intervene when current gap is clearly off.
    if mode == "fix_weak":
        if both_ok and _gap_plausible(gap0, mu, sigma, zmax=1.5):
            return p_peak0, p_idx0, s_peak0, s_idx0
        if not (freeze_p or freeze_s):
            return p_peak0, p_idx0, s_peak0, s_idx0

    best_score = float("-inf")
    best = (p_peak0, p_idx0, s_peak0, s_idx0)
    base_gap_ll = _log_gauss(gap0, mu, sigma) if both_ok else -1e9
    for p_idx, p_val in p_cands:
        for s_idx, s_val in s_cands:
            gap = idx_to_sec(s_idx, seq_len) - idx_to_sec(p_idx, seq_len)
            if gap < min_gap_sec or gap > max_gap_sec:
                continue
            gap_ll = _log_gauss(gap, mu, sigma)
            score = p_val + s_val + gap_weight * gap_ll
            if p_val >= threshold:
                score += 0.03
            if s_val >= threshold:
                score += 0.03
            # Conservative: require clear improvement over independent picks.
            if mode in {"repair", "anchor"} and both_ok:
                indep = p_peak0 + s_peak0 + gap_weight * base_gap_ll
                if score < indep + 0.08:
                    continue
            if mode == "fix_weak" and both_ok:
                # Accept only if gap likelihood improves enough.
                if gap_ll < base_gap_ll + 0.75:
                    continue
                # And do not collapse the weak-phase probability too hard.
                if freeze_p and s_val < max(threshold * 0.7, s_peak0 - 0.35):
                    continue
                if freeze_s and p_val < max(threshold * 0.7, p_peak0 - 0.35):
                    continue
            if score > best_score:
                best_score = score
                best = (p_val, p_idx, s_val, s_idx)

    if best_score == float("-inf"):
        return p_peak0, p_idx0, s_peak0, s_idx0
    # Never demote a frozen high-confidence peak below threshold accidentally.
    if freeze_p and best[0] < threshold <= p_peak0:
        return p_peak0, p_idx0, best[2], best[3]
    if freeze_s and best[2] < threshold <= s_peak0:
        return best[0], best[1], s_peak0, s_idx0
    return best


def _label(peak: float, pred: int, gt: int, th: float, tol: int) -> str:
    if peak < th:
        return "no_peak"
    if abs(pred - gt) <= tol:
        return "tp"
    return "wrong"


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ps-pair] device={device}", flush=True)
    model, _ = load_model(Path(args.checkpoint), device)
    ds = STEADPickingDataset("test", seq_len=args.seq_len, load_geometry=True)
    indices = _subsample_indices(ds, args.max_events, args.max_noise, args.subset_seed)
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False)
    tol = tolerance_bins(args.seq_len, args.tol_sec)
    n_ev = sum(1 for i in indices if ds.refs[i].is_event == 1)
    n_nz = len(indices) - n_ev
    print(
        f"[ps-pair] n={len(indices)} events={n_ev} noise={n_nz} batches={len(loader)}",
        flush=True,
    )

    strategies = {
        "baseline": _empty(),
        "fix_weak_empirical": _empty(),
        "fix_weak_pred_gap": _empty(),
        "fix_weak_pred_gap_nr_gate": _empty(),
        "fix_weak_distance": _empty(),
        "pair_anchor_distance": _empty(),
    }
    pair_names = (
        "fix_weak_empirical",
        "fix_weak_pred_gap",
        "fix_weak_pred_gap_nr_gate",
        "fix_weak_distance",
        "pair_anchor_distance",
    )
    gate_stats = Counter()

    t0 = time.time()
    for bi, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["x"], batch["t"])
        det_probs = torch.sigmoid(outputs["det"])
        if det_probs.dim() > 1:
            det_probs = det_probs.amax(dim=-1)
        p_probs = torch.sigmoid(outputs["p"])
        s_probs = torch.sigmoid(outputs["s"])
        p_base, s_base = apply_p_before_s_constraint(p_probs, s_probs, args.pick_threshold)
        noise_ratio = _noise_ratio_from_outputs(outputs)
        det_true = batch["det"] > 0.5
        bsz = batch["x"].size(0)

        for i in range(bsz):
            is_event = bool(det_true[i].item())
            det_ok = bool((det_probs[i] >= args.det_threshold).item())
            dist = float(batch["source_distance_km"][i].item())
            dist = dist if np.isfinite(dist) else None
            nr = (
                float(noise_ratio[i].item())
                if noise_ratio is not None
                else float("nan")
            )
            pred_gap = None
            pred_sigma = None
            if "ps_gap_sec" in outputs:
                pred_gap = float(outputs["ps_gap_sec"][i].item())
                if "ps_gap_log_sigma" in outputs:
                    import math as _math
                    raw = float(outputs["ps_gap_log_sigma"][i].item())
                    pred_sigma = float(_math.log1p(_math.exp(min(raw, 20.0))) + 0.2)  # softplus+0.2
                    pred_sigma = max(0.5, pred_sigma)

            # baseline independent
            bp = float(p_base[i].max().item())
            bi_p = int(p_base[i].argmax().item())
            bs_ = float(s_base[i].max().item())
            bi_s = int(s_base[i].argmax().item())

            common = dict(
                threshold=args.pick_threshold,
                seq_len=args.seq_len,
                topk=args.topk,
                min_frac=args.min_cand_frac,
                min_gap_sec=args.min_gap_sec,
                max_gap_sec=args.max_gap_sec,
                gap_weight=args.gap_weight,
                prior_mu=args.prior_mu_sec,
                prior_sigma=args.prior_sigma_sec,
                homog_s_per_km=args.homog_s_per_km,
                homog_sigma=args.homog_sigma_sec,
            )
            pred_gap_picks = _pair_picks(
                p_probs[i],
                s_probs[i],
                pred_gap_sec=pred_gap,
                pred_gap_sigma=pred_sigma,
                use_pred_gap=pred_gap is not None,
                mode="fix_weak",
                **common,
            )
            nr_gate_on = (
                is_event
                and pred_gap is not None
                and np.isfinite(nr)
                and nr >= float(args.pred_gap_noise_ratio_min)
            )
            if is_event:
                gate_stats["events"] += 1
                if nr_gate_on:
                    gate_stats["nr_gate_on"] += 1
            picks = {
                "baseline": (bp, bi_p, bs_, bi_s),
                "fix_weak_empirical": _pair_picks(
                    p_probs[i], s_probs[i], use_distance=False, mode="fix_weak", **common
                ),
                "fix_weak_pred_gap": pred_gap_picks,
                # High-noise_ratio only: otherwise keep independent baseline picks.
                "fix_weak_pred_gap_nr_gate": pred_gap_picks
                if nr_gate_on
                else (bp, bi_p, bs_, bi_s),
                "fix_weak_distance": _pair_picks(
                    p_probs[i],
                    s_probs[i],
                    distance_km=dist,
                    use_distance=True,
                    mode="fix_weak",
                    **common,
                ),
                "pair_anchor_distance": _pair_picks(
                    p_probs[i],
                    s_probs[i],
                    distance_km=dist,
                    use_distance=True,
                    mode="anchor",
                    **common,
                ),
            }

            gt_p = int(batch["p_idx"][i].item()) if batch["p_valid"][i] > 0 else None
            gt_s = int(batch["s_idx"][i].item()) if batch["s_valid"][i] > 0 else None

            # mode / rescue stats on events with both GT
            if is_event and gt_p is not None and gt_s is not None and det_ok:
                b_pl = _label(bp, bi_p, gt_p, args.pick_threshold, tol)
                b_sl = _label(bs_, bi_s, gt_s, args.pick_threshold, tol)
                strategies["baseline"]["mode_cases"][f"{b_pl}_{b_sl}"] += 1
                for name in pair_names:
                    pp, pi, sp, si = picks[name]
                    pl = _label(pp, pi, gt_p, args.pick_threshold, tol)
                    sl = _label(sp, si, gt_s, args.pick_threshold, tol)
                    strategies[name]["mode_cases"][f"{pl}_{sl}"] += 1
                    if b_pl == "wrong" and b_sl == "tp" and pl == "tp":
                        strategies[name]["rescued"]["p_from_wrong_tp"] += 1
                    if b_pl == "tp" and b_sl == "wrong" and sl == "tp":
                        strategies[name]["rescued"]["s_from_tp_wrong"] += 1
                    if b_pl == "tp" and b_sl == "tp" and (pl != "tp" or sl != "tp"):
                        strategies[name]["rescued"]["broke_tp_tp"] += 1

            for name, (pp, pi, sp, si) in picks.items():
                st = strategies[name]
                _score_one(
                    st, "p", pp, pi, gt_p, args.pick_threshold, det_ok, tol, args.seq_len, is_event
                )
                _score_one(
                    st, "s", sp, si, gt_s, args.pick_threshold, det_ok, tol, args.seq_len, is_event
                )

        if (bi + 1) % 20 == 0 or (bi + 1) == len(loader):
            elapsed = time.time() - t0
            rate = (bi + 1) / max(elapsed, 1e-6)
            eta = (len(loader) - bi - 1) / max(rate, 1e-6)
            print(
                f"[ps-pair] batch {bi+1}/{len(loader)} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "n_eval": len(indices),
        "n_events": n_ev,
        "n_noise": n_nz,
        "config": vars(args),
        "nr_gate": {
            "noise_ratio_min": float(args.pred_gap_noise_ratio_min),
            "events": int(gate_stats["events"]),
            "nr_gate_on": int(gate_stats["nr_gate_on"]),
            "gate_on_frac": float(gate_stats["nr_gate_on"]) / max(int(gate_stats["events"]), 1),
        },
        "strategies": {},
    }
    base_p = base_s = None
    for name, st in strategies.items():
        p_m = pick_prf(st["counts"]["p"]["tp"], st["counts"]["p"]["fp"], st["counts"]["p"]["fn"])
        s_m = pick_prf(st["counts"]["s"]["tp"], st["counts"]["s"]["fp"], st["counts"]["s"]["fn"])
        entry = {
            "p": p_m,
            "s": s_m,
            "p_mae_sec": st["err_sum"]["p"] / max(st["tp_n"]["p"], 1),
            "s_mae_sec": st["err_sum"]["s"] / max(st["tp_n"]["s"], 1),
            "error_breakdown": dict(st["breakdown"]),
            "mode_cases": dict(st["mode_cases"]),
            "rescued": dict(st["rescued"]),
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
    report = evaluate(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[ps-pair] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
