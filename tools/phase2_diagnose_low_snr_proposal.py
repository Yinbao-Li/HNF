#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase B condition 1: at fixed SNR, decompose misses into proposal vs refine.

On frozen imp_RSP, for SNR in {-5, 0} dB (and clean), report per phase P/S:
  (a) proposal recall: fraction of GT picks with physics proposal within tol
  (b) refine eccentricity among windows with a correct proposal:
      |argmax(full) − GT| in seconds (mean/median); also |argmax(full)−argmax(prop)|

  python tools/phase2_diagnose_low_snr_proposal.py --device cuda \
    --output results/phase2/2_4_low_snr/diagnosis_proposal_vs_refine.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from hnf.robust_aug import add_real_noise  # noqa: E402
from tools.eval_snr_curve import load_sets  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402

T = 6000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--snrs", default="clean,0,-5")
    p.add_argument("--n-events", type=int, default=3000)
    p.add_argument("--n-noise", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("results/phase2/2_4_low_snr/diagnosis_proposal_vs_refine.json"))
    return p.parse_args()


@torch.no_grad()
def physics_and_full(model, x):
    """Return det (B,), physics-pooled proposal logits, full logits (B,T) for P and S."""
    t = torch.linspace(0.0, 60.0, T, device=x.device).view(1, T, 1).expand(x.size(0), -1, -1)
    out = model(x, t)
    det = torch.sigmoid(out["det"].float())
    pw, sw = out["p_wave"].float(), out["s_wave"].float()
    pr, sr = out["p_ref"].float(), out["s_ref"].float()
    r_max = float(model.refine_max)
    eps = float(model.gate_floor)
    k = int(model.refine_pool_bins)
    p_base, s_base = pw, sw
    if k > 0:
        p_base = F.max_pool1d(pw.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)
        s_base = F.max_pool1d(sw.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)
    p_full = p_base + (eps + (1 - eps) * torch.sigmoid(p_base)) * r_max * torch.tanh(pr / r_max)
    s_full = s_base + (eps + (1 - eps) * torch.sigmoid(s_base)) * r_max * torch.tanh(sr / r_max)
    return det, p_base, s_base, p_full, s_full


def analyze_phase(prop_logit, full_logit, gate_bt, gt_idx, pick_th, tol_bins):
    """prop/full (N,T) logits; gate_bt (N,1) det gate."""
    prop_prob = torch.sigmoid(prop_logit) * gate_bt
    full_prob = torch.sigmoid(full_logit) * gate_bt
    prop_max, prop_idx = prop_prob.max(-1)
    full_max, full_idx = full_prob.max(-1)
    gt = gt_idx.long()
    n = gt.numel()
    # (a) proposal recall: GT covered if |prop_idx - gt|<=tol AND prop peak >= th
    # Also accept: any prop mass within tol of GT above th (local max search)
    # Use peak-based definition consistent with picker decode:
    prop_ok = (prop_max >= pick_th) & ((prop_idx - gt).abs() <= tol_bins)
    # Alternative coverage: max prop_prob in [gt-tol, gt+tol] >= th
    t = torch.arange(prop_prob.size(-1), device=prop_prob.device).view(1, -1)
    near = (t - gt.view(-1, 1)).abs() <= tol_bins
    local_max = prop_prob.masked_fill(~near, 0.0).amax(-1)
    prop_cover = local_max >= pick_th
    # among covered: refine eccentricity
    cover = prop_cover
    full_err = (full_idx - gt).abs().float() / 100.0  # seconds
    prop_err = (prop_idx - gt).abs().float() / 100.0
    refine_shift = (full_idx - prop_idx).abs().float() / 100.0
    full_hit = (full_max >= pick_th) & ((full_idx - gt).abs() <= tol_bins)
    # dominate test inputs
    miss = ~full_hit
    miss_due_no_prop = miss & (~prop_cover)
    miss_with_prop = miss & prop_cover  # proposal OK but full miss → refine/threshold issue
    return {
        "n": int(n),
        "proposal_recall_peak": float(prop_ok.float().mean()),
        "proposal_recall_local": float(prop_cover.float().mean()),
        "full_recall_hit": float(full_hit.float().mean()),
        "frac_miss": float(miss.float().mean()),
        "frac_miss_no_proposal": float(miss_due_no_prop.float().mean()),
        "frac_miss_with_proposal": float(miss_with_prop.float().mean()),
        "among_correct_proposal": {
            "n": int(cover.sum()),
            "full_hit_rate": float(full_hit[cover].float().mean()) if cover.any() else None,
            "refine_ecc_vs_gt_mean_sec": float(full_err[cover].mean()) if cover.any() else None,
            "refine_ecc_vs_gt_median_sec": float(full_err[cover].median()) if cover.any() else None,
            "prop_ecc_vs_gt_mean_sec": float(prop_err[cover].mean()) if cover.any() else None,
            "refine_shift_from_prop_mean_sec": float(refine_shift[cover].mean()) if cover.any() else None,
            "frac_full_farther_than_prop": float((full_err[cover] > prop_err[cover]).float().mean())
            if cover.any()
            else None,
        },
        "verdict_hint": (
            "proposal_dominated"
            if float(miss_due_no_prop.float().mean()) >= float(miss_with_prop.float().mean())
            else "refine_dominated"
        ),
    }


@torch.no_grad()
def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, _, _ = load_model(args.checkpoint, device)
    assert model.fusion == "propose_refine"
    ev_w, pool_w, only_w, y_ev = load_sets(args.n_events, args.n_noise, args.seed, "val")
    # diagnosis uses events only (proposal recall needs GT)
    n_ev = len(ev_w)
    tol = max(1, int(round(args.tol_sec * 100)))
    report = {
        "checkpoint": str(args.checkpoint),
        "split": "val",
        "n_events": n_ev,
        "seed": args.seed,
        "tol_sec": args.tol_sec,
        "pick_threshold": args.pick_threshold,
        "det_threshold": args.det_threshold,
        "refine_max": float(model.refine_max),
        "refine_pool_bins": int(model.refine_pool_bins),
        "gate_floor": float(model.gate_floor),
        "snrs": {},
    }
    for s in [x.strip() for x in args.snrs.split(",")]:
        if s == "clean":
            w = ev_w.astype(np.float32)
        else:
            w = np.stack(
                [add_real_noise(ev_w[i], int(y_ev["p_idx"][i]), pool_w[i], float(s)) for i in range(n_ev)]
            ).astype(np.float32)
        # run in batches with manuscript normalize
        dets, pbs, sbs, pfs, sfs = [], [], [], [], []
        for a in range(0, n_ev, args.batch_size):
            chunk = w[a : a + args.batch_size]
            x = torch.from_numpy(chunk).to(device)
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
            x = x.transpose(1, 2).contiguous()
            det, pb, sb, pf, sf = physics_and_full(model, x)
            dets.append(det.cpu())
            pbs.append(pb.cpu())
            sbs.append(sb.cpu())
            pfs.append(pf.cpu())
            sfs.append(sf.cpu())
        det = torch.cat(dets)
        # det is (N,) per-trace; broadcast as (N,1)
        gate_bt = (det >= args.det_threshold).float().view(-1, 1)
        pb, sb = torch.cat(pbs), torch.cat(sbs)
        pf, sf = torch.cat(pfs), torch.cat(sfs)
        row = {
            "p": analyze_phase(pb, pf, gate_bt, y_ev["p_idx"], args.pick_threshold, tol),
            "s": analyze_phase(sb, sf, gate_bt, y_ev["s_idx"], args.pick_threshold, tol),
        }
        # overall dominate: use P as primary (success metric is P F1)
        row["condition1_primary_phase"] = "p"
        row["condition1_verdict"] = row["p"]["verdict_hint"]
        report["snrs"][s] = row
        print(
            f"[snr={s}] P proposal_recall_local={row['p']['proposal_recall_local']:.3f} "
            f"full_hit={row['p']['full_recall_hit']:.3f} "
            f"miss_no_prop={row['p']['frac_miss_no_proposal']:.3f} "
            f"miss_with_prop={row['p']['frac_miss_with_proposal']:.3f} "
            f"refine_ecc_med={row['p']['among_correct_proposal']['refine_ecc_vs_gt_median_sec']} "
            f"→ {row['condition1_verdict']}",
            flush=True,
        )
        print(
            f"         S proposal_recall_local={row['s']['proposal_recall_local']:.3f} "
            f"miss_no_prop={row['s']['frac_miss_no_proposal']:.3f} "
            f"miss_with_prop={row['s']['frac_miss_with_proposal']:.3f} "
            f"→ {row['s']['verdict_hint']}",
            flush=True,
        )

    # Gate for proceeding: at -5 dB, P must be proposal_dominated
    v = report["snrs"].get("-5", {}).get("condition1_verdict")
    report["proceed_to_fix"] = v == "proposal_dominated"
    report["stop_reason"] = None if report["proceed_to_fix"] else (
        f"-5 dB P verdict={v}; refine eccentricity dominates — do not change proposal"
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] proceed={report['proceed_to_fix']} → {args.output}", flush=True)


if __name__ == "__main__":
    main()
