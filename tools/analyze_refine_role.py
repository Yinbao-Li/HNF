#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much does the CNN refine head actually decide? (propose_refine c2f; STEAD val and/or ETHZ dev.)

Physics-only readout = wave proposal w (r = 0); full = w' + σ(w')·R·tanh(r/R), w' = maxpool±pool(w).
Per phase, with the same det gate for both: picks created / erased by the CNN term, timing shift where both pick,
and F1 at several tolerances. Also the analytic bound: the lowest physics probability the CNN can lift over th.

  python tools/analyze_refine_role.py --checkpoint outputs/stead_c2f_imp_RSP/last.pt --datasets stead_val,ethz_dev
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
for sub in ("", "scripts/paper"):
    if str(_REPO / sub) not in sys.path:
        sys.path.insert(0, str(_REPO / sub))

from hnf.picking_metrics import EvalAccumulator, finalize_metrics, update_picking_counts  # noqa: E402
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402

T = 6000
TOLS = (0.1, 0.2, 0.5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--datasets", default="stead_val,ethz_dev")
    p.add_argument("--max-events", type=int, default=3000)
    p.add_argument("--max-noise", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/paper_stead_final_compare/final/interp/refine_role.json"))
    return p.parse_args()


def lift_bound(th, r_max):
    """Smallest physics probability σ(w') from which w' + σ(w')·R can still reach logit(th)."""
    target = math.log(th / (1 - th))
    lo, hi = -30.0, target
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid + r_max / (1 + math.exp(-mid)) >= target:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + math.exp(-hi))


def stead_batches(args):
    ds = STEADPickingDataset("val", seq_len=T, max_event_traces=args.max_events, max_noise_traces=args.max_noise,
                             seed=42, load_geometry=False)
    for b in DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2):
        yield b["x"], {k: b[k] for k in ("det", "p_idx", "s_idx", "p_valid", "s_valid")}


def ethz_batches(args):
    from tools.eval_ethz_cross import P_COLS, S_COLS, cut_window, first_arrival, load_dataset
    from scipy.signal import butter, sosfiltfilt

    data = load_dataset("ethz", "dev")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(data))[:args.max_events]
    sos = butter(4, [1, 45], btype="bandpass", fs=100.0, output="sos")
    buf, lab = [], {k: [] for k in ("det", "p_idx", "s_idx", "p_valid", "s_valid")}

    def flush():
        x = torch.from_numpy(sosfiltfilt(sos, np.stack(buf), axis=-1).astype(np.float32))
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        out = x.transpose(1, 2).contiguous(), {k: torch.tensor(v) for k, v in lab.items()}
        buf.clear()
        for v in lab.values():
            v.clear()
        return out

    for i in order:
        w, meta = data.get_sample(int(i))
        p = first_arrival(meta, P_COLS)
        if p is None or w.shape[0] != 3:
            continue
        s = first_arrival(meta, S_COLS)
        tp = int(round(rng.uniform(5, 30) * 100))
        win, _, _ = cut_window(np.nan_to_num(np.asarray(w, dtype=np.float32)), int(round(p)), tp)
        s_rel = None if s is None else int(round(s)) - int(round(p)) + tp
        s_ok = s_rel is not None and 50 <= s_rel <= T - 50 and s_rel > tp
        buf.append(win)
        lab["det"].append(1.0)
        lab["p_idx"].append(tp)
        lab["s_idx"].append(s_rel if s_ok else 0)
        lab["p_valid"].append(1.0)
        lab["s_valid"].append(1.0 if s_ok else 0.0)
        if len(buf) == args.batch_size:
            yield flush()
    if buf:
        yield flush()


@torch.no_grad()
def analyse(model, batches, args):
    th = args.pick_threshold
    k = model.refine_pool_bins
    acc = {v: {ph: {t: EvalAccumulator() for t in TOLS} for ph in "ps"} for v in ("full", "physics")}
    st = {ph: {"n_traces": 0, "created": 0, "erased": 0, "created_correct": 0, "erased_correct": 0,
               "shift_s": []} for ph in "ps"}
    t = torch.linspace(0.0, 60.0, T).view(1, T, 1)
    for x, y in batches:
        x = x.to(args.device)
        out = model(x, t.expand(x.size(0), -1, -1).to(args.device))
        gate = (torch.sigmoid(out["det"].float()) >= args.det_threshold).cpu()
        det_true = y["det"] > 0.5
        for ph in "ps":
            full = torch.sigmoid(out[ph].float()).cpu() * gate.unsqueeze(-1)
            phys = torch.sigmoid(out[f"{ph}_wave"].float()).cpu() * gate.unsqueeze(-1)
            valid = (y[f"{ph}_valid"] > 0) & det_true
            idx = y[f"{ph}_idx"]
            for tol in TOLS:
                tb = max(1, int(round(tol * 100)))
                for name, prob in (("full", full), ("physics", phys)):
                    a = acc[name][ph][tol]
                    update_picking_counts(a.p, prob, gate, det_true, valid, idx, th, tb, T)
            fmax, fi = full.max(-1)
            pmax, pi = phys.max(-1)
            fy, py = fmax >= th, pmax >= th
            ok_f = valid & ((fi - idx).abs() <= 50)
            ok_p = valid & ((pi - idx).abs() <= 50)
            s = st[ph]
            s["n_traces"] += len(fy)
            s["created"] += int((fy & ~py).sum())
            s["erased"] += int((~fy & py).sum())
            s["created_correct"] += int((fy & ~py & ok_f).sum())
            s["erased_correct"] += int((~fy & py & ok_p).sum())
            both = fy & py
            s["shift_s"] += ((fi[both] - pi[both]).float() / 100.0).tolist()
    res = {}
    for ph in "ps":
        s = st[ph]
        sh = np.abs(np.asarray(s.pop("shift_s")))
        s["n_both"] = int(len(sh))
        if len(sh):
            s["abs_shift_s"] = {q: float(np.quantile(sh, float(q))) for q in ("0.5", "0.9", "0.99")}
            s["frac_shift_le_pool"] = float((sh <= k / 100.0 + 1e-6).mean())
        s["f1"] = {name: {f"{tol:g}": finalize_metrics(acc[name][ph][tol])["p_f1"] for tol in TOLS}
                   for name in acc}
        res[ph] = s
    return res


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, torch.device(args.device))[0].eval()
    bound = lift_bound(args.pick_threshold, model.refine_max)
    report = {"checkpoint": str(args.checkpoint), "threshold": args.pick_threshold,
              "refine_max": model.refine_max, "refine_pool_s": model.refine_pool_bins / 100.0,
              "min_physics_prob_liftable": bound, "datasets": {}}
    print(f"[bound] CNN can only lift picks whose physics prob ≥ {bound:.3f} (th {args.pick_threshold})", flush=True)
    for name in [s for s in args.datasets.split(",") if s]:
        gen = stead_batches(args) if name == "stead_val" else ethz_batches(args)
        r = analyse(model, gen, args)
        report["datasets"][name] = r
        for ph in "ps":
            s = r[ph]
            print(f"  {name} {ph.upper()}: traces={s['n_traces']} created={s['created']} "
                  f"(correct {s['created_correct']}) erased={s['erased']} (correct {s['erased_correct']}) "
                  f"|shift| med/90/99={s.get('abs_shift_s')} ≤pool={s.get('frac_shift_le_pool')} "
                  f"F1 full={s['f1']['full']} phys={s['f1']['physics']}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output}")


if __name__ == "__main__":
    main()
