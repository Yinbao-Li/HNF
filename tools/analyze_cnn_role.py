#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much does the CNN refine head actually change the picks? (propose_refine c2f; det-gated, th 0.3)

physics-only logit = w' (= maxpool±pool of the wave head), full logit = w' + σ(w')·R·tanh(r/R).
Per phase reports: pick-existence agreement (CNN creates / erases a pick), the time shift the CNN applies
(argmax full vs argmax of the unpooled wave head), errors of both against the label, and the largest
full probability anywhere the physics probability is < 0.05.

  python tools/analyze_cnn_role.py --checkpoint outputs/stead_c2f_imp_RSP/last.pt --datasets stead_val,ethz_dev
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
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402

T = 6000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--datasets", default="stead_val,ethz_dev")
    p.add_argument("--n-events", type=int, default=3000)
    p.add_argument("--n-noise", type=int, default=1000)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--output", type=Path, default=Path("outputs/paper_stead_final_compare/final/interp/cnn_role.json"))
    return p.parse_args()


def stead_val(n_ev, n_nz):
    from hnf.stead_picking_dataset import STEADPickingDataset

    ds = STEADPickingDataset("val", seq_len=T, max_event_traces=n_ev, max_noise_traces=n_nz, seed=42,
                             load_geometry=False)
    xs, y = [], {k: [] for k in ("p_idx", "s_idx", "p_valid", "s_valid", "is_event")}
    for i in range(len(ds)):
        b = ds[i]
        xs.append(b["x"].numpy().T)
        y["p_idx"].append(int(b["p_idx"]))
        y["s_idx"].append(int(b["s_idx"]))
        y["p_valid"].append(bool(b["p_valid"] > 0))
        y["s_valid"].append(bool(b["s_valid"] > 0))
        y["is_event"].append(bool(b["det"] > 0.5))
    return np.stack(xs).astype(np.float32), {k: np.asarray(v) for k, v in y.items()}


def ethz_dev(n_ev):
    from scipy.signal import butter, sosfiltfilt

    from tools.eval_ethz_cross import P_COLS, S_COLS, cut_window, first_arrival, load_dataset

    data = load_dataset("ethz", "dev")
    rng = np.random.default_rng(0)
    xs, y = [], {k: [] for k in ("p_idx", "s_idx", "p_valid", "s_valid", "is_event")}
    for i in rng.permutation(len(data)):
        if len(xs) >= n_ev:
            break
        w, meta = data.get_sample(int(i))
        p = first_arrival(meta, P_COLS)
        if p is None or w.shape[0] != 3:
            continue
        s = first_arrival(meta, S_COLS)
        tp = int(round(rng.uniform(5, 30) * 100))
        win, _, _ = cut_window(np.nan_to_num(np.asarray(w, dtype=np.float32)), int(round(p)), tp)
        s_rel = None if s is None else int(round(s)) - int(round(p)) + tp
        s_ok = s_rel is not None and 50 <= s_rel <= T - 50 and s_rel > tp
        xs.append(win)
        y["p_idx"].append(tp)
        y["s_idx"].append(s_rel if s_ok else 0)
        y["p_valid"].append(True)
        y["s_valid"].append(bool(s_ok))
        y["is_event"].append(True)
    sos = butter(4, [1.0, 45.0], btype="bandpass", fs=100.0, output="sos")
    xs = sosfiltfilt(sos, np.stack(xs), axis=-1).astype(np.float32)
    return xs, {k: np.asarray(v) for k, v in y.items()}


@torch.no_grad()
def collect(model, waves, bs, det_th):
    k = model.refine_pool_bins
    r_max = model.refine_max
    keys = ("full", "base", "wave")
    out = {f"{ph}_{kk}": [] for ph in "ps" for kk in keys}
    out["gate"] = []
    for a in range(0, len(waves), bs):
        x = torch.from_numpy(waves[a:a + bs])
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
        o = model(x.transpose(1, 2).contiguous())
        gate = torch.sigmoid(o["det"].float()) >= det_th
        out["gate"].append(gate)
        for ph in "ps":
            w = o[f"{ph}_wave"].float()
            base = F.max_pool1d(w.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1) if k > 0 else w
            full = o[ph].float()
            assert torch.allclose(full, base + torch.sigmoid(base) * r_max * torch.tanh(o[f"{ph}_ref"].float() / r_max),
                                  atol=1e-3)
            out[f"{ph}_full"].append(torch.sigmoid(full).half())
            out[f"{ph}_base"].append(torch.sigmoid(base).half())
            out[f"{ph}_wave"].append(torch.sigmoid(w).half())
    return {kk: torch.cat(v) for kk, v in out.items()}


def summarise(c, y, th):
    gate = c["gate"].numpy()
    rep = {}
    for ph in "ps":
        full, base, wave = (c[f"{ph}_{k}"].float().numpy() for k in ("full", "base", "wave"))
        has_full = (full.max(-1) >= th) & gate
        has_base = (base.max(-1) >= th) & gate
        both = has_full & has_base
        t_full, t_wave = full.argmax(-1), wave.argmax(-1)
        shift_ms = (t_full - t_wave)[both] * 10.0
        lab = y[f"{ph}_valid"] & y["is_event"]
        m = both & lab
        err_full = np.abs(t_full - y[f"{ph}_idx"])[m] * 10.0
        err_wave = np.abs(t_wave - y[f"{ph}_idx"])[m] * 10.0
        quiet = base < 0.05
        rep[ph] = {
            "n": int(len(full)),
            "pick_agree": float((has_full == has_base).mean()),
            "cnn_creates": int((has_full & ~has_base).sum()),
            "cnn_erases": int((~has_full & has_base).sum()),
            "n_picks_full": int(has_full.sum()),
            "shift_ms_abs_quantiles_50_90_99": [float(q) for q in np.percentile(np.abs(shift_ms), [50, 90, 99])],
            "shift_within_pool": float((np.abs(shift_ms) <= model_pool_ms + 10).mean()),
            "err_ms_median_full_vs_wave": [float(np.median(err_full)), float(np.median(err_wave))],
            "within_100ms_full_vs_wave": [float((err_full <= 100).mean()), float((err_wave <= 100).mean())],
            "max_full_prob_where_physics_below_0.05": float(full[quiet].max()) if quiet.any() else None,
        }
    return rep


def main():
    global model_pool_ms
    args = parse_args()
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, torch.device("cpu"))[0].eval()
    model_pool_ms = model.refine_pool_bins * 10.0
    report = {"checkpoint": str(args.checkpoint), "pool_ms": model_pool_ms, "refine_max": model.refine_max}
    for name in [s for s in args.datasets.split(",") if s]:
        waves, y = stead_val(args.n_events, args.n_noise) if name == "stead_val" else ethz_dev(args.n_events)
        print(f"[{name}] {len(waves)} windows", flush=True)
        c = collect(model, waves, args.batch_size, args.det_threshold)
        report[name] = summarise(c, y, args.pick_threshold)
        for ph in "ps":
            r = report[name][ph]
            print(f"  {ph.upper()}: agree={r['pick_agree']:.4f} creates={r['cnn_creates']} erases={r['cnn_erases']} "
                  f"of {r['n_picks_full']} | |shift| p50/90/99={r['shift_ms_abs_quantiles_50_90_99']} ms "
                  f"within pool={r['shift_within_pool']:.3f} | err med full/wave={r['err_ms_median_full_vs_wave']} "
                  f"≤100ms full/wave={r['within_100ms_full_vs_wave']} | max p(full) where phys<0.05="
                  f"{r['max_full_prob_where_physics_below_0.05']}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output}")


model_pool_ms = 0.0

if __name__ == "__main__":
    main()
