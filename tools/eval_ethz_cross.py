#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-dataset test: STEAD-trained pickers on the SeisBench ETHZ test split (Swiss network).

Traces are resampled to 100 Hz (channel order E, N, Z). P = earliest of P/P1/Pg/Pn, S = earliest of
S/S1/Sg/Sn. Each event trace is cut to a 60 s window with P placed uniformly in [p_min, p_max] s
(seeded); if the trace is too short, the window is padded with the trace's own pre-P noise.
Same threshold for all models (th 0.3, det 0.5); c2f uses the hard det gate.

  python tools/eval_ethz_cross.py --c2f final=outputs/stead_c2f_lft_RSP/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for sub in ("", "scripts/paper"):
    if str(_REPO / sub) not in sys.path:
        sys.path.insert(0, str(_REPO / sub))

import seisbench.data as sbd  # noqa: E402
from run_paper_stead_triple_compare import load_sb  # noqa: E402
from tools.eval_shift_invariance import load_sbc, run_c2f, run_sb, run_sbc  # noqa: E402
from tools.sweep_seisbench_threshold import MODEL_MAP, metrics_at  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402

T = 6000
P_COLS = ["trace_P_arrival_sample", "trace_P1_arrival_sample", "trace_Pg_arrival_sample", "trace_Pn_arrival_sample",
          "trace_p_arrival_sample"]
S_COLS = ["trace_S_arrival_sample", "trace_S1_arrival_sample", "trace_Sg_arrival_sample", "trace_Sn_arrival_sample",
          "trace_s_arrival_sample"]
GROUP_COL = {"ethz": "trace_channel", "obs": "trace_instrument_type"}


def load_dataset(name: str, split: str):
    if name == "ethz":
        data = sbd.ETHZ(sampling_rate=100, component_order="ENZ")
    elif name == "obs":
        # OBS horizontals 1/2 have unknown azimuth; only Z/1/2-complete traces are kept.
        data = sbd.OBS(sampling_rate=100, component_order="21Z", download_if_missing=False)
    else:
        raise ValueError(name)
    data = getattr(data, split)()
    if name == "obs":
        order = data.metadata["trace_component_order"].astype(str)
        data.filter(order.str.contains("Z") & order.str.contains("1") & order.str.contains("2"))
    return data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c2f", default="final=outputs/stead_c2f_lft_RSP/best.pt")
    p.add_argument("--seisbench", default="eqt,phasenet")
    p.add_argument("--sbc", default="", help="name=ckpt,... from tools/train_sb_stead_aug.py")
    p.add_argument("--dataset", choices=["ethz", "obs"], default="ethz")
    p.add_argument("--split", default="test")
    p.add_argument("--z-only", action="store_true", help="zero both horizontals (diagnostic)")
    p.add_argument("--max-traces", type=int, default=0, help="0 = all")
    p.add_argument("--p-min", type=float, default=5.0)
    p.add_argument("--p-max", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tolerances", default="0.1,0.2,0.5")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bandpass", default="1,45",
                   help="'fmin,fmax' Hz zero-phase Butterworth applied to every window for all models "
                        "(STEAD's own band); '' = raw")
    p.add_argument("--output", type=Path, default=Path("outputs/paper_stead_final_compare/final/ethz_cross.json"))
    return p.parse_args()


def first_arrival(meta: dict, cols: list[str]) -> float | None:
    vals = [meta.get(c) for c in cols]
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return min(vals) if vals else None


def cut_window(w: np.ndarray, p: int, target_p: int) -> tuple[np.ndarray, int, int]:
    """Return a (3, T) window with P at target_p (padding with pre-P noise), and pad sizes."""
    start = p - target_p
    noise = w[:, :max(p - 50, 100)]
    if noise.shape[1] < 10:
        noise = np.zeros((3, 100), dtype=w.dtype)
    left = max(0, -start)
    seg = w[:, max(0, start):max(0, start) + T - left]
    right = T - left - seg.shape[1]

    def pad(n):
        reps = int(np.ceil(n / noise.shape[1]))
        return np.tile(noise, reps)[:, :n]

    out = np.concatenate([pad(left), seg, pad(right)], axis=1) if (left or right) else seg
    return out.astype(np.float32), left, right


def main():
    args = parse_args()
    device = torch.device(args.device)
    tols = [float(x) for x in args.tolerances.split(",")]
    rng = np.random.default_rng(args.seed)

    data = load_dataset(args.dataset, args.split)
    n = len(data) if args.max_traces <= 0 else min(args.max_traces, len(data))
    order = rng.permutation(len(data))[:n]

    waves, y = [], {"p_idx": [], "s_idx": [], "p_valid": [], "s_valid": [], "is_event": []}
    groups = []
    n_pad = n_dead = 0
    for i in order:
        w, meta = data.get_sample(int(i))
        p = first_arrival(meta, P_COLS)
        if p is None or w.shape[0] != 3:
            continue
        if args.dataset == "obs" and not (np.nan_to_num(np.asarray(w, dtype=np.float32)).std(axis=1) > 0).all():
            n_dead += 1
            continue
        g = meta.get(GROUP_COL[args.dataset])
        groups.append(str(g) if isinstance(g, str) and g else "land/other")
        s = first_arrival(meta, S_COLS)
        p = int(round(p))
        tp = int(round(rng.uniform(args.p_min, args.p_max) * 100))
        win, left, right = cut_window(np.nan_to_num(np.asarray(w, dtype=np.float32)), p, tp)
        n_pad += int(left > 0 or right > 0)
        s_rel = None if s is None else int(round(s)) - p + tp
        s_ok = s_rel is not None and 50 <= s_rel <= T - 50 and s_rel > tp
        waves.append(win)
        y["p_idx"].append(tp)
        y["s_idx"].append(s_rel if s_ok else 0)
        y["p_valid"].append(True)
        y["s_valid"].append(bool(s_ok))
        y["is_event"].append(True)
    waves = np.stack(waves)
    if args.bandpass:
        from scipy.signal import butter, sosfiltfilt
        lo, hi = (float(x) for x in args.bandpass.split(","))
        sos = butter(4, [lo, hi], btype="bandpass", fs=100.0, output="sos")
        waves = sosfiltfilt(sos, waves, axis=-1).astype(np.float32)
    if args.z_only:
        waves[:, :2] = 0.0
    y = {k: torch.tensor(v) for k, v in y.items()}
    print(f"[{args.dataset}] {len(waves)} event windows (dead-channel dropped {n_dead}) ({int(y['s_valid'].sum())} with S), padded {n_pad}", flush=True)

    models = {}
    for spec in [s for s in args.c2f.split(",") if s]:
        name, path = spec.split("=")
        models[name] = ("c2f", load_model(Path(path), device)[0])
    for spec in [s for s in args.sbc.split(",") if s]:
        name, path = spec.split("=")
        kind, m = load_sbc(Path(path), device)
        models[name] = ("sbc:" + kind, m)
    for key in [s for s in args.seisbench.split(",") if s]:
        label, cls, weights, kind = MODEL_MAP[key]
        models[label] = (kind, load_sb(cls, weights).to(device).eval())

    report = {"dataset": args.dataset, "split": args.split,
              "n_windows": len(waves), "n_with_s": int(y["s_valid"].sum()), "n_padded": n_pad,
              "n_dead_channel_dropped": n_dead,
              "bandpass": args.bandpass,
              "p_range_sec": [args.p_min, args.p_max], "tolerances": tols, "results": {}}
    for name, (kind, model) in models.items():
        if kind == "c2f":
            det, p, s = run_c2f(model, waves, device, args.batch_size, args.det_threshold)
            mk = "eqt"
        elif kind.startswith("sbc:"):
            mk = kind[4:]
            det, p, s = run_sbc(model, mk, waves, device, args.batch_size)
        else:
            det, p, s = run_sb(model, kind, waves, device, args.batch_size)
            mk = kind
        row = {}
        for tol in tols:
            m = metrics_at(det, p, s, y, mk, args.pick_threshold, args.det_threshold,
                           max(1, int(round(tol * 100))), device)
            row[f"{tol:g}"] = {k: m[k] for k in m if k.startswith(("p_", "s_", "det_"))}
        by_group = {}
        for g in sorted(set(groups)):
            idx = torch.tensor([j for j, gg in enumerate(groups) if gg == g])
            if len(idx) < 50:
                continue
            m = metrics_at(None if det is None else det[idx], p[idx], s[idx],
                           {k: v[idx] for k, v in y.items()}, mk, args.pick_threshold,
                           args.det_threshold, 50, device)
            by_group[g] = {"n": len(idx), "p_f1": m["p_f1"], "s_f1": m["s_f1"]}
        row["by_group_tol0.5"] = by_group
        report["results"][name] = row
        print("  " + f"{name:16s} " + "  ".join(
            f"tol{t:g} P={row[f'{t:g}']['p_f1']:.4f} S={row[f'{t:g}']['s_f1']:.4f}" for t in tols), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output}", flush=True)


if __name__ == "__main__":
    main()
