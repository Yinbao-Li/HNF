#!/usr/bin/env python3
"""Seed-averaged AE zero-shot metrics with paired-bootstrap CIs (per-trace dumps of tools/train_ae_picking.py).

For each material / tolerance: F1 and AP per model averaged over evaluation seeds (± SD across seeds), and
ours − baseline differences with 95% CIs from a paired bootstrap over traces (same resample for all models
within a seed; the seed-mean difference is formed per bootstrap replicate).

  python tools/ae_zs_stats.py --root outputs/ae/zs_seeds --output outputs/ae/zs_seeds/stats.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.train_ae_picking import average_precision  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/ae/zs_seeds"))
    p.add_argument("--models", default="ours,eqt_aug,pn_aug,eqt_official,pn_official")
    p.add_argument("--ref", default="ours")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--tols", default="10,20,50")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--output", type=Path, default=Path("outputs/ae/zs_seeds/stats.json"))
    return p.parse_args()


def f1(c):
    tp, fp, fn = c.sum(0)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def ap(pt, t, idx):
    return average_precision([pt["scores"][i] for i in idx], [pt["hits"][t][i] for i in idx], int(pt["n_true"][idx].sum()))


def main():
    args = parse_args()
    rng = np.random.default_rng(0)
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    tols = [int(t) for t in args.tols.split(",")]
    report = {}
    for mat in ("Minusil_test", "Glass_beads_test"):
        data = {m: {s: pickle.load(open(args.root / m / f"zs_s{s}_{mat}_per_trace.pkl", "rb")) for s in seeds}
                for m in models}
        n = {s: len(data[models[0]][s]["n_true"]) for s in seeds}
        for s in seeds:
            assert all(len(data[m][s]["n_true"]) == n[s] and np.array_equal(data[m][s]["n_true"], data[models[0]][s]["n_true"])
                       for m in models), "trace sets differ across models"
        boots = {s: [rng.integers(0, n[s], n[s]) for _ in range(args.n_boot)] for s in seeds}
        rep = {}
        for t in tols:
            row = {}
            point = {m: {"f1": [f1(data[m][s]["counts"][t]) for s in seeds],
                         "ap": [ap(data[m][s], t, np.arange(n[s])) for s in seeds]} for m in models}
            bs = {m: {"f1": np.zeros((args.n_boot, len(seeds))), "ap": np.zeros((args.n_boot, len(seeds)))} for m in models}
            for j, s in enumerate(seeds):
                for b, idx in enumerate(boots[s]):
                    for m in models:
                        bs[m]["f1"][b, j] = f1(data[m][s]["counts"][t][idx])
                        bs[m]["ap"][b, j] = ap(data[m][s], t, idx)
            for m in models:
                r = {k: {"mean": float(np.mean(v)), "sd_seeds": float(np.std(v)),
                         "ci": [float(np.percentile(bs[m][k].mean(1), 2.5)), float(np.percentile(bs[m][k].mean(1), 97.5))]}
                     for k, v in point[m].items()}
                if m != args.ref:
                    for k in ("f1", "ap"):
                        d = (bs[args.ref][k] - bs[m][k]).mean(1)
                        r[f"{k}_diff_ref_minus_this"] = {
                            "mean": float(np.mean(point[args.ref][k]) - np.mean(point[m][k])),
                            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                            "p_ref_not_better": float((d <= 0).mean())}
                row[m] = r
            rep[f"tol{t}"] = row
            print(f"[{mat} tol{t}] " + " | ".join(
                f"{m} F1 {row[m]['f1']['mean']:.3f}±{row[m]['f1']['sd_seeds']:.3f} AP {row[m]['ap']['mean']:.3f}"
                + (f" (Δ F1 {row[m]['f1_diff_ref_minus_this']['mean']:+.3f} [{row[m]['f1_diff_ref_minus_this']['ci'][0]:+.3f},"
                   f"{row[m]['f1_diff_ref_minus_this']['ci'][1]:+.3f}] Δ AP {row[m]['ap_diff_ref_minus_this']['mean']:+.3f} "
                   f"[{row[m]['ap_diff_ref_minus_this']['ci'][0]:+.3f},{row[m]['ap_diff_ref_minus_this']['ci'][1]:+.3f}])"
                   if m != args.ref else "") for m in models), flush=True)
        report[mat] = rep
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output}", flush=True)


if __name__ == "__main__":
    main()
