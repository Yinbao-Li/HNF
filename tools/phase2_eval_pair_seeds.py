#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Phase 2.1 pair × seeds: STEAD val from history + ETHZ/OBS via eval_ethz_cross."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/phase2/2_1_pair_seeds"))
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def history_last_val(hist_path: Path) -> dict:
    h = json.loads(hist_path.read_text())
    last = h[-1]
    v = last.get("val") or last
    return {
        "epoch": last.get("epoch"),
        "p_f1": float(v.get("p_f1", v.get("val", {}).get("p_f1", float("nan")))),
        "s_f1": float(v.get("s_f1", float("nan"))) if "s_f1" in v else float(v.get("val", {}).get("s_f1", float("nan"))),
        "raw": last,
    }


def mean_std(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    report = {"seeds": seeds, "arms": {}, "table_viii_style": {}}
    for arm in ("huygens", "free"):
        rows = []
        for seed in seeds:
            d = args.root / f"{arm}_seed{seed}"
            hist = d / "history.json"
            ckpt = d / "last.pt"
            row = {"seed": seed, "dir": str(d), "status": "ok"}
            if not hist.exists() or not ckpt.exists():
                row["status"] = "MISSING"
                rows.append(row)
                continue
            hv = history_last_val(hist)
            row["stead_val"] = {"p_f1": hv["p_f1"], "s_f1": hv["s_f1"], "epoch": hv["epoch"]}
            # cross-domain eval
            for ds in ("ethz", "obs"):
                out = args.root / "eval" / f"{arm}_seed{seed}_{ds}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    cmd = [
                        sys.executable,
                        "tools/eval_ethz_cross.py",
                        "--c2f",
                        f"{arm}={ckpt}",
                        "--seisbench",
                        "",
                        "--dataset",
                        ds,
                        "--device",
                        args.device,
                        "--output",
                        str(out),
                    ]
                    print("[eval]", " ".join(cmd), flush=True)
                    subprocess.check_call(cmd)
                ej = json.loads(out.read_text())
                r = ej["results"][arm]
                row[ds] = {
                    "p_f1_0.5": r["0.5"]["p_f1"],
                    "s_f1_0.5": r["0.5"]["s_f1"],
                    "p_f1_0.1": r["0.1"]["p_f1"],
                    "s_f1_0.1": r["0.1"]["s_f1"],
                }
            rows.append(row)
        report["arms"][arm] = rows
        # mean±std
        def collect(keypath):
            vals = []
            for r in rows:
                if r.get("status") != "ok":
                    continue
                cur = r
                for k in keypath:
                    cur = cur[k]
                vals.append(cur)
            return mean_std(vals) if vals else (float("nan"), float("nan"))

        report["table_viii_style"][arm] = {
            "stead_val_p_f1": collect(["stead_val", "p_f1"]),
            "stead_val_s_f1": collect(["stead_val", "s_f1"]),
            "ethz_p_f1_0.5": collect(["ethz", "p_f1_0.5"]),
            "ethz_s_f1_0.5": collect(["ethz", "s_f1_0.5"]),
            "obs_p_f1_0.5": collect(["obs", "p_f1_0.5"]),
            "obs_s_f1_0.5": collect(["obs", "s_f1_0.5"]),
        }

    (args.root / "pair_seeds_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Phase 2.1 pair × 3 seeds (Table VIII style)", "", "| Arm | Metric | mean±std |", "|---|---|---|"]
    for arm, metrics in report["table_viii_style"].items():
        for name, (mu, sd) in metrics.items():
            lines.append(f"| {arm} | {name} | {mu:.4f}±{sd:.4f} |")
    # deltas huygens - free
    lines.append("")
    lines.append("## Δ (huygens − free) mean")
    for name in report["table_viii_style"]["huygens"]:
        dh = report["table_viii_style"]["huygens"][name][0] - report["table_viii_style"]["free"][name][0]
        lines.append(f"- {name}: {dh:+.4f}")
    (args.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"[done] → {args.root}", flush=True)


if __name__ == "__main__":
    main()
