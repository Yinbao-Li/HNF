#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run22: stepwise obliquity χ ablation on STEAD (from run20).

Modes (see --obliquity-mode in train_stead_picking.py):
  soft_shared  — χ only on shared encoder (pick branches pure Huygens)
  det_soft     — duplicate det shared stack with soft χ
  det_fresnel  — det shared stack full Fresnel
  noise_soft   — χ only in noise-cancel propagation kernel

Each variant: short 2-epoch sharp pass, then eval vs EQT on 2k-event subset.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "run22"
BASE = ROOT / "outputs" / "run20" / "20_wrongpeak_sharp" / "best.pt"
EQT_REF = {"det_f1": 0.9992, "p_f1": 0.9894, "s_f1": 0.9707}

COMMON = [
    sys.executable,
    str(ROOT / "tools" / "tools/train_stead_picking.py"),
    "--seq-len", "800",
    "--batch-size", "8",
    "--grad-accum-steps", "6",
    "--num-workers", "1",
    "--embed-dim", "64",
    "--num-shared-layers", "2",
    "--num-branch-layers", "2",
    "--local-window-sec", "15.0",
    "--seed", "42",
    "--pick-head-hidden", "48",
    "--pick-head-layers", "4",
    "--pick-head-kernel", "7",
    "--score-mode", "det_guard",
    "--det-score-floor", "0.988",
    "--post-process-p-before-s",
    "--no-residual-det-head",
    "--enhanced-det-head",
    "--noise-cancel",
    "--noise-source-dim", "16",
    "--noise-det-pick-split",
    "--noise-pick-cues",
    "--reset-best-score",
    "--resume", str(BASE),
    "--epochs", "2",
    "--lr", "3e-5",
    "--label-sigma-sec", "0.35",
    "--pick-pos-weight", "28",
    "--pick-loss-weight", "2.8",
    "--p-pick-loss-weight", "1.3",
    "--s-pick-loss-weight", "1.6",
    "--det-loss-weight", "1.0",
    "--det-event-weight", "2.0",
    "--ps-order-loss-weight", "0.12",
    "--wrong-peak-loss-weight", "0.15",
    "--wrong-peak-radius-sec", "0.45",
    "--wrong-peak-margin", "0.25",
    "--s-wrong-peak-scale", "1.35",
    "--noise-cancel-weight", "0.05",
]

VARIANTS = [
    ("22_soft_shared_02", "soft_shared", "0.25"),
    ("22_soft_shared_03", "soft_shared", "0.30"),
    ("22_det_soft_025", "det_soft", "0.25"),
    ("22_det_fresnel", "det_fresnel", "1.0"),
    ("22_noise_soft_025", "noise_soft", "0.25"),
]


def train_one(name: str, mode: str, mix: str, dry: bool) -> Path:
    out = OUT / name
    cmd = COMMON + [
        "--output-dir", str(out),
        "--obliquity-mode", mode,
        "--obliquity-mix", mix,
    ]
    print(f"[run22] train {name} mode={mode} mix={mix}", flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    return out / "best.pt"


def eval_subset(ckpt: Path) -> dict:
    out_dir = OUT / "eval" / ckpt.parent.name
    report_path = out_dir / "stead_baseline_compare_report.json"
    cmd = [
        sys.executable,
        str(ROOT / "run_paper_stead_baseline_compare.py"),
        "--checkpoint", str(ckpt),
        "--output-dir", str(out_dir),
        "--max-events", "2000",
        "--max-noise", "500",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr[-500:]}
    if not report_path.is_file():
        return {"error": "missing report", "stdout_tail": proc.stdout[-300:]}
    return json.loads(report_path.read_text())


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default=None)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-completed", action="store_true", help="Skip train if best.pt exists")
    args = p.parse_args()

    if not BASE.is_file():
        raise SystemExit(f"Missing base ckpt: {BASE}")

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "obliquity_ablation_report.json"
    if report_path.is_file():
        results = json.loads(report_path.read_text())
        results.setdefault("variants", {})
    else:
        results = {
            "eqt_ref": EQT_REF,
            "run20_ref": json.loads(
                (ROOT / "outputs/run20/20_wrongpeak_sharp/test_metrics.json").read_text()
            ),
            "variants": {},
        }

    # Baseline eval only (once)
    if args.only in (None, "run20") and "run20_baseline" not in results.get("variants", {}):
        print("[run22] eval run20 baseline...", flush=True)
        if not args.dry_run:
            results["variants"]["run20_baseline"] = eval_subset(BASE)
            report_path.write_text(json.dumps(results, indent=2))

    variants = VARIANTS
    if args.only:
        variants = [v for v in VARIANTS if v[0] == args.only]
        if not variants and args.only != "run20":
            raise SystemExit(f"Unknown variant: {args.only}")

    for name, mode, mix in variants:
        ckpt = OUT / name / "best.pt"
        if args.skip_train or (args.skip_completed and ckpt.is_file()):
            if not ckpt.is_file():
                print(f"[run22] skip missing {name}", flush=True)
                continue
            print(f"[run22] reuse {name}", flush=True)
        else:
            ckpt = train_one(name, mode, mix, args.dry_run)
        if args.dry_run:
            continue
        if not ckpt.is_file():
            results["variants"][name] = {"error": "no checkpoint"}
            report_path.write_text(json.dumps(results, indent=2))
            continue
        if name in results["variants"] and "results" in results["variants"][name]:
            print(f"[run22] already eval {name}", flush=True)
            continue
        print(f"[run22] eval {name}...", flush=True)
        ev = eval_subset(ckpt)
        results["variants"][name] = ev
        report_path.write_text(json.dumps(results, indent=2))

    # Rank by P+S vs EQT
    ranked = []
    for k, v in results["variants"].items():
        if "error" in v:
            continue
        coupled = v.get("results", {})
        hnf = coupled.get("HNF(run20)", {})
        if not hnf and coupled:
            hnf = next(iter(coupled.values()))
        m = hnf.get("coupled", hnf) if isinstance(hnf, dict) else {}
        if not m:
            continue
        ranked.append((k, m.get("p_f1", 0) + m.get("s_f1", 0), m))
    ranked.sort(key=lambda x: x[1], reverse=True)
    results["ranking_ps_sum"] = [
        {"name": k, "p_f1": m.get("p_f1"), "s_f1": m.get("s_f1"), "det_f1": m.get("det_f1"), "ps_sum": s}
        for k, s, m in ranked
    ]

    report = OUT / "obliquity_ablation_report.json"
    report.write_text(json.dumps(results, indent=2))
    print(json.dumps({"report": str(report), "ranking": results.get("ranking_ps_sum", [])}, indent=2))


if __name__ == "__main__":
    main()
