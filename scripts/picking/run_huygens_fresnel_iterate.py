#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Huygens–Fresnel iteration: replay the two locked conclusions.

1) Picking: short sharp pass from run19, same recipe as run20, with --principle huygens_fresnel
2) Inversion: freeze fresnel picking ckpt, train macro head, Route A2 + STEAD geom (proof subset)

Usage:
  python run_huygens_fresnel_iterate.py --stage all --device cuda
  python run_huygens_fresnel_iterate.py --stage pick
  python run_huygens_fresnel_iterate.py --stage invert
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "huygens_fresnel"
PICK_DIR = OUT / "picking_sharp"
INV_DIR = OUT / "zhizi_macro"
PROOF_DIR = OUT / "proof_suite"
BASE_RESUME = ROOT / "outputs" / "run19" / "19_detpick_split" / "best.pt"
RUN20_REF = {
    "det_f1": 0.9940792470016699,
    "p_f1": 0.9593805429066354,
    "s_f1": 0.9492358762370036,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huygens–Fresnel iterate vs locked baselines")
    p.add_argument("--stage", choices=["pick", "invert", "all"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pick-epochs", type=int, default=4)
    p.add_argument("--inv-epochs", type=int, default=8)
    p.add_argument("--n-synth", type=int, default=32)
    p.add_argument("--max-events", type=int, default=48)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(cmd: list[str], dry: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry:
        return
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def stage_pick(args: argparse.Namespace) -> Path:
    PICK_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_RESUME.is_file():
        raise FileNotFoundError(f"Missing run19 resume: {BASE_RESUME}")
    cmd = [
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
        "--principle", "huygens_fresnel",
        "--obliquity-scale", "1.0",
        "--resume", str(BASE_RESUME),
        "--epochs", str(args.pick_epochs),
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
        "--device", args.device,
        "--output-dir", str(PICK_DIR),
    ]
    run(cmd, args.dry_run)
    best = PICK_DIR / "best.pt"
    metrics = {}
    test_path = PICK_DIR / "test_metrics.json"
    if test_path.is_file():
        metrics = json.loads(test_path.read_text())
    # prefer state-like summary from history if present
    hist = PICK_DIR / "history.csv"
    report = {
        "fresnel_checkpoint": str(best),
        "run20_ref": RUN20_REF,
        "fresnel_test": metrics,
        "delta_vs_run20": None,
    }
    if metrics:
        def g(d, *keys):
            for k in keys:
                if k in d:
                    return float(d[k])
            return None
        f = {
            "det_f1": g(metrics, "det_f1", "f1_det"),
            "p_f1": g(metrics, "p_f1", "f1_p"),
            "s_f1": g(metrics, "s_f1", "f1_s"),
        }
        # also try nested
        if f["det_f1"] is None and "det" in metrics:
            f["det_f1"] = metrics["det"].get("f1") if isinstance(metrics["det"], dict) else None
        report["fresnel_summary"] = f
        if all(v is not None for v in f.values()):
            report["delta_vs_run20"] = {
                "det_f1": f["det_f1"] - RUN20_REF["det_f1"],
                "p_f1": f["p_f1"] - RUN20_REF["p_f1"],
                "s_f1": f["s_f1"] - RUN20_REF["s_f1"],
            }
    (OUT / "pick_compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return best


def stage_invert(args: argparse.Namespace, pick_ckpt: Path) -> None:
    INV_DIR.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable,
        str(ROOT / "tools" / "train_zhizi_inversion.py"),
        "--checkpoint", str(pick_ckpt),
        "--head-mode", "macro",
        "--epochs", str(args.inv_epochs),
        "--n-train", "96",
        "--n-val", "16",
        "--unrolled-weight", "0.5",
        "--unrolled-steps", "5",
        "--vp-sup-weight", "0.05",
        "--lr", "3e-3",
        "--device", args.device,
        "--output-dir", str(INV_DIR),
    ]
    run(train_cmd, args.dry_run)
    head = INV_DIR / "best_physics_head.pt"
    a2 = OUT / "route_a2_32"
    a2_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "inversion" / "run_route_a2_waveform.py"),
        "--checkpoint", str(pick_ckpt),
        "--head-mode", "macro",
        "--physics-head", str(head),
        "--n-test", str(args.n_synth),
        "--fwi-steps", "60",
        "--device", args.device,
        "--output-dir", str(a2),
    ]
    run(a2_cmd, args.dry_run)
    proof_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "inversion" / "run_proof_suite.py"),
        "--checkpoint", str(pick_ckpt),
        "--physics-head", str(head),
        "--head-mode", "macro",
        "--device", args.device,
        "--max-events", str(args.max_events),
        "--n-synth", str(args.n_synth),
        "--output-dir", str(PROOF_DIR),
    ]
    run(proof_cmd, args.dry_run)

    base_a2 = ROOT / "outputs" / "route_a2_waveform_macro_32" / "report.json"
    base_proof = ROOT / "outputs" / "proof_suite" / "proof_report.json"
    fres_a2 = a2 / "report.json"
    fres_proof = PROOF_DIR / "proof_report.json"
    compare = {
        "pick_ckpt": str(pick_ckpt),
        "physics_head": str(head),
        "baseline_route_a2": json.loads(base_a2.read_text()) if base_a2.is_file() else None,
        "fresnel_route_a2": json.loads(fres_a2.read_text()) if fres_a2.is_file() else None,
        "baseline_proof": json.loads(base_proof.read_text()) if base_proof.is_file() else None,
        "fresnel_proof": json.loads(fres_proof.read_text()) if fres_proof.is_file() else None,
    }
    (OUT / "invert_compare.json").write_text(json.dumps(compare, indent=2))
    print(json.dumps({
        "baseline_a2": compare["baseline_route_a2"],
        "fresnel_a2": compare["fresnel_route_a2"],
        "baseline_stead": (compare["baseline_proof"] or {}).get("stead_summary"),
        "fresnel_stead": (compare["fresnel_proof"] or {}).get("stead_summary"),
        "baseline_synth": (compare["baseline_proof"] or {}).get("synth_summary"),
        "fresnel_synth": (compare["fresnel_proof"] or {}).get("synth_summary"),
    }, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pick_ckpt = PICK_DIR / "best.pt"
    if args.stage in {"pick", "all"}:
        pick_ckpt = stage_pick(args)
    if args.stage in {"invert", "all"}:
        if not pick_ckpt.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Need picking checkpoint first: {pick_ckpt}")
        stage_invert(args, pick_ckpt)


if __name__ == "__main__":
    main()
