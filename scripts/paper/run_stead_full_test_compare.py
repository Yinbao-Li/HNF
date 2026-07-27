#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full STEAD test compare: HNF (official metrics) + EQT + PhaseNet."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PAPER = _REPO_ROOT / "scripts" / "paper"
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from run_paper_stead_triple_compare import eval_seisbench, load_sb  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hnf-metrics",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/test_metrics.json",
    )
    p.add_argument("--output-dir", default="outputs/paper_stead_full_test_compare")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument(
        "--models",
        default="eqt,phasenet",
        help="Comma list among: eqt,phasenet (HNF always included from metrics file)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[full-test] device={device}", flush=True)

    ds = STEADPickingDataset("test", seq_len=800)
    indices = list(range(len(ds)))
    n_ev = sum(1 for r in ds.refs if r.is_event == 1)
    n_nz = len(ds) - n_ev
    print(
        f"[full-test] n={len(ds)} events={n_ev} noise={n_nz} tol={args.tol_sec}s",
        flush=True,
    )

    hnf = json.loads(Path(args.hnf_metrics).read_text())
    results = {
        "HNF(run28)": {
            "source": args.hnf_metrics,
            "coupled": {
                k: hnf[k]
                for k in (
                    "det_f1",
                    "p_f1",
                    "s_f1",
                    "p_mae_sec",
                    "s_mae_sec",
                    "p_precision",
                    "p_recall",
                    "s_precision",
                    "s_recall",
                    "det_precision",
                    "det_recall",
                )
                if k in hnf
            },
        }
    }

    model_map = {
        "eqt": ("EQT(STEAD)", "EQTransformer", "stead", "eqt"),
        "phasenet": ("PhaseNet(STEAD)", "PhaseNet", "stead", "phasenet"),
    }
    wanted = [x.strip().lower() for x in args.models.split(",") if x.strip()]

    for key in wanted:
        if key not in model_map:
            raise ValueError(f"unknown model {key}")
        label, cls_name, weights, kind = model_map[key]
        print(f"[full-test] evaluating {label} ...", flush=True)
        t0 = time.time()
        try:
            sb = load_sb(cls_name, weights)
            res = eval_seisbench(
                sb,
                ds,
                indices,
                device,
                kind,
                pick_th=args.pick_threshold,
                det_th=args.det_threshold,
                tol_sec=args.tol_sec,
                batch_size=args.batch_size,
            )
            elapsed = time.time() - t0
            res["elapsed_sec"] = elapsed
            results[label] = res
            c = res["coupled"]
            print(
                f"[full-test] {label} done in {elapsed:.0f}s | "
                f"det={c['det_f1']:.4f} P={c['p_f1']:.4f} S={c['s_f1']:.4f} "
                f"MAE P/S={c['p_mae_sec']:.4f}/{c['s_mae_sec']:.4f}",
                flush=True,
            )
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[full-test] {label} ERROR: {e}", flush=True)
            import traceback

            traceback.print_exc()

        # partial checkpoint (drop large residual lists for size)
        slim = {}
        for k, v in results.items():
            if not isinstance(v, dict):
                slim[k] = v
                continue
            slim[k] = {kk: vv for kk, vv in v.items() if kk != "residuals"}
            if "residuals" in v:
                slim[k]["residuals"] = {
                    ph: {sk: sv for sk, sv in v["residuals"][ph].items()}
                    for ph in ("p", "s")
                    if ph in v["residuals"]
                }
        report = {
            "protocol": {
                "split": "STEAD EQTransformer full test",
                "n": len(ds),
                "n_events": n_ev,
                "n_noise": n_nz,
                "tol_sec": args.tol_sec,
                "pick_threshold": args.pick_threshold,
                "det_threshold": args.det_threshold,
                "hnf_source": args.hnf_metrics,
                "sb_input": "native 6000 @ 100Hz, peak-norm",
            },
            "results": slim,
        }
        (out / "stead_full_test_compare.json").write_text(json.dumps(report, indent=2))

    lines = [
        f"# STEAD full-test compare (tol={args.tol_sec}s)",
        "",
        f"- n={len(ds)} (events={n_ev}, noise={n_nz})",
        f"- tol={args.tol_sec}s  pick_th={args.pick_threshold}  det_th={args.det_threshold}",
        "",
        "| Model | det F1 | P F1 | S F1 | P MAE | S MAE |",
        "|-------|-------:|-----:|-----:|------:|------:|",
    ]
    for name in ["HNF(run28)", "EQT(STEAD)", "PhaseNet(STEAD)"]:
        d = results.get(name, {})
        if not d:
            continue
        if "error" in d:
            lines.append(f"| {name} | ERROR | | | | |")
            continue
        c = d.get("coupled") or {}
        lines.append(
            f"| {name} | {c.get('det_f1', float('nan')):.4f} | "
            f"{c.get('p_f1', float('nan')):.4f} | {c.get('s_f1', float('nan')):.4f} | "
            f"{c.get('p_mae_sec', float('nan')):.4f} | {c.get('s_mae_sec', float('nan')):.4f} |"
        )
    md = "\n".join(lines)
    (out / "stead_full_test_compare.md").write_text(md)
    print(md, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
