#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBS zero-shot board: dense HNF + foveated HNF vs EQT/PhaseNet (STEAD).

Fairness: same holdout split as Step 4 (`obs_matched_adapt_split_randoffset`),
pick-only F1, 0.5 s tol, random p_offset∈[4,12]. Does NOT mix adapt rows into
the primary ZS table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    tolerance_bins,
)
from tools.analyze_stead_picking import load_model
from tools.obs_matched_split import load_split_samples
from tools.train_foveated import build_engine
from tools.train_stead_picking import set_seed

# Reuse OBS compare helpers (file is not a package module)
import importlib.util

_obs_path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
_spec = importlib.util.spec_from_file_location("obs_cmp", _obs_path)
obs_cmp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(obs_cmp)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS ZS: dense + foveated vs EQT/PN")
    p.add_argument("--stead-checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--obs-full-checkpoint", default="outputs/run28_obs_full_800/best.pt")
    p.add_argument(
        "--split-json",
        default="outputs/obs_matched_adapt_split_randoffset/split.json",
    )
    p.add_argument("--output-dir", default="outputs/foveated/obs_zs_board")
    p.add_argument("--max-gazes", type=int, default=8)
    p.add_argument("--foveated-seq-len", type=int, default=6000)
    p.add_argument("--dense-seq-len", type=int, default=800)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--skip-obs-full", action="store_true")
    p.add_argument("--skip-seisbench", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def _pick_row(m: dict) -> dict:
    po = m.get("pick_only", m)
    return {
        "p_f1": float(po["p_f1"]),
        "s_f1": float(po["s_f1"]),
        "p_mae_sec": float(po.get("p_mae_sec", 0.0)),
        "s_mae_sec": float(po.get("s_mae_sec", 0.0)),
        "n_gazes_mean": float(m.get("n_gazes_mean", float("nan"))),
        "sec_per_trace": float(m.get("sec_per_trace", float("nan"))),
        "coverage_mean": float(m.get("coverage_mean", float("nan"))),
    }


@torch.no_grad()
def eval_foveated_obs(
    engine,
    samples: list[dict],
    *,
    seq_len: int,
    window_sec: float,
    max_gazes: int,
    pick_th: float,
    tol_sec: float,
    device: torch.device,
) -> dict:
    engine.eval()
    pick_acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, tol_sec)
    t0 = time.time()
    n_gazes = 0.0
    cover = 0.0
    n = 0
    for s in samples:
        x, t, p_idx, s_idx, p_valid, s_valid = obs_cmp.to_hnf_batch(
            [s], seq_len, window_sec, device
        )
        # Engine wants (B,3,T)
        wave = x.transpose(1, 2).contiguous()
        out = engine(wave, max_gazes=max_gazes)
        p_prob, s_prob = apply_p_before_s_constraint(out.p_prob, out.s_prob, pick_th)
        obs_cmp._pick_only_counts(p_prob, p_valid, p_idx, pick_th, tol, seq_len, pick_acc.p)
        obs_cmp._pick_only_counts(s_prob, s_valid, s_idx, pick_th, tol, seq_len, pick_acc.s)
        n_gazes += float(out.n_gazes.float().sum().item())
        if out.coverage is not None:
            cover += float(out.coverage.mean().item())
        n += 1
    m = finalize_metrics(pick_acc)
    return {
        "pick_only": m,
        "n_gazes_mean": n_gazes / max(n, 1),
        "coverage_mean": cover / max(n, 1),
        "sec_per_trace": (time.time() - t0) / max(n, 1),
        "n": n,
    }


def write_figure(rows: list[tuple[str, dict]], out_path: Path) -> None:
    labels = [r[0] for r in rows]
    p_f1 = [r[1]["p_f1"] for r in rows]
    s_f1 = [r[1]["s_f1"] for r in rows]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, p_f1, w, label="P-F1", color="#3d7ea6")
    ax.bar(x + w / 2, s_f1, w, label="S-F1", color="#c47a3a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pick-only F1")
    ax.set_title("OBS holdout zero-shot (random p_offset)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    docs = _REPO_ROOT / "docs" / "figures" / "foveated_obs_zs_board.png"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_bytes(out_path.read_bytes())


def write_md(report: dict, path: Path) -> None:
    lines = [
        "# OBS Zero-Shot Board: Dense / Foveated / EQT / PhaseNet",
        "",
        f"- holdout n: **{report['n']}** (with S: {report['n_with_s']})",
        f"- split: `{report['split_json']}`",
        f"- protocol: random p_offset, pick-only F1, tol={report['tol_sec']} s",
        f"- primary table = **zero-shot only** (no adapt mix)",
        "",
        "## A. Zero-shot (train≠OBS → eval=OBS)",
        "",
        "| Model | P-F1 | S-F1 | P-MAE | S-MAE | gazes | sec/trace |",
        "|-------|-----:|-----:|------:|------:|------:|----------:|",
    ]
    for name, row in report["zs_rows"]:
        g = f"{row['n_gazes_mean']:.2f}" if np.isfinite(row["n_gazes_mean"]) else "—"
        t = f"{row['sec_per_trace']:.3f}" if np.isfinite(row["sec_per_trace"]) else "—"
        lines.append(
            f"| `{name}` | {row['p_f1']:.3f} | {row['s_f1']:.3f} | "
            f"{row['p_mae_sec']:.3f} | {row['s_mae_sec']:.3f} | {g} | {t} |"
        )
    if report.get("ref_rows"):
        lines += [
            "",
            "## B. OBS-exposed reference (not same budget)",
            "",
            "| Model | P-F1 | S-F1 | note |",
            "|-------|-----:|-----:|------|",
        ]
        for name, row, note in report["ref_rows"]:
            lines.append(f"| `{name}` | {row['p_f1']:.3f} | {row['s_f1']:.3f} | {note} |")
    lines += [
        "",
        f"- figure: `docs/figures/foveated_obs_zs_board.png`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("[obs-fov] loading holdout …", flush=True)
    samples, info, meta = load_split_samples(args.split_json, "holdout")
    n_with_s = sum(1 for s in samples if s["s_valid"])
    print(f"  n={len(samples)} with_s={n_with_s}", flush=True)

    results: dict[str, dict] = {}

    # Dense STEAD ZS
    print("[obs-fov] dense HNF(run28/STEAD) @800 …", flush=True)
    model_stead, _ = load_model(Path(args.stead_checkpoint), device)
    t0 = time.time()
    dense_zs = obs_cmp.eval_hnf(
        model_stead, samples, device, args.dense_seq_len, args.window_sec,
        args.pick_threshold, 0.5, args.tol_sec, batch_size=8,
    )
    dense_zs["sec_per_trace"] = (time.time() - t0) / max(len(samples), 1)
    results["HNF(run28/STEAD)-dense"] = dense_zs
    print(
        f"  P={dense_zs['pick_only']['p_f1']:.3f} S={dense_zs['pick_only']['s_f1']:.3f}",
        flush=True,
    )

    # Foveated STEAD ZS
    print(f"[obs-fov] foveated HNF(run28/STEAD) @{args.foveated_seq_len} …", flush=True)

    class A:
        pass

    a = A()
    a.checkpoint = args.stead_checkpoint
    a.seq_len = args.foveated_seq_len
    a.max_gazes = args.max_gazes
    a.freeze_backbone = True
    a.unfreeze_backbone = False
    a.scanner = "energy"
    engine = build_engine(a, device)
    engine.coverage_complete_ratio = 0.85
    fov = eval_foveated_obs(
        engine, samples,
        seq_len=args.foveated_seq_len, window_sec=args.window_sec,
        max_gazes=args.max_gazes, pick_th=args.pick_threshold,
        tol_sec=args.tol_sec, device=device,
    )
    results["HNF(run28/STEAD)-foveated"] = fov
    print(
        f"  P={fov['pick_only']['p_f1']:.3f} S={fov['pick_only']['s_f1']:.3f} "
        f"gazes={fov['n_gazes_mean']:.2f}",
        flush=True,
    )

    # EQT / PhaseNet STEAD ZS
    if not args.skip_seisbench:
        for name, weights, kind, n_ch, norm in [
            ("EQT(STEAD)", "stead", "eqt", 3, "peak"),
            ("PhaseNet(STEAD)", "stead", "phasenet", 3, "peak"),
        ]:
            print(f"[obs-fov] {name} …", flush=True)
            try:
                model_name = "EQTransformer" if kind == "eqt" else "PhaseNet"
                sb = obs_cmp.load_sb_model(model_name, weights)
                sb = sb.to(device)
                t0 = time.time()
                m = obs_cmp.eval_seisbench(
                    sb, samples, device, args.pick_threshold, 0.5, args.tol_sec,
                    batch_size=8, kind=kind, n_channels=n_ch, norm_mode=norm,
                )
                m["sec_per_trace"] = (time.time() - t0) / max(len(samples), 1)
                results[name] = m
                print(
                    f"  P={m['pick_only']['p_f1']:.3f} S={m['pick_only']['s_f1']:.3f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  SKIP {name}: {e}", flush=True)

    # OBS-full dense reference
    ref_rows = []
    if not args.skip_obs_full and Path(args.obs_full_checkpoint).exists():
        print("[obs-fov] dense HNF(run28/OBS-full) reference …", flush=True)
        model_obs, _ = load_model(Path(args.obs_full_checkpoint), device)
        t0 = time.time()
        obs_full = obs_cmp.eval_hnf(
            model_obs, samples, device, args.dense_seq_len, args.window_sec,
            args.pick_threshold, 0.5, args.tol_sec, batch_size=8,
        )
        obs_full["sec_per_trace"] = (time.time() - t0) / max(len(samples), 1)
        results["HNF(run28/OBS-full)-dense"] = obs_full
        ref_rows.append((
            "HNF(run28/OBS-full)-dense",
            _pick_row(obs_full),
            "OBS-full retrain reference (not ZS)",
        ))
        print(
            f"  P={obs_full['pick_only']['p_f1']:.3f} S={obs_full['pick_only']['s_f1']:.3f}",
            flush=True,
        )

    zs_order = [
        "HNF(run28/STEAD)-dense",
        "HNF(run28/STEAD)-foveated",
        "EQT(STEAD)",
        "PhaseNet(STEAD)",
    ]
    zs_rows = [(k, _pick_row(results[k])) for k in zs_order if k in results]

    fig_rows = list(zs_rows)
    for name, row, _ in ref_rows:
        fig_rows.append((name, row))
    write_figure(fig_rows, out / "foveated_obs_zs_board.png")

    report = {
        "split_json": args.split_json,
        "n": len(samples),
        "n_with_s": n_with_s,
        "tol_sec": args.tol_sec,
        "pick_threshold": args.pick_threshold,
        "max_gazes": args.max_gazes,
        "stead_checkpoint": args.stead_checkpoint,
        "results_raw": {k: {
            "pick_only": v.get("pick_only", v),
            "n_gazes_mean": v.get("n_gazes_mean"),
            "sec_per_trace": v.get("sec_per_trace"),
            "coverage_mean": v.get("coverage_mean"),
        } for k, v in results.items()},
        "zs_rows": zs_rows,
        "ref_rows": ref_rows,
        "load_info": info,
    }
    # JSON-safe
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(v) for v in o]
        if isinstance(o, tuple):
            return [_san(v) for v in o]
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    (out / "obs_zs_board.json").write_text(json.dumps(_san(report), indent=2))
    write_md(report, out / "obs_zs_board.md")
    print(open(out / "obs_zs_board.md").read(), flush=True)


if __name__ == "__main__":
    main()
