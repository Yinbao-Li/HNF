#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval EQT-aug / PN-aug checkpoints at epochs 20/40/80/182 (Phase 2.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

_REPO = Path(__file__).resolve().parents[1]
for sub in ("", "scripts/paper"):
    if str(_REPO / sub) not in sys.path:
        sys.path.insert(0, str(_REPO / sub))

from tools.eval_ethz_cross import (  # noqa: E402
    GROUP_COL,
    P_COLS,
    S_COLS,
    T,
    cut_window,
    first_arrival,
    load_dataset,
)
from tools.eval_shift_invariance import load_sbc, run_sbc  # noqa: E402
from tools.sweep_seisbench_threshold import metrics_at  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/phase2/2_2_matched_budget"))
    p.add_argument("--epochs", default="20,40,80,182")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sets", default="ethz,obs")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def eval_one(ckpt: Path, device, dataset: str, batch_size: int) -> dict:
    kind_m, model = load_sbc(ckpt, device)
    data = load_dataset(dataset, "test")
    rng = np.random.default_rng(0)
    waves, y = [], {"p_idx": [], "s_idx": [], "p_valid": [], "s_valid": [], "is_event": []}
    n_dead = n_pad = 0
    for i in rng.permutation(len(data)):
        w, meta = data.get_sample(int(i))
        p = first_arrival(meta, P_COLS)
        if p is None or w.shape[0] != 3:
            continue
        w = np.nan_to_num(np.asarray(w, dtype=np.float32))
        if dataset == "obs" and not (w.std(axis=1) > 0).all():
            n_dead += 1
            continue
        s = first_arrival(meta, S_COLS)
        p = int(round(p))
        tp = int(round(rng.uniform(5.0, 30.0) * 100))
        win, left, right = cut_window(w, p, tp)
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
    sos = butter(4, [1.0, 45.0], btype="bandpass", fs=100.0, output="sos")
    waves = sosfiltfilt(sos, waves, axis=-1).astype(np.float32)
    y = {k: torch.tensor(v) for k, v in y.items()}
    det, p, s = run_sbc(model, kind_m, waves, device, batch_size)
    row = {
        "n_windows": len(waves),
        "n_with_s": int(y["s_valid"].sum()),
        "n_dead": n_dead,
        "n_pad": n_pad,
        "tolerances": {},
    }
    for tol in (0.1, 0.2, 0.5):
        m = metrics_at(det, p, s, y, kind_m, 0.3, 0.5, max(1, int(round(tol * 100))), device)
        row["tolerances"][f"{tol:g}"] = {k: m[k] for k in m if k.startswith(("p_", "s_", "det_"))}
        print(
            f"  [{dataset}] {ckpt.parent.name}/{ckpt.name} tol{tol:g} "
            f"P={row['tolerances'][f'{tol:g}']['p_f1']:.4f} "
            f"S={row['tolerances'][f'{tol:g}']['s_f1']:.4f}",
            flush=True,
        )
    out = ckpt.parent.parent / "eval" / f"{ckpt.parent.name}_{ckpt.stem}_{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"checkpoint": str(ckpt), "dataset": dataset, "kind": kind_m, **row}, indent=2)
    )
    return row


def resolve_ckpt(root: Path, model: str, ep: int) -> Path | None:
    ckpt = root / model / f"ep{ep:04d}.pt"
    if ckpt.exists():
        return ckpt
    last = root / model / "last.pt"
    if last.exists():
        e = int(torch.load(last, map_location="cpu").get("epoch", -1))
        if e == ep:
            return last
    return None


def main():
    args = parse_args()
    device = torch.device(args.device)
    epochs = [int(x) for x in args.epochs.split(",")]
    summary = {"protocol": "matched_budget_sb_aug", "epochs": epochs, "models": {}}
    for model in ("eqt", "phasenet"):
        summary["models"][model] = {}
        for ep in epochs:
            ckpt = resolve_ckpt(args.root, model, ep)
            if ckpt is None:
                summary["models"][model][f"ep{ep}"] = {"status": "MISSING"}
                print(f"[skip] missing {model} ep{ep}", flush=True)
                continue
            summary["models"][model][f"ep{ep}"] = {"checkpoint": str(ckpt), "sets": {}}
            for ds in args.sets.split(","):
                print(f"[eval] {model} ep{ep} {ds}", flush=True)
                summary["models"][model][f"ep{ep}"]["sets"][ds] = eval_one(
                    ckpt, device, ds, args.batch_size
                )
    eval_dir = args.root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Phase 2.2 matched-budget eval",
        "",
        "| Model | Ep | Set | P@0.1 | S@0.1 | P@0.5 | S@0.5 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for model, eps in summary["models"].items():
        for ep_key, block in eps.items():
            if block.get("status") == "MISSING":
                lines.append(f"| {model} | {ep_key} | — | MISSING | | | |")
                continue
            for ds, row in block.get("sets", {}).items():
                t = row["tolerances"]
                lines.append(
                    f"| {model} | {ep_key} | {ds} | {t['0.1']['p_f1']:.4f} | {t['0.1']['s_f1']:.4f} | "
                    f"{t['0.5']['p_f1']:.4f} | {t['0.5']['s_f1']:.4f} |"
                )
    (args.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"[done] → {args.root}", flush=True)


if __name__ == "__main__":
    main()
