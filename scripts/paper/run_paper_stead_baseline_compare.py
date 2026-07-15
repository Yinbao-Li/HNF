#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEAD in-domain baseline: HNF(run20) vs EQT(STEAD) vs PhaseNet(STEAD).

Shows EQT/PhaseNet are also strong on STEAD (like HNF's ~0.99 det), so OBS
low scores are cross-domain transfer — not weak models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analyze_stead_picking import load_model
from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset
from run_paper_obs_picking_compare import _pick_only_counts, normalize_wave
from train_stead_picking import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STEAD in-domain HNF vs EQT vs PhaseNet")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/paper_stead_baseline_compare")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-events", type=int, default=2000)
    p.add_argument("--max-noise", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def subsample_indices(ds: STEADPickingDataset, max_events: int, max_noise: int, seed: int):
    ev = [i for i, r in enumerate(ds.refs) if r.is_event == 1]
    nz = [i for i, r in enumerate(ds.refs) if r.is_event == 0]
    rng = np.random.default_rng(seed)
    if len(ev) > max_events:
        ev = sorted(rng.choice(ev, size=max_events, replace=False).tolist())
    if len(nz) > max_noise:
        nz = sorted(rng.choice(nz, size=max_noise, replace=False).tolist())
    return ev + nz


def _load_raw_stead(ds: STEADPickingDataset, idx: int) -> dict:
    ref = ds.refs[idx]
    waveform = ds._get_handle(ref.chunk)["data"][ref.trace_name][()]  # (6000, 3)
    wave = np.asarray(waveform, dtype=np.float32).T  # (3, 6000)
    return {
        "wave": wave,
        "is_event": ref.is_event == 1,
        "p_idx": int(ref.p_sample) if ref.p_sample is not None else 0,
        "s_idx": int(ref.s_sample) if ref.s_sample is not None else 0,
        "p_valid": ref.p_sample is not None,
        "s_valid": ref.s_sample is not None,
    }


@torch.no_grad()
def eval_seisbench_stead(
    model,
    ds: STEADPickingDataset,
    indices: list[int],
    device: torch.device,
    kind: str,
    pick_th: float,
    det_th: float,
    tol_sec: float,
    batch_size: int,
):
    acc = EvalAccumulator()
    pick_acc = EvalAccumulator()
    native_len = 6000
    tol = max(1, int(round(tol_sec * 100.0)))  # 100 Hz
    model = model.to(device).eval()
    labels = "".join(getattr(model, "labels", "PSN") or "PSN")

    for start in range(0, len(indices), batch_size):
        chunk_idx = indices[start : start + batch_size]
        samples = [_load_raw_stead(ds, i) for i in chunk_idx]
        waves = [normalize_wave(s["wave"], "peak") for s in samples]
        x = torch.from_numpy(np.stack(waves)).to(device)
        p_idx = torch.tensor([s["p_idx"] for s in samples], device=device)
        s_idx = torch.tensor([s["s_idx"] for s in samples], device=device)
        p_valid = torch.tensor([s["p_valid"] for s in samples], device=device)
        s_valid = torch.tensor([s["s_valid"] for s in samples], device=device)
        det_true = torch.tensor([s["is_event"] for s in samples], device=device)

        out = model(x)
        if kind == "eqt":
            det_prob, p_prob, s_prob = out
            if det_prob.dim() == 3:
                det_prob = det_prob.squeeze(1)
            if p_prob.dim() == 3:
                p_prob = p_prob.squeeze(1)
            if s_prob.dim() == 3:
                s_prob = s_prob.squeeze(1)
            det_pred = det_prob.amax(dim=-1) >= det_th
        else:
            if labels.upper().startswith("PS"):
                p_prob, s_prob = out[:, 0], out[:, 1]
            elif labels.upper().startswith("NP"):
                p_prob, s_prob = out[:, 1], out[:, 2]
            else:
                p_prob, s_prob = out[:, 0], out[:, 1]
            det_pred = torch.maximum(p_prob.amax(-1), s_prob.amax(-1)) >= pick_th

        p_prob, s_prob = apply_p_before_s_constraint(p_prob, s_prob, pick_th)
        update_detection_counts(acc, det_pred, det_true)
        update_picking_counts(acc.p, p_prob, det_pred, det_true, p_valid, p_idx, pick_th, tol, native_len)
        update_picking_counts(acc.s, s_prob, det_pred, det_true, s_valid, s_idx, pick_th, tol, native_len)

        force_det = torch.ones_like(det_pred)
        if det_true.any():
            m = det_true
            _pick_only_counts(p_prob[m], p_valid[m], p_idx[m], pick_th, tol, native_len, pick_acc.p)
            _pick_only_counts(s_prob[m], s_valid[m], s_idx[m], pick_th, tol, native_len, pick_acc.s)
            update_detection_counts(pick_acc, force_det[m], det_true[m])

    return {
        "coupled": finalize_metrics(acc),
        "pick_only_events": finalize_metrics(pick_acc),
        "norm_mode": "peak",
        "labels": labels,
    }


def load_sb(name: str, weights: str):
    import seisbench.models as sbm

    return getattr(sbm, name).from_pretrained(weights)


def plot_board(results: dict, official_hnf: dict, out_path: Path) -> None:
    names = [k for k in results if "error" not in results[k]]
    metrics = ["det_f1", "p_f1", "s_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.25

    ax = axes[0]
    x = np.arange(len(names))
    for i, m in enumerate(metrics):
        vals = [results[n]["coupled"][m] for n in names]
        ax.bar(x + (i - 1) * w, vals, width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("STEAD subsample (coupled)")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["HNF official\nfull test"] + [f"{n}\ncoupled" for n in names]
    d_vals = [official_hnf["det_f1"]] + [results[n]["coupled"]["det_f1"] for n in names]
    p_vals = [official_hnf["p_f1"]] + [results[n]["coupled"]["p_f1"] for n in names]
    s_vals = [official_hnf["s_f1"]] + [results[n]["coupled"]["s_f1"] for n in names]
    x = np.arange(len(labels))
    ax.bar(x - w, d_vals, width=w, label="det_f1")
    ax.bar(x, p_vals, width=w, label="p_f1")
    ax.bar(x + w, s_vals, width=w, label="s_f1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("In-domain reference")
    ax.legend(fontsize=8)

    fig.suptitle("STEAD in-domain: HNF vs EQT vs PhaseNet", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    official = json.loads(
        Path("outputs/run20/20_wrongpeak_sharp/test_metrics.json").read_text()
    )

    print("[stead-baseline] loading STEAD test refs...", flush=True)
    ds = STEADPickingDataset(
        "test",
        seq_len=800,
        load_geometry=False,
        max_event_traces=None,
        max_noise_traces=None,
    )
    indices = subsample_indices(ds, args.max_events, args.max_noise, args.seed)
    n_ev = sum(1 for i in indices if ds.refs[i].is_event == 1)
    n_nz = len(indices) - n_ev
    print(f"[stead-baseline] subset n={len(indices)} events={n_ev} noise={n_nz}", flush=True)

    results = {}

    print("[stead-baseline] HNF run20...", flush=True)
    model, cfg = load_model(Path(args.checkpoint), device)
    model.eval()
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False)
    hnf_m = evaluate(
        model,
        loader,
        device,
        seq_len=int(cfg.get("seq_len", 800)),
        pick_threshold=args.pick_threshold,
        pick_tolerance_sec=args.tol_sec,
        post_process_p_before_s=True,
    )
    keys = (
        "det_precision", "det_recall", "det_f1",
        "p_precision", "p_recall", "p_f1", "p_mae_sec",
        "s_precision", "s_recall", "s_f1", "s_mae_sec",
    )
    results["HNF(run20)"] = {
        "coupled": {k: hnf_m[k] for k in keys if k in hnf_m},
        "pick_only_events": {
            "p_f1": hnf_m["p_f1"],
            "s_f1": hnf_m["s_f1"],
            "det_f1": hnf_m["det_f1"],
            "note": "same coupled protocol as official test_metrics",
        },
    }
    print(results["HNF(run20)"]["coupled"], flush=True)

    for label, cls_name, weights, kind in [
        ("EQT(STEAD)", "EQTransformer", "stead", "eqt"),
        ("PhaseNet(STEAD)", "PhaseNet", "stead", "phasenet"),
    ]:
        print(f"[stead-baseline] {label}...", flush=True)
        try:
            sb = load_sb(cls_name, weights)
            results[label] = eval_seisbench_stead(
                sb,
                ds,
                indices,
                device,
                kind,
                args.pick_threshold,
                args.det_threshold,
                args.tol_sec,
                args.batch_size,
            )
            print(results[label]["coupled"], flush=True)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            print(results[label], flush=True)

    fig_path = out_dir / "stead_baseline_compare.png"
    plot_board(results, official, fig_path)
    fig_docs = Path("docs/figures/fig5_stead_baseline_compare.png")
    fig_docs.parent.mkdir(parents=True, exist_ok=True)
    plot_board(results, official, fig_docs)

    report = {
        "protocol": {
            "split": "STEAD EQTransformer test split",
            "subset": {"n_events": n_ev, "n_noise": n_nz, "seed": args.seed},
            "tol_sec": args.tol_sec,
            "pick_threshold": args.pick_threshold,
            "det_threshold": args.det_threshold,
            "hnf_norm": "per-channel demean+std + resample 800",
            "sb_norm": "peak (SeisBench STEAD default)",
            "note": "In-domain sanity check; OBS table is cross-domain.",
        },
        "hnf_official_full_test": {
            k: official[k] for k in ("det_f1", "p_f1", "s_f1", "p_mae_sec", "s_mae_sec")
        },
        "results": results,
        "figure": str(fig_path),
    }
    (out_dir / "stead_baseline_compare_report.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# STEAD in-domain baseline (HNF vs EQT vs PhaseNet)",
        "",
        f"Subset: {n_ev} events + {n_nz} noise (seed={args.seed}).",
        "",
        "## HNF official full test",
        f"- det_f1={official['det_f1']:.4f}  p_f1={official['p_f1']:.4f}  s_f1={official['s_f1']:.4f}",
        "",
        "## Subset coupled F1",
    ]
    for k, v in results.items():
        if "error" in v:
            lines.append(f"- {k}: ERROR {v['error']}")
        else:
            c = v["coupled"]
            lines.append(
                f"- {k}: det={c['det_f1']:.4f}  p={c['p_f1']:.4f}  s={c['s_f1']:.4f}"
            )
    lines += ["", f"Figure: `{fig_path}`"]
    (out_dir / "stead_baseline_compare_report.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "n_events": n_ev,
        "hnf_official": {k: round(official[k], 4) for k in ("det_f1", "p_f1", "s_f1")},
        "subset_coupled": {
            k: {m: round(v["coupled"][m], 4) for m in ("det_f1", "p_f1", "s_f1")}
            for k, v in results.items() if "coupled" in v
        },
    }, indent=2))


if __name__ == "__main__":
    main()
