#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-metric STEAD in-domain compare: HNF vs EQT vs PhaseNet.

Reports F1, MAE, MAD, residual σ, and inference throughput on one shared subset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PAPER = _REPO_ROOT / "scripts" / "paper"
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from hnf.picking_metrics import (  # noqa: E402
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from run_paper_obs_picking_compare import _pick_only_counts, normalize_wave  # noqa: E402
from tools.analyze_stead_picking import load_model  # noqa: E402
from tools.train_stead_picking import evaluate, move_batch_to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hnf-checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/paper_stead_triple_compare_50ep",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-events", type=int, default=10000)
    p.add_argument("--max-noise", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--speed-warmup", type=int, default=20)
    p.add_argument("--speed-batches", type=int, default=80)
    return p.parse_args()


def subsample_indices(ds, max_events: int, max_noise: int, seed: int):
    ev = [i for i, r in enumerate(ds.refs) if r.is_event == 1]
    nz = [i for i, r in enumerate(ds.refs) if r.is_event == 0]
    rng = np.random.default_rng(seed)
    if len(ev) > max_events:
        ev = sorted(rng.choice(ev, size=max_events, replace=False).tolist())
    if len(nz) > max_noise:
        nz = sorted(rng.choice(nz, size=max_noise, replace=False).tolist())
    return ev + nz


def residual_stats(residuals: list[float]) -> dict:
    if not residuals:
        return {
            "n": 0,
            "mae_sec": None,
            "mad_sec": None,
            "sigma_sec": None,
            "median_sec": None,
            "p90_abs_sec": None,
        }
    r = np.asarray(residuals, dtype=np.float64)
    med = float(np.median(r))
    return {
        "n": int(r.size),
        "mae_sec": float(np.mean(np.abs(r))),
        "mad_sec": float(np.median(np.abs(r - med))),  # classic MAD around median
        "mad_abs_sec": float(np.median(np.abs(r))),  # median |error|
        "sigma_sec": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
        "median_sec": med,
        "mean_sec": float(np.mean(r)),
        "p90_abs_sec": float(np.percentile(np.abs(r), 90)),
    }


def _load_raw_stead(ds, idx: int) -> dict:
    ref = ds.refs[idx]
    waveform = ds._get_handle(ref.chunk)["data"][ref.trace_name][()]
    wave = np.asarray(waveform, dtype=np.float32).T
    return {
        "wave": wave,
        "is_event": ref.is_event == 1,
        "p_idx": int(ref.p_sample) if ref.p_sample is not None else 0,
        "s_idx": int(ref.s_sample) if ref.s_sample is not None else 0,
        "p_valid": ref.p_sample is not None,
        "s_valid": ref.s_sample is not None,
    }


def collect_pick_residuals(
    probs: torch.Tensor,
    valid: torch.Tensor,
    gt_idx: torch.Tensor,
    pick_th: float,
    tol: int,
    seq_len: int,
    native_hz: float = 100.0,
) -> list[float]:
    """Residuals in seconds for peak>=th and within tol (pick-only style)."""
    del native_hz  # both pred/gt live on the same seq_len grid
    out = []
    for i in range(probs.size(0)):
        if not bool(valid[i].item()):
            continue
        pk = float(probs[i].max().item())
        if pk < pick_th:
            continue
        pred = int(probs[i].argmax().item())
        gt = int(gt_idx[i].item())
        if abs(pred - gt) > tol:
            continue
        pred_sec = pred * 60.0 / seq_len
        gt_sec = gt * 60.0 / seq_len
        out.append(pred_sec - gt_sec)
    return out


@torch.no_grad()
def eval_seisbench(
    model,
    ds,
    indices,
    device,
    kind: str,
    pick_th: float,
    det_th: float,
    tol_sec: float,
    batch_size: int,
):
    acc = EvalAccumulator()
    pick_acc = EvalAccumulator()
    native_len = 6000
    native_hz = 100.0
    tol = max(1, int(round(tol_sec * native_hz)))
    model = model.to(device).eval()
    labels = "".join(getattr(model, "labels", "PSN") or "PSN")
    p_res, s_res = [], []

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
            p_res += collect_pick_residuals(p_prob[m], p_valid[m], p_idx[m], pick_th, tol, native_len, native_hz)
            s_res += collect_pick_residuals(s_prob[m], s_valid[m], s_idx[m], pick_th, tol, native_len, native_hz)

    return {
        "coupled": finalize_metrics(acc),
        "pick_only_events": finalize_metrics(pick_acc),
        "residuals": {"p": residual_stats(p_res), "s": residual_stats(s_res)},
        "labels": labels,
        "seq_len": native_len,
    }


@torch.no_grad()
def eval_hnf_with_residuals(
    model,
    loader,
    device,
    seq_len: int,
    pick_th: float,
    tol_sec: float,
):
    from hnf.picking_metrics import det_pred_from_logits, tolerance_bins

    model.eval()
    acc = EvalAccumulator()
    p_res, s_res = [], []
    tol = tolerance_bins(seq_len, tol_sec)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        det_pred = det_pred_from_logits(out["det"])
        det_true = batch["det"] > 0.5
        update_detection_counts(acc, det_pred, det_true)
        p_probs = torch.sigmoid(out["p"])
        s_probs = torch.sigmoid(out["s"])
        p_probs, s_probs = apply_p_before_s_constraint(p_probs, s_probs, pick_th)
        update_picking_counts(
            acc.p, p_probs, det_pred, det_true, batch["p_valid"], batch["p_idx"], pick_th, tol, seq_len
        )
        update_picking_counts(
            acc.s, s_probs, det_pred, det_true, batch["s_valid"], batch["s_idx"], pick_th, tol, seq_len
        )
        # residuals on event windows with valid phase (pick-only style)
        m = det_true
        if m.any():
            p_res += collect_pick_residuals(
                p_probs[m], batch["p_valid"][m], batch["p_idx"][m], pick_th, tol, seq_len, 100.0
            )
            s_res += collect_pick_residuals(
                s_probs[m], batch["s_valid"][m], batch["s_idx"][m], pick_th, tol, seq_len, 100.0
            )
    metrics = finalize_metrics(acc)
    return {
        "coupled": metrics,
        "pick_only_events": metrics,  # same evaluate protocol as training test
        "residuals": {"p": residual_stats(p_res), "s": residual_stats(s_res)},
        "seq_len": seq_len,
    }


@torch.no_grad()
def bench_hnf(model, loader, device, warmup: int, n_batches: int) -> dict:
    model.eval()
    it = iter(loader)
    batches = []
    for _ in range(warmup + n_batches):
        try:
            batches.append(move_batch_to_device(next(it), device))
        except StopIteration:
            it = iter(loader)
            batches.append(move_batch_to_device(next(it), device))
    for b in batches[:warmup]:
        model(b["x"], b["t"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for b in batches[warmup : warmup + n_batches]:
        model(b["x"], b["t"])
        n += int(b["x"].size(0))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"samples": n, "seconds": dt, "samples_per_sec": n / max(dt, 1e-9)}


@torch.no_grad()
def bench_sb(model, ds, indices, device, kind: str, batch_size: int, warmup: int, n_batches: int) -> dict:
    model = model.to(device).eval()
    # prebuild a few batches
    built = []
    for start in range(0, min(len(indices), (warmup + n_batches) * batch_size), batch_size):
        chunk = indices[start : start + batch_size]
        samples = [_load_raw_stead(ds, i) for i in chunk]
        waves = [normalize_wave(s["wave"], "peak") for s in samples]
        x = torch.from_numpy(np.stack(waves)).to(device)
        built.append(x)
        if len(built) >= warmup + n_batches:
            break
    for x in built[:warmup]:
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for x in built[warmup : warmup + n_batches]:
        model(x)
        n += int(x.size(0))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "samples": n,
        "seconds": dt,
        "samples_per_sec": n / max(dt, 1e-9),
        "input_len": 6000,
        "kind": kind,
    }


def load_sb(name: str, weights: str):
    import seisbench.models as sbm

    return getattr(sbm, name).from_pretrained(weights)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("[triple] loading STEAD test...", flush=True)
    ds = STEADPickingDataset("test", seq_len=800, load_geometry=False)
    indices = subsample_indices(ds, args.max_events, args.max_noise, args.seed)
    n_ev = sum(1 for i in indices if ds.refs[i].is_event == 1)
    n_nz = len(indices) - n_ev
    print(f"[triple] subset n={len(indices)} events={n_ev} noise={n_nz}", flush=True)

    results = {}

    # HNF
    print(f"[triple] HNF {args.hnf_checkpoint}...", flush=True)
    hnf, cfg = load_model(Path(args.hnf_checkpoint), device)
    hnf.eval()
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False)
    seq_len = int(cfg.get("seq_len", 800))
    results["HNF(run28-50ep)"] = eval_hnf_with_residuals(
        hnf, loader, device, seq_len, args.pick_threshold, args.tol_sec
    )
    # also official evaluate for parity with train test_metrics
    hnf_eval = evaluate(
        hnf,
        loader,
        device,
        seq_len=seq_len,
        pick_threshold=args.pick_threshold,
        pick_tolerance_sec=args.tol_sec,
        post_process_p_before_s=True,
    )
    results["HNF(run28-50ep)"]["train_eval_protocol"] = {
        k: hnf_eval[k]
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
        )
        if k in hnf_eval
    }
    results["HNF(run28-50ep)"]["throughput"] = bench_hnf(
        hnf, loader, device, args.speed_warmup, args.speed_batches
    )
    results["HNF(run28-50ep)"]["throughput"]["input_len"] = seq_len
    print(results["HNF(run28-50ep)"]["train_eval_protocol"], flush=True)
    print("throughput", results["HNF(run28-50ep)"]["throughput"], flush=True)

    for label, cls_name, weights, kind in [
        ("EQT(STEAD)", "EQTransformer", "stead", "eqt"),
        ("PhaseNet(STEAD)", "PhaseNet", "stead", "phasenet"),
    ]:
        print(f"[triple] {label}...", flush=True)
        try:
            sb = load_sb(cls_name, weights)
            results[label] = eval_seisbench(
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
            results[label]["throughput"] = bench_sb(
                sb,
                ds,
                indices,
                device,
                kind,
                args.batch_size,
                args.speed_warmup,
                args.speed_batches,
            )
            print(results[label]["coupled"], flush=True)
            print("throughput", results[label]["throughput"], flush=True)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            print(results[label], flush=True)

    full_test = {}
    ft = Path("outputs/run28/28_ms_fresnel_phys_50ep_local/test_metrics.json")
    if ft.exists():
        full_test = json.loads(ft.read_text())

    report = {
        "protocol": {
            "split": "STEAD EQTransformer test split",
            "subset": {"n_events": n_ev, "n_noise": n_nz, "seed": args.seed, "n": len(indices)},
            "tol_sec": args.tol_sec,
            "pick_threshold": args.pick_threshold,
            "det_threshold": args.det_threshold,
            "hnf_input": f"resampled seq_len={seq_len} over 60s",
            "sb_input": "native 6000 @ 100Hz, peak-norm",
            "mad_definition": "median(|r - median(r)|); also report mad_abs=median(|r|)",
            "sigma_definition": "sample std of (pred-gt) seconds on in-tol picks",
            "throughput_note": "GPU forward only; excludes dataloader disk I/O for HNF (cached tensors), SB includes tensorized waves from prebuilt batches",
        },
        "hnf_full_test_official": {
            k: full_test.get(k)
            for k in (
                "best_epoch",
                "det_f1",
                "p_f1",
                "s_f1",
                "p_mae_sec",
                "s_mae_sec",
                "p_precision",
                "p_recall",
                "s_precision",
                "s_recall",
            )
        },
        "results": results,
    }
    (out / "stead_triple_compare.json").write_text(json.dumps(report, indent=2))

    # markdown table
    def row(name: str, d: dict) -> str:
        if "error" in d:
            return f"| {name} | ERROR | | | | | | | | |"
        c = d.get("train_eval_protocol") or d.get("coupled") or {}
        rp = d.get("residuals", {}).get("p", {})
        rs = d.get("residuals", {}).get("s", {})
        thr = d.get("throughput", {})
        return (
            f"| {name} | {c.get('det_f1', float('nan')):.4f} | {c.get('p_f1', float('nan')):.4f} | "
            f"{c.get('s_f1', float('nan')):.4f} | {c.get('p_mae_sec', float('nan')):.4f} | "
            f"{c.get('s_mae_sec', float('nan')):.4f} | "
            f"{(rp.get('mad_sec') or float('nan')):.4f} / {(rs.get('mad_sec') or float('nan')):.4f} | "
            f"{(rp.get('sigma_sec') or float('nan')):.4f} / {(rs.get('sigma_sec') or float('nan')):.4f} | "
            f"{thr.get('samples_per_sec', float('nan')):.1f} |"
        )

    md = [
        "# STEAD triple compare: HNF vs EQT vs PhaseNet",
        "",
        f"- subset: {n_ev} events + {n_nz} noise (seed={args.seed})",
        f"- tol={args.tol_sec}s  pick_th={args.pick_threshold}",
        f"- HNF ckpt: `{args.hnf_checkpoint}`",
        "",
        "## Shared subset",
        "",
        "| Model | det F1 | P F1 | S F1 | P MAE | S MAE | P/S MAD | P/S σ | samples/s |",
        "|-------|-------:|-----:|-----:|------:|------:|--------:|------:|----------:|",
    ]
    for name in ["HNF(run28-50ep)", "EQT(STEAD)", "PhaseNet(STEAD)"]:
        if name in results:
            md.append(row(name, results[name]))
    if full_test:
        md += [
            "",
            "## HNF full test (official, n≈126k)",
            f"- det={full_test['det_f1']:.4f}  P={full_test['p_f1']:.4f}  S={full_test['s_f1']:.4f}  "
            f"MAE P/S={full_test['p_mae_sec']:.4f}/{full_test['s_mae_sec']:.4f}  best_ep={full_test['best_epoch']}",
        ]
    md += [
        "",
        "## Residual detail (in-tolerance picks)",
        "",
    ]
    for name, d in results.items():
        if "residuals" not in d:
            continue
        md.append(f"### {name}")
        for ph in ("p", "s"):
            r = d["residuals"][ph]
            md.append(
                f"- {ph.upper()}: n={r['n']}  MAE={r.get('mae_sec')}  "
                f"MAD={r.get('mad_sec')}  MAD_abs={r.get('mad_abs_sec')}  "
                f"σ={r.get('sigma_sec')}  median={r.get('median_sec')}"
            )
        md.append("")
    (out / "stead_triple_compare.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
