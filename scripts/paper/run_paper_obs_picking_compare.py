#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig5 picking cross-dataset: OBS (SeisBench) zero-shot compare.

Models:
  - HNF run20 (STEAD-trained)
  - EQTransformer / PhaseNet pretrained on STEAD  (fair zero-shot)
  - EQTransformer / PhaseNet pretrained on OBS   (domain-matched reference)

Uses OBS chunk 201805 (local SeisBench cache). Same 0.5 s tolerance protocol
as STEAD picking metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tools.analyze_stead_picking import load_model
from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    det_pred_from_logits,
    finalize_metrics,
    tolerance_bins,
    update_detection_counts,
    update_picking_counts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS picking cross-dataset compare")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/paper_obs_picking_compare")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk", default="201805")
    p.add_argument("--max-events", type=int, default=400)
    p.add_argument("--seq-len", type=int, default=800, help="HNF resampled length over 60 s")
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--p-offset-sec", type=float, default=15.0, help="P arrival target offset in window")
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--require-full-3c", action="store_true", default=True,
                   help="Keep only traces with energetic Z/1/2 (drop ZH/Z1H incomplete)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def _channel_alive(wave: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(wave ** 2, axis=-1))
    return rms > eps


def normalize_wave(wave: np.ndarray, mode: str) -> np.ndarray:
    """Per-channel demean + scale. mode in {peak, std}."""
    x = wave.astype(np.float32, copy=True)
    x = x - x.mean(axis=-1, keepdims=True)
    if mode == "peak":
        scale = np.max(np.abs(x), axis=-1, keepdims=True)
    elif mode == "std":
        scale = x.std(axis=-1, keepdims=True)
    else:
        raise ValueError(mode)
    return x / (scale + 1e-8)


def load_obs_windows(
    chunk: str,
    max_events: int,
    window_sec: float,
    p_offset_sec: float,
    seed: int,
    require_full_3c: bool = True,
):
    import seisbench.data as sbd

    # Keep hydrophone so OBS-pretrained 4-channel models can run; land models use first 3.
    ds = sbd.OBS(chunks=[chunk], download_if_missing=False, component_order="Z12H")
    meta = ds.metadata
    idxs = []
    for i in range(len(ds)):
        row = meta.iloc[i]
        p = row.get("trace_p_arrival_sample")
        if p is None or (isinstance(p, float) and not np.isfinite(p)):
            continue
        # Prefer complete horizontal components for fair 3C land-model transfer.
        order = str(row.get("trace_component_order", ""))
        if require_full_3c and ("2" not in order):
            continue
        idxs.append(i)
    rng = np.random.default_rng(seed)
    if len(idxs) > max_events:
        idxs = sorted(rng.choice(idxs, size=max_events, replace=False).tolist())

    samples = []
    n_drop_energy = 0
    for i in idxs:
        wave, row = ds.get_sample(i)  # (4, npts) Z12H layout
        wave = np.asarray(wave, dtype=np.float32)
        sr = float(row.get("trace_sampling_rate_hz", 100.0))
        npts = wave.shape[-1]
        p_abs = float(row["trace_p_arrival_sample"])
        s_raw = row.get("trace_s_arrival_sample")
        s_abs = float(s_raw) if s_raw is not None and np.isfinite(float(s_raw)) else float("nan")

        win = int(round(window_sec * sr))
        start = int(round(p_abs - p_offset_sec * sr))
        start = max(0, min(start, max(0, npts - win)))
        end = start + win
        if end > npts:
            pad = end - npts
            seg = np.pad(wave[:, start:npts], ((0, 0), (0, pad)), mode="constant")
        else:
            seg = wave[:, start:end]

        p_rel = p_abs - start
        s_rel = s_abs - start if np.isfinite(s_abs) else float("nan")
        p_valid = 0.0 <= p_rel < win
        s_valid = np.isfinite(s_rel) and 0.0 <= s_rel < win
        if not p_valid:
            continue

        alive = _channel_alive(seg)
        if require_full_3c and not bool(alive[:3].all()):
            n_drop_energy += 1
            continue

        # Store RAW window; apply model-specific normalization at eval time.
        samples.append({
            "wave_4_raw": seg.copy(),
            "wave_3_raw": seg[:3].copy(),
            "sr": sr,
            "p_idx_native": int(round(p_rel)),
            "s_idx_native": int(round(s_rel)) if s_valid else -1,
            "p_valid": True,
            "s_valid": bool(s_valid),
            "trace_name": str(row.get("trace_name_original", row.get("trace_name", i))),
            "station": str(row.get("station_code", "")),
            "component_order": str(row.get("trace_component_order", "")),
            "split": str(row.get("split", "")),
        })
    return samples, {"n_drop_energy": n_drop_energy, "n_candidate_idxs": len(idxs)}


def to_hnf_batch(samples: list[dict], seq_len: int, window_sec: float, device: torch.device):
    xs, ts, p_idx, s_idx, p_valid, s_valid = [], [], [], [], [], []
    for s in samples:
        # Match STEADPickingDataset: per-channel demean + std, then resample.
        x = torch.from_numpy(normalize_wave(s["wave_3_raw"], "std")).float()  # (3, Tn)
        x = F.interpolate(x.unsqueeze(0), size=seq_len, mode="linear", align_corners=False).squeeze(0)
        x = x.transpose(0, 1)  # (T,3)
        scale = seq_len / float(s["wave_3_raw"].shape[-1])
        xs.append(x)
        ts.append(torch.linspace(0.0, window_sec, seq_len).unsqueeze(-1))
        p_idx.append(int(round(s["p_idx_native"] * scale)))
        s_idx.append(int(round(s["s_idx_native"] * scale)) if s["s_valid"] else 0)
        p_valid.append(1.0)
        s_valid.append(1.0 if s["s_valid"] else 0.0)
    return (
        torch.stack(xs).to(device),
        torch.stack(ts).to(device),
        torch.tensor(p_idx, device=device),
        torch.tensor(s_idx, device=device),
        torch.tensor(p_valid, device=device),
        torch.tensor(s_valid, device=device),
    )


def to_sb_batch(samples: list[dict], device: torch.device, n_channels: int, norm_mode: str):
    xs = []
    for s in samples:
        raw = s["wave_4_raw"] if n_channels >= 4 else s["wave_3_raw"]
        xs.append(torch.from_numpy(normalize_wave(raw[:n_channels], norm_mode)).float())
    tlen = max(x.shape[-1] for x in xs)
    out = []
    for x in xs:
        if x.shape[-1] < tlen:
            x = F.pad(x, (0, tlen - x.shape[-1]))
        out.append(x)
    x = torch.stack(out).to(device)
    p_idx = torch.tensor([s["p_idx_native"] for s in samples], device=device)
    s_idx = torch.tensor([s["s_idx_native"] if s["s_valid"] else 0 for s in samples], device=device)
    p_valid = torch.tensor([1.0] * len(samples), device=device)
    s_valid = torch.tensor([1.0 if s["s_valid"] else 0.0 for s in samples], device=device)
    return x, p_idx, s_idx, p_valid, s_valid


def _pick_only_counts(probs, valid, gt_idx, pick_th, tol_bins, seq_len, counts):
    """Event-window pick-only: ignore detection gate; peak>th within tol = TP."""
    from hnf.picking_metrics import idx_to_sec

    max_prob, pred_idx = probs.max(dim=-1)
    for i in range(probs.size(0)):
        if not bool(valid[i].item()):
            continue
        pred_exists = float(max_prob[i]) >= pick_th
        pred_i = int(pred_idx[i])
        gt_i = int(gt_idx[i])
        if pred_exists and abs(pred_i - gt_i) <= tol_bins:
            counts.tp += 1
            counts.mae_sec_sum += abs(idx_to_sec(pred_i, seq_len) - idx_to_sec(gt_i, seq_len))
        elif pred_exists:
            counts.fp += 1
            counts.fn += 1
        else:
            counts.fn += 1


@torch.no_grad()
def eval_hnf(model, samples, device, seq_len, window_sec, pick_th, det_th, tol_sec, batch_size):
    """HNF forward returns LOGITS in det/p/s — must sigmoid before threshold (STEAD protocol)."""
    acc = EvalAccumulator()
    pick_acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, tol_sec)
    n_det_nan = 0
    n_pick_nan = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        x, t, p_idx, s_idx, p_valid, s_valid = to_hnf_batch(chunk, seq_len, window_sec, device)
        out = model(x, t)
        det_true = torch.ones(len(chunk), device=device)
        if "det_logits" in out:
            det_logits = out["det_logits"]
            p_logits = out["p_logits"]
            s_logits = out["s_logits"]
        else:
            # run20 / picking_model: keys are det/p/s but values are logits
            det_logits = out["det"]
            p_logits = out["p"]
            s_logits = out["s"]

        pick_bad = (~torch.isfinite(p_logits)).any(dim=-1) | (~torch.isfinite(s_logits)).any(dim=-1)
        n_pick_nan += int(pick_bad.sum().item())
        p_logits = torch.nan_to_num(p_logits, nan=-50.0, posinf=50.0, neginf=-50.0)
        s_logits = torch.nan_to_num(s_logits, nan=-50.0, posinf=50.0, neginf=-50.0)
        p_probs = torch.sigmoid(p_logits)
        s_probs = torch.sigmoid(s_logits)

        if det_logits.dim() == 1:
            det_nan = ~torch.isfinite(det_logits)
            n_det_nan += int(det_nan.sum().item())
            det_prob = torch.sigmoid(torch.nan_to_num(det_logits, nan=-50.0))
            det_pred = det_prob >= det_th
        else:
            det_nan = (~torch.isfinite(det_logits)).any(dim=-1)
            n_det_nan += int(det_nan.sum().item())
            det_prob = torch.sigmoid(torch.nan_to_num(det_logits, nan=-50.0))
            det_pred = det_prob.amax(dim=-1) >= det_th

        p_probs, s_probs = apply_p_before_s_constraint(p_probs, s_probs, pick_th)
        update_detection_counts(acc, det_pred, det_true)
        update_picking_counts(acc.p, p_probs, det_pred, det_true, p_valid, p_idx, pick_th, tol, seq_len)
        update_picking_counts(acc.s, s_probs, det_pred, det_true, s_valid, s_idx, pick_th, tol, seq_len)
        # pick-only: ignore detection gate (OBS often NaNs det head)
        force_det = torch.ones_like(det_pred)
        _pick_only_counts(p_probs, p_valid, p_idx, pick_th, tol, seq_len, pick_acc.p)
        _pick_only_counts(s_probs, s_valid, s_idx, pick_th, tol, seq_len, pick_acc.s)
        update_detection_counts(pick_acc, force_det, det_true)
    coupled = finalize_metrics(acc)
    pick_only = finalize_metrics(pick_acc)
    return {
        "coupled": coupled,
        "pick_only": pick_only,
        "n_det_nan": n_det_nan,
        "n_pick_nan": n_pick_nan,
        "note": "sigmoid on logits before threshold; matches train_stead_picking.py",
    }


@torch.no_grad()
def eval_seisbench(
    model,
    samples,
    device,
    pick_th,
    det_th,
    tol_sec,
    batch_size,
    kind: str,
    n_channels: int,
    norm_mode: str,
):
    acc = EvalAccumulator()
    pick_acc = EvalAccumulator()
    native_len = samples[0]["wave_3_raw"].shape[-1]
    tol = max(1, int(round(tol_sec * samples[0]["sr"])))
    model = model.to(device).eval()
    # PhaseNet labels are "PSN" in SeisBench (P, S, Noise) — NOT NPS.
    labels = "".join(getattr(model, "labels", "PSN") or "PSN")
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        x, p_idx, s_idx, p_valid, s_valid = to_sb_batch(
            chunk, device, n_channels=n_channels, norm_mode=norm_mode
        )
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
            if out.dim() != 3 or out.shape[1] != 3:
                raise RuntimeError(f"Unexpected PhaseNet out shape {tuple(out.shape)}")
            if labels.upper().startswith("PS"):
                p_prob, s_prob = out[:, 0], out[:, 1]
            elif labels.upper().startswith("NP"):
                p_prob, s_prob = out[:, 1], out[:, 2]
            else:
                # fallback: assume PSN
                p_prob, s_prob = out[:, 0], out[:, 1]
            det_pred = torch.maximum(p_prob.amax(-1), s_prob.amax(-1)) >= pick_th
        det_true = torch.ones(len(chunk), device=device)
        p_prob, s_prob = apply_p_before_s_constraint(p_prob, s_prob, pick_th)
        update_detection_counts(acc, det_pred, det_true)
        update_picking_counts(acc.p, p_prob, det_pred, det_true, p_valid, p_idx, pick_th, tol, native_len)
        update_picking_counts(acc.s, s_prob, det_pred, det_true, s_valid, s_idx, pick_th, tol, native_len)
        force_det = torch.ones_like(det_pred)
        _pick_only_counts(p_prob, p_valid, p_idx, pick_th, tol, native_len, pick_acc.p)
        _pick_only_counts(s_prob, s_valid, s_idx, pick_th, tol, native_len, pick_acc.s)
        update_detection_counts(pick_acc, force_det, det_true)
    return {
        "coupled": finalize_metrics(acc),
        "pick_only": finalize_metrics(pick_acc),
        "norm_mode": norm_mode,
        "labels": labels,
    }


def load_sb_model(name: str, weights: str):
    import seisbench.models as sbm

    cls = getattr(sbm, name)
    return cls.from_pretrained(weights)


def plot_compare(results: dict, out_dir: Path) -> str:
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    p_f1 = [results[n]["pick_only"]["p_f1"] for n in names]
    s_f1 = [results[n]["pick_only"]["s_f1"] for n in names]
    p_mae = [results[n]["pick_only"]["p_mae_sec"] for n in names]
    s_mae = [results[n]["pick_only"]["s_mae_sec"] for n in names]
    colors = ["C2" if "HNF" in n else ("C3" if "(OBS)" in n else "C0") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w / 2, p_f1, width=w, label="P-F1", color="C0")
    axes[0].bar(x + w / 2, s_f1, width=w, label="S-F1", color="C1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("F1 (pick-only)")
    axes[0].set_title("OBS event-window picking F1")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x - w / 2, p_mae, width=w, label="P-MAE", color="C0")
    axes[1].bar(x + w / 2, s_mae, width=w, label="S-MAE", color="C1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    axes[1].set_ylabel("MAE (s)")
    axes[1].set_title("Timing error on true positives")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Fig5 picking cross-dataset: OBS 201805 (HNF vs EQT vs PhaseNet)", fontsize=12)
    p = out_dir / "obs_picking_compare.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    (docs / "fig5_obs_picking_compare.png").write_bytes(p.read_bytes())
    return str(p)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[obs-compare] loading OBS chunk={args.chunk}", flush=True)
    samples, load_info = load_obs_windows(
        args.chunk, args.max_events, args.window_sec, args.p_offset_sec, args.seed,
        require_full_3c=args.require_full_3c,
    )
    n_s = sum(1 for s in samples if s["s_valid"])
    print(
        f"[obs-compare] n={len(samples)} with_S={n_s} device={device} "
        f"drop_energy={load_info['n_drop_energy']} candidates={load_info['n_candidate_idxs']}",
        flush=True,
    )
    if len(samples) < 10:
        raise RuntimeError("Too few OBS samples")

    results = {}

    print("[obs-compare] HNF run20...", flush=True)
    hnf, _ = load_model(Path(args.checkpoint), device, bypass_noise_cancel=False)
    results["HNF(run20/STEAD)"] = eval_hnf(
        hnf, samples, device, args.seq_len, args.window_sec,
        args.pick_threshold, args.det_threshold, args.tol_sec, args.batch_size,
    )
    print(results["HNF(run20/STEAD)"]["pick_only"], flush=True)

    # STEAD land models: SeisBench default norm=peak; OBS models: norm=std.
    for label, name, weights, kind, nch, norm_mode in [
        ("EQT(STEAD)", "EQTransformer", "stead", "eqt", 3, "peak"),
        ("PhaseNet(STEAD)", "PhaseNet", "stead", "phasenet", 3, "peak"),
        ("EQT(OBS)", "EQTransformer", "obs", "eqt", 4, "std"),
        ("PhaseNet(OBS)", "PhaseNet", "obs", "phasenet", 4, "std"),
    ]:
        print(f"[obs-compare] {label} (norm={norm_mode})...", flush=True)
        try:
            m = load_sb_model(name, weights)
            # Prefer model.norm if present
            model_norm = getattr(m, "norm", None)
            if model_norm in ("peak", "std"):
                norm_mode = model_norm
            results[label] = eval_seisbench(
                m, samples, device, args.pick_threshold, args.det_threshold,
                args.tol_sec, args.batch_size, kind=kind, n_channels=nch,
                norm_mode=norm_mode,
            )
            print(results[label]["pick_only"], flush=True)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            print("  failed:", e, flush=True)

    ok = {k: v for k, v in results.items() if "error" not in v}
    fig = plot_compare(ok, out_dir)
    report = {
        "dataset": "SeisBench OBS",
        "chunk": args.chunk,
        "n_events": len(samples),
        "n_with_s": n_s,
        "load_info": load_info,
        "require_full_3c": args.require_full_3c,
        "tol_sec": args.tol_sec,
        "pick_threshold": args.pick_threshold,
        "det_threshold": args.det_threshold,
        "window_sec": args.window_sec,
        "results": results,
        "figure": fig,
        "protocol_fixes": [
            "PhaseNet labels are PSN (was wrongly decoded as NPS)",
            "STEAD models use peak-norm; OBS models use std-norm (SeisBench defaults)",
            "HNF uses per-channel demean+std then resample (matches STEAD training)",
            "Drop incomplete 3C traces (ZH/Z1H) for fair land-model transfer",
            "HNF forward returns logits; must sigmoid before threshold (matches train_stead_picking.py)",
            "Primary metric: pick-only F1 on event windows",
        ],
        "notes": {
            "fair_zero_shot": ["HNF(run20/STEAD)", "EQT(STEAD)", "PhaseNet(STEAD)"],
            "domain_matched_reference": ["EQT(OBS)", "PhaseNet(OBS)"],
            "primary_metric": "pick_only on event windows (detection gate disabled)",
            "secondary_metric": "coupled EQT-style detection+picking",
            "channels": "land/HNF/STEAD models use Z12; OBS-pretrained use Z12H",
        },
    }
    (out_dir / "obs_picking_compare_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# OBS Picking Cross-Dataset Compare",
        "",
        f"- chunk: `{args.chunk}`",
        f"- n: {len(samples)} (with S: {n_s})",
        f"- tolerance: {args.tol_sec} s",
        "- primary: **pick-only** F1 on event windows",
        "",
        "| Model | P-F1 | S-F1 | P-MAE | S-MAE | role |",
        "|------|-----:|-----:|------:|------:|------|",
    ]
    roles = {
        "HNF(run20/STEAD)": "zero-shot",
        "EQT(STEAD)": "zero-shot baseline",
        "PhaseNet(STEAD)": "zero-shot baseline",
        "EQT(OBS)": "domain reference",
        "PhaseNet(OBS)": "domain reference",
    }
    for k, v in results.items():
        if "error" in v:
            md.append(f"| `{k}` | ERR | | | | |")
        else:
            po = v["pick_only"]
            md.append(
                f"| `{k}` | {po['p_f1']:.3f} | {po['s_f1']:.3f} | "
                f"{po['p_mae_sec']:.3f} | {po['s_mae_sec']:.3f} | {roles.get(k,'')} |"
            )
    md += [
        "",
        "## Protocol fixes (v2)",
        "- PhaseNet channel order: **PSN** (previous run wrongly used NPS)",
        "- Normalization: STEAD models `peak`, OBS models `std`, HNF per-channel `std`",
        "- Sample filter: require full energetic Z/1/2 (drop ZH/Z1H incompletes)",
        "",
        "## Interpretation",
        "- Fair zero-shot: HNF / EQT / PhaseNet all STEAD-trained on 3C",
        "- EQT(OBS) / PhaseNet(OBS) use 4C hydrophone and are **not** zero-shot",
        "- Coupled detection metrics are secondary (OBS windows are already events)",
        "",
        f"## Figure",
        f"- `{Path(fig).name}` → `docs/figures/fig5_obs_picking_compare.png`",
    ]
    (out_dir / "obs_picking_compare_report.md").write_text("\n".join(md))
    slim = {}
    for k, v in results.items():
        if "error" in v:
            slim[k] = v
        else:
            slim[k] = {kk: v["pick_only"][kk] for kk in ("p_f1", "s_f1", "p_mae_sec", "s_mae_sec")}
    print(json.dumps({"n": len(samples), "results_pick_only": slim, "figure": fig}, indent=2))


if __name__ == "__main__":
    main()
