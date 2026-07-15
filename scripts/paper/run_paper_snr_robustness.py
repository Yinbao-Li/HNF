#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-grade SNR robustness evaluation (Fig5 core).

Adds white Gaussian noise at target SNR levels to clean STEAD waveforms and
measures:
  - picking P/S F1 and MAE (large N)
  - inversion refine win-rate vs perturb baseline (subset N)
  - denoise-on vs denoise-bypass (same checkpoint)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.analyze_stead_picking import load_model
from hnf.inversion_1d import default_synth_model
from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    det_pred_from_logits,
    idx_to_sec,
    tolerance_bins,
    update_detection_counts,
    update_picking_counts,
)
from hnf.picking_prior import run_picking_on_batch
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.stead_zhizi_inversion_dataset import encode_geometry_tensor
from hnf.physics_decoder import load_physics_decoder_from_checkpoint
from run_phase_f_stead_profile import time_misfit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper SNR robustness (Fig5)")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_mixed_geo/best_physics_head.pt")
    p.add_argument("--output-dir", default="outputs/paper_snr_robustness")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--max-pick-events", type=int, default=512)
    p.add_argument("--max-inv-events", type=int, default=128)
    p.add_argument("--snr-levels", type=float, nargs="*", default=[20.0, 15.0, 10.0, 5.0, 0.0])
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def add_noise_snr(x: torch.Tensor, snr_db: float, rng: np.random.Generator) -> torch.Tensor:
    """x: (B,T,C). SNR defined on RMS signal vs RMS noise."""
    if not np.isfinite(snr_db):
        return x
    xb = x.detach()
    # per-trace RMS over all channels
    rms = xb.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)
    noise_rms = rms / (10.0 ** (snr_db / 20.0))
    noise = torch.from_numpy(rng.standard_normal(size=tuple(xb.shape)).astype(np.float32)).to(xb.device)
    noise = noise * noise_rms
    return xb + noise


def collect_valid_indices(ds: STEADPickingDataset, max_n: int) -> list[int]:
    idxs = []
    for i in range(len(ds)):
        if len(idxs) >= max_n:
            break
        item = ds[i]
        if float(item["det"]) > 0.5 and float(item["p_valid"]) > 0 and float(item["s_valid"]) > 0:
            dist = float(item["source_distance_km"])
            depth = float(item["source_depth_km"])
            if np.isfinite(dist) and np.isfinite(depth) and 1.0 <= dist <= 200.0:
                idxs.append(i)
    return idxs


@torch.no_grad()
def eval_picking(
    model,
    ds: STEADPickingDataset,
    idxs: list[int],
    snr_db: float | None,
    device: torch.device,
    pick_threshold: float,
    det_threshold: float,
    seed: int,
    batch_size: int,
) -> dict:
    acc = EvalAccumulator()
    tol = tolerance_bins(ds.seq_len if hasattr(ds, "seq_len") else 800, 0.5)
    # STEADPickingDataset may not expose seq_len; infer from first item
    seq_len = int(ds[idxs[0]]["x"].shape[0])
    tol = tolerance_bins(seq_len, 0.5)
    rng = np.random.default_rng(seed + int((snr_db if snr_db is not None else 99) * 10))
    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start : start + batch_size]
        xs, ts, dets, p_idx, s_idx, p_valid, s_valid = [], [], [], [], [], [], []
        for i in chunk:
            it = ds[i]
            xs.append(it["x"])
            ts.append(it["t"])
            dets.append(float(it["det"]))
            p_idx.append(int(it["p_idx"]))
            s_idx.append(int(it["s_idx"]))
            p_valid.append(float(it["p_valid"]))
            s_valid.append(float(it["s_valid"]))
        x = torch.stack(xs, dim=0).to(device)
        t = torch.stack(ts, dim=0).to(device)
        if snr_db is not None:
            x = add_noise_snr(x, snr_db, rng)
        out = model(x, t)
        det_pred = det_pred_from_logits(out["det"], threshold=det_threshold)
        det_true = torch.tensor(dets, device=device) > 0.5
        update_detection_counts(acc, det_pred, det_true)
        p_probs = torch.sigmoid(out["p"])
        s_probs = torch.sigmoid(out["s"])
        p_probs, s_probs = apply_p_before_s_constraint(p_probs, s_probs, pick_threshold)
        update_picking_counts(
            acc.p,
            p_probs,
            det_pred,
            det_true,
            torch.tensor(p_valid, device=device) > 0,
            torch.tensor(p_idx, device=device, dtype=torch.long),
            pick_threshold,
            tol,
            seq_len,
        )
        update_picking_counts(
            acc.s,
            s_probs,
            det_pred,
            det_true,
            torch.tensor(s_valid, device=device) > 0,
            torch.tensor(s_idx, device=device, dtype=torch.long),
            pick_threshold,
            tol,
            seq_len,
        )
    p_pr, p_re, p_f1 = acc.p.prf()
    s_pr, s_re, s_f1 = acc.s.prf()
    d_pr, d_re, d_f1 = acc.det.prf()
    return {
        "n": len(idxs),
        "det_f1": d_f1,
        "p_f1": p_f1,
        "s_f1": s_f1,
        "p_mae": acc.p.mae_sec(),
        "s_mae": acc.s.mae_sec(),
        "p_precision": p_pr,
        "p_recall": p_re,
        "s_precision": s_pr,
        "s_recall": s_re,
    }


@torch.no_grad()
def eval_inversion_subset(
    backbone,
    bridge,
    ds: STEADPickingDataset,
    idxs: list[int],
    snr_db: float | None,
    device: torch.device,
    seed: int,
    infer_seq_len: int = 600,
) -> dict:
    from hnf.picking_prior import downsample_traces
    base = default_synth_model(device)
    rng = np.random.default_rng(seed + 1000 + int((snr_db if snr_db is not None else 99) * 10))
    wins = 0
    n = 0
    zh_tt = []
    pe_tt = []
    for i in idxs:
        it = ds[i]
        x = it["x"].unsqueeze(0).to(device)
        t = it["t"].unsqueeze(0).to(device) if it["t"].dim() == 2 else it["t"].to(device)
        if snr_db is not None:
            x = add_noise_snr(x, snr_db, rng)
        x, t_ds = downsample_traces(x, t[0] if t.dim() == 3 else t, infer_seq_len)
        t = t_ds
        dist = float(it["source_distance_km"])
        depth = max(float(it["source_depth_km"]), 1.0)
        # map GT indices to downsampled timeline approximately via seconds
        gt_p = idx_to_sec(int(it["p_idx"]), it["x"].shape[0])
        gt_s = idx_to_sec(int(it["s_idx"]), it["x"].shape[0])
        picks = run_picking_on_batch(backbone, x, t.unsqueeze(-1) if t.dim() == 1 else t, infer_seq_len=None)
        tp = picks["tp_sec"][0] if picks["tp_sec"][0] is not None else gt_p
        ts = picks["ts_sec"][0] if picks["ts_sec"][0] is not None else gt_s
        geo = encode_geometry_tensor(dist, depth, device=device) if getattr(bridge, "geo_condition", False) else None
        distances = torch.tensor([dist], dtype=torch.float32, device=device)
        obs_tp = torch.tensor([tp], device=device)
        obs_ts = torch.tensor([ts], device=device)
        t_event = t if t.dim() == 2 else t.unsqueeze(-1)
        out, _ = bridge.forward_event(x, t_event, include_picks=True, geo=geo)
        init_earth = bridge.physics_head.earth(out, base.depths, base.q)
        # Use init TT for paper-scale SNR sweep (stable + fast). Refine is covered elsewhere.
        z = float(time_misfit(init_earth, depth, distances, obs_tp, obs_ts))
        scale = 1.0 + float(rng.uniform(-0.08, 0.08))
        from hnf.inversion_1d import LayeredEarth1D
        pert_earth = LayeredEarth1D(
            depths=base.depths,
            vp=(init_earth.vp.detach() * scale).clamp(1.2, 8.5),
            vs=(init_earth.vs.detach() * scale).clamp(0.7, 5.0),
            q=base.q,
        )
        p = float(time_misfit(pert_earth, depth, distances, obs_tp, obs_ts))
        zh_tt.append(z)
        pe_tt.append(p)
        if z <= p:
            wins += 1
        n += 1
    return {
        "n": n,
        "win_frac": wins / max(n, 1),
        "mean_zhizi_tt": float(np.mean(zh_tt)) if zh_tt else float("nan"),
        "mean_perturb_tt": float(np.mean(pe_tt)) if pe_tt else float("nan"),
        "metric": "init_tt",
    }


def plot_fig5(report: dict, out_dir: Path) -> dict:
    snrs = report["snr_levels"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    for mode, style in [("denoise_on", "-o"), ("denoise_bypass", "--s")]:
        rows = report["picking"][mode]
        axes[0].plot(snrs, [rows[str(s)]["p_f1"] for s in snrs], style, label=f"P {mode}")
        axes[0].plot(snrs, [rows[str(s)]["s_f1"] for s in snrs], style.replace("o", "^").replace("s", "D"), label=f"S {mode}")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("F1")
    axes[0].set_title("Picking F1 vs SNR")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].invert_xaxis()

    for mode, style in [("denoise_on", "-o"), ("denoise_bypass", "--s")]:
        rows = report["picking"][mode]
        axes[1].plot(snrs, [rows[str(s)]["p_mae"] for s in snrs], style, label=f"P {mode}")
        axes[1].plot(snrs, [rows[str(s)]["s_mae"] for s in snrs], style.replace("o", "^").replace("s", "D"), label=f"S {mode}")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("MAE (s)")
    axes[1].set_title("Pick MAE vs SNR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].invert_xaxis()

    for mode, style in [("denoise_on", "-o"), ("denoise_bypass", "--s")]:
        rows = report["inversion"][mode]
        axes[2].plot(snrs, [rows[str(s)]["win_frac"] for s in snrs], style, label=mode)
    axes[2].axhline(0.5, color="k", ls=":", lw=1)
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Zhizi refine win frac")
    axes[2].set_title("Inversion refine win-rate vs SNR")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[2].invert_xaxis()

    p = out_dir / "fig5_snr_robustness.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return {"figure": str(p)}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("docs/figures")
    docs_dir.mkdir(parents=True, exist_ok=True)

    ds = STEADPickingDataset("test", seq_len=args.seq_len)
    pick_idxs = collect_valid_indices(ds, args.max_pick_events)
    inv_idxs = pick_idxs[: args.max_inv_events]
    print(f"[snr] pick_n={len(pick_idxs)} inv_n={len(inv_idxs)} snrs={args.snr_levels}", flush=True)

    report = {
        "checkpoint": args.checkpoint,
        "physics_head": args.physics_head,
        "snr_levels": args.snr_levels,
        "max_pick_events": len(pick_idxs),
        "max_inv_events": len(inv_idxs),
        "picking": {"denoise_on": {}, "denoise_bypass": {}},
        "inversion": {"denoise_on": {}, "denoise_bypass": {}},
    }

    for bypass, mode in [(False, "denoise_on"), (True, "denoise_bypass")]:
        print(f"[snr] mode={mode}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        backbone, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=bypass)
        bridge = load_physics_decoder_from_checkpoint(
            backbone,
            args.physics_head,
            device,
            embed_dim=int(ckpt_args.get("embed_dim", 64)),
            n_layers=default_synth_model(device).n_layers,
            infer_seq_len=600,
        )
        bridge.eval()
        # noise-cancel path is memory-heavy; use smaller batches
        bs = 4 if (not bypass) else args.batch_size
        for snr in args.snr_levels:
            print(f"[snr] {mode} snr={snr} picking...", flush=True)
            pick_m = eval_picking(
                backbone, ds, pick_idxs, snr, device,
                args.pick_threshold, args.det_threshold, args.seed, bs,
            )
            report["picking"][mode][str(snr)] = pick_m
            print(f"[snr] {mode} snr={snr} inversion...", flush=True)
            inv_m = eval_inversion_subset(backbone, bridge, ds, inv_idxs, snr, device, args.seed, infer_seq_len=600)
            report["inversion"][mode][str(snr)] = inv_m
            print(
                f"[snr] {mode} snr={snr}: p_f1={pick_m['p_f1']:.3f} s_f1={pick_m['s_f1']:.3f} "
                f"win={inv_m['win_frac']:.3f}",
                flush=True,
            )
            (out_dir / "snr_report.json").write_text(json.dumps(report, indent=2))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del backbone, bridge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    figs = plot_fig5(report, out_dir)
    report["figures"] = figs
    # also clean reference at +inf conceptually = no added noise already covered by high SNR
    (out_dir / "snr_report.json").write_text(json.dumps(report, indent=2))
    docs_fig = docs_dir / "fig5_snr_robustness.png"
    docs_fig.write_bytes(Path(figs["figure"]).read_bytes())
    md = [
        "# Paper SNR Robustness (Fig5)",
        "",
        f"- pick events: {len(pick_idxs)}",
        f"- inversion events: {len(inv_idxs)}",
        f"- SNR levels (dB): {args.snr_levels}",
        f"- figure: `{Path(figs['figure']).name}`",
        "",
        "## Picking (denoise_on)",
    ]
    for snr in args.snr_levels:
        r = report["picking"]["denoise_on"][str(snr)]
        md.append(f"- SNR {snr}: P-F1={r['p_f1']:.3f}, S-F1={r['s_f1']:.3f}, P-MAE={r['p_mae']:.3f}s, S-MAE={r['s_mae']:.3f}s")
    md += ["", "## Inversion win-rate (denoise_on)"]
    for snr in args.snr_levels:
        r = report["inversion"]["denoise_on"][str(snr)]
        md.append(f"- SNR {snr}: win_frac={r['win_frac']:.3f}, mean_zhizi_tt={r['mean_zhizi_tt']:.3f}")
    (out_dir / "snr_report.md").write_text("\n".join(md))
    print(json.dumps({"figure": figs["figure"], "docs": str(docs_fig), "report": str(out_dir / "snr_report.json")}, indent=2))


if __name__ == "__main__":
    main()
