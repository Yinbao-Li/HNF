#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-head / cross-checkpoint knowledge mining (pass 4).

Builds on sample-level exports from run_knowledge_mining.py and interpret
ablation curves. Focus:
  - ablation-derived gamma/omega sensitivities
  - per-event multi-head comparison on the same traces
  - rho_p_lag -> fit-quality stability across heads
  - kernel-parameter comparison across picking checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_stead_picking import load_model
from hnf.inversion_1d import default_synth_model
from hnf.picking_metrics import idx_to_sec
from hnf.picking_prior import run_picking_on_batch
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.stead_zhizi_inversion_dataset import encode_geometry_tensor
from hnf.zhizi_inversion_bridge import load_inversion_bridge_from_checkpoint
from run_knowledge_mining import (
    bootstrap_partial_ci,
    partial_spearman_corr,
    spearman_corr,
    _normal_p_from_r,
)
from run_phase_f_stead_profile import refine_tt, time_misfit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF cross-model knowledge mining")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--sample-csv", default="outputs/knowledge_mining_v3/sample_level_stats.csv")
    p.add_argument("--interpret-report", default="outputs/interpret_suite_extended_v6/interpret_report.json")
    p.add_argument("--output-dir", default="outputs/knowledge_mining_v4")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--max-events", type=int, default=36)
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--physics-heads",
        nargs="*",
        default=[
            "outputs/zhizi_inversion_bridge_macro/best_physics_head.pt",
            "outputs/zhizi_inversion_bridge_residual/best_physics_head.pt",
            "outputs/zhizi_inversion_mixed_geo/best_physics_head.pt",
            "outputs/zhizi_inversion_stead_macro/best_physics_head.pt",
        ],
    )
    p.add_argument(
        "--picking-checkpoints",
        nargs="*",
        default=[
            "outputs/run19/19_detpick_split/best.pt",
            "outputs/run20/20_wrongpeak_sharp/best.pt",
            "outputs/run21/21_s_only_refine/best.pt",
        ],
    )
    return p.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [{k: float(v) if k not in ("trace_name",) and _is_num(v) else v for k, v in row.items()} for row in csv.DictReader(f)]


def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def ablation_sensitivity(interpret_report: Path) -> list[dict[str, Any]]:
    data = json.loads(interpret_report.read_text())
    results = data.get("branch_ablation", {}).get("results", {})
    out = []
    for key, rows in results.items():
        if not rows:
            continue
        vals = np.array([r["value"] for r in rows], dtype=np.float64)
        for target in ["p_lag", "s_lag", "rho_mean", "vp_mean", "vs_mean", "vpvs_mean"]:
            y = np.array([r[target] for r in rows], dtype=np.float64)
            span = float(np.max(y) - np.min(y))
            slope = float((y[-1] - y[0]) / max(vals[-1] - vals[0], 1e-9))
            out.append({
                "scan": key,
                "target": target,
                "value_span": span,
                "finite_diff_slope": slope,
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
            })
    out.sort(key=lambda r: abs(r["finite_diff_slope"]), reverse=True)
    return out


def plot_ablation_sensitivity(rows: list[dict[str, Any]], out_dir: Path) -> str:
    top = [r for r in rows if r["target"] in ("p_lag", "s_lag", "vp_mean", "vpvs_mean")][:12]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    labels = [f"{r['scan']}->{r['target']}" for r in top]
    vals = [r["finite_diff_slope"] for r in top]
    colors = ["C0" if abs(v) > 0.05 else "0.75" for v in vals]
    ax.barh(range(len(top)), vals, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("finite-difference slope")
    ax.set_title("Branch ablation sensitivities (strongest first)")
    ax.grid(True, alpha=0.3, axis="x")
    p = out_dir / "ablation_sensitivity_ranking.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def collect_checkpoint_kernels(checkpoint_paths: list[str], device: torch.device) -> list[dict[str, Any]]:
    out = []
    for ckpt in checkpoint_paths:
        p = Path(ckpt)
        if not p.exists():
            continue
        model, args = load_model(p, device, bypass_noise_cancel=True)
        params = model.collect_kernel_params()
        rec = {"checkpoint": str(p), "name": p.parent.name}
        for branch in ["p_branch_0", "s_branch_0"]:
            if branch in params:
                rec[f"{branch}_gamma"] = float(params[branch]["gamma"])
                rec[f"{branch}_omega"] = float(params[branch]["omega"])
                rec[f"{branch}_wave_speed"] = float(params[branch].get("wave_speed", float("nan")))
        out.append(rec)
    return out


def per_event_head_matrix(
    sample_rows: list[dict[str, Any]],
    backbone,
    ckpt_args: dict,
    base,
    device: torch.device,
    infer_seq_len: int,
    head_paths: list[str],
    max_events: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    ds = STEADPickingDataset("test", seq_len=int(ckpt_args.get("seq_len", 800)))
    target_rows = sample_rows[: max(1, int(max_events))]
    wanted = {r["trace_name"] for r in target_rows}
    matched: dict[str, Any] = {}
    for i in range(len(ds)):
        item = ds[i]
        trace = str(item["trace_name"])
        if trace in wanted:
            matched[trace] = item
            if len(matched) == len(wanted):
                break
    head_names = []
    bridges = []
    for hp in head_paths:
        p = Path(hp)
        if not p.exists():
            continue
        head_names.append(p.parent.name)
        bridges.append(load_inversion_bridge_from_checkpoint(
            backbone, str(p), device,
            embed_dim=int(ckpt_args.get("embed_dim", 64)),
            n_layers=base.n_layers,
            infer_seq_len=infer_seq_len,
        ))
    rows_out = []
    for base_row in target_rows:
        trace = base_row["trace_name"]
        if trace not in matched:
            continue
        item = matched[trace]
        x = item["x"].unsqueeze(0).to(device)
        t = item["t"].unsqueeze(0).to(device) if item["t"].dim() == 2 else item["t"].to(device)
        dist = float(item["source_distance_km"])
        depth = max(float(item["source_depth_km"]), 1.0)
        gt_p = idx_to_sec(int(item["p_idx"]), x.shape[1])
        gt_s = idx_to_sec(int(item["s_idx"]), x.shape[1])
        picks = run_picking_on_batch(backbone, x, t, infer_seq_len=None)
        tp = picks["tp_sec"][0] if picks["tp_sec"][0] is not None else gt_p
        ts = picks["ts_sec"][0] if picks["ts_sec"][0] is not None else gt_s
        distances = torch.tensor([dist], dtype=torch.float32, device=device)
        obs_tp = torch.tensor([tp], device=device)
        obs_ts = torch.tensor([ts], device=device)
        rec = {
            "trace_name": trace,
            "distance_km": dist,
            "source_depth_km": depth,
            "rho_p_lag": float(base_row.get("rho_p_lag", float("nan"))),
            "rho_mean": float(base_row.get("rho_mean", float("nan"))),
            "pick_err_p": float(base_row.get("pick_err_p", float("nan"))),
        }
        for name, bridge in zip(head_names, bridges):
            bridge.eval()
            geo = encode_geometry_tensor(dist, depth, device=device) if getattr(bridge, "geo_condition", False) else None
            with torch.no_grad():
                out_head, _ = bridge.forward_event(x, t[0] if t.dim() == 3 else t, include_picks=True, geo=geo)
            init_earth = bridge.physics_head.earth(out_head, base.depths, base.q)
            ref = refine_tt(
                base.depths, init_earth.vp.detach(), init_earth.vs.detach(), base.q,
                depth, distances, obs_tp, obs_ts,
            )
            vp = ref.earth.vp.detach().cpu().numpy()
            vs = ref.earth.vs.detach().cpu().numpy()
            rec[f"{name}_vp_mean"] = float(np.mean(vp))
            rec[f"{name}_vpvs_mean"] = float(np.mean(vp / np.clip(vs, 1e-6, None)))
            rec[f"{name}_refined_tt"] = float(ref.time_misfit)
        rows_out.append(rec)
    return rows_out, head_names


def head_pair_agreement(matrix_rows: list[dict[str, Any]], head_names: list[str]) -> list[dict[str, Any]]:
    out = []
    for i, h1 in enumerate(head_names):
        for h2 in head_names[i + 1:]:
            x = [r[f"{h1}_vpvs_mean"] for r in matrix_rows]
            y = [r[f"{h2}_vpvs_mean"] for r in matrix_rows]
            out.append({
                "head_a": h1,
                "head_b": h2,
                "vpvs_spearman": spearman_corr(x, y),
                "vpvs_mae": float(np.mean(np.abs(np.asarray(x) - np.asarray(y)))),
            })
    return out


def rho_p_lag_stability_by_head(matrix_rows: list[dict[str, Any]], head_names: list[str], n_boot: int, seed: int) -> list[dict[str, Any]]:
    controls = ["distance_km", "source_depth_km", "pick_err_p"]
    out = []
    for i, h in enumerate(head_names):
        x = [r["rho_p_lag"] for r in matrix_rows]
        y = [r[f"{h}_refined_tt"] for r in matrix_rows]
        c = [[r[cname] for r in matrix_rows] for cname in controls]
        rho, n_eff = partial_spearman_corr(x, y, c)
        ci_lo, ci_hi = bootstrap_partial_ci(x, y, c, n_boot=n_boot, seed=seed + i * 31)
        out.append({
            "head": h,
            "n": n_eff,
            "partial_rho_p_lag_to_refined_tt": rho,
            "ci95": [ci_lo, ci_hi],
            "p_approx": _normal_p_from_r(rho, n_eff),
        })
    return out


def plot_head_heatmap(matrix_rows: list[dict[str, Any]], head_names: list[str], out_dir: Path) -> str:
    if not matrix_rows or not head_names:
        return ""
    mat = np.array([[r[f"{h}_vpvs_mean"] for h in head_names] for r in matrix_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.22 * len(matrix_rows))), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(head_names)))
    ax.set_xticklabels(head_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix_rows)))
    ax.set_yticklabels([r["trace_name"][:18] for r in matrix_rows], fontsize=7)
    ax.set_title("Per-event Vp/Vs across physics heads")
    fig.colorbar(im, ax=ax, fraction=0.03)
    p = out_dir / "cross_head_vpvs_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = load_csv_rows(Path(args.sample_csv))
    ablation_rows = ablation_sensitivity(Path(args.interpret_report))
    ablation_fig = plot_ablation_sensitivity(ablation_rows, out_dir)
    kernel_rows = collect_checkpoint_kernels(args.picking_checkpoints, device)

    backbone, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=True)
    base = default_synth_model(device)
    matrix_rows, head_names = per_event_head_matrix(
        sample_rows, backbone, ckpt_args, base, device, args.infer_seq_len,
        args.physics_heads, args.max_events,
    )
    with (out_dir / "cross_head_per_event.csv").open("w", newline="") as f:
        if matrix_rows:
            writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
            writer.writeheader()
            writer.writerows(matrix_rows)

    pair_agree = head_pair_agreement(matrix_rows, head_names)
    rho_stability = rho_p_lag_stability_by_head(matrix_rows, head_names, args.n_bootstrap, args.seed)
    heatmap = plot_head_heatmap(matrix_rows, head_names, out_dir)

    report = {
        "sample_csv": args.sample_csv,
        "interpret_report": args.interpret_report,
        "n_matrix_rows": len(matrix_rows),
        "head_names": head_names,
        "ablation_sensitivity": ablation_rows[:20],
        "ablation_figure": ablation_fig,
        "checkpoint_kernels": kernel_rows,
        "head_pair_agreement": pair_agree,
        "rho_p_lag_stability_by_head": rho_stability,
        "cross_head_heatmap": heatmap,
        "notes": {
            "ablation": "Finite-difference slopes from interpret-suite branch scans; large |slope| on p_lag/s_lag with small slope on vp/vs supports weak bridge propagation.",
            "rho_p_lag": "If partial rho_p_lag->refined_tt is positive across multiple heads, the candidate is head-robust rather than head-specific.",
        },
    }
    (out_dir / "cross_mining_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Cross-Model Knowledge Mining (Pass 4)",
        "",
        f"- n per-event rows: {len(matrix_rows)}",
        f"- ablation figure: `{Path(ablation_fig).name}`",
        f"- heatmap: `{Path(heatmap).name}`",
        "",
        "## Top ablation sensitivities",
    ]
    for r in ablation_rows[:8]:
        md.append(
            f"- `{r['scan']} -> {r['target']}`: slope={r['finite_diff_slope']:.4f}, span={r['value_span']:.4f}"
        )
    md += ["", "## Checkpoint kernels"]
    for r in kernel_rows:
        md.append(
            f"- `{r['name']}`: p_gamma={r.get('p_branch_0_gamma', float('nan')):.3f}, "
            f"p_omega={r.get('p_branch_0_omega', float('nan')):.3f}, "
            f"s_gamma={r.get('s_branch_0_gamma', float('nan')):.3f}, "
            f"s_omega={r.get('s_branch_0_omega', float('nan')):.3f}"
        )
    md += ["", "## Head pair agreement (Vp/Vs)"]
    for r in pair_agree:
        md.append(f"- `{r['head_a']}` vs `{r['head_b']}`: spearman={r['vpvs_spearman']:.3f}, mae={r['vpvs_mae']:.3f}")
    md += ["", "## rho_p_lag stability by head"]
    for r in rho_stability:
        md.append(
            f"- `{r['head']}`: partial={r['partial_rho_p_lag_to_refined_tt']:.3f}, "
            f"CI=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}], p≈{r['p_approx']:.3g}"
        )
    (out_dir / "cross_mining_report.md").write_text("\n".join(md))
    print(json.dumps({"n_rows": len(matrix_rows), "report": str(out_dir / "cross_mining_report.json")}, indent=2))


if __name__ == "__main__":
    main()
