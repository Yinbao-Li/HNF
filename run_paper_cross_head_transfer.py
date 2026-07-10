#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-head transfer check for rediscovery priority laws.

Latents (rho_*, noise_ratio, pick_err_*) come from the picking backbone and
are reused from the robust clustered CSV. Only physics-head outputs
(init_tt, vp_mean, vpvs_mean) are recomputed per head.

A law is head-robust if its sign is stable and CI excludes 0 on >=3/4 heads
(or all available heads if fewer).
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
    _normal_p_from_r,
    bootstrap_ci,
    bootstrap_partial_ci,
    fdr_bh,
    partial_spearman_corr,
    spearman_corr,
)
from run_phase_f_stead_profile import time_misfit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-head transfer of priority laws")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--sample-csv", default="outputs/paper_cluster_rediscovery/sample_clustered_robust.csv")
    p.add_argument("--output-dir", default="outputs/paper_cross_head_transfer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--max-events", type=int, default=200)
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=11)
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
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            item: dict[str, Any] = {}
            for k, v in row.items():
                if k == "trace_name":
                    item[k] = v
                else:
                    try:
                        item[k] = float(v)
                    except (TypeError, ValueError):
                        item[k] = v
            rows.append(item)
    return rows


def short_name(path: str) -> str:
    return Path(path).parent.name.replace("zhizi_inversion_", "")


def priority_specs() -> list[dict]:
    return [
        {
            "id": "rho_p_lag_init_tt",
            "kind": "partial",
            "x": "rho_p_lag",
            "y": "init_tt",
            "controls": ["distance_km", "source_depth_km", "pick_err_p"],
            "head_dependent": True,
            "note": "priority causal-chain fit quality",
        },
        {
            "id": "rho_mean_vp_mean",
            "kind": "partial",
            "x": "rho_mean",
            "y": "vp_mean",
            "controls": ["distance_km", "source_depth_km"],
            "head_dependent": True,
            "note": "priority latent-physical coupling",
        },
        {
            "id": "rho_mean_vpvs_mean",
            "kind": "partial",
            "x": "rho_mean",
            "y": "vpvs_mean",
            "controls": ["distance_km", "source_depth_km"],
            "head_dependent": True,
            "note": "strong global candidate; suspect head-induced",
        },
        {
            "id": "noise_ratio_pick_err_p",
            "kind": "partial",
            "x": "noise_ratio",
            "y": "pick_err_p",
            "controls": ["distance_km", "source_depth_km"],
            "head_dependent": False,
            "note": "QC latent; independent of physics head",
        },
        {
            "id": "noise_ratio_init_tt",
            "kind": "partial",
            "x": "noise_ratio",
            "y": "init_tt",
            "controls": ["distance_km", "source_depth_km", "pick_err_p"],
            "head_dependent": True,
            "note": "scene-specific in rediscovery; check head stability",
        },
    ]


def eval_relation(rows: list[dict], spec: dict, n_boot: int, seed: int) -> dict:
    x = [r[spec["x"]] for r in rows]
    y = [r[spec["y"]] for r in rows]
    if spec["kind"] == "pairwise":
        rho = spearman_corr(x, y)
        ci_lo, ci_hi = bootstrap_ci(x, y, n_boot=n_boot, seed=seed)
        n_eff = int(np.sum(np.isfinite(np.asarray(x, dtype=float)) & np.isfinite(np.asarray(y, dtype=float))))
    else:
        controls = [[r[c] for r in rows] for c in spec["controls"]]
        rho, n_eff = partial_spearman_corr(x, y, controls)
        ci_lo, ci_hi = bootstrap_partial_ci(x, y, controls, n_boot=n_boot, seed=seed)
    p = _normal_p_from_r(rho, n_eff)
    ci_ok = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0))
    return {
        "stat": float(rho) if np.isfinite(rho) else float("nan"),
        "ci95": [float(ci_lo), float(ci_hi)],
        "n": int(n_eff),
        "p_approx": float(p) if np.isfinite(p) else 1.0,
        "ci_excludes_zero": ci_ok,
        "sign": 0 if not np.isfinite(rho) or abs(rho) < 1e-12 else (1 if rho > 0 else -1),
    }


def collect_per_head(
    sample_rows: list[dict],
    backbone,
    ckpt_args: dict,
    base,
    device: torch.device,
    infer_seq_len: int,
    head_paths: list[str],
    max_events: int,
) -> tuple[dict[str, list[dict]], list[str]]:
    ds = STEADPickingDataset("test", seq_len=int(ckpt_args.get("seq_len", 800)))
    target = sample_rows[: max(1, int(max_events))]
    wanted = {r["trace_name"] for r in target}
    matched: dict[str, Any] = {}
    for i in range(len(ds)):
        item = ds[i]
        trace = str(item["trace_name"])
        if trace in wanted:
            matched[trace] = item
            if len(matched) == len(wanted):
                break

    head_names: list[str] = []
    bridges = []
    for hp in head_paths:
        p = Path(hp)
        if not p.exists():
            print(f"[cross-head] missing head: {hp}", flush=True)
            continue
        name = short_name(hp)
        head_names.append(name)
        bridges.append(
            load_inversion_bridge_from_checkpoint(
                backbone,
                str(p),
                device,
                embed_dim=int(ckpt_args.get("embed_dim", 64)),
                n_layers=base.n_layers,
                infer_seq_len=infer_seq_len,
            )
        )

    by_head: dict[str, list[dict]] = {h: [] for h in head_names}
    n_ok = 0
    for base_row in target:
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

        shared = {
            "trace_name": trace,
            "distance_km": dist,
            "source_depth_km": depth,
            "cluster": int(base_row.get("cluster", -1)),
            "rho_p_lag": float(base_row["rho_p_lag"]),
            "rho_mean": float(base_row["rho_mean"]),
            "noise_ratio": float(base_row["noise_ratio"]),
            "pick_err_p": float(base_row["pick_err_p"]),
            "pick_err_s": float(base_row.get("pick_err_s", float("nan"))),
        }
        for name, bridge in zip(head_names, bridges):
            bridge.eval()
            geo = (
                encode_geometry_tensor(dist, depth, device=device)
                if getattr(bridge, "geo_condition", False)
                else None
            )
            with torch.no_grad():
                out_head, _ = bridge.forward_event(
                    x, t[0] if t.dim() == 3 else t, include_picks=True, geo=geo
                )
            init_earth = bridge.physics_head.earth(out_head, base.depths, base.q)
            vp = init_earth.vp.detach().cpu().numpy()
            vs = init_earth.vs.detach().cpu().numpy()
            init_tt = float(time_misfit(init_earth, depth, distances, obs_tp, obs_ts))
            rec = {
                **shared,
                "init_tt": init_tt,
                "vp_mean": float(np.mean(vp)),
                "vs_mean": float(np.mean(vs)),
                "vpvs_mean": float(np.mean(vp / np.clip(vs, 1e-6, None))),
            }
            by_head[name].append(rec)
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"[cross-head] events {n_ok}/{len(target)} matched={len(matched)}", flush=True)
    return by_head, head_names


def classify_transfer(per_head: dict[str, dict], head_dependent: bool) -> str:
    if not head_dependent:
        # single evaluation reused; treat as robust if supported
        any_row = next(iter(per_head.values()))
        return "head_independent_supported" if any_row["supported"] else "head_independent_unsupported"
    supported = [h for h, r in per_head.items() if r["supported"]]
    signs = [r["sign"] for r in per_head.values() if r["sign"] != 0]
    if len(supported) >= max(3, len(per_head) - 1) and len(set(signs)) == 1:
        return "head_robust"
    if len(supported) >= 1 and len(set(signs)) == 1:
        return "partially_stable"
    if len(set(signs)) > 1:
        return "sign_unstable"
    return "head_specific_or_weak"


def plot_transfer(results: list[dict], head_names: list[str], out_dir: Path) -> str:
    specs = [r for r in results if r["head_dependent"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)

    # left: effect sizes by head
    x = np.arange(len(specs))
    width = 0.18
    for i, h in enumerate(head_names):
        vals = [r["per_head"][h]["stat"] for r in specs]
        axes[0].bar(x + (i - 1.5) * width, vals, width=width, label=h)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r["id"] for r in specs], rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("partial Spearman")
    axes[0].set_title("Priority laws by physics head")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, axis="y", alpha=0.3)

    # right: transfer labels
    labels = ["head_robust", "partially_stable", "sign_unstable", "head_specific_or_weak", "head_independent_supported"]
    counts = [sum(1 for r in results if r["transfer_label"] == lab) for lab in labels]
    axes[1].barh(range(len(labels)), counts, color=["C2", "C0", "C3", "0.6", "C1"])
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("number of priority relations")
    axes[1].set_title("Transfer classification")
    axes[1].grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(counts):
        axes[1].text(v + 0.05, i, str(v), va="center")

    p = out_dir / "cross_head_transfer_summary.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)

    sample_rows = load_rows(Path(args.sample_csv))
    print(f"[cross-head] sample={len(sample_rows)} max_events={args.max_events} device={args.device}", flush=True)

    device = torch.device(args.device)
    backbone, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=False)
    base = default_synth_model(device)
    by_head, head_names = collect_per_head(
        sample_rows,
        backbone,
        ckpt_args,
        base,
        device,
        args.infer_seq_len,
        args.physics_heads,
        args.max_events,
    )
    if not head_names:
        raise RuntimeError("No physics heads loaded")

    # save wide CSV
    wide_rows = []
    n = len(by_head[head_names[0]])
    for i in range(n):
        row = {k: by_head[head_names[0]][i][k] for k in [
            "trace_name", "distance_km", "source_depth_km", "cluster",
            "rho_p_lag", "rho_mean", "noise_ratio", "pick_err_p", "pick_err_s",
        ]}
        for h in head_names:
            r = by_head[h][i]
            row[f"{h}_init_tt"] = r["init_tt"]
            row[f"{h}_vp_mean"] = r["vp_mean"]
            row[f"{h}_vpvs_mean"] = r["vpvs_mean"]
        wide_rows.append(row)
    with (out_dir / "cross_head_per_event.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
        w.writeheader()
        w.writerows(wide_rows)

    specs = priority_specs()
    results = []
    for si, spec in enumerate(specs):
        per_head = {}
        pvals = []
        if not spec["head_dependent"]:
            # Head-independent QC laws: evaluate on the full robust CSV, not the
            # head-forward subsample (n=200 can lose borderline CI support).
            ev = eval_relation(sample_rows, spec, args.n_bootstrap, args.seed + si * 17)
            ev["fdr_q"] = ev["p_approx"]
            ev["supported"] = bool(ev["ci_excludes_zero"] and ev["fdr_q"] <= 0.10)
            ev["eval_n_source"] = "full_sample_csv"
            for h in head_names:
                per_head[h] = dict(ev)
        else:
            for hi, h in enumerate(head_names):
                ev = eval_relation(by_head[h], spec, args.n_bootstrap, args.seed + si * 17 + hi)
                per_head[h] = ev
                pvals.append(ev["p_approx"])
            qvals = fdr_bh(pvals)
            for h, q in zip(head_names, qvals):
                per_head[h]["fdr_q"] = q
                per_head[h]["supported"] = bool(
                    per_head[h]["ci_excludes_zero"] and q <= 0.10
                )
        label = classify_transfer(per_head, spec["head_dependent"])
        results.append({
            **spec,
            "per_head": per_head,
            "transfer_label": label,
            "n_events": n,
        })

    fig = plot_transfer(results, head_names, out_dir)
    (docs / "cross_head_transfer_summary.png").write_bytes(Path(fig).read_bytes())

    # head agreement on vpvs
    agree = []
    for i, h1 in enumerate(head_names):
        for h2 in head_names[i + 1:]:
            x = [r[f"{h1}_vpvs_mean"] for r in wide_rows]
            y = [r[f"{h2}_vpvs_mean"] for r in wide_rows]
            agree.append({
                "head_a": h1,
                "head_b": h2,
                "vpvs_spearman": spearman_corr(x, y),
                "vpvs_mae": float(np.mean(np.abs(np.asarray(x) - np.asarray(y)))),
            })

    report = {
        "n_events": n,
        "n_sample_csv": len(sample_rows),
        "heads": head_names,
        "support_rule": "CI excludes 0 AND FDR q<=0.10 (FDR across heads per relation)",
        "transfer_rule": "head_robust if supported on >=3 heads (or all-1) with same sign",
        "results": results,
        "head_pair_agreement": agree,
        "figure": fig,
        "notes": {
            "noise_ratio_pick_err_p": "Does not use physics-head outputs; transfer is automatic if supported.",
            "init_tt": "Uses init travel-time misfit (no refine) for scale/stability, matching paper clustering.",
        },
    }
    (out_dir / "transfer_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Cross-Head Transfer of Priority Laws",
        "",
        f"- n events: {n}",
        f"- heads: {', '.join(head_names)}",
        f"- support: CI excludes 0 and FDR q<=0.10",
        "",
        "## Transfer labels",
    ]
    for r in results:
        md.append(f"- `{r['id']}` → **{r['transfer_label']}** ({r['note']})")
        for h in head_names:
            ev = r["per_head"][h]
            md.append(
                f"  - `{h}`: stat={ev['stat']:.3f}, CI=[{ev['ci95'][0]:.3f},{ev['ci95'][1]:.3f}], "
                f"q={ev.get('fdr_q', float('nan')):.3g}, support={ev['supported']}"
            )
    md += ["", "## Head Vp/Vs agreement"]
    for a in agree:
        md.append(f"- `{a['head_a']}` vs `{a['head_b']}`: spearman={a['vpvs_spearman']:.3f}, mae={a['vpvs_mae']:.3f}")
    md += ["", f"## Figure", f"- `{Path(fig).name}`"]
    (out_dir / "transfer_report.md").write_text("\n".join(md))
    print(json.dumps({
        "n_events": n,
        "heads": head_names,
        "labels": {r["id"]: r["transfer_label"] for r in results},
        "report": str(out_dir / "transfer_report.json"),
        "figure": fig,
    }, indent=2))


if __name__ == "__main__":
    main()
