#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full knowledge rediscovery conditioned on scene clusters.

Why this exists:
  Earlier mining mixed heterogeneous STEAD scenes. Weak global correlations can
  hide laws that are stable only inside a scene type. This script re-runs a
  broad candidate graph globally and within each robust cluster.

Inputs:
  outputs/paper_scene_clustering/sample_clustered.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_knowledge_mining import (
    _normal_p_from_r,
    bootstrap_ci,
    bootstrap_partial_ci,
    fdr_bh,
    partial_spearman_corr,
    spearman_corr,
)
from run_paper_scene_clustering import kmeans


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster-conditioned full rediscovery")
    p.add_argument("--sample-csv", default="outputs/paper_scene_clustering/sample_clustered.csv")
    p.add_argument("--output-dir", default="outputs/paper_cluster_rediscovery")
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--trim-quantile", type=float, default=0.95)
    p.add_argument("--min-cluster-n", type=int, default=30)
    p.add_argument("--n-bootstrap", type=int, default=300)
    # seed=11 reproduces outputs/paper_scene_clustering/cluster_report_robust.*
    p.add_argument("--seed", type=int, default=11)
    p.add_argument(
        "--reuse-cluster-col",
        action="store_true",
        help="Use existing CSV cluster labels instead of re-clustering after trim "
             "(not recommended: original CSV labels are pre-trim and unbalanced).",
    )
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    out = []
    for r in rows:
        item = {}
        for k, v in r.items():
            if k == "trace_name":
                item[k] = v
            else:
                try:
                    item[k] = float(v)
                except (TypeError, ValueError):
                    item[k] = v
        out.append(item)
    return out


def robust_trim(rows: list[dict], q: float) -> tuple[list[dict], float]:
    tt = np.array([r["init_tt"] for r in rows], dtype=np.float64)
    cut = float(np.quantile(tt, q))
    kept = [r for r in rows if r["init_tt"] <= cut]
    return kept, cut


def assign_clusters(rows: list[dict], n_clusters: int, seed: int) -> list[dict]:
    feat_keys = [
        "distance_km", "source_depth_km", "pick_err_p", "pick_err_s",
        "rho_mean", "rho_p_lag", "rho_s_lag", "noise_ratio", "init_tt",
    ]
    X = np.array([[r[k] for k in feat_keys] for r in rows], dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        if not np.all(np.isfinite(col)):
            med = np.nanmedian(col)
            col[~np.isfinite(col)] = med if np.isfinite(med) else 0.0
            X[:, j] = col
    labels, _ = kmeans(X, n_clusters, seed)
    for r, lab in zip(rows, labels):
        r["cluster"] = int(lab)
    return rows


def candidate_specs() -> list[dict]:
    """Broad but physically motivated candidate graph."""
    # pairwise Spearman (no controls)
    pairs = [
        ("distance_km", "rho_mean"),
        ("source_depth_km", "rho_mean"),
        ("distance_km", "vp_mean"),
        ("distance_km", "vpvs_mean"),
        ("source_depth_km", "vp_mean"),
        ("rho_mean", "vp_mean"),
        ("rho_mean", "vpvs_mean"),
        ("rho_mean", "init_tt"),
        ("rho_p_lag", "init_tt"),
        ("rho_s_lag", "init_tt"),
        ("p_prob_lag", "init_tt"),
        ("s_prob_lag", "init_tt"),
        ("pick_err_p", "init_tt"),
        ("pick_err_s", "init_tt"),
        ("noise_ratio", "pick_err_p"),
        ("noise_ratio", "pick_err_s"),
        ("noise_ratio", "init_tt"),
        ("noise_ratio", "rho_mean"),
        ("noise_energy", "pick_err_p"),
        ("residual_energy", "pick_err_p"),
        ("rho_p_lag", "p_prob_lag"),
        ("rho_s_lag", "s_prob_lag"),
    ]
    # partial Spearman with controls
    partials = [
        ("rho_mean", "vp_mean", ["distance_km", "source_depth_km"]),
        ("rho_mean", "vpvs_mean", ["distance_km", "source_depth_km"]),
        ("rho_mean", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("rho_p_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("rho_s_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_s"]),
        ("p_prob_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("s_prob_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_s"]),
        ("noise_ratio", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("noise_ratio", "pick_err_s", ["distance_km", "source_depth_km"]),
        ("noise_ratio", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("noise_ratio", "rho_mean", ["distance_km", "source_depth_km"]),
        ("rho_p_lag", "vp_mean", ["distance_km", "source_depth_km"]),
        ("noise_energy", "init_tt", ["distance_km", "source_depth_km"]),
    ]
    specs = []
    for x, y in pairs:
        specs.append({"kind": "pairwise", "x": x, "y": y, "controls": []})
    for x, y, c in partials:
        specs.append({"kind": "partial", "x": x, "y": y, "controls": c})
    return specs


def eval_spec(rows: list[dict], spec: dict, n_boot: int, seed: int) -> dict:
    x = [r[spec["x"]] for r in rows]
    y = [r[spec["y"]] for r in rows]
    if spec["kind"] == "pairwise":
        rho = spearman_corr(x, y)
        ci_lo, ci_hi = bootstrap_ci(x, y, n_boot=n_boot, seed=seed)
        n_eff = int(np.sum(np.isfinite(np.asarray(x)) & np.isfinite(np.asarray(y))))
        p = _normal_p_from_r(rho, n_eff)
        return {
            **spec,
            "n": n_eff,
            "stat": rho,
            "ci95": [ci_lo, ci_hi],
            "p_approx": p,
            "ci_excludes_zero": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0)),
        }
    controls = [[r[c] for r in rows] for c in spec["controls"]]
    rho, n_eff = partial_spearman_corr(x, y, controls)
    ci_lo, ci_hi = bootstrap_partial_ci(x, y, controls, n_boot=n_boot, seed=seed)
    p = _normal_p_from_r(rho, n_eff)
    return {
        **spec,
        "n": n_eff,
        "stat": rho,
        "ci95": [ci_lo, ci_hi],
        "p_approx": p,
        "ci_excludes_zero": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0)),
    }


def screen_scope(rows: list[dict], specs: list[dict], n_boot: int, seed: int, scope: str) -> list[dict]:
    results = []
    pvals = []
    for i, spec in enumerate(specs):
        rec = eval_spec(rows, spec, n_boot=n_boot, seed=seed + i * 13)
        rec["scope"] = scope
        results.append(rec)
        pvals.append(1.0 if not np.isfinite(rec["p_approx"]) else float(rec["p_approx"]))
    qvals = fdr_bh(pvals)
    for rec, q in zip(results, qvals):
        rec["fdr_q"] = q
        rec["supported"] = bool(rec["ci_excludes_zero"] and q <= 0.10)
    return results


def classify_laws(global_res: list[dict], cluster_res: dict[int, list[dict]], min_cluster_n: int) -> dict:
    """
    global: supported in all-sample screen
    scene_specific: not global, but supported in >=1 eligible cluster
    rejected: neither
    """
    key = lambda r: (r["kind"], r["x"], r["y"], tuple(r["controls"]))
    gmap = {key(r): r for r in global_res}
    scene = []
    global_laws = []
    rejected = []
    for k, g in gmap.items():
        cluster_hits = []
        for c, rows in cluster_res.items():
            for r in rows:
                if key(r) == k and r["supported"] and int(r.get("n", 0)) >= min_cluster_n:
                    cluster_hits.append({"cluster": c, "stat": r["stat"], "ci95": r["ci95"], "n": r["n"], "fdr_q": r["fdr_q"]})
        item = {
            "kind": g["kind"],
            "x": g["x"],
            "y": g["y"],
            "controls": g["controls"],
            "global": {
                "supported": g["supported"],
                "stat": g["stat"],
                "ci95": g["ci95"],
                "fdr_q": g["fdr_q"],
                "n": g["n"],
            },
            "cluster_hits": cluster_hits,
        }
        if g["supported"]:
            item["label"] = "global"
            global_laws.append(item)
        elif cluster_hits:
            item["label"] = "scene_specific"
            scene.append(item)
        else:
            item["label"] = "rejected"
            rejected.append(item)
    return {"global_laws": global_laws, "scene_specific_laws": scene, "rejected_count": len(rejected)}


def plot_summary(classed: dict, cluster_profiles: list[dict], out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)

    # left: counts
    labels = ["global", "scene-specific", "rejected"]
    vals = [
        len(classed["global_laws"]),
        len(classed["scene_specific_laws"]),
        classed["rejected_count"],
    ]
    axes[0].bar(labels, vals, color=["C2", "C1", "0.7"])
    axes[0].set_ylabel("number of candidate relations")
    axes[0].set_title("Rediscovery labels (CI + FDR<=0.10)")
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.2, str(v), ha="center")

    # right: top supported effects
    items = []
    for g in classed["global_laws"]:
        items.append((f"G:{g['x']}->{g['y']}", g["global"]["stat"]))
    for s in classed["scene_specific_laws"]:
        best = max(s["cluster_hits"], key=lambda h: abs(h["stat"]))
        items.append((f"C{best['cluster']}:{s['x']}->{s['y']}", best["stat"]))
    items = sorted(items, key=lambda t: abs(t[1]), reverse=True)[:12]
    if items:
        axes[1].barh(range(len(items)), [v for _, v in items], color=["C0" if n.startswith("G") else "C1" for n, _ in items])
        axes[1].set_yticks(range(len(items)))
        axes[1].set_yticklabels([n for n, _ in items], fontsize=8)
        axes[1].axvline(0, color="k", lw=1)
        axes[1].set_xlabel("effect size (Spearman / partial)")
        axes[1].set_title("Top supported effects")
        axes[1].grid(True, axis="x", alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.1, 0.5, "No supported laws", fontsize=12)

    p = out_dir / "cluster_rediscovery_summary.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.sample_csv))
    kept, tt_cut = robust_trim(rows, args.trim_quantile)
    if args.reuse_cluster_col and all("cluster" in r for r in kept):
        for r in kept:
            r["cluster"] = int(r["cluster"])
        print("[rediscover] reusing CSV cluster labels (pre-trim assignment)", flush=True)
    else:
        kept = assign_clusters(kept, args.n_clusters, args.seed)
        print(f"[rediscover] reclustered after trim with seed={args.seed}", flush=True)
    specs = candidate_specs()

    print(f"[rediscover] n_kept={len(kept)}/{len(rows)} tt_cut={tt_cut:.3f} n_specs={len(specs)}", flush=True)

    # profiles
    profiles = []
    for c in range(args.n_clusters):
        sel = [r for r in kept if r["cluster"] == c]
        if not sel:
            continue
        profiles.append({
            "cluster": c,
            "n": len(sel),
            "distance_km_mean": float(np.mean([r["distance_km"] for r in sel])),
            "source_depth_km_mean": float(np.mean([r["source_depth_km"] for r in sel])),
            "noise_ratio_mean": float(np.mean([r["noise_ratio"] for r in sel])),
            "init_tt_mean": float(np.mean([r["init_tt"] for r in sel])),
            "pick_err_p_mean": float(np.mean([r["pick_err_p"] for r in sel])),
            "rho_mean_mean": float(np.mean([r["rho_mean"] for r in sel])),
        })

    global_res = screen_scope(kept, specs, args.n_bootstrap, args.seed, scope="all")
    cluster_res: dict[int, list[dict]] = {}
    for c in range(args.n_clusters):
        sel = [r for r in kept if r["cluster"] == c]
        if len(sel) < args.min_cluster_n:
            print(f"[rediscover] skip cluster {c}: n={len(sel)} < {args.min_cluster_n}", flush=True)
            continue
        print(f"[rediscover] screening cluster {c} n={len(sel)}", flush=True)
        cluster_res[c] = screen_scope(sel, specs, args.n_bootstrap, args.seed + 1000 + c * 97, scope=f"C{c}")

    classed = classify_laws(global_res, cluster_res, args.min_cluster_n)
    fig = plot_summary(classed, profiles, out_dir)
    (docs / "cluster_rediscovery_summary.png").write_bytes(Path(fig).read_bytes())

    # save clustered table
    fields = list(kept[0].keys())
    with (out_dir / "sample_clustered_robust.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)

    report = {
        "n_total": len(rows),
        "n_kept": len(kept),
        "tt_cut": tt_cut,
        "n_specs": len(specs),
        "min_cluster_n": args.min_cluster_n,
        "support_rule": "bootstrap CI excludes 0 AND FDR q<=0.10",
        "profiles": profiles,
        "global_results": global_res,
        "cluster_results": {str(k): v for k, v in cluster_res.items()},
        "classification": classed,
        "figure": fig,
        "notes": {
            "why_cluster": "Full-sample mining mixes heterogeneous scenes; cluster-conditioned rediscovery separates global laws from scene-specific laws.",
            "previous_gap": "Earlier cluster pass only tested 4 hand-picked relations; this pass screens the full candidate graph.",
        },
    }
    (out_dir / "rediscovery_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# Cluster-Conditioned Knowledge Rediscovery",
        "",
        f"- kept: {len(kept)} / {len(rows)} (init_tt <= q{args.trim_quantile} = {tt_cut:.3f})",
        f"- candidate relations: {len(specs)}",
        f"- support rule: bootstrap CI excludes 0 and FDR q<=0.10",
        f"- global laws: {len(classed['global_laws'])}",
        f"- scene-specific laws: {len(classed['scene_specific_laws'])}",
        f"- rejected: {classed['rejected_count']}",
        "",
        "## Cluster profiles",
    ]
    for p0 in profiles:
        md.append(
            f"- C{p0['cluster']} n={p0['n']}: dist={p0['distance_km_mean']:.1f}, "
            f"depth={p0['source_depth_km_mean']:.1f}, noise={p0['noise_ratio_mean']:.3f}, "
            f"init_tt={p0['init_tt_mean']:.2f}"
        )
    md += ["", "## Global laws"]
    if not classed["global_laws"]:
        md.append("- none")
    for g in sorted(classed["global_laws"], key=lambda r: abs(r["global"]["stat"]), reverse=True):
        ctrl = f" | {', '.join(g['controls'])}" if g["controls"] else ""
        md.append(
            f"- `{g['kind']}` `{g['x']}->{g['y']}{ctrl}`: "
            f"stat={g['global']['stat']:.3f}, CI=[{g['global']['ci95'][0]:.3f},{g['global']['ci95'][1]:.3f}], "
            f"q={g['global']['fdr_q']:.3g}, n={g['global']['n']}"
        )
    md += ["", "## Scene-specific laws"]
    if not classed["scene_specific_laws"]:
        md.append("- none")
    for s in classed["scene_specific_laws"]:
        hits = ", ".join([f"C{h['cluster']}({h['stat']:+.3f},n={h['n']})" for h in s["cluster_hits"]])
        ctrl = f" | {', '.join(s['controls'])}" if s["controls"] else ""
        md.append(f"- `{s['kind']}` `{s['x']}->{s['y']}{ctrl}`: {hits}")
    md += ["", f"## Figure", f"- `{Path(fig).name}`"]
    (out_dir / "rediscovery_report.md").write_text("\n".join(md))
    print(json.dumps({
        "n_kept": len(kept),
        "n_global": len(classed["global_laws"]),
        "n_scene": len(classed["scene_specific_laws"]),
        "n_rejected": classed["rejected_count"],
        "report": str(out_dir / "rediscovery_report.json"),
        "figure": fig,
    }, indent=2))


if __name__ == "__main__":
    main()
