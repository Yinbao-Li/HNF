#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geo-conditioned knowledge rediscovery (CPU-only).

Attaches STEAD source/receiver lat-lon onto an existing mining sample and
re-screens candidate laws:
  1) globally
  2) inside spatial (lat/lon) clusters
  3) with absolute geography as predictors / controls

Does NOT touch OBS and does not use GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_knowledge_mining import (
    _normal_p_from_r,
    bootstrap_ci,
    bootstrap_partial_ci,
    fdr_bh,
    partial_spearman_corr,
    spearman_corr,
)
from run_paper_scene_clustering import kmeans


STEAD_DIR = Path(__file__).resolve().parent / "STEAD"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geo-conditioned rediscovery (CPU)")
    p.add_argument(
        "--sample-csv",
        default="outputs/paper_scene_clustering/sample_clustered.csv",
    )
    p.add_argument("--output-dir", default="outputs/paper_geo_rediscovery")
    p.add_argument("--n-geo-clusters", type=int, default=4)
    p.add_argument("--trim-quantile", type=float, default=0.95)
    p.add_argument("--min-cluster-n", type=int, default=30)
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def load_stead_geo_index() -> dict[str, dict]:
    """trace_name -> geo fields from STEAD chunk CSVs."""
    cols = [
        "trace_name",
        "network_code",
        "receiver_code",
        "receiver_latitude",
        "receiver_longitude",
        "receiver_elevation_m",
        "source_latitude",
        "source_longitude",
        "source_depth_km",
        "source_distance_km",
        "back_azimuth_deg",
        "snr_db",
    ]
    index: dict[str, dict] = {}
    for chunk in range(1, 7):
        path = STEAD_DIR / f"chunk{chunk}_eofextract" / f"chunk{chunk}.csv"
        if not path.exists():
            continue
        usecols = [c for c in cols if c == "trace_name" or True]
        # chunk1 is noise-only and may lack some event columns
        header = pd.read_csv(path, nrows=0).columns.tolist()
        use = [c for c in cols if c in header]
        df = pd.read_csv(path, usecols=use, low_memory=False)
        for row in df.itertuples(index=False):
            name = row.trace_name
            item = {"trace_name": name, "chunk": chunk}
            for c in use:
                if c == "trace_name":
                    continue
                val = getattr(row, c)
                if c in {"network_code", "receiver_code"}:
                    item[c] = val
                else:
                    item[c] = _as_float(val)
            index[name] = item
    return index


def load_sample(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    out = []
    for r in rows:
        item = {}
        for k, v in r.items():
            if k in {"trace_name", "network_code", "receiver_code"}:
                item[k] = v
            else:
                try:
                    item[k] = float(v)
                except (TypeError, ValueError):
                    item[k] = v
        out.append(item)
    return out


def _as_float(val) -> float:
    if val is None:
        return float("nan")
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "nan":
            return float("nan")
        # STEAD snr_db can be a stringified vector like "[56.8 55.4 47.4]"
        if s.startswith("[") and s.endswith("]"):
            nums = []
            for tok in s.strip("[]").replace(",", " ").split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue
            return float(np.mean(nums)) if nums else float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def attach_geo(rows: list[dict], geo_index: dict[str, dict]) -> tuple[list[dict], dict]:
    kept = []
    miss = 0
    for r in rows:
        g = geo_index.get(r["trace_name"])
        if g is None:
            miss += 1
            continue
        item = dict(r)
        for k, v in g.items():
            if k == "trace_name":
                continue
            item[k] = v
        # aliases used by mining specs
        item["source_lat"] = _as_float(g.get("source_latitude", float("nan")))
        item["source_lon"] = _as_float(g.get("source_longitude", float("nan")))
        item["receiver_lat"] = _as_float(g.get("receiver_latitude", float("nan")))
        item["receiver_lon"] = _as_float(g.get("receiver_longitude", float("nan")))
        item["back_azimuth"] = _as_float(g.get("back_azimuth_deg", float("nan")))
        item["snr_db"] = _as_float(g.get("snr_db", float("nan")))
        if not (np.isfinite(item["source_lat"]) and np.isfinite(item["source_lon"])):
            miss += 1
            continue
        kept.append(item)
    stats = {"n_in": len(rows), "n_geo": len(kept), "n_miss": miss}
    return kept, stats


def robust_trim(rows: list[dict], q: float) -> tuple[list[dict], float]:
    tt = np.array([r["init_tt"] for r in rows], dtype=np.float64)
    cut = float(np.quantile(tt, q))
    return [r for r in rows if r["init_tt"] <= cut], cut


def assign_geo_clusters(rows: list[dict], n_clusters: int, seed: int) -> list[dict]:
    X = np.array([[r["source_lat"], r["source_lon"]] for r in rows], dtype=np.float64)
    labels, centers = kmeans(X, n_clusters, seed)
    for r, lab in zip(rows, labels):
        r["geo_cluster"] = int(lab)
    # also a coarse lon bin for California-ish STEAD
    lons = np.array([r["source_lon"] for r in rows])
    qs = np.quantile(lons, [0.33, 0.66])
    for r in rows:
        lon = r["source_lon"]
        if lon <= qs[0]:
            r["lon_tertile"] = 0
        elif lon <= qs[1]:
            r["lon_tertile"] = 1
        else:
            r["lon_tertile"] = 2
    return rows


def candidate_specs() -> list[dict]:
    pairs = [
        # absolute geography -> quality / latents
        ("source_lat", "pick_err_p"),
        ("source_lon", "pick_err_p"),
        ("source_lat", "pick_err_s"),
        ("source_lon", "pick_err_s"),
        ("source_lat", "init_tt"),
        ("source_lon", "init_tt"),
        ("source_lat", "rho_mean"),
        ("source_lon", "rho_mean"),
        ("source_lat", "noise_ratio"),
        ("source_lon", "noise_ratio"),
        ("receiver_lat", "pick_err_p"),
        ("receiver_lon", "pick_err_p"),
        ("back_azimuth", "pick_err_p"),
        ("back_azimuth", "rho_p_lag"),
        ("snr_db", "pick_err_p"),
        # keep priority latent laws for geo stratification
        ("noise_ratio", "pick_err_p"),
        ("noise_ratio", "init_tt"),
        ("rho_p_lag", "init_tt"),
        ("rho_mean", "vp_mean"),
        ("rho_mean", "vpvs_mean"),
        ("distance_km", "pick_err_p"),
        ("source_depth_km", "pick_err_p"),
    ]
    partials = [
        ("source_lat", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("source_lon", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("source_lat", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("source_lon", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        ("source_lat", "rho_mean", ["distance_km", "source_depth_km"]),
        ("source_lon", "rho_mean", ["distance_km", "source_depth_km"]),
        ("source_lat", "noise_ratio", ["distance_km", "source_depth_km"]),
        ("source_lon", "noise_ratio", ["distance_km", "source_depth_km"]),
        ("receiver_lat", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("receiver_lon", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("back_azimuth", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("noise_ratio", "pick_err_p", ["distance_km", "source_depth_km", "source_lat", "source_lon"]),
        ("rho_p_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon"]),
        ("rho_mean", "vp_mean", ["distance_km", "source_depth_km", "source_lat", "source_lon"]),
        ("noise_ratio", "init_tt", ["distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon"]),
    ]
    specs = []
    for x, y in pairs:
        specs.append({"kind": "pairwise", "x": x, "y": y, "controls": []})
    for x, y, c in partials:
        specs.append({"kind": "partial", "x": x, "y": y, "controls": c})
    return specs


def eval_spec(rows: list[dict], spec: dict, n_boot: int, seed: int) -> dict:
    x = [r.get(spec["x"], float("nan")) for r in rows]
    y = [r.get(spec["y"], float("nan")) for r in rows]
    if spec["kind"] == "pairwise":
        rho = spearman_corr(x, y)
        ci_lo, ci_hi = bootstrap_ci(x, y, n_boot=n_boot, seed=seed)
        n_eff = int(np.sum(np.isfinite(np.asarray(x, dtype=float)) & np.isfinite(np.asarray(y, dtype=float))))
        p = _normal_p_from_r(rho, n_eff)
        return {
            **spec,
            "n": n_eff,
            "stat": rho,
            "ci95": [ci_lo, ci_hi],
            "p_approx": p,
            "ci_excludes_zero": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0)),
        }
    controls = [[r.get(c, float("nan")) for r in rows] for c in spec["controls"]]
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
        rec = eval_spec(rows, spec, n_boot=n_boot, seed=seed + i * 17)
        rec["scope"] = scope
        results.append(rec)
        pvals.append(1.0 if not np.isfinite(rec["p_approx"]) else float(rec["p_approx"]))
    qvals = fdr_bh(pvals)
    for rec, q in zip(results, qvals):
        rec["fdr_q"] = q
        rec["supported"] = bool(rec["ci_excludes_zero"] and q <= 0.10)
    return results


def classify_laws(global_res: list[dict], cluster_res: dict[int, list[dict]], min_n: int) -> dict:
    key = lambda r: (r["kind"], r["x"], r["y"], tuple(r["controls"]))
    gmap = {key(r): r for r in global_res}
    out = {"global": [], "geo_specific": [], "rejected": []}
    for k, g in gmap.items():
        hits = []
        for c, rows in cluster_res.items():
            for r in rows:
                if key(r) == k and r["supported"] and int(r.get("n", 0)) >= min_n:
                    hits.append(
                        {
                            "geo_cluster": c,
                            "stat": r["stat"],
                            "ci95": r["ci95"],
                            "n": r["n"],
                            "fdr_q": r["fdr_q"],
                        }
                    )
        item = {
            "kind": g["kind"],
            "x": g["x"],
            "y": g["y"],
            "controls": g["controls"],
            "global": {
                "supported": g["supported"],
                "stat": g["stat"],
                "ci95": g["ci95"],
                "n": g["n"],
                "fdr_q": g["fdr_q"],
            },
            "geo_hits": hits,
        }
        if g["supported"]:
            out["global"].append(item)
        elif hits:
            out["geo_specific"].append(item)
        else:
            out["rejected"].append(item)
    return out


def plot_map(rows: list[dict], out_png: Path) -> None:
    lats = np.array([r["source_lat"] for r in rows])
    lons = np.array([r["source_lon"] for r in rows])
    labs = np.array([r["geo_cluster"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in sorted(set(labs.tolist())):
        m = labs == c
        ax.scatter(lons[m], lats[m], s=18, alpha=0.75, label=f"geo C{c} (n={m.sum()})")
    ax.set_xlabel("source longitude")
    ax.set_ylabel("source latitude")
    ax.set_title("STEAD mining sample — source locations by geo cluster")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_md(report: dict, path: Path) -> None:
    lines = [
        "# Geo-conditioned rediscovery",
        "",
        f"- n_geo={report['n_geo']} (trim cut init_tt<={report['trim_cut']:.3f})",
        f"- geo_clusters={report['n_geo_clusters']}, min_cluster_n={report['min_cluster_n']}",
        f"- labels: global={len(report['laws']['global'])}, "
        f"geo_specific={len(report['laws']['geo_specific'])}, "
        f"rejected={len(report['laws']['rejected'])}",
        "",
        "## Geo cluster sizes",
        "",
    ]
    for c, n in sorted(report["geo_cluster_sizes"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"- C{c}: n={n}")
    lines += ["", "## Priority claims under geography", ""]
    for claim in report["priority_claims"]:
        lines.append(
            f"- `{claim['id']}`: global_supported={claim['global_supported']}, "
            f"geo_hits={claim['n_geo_hits']}, note={claim['note']}"
        )
    lines += ["", "## Top geo-specific laws", ""]
    for item in report["laws"]["geo_specific"][:12]:
        hits = ", ".join(
            f"C{h['geo_cluster']} ρ={h['stat']:.2f} (n={h['n']})" for h in item["geo_hits"]
        )
        ctrl = ",".join(item["controls"]) if item["controls"] else "-"
        lines.append(f"- {item['kind']} `{item['x']} -> {item['y']}` | {ctrl} | {hits}")
    lines += ["", "## Newly supported absolute-geo predictors (global)", ""]
    for item in report["laws"]["global"]:
        if item["x"] in {
            "source_lat",
            "source_lon",
            "receiver_lat",
            "receiver_lon",
            "back_azimuth",
        }:
            g = item["global"]
            lines.append(
                f"- {item['kind']} `{item['x']} -> {item['y']}` ρ={g['stat']:.3f} "
                f"CI=[{g['ci95'][0]:.3f},{g['ci95'][1]:.3f}] n={g['n']}"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[geo] loading STEAD geo index (CPU)...", flush=True)
    geo_index = load_stead_geo_index()
    print(f"[geo] indexed traces={len(geo_index)}", flush=True)

    rows = load_sample(Path(args.sample_csv))
    rows, join_stats = attach_geo(rows, geo_index)
    print(f"[geo] join {join_stats}", flush=True)
    rows, cut = robust_trim(rows, args.trim_quantile)
    rows = assign_geo_clusters(rows, args.n_geo_clusters, args.seed)
    print(
        f"[geo] after trim n={len(rows)} cut={cut:.3f} "
        f"clusters={dict(Counter(r['geo_cluster'] for r in rows))}",
        flush=True,
    )

    # write enriched sample
    fieldnames = sorted({k for r in rows for k in r.keys()})
    sample_path = out_dir / "sample_with_geo.csv"
    with sample_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    specs = candidate_specs()
    print(f"[geo] screening {len(specs)} specs globally...", flush=True)
    global_res = screen_scope(rows, specs, args.n_bootstrap, args.seed, "global")

    cluster_res: dict[int, list[dict]] = {}
    for c in sorted(set(r["geo_cluster"] for r in rows)):
        sub = [r for r in rows if r["geo_cluster"] == c]
        if len(sub) < args.min_cluster_n:
            print(f"[geo] skip geo C{c} n={len(sub)}", flush=True)
            continue
        print(f"[geo] screening geo C{c} n={len(sub)}...", flush=True)
        cluster_res[c] = screen_scope(sub, specs, args.n_bootstrap, args.seed + 100 * (c + 1), f"geo_C{c}")

    laws = classify_laws(global_res, cluster_res, args.min_cluster_n)

    # priority claim summary
    priority_ids = [
        ("noise_ratio", "pick_err_p", ("distance_km", "source_depth_km", "source_lat", "source_lon")),
        ("rho_p_lag", "init_tt", ("distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon")),
        ("rho_mean", "vp_mean", ("distance_km", "source_depth_km", "source_lat", "source_lon")),
        ("source_lat", "pick_err_p", ("distance_km", "source_depth_km")),
        ("source_lon", "pick_err_p", ("distance_km", "source_depth_km")),
    ]
    priority = []
    for x, y, ctrl in priority_ids:
        matches = [
            item
            for bucket in ("global", "geo_specific", "rejected")
            for item in laws[bucket]
            if item["x"] == x and item["y"] == y and tuple(item["controls"]) == ctrl
        ]
        # also allow pairwise / shorter controls
        if not matches:
            matches = [
                item
                for bucket in ("global", "geo_specific", "rejected")
                for item in laws[bucket]
                if item["x"] == x and item["y"] == y
            ]
        if not matches:
            continue
        item = matches[0]
        note = "global" if item["global"]["supported"] else ("geo_specific" if item["geo_hits"] else "unsupported")
        priority.append(
            {
                "id": f"{x}->{y}",
                "controls": item["controls"],
                "global_supported": item["global"]["supported"],
                "global_stat": item["global"]["stat"],
                "n_geo_hits": len(item["geo_hits"]),
                "geo_hits": item["geo_hits"],
                "note": note,
            }
        )

    plot_map(rows, out_dir / "geo_cluster_map.png")
    # also copy-friendly figure path used by docs if present
    docs_fig = Path("docs/figures/geo_cluster_map.png")
    if docs_fig.parent.exists():
        plot_map(rows, docs_fig)

    report = {
        "join_stats": join_stats,
        "n_geo": len(rows),
        "trim_cut": cut,
        "n_geo_clusters": args.n_geo_clusters,
        "min_cluster_n": args.min_cluster_n,
        "geo_cluster_sizes": dict(Counter(str(r["geo_cluster"]) for r in rows)),
        "lon_tertile_sizes": dict(Counter(str(r["lon_tertile"]) for r in rows)),
        "laws": {
            "global": laws["global"],
            "geo_specific": laws["geo_specific"],
            "rejected": laws["rejected"],
            "counts": {
                "global": len(laws["global"]),
                "geo_specific": len(laws["geo_specific"]),
                "rejected": len(laws["rejected"]),
            },
        },
        "priority_claims": priority,
        "global_results": global_res,
        "geo_cluster_results": {str(k): v for k, v in cluster_res.items()},
    }
    (out_dir / "geo_rediscovery_report.json").write_text(json.dumps(report, indent=2))
    write_md(report, out_dir / "geo_rediscovery_report.md")
    print(json.dumps(report["laws"]["counts"], indent=2), flush=True)
    print(json.dumps(priority, indent=2), flush=True)
    print(f"[geo] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
