#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.7: cluster-bootstrap CIs for AE + headline dumps.

Fast path:
  - F1: per-trace [tp,fp,fn] sums (trace/event/sensor @ n_boot=1000)
  - AP: event-cluster only @ n_boot_ap (default 200), using pre-flattened pick arrays

  python tools/phase2_cluster_bootstrap_ci.py --output-dir results/phase2/2_7_ci
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ae-root", type=Path, default=Path("outputs/ae/zs_seeds"))
    p.add_argument("--ae-data", type=Path, default=Path("/media/bob/Work/aesnet/AEsNet_Dataset"))
    p.add_argument("--models", default="ours,eqt_aug,pn_aug,eqt_official,pn_official")
    p.add_argument("--ref", default="ours")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--tols", default="10,20,50")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-boot-ap", type=int, default=200)
    p.add_argument("--headline-dir", type=Path, default=Path("results/phase2/2_7_ci"))
    p.add_argument("--output-dir", type=Path, default=Path("results/phase2/2_7_ci"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def f1_from_counts(c: np.ndarray) -> float:
    tp, fp, fn = c.sum(0)
    return float(2 * tp / max(1.0, 2 * tp + fp + fn))


def parse_ae_name(name: str) -> tuple[str, str]:
    sensor = name.split("/")[-1] if "/" in name else "UNK"
    m = re.search(r"(?:Noise_)?AE_(\d+)", name)
    event = f"AE_{m.group(1)}" if m else name.split("/")[0]
    return event, sensor


def load_ae_names(ae_data: Path, mat: str) -> list[str]:
    base = "Minusil" if mat.startswith("Minusil") else "Glass_beads"
    ev = [l.strip() for l in open(ae_data / base / "data_list_test.txt") if l.strip() and l.strip() != "file_name"]
    nz = [l.strip() for l in open(ae_data / base / "data_list_testNOISE.txt") if l.strip() and l.strip() != "file_name"]
    return ev + nz


def cluster_index(ids: list[str]) -> tuple[np.ndarray, list[str]]:
    uniq = sorted(set(ids))
    mp = {u: i for i, u in enumerate(uniq)}
    return np.asarray([mp[x] for x in ids], dtype=np.int32), uniq


def members_list(cluster_of_trace: np.ndarray) -> list[np.ndarray]:
    n_c = int(cluster_of_trace.max()) + 1
    return [np.where(cluster_of_trace == c)[0] for c in range(n_c)]


def bootstrap_indices(rng, members: list[np.ndarray], n_boot: int) -> list[np.ndarray]:
    n_c = len(members)
    out = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_c, n_c)
        out.append(np.concatenate([members[c] for c in chosen]))
    return out


def flatten_picks(scores_list, hits_list, n_true_arr):
    """Pre-flatten variable-length per-trace picks into contiguous arrays."""
    n = len(scores_list)
    lengths = np.array([len(np.asarray(scores_list[i])) for i in range(n)], dtype=np.int32)
    total = int(lengths.sum())
    scores = np.empty(total, dtype=np.float64)
    hits = np.empty(total, dtype=bool)
    trace_of = np.empty(total, dtype=np.int32)
    cursor = 0
    for i in range(n):
        s = np.asarray(scores_list[i], dtype=np.float64)
        h = np.asarray(hits_list[i], dtype=bool)
        k = s.size
        if k:
            scores[cursor : cursor + k] = s
            hits[cursor : cursor + k] = h
            trace_of[cursor : cursor + k] = i
            cursor += k
    return {
        "scores": scores[:cursor],
        "hits": hits[:cursor],
        "trace_of": trace_of[:cursor],
        "n_true": np.asarray(n_true_arr, dtype=np.int32),
    }


def ap_from_flat(flat: dict, selected_traces: np.ndarray) -> float:
    """AP over picks belonging to selected traces (with replacement multiplicity)."""
    # multiplicity of each trace in the bootstrap sample
    n = int(flat["n_true"].shape[0])
    mult = np.bincount(selected_traces, minlength=n)
    n_true = int((flat["n_true"] * mult).sum())
    if n_true == 0:
        return 0.0
    to = flat["trace_of"]
    if to.size == 0:
        return 0.0
    m = mult[to]
    keep = m > 0
    if not np.any(keep):
        return 0.0
    # Expand picks by multiplicity via repeats (rare; usually mult is 0/1)
    s = flat["scores"][keep]
    h = flat["hits"][keep]
    m_keep = m[keep]
    if m_keep.max() > 1:
        s = np.repeat(s, m_keep)
        h = np.repeat(h, m_keep)
    order = np.argsort(-s)
    h = h[order]
    tp = np.cumsum(h)
    prec = tp / np.arange(1, len(h) + 1)
    rec = tp / n_true
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def ae_block(args, rng) -> dict:
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    tols = [int(t) for t in args.tols.split(",")]
    report = {
        "method": "cluster_bootstrap",
        "n_boot_f1": args.n_boot,
        "n_boot_ap": args.n_boot_ap,
        "clusterings_f1": ["trace", "event", "sensor"],
        "clusterings_ap": ["event"],
        "materials": {},
    }
    for mat in ("Minusil_test", "Glass_beads_test"):
        print(f"[ae] loading {mat}", flush=True)
        names = load_ae_names(args.ae_data, mat)
        event_of, event_uniq = cluster_index([parse_ae_name(n)[0] for n in names])
        sensor_of, sensor_uniq = cluster_index([parse_ae_name(n)[1] for n in names])
        data = {
            m: {s: pickle.load(open(args.ae_root / m / f"zs_s{s}_{mat}_per_trace.pkl", "rb")) for s in seeds}
            for m in models
        }
        n = len(names)
        for s in seeds:
            for m in models:
                assert len(data[m][s]["n_true"]) == n

        members = {
            "event": members_list(event_of),
            "sensor": members_list(sensor_of),
        }
        boot_idx = {
            "trace": {s: [rng.integers(0, n, n) for _ in range(args.n_boot)] for s in seeds},
            "event": {s: bootstrap_indices(rng, members["event"], args.n_boot) for s in seeds},
            "sensor": {s: bootstrap_indices(rng, members["sensor"], args.n_boot) for s in seeds},
        }
        boot_idx_ap = {s: idxs[: args.n_boot_ap] for s, idxs in boot_idx["event"].items()}

        mat_rep = {
            "n_traces": n,
            "n_events": len(event_uniq),
            "n_sensors": len(sensor_uniq),
            "tolerances": {},
        }
        for t in tols:
            print(f"[ae] {mat} tol{t} F1 boots...", flush=True)
            # Pre-flatten AP arrays once per model/seed/tol
            flat = {
                m: {s: flatten_picks(data[m][s]["scores"], data[m][s]["hits"][t], data[m][s]["n_true"]) for s in seeds}
                for m in models
            }
            point = {
                m: {
                    "f1": [f1_from_counts(data[m][s]["counts"][t]) for s in seeds],
                    "ap": [ap_from_flat(flat[m][s], np.arange(n)) for s in seeds],
                }
                for m in models
            }
            tol_row = {"models": {}, "diffs_ref_minus_baseline": {}}

            for scheme in ("trace", "event", "sensor"):
                bs_f1 = {m: np.zeros((args.n_boot, len(seeds))) for m in models}
                for j, s in enumerate(seeds):
                    for b, idx in enumerate(boot_idx[scheme][s]):
                        for m in models:
                            bs_f1[m][b, j] = f1_from_counts(data[m][s]["counts"][t][idx])
                for m in models:
                    tol_row["models"].setdefault(
                        m,
                        {
                            "point": {
                                "f1_mean": float(np.mean(point[m]["f1"])),
                                "ap_mean": float(np.mean(point[m]["ap"])),
                                "f1_sd_seeds": float(np.std(point[m]["f1"])),
                                "ap_sd_seeds": float(np.std(point[m]["ap"])),
                            }
                        },
                    )
                    dist = bs_f1[m].mean(1)
                    tol_row["models"][m][f"ci95_{scheme}_f1"] = [
                        float(np.percentile(dist, 2.5)),
                        float(np.percentile(dist, 97.5)),
                    ]
                for m in models:
                    if m == args.ref:
                        continue
                    key = f"{args.ref}_minus_{m}"
                    tol_row["diffs_ref_minus_baseline"].setdefault(key, {})
                    d = (bs_f1[args.ref] - bs_f1[m]).mean(1)
                    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                    tol_row["diffs_ref_minus_baseline"][key][f"{scheme}_f1"] = {
                        "mean_point": float(np.mean(point[args.ref]["f1"]) - np.mean(point[m]["f1"])),
                        "ci95": [lo, hi],
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "frac_leq_zero": float((d <= 0).mean()),
                    }

            print(f"[ae] {mat} tol{t} AP boots event-only (n={args.n_boot_ap})...", flush=True)
            bs_ap = {m: np.zeros((args.n_boot_ap, len(seeds))) for m in models}
            for j, s in enumerate(seeds):
                for b, idx in enumerate(boot_idx_ap[s]):
                    for m in models:
                        bs_ap[m][b, j] = ap_from_flat(flat[m][s], idx)
            for m in models:
                dist = bs_ap[m].mean(1)
                tol_row["models"][m]["ci95_event_ap"] = [
                    float(np.percentile(dist, 2.5)),
                    float(np.percentile(dist, 97.5)),
                ]
            for m in models:
                if m == args.ref:
                    continue
                key = f"{args.ref}_minus_{m}"
                d = (bs_ap[args.ref] - bs_ap[m]).mean(1)
                lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                tol_row["diffs_ref_minus_baseline"][key]["event_ap"] = {
                    "mean_point": float(np.mean(point[args.ref]["ap"]) - np.mean(point[m]["ap"])),
                    "ci95": [lo, hi],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "frac_leq_zero": float((d <= 0).mean()),
                }

            mat_rep["tolerances"][f"tol{t}"] = tol_row
            dkey = f"{args.ref}_minus_pn_aug"
            if dkey in tol_row["diffs_ref_minus_baseline"]:
                evf = tol_row["diffs_ref_minus_baseline"][dkey]["event_f1"]
                trf = tol_row["diffs_ref_minus_baseline"][dkey]["trace_f1"]
                eva = tol_row["diffs_ref_minus_baseline"][dkey]["event_ap"]
                print(
                    f"[{mat} tol{t}] ΔF1 point={evf['mean_point']:+.3f} "
                    f"traceCI={trf['ci95']} excl0={trf['excludes_zero']} | "
                    f"eventCI={evf['ci95']} excl0={evf['excludes_zero']} | "
                    f"eventAP_CI={eva['ci95']} excl0={eva['excludes_zero']}",
                    flush=True,
                )
        report["materials"][mat] = mat_rep
    return report


def headline_block(args, rng) -> dict:
    out = {"method": "cluster_bootstrap_on_dumps", "sets": {}}
    for name in ("stead_full", "ethz_test", "obs_test"):
        path = args.headline_dir / f"{name}_hits.json"
        if not path.exists():
            out["sets"][name] = {"status": "MISSING_DUMP", "path": str(path)}
            continue
        d = json.loads(path.read_text())
        clusters = np.asarray(d["cluster_id"], dtype=np.int32)
        members = members_list(clusters)
        boots = bootstrap_indices(rng, members, args.n_boot)
        set_rep = {"source": d.get("source"), "n": d["n"], "n_clusters": d.get("n_clusters"), "models": {}}
        for model, md in d["models"].items():
            set_rep["models"][model] = {}
            for phase in ("p", "s"):
                for tol_key, arr in md[phase].items():
                    hit = np.asarray(arr["hit"], dtype=bool)
                    valid = np.asarray(arr["valid"], dtype=bool)
                    pred = np.asarray(arr["pred"], dtype=bool)
                    counts = np.stack(
                        [
                            (hit & valid & pred).astype(np.int32),
                            (pred & ~hit).astype(np.int32),
                            (valid & ~hit).astype(np.int32),
                        ],
                        axis=1,
                    )
                    point = f1_from_counts(counts)
                    dist = np.array([f1_from_counts(counts[idx]) for idx in boots])
                    ci = [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]
                    ms = None
                    if "manuscript_refs" in d:
                        ms = d["manuscript_refs"].get(f"{model}_{phase}_{tol_key}")
                    set_rep["models"][model][f"{phase}@{tol_key}"] = {
                        "f1_point": point,
                        "ci95_cluster": ci,
                        "manuscript_f1": ms,
                        "diff_vs_manuscript": (None if ms is None else point - float(ms)),
                    }
                    print(f"[headline {name}] {model} {phase}@{tol_key} F1={point:.4f} CI={ci} MS={ms}", flush=True)
        out["sets"][name] = set_rep
    return out


def write_summary(ae: dict, hl: dict, path: Path):
    lines = ["# Phase 2.7 cluster-bootstrap CI summary", ""]
    lines.append("## AE (primary): event-cluster vs paired-trace")
    lines.append("")
    lines.append("| Material | Tol | ΔF1 ours−pn_aug | Trace CI excl0? | Event CI excl0? | ΔAP event CI excl0? |")
    lines.append("|---|---:|---:|---|---|---|")
    for mat, mr in ae["materials"].items():
        for tol, tr in mr["tolerances"].items():
            key = "ours_minus_pn_aug"
            d = tr["diffs_ref_minus_baseline"][key]
            lines.append(
                f"| {mat} | {tol} | {d['event_f1']['mean_point']:+.3f} | "
                f"{d['trace_f1']['excludes_zero']} {d['trace_f1']['ci95']} | "
                f"{d['event_f1']['excludes_zero']} {d['event_f1']['ci95']} | "
                f"{d['event_ap']['excludes_zero']} {d['event_ap']['ci95']} |"
            )
    lines.append("")
    lines.append("## Headline tables")
    for name, sr in hl["sets"].items():
        if sr.get("status") == "MISSING_DUMP":
            lines.append(f"- **{name}**: MISSING dump (`{sr['path']}`)")
        else:
            lines.append(f"- **{name}**: n={sr['n']} clusters={sr.get('n_clusters')}")
            for model, md in sr["models"].items():
                for k, v in md.items():
                    lines.append(
                        f"  - {model} {k}: F1={v['f1_point']:.4f} clusterCI={v['ci95_cluster']} "
                        f"manuscript={v.get('manuscript_f1')} Δ={v.get('diff_vs_manuscript')}"
                    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ae = ae_block(args, rng)
    (args.output_dir / "ae_cluster_bootstrap.json").write_text(json.dumps(ae, indent=2))
    hl = headline_block(args, rng)
    (args.output_dir / "headline_cluster_bootstrap.json").write_text(json.dumps(hl, indent=2))
    write_summary(ae, hl, args.output_dir / "SUMMARY.md")
    print(f"[done] -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
