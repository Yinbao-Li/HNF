#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dig into two physics questions with causal-chain + summary features.

1. Can magnitude be predicted from the chain / summary (beyond distance)?
2. Do chain modes associate with geography? Do residuals suggest path/structure?

Uses STEAD metadata (source_magnitude, lat/lon, depth, distance) joined by
trace_name. CPU-friendly.

Example:
  PYTHONPATH=. python tools/analyze_causal_physics.py \\
    --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt \\
    --split val --max-event 500 --device cpu \\
    --output-dir outputs/causal_physics_run28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.causal_chain import (
    CAUSAL_SHAPE_NAMES,
    causal_chain_features,
    extract_causal_observables,
    features_to_vector,
    has_valid_chain,
)
from hnf.pattern_library import FEATURE_NAMES, _kmeans, extract_pattern_features, features_to_vector as summary_to_vector
from hnf.stead_picking_dataset import STEAD_DIR, STEADPickingDataset
from tools.analyze_stead_picking import load_model
from tools.train_stead_picking import move_batch_to_device


META_COLS = [
    "trace_name",
    "source_magnitude",
    "source_magnitude_type",
    "source_latitude",
    "source_longitude",
    "source_depth_km",
    "source_distance_km",
    "receiver_latitude",
    "receiver_longitude",
    "network_code",
    "receiver_code",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal physics: magnitude + geography")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/causal_physics_run28")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max-event", type=int, default=500)
    p.add_argument("--max-noise", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_stead_meta() -> pd.DataFrame:
    frames = []
    for chunk in range(1, 7):
        csv_path = STEAD_DIR / f"chunk{chunk}_eofextract" / f"chunk{chunk}.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path, usecols=META_COLS, low_memory=False)
        df["chunk"] = chunk
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta = meta.drop_duplicates(subset=["trace_name"], keep="first")
    return meta.set_index("trace_name")


def ridge_fit_predict(
    x: np.ndarray, y: np.ndarray, *, lam: float = 1.0, seed: int = 0
) -> dict[str, float]:
    """5-fold ridge regression; returns mean R² / MAE / RMSE on held-out folds."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    r2s, maes, rmses = [], [], []
    for fi in range(5):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(5) if j != fi])
        xtr, ytr = x[tr], y[tr]
        xte, yte = x[te], y[te]
        mu = xtr.mean(0)
        sd = xtr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        ztr = (xtr - mu) / sd
        zte = (xte - mu) / sd
        # augment with bias column
        ztr = np.concatenate([ztr, np.ones((ztr.shape[0], 1))], axis=1)
        zte = np.concatenate([zte, np.ones((zte.shape[0], 1))], axis=1)
        a = ztr.T @ ztr + lam * np.eye(ztr.shape[1])
        a[-1, -1] = 0.0  # don't regularise bias
        w = np.linalg.solve(a, ztr.T @ ytr)
        pred = zte @ w
        ss_res = float(((yte - pred) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
        r2s.append(1.0 - ss_res / ss_tot)
        maes.append(float(np.abs(yte - pred).mean()))
        rmses.append(float(np.sqrt(((yte - pred) ** 2).mean())))
    return {
        "r2": round(float(np.mean(r2s)), 3),
        "r2_std": round(float(np.std(r2s)), 3),
        "mae": round(float(np.mean(maes)), 3),
        "rmse": round(float(np.mean(rmses)), 3),
        "n": int(n),
    }


def _merge_small(z: np.ndarray, labels: np.ndarray, min_frac: float = 0.03) -> np.ndarray:
    n = z.shape[0]
    min_n = max(3, int(round(min_frac * n)))
    labels = labels.copy()
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) <= 1:
            break
        small = uniq[counts < min_n]
        if small.size == 0:
            break
        big = uniq[counts >= min_n]
        if big.size == 0:
            labels[:] = uniq[counts.argmax()]
            break
        cents = {int(c): z[labels == c].mean(0) for c in big}
        victim = int(small[0])
        for i in np.where(labels == victim)[0]:
            d = {c: np.linalg.norm(z[i] - v) for c, v in cents.items()}
            labels[i] = min(d, key=d.get)
    remap = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    return np.asarray([remap[int(v)] for v in labels], dtype=np.int64)


def region_key(lat: float, lon: float, step: float = 5.0) -> str:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return "unknown"
    la = int(np.floor(lat / step) * step)
    lo = int(np.floor(lon / step) * step)
    return f"{la:+d}/{lo:+d}"


def chi2_contingency(table: np.ndarray) -> tuple[float, float]:
    """Return (chi2, Cramer's V) without scipy."""
    table = table.astype(np.float64)
    if table.size == 0 or table.sum() == 0:
        return 0.0, 0.0
    row = table.sum(1, keepdims=True)
    col = table.sum(0, keepdims=True)
    exp = row @ col / table.sum()
    mask = exp > 0
    chi2 = float(((table[mask] - exp[mask]) ** 2 / exp[mask]).sum())
    n = float(table.sum())
    r, c = table.shape
    v = float(np.sqrt(chi2 / (n * max(min(r - 1, c - 1), 1))))
    return chi2, v


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("[physics] loading STEAD metadata…", flush=True)
    meta = load_stead_meta()
    print(f"[physics] meta rows={len(meta)}", flush=True)

    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    ds = STEADPickingDataset(
        args.split,
        seq_len=args.seq_len,
        max_event_traces=args.max_event,
        max_noise_traces=args.max_noise,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows = []
    causal_vecs = []
    summary_vecs = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            name = batch["trace_name"][0] if isinstance(batch["trace_name"], (list, tuple)) else str(batch["trace_name"][0])
            if float(batch["det"][0].item()) <= 0.5:
                continue
            x = batch["x"]
            t = batch["t"][0] if batch["t"].dim() == 3 else batch["t"]
            obs = extract_causal_observables(
                model, x, t, pick_threshold=args.pick_threshold, is_event=True
            )
            if not has_valid_chain(obs):
                continue
            summ = extract_pattern_features(model, x, t, pick_threshold=args.pick_threshold)
            if name not in meta.index:
                continue
            m = meta.loc[name]
            mag = float(m["source_magnitude"]) if pd.notna(m["source_magnitude"]) else float("nan")
            row = {
                "trace_name": name,
                "mag": mag,
                "mag_type": str(m["source_magnitude_type"]) if pd.notna(m["source_magnitude_type"]) else "",
                "src_lat": float(m["source_latitude"]) if pd.notna(m["source_latitude"]) else float("nan"),
                "src_lon": float(m["source_longitude"]) if pd.notna(m["source_longitude"]) else float("nan"),
                "rcv_lat": float(m["receiver_latitude"]) if pd.notna(m["receiver_latitude"]) else float("nan"),
                "rcv_lon": float(m["receiver_longitude"]) if pd.notna(m["receiver_longitude"]) else float("nan"),
                "depth_km": float(m["source_depth_km"]) if pd.notna(m["source_depth_km"]) else float("nan"),
                "dist_km": float(m["source_distance_km"]) if pd.notna(m["source_distance_km"]) else float("nan"),
                "network": str(m["network_code"]) if pd.notna(m["network_code"]) else "",
                "station": str(m["receiver_code"]) if pd.notna(m["receiver_code"]) else "",
                "ps_gap": obs.ps_gap_sec,
                "det": summ["det"],
                "p_peak": summ["p_peak"],
                "s_peak": summ["s_peak"],
                "coda_slope": causal_chain_features(obs)["coda_slope"],
                "onset_sharp": causal_chain_features(obs)["onset_sharp"],
                "n_rho_peaks": causal_chain_features(obs)["n_rho_peaks"],
                "log_sp_amp_ratio": causal_chain_features(obs)["log_sp_amp_ratio"],
                "pre_p_energy_frac": causal_chain_features(obs)["pre_p_energy_frac"],
                "log_wave_energy": float(np.log10(obs.wave_env.mean() + 1e-12)),
            }
            # midpath for geographic path effects
            if np.isfinite(row["src_lat"]) and np.isfinite(row["rcv_lat"]):
                row["mid_lat"] = 0.5 * (row["src_lat"] + row["rcv_lat"])
                row["mid_lon"] = 0.5 * (row["src_lon"] + row["rcv_lon"])
            else:
                row["mid_lat"] = row["mid_lon"] = float("nan")
            rows.append(row)
            causal_vecs.append(features_to_vector(causal_chain_features(obs)))
            summary_vecs.append(summary_to_vector(summ))
            if (bi + 1) % 50 == 0:
                print(f"[physics] {bi+1}/{len(ds)} kept={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    C = np.stack(causal_vecs)
    S = np.stack(summary_vecs)
    print(f"[physics] kept {len(df)} event chains in {time.time()-t0:.0f}s", flush=True)

    # ---- cluster causal modes (same recipe as v2) ----
    med = np.median(C, axis=0)
    mad = np.median(np.abs(C - med), axis=0)
    scale = np.where(mad < 1e-9, 1.0, 1.4826 * mad)
    zc = np.clip((C - med) / scale, -5.0, 5.0)
    labels, _ = _kmeans(zc, k=args.k, seed=args.seed)
    labels = _merge_small(zc, labels)
    df["mode"] = labels

    # ---- Q1: magnitude prediction ----
    mag_ok = df["mag"].notna() & np.isfinite(df["mag"].to_numpy())
    dist_ok = df["dist_km"].notna() & (df["dist_km"] > 0)
    use = mag_ok & dist_ok
    y = df.loc[use, "mag"].to_numpy(dtype=np.float64)
    dist = df.loc[use, "dist_km"].to_numpy(dtype=np.float64)
    # classical single-station baseline: log-amplitude + log-distance (Richter-like)
    log_e = df.loc[use, "log_wave_energy"].to_numpy(dtype=np.float64)
    log_d = np.log10(dist + 1.0)
    gap = df.loc[use, "ps_gap"].to_numpy(dtype=np.float64)
    depth = df.loc[use, "depth_km"].fillna(0).to_numpy(dtype=np.float64)

    scalar_causal = df.loc[use, [
        "coda_slope", "onset_sharp", "n_rho_peaks", "log_sp_amp_ratio", "pre_p_energy_frac"
    ]].to_numpy(dtype=np.float64)
    scalar_summary = df.loc[use, ["det", "p_peak", "s_peak", "ps_gap"]].to_numpy(dtype=np.float64)
    C_use = C[use.to_numpy()]
    S_use = S[use.to_numpy()]
    mode_oh = np.eye(int(labels.max()) + 1, dtype=np.float64)[labels[use.to_numpy()]]

    mag_report = {
        "baseline_logE_logD": ridge_fit_predict(np.column_stack([log_e, log_d]), y, seed=args.seed),
        "distance_gap_only": ridge_fit_predict(np.column_stack([log_d, gap, depth]), y, seed=args.seed),
        "summary_scalars": ridge_fit_predict(
            np.column_stack([log_e, log_d, scalar_summary]), y, seed=args.seed
        ),
        "causal_scalars": ridge_fit_predict(
            np.column_stack([log_e, log_d, scalar_causal]), y, seed=args.seed
        ),
        "causal_full_shape": ridge_fit_predict(
            np.column_stack([log_e, log_d, C_use]), y, seed=args.seed
        ),
        "summary_full": ridge_fit_predict(
            np.column_stack([log_e, log_d, S_use]), y, seed=args.seed
        ),
        "causal_plus_summary": ridge_fit_predict(
            np.column_stack([log_e, log_d, C_use, S_use]), y, seed=args.seed
        ),
        "mode_onehot_plus_logE_logD": ridge_fit_predict(
            np.column_stack([log_e, log_d, mode_oh]), y, seed=args.seed
        ),
    }
    # per-mode magnitude stats
    mode_mag = {}
    for j in sorted(df["mode"].unique()):
        sub = df[(df["mode"] == j) & use]
        if len(sub) == 0:
            continue
        mode_mag[f"mode{j}"] = {
            "n": int(len(sub)),
            "mag_mean": round(float(sub["mag"].mean()), 2),
            "mag_std": round(float(sub["mag"].std()), 2),
            "dist_mean_km": round(float(sub["dist_km"].mean()), 1),
            "coda_slope": round(float(sub["coda_slope"].mean()), 3),
            "onset_sharp": round(float(sub["onset_sharp"].mean()), 3),
        }

    # ---- Q2: geography ----
    df["src_region"] = [region_key(a, b, 5.0) for a, b in zip(df["src_lat"], df["src_lon"])]
    df["path_region"] = [region_key(a, b, 5.0) for a, b in zip(df["mid_lat"], df["mid_lon"])]
    df["rcv_region"] = [region_key(a, b, 5.0) for a, b in zip(df["rcv_lat"], df["rcv_lon"])]

    def region_table(col: str, top_n: int = 12) -> dict:
        vc = df[col].value_counts()
        keep = list(vc.head(top_n).index)
        modes = sorted(df["mode"].unique())
        table = np.zeros((len(keep), len(modes)), dtype=np.int64)
        for i, r in enumerate(keep):
            for j, m in enumerate(modes):
                table[i, j] = int(((df[col] == r) & (df["mode"] == m)).sum())
        chi2, v = chi2_contingency(table)
        # mode fractions per region
        frac = {}
        for i, r in enumerate(keep):
            tot = max(int(table[i].sum()), 1)
            frac[r] = {
                f"mode{m}": round(float(table[i, j]) / tot, 3)
                for j, m in enumerate(modes)
            }
            frac[r]["n"] = int(table[i].sum())
        return {
            "chi2": round(chi2, 1),
            "cramers_v": round(v, 3),
            "mode_fractions": frac,
        }

    geo_report = {
        "source_region_5deg": region_table("src_region"),
        "path_midpoint_5deg": region_table("path_region"),
        "receiver_region_5deg": region_table("rcv_region"),
    }

    # Distance-controlled geographic residual: within a distance bin, does mode
    # still vary by region? If yes, that points to path/structure, not just range.
    dist_bins = pd.cut(df["dist_km"], bins=[0, 50, 150, 400, 2000], labels=["0-50", "50-150", "150-400", "400+"])
    df["dist_bin"] = dist_bins.astype(str)
    controlled = {}
    for b in ["0-50", "50-150", "150-400", "400+"]:
        sub = df[df["dist_bin"] == b]
        if len(sub) < 30:
            continue
        # rebuild tiny table for this bin
        vc = sub["path_region"].value_counts()
        keep = list(vc.head(8).index)
        modes = sorted(sub["mode"].unique())
        table = np.zeros((len(keep), len(modes)), dtype=np.int64)
        for i, r in enumerate(keep):
            for j, m in enumerate(modes):
                table[i, j] = int(((sub["path_region"] == r) & (sub["mode"] == m)).sum())
        chi2, v = chi2_contingency(table)
        controlled[b] = {"n": int(len(sub)), "chi2": round(chi2, 1), "cramers_v": round(v, 3)}

    # Anomaly mode candidates: modes with extreme coda/onset relative to peers
    # at similar distance — possible structure/path signatures
    anomalies = []
    for j in sorted(df["mode"].unique()):
        sub = df[df["mode"] == j]
        peers = df[df["mode"] != j]
        # compare coda within overlapping distance
        dlo, dhi = sub["dist_km"].quantile(0.25), sub["dist_km"].quantile(0.75)
        peer_same = peers[(peers["dist_km"] >= dlo) & (peers["dist_km"] <= dhi)]
        if len(sub) < 10 or len(peer_same) < 10:
            continue
        coda_delta = float(sub["coda_slope"].mean() - peer_same["coda_slope"].mean())
        onset_delta = float(sub["onset_sharp"].mean() - peer_same["onset_sharp"].mean())
        top_regions = sub["path_region"].value_counts().head(5).to_dict()
        anomalies.append(
            {
                "mode": int(j),
                "n": int(len(sub)),
                "coda_delta_vs_peers": round(coda_delta, 3),
                "onset_delta_vs_peers": round(onset_delta, 3),
                "mag_mean": round(float(sub["mag"].mean()), 2) if sub["mag"].notna().any() else None,
                "dist_mean_km": round(float(sub["dist_km"].mean()), 1),
                "top_path_regions": {str(k): int(v) for k, v in top_regions.items()},
                "interpretation": (
                    "faster coda than peers at same distance → higher path attenuation / lower Q"
                    if coda_delta < -0.03
                    else (
                        "slower coda than peers → scattering-rich / high-Q path"
                        if coda_delta > 0.03
                        else "coda similar to peers; look at onset / multipath instead"
                    )
                ),
            }
        )

    report = {
        "n_traces": int(len(df)),
        "elapsed_sec": round(time.time() - t0, 1),
        "magnitude_prediction": mag_report,
        "magnitude_by_mode": mode_mag,
        "geography": geo_report,
        "geography_distance_controlled": controlled,
        "structure_anomaly_candidates": anomalies,
        "reading": {
            "magnitude": (
                "Baseline is Richter-like (log energy + log distance). "
                "If causal/summary lift R² beyond that, the chain carries source info "
                "(not just path length)."
            ),
            "geography": (
                "Cramer's V measures mode↔region association (0=none, >0.2 moderate). "
                "If V stays high inside a fixed distance bin, modes track path/structure, "
                "not just epicentral range."
            ),
        },
    }
    (out / "physics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    df.to_csv(out / "traces.csv", index=False)
    np.savez(out / "features.npz", causal=C, summary=S, labels=labels)

    # ---- plots ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        # mag vs mode
        ax = axes[0]
        modes = sorted(df["mode"].unique())
        data = [df.loc[(df["mode"] == m) & use, "mag"].dropna().to_numpy() for m in modes]
        ax.boxplot(data, labels=[f"m{m}" for m in modes], showfliers=False)
        ax.set_ylabel("magnitude")
        ax.set_title("Magnitude by causal mode")

        # mag residual of baseline vs causal coda
        ax = axes[1]
        # fit simple baseline on all use rows
        xb = np.column_stack([log_e, log_d, np.ones(len(y))])
        w = np.linalg.lstsq(xb, y, rcond=None)[0]
        resid = y - xb @ w
        coda = df.loc[use, "coda_slope"].to_numpy()
        ax.scatter(coda, resid, s=8, alpha=0.45, c=labels[use.to_numpy()], cmap="tab10")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("coda_slope (1/s)")
        ax.set_ylabel("mag residual (vs logE+logD)")
        ax.set_title("Does coda explain leftover magnitude?")

        # map: source locations coloured by mode
        ax = axes[2]
        for m in modes:
            sub = df[df["mode"] == m]
            ax.scatter(sub["src_lon"], sub["src_lat"], s=10, alpha=0.55, label=f"m{m}")
        ax.set_xlabel("source lon")
        ax.set_ylabel("source lat")
        ax.set_title("Sources by causal mode")
        ax.legend(fontsize=7, markerscale=1.5)

        fig.tight_layout()
        fig.savefig(out / "physics_overview.png", dpi=130)
        print(f"[physics] wrote {out / 'physics_overview.png'}", flush=True)
    except Exception as exc:
        print(f"[physics] plot skipped: {exc}", flush=True)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
