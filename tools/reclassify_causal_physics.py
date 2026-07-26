#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-classify events with interpretable HNF features; raise mag + geo resolution.

Upgrades vs the first causal_physics pass:
  1. Recover RAW STEAD amplitude (training normalises it away) → true Richter-like
     ``reduced_amp = log10(A) + log10(D)``.
  2. Cluster on a compact *named* feature set (chain shape + amplitude + model
     confidence), not the 101-d trajectory — modes stay readable.
  3. Auto-name each mode from its physical centroid
     (e.g. ``impulsive_fastQ_weak_near``).
  4. Re-evaluate magnitude R² and geography Cramér's V under the new taxonomy;
     add path-residual (coda after regressing out distance) as a structure proxy.

CPU-friendly. Example:
  PYTHONPATH=. python tools/reclassify_causal_physics.py \\
    --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt \\
    --split val --max-event 600 --k 6 --device cpu \\
    --output-dir outputs/causal_reclass_run28
"""

from __future__ import annotations

import argparse
import ast
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
    INTERPRETABLE_NAMES,
    causal_chain_features,
    extract_causal_observables,
    has_valid_chain,
    interpretable_feature_dict,
    interpretable_to_vector,
    name_causal_mode,
    raw_amplitude_features,
)
from hnf.kernel_response import extract_kernel_response_features
from hnf.pattern_library import _kmeans, extract_pattern_features
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
    "snr_db",
    "network_code",
    "receiver_code",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interpretable causal reclassification")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/causal_reclass_run28")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max-event", type=int, default=600)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_meta() -> pd.DataFrame:
    frames = []
    for chunk in range(1, 7):
        p = STEAD_DIR / f"chunk{chunk}_eofextract" / f"chunk{chunk}.csv"
        if p.is_file():
            frames.append(pd.read_csv(p, usecols=META_COLS, low_memory=False))
    return pd.concat(frames).drop_duplicates("trace_name").set_index("trace_name")


def parse_snr(v) -> float:
    if isinstance(v, (float, int)) and np.isfinite(v):
        return float(v)
    if isinstance(v, str):
        try:
            arr = np.asarray(ast.literal_eval(v.replace(" ", ",")), dtype=float)
            return float(np.nanmean(arr))
        except Exception:
            try:
                return float(np.nanmean(np.fromstring(v.strip("[]"), sep=" ")))
            except Exception:
                return float("nan")
    return float("nan")


def ridge(x: np.ndarray, y: np.ndarray, *, lam: float = 10.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    r2s, maes = [], []
    for fi in range(5):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(5) if j != fi])
        xtr, ytr = x[tr], y[tr]
        xte, yte = x[te], y[te]
        good = np.isfinite(xtr).all(0) & (np.nanstd(xtr, 0) > 1e-12)
        if good.sum() == 0:
            r2s.append(0.0)
            maes.append(float("nan"))
            continue
        xtr, xte = xtr[:, good], xte[:, good]
        mu, sd = xtr.mean(0), xtr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        ztr, zte = (xtr - mu) / sd, (xte - mu) / sd
        u, s, vt = np.linalg.svd(ztr, full_matrices=False)
        d = s / (s * s + lam)
        w = (vt.T * d) @ (u.T @ ytr)
        bias = ytr.mean() - ztr.mean(0) @ w
        pred = zte @ w + bias
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-12
        r2s.append(1.0 - float(ss_res / ss_tot))
        maes.append(float(np.abs(yte - pred).mean()))
    return {
        "r2": round(float(np.mean(r2s)), 3),
        "r2_std": round(float(np.std(r2s)), 3),
        "mae": round(float(np.mean(maes)), 3),
        "n": int(n),
    }


def robust_z(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    scale = np.where(mad < 1e-9, 1.0, 1.4826 * mad)
    z = np.clip((x - med) / scale, -5.0, 5.0)
    return z, med, scale


def merge_small(z: np.ndarray, labels: np.ndarray, min_frac: float = 0.04) -> np.ndarray:
    n = z.shape[0]
    min_n = max(4, int(round(min_frac * n)))
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
    remap = {o: n for n, o in enumerate(sorted(np.unique(labels)))}
    return np.asarray([remap[int(v)] for v in labels], dtype=np.int64)


def region_key(lat: float, lon: float, step: float = 5.0) -> str:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return "unknown"
    return f"{int(np.floor(lat / step) * step):+d}/{int(np.floor(lon / step) * step):+d}"


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    table = table.astype(np.float64)
    if table.sum() == 0:
        return 0.0, 0.0
    row, col = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
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

    meta = load_meta()
    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    ds = STEADPickingDataset(
        args.split,
        seq_len=args.seq_len,
        max_event_traces=args.max_event,
        max_noise_traces=0,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Feature subsets for taxonomies
    shape_idx = [INTERPRETABLE_NAMES.index(n) for n in (
        "coda_slope", "onset_sharp", "n_rho_peaks", "log_sp_amp_ratio", "pre_p_energy_frac"
    )]
    kernel_idx = [INTERPRETABLE_NAMES.index(n) for n in (
        "coda_slope", "onset_sharp", "n_rho_peaks", "log_sp_amp_ratio", "pre_p_energy_frac",
        "p_kern_mean_lag_sec", "p_kern_spread_sec", "p_kern_entropy",
        "s_kern_mean_lag_sec", "s_kern_spread_sec", "s_kern_entropy", "ps_kern_lag_ratio",
    )]
    full_idx = list(range(len(INTERPRETABLE_NAMES)))

    rows = []
    vecs = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            ref = ds.refs[bi]
            if not ref.is_event:
                continue
            batch = move_batch_to_device(batch, device)
            name = batch["trace_name"][0]
            if isinstance(name, bytes):
                name = name.decode()
            name = str(name)
            if name not in meta.index:
                continue

            x = batch["x"]
            t = batch["t"][0] if batch["t"].dim() == 3 else batch["t"]
            obs = extract_causal_observables(
                model, x, t, pick_threshold=args.pick_threshold, is_event=True
            )
            if not has_valid_chain(obs):
                continue

            # RAW waveform from HDF5 (before dataset normalisation)
            raw = np.asarray(ds._get_handle(ref.chunk)["data"][ref.trace_name][()], dtype=np.float64)
            mrow = meta.loc[name]
            dist = float(mrow["source_distance_km"]) if pd.notna(mrow["source_distance_km"]) else float("nan")
            amp = raw_amplitude_features(
                raw,
                p_sample=ref.p_sample,
                s_sample=ref.s_sample,
                dist_km=dist if np.isfinite(dist) else 1.0,
            )
            chain = causal_chain_features(obs)
            summ = extract_pattern_features(model, x, t, pick_threshold=args.pick_threshold)
            # map pick seconds → bins on the model grid
            seq = int(x.size(1))
            p_idx = int(round(obs.p_sec / 60.0 * (seq - 1))) if obs.p_sec >= 0 else -1
            s_idx = int(round(obs.s_sec / 60.0 * (seq - 1))) if obs.s_sec >= 0 else -1
            kern = extract_kernel_response_features(
                model, x, t, p_idx=p_idx, s_idx=s_idx, bypass_noise_cancel=True
            )
            feat = interpretable_feature_dict(chain, amp, summ, kern)
            vecs.append(interpretable_to_vector(feat))

            rows.append({
                "trace_name": name,
                "mag": float(mrow["source_magnitude"]) if pd.notna(mrow["source_magnitude"]) else float("nan"),
                "mag_type": str(mrow["source_magnitude_type"]) if pd.notna(mrow["source_magnitude_type"]) else "",
                "src_lat": float(mrow["source_latitude"]) if pd.notna(mrow["source_latitude"]) else float("nan"),
                "src_lon": float(mrow["source_longitude"]) if pd.notna(mrow["source_longitude"]) else float("nan"),
                "rcv_lat": float(mrow["receiver_latitude"]) if pd.notna(mrow["receiver_latitude"]) else float("nan"),
                "rcv_lon": float(mrow["receiver_longitude"]) if pd.notna(mrow["receiver_longitude"]) else float("nan"),
                "depth_km": float(mrow["source_depth_km"]) if pd.notna(mrow["source_depth_km"]) else float("nan"),
                "dist_km": dist,
                "snr_db": parse_snr(mrow["snr_db"]),
                "network": str(mrow["network_code"]) if pd.notna(mrow["network_code"]) else "",
                "station": str(mrow["receiver_code"]) if pd.notna(mrow["receiver_code"]) else "",
                "ps_gap": obs.ps_gap_sec,
                **{k: feat[k] for k in INTERPRETABLE_NAMES},
            })
            if (bi + 1) % 50 == 0:
                print(f"[reclass] {bi+1}/{len(ds)} kept={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    X = np.stack(vecs)
    print(f"[reclass] kept {len(df)} in {time.time()-t0:.0f}s", flush=True)

    # ---- two taxonomies ----
    taxonomies = {
        "shape_only": shape_idx,   # path/mechanism; distance-independent
        "shape_plus_kernel": kernel_idx,  # shape + per-trace kernel-row responses
        "full_interpretable": full_idx,  # includes amplitude → better mag separation
    }
    reports = {}
    for tax_name, idxs in taxonomies.items():
        xt = X[:, idxs]
        z, _, _ = robust_z(xt)
        labels, _ = _kmeans(z, k=args.k, seed=args.seed)
        labels = merge_small(z, labels)
        col = f"mode_{tax_name}"
        df[col] = labels

        mode_info = []
        for j in sorted(np.unique(labels)):
            sub = df[df[col] == j]
            stats = {n: float(sub[n].mean()) for n in INTERPRETABLE_NAMES if n in sub}
            stats["ps_gap_sec"] = float(sub["ps_gap"].mean())
            stats["mag_mean"] = float(sub["mag"].mean()) if sub["mag"].notna().any() else float("nan")
            stats["dist_mean_km"] = float(sub["dist_km"].mean()) if sub["dist_km"].notna().any() else float("nan")
            name = name_causal_mode(stats)
            mode_info.append({
                "mode": int(j),
                "name": name,
                "n": int(len(sub)),
                "mag_mean": round(stats["mag_mean"], 2) if np.isfinite(stats["mag_mean"]) else None,
                "dist_mean_km": round(stats["dist_mean_km"], 1) if np.isfinite(stats["dist_mean_km"]) else None,
                "ps_gap": round(stats["ps_gap_sec"], 2),
                "coda_slope": round(stats["coda_slope"], 3),
                "onset_sharp": round(stats["onset_sharp"], 3),
                "n_rho_peaks": round(stats["n_rho_peaks"], 2),
                "reduced_amp": round(stats["reduced_amp"], 3),
                "log_peak_amp": round(stats["log_peak_amp"], 3),
            })
            df.loc[df[col] == j, f"name_{tax_name}"] = name
        reports[tax_name] = {"modes": mode_info}

    # Prefer full taxonomy for primary mag; shape_only for structure
    primary = "full_interpretable"
    shape_tax = "shape_only"

    # ---- magnitude ----
    use = df["mag"].notna() & df["dist_km"].notna() & (df["dist_km"] > 0) & df["log_peak_amp"].notna()
    y = df.loc[use, "mag"].to_numpy(float)
    log_a = df.loc[use, "log_peak_amp"].to_numpy(float)
    log_d = np.log10(df.loc[use, "dist_km"].to_numpy(float) + 1.0)
    reduced = df.loc[use, "reduced_amp"].to_numpy(float)
    depth = df.loc[use, "depth_km"].fillna(0).to_numpy(float)
    snr = df.loc[use, "snr_db"].fillna(0).to_numpy(float)
    shape_s = df.loc[use, ["coda_slope", "onset_sharp", "n_rho_peaks", "log_sp_amp_ratio", "pre_p_energy_frac"]].to_numpy(float)
    conf = df.loc[use, ["det", "p_peak", "s_peak"]].to_numpy(float)
    amp_s = df.loc[use, ["log_p_rms", "log_s_rms", "log_coda_rms"]].to_numpy(float)
    mode_labels = df.loc[use, f"mode_{primary}"].to_numpy(int)
    mode_oh = np.eye(int(mode_labels.max()) + 1)[mode_labels]
    shape_mode = df.loc[use, f"mode_{shape_tax}"].to_numpy(int)
    shape_oh = np.eye(int(shape_mode.max()) + 1)[shape_mode]

    mag_report = {
        "richter_logA_logD": ridge(np.c_[log_a, log_d], y, seed=args.seed),
        "richter_plus_depth": ridge(np.c_[log_a, log_d, depth], y, seed=args.seed),
        "reduced_amp_only": ridge(reduced.reshape(-1, 1), y, seed=args.seed),
        "old_baseline_snr_logD": ridge(np.c_[snr, log_d], y, seed=args.seed),
        "richter_plus_causal_shape": ridge(np.c_[log_a, log_d, shape_s], y, seed=args.seed),
        "richter_plus_amp_windows": ridge(np.c_[log_a, log_d, amp_s], y, seed=args.seed),
        "richter_plus_hnf_confidence": ridge(np.c_[log_a, log_d, conf], y, seed=args.seed),
        "richter_plus_all_interpretable": ridge(
            np.c_[log_a, log_d, depth, shape_s, amp_s, conf], y, seed=args.seed
        ),
        "richter_plus_reclass_mode": ridge(np.c_[log_a, log_d, mode_oh], y, seed=args.seed),
        "richter_plus_shape_mode": ridge(np.c_[log_a, log_d, shape_oh], y, seed=args.seed),
        "causal_shape_only_no_amp": ridge(shape_s, y, seed=args.seed),
    }
    # lift vs Richter
    base_r2 = mag_report["richter_logA_logD"]["r2"]
    mag_report["lift_vs_richter"] = {
        k: round(v["r2"] - base_r2, 3)
        for k, v in mag_report.items()
        if isinstance(v, dict) and "r2" in v and k != "richter_logA_logD"
    }

    # ---- geography / structure ----
    df["path_region"] = [
        region_key(0.5 * (a + c) if np.isfinite(a) and np.isfinite(c) else float("nan"),
                   0.5 * (b + d) if np.isfinite(b) and np.isfinite(d) else float("nan"))
        for a, b, c, d in zip(df["src_lat"], df["src_lon"], df["rcv_lat"], df["rcv_lon"])
    ]
    df["src_region"] = [region_key(a, b) for a, b in zip(df["src_lat"], df["src_lon"])]

    # coda residual after removing distance trend → path/structure proxy
    ok = df["coda_slope"].notna() & df["dist_km"].notna() & (df["dist_km"] > 0)
    ld = np.log10(df.loc[ok, "dist_km"].to_numpy(float) + 1.0)
    cs = df.loc[ok, "coda_slope"].to_numpy(float)
    # simple linear: coda ~ a + b logD
    A = np.c_[ld, np.ones(len(ld))]
    coef, _, _, _ = np.linalg.lstsq(A, cs, rcond=None)
    resid = cs - A @ coef
    df.loc[ok, "coda_path_residual"] = resid

    def geo_assoc(mode_col: str) -> dict:
        vc = df["path_region"].value_counts()
        keep = list(vc.head(12).index)
        modes = sorted(df[mode_col].unique())
        table = np.zeros((len(keep), len(modes)), dtype=np.int64)
        for i, r in enumerate(keep):
            for j, m in enumerate(modes):
                table[i, j] = int(((df["path_region"] == r) & (df[mode_col] == m)).sum())
        chi2, v = cramers_v(table)
        fracs = {}
        for i, r in enumerate(keep):
            tot = max(int(table[i].sum()), 1)
            fracs[r] = {f"mode{m}": round(float(table[i, j]) / tot, 3) for j, m in enumerate(modes)}
            fracs[r]["n"] = int(table[i].sum())
        # distance-controlled
        controlled = {}
        bins = pd.cut(df["dist_km"], bins=[0, 50, 150, 400, 5000], labels=["0-50", "50-150", "150-400", "400+"])
        for b in ["0-50", "50-150", "150-400"]:
            sub = df[bins == b]
            if len(sub) < 40:
                continue
            vc2 = sub["path_region"].value_counts()
            keep2 = list(vc2.head(8).index)
            modes2 = sorted(sub[mode_col].unique())
            t2 = np.zeros((len(keep2), len(modes2)), dtype=np.int64)
            for i, r in enumerate(keep2):
                for j, m in enumerate(modes2):
                    t2[i, j] = int(((sub["path_region"] == r) & (sub[mode_col] == m)).sum())
            c2, v2 = cramers_v(t2)
            controlled[b] = {"n": int(len(sub)), "chi2": round(c2, 1), "cramers_v": round(v2, 3)}
        return {"chi2": round(chi2, 1), "cramers_v": round(v, 3), "fractions": fracs, "distance_controlled": controlled}

    geo_report = {
        "shape_only": geo_assoc(f"mode_{shape_tax}"),
        "shape_plus_kernel": geo_assoc("mode_shape_plus_kernel"),
        "full_interpretable": geo_assoc(f"mode_{primary}"),
    }

    # structure anomaly: modes with extreme coda_path_residual + regional concentration
    anomalies = []
    for j in sorted(df[f"mode_{shape_tax}"].unique()):
        sub = df[df[f"mode_{shape_tax}"] == j]
        if len(sub) < 15 or sub["coda_path_residual"].isna().all():
            continue
        peers = df[df[f"mode_{shape_tax}"] != j]
        delta = float(sub["coda_path_residual"].mean() - peers["coda_path_residual"].mean())
        top = sub["path_region"].value_counts().head(4).to_dict()
        name = str(sub[f"name_{shape_tax}"].iloc[0])
        anomalies.append({
            "mode": int(j),
            "name": name,
            "n": int(len(sub)),
            "coda_path_residual_delta": round(delta, 3),
            "mag_mean": round(float(sub["mag"].mean()), 2),
            "dist_mean_km": round(float(sub["dist_km"].mean()), 1),
            "top_path_regions": {str(k): int(v) for k, v in top.items()},
            "interpretation": (
                "path residual more negative than peers → stronger attenuation / lower Q along typical paths"
                if delta < -0.02
                else (
                    "path residual more positive → scattering-rich / higher-Q paths"
                    if delta > 0.02
                    else "path residual near peers"
                )
            ),
        })

    report = {
        "n_traces": int(len(df)),
        "elapsed_sec": round(time.time() - t0, 1),
        "feature_names": list(INTERPRETABLE_NAMES),
        "taxonomies": reports,
        "magnitude_prediction": mag_report,
        "geography": geo_report,
        "structure_anomalies": anomalies,
        "comparison_to_previous": {
            "prev_best_mag_r2_approx": 0.658,
            "prev_geo_cramers_v": 0.366,
            "note": "Previous pass lacked raw amplitude (normalised waveforms). "
                    "This pass restores Richter logA and re-clusters on named interpretable knobs.",
        },
    }
    (out / "reclass_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    df.to_csv(out / "traces.csv", index=False)
    np.savez(out / "features.npz", X=X, names=np.asarray(INTERPRETABLE_NAMES))

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        # mag by full mode
        ax = axes[0, 0]
        modes = sorted(df[f"mode_{primary}"].unique())
        data = [df.loc[(df[f"mode_{primary}"] == m) & use, "mag"].dropna().to_numpy() for m in modes]
        labels_box = []
        for m in modes:
            nm = df.loc[df[f"mode_{primary}"] == m, f"name_{primary}"].iloc[0]
            labels_box.append(f"m{m}\n{nm[:18]}")
        ax.boxplot(data, labels=labels_box, showfliers=False)
        ax.set_ylabel("magnitude")
        ax.set_title(f"Mag by reclass mode (Richter R²={mag_report['richter_logA_logD']['r2']})")
        ax.tick_params(axis="x", labelsize=7)

        # Richter residual vs reduced_amp
        ax = axes[0, 1]
        xb = np.c_[log_a, log_d]
        mu, sd = xb.mean(0), np.where(xb.std(0) < 1e-8, 1, xb.std(0))
        zb = (xb - mu) / sd
        u, s, vt = np.linalg.svd(zb, full_matrices=False)
        d = s / (s * s + 10)
        w = (vt.T * d) @ (u.T @ y)
        bias = y.mean() - zb.mean(0) @ w
        resid_m = y - (zb @ w + bias)
        ax.scatter(reduced, resid_m, s=10, alpha=0.5, c=mode_labels, cmap="tab10")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("reduced_amp = logA + logD")
        ax.set_ylabel("mag residual vs Richter")
        ax.set_title(f"all-interp R²={mag_report['richter_plus_all_interpretable']['r2']}")

        # map by shape mode (structure)
        ax = axes[1, 0]
        for m in sorted(df[f"mode_{shape_tax}"].unique()):
            sub = df[df[f"mode_{shape_tax}"] == m]
            nm = sub[f"name_{shape_tax}"].iloc[0]
            ax.scatter(sub["src_lon"], sub["src_lat"], s=12, alpha=0.55, label=f"m{m}:{nm[:14]}")
        ax.legend(fontsize=6, loc="best")
        ax.set_title(f"Shape modes (geo V={geo_report['shape_only']['cramers_v']})")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

        # coda path residual by region (top regions)
        ax = axes[1, 1]
        top_regs = list(df["path_region"].value_counts().head(8).index)
        data = [df.loc[df["path_region"] == r, "coda_path_residual"].dropna().to_numpy() for r in top_regs]
        ax.boxplot(data, labels=top_regs, showfliers=False)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("coda path residual")
        ax.set_title("Path residual by region (structure proxy)")
        ax.tick_params(axis="x", rotation=30, labelsize=7)

        fig.tight_layout()
        fig.savefig(out / "reclass_overview.png", dpi=130)
        print(f"[reclass] wrote {out / 'reclass_overview.png'}", flush=True)
    except Exception as exc:
        print(f"[reclass] plot skipped: {exc}", flush=True)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
