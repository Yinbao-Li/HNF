#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand 5-class shape labels to regional STEAD subsets (GPU).

Selects a fixed-station panel (multi-year stations) plus stratified fill,
runs the frozen run28 Huygens model, assigns shape with *ceiling* thresholds,
and writes a traces CSV compatible with ``analyze_shape_temporal_evolution.py``.

Example
-------
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/expand_shape_labels_regional.py \\
    --regions socal,pnw,alaska --device cuda --batch-size 24 \\
    --max-per-region 12000 --output-dir outputs/shape_labels_expanded
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.causal_chain import (
    CausalObservables,
    causal_chain_features,
    has_valid_chain,
)
from hnf.stead_picking_dataset import STEAD_DIR
from tools.analyze_stead_picking import load_model
from tools.analyze_shape_temporal_evolution import REGIONS
from tools.interpretable_ceiling import assign_shape
from tools.reclassify_causal_physics import parse_snr

# Frozen thresholds from interpretable_physics_best/ceiling (keep taxonomy fixed).
DEFAULT_THRESH = {
    "peak_thr": 1.0,
    "coda_fast": -0.198,
    "coda_slow": -0.113,
    "onset_hi": 0.877,
    "onset_lo": 0.733,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expand regional shape labels on GPU")
    p.add_argument("--checkpoint", default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt")
    p.add_argument("--stead-dir", default=str(STEAD_DIR))
    p.add_argument("--regions", default="socal,pnw,alaska")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--max-per-region", type=int, default=12000)
    p.add_argument("--max-fixed-fraction", type=float, default=0.70)
    p.add_argument("--min-station-years", type=int, default=5)
    p.add_argument("--min-station-traces", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--thresholds-json",
        default="outputs/interpretable_physics_best/ceiling/ceiling_report.json",
    )
    p.add_argument("--output-dir", default="outputs/shape_labels_expanded")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def _load_thresholds(path: Path) -> dict:
    thr = dict(DEFAULT_THRESH)
    if path.is_file():
        rep = json.loads(path.read_text(encoding="utf-8"))
        st = rep.get("discrimination", {}).get("shape_thresholds", {})
        for k in thr:
            if k in st:
                thr[k] = float(st[k])
    return thr


def _parse_time(name: str) -> pd.Timestamp:
    m = re.search(r"_(\d{14})_", str(name))
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S", errors="coerce")


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size < 3:
        return x
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, k, mode="same")


def _peak_sec(prob: np.ndarray, window_sec: float, thr: float) -> tuple[float, float]:
    if prob.size == 0:
        return 0.0, -1.0
    idx = int(np.argmax(prob))
    peak = float(prob[idx])
    if peak < thr:
        return peak, -1.0
    return peak, float(idx) / max(prob.size - 1, 1) * window_sec


def observables_from_batch(
    out: dict,
    x: torch.Tensor,
    i: int,
    *,
    window_sec: float,
    pick_threshold: float,
) -> CausalObservables:
    det_p = torch.sigmoid(out["det"])
    if det_p.dim() > 1:
        det_v = float(det_p[i].amax().item())
    else:
        det_v = float(det_p[i].item())
    p_prob = torch.sigmoid(out["p"][i]).detach().float().cpu().numpy()
    s_prob = torch.sigmoid(out["s"][i]).detach().float().cpu().numpy()
    seq = x.size(1)
    smooth_win = max(3, seq // 100)
    rho = _smooth(out["rho"][i].detach().float().cpu().numpy(), smooth_win)
    p_env = _smooth(out["p_field_env"][i].detach().float().cpu().numpy(), smooth_win)
    s_env = _smooth(out["s_field_env"][i].detach().float().cpu().numpy(), smooth_win)
    wave_env = _smooth(x[i].detach().float().cpu().pow(2).mean(dim=-1).numpy(), smooth_win)
    _, p_sec = _peak_sec(p_prob, window_sec, pick_threshold)
    _, s_sec = _peak_sec(s_prob, window_sec, pick_threshold)
    gap = s_sec - p_sec if (p_sec >= 0 and s_sec >= 0 and s_sec > p_sec) else -1.0
    return CausalObservables(
        rho=rho,
        p_env=p_env,
        s_env=s_env,
        p_prob=p_prob,
        s_prob=s_prob,
        wave_env=wave_env,
        det=det_v,
        p_sec=p_sec,
        s_sec=s_sec,
        ps_gap_sec=gap,
        window_sec=window_sec,
        is_event=True,
    )


def collect_region_catalog(stead_dir: Path, region_key: str) -> pd.DataFrame:
    meta = REGIONS[region_key]
    lat_rng, lon_rng = meta["lat"], meta["lon"]
    usecols = [
        "trace_name",
        "trace_category",
        "source_origin_time",
        "source_latitude",
        "source_longitude",
        "source_magnitude",
        "source_magnitude_type",
        "source_depth_km",
        "source_distance_km",
        "receiver_latitude",
        "receiver_longitude",
        "network_code",
        "receiver_code",
        "snr_db",
        "p_arrival_sample",
        "s_arrival_sample",
    ]
    frames = []
    for chunk in range(1, 7):
        csv = Path(stead_dir) / f"chunk{chunk}_eofextract" / f"chunk{chunk}.csv"
        if not csv.is_file():
            continue
        print(f"[{region_key}] scan {csv.name}", flush=True)
        df = pd.read_csv(csv, usecols=lambda c: c in usecols, low_memory=False)
        if "trace_category" in df.columns:
            df = df[df["trace_category"].astype(str).str.contains("earthquake", case=False, na=False)]
        # chunk1 may lack picks / coords
        if "source_latitude" not in df.columns:
            continue
        lat = pd.to_numeric(df["source_latitude"], errors="coerce")
        lon = pd.to_numeric(df["source_longitude"], errors="coerce")
        m = lat.between(*lat_rng) & lon.between(*lon_rng)
        if not m.any():
            continue
        sub = df.loc[m].copy()
        sub["chunk"] = chunk
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    cat = pd.concat(frames, ignore_index=True)
    cat["t"] = pd.to_datetime(cat["source_origin_time"], errors="coerce")
    # fallback: parse from trace name
    miss = cat["t"].isna()
    if miss.any():
        cat.loc[miss, "t"] = cat.loc[miss, "trace_name"].map(_parse_time)
    cat["year"] = cat["t"].dt.year
    cat["network"] = cat["network_code"].astype(str)
    cat["station"] = cat["receiver_code"].astype(str)
    cat["site"] = cat["network"] + "." + cat["station"]
    cat = cat.dropna(subset=["trace_name", "t", "p_arrival_sample", "s_arrival_sample"])
    return cat


def select_panel(cat: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Prefer multi-year fixed stations, then stratified fill by year."""
    rng = np.random.default_rng(args.seed)
    if cat.empty:
        return cat

    site_years = cat.groupby("site")["year"].nunique()
    site_n = cat.groupby("site").size()
    fixed_sites = site_years[
        (site_years >= args.min_station_years) & (site_n.reindex(site_years.index).fillna(0) >= args.min_station_traces)
    ].index
    cat = cat.copy()
    cat["fixed_station"] = cat["site"].isin(fixed_sites)

    max_n = int(args.max_per_region)
    max_fixed = int(round(max_n * args.max_fixed_fraction))
    fixed = cat[cat["fixed_station"]]
    other = cat[~cat["fixed_station"]]

    def _strat_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df
        # proportional by year, then top-up
        years = df["year"].dropna().unique()
        if len(years) == 0:
            idx = rng.choice(len(df), size=n, replace=False)
            return df.iloc[idx]
        per = max(1, n // max(len(years), 1))
        parts = []
        for y in sorted(years):
            sub = df[df["year"] == y]
            take = min(len(sub), per)
            if take <= 0:
                continue
            parts.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
        out = pd.concat(parts, ignore_index=False) if parts else df.iloc[:0]
        if len(out) < n:
            remain = df.drop(index=out.index, errors="ignore")
            need = min(len(remain), n - len(out))
            if need > 0:
                out = pd.concat([out, remain.iloc[rng.choice(len(remain), size=need, replace=False)]])
        if len(out) > n:
            out = out.iloc[rng.choice(len(out), size=n, replace=False)]
        return out

    picked_fixed = _strat_sample(fixed, max_fixed)
    remain_n = max_n - len(picked_fixed)
    # fill from non-fixed first, then leftover fixed
    pool = pd.concat([other, fixed.drop(index=picked_fixed.index, errors="ignore")], ignore_index=False)
    picked_fill = _strat_sample(pool, remain_n) if remain_n > 0 else pool.iloc[:0]
    out = pd.concat([picked_fixed, picked_fill]).drop_duplicates("trace_name")
    print(
        f"[select] candidates={len(cat)} fixed_sites={len(fixed_sites)} "
        f"picked={len(out)} (fixed={int(out['fixed_station'].sum())})",
        flush=True,
    )
    return out.reset_index(drop=True)


class RegionalTraceDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, stead_dir: Path, seq_len: int):
        self.rows = rows.reset_index(drop=True)
        self.seq_len = seq_len
        self.original_len = 6000
        self._paths = {i: Path(stead_dir) / f"chunk{i}_eofextract" / f"chunk{i}.hdf5" for i in range(1, 7)}
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _handle(self, chunk: int) -> h5py.File:
        h = self._handles.get(chunk)
        if h is None:
            h = h5py.File(self._paths[chunk], "r")
            self._handles[chunk] = h
        return h

    def __getitem__(self, idx: int) -> dict:
        r = self.rows.iloc[idx]
        chunk = int(r["chunk"])
        name = str(r["trace_name"])
        waveform = self._handle(chunk)["data"][name][()]
        x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).transpose(0, 1)  # 3,T
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        x = F.interpolate(x.unsqueeze(0), size=self.seq_len, mode="linear", align_corners=False).squeeze(0)
        x = x.transpose(0, 1)  # T,3
        t = torch.linspace(0.0, 60.0, self.seq_len, dtype=torch.float32).unsqueeze(-1)
        return {
            "x": x,
            "t": t,
            "idx": torch.tensor(idx, dtype=torch.int64),
        }

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()


def _lstsq(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def label_region(region_key: str, args: argparse.Namespace, thr: dict, device: torch.device, model) -> Path:
    stead_dir = Path(args.stead_dir)
    out_dir = Path(args.output_dir) / region_key
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = collect_region_catalog(stead_dir, region_key)
    if cat.empty:
        raise SystemExit(f"[{region_key}] no STEAD events in box")
    panel = select_panel(cat, args)
    panel_path = out_dir / "panel_selection.csv"
    panel.to_csv(panel_path, index=False)

    ds = RegionalTraceDataset(panel, stead_dir, args.seq_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True)
            out = model(x, t)
            idxs = batch["idx"].cpu().numpy().tolist()
            for j, idx in enumerate(idxs):
                obs = observables_from_batch(
                    out, x, j, window_sec=60.0, pick_threshold=args.pick_threshold
                )
                if not has_valid_chain(obs):
                    continue
                feat = causal_chain_features(obs)
                r = panel.iloc[int(idx)]
                rows.append(
                    {
                        "trace_name": str(r["trace_name"]),
                        "mag": float(r["source_magnitude"]) if pd.notna(r["source_magnitude"]) else float("nan"),
                        "mag_type": str(r["source_magnitude_type"]) if pd.notna(r.get("source_magnitude_type", "")) else "",
                        "src_lat": float(r["source_latitude"]),
                        "src_lon": float(r["source_longitude"]),
                        "rcv_lat": float(r["receiver_latitude"]) if pd.notna(r["receiver_latitude"]) else float("nan"),
                        "rcv_lon": float(r["receiver_longitude"]) if pd.notna(r["receiver_longitude"]) else float("nan"),
                        "depth_km": float(r["source_depth_km"]) if pd.notna(r["source_depth_km"]) else float("nan"),
                        "dist_km": float(r["source_distance_km"]) if pd.notna(r["source_distance_km"]) else float("nan"),
                        "snr_db": parse_snr(r["snr_db"]),
                        "network": str(r["network"]),
                        "station": str(r["station"]),
                        "site": str(r["site"]),
                        "fixed_station": bool(r["fixed_station"]),
                        "chunk": int(r["chunk"]),
                        "t": r["t"].isoformat() if pd.notna(r["t"]) else "",
                        "year": int(r["year"]) if pd.notna(r["year"]) else -1,
                        "ps_gap": float(obs.ps_gap_sec),
                        "coda_slope": float(feat["coda_slope"]),
                        "onset_sharp": float(feat["onset_sharp"]),
                        "n_rho_peaks": float(feat["n_rho_peaks"]),
                        "det": float(obs.det),
                    }
                )
            if (bi + 1) % 20 == 0:
                rate = len(rows) / max(time.time() - t0, 1e-6)
                print(
                    f"[{region_key}] batch {bi+1}/{len(loader)} kept={len(rows)} ({rate:.1f} tr/s)",
                    flush=True,
                )

    ds.close()
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"[{region_key}] no valid causal chains")

    # assign shape with frozen thresholds
    df["shape"] = df.apply(
        lambda r: assign_shape(
            r,
            peak_thr=thr["peak_thr"],
            coda_fast=thr["coda_fast"],
            coda_slow=thr["coda_slow"],
            onset_hi=thr["onset_hi"],
            onset_lo=thr["onset_lo"],
        ),
        axis=1,
    )

    # coda path residual: regress coda_slope ~ log10(dist)
    ok = df["coda_slope"].notna() & df["dist_km"].notna() & (df["dist_km"] > 0)
    ld = np.log10(df.loc[ok, "dist_km"].to_numpy(float) + 1.0)
    cs = df.loc[ok, "coda_slope"].to_numpy(float)
    A = np.column_stack([ld, np.ones(len(ld))])
    coef = _lstsq(A, cs)
    df["coda_path_residual"] = np.nan
    df.loc[ok, "coda_path_residual"] = cs - A @ coef
    df["log_d"] = np.log10(df["dist_km"].to_numpy(float) + 1.0)

    mid_lat = 0.5 * (df["src_lat"] + df["rcv_lat"])
    mid_lon = 0.5 * (df["src_lon"] + df["rcv_lon"])
    df["path_region"] = [
        f"{int(np.floor(a / 5) * 5):+d}/{int(np.floor(b / 5) * 5):+d}"
        if np.isfinite(a) and np.isfinite(b)
        else "unknown"
        for a, b in zip(mid_lat, mid_lon)
    ]
    df["src_region"] = [
        f"{int(np.floor(a / 5) * 5):+d}/{int(np.floor(b / 5) * 5):+d}"
        if np.isfinite(a) and np.isfinite(b)
        else "unknown"
        for a, b in zip(df["src_lat"], df["src_lon"])
    ]

    out_csv = out_dir / "traces_labeled.csv"
    df.to_csv(out_csv, index=False)
    summary = {
        "region": region_key,
        "n_kept": int(len(df)),
        "n_fixed": int(df["fixed_station"].sum()),
        "shape_counts": df["shape"].value_counts().to_dict(),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "thresholds": thr,
        "beta_reg_coef_logd_intercept": [float(coef[0]), float(coef[1])],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{region_key}] wrote {out_csv} n={len(df)} in {summary['elapsed_sec']}s", flush=True)
    print(f"[{region_key}] shapes {summary['shape_counts']}", flush=True)
    return out_csv


def main() -> None:
    args = parse_args()
    thr = _load_thresholds(Path(args.thresholds_json))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    print(f"[expand] device={device} thr={thr}", flush=True)
    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    keys = [k.strip() for k in args.regions.split(",") if k.strip()]
    unknown = [k for k in keys if k not in REGIONS]
    if unknown:
        raise SystemExit(f"unknown regions {unknown}; choose from {sorted(REGIONS)}")

    paths = {}
    for key in keys:
        paths[key] = str(label_region(key, args, thr, device, model))

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    print(f"[done] {root / 'index.json'}", flush=True)


if __name__ == "__main__":
    main()
