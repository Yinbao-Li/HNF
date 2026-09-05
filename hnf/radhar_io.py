"""RadHAR (nesl/RadHAR) I/O — public TI mmWave FMCW point-cloud recordings.

Dataset: Singh et al., RadHAR, ACM mmNets 2019.
https://github.com/nesl/RadHAR

Each ``.txt`` file is one activity session (~20 s, ~30 Hz). Points include
``range`` (m), ``doppler_bin``, ``velocity``, ``intensity``. Aggregation modes:

- ``intensity``: range × slow-time power (legacy)
- ``rich``: intensity + |vel| + signed vel + count + |bearing| + range–Doppler
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

C_LIGHT = 299_792_458.0
ACTIVITIES = ("boxing", "jack", "jump", "squats", "walk")

_PT_VEL = 3
_PT_INT = 4
_PT_RNG = 5
_PT_BRG = 6
_PT_DOP = 7

CLS_FEATURE_NAMES = ("intensity", "abs_vel", "signed_vel", "count", "abs_bearing")


def parse_radhar_txt(path: Path) -> dict[int, list[list[float]]]:
    """Parse one RadHAR text file → ``{frame_id: [[x,y,z,vel,intensity,range,bearing], ...]}``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = text.split()
    frame_num = -1
    frames: dict[int, list[list[float]]] = {}
    i = 0
    while i < len(tokens):
        key = tokens[i].rstrip(":")
        if key == "point_id" and i + 1 < len(tokens) and tokens[i + 1] == "0":
            frame_num += 1
        if key == "point_id" and frame_num >= 0:
            pt: dict[str, float] = {}
            j = i
            while j < len(tokens) - 1:
                k = tokens[j].rstrip(":")
                if k == "point_id" and j > i:
                    break
                if k in {"x", "y", "z", "velocity", "intensity", "range", "bearing", "doppler_bin"}:
                    try:
                        pt[k] = float(tokens[j + 1])
                    except ValueError:
                        pass
                j += 1
            if "range" in pt and "intensity" in pt:
                frames.setdefault(frame_num, []).append([
                    pt.get("x", 0.0),
                    pt.get("y", 0.0),
                    pt.get("z", 0.0),
                    pt.get("velocity", 0.0),
                    pt["intensity"],
                    pt["range"],
                    pt.get("bearing", 0.0),
                    pt.get("doppler_bin", 0.0),
                ])
            i = j
            continue
        i += 1
    return frames


def _zscore_time(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def frames_to_rich_range_time(
    frames: dict[int, list[list[float]]],
    *,
    n_range_bins: int = 24,
    range_max_m: float = 8.0,
    n_doppler_bins: int = 8,
    doppler_max: float = 16.0,
    include_range_doppler: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-cue ``(C, T)``: range cues + optional range–Doppler occupancy."""
    fids = sorted(frames.keys())
    edges = np.linspace(0.0, range_max_m, n_range_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    T = len(fids)
    n_feat = len(CLS_FEATURE_NAMES)
    feats = np.zeros((n_feat, n_range_bins, T), dtype=np.float32)
    wsum = np.zeros((n_range_bins, T), dtype=np.float32)
    rd = None
    if include_range_doppler:
        rd = np.zeros((n_range_bins, n_doppler_bins, T), dtype=np.float32)

    for ti, fid in enumerate(fids):
        for p in frames.get(fid, []):
            r = p[_PT_RNG]
            if r <= 0 or r >= range_max_m:
                continue
            bi = int(np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_range_bins - 1))
            w = float(np.exp(p[_PT_INT] / 20.0))
            v = float(p[_PT_VEL])
            feats[0, bi, ti] += w
            feats[1, bi, ti] += w * abs(v)
            feats[2, bi, ti] += w * v
            feats[3, bi, ti] += 1.0
            feats[4, bi, ti] += w * abs(float(p[_PT_BRG]))
            wsum[bi, ti] += w
            if rd is not None:
                d = float(p[_PT_DOP])
                di = int(np.clip(d, 0, doppler_max - 1e-6) * n_doppler_bins / doppler_max)
                di = int(np.clip(di, 0, n_doppler_bins - 1))
                rd[bi, di, ti] += w

    denom = np.maximum(wsum, 1e-6)
    feats[1] /= denom
    feats[2] /= denom
    feats[4] /= denom
    chunks = [_zscore_time(feats[i]) for i in range(n_feat)]
    x = np.concatenate(chunks, axis=0)
    if rd is not None:
        x = np.concatenate([x, _zscore_time(rd.reshape(n_range_bins * n_doppler_bins, T))], axis=0)
    return x.astype(np.float32), centers.astype(np.float64)


def rich_channel_count(
    n_range_bins: int = 24,
    n_doppler_bins: int = 8,
    include_range_doppler: bool = True,
) -> int:
    n = len(CLS_FEATURE_NAMES) * n_range_bins
    if include_range_doppler:
        n += n_range_bins * n_doppler_bins
    return n


def frames_to_range_time(
    frames: dict[int, list[list[float]]],
    *,
    n_range_bins: int = 24,
    range_max_m: float = 12.0,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build range-bin slow-time envelope ``(N, T)`` and bin centre ranges (m).

    Returns ``x_nt, range_centers_m, frame_ids``.
    """
    fids = sorted(frames.keys())
    if max_frames is not None:
        fids = fids[:max_frames]
    fids = fids[:: max(1, frame_step)]
    edges = np.linspace(0.0, range_max_m, n_range_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    T = len(fids)
    x = np.zeros((n_range_bins, T), dtype=np.float32)
    for ti, fid in enumerate(fids):
        pts = frames[fid]
        if not pts:
            continue
        ranges = np.array([p[5] for p in pts], dtype=np.float64)
        ints = np.array([p[4] for p in pts], dtype=np.float64)
        # log-power aggregate per bin
        for r, w in zip(ranges, ints):
            if r <= 0 or r >= range_max_m:
                continue
            bi = int(np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_range_bins - 1))
            x[bi, ti] += np.exp(w / 20.0)
    # per-bin z-score over time (shape focus)
    x = _zscore_time(x)
    return x, centers.astype(np.float64), np.array(fids, dtype=np.int64)


def range_geometry_km(centers_m: np.ndarray) -> np.ndarray:
    """Pairwise range separation (km) — 1-D radar axis graph."""
    d = np.abs(centers_m.reshape(-1, 1) - centers_m.reshape(1, -1))
    return (d / 1000.0).astype(np.float64)


def two_way_delay_sec(centers_m: np.ndarray) -> np.ndarray:
    """Round-trip delay per range bin (s), relative to nearest bin."""
    tau = 2.0 * centers_m / C_LIGHT
    return (tau - tau.min()).astype(np.float64)


def iter_radhar_files(data_root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(split, activity, path)`` for every ``.txt`` under Train/Test."""
    for split in ("Train", "Test"):
        base = data_root / split
        if not base.is_dir():
            continue
        for act in ACTIVITIES:
            adir = base / act
            if not adir.is_dir():
                continue
            for fp in sorted(adir.glob("*.txt")):
                yield split.lower(), act, fp


def build_regime_window(
    path: Path,
    *,
    n_range_bins: int = 24,
    t_steps: int = 64,
    t_start: int = 0,
    fps: float = 30.0,
    frames: Optional[dict[int, list[list[float]]]] = None,
) -> Optional[dict]:
    """One STEAD-compatible sample dict from a RadHAR file segment."""
    frames = frames if frames is not None else parse_radhar_txt(path)
    fids = sorted(frames.keys())
    if t_start + t_steps > len(fids):
        return None
    sel = fids[t_start : t_start + t_steps]
    sub = {j: frames[sel[j]] for j in range(t_steps)}
    x_np, centers, _ = frames_to_range_time(sub, n_range_bins=n_range_bins)
    dist = range_geometry_km(centers)
    p_rel = two_way_delay_sec(centers)
    dt = 1.0 / fps
    return {
        "x": torch.from_numpy(x_np),
        "dist_ij_km": torch.from_numpy(dist),
        "dt_sec": torch.tensor(dt),
        "p_rel_sec": torch.from_numpy(p_rel),
        "source_file": str(path),
        "domain": "radhar",
    }


def build_cls_window(
    path: Path,
    *,
    frames: dict[int, list[list[float]]],
    t_start: int,
    t_steps: int = 60,
    n_range_bins: int = 24,
    feature_mode: str = "rich",
    include_range_doppler: bool = True,
    n_doppler_bins: int = 8,
    fps: float = 30.0,
) -> Optional[dict]:
    fids = sorted(frames.keys())
    if t_start + t_steps > len(fids):
        return None
    sel = fids[t_start : t_start + t_steps]
    sub = {j: frames[sel[j]] for j in range(t_steps)}
    if feature_mode == "intensity":
        x_np, _, _ = frames_to_range_time(sub, n_range_bins=n_range_bins, range_max_m=8.0)
    elif feature_mode == "rich":
        x_np, _ = frames_to_rich_range_time(
            sub,
            n_range_bins=n_range_bins,
            include_range_doppler=include_range_doppler,
            n_doppler_bins=n_doppler_bins,
        )
    else:
        raise ValueError(f"Unknown feature_mode={feature_mode!r}")
    return {
        "x": torch.from_numpy(x_np),
        "source_file": str(path),
        "domain": "radhar",
        "t_start": int(t_start),
        "fps": float(fps),
    }


def collect_radhar_windows(
    data_root: Path,
    *,
    n_range_bins: int = 24,
    t_steps: int = 64,
    stride_frames: int = 32,
    max_files: Optional[int] = None,
    max_windows_per_file: Optional[int] = 8,
    fps: float = 30.0,
    splits: Optional[tuple[str, ...]] = None,
) -> list[dict]:
    """Slide over public RadHAR recordings (optional Train/Test filter)."""
    windows: list[dict] = []
    n_files = 0
    allow = None if splits is None else {s.lower() for s in splits}
    for _split, act, fp in iter_radhar_files(data_root):
        if allow is not None and _split not in allow:
            continue
        frames = parse_radhar_txt(fp)
        n_frames = len(frames)
        if n_frames < t_steps:
            continue
        starts = list(range(0, n_frames - t_steps + 1, stride_frames))
        if max_windows_per_file is not None:
            starts = starts[:max_windows_per_file]
        for t0 in starts:
            w = build_regime_window(
                fp,
                frames=frames,
                n_range_bins=n_range_bins,
                t_steps=t_steps,
                t_start=t0,
                fps=fps,
            )
            if w is not None:
                w["activity"] = act
                w["split"] = _split
                windows.append(w)
        n_files += 1
        if max_files is not None and n_files >= max_files:
            break
    return windows


def collect_radhar_cls_windows(
    data_root: Path,
    *,
    n_range_bins: int = 24,
    t_steps: int = 60,
    stride_frames: int = 10,
    max_files: Optional[int] = None,
    max_windows_per_file: Optional[int] = None,
    fps: float = 30.0,
    splits: Optional[tuple[str, ...]] = None,
    feature_mode: str = "rich",
    include_range_doppler: bool = True,
    n_doppler_bins: int = 8,
) -> list[dict]:
    """Classification windows; parses each file once."""
    windows: list[dict] = []
    n_files = 0
    allow = None if splits is None else {s.lower() for s in splits}
    for _split, act, fp in iter_radhar_files(data_root):
        if allow is not None and _split not in allow:
            continue
        frames = parse_radhar_txt(fp)
        n_frames = len(frames)
        if n_frames < t_steps:
            continue
        starts = list(range(0, n_frames - t_steps + 1, stride_frames))
        if max_windows_per_file is not None:
            starts = starts[:max_windows_per_file]
        for t0 in starts:
            w = build_cls_window(
                fp,
                frames=frames,
                t_start=t0,
                t_steps=t_steps,
                n_range_bins=n_range_bins,
                feature_mode=feature_mode,
                include_range_doppler=include_range_doppler,
                n_doppler_bins=n_doppler_bins,
                fps=fps,
            )
            if w is not None:
                w["activity"] = act
                w["split"] = _split
                windows.append(w)
        n_files += 1
        if max_files is not None and n_files >= max_files:
            break
    return windows
