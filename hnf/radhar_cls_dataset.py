"""RadHAR official-split dataset for 5-class activity detection (task C)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hnf.radhar_io import (
    ACTIVITIES,
    collect_radhar_cls_windows,
    rich_channel_count,
)

ACTIVITY_TO_IDX = {a: i for i, a in enumerate(ACTIVITIES)}


def _cache_key(
    split: str,
    *,
    n_range_bins: int,
    t_steps: int,
    stride_frames: int,
    feature_mode: str,
    include_range_doppler: bool,
    n_doppler_bins: int,
) -> str:
    rd = "rd" if include_range_doppler else "nord"
    return (
        f"radhar_cls_{split}_rb{n_range_bins}_t{t_steps}_s{stride_frames}_"
        f"{feature_mode}_{rd}{n_doppler_bins}.npz"
    )


def load_or_build_windows(
    data_root: Path,
    split: str,
    cache_dir: Path,
    *,
    rebuild_cache: bool = False,
    **kwargs,
) -> tuple[list[dict], int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_key(split, **{k: kwargs[k] for k in (
        "n_range_bins", "t_steps", "stride_frames", "feature_mode",
        "include_range_doppler", "n_doppler_bins",
    )})
    if cache_path.exists() and not rebuild_cache:
        print(f"[cache] load {cache_path}", flush=True)
        z = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        xs = z["x"]
        ys = z["y"]
        acts = [str(a) for a in z["activities"]]
        files = [str(f) for f in z["source_files"]]
        samples = []
        for i in range(len(ys)):
            samples.append({
                "x": torch.from_numpy(xs[i]),
                "y": int(ys[i]),
                "activity": acts[i],
                "source_file": files[i],
                "split": split,
            })
        return samples, int(z["n_channels"])

    print(f"[cache] build {split} windows from {data_root} ...", flush=True)
    raw = collect_radhar_cls_windows(data_root, splits=(split,), **kwargs)
    if not raw:
        raise FileNotFoundError(f"No windows for split={split} under {data_root}")
    n_ch = int(raw[0]["x"].shape[0])
    xs = np.stack([w["x"].numpy() for w in raw])
    ys = np.array([ACTIVITY_TO_IDX[str(w["activity"])] for w in raw], dtype=np.int64)
    acts = np.array([str(w["activity"]) for w in raw])
    files = np.array([str(w.get("source_file", "")) for w in raw])
    meta = json.dumps({"n": len(raw), "n_channels": n_ch, "split": split})
    np.savez(
        cache_path,
        x=xs,
        y=ys,
        activities=acts,
        source_files=files,
        meta=np.array(meta),
        n_channels=np.array(n_ch),
    )
    print(f"[cache] wrote {len(raw)} → {cache_path}", flush=True)
    samples = []
    for i, w in enumerate(raw):
        samples.append({
            "x": torch.from_numpy(xs[i]),
            "y": int(ys[i]),
            "activity": str(acts[i]),
            "source_file": str(files[i]),
            "split": split,
        })
    return samples, n_ch


class RadHARClsDataset(Dataset):
    """Windows ``(C, T)`` with class index ``y``."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        *,
        n_range_bins: int = 24,
        t_steps: int = 60,
        stride_frames: int = 10,
        max_windows_per_file: Optional[int] = None,
        max_files: Optional[int] = None,
        fps: float = 30.0,
        feature_mode: str = "rich",
        include_range_doppler: bool = True,
        n_doppler_bins: int = 8,
        cache_dir: Optional[Path] = None,
        rebuild_cache: bool = False,
        augment: bool = False,
        seed: int = 42,
    ):
        split_key = split.lower().strip()
        if split_key not in {"train", "test"}:
            raise ValueError(f"split must be train|test, got {split!r}")
        self.split = split_key
        self.fps = float(fps)
        self.epoch_sec = float(t_steps) / self.fps
        self.n_range_bins = int(n_range_bins)
        self.t_steps = int(t_steps)
        self.augment = bool(augment) and split_key == "train"
        self.seed = int(seed)
        self.feature_mode = feature_mode

        kw = dict(
            n_range_bins=n_range_bins,
            t_steps=t_steps,
            stride_frames=stride_frames,
            max_files=max_files,
            max_windows_per_file=max_windows_per_file,
            fps=fps,
            feature_mode=feature_mode,
            include_range_doppler=include_range_doppler,
            n_doppler_bins=n_doppler_bins,
        )
        if cache_dir is not None:
            self.samples, self.n_channels = load_or_build_windows(
                Path(data_root), split_key, Path(cache_dir),
                rebuild_cache=rebuild_cache, **kw,
            )
        else:
            raw = collect_radhar_cls_windows(Path(data_root), splits=(split_key,), **kw)
            if not raw:
                raise FileNotFoundError(f"No RadHAR windows for split={split_key}")
            self.samples = [
                {
                    "x": w["x"],
                    "y": ACTIVITY_TO_IDX[str(w["activity"])],
                    "activity": str(w["activity"]),
                    "source_file": w.get("source_file", ""),
                }
                for w in raw
            ]
            self.n_channels = int(self.samples[0]["x"].shape[0])

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed + idx)
        scale = float(rng.uniform(0.85, 1.15))
        x = x * scale
        shift = int(rng.integers(-5, 6))
        if shift:
            x = torch.roll(x, shifts=shift, dims=-1)
        noise = float(rng.uniform(0.0, 0.05))
        if noise > 0:
            x = x + torch.randn_like(x) * noise
        return x

    def __getitem__(self, idx: int) -> dict:
        w = self.samples[idx]
        x = w["x"].clone()
        if self.augment:
            x = self._augment(x, idx)
        t = torch.linspace(0.0, self.epoch_sec, self.t_steps, dtype=torch.float32).unsqueeze(-1)
        return {
            "x": x,
            "t": t,
            "y": torch.tensor(int(w["y"]), dtype=torch.long),
            "activity": w["activity"],
            "source_file": w.get("source_file", ""),
        }


def class_weights_from_samples(samples: list[dict], n_classes: int = 5) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for w in samples:
        counts[int(w["y"])] += 1
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
