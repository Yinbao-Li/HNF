#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a disjoint OBS adapt split with per-event p_offset assignments."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sample_key(s: dict) -> str:
    if s.get("event_key"):
        return str(s["event_key"])
    return f"{s.get('chunk','')}|{s.get('ds_index', s.get('trace_name',''))}"


def _entry_from_sample(s: dict) -> dict:
    return {
        "chunk": s["chunk"],
        "ds_index": int(s["ds_index"]),
        "p_offset_sec": float(s["p_offset_sec"]),
        "event_key": sample_key(s),
        "s_valid": bool(s["s_valid"]),
    }


def build_disjoint_split(
    chunks: list[str],
    train_n: int,
    holdout_n: int,
    window_sec: float,
    seed: int,
    p_offset_min: float,
    p_offset_max: float,
) -> tuple[dict, list[dict], list[dict]]:
    obs_mod = _load_obs_compare_module()
    pool_n = train_n + holdout_n
    # Fixed mid offset only for candidate selection stability; then reassign random.
    samples, load_info = obs_mod.load_obs_windows(
        chunks,
        pool_n,
        window_sec,
        p_offset_sec=0.5 * (p_offset_min + p_offset_max),
        seed=seed,
        require_full_3c=True,
        p_offset_min=p_offset_min,
        p_offset_max=p_offset_max,
    )
    if len(samples) < pool_n:
        raise RuntimeError(f"Need >= {pool_n} samples, got {len(samples)}")
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(samples))
    rng.shuffle(idxs)
    train_idx = idxs[:train_n].tolist()
    holdout_idx = idxs[train_n : train_n + holdout_n].tolist()
    train_samples = [samples[i] for i in train_idx]
    holdout_samples = [samples[i] for i in holdout_idx]
    train_entries = [_entry_from_sample(s) for s in train_samples]
    holdout_entries = [_entry_from_sample(s) for s in holdout_samples]
    train_keys = [e["event_key"] for e in train_entries]
    holdout_keys = [e["event_key"] for e in holdout_entries]
    assert not (set(train_keys) & set(holdout_keys)), "train/holdout overlap"
    meta = {
        "chunks": chunks,
        "seed": seed,
        "window_sec": window_sec,
        "p_offset_min": p_offset_min,
        "p_offset_max": p_offset_max,
        "protocol": "random_p_offset",
        "train_n": train_n,
        "holdout_n": holdout_n,
        "train_entries": train_entries,
        "holdout_entries": holdout_entries,
        "train_keys": train_keys,
        "holdout_keys": holdout_keys,
        "load_info": load_info,
        "n_with_s_train": sum(1 for s in train_samples if s["s_valid"]),
        "n_with_s_holdout": sum(1 for s in holdout_samples if s["s_valid"]),
        "holdout_p_offset_mean": float(np.mean([e["p_offset_sec"] for e in holdout_entries])),
        "holdout_p_offset_std": float(np.std([e["p_offset_sec"] for e in holdout_entries])),
    }
    return meta, train_samples, holdout_samples


def load_split(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def filter_by_keys(samples: list[dict], keys: list[str]) -> list[dict]:
    want = set(keys)
    by = {sample_key(s): s for s in samples if sample_key(s) in want}
    return [by[k] for k in keys if k in by]


def load_split_samples(split_path: str, which: str = "train"):
    """Load rematerialized train or holdout windows from split entries."""
    obs_mod = _load_obs_compare_module()
    meta = load_split(split_path)
    key = "train_entries" if which == "train" else "holdout_entries"
    if key not in meta:
        raise RuntimeError(f"split missing {key}; rebuild with tools/obs_matched_split.py")
    samples, info = obs_mod.load_obs_windows_from_entries(
        meta[key], float(meta["window_sec"]), require_full_3c=True,
    )
    return samples, info, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/obs_matched_adapt_split_randoffset/split.json")
    p.add_argument("--chunks", default="201805,201806,201807,201808")
    p.add_argument("--train-n", type=int, default=2400)
    p.add_argument("--holdout-n", type=int, default=800)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--p-offset-min", type=float, default=4.0)
    p.add_argument("--p-offset-max", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
    meta, train_samples, holdout_samples = build_disjoint_split(
        chunks,
        args.train_n,
        args.holdout_n,
        args.window_sec,
        args.seed,
        args.p_offset_min,
        args.p_offset_max,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "output": str(out),
                "train_n": len(train_samples),
                "holdout_n": len(holdout_samples),
                "train_with_s": meta["n_with_s_train"],
                "holdout_with_s": meta["n_with_s_holdout"],
                "p_offset_range": [args.p_offset_min, args.p_offset_max],
                "holdout_offset_mean": meta["holdout_p_offset_mean"],
                "holdout_offset_std": meta["holdout_p_offset_std"],
                "overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
