#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a full-corpus OBS train/val/test split (STEAD-style).

Metadata-only: does not materialize waveforms. Entries are rematerialized
at train/eval time via load_obs_windows_from_entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full OBS train/val/test split")
    p.add_argument(
        "--chunks",
        default=(
            "201805,201806,201807,201808,201809,201810,201811,201812,"
            "201901,201902,201903,201904,201905,201906,201907,201908,000000"
        ),
    )
    p.add_argument("--output", default="outputs/obs_full_native_split/split.json")
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--p-offset-min", type=float, default=4.0)
    p.add_argument("--p-offset-max", type=float, default=12.0)
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--require-full-3c-meta",
        action="store_true",
        default=True,
        help="Require '2' in trace_component_order (horizontal present)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    import seisbench.data as sbd

    candidates: list[dict] = []
    per_chunk: dict[str, int] = {}
    for chunk in chunks:
        print(f"[obs-full-split] scan chunk={chunk}", flush=True)
        ds = sbd.OBS(chunks=[chunk], download_if_missing=False, component_order="Z12H")
        meta = ds.metadata
        n_c = 0
        for i in range(len(ds)):
            row = meta.iloc[i]
            p = row.get("trace_p_arrival_sample")
            if p is None or (isinstance(p, float) and not np.isfinite(p)):
                continue
            order = str(row.get("trace_component_order", ""))
            if args.require_full_3c_meta and ("2" not in order):
                continue
            s_raw = row.get("trace_s_arrival_sample")
            s_valid = s_raw is not None and np.isfinite(float(s_raw))
            # Deterministic random offset per event (same scheme as load_obs_windows).
            local = np.random.default_rng(args.seed + (hash((chunk, int(i))) % 1_000_000_007))
            p_offset = float(local.uniform(args.p_offset_min, args.p_offset_max))
            event_key = f"{chunk}|{int(i)}"
            candidates.append(
                {
                    "chunk": chunk,
                    "ds_index": int(i),
                    "p_offset_sec": p_offset,
                    "event_key": event_key,
                    "s_valid": bool(s_valid),
                }
            )
            n_c += 1
        per_chunk[chunk] = n_c
        print(f"[obs-full-split]   kept={n_c}", flush=True)

    n = len(candidates)
    if n < 500:
        raise RuntimeError(f"Too few OBS candidates: {n}")

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    n_test = n - n_train - n_val
    if n_test < 100:
        deficit = 100 - n_test
        n_train = max(100, n_train - deficit)
        n_test = n - n_train - n_val

    train_e = [candidates[int(i)] for i in idxs[:n_train]]
    val_e = [candidates[int(i)] for i in idxs[n_train : n_train + n_val]]
    test_e = [candidates[int(i)] for i in idxs[n_train + n_val :]]

    train_keys = [e["event_key"] for e in train_e]
    val_keys = [e["event_key"] for e in val_e]
    test_keys = [e["event_key"] for e in test_e]
    assert not (set(train_keys) & set(val_keys))
    assert not (set(train_keys) & set(test_keys))
    assert not (set(val_keys) & set(test_keys))

    meta = {
        "protocol": "obs_full_native_random_p_offset",
        "chunks": chunks,
        "seed": args.seed,
        "window_sec": args.window_sec,
        "p_offset_min": args.p_offset_min,
        "p_offset_max": args.p_offset_max,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "n_total": n,
        "train_n": len(train_e),
        "val_n": len(val_e),
        "holdout_n": len(test_e),
        "test_n": len(test_e),
        "train_entries": train_e,
        "val_entries": val_e,
        "holdout_entries": test_e,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "holdout_keys": test_keys,
        "n_with_s_train": sum(1 for e in train_e if e["s_valid"]),
        "n_with_s_val": sum(1 for e in val_e if e["s_valid"]),
        "n_with_s_test": sum(1 for e in test_e if e["s_valid"]),
        "per_chunk_candidates": per_chunk,
        "note": (
            "STEAD-style native OBS split over full SeisBench OBS corpus. "
            "holdout_entries == test for train_obs_picking.py compatibility. "
            "Metadata-only; waveforms rematerialized at train/eval."
        ),
    }
    out.write_text(json.dumps(meta, indent=2))
    summary = {
        "output": str(out),
        "n_total": n,
        "train": len(train_e),
        "val": len(val_e),
        "test": len(test_e),
        "chunks": len(chunks),
        "with_S_train": meta["n_with_s_train"],
        "per_chunk": per_chunk,
    }
    (out.parent / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chunk"}, indent=2), flush=True)
    print("[obs-full-split] per_chunk:", json.dumps(per_chunk), flush=True)


if __name__ == "__main__":
    main()
