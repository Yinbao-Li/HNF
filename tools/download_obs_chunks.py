#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download extra SeisBench OBS chunks for transfer learning."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SeisBench OBS chunks")
    p.add_argument(
        "--chunks",
        default="201805,201806,201807,201808",
        help="Comma-separated chunk ids",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    want = [c.strip() for c in args.chunks.split(",") if c.strip()]
    print(f"[obs-dl] requesting chunks={want}", flush=True)
    import seisbench.data as sbd
    import numpy as np

    ds = sbd.OBS(chunks=want, download_if_missing=True, component_order="Z12H")
    meta = ds.metadata
    p = meta["trace_p_arrival_sample"]
    s = meta["trace_s_arrival_sample"]
    p_ok = p.notna() & np.isfinite(p.astype(float))
    s_ok = s.notna() & np.isfinite(s.astype(float))
    print(
        f"[obs-dl] done  n={len(ds)}  with_P={int(p_ok.sum())}  "
        f"with_S={int(s_ok.sum())}  with_PS={int((p_ok & s_ok).sum())}",
        flush=True,
    )
    root = Path.home() / ".seisbench/datasets/obs"
    files = sorted(root.glob("waveforms*.hdf5"))
    for f in files:
        print(f"[obs-dl] {f.name} {f.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
