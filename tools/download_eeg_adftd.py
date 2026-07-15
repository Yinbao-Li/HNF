#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download OpenNeuro ds004504 (AD/FTD/HC EEG, CC0)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OpenNeuro ds004504")
    p.add_argument("--target-dir", default="external_data/eeg_adftd")
    p.add_argument("--tag", default="1.0.9")
    p.add_argument("--dataset", default="ds004504")
    p.add_argument(
        "--include",
        default="participants.tsv,dataset_description.json,derivatives",
        help="Comma-separated OpenNeuro include filters (empty = full dataset)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)
    try:
        from openneuro import download
    except ImportError as exc:
        raise SystemExit(
            "openneuro is not installed. Run: pip install 'openneuro-py==2023.1.0' 'mne>=1.5,<1.7'"
        ) from exc
    include = [x.strip() for x in args.include.split(",") if x.strip()] or None
    print(f"[download] {args.dataset}@{args.tag} → {target} include={include}", flush=True)
    download(
        dataset=args.dataset,
        tag=args.tag,
        target_dir=str(target),
        include=include,
    )
    print("[download] done", flush=True)


if __name__ == "__main__":
    main()
