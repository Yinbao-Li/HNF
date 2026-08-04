#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download OpenNeuro ds005385 (Dortmund Vital Study) for longitudinal EEG pilot.

Default: EyesClosed + acq-pre only, for subjects with both ses-1 and ses-2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", default="external_data/eeg_ds005385")
    p.add_argument("--tag", default="1.0.2")
    p.add_argument("--max-subjects", type=int, default=60, help="0 = all longitudinal pairs")
    p.add_argument("--task", default="EyesClosed", choices=["EyesClosed", "EyesOpen"])
    p.add_argument("--acq", default="pre", choices=["pre", "post"])
    return p.parse_args()


def longitudinal_subjects(participants_tsv: Path) -> list[str]:
    rows = list(csv.DictReader(participants_tsv.open(), delimiter="\t"))
    out = []
    for r in rows:
        if str(r.get("session1", "")).lower() == "yes" and str(r.get("session2", "")).lower() == "yes":
            out.append(str(r["participant_id"]))
    return out


def main() -> None:
    args = parse_args()
    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)
    meta = target / "participants.tsv"
    if not meta.is_file():
        from openneuro import download as on_download

        on_download(
            dataset="ds005385",
            tag=args.tag,
            target_dir=str(target),
            include=["participants.tsv", "participants.json", "dataset_description.json", "README"],
        )
    subjects = longitudinal_subjects(meta)
    if args.max_subjects and args.max_subjects > 0:
        subjects = subjects[: int(args.max_subjects)]
    include: list[str] = ["participants.tsv", "participants.json"]
    for sid in subjects:
        include.append(f"{sid}/{sid}_sessions.tsv")
        for ses in ("ses-1", "ses-2"):
            stem = f"{sid}/{ses}/eeg/{sid}_{ses}_task-{args.task}_acq-{args.acq}"
            include.extend(
                [
                    f"{stem}_eeg.edf",
                    f"{stem}_eeg.json",
                    f"{stem}_channels.tsv",
                ]
            )
    print(f"[ds005385] subjects={len(subjects)} files≈{len(include)} → {target}", flush=True)
    from openneuro import download as on_download

    on_download(dataset="ds005385", tag=args.tag, target_dir=str(target), include=include)
    print("[ds005385] done", flush=True)


if __name__ == "__main__":
    main()
