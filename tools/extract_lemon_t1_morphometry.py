#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast T1 proxies for LEMON (brain volume + 3-class GMM GM/WM/CSF).

Not FreeSurfer thickness. Use inv-2 (or skull-stripped UNI) masked to brain.
Hippocampal volumes require SynthSeg/FastSurfer (separate job).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from hnf.eeg_lemon import list_lemon_subjects


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="external_data/eeg_lemon")
    p.add_argument("--output-dir", default="outputs/eeg/lemon_morphometry")
    p.add_argument("--max-subjects", type=int, default=0)
    return p.parse_args()


def _load_nii(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    return data, zooms


def morphometry_one(inv2: Path | None, brain: Path | None, t1w: Path | None) -> dict:
    src = inv2 or brain or t1w
    if src is None or not src.is_file():
        return {"ok": False, "reason": "no_t1"}
    data, zooms = _load_nii(src)
    vox_mm3 = float(np.prod(zooms))
    if brain is not None and brain.is_file():
        bdata, bzooms = _load_nii(brain)
        if bdata.shape != data.shape:
            # derivatives brain is often a different grid; use it alone
            data, zooms, vox_mm3 = bdata, bzooms, float(np.prod(bzooms))
            mask = np.abs(data) > 0
        else:
            mask = np.abs(bdata) > 0
    else:
        finite = np.isfinite(data)
        vals = data[finite]
        if vals.size < 1000:
            return {"ok": False, "reason": "empty"}
        thr = float(np.quantile(vals[vals > 0], 0.15)) if np.any(vals > 0) else 0.0
        mask = finite & (data > thr)
    n_brain = int(mask.sum())
    if n_brain < 1000:
        return {"ok": False, "reason": "tiny_mask", "n_brain": n_brain}
    brain_cm3 = n_brain * vox_mm3 / 1000.0
    x = data[mask].reshape(-1, 1)
    x = x[np.isfinite(x[:, 0])]
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0, max_iter=80)
    gmm.fit(x)
    lab = gmm.predict(x)
    order = np.argsort(gmm.means_.ravel())  # CSF, GM, WM on T1-like contrast
    counts = {int(k): int((lab == k).sum()) for k in range(3)}
    csf_n = counts[int(order[0])]
    gm_n = counts[int(order[1])]
    wm_n = counts[int(order[2])]
    gm_cm3 = gm_n * vox_mm3 / 1000.0
    wm_cm3 = wm_n * vox_mm3 / 1000.0
    csf_cm3 = csf_n * vox_mm3 / 1000.0
    icv_cm3 = gm_cm3 + wm_cm3 + csf_cm3
    return {
        "ok": True,
        "source": src.name,
        "brain_cm3": brain_cm3,
        "gm_cm3": gm_cm3,
        "wm_cm3": wm_cm3,
        "csf_cm3": csf_cm3,
        "icv_cm3": icv_cm3,
        "gm_icv": gm_cm3 / icv_cm3 if icv_cm3 > 1 else float("nan"),
        "wm_icv": wm_cm3 / icv_cm3 if icv_cm3 > 1 else float("nan"),
        "brain_icv": brain_cm3 / icv_cm3 if icv_cm3 > 1 else float("nan"),
        "n_brain_voxels": n_brain,
        "vox_mm3": vox_mm3,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subjects = list_lemon_subjects(root)
    if args.max_subjects and args.max_subjects > 0:
        subjects = subjects[: int(args.max_subjects)]
    rows = []
    n_fail = 0
    for i, s in enumerate(subjects, 1):
        morph = morphometry_one(s.inv2_path, s.brain_path, s.t1w_path)
        if not morph.get("ok"):
            n_fail += 1
        row = {
            "subject_id": s.subject_id,
            "age": s.age,
            "sex": s.sex,
            "age_group": s.age_group,
            **{k: v for k, v in morph.items() if k != "ok"},
            "ok": bool(morph.get("ok")),
        }
        rows.append(row)
        if i % 20 == 0 or i == len(subjects):
            print(f"[morph] {i}/{len(subjects)} fail={n_fail}", flush=True)
    csv_path = out / "lemon_t1_morphometry.csv"
    keys = sorted({k for r in rows for k in r})
    preferred = [
        "subject_id",
        "age",
        "sex",
        "age_group",
        "ok",
        "brain_cm3",
        "gm_cm3",
        "wm_cm3",
        "csf_cm3",
        "icv_cm3",
        "gm_icv",
        "wm_icv",
        "brain_icv",
        "source",
        "n_brain_voxels",
        "vox_mm3",
        "reason",
    ]
    fieldnames = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = {
        "n": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_fail": n_fail,
        "csv": str(csv_path),
        "note": "GMM tissue proxy on inv-2/brain, not FreeSurfer thickness.",
    }
    (out / "MORPH_STATUS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
