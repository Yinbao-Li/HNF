#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cache RACLETTE GroundTruth mid-aorta slices as (2,H,W) velocity fields.

Uses system Python 3.10 + pyvista_zstd (anaconda 3.8 cannot install pyvista-zstd).
Output: ``external_data/raclette_cache/gt_slices.npz`` for Stage-0b training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess RACLETTE GT slices")
    p.add_argument(
        "--subject-dir",
        default=(
            "external_data/raclette/Tutorials/DataDownload/Downloaded/"
            "7p_2Venc/VirtualSubject_n001"
        ),
    )
    p.add_argument("--out", default="external_data/raclette_cache/gt_slices.npz")
    p.add_argument("--out-size", type=int, default=32)
    p.add_argument("--z-radius", type=int, default=4, help="± slices around mid-z")
    p.add_argument("--scripts-dir", default="external_data/raclette/Scripts")
    return p.parse_args()


def _bbox_mask(mask2d: np.ndarray, pad: int = 2) -> tuple[slice, slice]:
    ys, xs = np.where(mask2d > 0.5)
    if len(ys) == 0:
        h, w = mask2d.shape
        return slice(0, h), slice(0, w)
    y0, y1 = max(0, int(ys.min()) - pad), min(mask2d.shape[0], int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(mask2d.shape[1], int(xs.max()) + 1 + pad)
    return slice(y0, y1), slice(x0, x1)


def _resize_vel(vel: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """vel (2,H,W), mask (H,W) → size×size."""
    zy = size / vel.shape[1]
    zx = size / vel.shape[2]
    out_v = np.stack(
        [zoom(vel[c], (zy, zx), order=1) for c in range(2)],
        axis=0,
    ).astype(np.float32)
    out_m = zoom(mask.astype(np.float32), (zy, zx), order=0)
    out_m = (out_m > 0.5).astype(np.float32)
    return out_v, out_m


def main() -> None:
    args = parse_args()
    scripts = Path(args.scripts_dir).resolve()
    sys.path.insert(0, str(scripts))
    from dataHandler import read_fields_pv  # noqa: E402

    gt_dir = Path(args.subject_dir) / "GroundTruth"
    frames = sorted(gt_dir.glob("aorta_*.pv"))
    if not frames:
        raise FileNotFoundError(f"No aorta_*.pv under {gt_dir}")

    vel_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for fi, fpath in enumerate(frames):
        dims, spacing, fields, _ = read_fields_pv(str(fpath))
        U = np.asarray(fields["U"], dtype=np.float32)  # (X,Y,Z,3)
        mask = np.asarray(fields["mask"], dtype=np.float32)
        # pyvista dims are (nx,ny,nz); treat as (X,Y,Z)
        zmid = U.shape[2] // 2
        z0 = max(0, zmid - args.z_radius)
        z1 = min(U.shape[2], zmid + args.z_radius + 1)
        for z in range(z0, z1):
            # In-plane components Ux, Uy on this z-slice; transpose to (H=Y, W=X)
            u_xy = np.transpose(U[:, :, z, :2], (1, 0, 2))  # (Y,X,2)
            m2 = np.transpose(mask[:, :, z], (1, 0))
            if float(m2.mean()) < 0.01:
                continue
            ys, xs = _bbox_mask(m2)
            crop_v = np.transpose(u_xy[ys, xs, :], (2, 0, 1))  # (2,h,w)
            crop_m = m2[ys, xs]
            vel_r, mask_r = _resize_vel(crop_v, crop_m, args.out_size)
            # Zero outside vessel for cleaner targets
            vel_r = vel_r * mask_r[None]
            vel_list.append(vel_r)
            mask_list.append(mask_r)
            meta_rows.append(
                {
                    "frame": fpath.name,
                    "z": int(z),
                    "spacing": [float(s) for s in spacing],
                    "dims": [int(d) for d in dims],
                }
            )
        print(f"[raclette-pp] {fpath.name} → running total {len(vel_list)} slices", flush=True)

    if not vel_list:
        raise RuntimeError("No valid slices extracted")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vel = np.stack(vel_list, axis=0)
    masks = np.stack(mask_list, axis=0)
    np.savez_compressed(
        out,
        velocity=vel.astype(np.float32),
        vessel_mask=masks.astype(np.float32),
        meta_json=json.dumps(meta_rows),
    )
    print(
        f"[raclette-pp] wrote {out}  n={vel.shape[0]} shape={tuple(vel.shape[1:])} "
        f"|U|mean={float(np.abs(vel).mean()):.4g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
