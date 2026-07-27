#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cache RACLETTE 3D velocity patches (Ux,Uy,Uz) for Stage-0c."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess RACLETTE 3D patches")
    p.add_argument(
        "--subject-dir",
        default="external_data/raclette/Tutorials/DataDownload/Downloaded/7p_2Venc/VirtualSubject_n001",
    )
    p.add_argument("--out", default="external_data/raclette_cache/gt_volumes.npz")
    p.add_argument("--out-d", type=int, default=8)
    p.add_argument("--out-h", type=int, default=16)
    p.add_argument("--out-w", type=int, default=16)
    p.add_argument("--z-radius", type=int, default=4)
    p.add_argument("--scripts-dir", default="external_data/raclette/Scripts")
    return p.parse_args()


def _bbox3(mask3d: np.ndarray, pad: int = 1) -> tuple[slice, slice, slice]:
    zs, ys, xs = np.where(mask3d > 0.5)
    if len(zs) == 0:
        d, h, w = mask3d.shape
        return slice(0, d), slice(0, h), slice(0, w)
    return (
        slice(max(0, int(zs.min()) - pad), min(mask3d.shape[0], int(zs.max()) + 1 + pad)),
        slice(max(0, int(ys.min()) - pad), min(mask3d.shape[1], int(ys.max()) + 1 + pad)),
        slice(max(0, int(xs.min()) - pad), min(mask3d.shape[2], int(xs.max()) + 1 + pad)),
    )


def _resize3(vel: np.ndarray, mask: np.ndarray, od: int, oh: int, ow: int) -> tuple[np.ndarray, np.ndarray]:
    """vel (3,D,H,W), mask (D,H,W)."""
    zd, zy, zx = od / vel.shape[1], oh / vel.shape[2], ow / vel.shape[3]
    out_v = np.stack([zoom(vel[c], (zd, zy, zx), order=1) for c in range(3)], axis=0).astype(np.float32)
    out_m = zoom(mask.astype(np.float32), (zd, zy, zx), order=0)
    out_m = (out_m > 0.5).astype(np.float32)
    out_v = out_v * out_m[None]
    return out_v, out_m


def main() -> None:
    args = parse_args()
    scripts = Path(args.scripts_dir).resolve()
    sys.path.insert(0, str(scripts))
    from dataHandler import read_fields_pv  # noqa: E402

    gt_dir = Path(args.subject_dir) / "GroundTruth"
    frames = sorted(gt_dir.glob("aorta_*.pv"))
    if not frames:
        raise FileNotFoundError(f"No frames in {gt_dir}")

    vel_list, mask_list, meta = [], [], []
    for fpath in frames:
        _, spacing, fields, _ = read_fields_pv(str(fpath))
        U = np.asarray(fields["U"], dtype=np.float32)  # (X,Y,Z,3)
        mask = np.asarray(fields["mask"], dtype=np.float32)
        Uc = np.transpose(U, (3, 2, 1, 0))  # (3,Z,Y,X)
        Mc = np.transpose(mask, (2, 1, 0))
        zmid = Uc.shape[1] // 2
        z0, z1 = max(0, zmid - args.z_radius), min(Uc.shape[1], zmid + args.z_radius + 1)
        sub_v = Uc[:, z0:z1]
        sub_m = Mc[z0:z1]
        if float(sub_m.mean()) < 0.01:
            continue
        sl = _bbox3(sub_m)
        crop_v = sub_v[:, sl[0], sl[1], sl[2]]
        crop_m = sub_m[sl[0], sl[1], sl[2]]
        vel_r, mask_r = _resize3(crop_v, crop_m, args.out_d, args.out_h, args.out_w)
        vel_list.append(vel_r)
        mask_list.append(mask_r)
        meta.append({"frame": fpath.name, "z0": int(z0), "z1": int(z1), "spacing": [float(s) for s in spacing]})
        print(f"[raclette-vol] {fpath.name} total={len(vel_list)}", flush=True)

    if not vel_list:
        raise RuntimeError("No 3D patches extracted")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        velocity=np.stack(vel_list, axis=0),
        vessel_mask=np.stack(mask_list, axis=0),
        meta_json=json.dumps(meta),
    )
    print(f"[raclette-vol] wrote {out} n={len(vel_list)} shape={vel_list[0].shape}", flush=True)


if __name__ == "__main__":
    main()
