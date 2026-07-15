#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download RACLETTE example cohort via public WebDAV (no pyvista required).

Pulls:
  - meta.json for all subjects in all 4 variants (~1 MB)
  - tutorial example Data/ (+ GroundTruth for 7p_2Venc) (~13 GB)

Source: ETH RACLETTE DOI 10.3929/ethz-c-000799752"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REMOTE_URL = "https://libdrive.ethz.ch/public.php/webdav"
REMOTE_TOKEN = "YLRgP6R5mDz7fLg"

VARIANTS = ["7p_2Venc", "7p_2Venc_Breathing", "7p_2Venc_Super", "13p_2Venc_ICOSA"]
VARIANT_META = {
    "7p_2Venc": "meta.json",
    "7p_2Venc_Breathing": "meta.json",
    "7p_2Venc_Super": "meta_mid.json",
    "13p_2Venc_ICOSA": "meta.json",
}
# Example subjects used by official tutorials
EXAMPLE = {
    "7p_2Venc": ("VirtualSubject_n001", ["Data", "GroundTruth"]),
    "7p_2Venc_Breathing": ("VirtualSubject_n004", ["Data"]),
    "7p_2Venc_Super": ("VirtualSubject_n001", ["Data"]),
    "13p_2Venc_ICOSA": ("VirtualSubject_n020", ["Data"]),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download RACLETTE example data")
    p.add_argument(
        "--out-dir",
        default="external_data/raclette/Tutorials/DataDownload/Downloaded",
        help="Local download root",
    )
    p.add_argument("--skip-meta", action="store_true")
    p.add_argument("--skip-volumes", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def _basename(item: dict) -> str:
    return item["name"].rsplit("/", 1)[-1]


def list_files_recursive(client, remote_dir: str) -> list:
    files = []
    for item in client.ls(remote_dir, detail=True):
        if item["type"] == "directory":
            files.extend(list_files_recursive(client, f"{remote_dir}/{_basename(item)}"))
        else:
            files.append(item)
    return files


def download_remote_dir(client, remote_dir: str, local_dir: Path, workers: int) -> tuple[float, int, int]:
    prefix = remote_dir.lstrip("/") + "/"
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_files = list_files_recursive(client, remote_dir)
    total_mb = sum(item["content_length"] for item in remote_files) / 1024**2

    def _one(item):
        rel = item["name"].split(prefix, 1)[-1]
        local_path = local_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size == item["content_length"]:
            return False
        client.download_file("/" + item["name"], str(local_path))
        return True

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, remote_files))
    dt = time.time() - t0
    n_new = sum(1 for x in results if x)
    print(
        f"  {remote_dir}: {total_mb:.1f} MB, {n_new}/{len(results)} new files, {dt:.1f}s",
        flush=True,
    )
    return total_mb, n_new, len(results)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from webdav4.client import Client

    print(f"[raclette] connect {REMOTE_URL}", flush=True)
    client = Client(REMOTE_URL, auth=(REMOTE_TOKEN, ""))

    for v in VARIANTS:
        n = sum(1 for item in client.ls(f"/{v}", detail=True) if item["type"] == "directory")
        print(f"  catalog {v}: {n} subjects", flush=True)

    if not args.skip_meta:
        print("[raclette] Part A: meta for all subjects …", flush=True)
        for variant in VARIANTS:
            metafile = VARIANT_META[variant]
            subjects = sorted(
                _basename(item)
                for item in client.ls(f"/{variant}", detail=True)
                if item["type"] == "directory"
            )
            meta_root = out / variant / "_meta_all"
            meta_root.mkdir(parents=True, exist_ok=True)

            def _meta(subject, variant=variant, metafile=metafile):
                dest = meta_root / f"{subject}_{metafile}"
                if dest.exists() and dest.stat().st_size > 10:
                    return
                with client.open(f"/{variant}/{subject}/{metafile}") as f:
                    data = json.load(f)
                dest.write_text(json.dumps(data), encoding="utf-8")

            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(_meta, subjects))
            print(f"  {variant}: {len(subjects)} metas", flush=True)

    if not args.skip_volumes:
        print("[raclette] Part B: example volumes (~13 GB) …", flush=True)
        for variant, (subject, subfolders) in EXAMPLE.items():
            for sub in subfolders:
                remote = f"/{variant}/{subject}/{sub}"
                local = out / variant / subject / sub
                print(f"[raclette] {remote} → {local}", flush=True)
                download_remote_dir(client, remote, local, args.workers)

    print("[raclette] done", flush=True)


if __name__ == "__main__":
    main()
