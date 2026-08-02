#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download Leeds PS SAOS rheology data (Elliott et al. 2025, DOI 10.5518/1689)."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import Request, urlretrieve

RHEO_ZIP = "https://archive.researchdata.leeds.ac.uk/1428/3/Rheo_Data.zip"
README_URL = "https://archive.researchdata.leeds.ac.uk/1428/7/README_Elliott-etal_2025.txt"
SOURCES_URL = "https://archive.researchdata.leeds.ac.uk/1428/6/Data_Sources.txt"
UA = "Mozilla/5.0 (compatible; HNF-rheo-download/1.0)"


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": UA})
    # urlretrieve does not take Request; use urlopen
    from urllib.request import urlopen

    with urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        default="external_data/rheo_leeds_ps",
        help="Destination directory",
    )
    p.add_argument("--with-gpc", action="store_true", default=True, help="Also download GPC_Data")
    args = p.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "Rheo_Data.zip"
    print("Downloading", RHEO_ZIP)
    _fetch(RHEO_ZIP, zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(root)
    if args.with_gpc:
        gpc_url = "https://archive.researchdata.leeds.ac.uk/1428/1/GPC_Data.zip"
        gz = root / "GPC_Data.zip"
        print("Downloading", gpc_url)
        _fetch(gpc_url, gz)
        with zipfile.ZipFile(gz, "r") as zf:
            zf.extractall(root)
    _fetch(README_URL, root / "README_Elliott.txt")
    _fetch(SOURCES_URL, root / "Data_Sources.txt")
    n = len(list((root / "Rheo_Data").glob("*_rheo.dat")))
    print(f"OK: {n} rheo samples under {root / 'Rheo_Data'}")
    gpc_dir = root / "GPC_Data"
    if gpc_dir.exists():
        print(f"OK: {len(list(gpc_dir.glob('*_GPC.dat')))} GPC samples")


if __name__ == "__main__":
    main()
