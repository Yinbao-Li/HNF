#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inventory external rheology corpora vs Leeds (DOI 10.5518/1689)."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "outputs" / "rheo" / "external_inventory"


def main() -> None:
    leeds_rheo = sorted((_REPO / "external_data/rheo_leeds_ps/Rheo_Data").glob("*_rheo.dat"))
    leeds_gpc = sorted((_REPO / "external_data/rheo_leeds_ps/GPC_Data").glob("*_GPC.dat"))
    reptate_ps = _REPO / "external_data/RepTate/data/PS_Linear_Polydisperse"
    reptate_ids = []
    if reptate_ps.is_dir():
        for gpc in sorted(reptate_ps.glob("*.gpc")):
            if "_header" in gpc.stem:
                continue
            sid = gpc.stem.upper()
            tts = reptate_ps / f"{gpc.stem}.tts"
            reptate_ids.append(
                {
                    "sample_id": sid,
                    "gpc": str(gpc.relative_to(_REPO)),
                    "tts": str(tts.relative_to(_REPO)) if tts.is_file() else None,
                    "already_in_leeds": ( _REPO / f"external_data/rheo_leeds_ps/Rheo_Data/{sid}_rheo.dat").is_file(),
                }
            )

    umn_root = _REPO / "external_data" / "rheo_umn_bottlebrush"
    umn_arch = umn_root / "archival"
    umn_saos = sorted(umn_arch.glob("SAOS Data_Nbb*.csv")) if umn_arch.is_dir() else []
    umn_status = "present" if umn_saos else ("partial" if umn_root.is_dir() else "missing")

    report = {
        "leeds": {
            "n_saos": len(leeds_rheo),
            "n_gpc": len(leeds_gpc),
            "samples": [p.stem.replace("_rheo", "") for p in leeds_rheo],
            "path": "external_data/rheo_leeds_ps",
            "doi": "10.5518/1689",
        },
        "reptate_ps_linear_polydisperse": {
            "path": "external_data/RepTate/data/PS_Linear_Polydisperse",
            "pairs": reptate_ids,
            "note": "PS1–PS3 TTS+GPC overlap Leeds (same BASF/RepTate lineage); adds 0 new PS melts.",
        },
        "umn_star_bottlebrush": {
            "doi": "10.13020/y7as-3w53",
            "path": "external_data/rheo_umn_bottlebrush",
            "status": umn_status,
            "n_saos_csv": len(umn_saos),
            "saos_files": [p.name for p in umn_saos],
            "sec": "chromatogram time–dRI (not calibrated w(M)); not Leeds-equivalent GPC",
            "chemistry": "PLA graft copolymers; unentangled star→bottlebrush (Nsc≈72)",
            "note": "Useful LVE diversity; does not enlarge linear-PS tube–MWD n.",
        },
        "next": [
            "dig Leeds: harden n=9 Prony↔tube–MWD claim",
            "optional: UMN SAOS board + star/bottlebrush regime labels (not tube–MWD)",
            "RepTate PI_LINEAR: TTS without paired GPC — skip for MWD alignment",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INVENTORY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[wrote] {OUT / 'INVENTORY.json'}")


if __name__ == "__main__":
    main()
