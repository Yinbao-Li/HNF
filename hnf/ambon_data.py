# -*- coding: utf-8 -*-
"""Parse Ambon Mendeley catalog and station/velocity tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from hnf.inversion_1d import LayeredEarth1D


@dataclass
class AmbonStation:
    code: str
    longitude: float
    latitude: float


@dataclass
class AmbonEvent:
    year: int
    month: int
    day: int
    longitude: float
    latitude: float
    depth_km: float


def _ambon_root() -> Path:
    return Path(__file__).resolve().parents[1] / "external_data" / "ambon_mendeley"


def load_ambon_stations(root: Path | None = None) -> list[AmbonStation]:
    root = root or _ambon_root()
    path = root / "[01]Station_Velocity_Model_Datainbrief_Ambon.xlsx"
    df = pd.read_excel(path, sheet_name="Station Data", header=None)
    header = [str(x).strip().lower() for x in df.iloc[1].tolist()]
    lon_i = header.index("longitude")
    lat_i = header.index("latitude")
    code_i = header.index("network") if "network" in header else 4
    stations: list[AmbonStation] = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        lon = row.iloc[lon_i]
        lat = row.iloc[lat_i]
        if pd.isna(lon) or pd.isna(lat):
            continue
        code = str(row.iloc[code_i]).strip() if not pd.isna(row.iloc[code_i]) else f"S{i}"
        if code.lower() == "nan" or not code:
            code = f"S{i}"
        stations.append(AmbonStation(code=code, longitude=float(lon), latitude=float(lat)))
    return stations


def load_ambon_velocity_model(root: Path | None = None, use_velest: bool = True) -> LayeredEarth1D:
    """Load layered vp/vs from Ambon xlsx (AK135 or VELEST updated section)."""
    root = root or _ambon_root()
    path = root / "[01]Station_Velocity_Model_Datainbrief_Ambon.xlsx"
    vm = pd.read_excel(path, sheet_name="Velocity Model", header=None)
    start = 11 if use_velest else 2
    depths = [0.0]
    vp_list, vs_list = [], []
    for i in range(start + 1, len(vm)):
        d, vp, vs = vm.iloc[i, 0], vm.iloc[i, 1], vm.iloc[i, 2]
        if pd.isna(d) or pd.isna(vp):
            break
        depths.append(float(d))
        vp_list.append(float(vp))
        vs_list.append(float(vs))
    if not vp_list:
        raise ValueError("No velocity layers parsed from Ambon xlsx")
    q = torch.full((len(vp_list),), 150.0)
    return LayeredEarth1D(
        depths=torch.tensor(depths, dtype=torch.float32),
        vp=torch.tensor(vp_list, dtype=torch.float32),
        vs=torch.tensor(vs_list, dtype=torch.float32),
        q=q,
    )


def load_ambon_events(root: Path | None = None, sheet: str = "Updated Hypocenter VELEST") -> list[AmbonEvent]:
    root = root or _ambon_root()
    path = root / "[02]Catalog_Hypocenter_Datainbrief_Ambon.xlsx"
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    events: list[AmbonEvent] = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        if pd.isna(row.iloc[0]):
            continue
        try:
            events.append(
                AmbonEvent(
                    year=int(row.iloc[0]),
                    month=int(row.iloc[1]),
                    day=int(row.iloc[2]),
                    longitude=float(row.iloc[6]),
                    latitude=float(row.iloc[7]),
                    depth_km=float(row.iloc[8]),
                )
            )
        except (TypeError, ValueError):
            continue
    return events


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance (km) on WGS84 sphere."""
    import math

    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
