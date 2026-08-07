# -*- coding: utf-8 -*-
"""UMN star→bottlebrush PLA graft SAOS loader (Zografos et al., DOI 10.13020/y7as-3w53).

Prefer archival CSVs under ``external_data/rheo_umn_bottlebrush/archival/``.
Master curves are TTS-shifted to T_ref = 86 °C (≈ Tg + 34 °C).
G', G'' in source files are in MPa; this loader returns Pa.

SEC in this deposit is chromatogram time–dRI, not a calibrated w(M) MWD —
do not treat as Leeds-style GPC for tube–MWD alignment without extra calibration.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "external_data" / "rheo_umn_bottlebrush"
DEFAULT_ARCHIVAL = DEFAULT_ROOT / "archival"
DOI = "10.13020/y7as-3w53"
CITE = "Zografos et al. ACS Macro Lett. / DRUM 10.13020/y7as-3w53"
T_REF_C = 86.0
DEFAULT_NSC = 72

_SAOS_NAME_RE = re.compile(
    r"SAOS Data_Nbb\s*=\s*(?P<nbb>\d+)(?:\s*,\s*Nsc\s*=\s*(?P<nsc>\d+))?",
    re.IGNORECASE,
)


@dataclass
class UmnSAOSSample:
    sample_id: str
    nbb: int
    nsc: int
    temperature_c: float
    omega: np.ndarray  # rad/s
    g_prime: np.ndarray  # Pa
    g_double_prime: np.ndarray  # Pa
    eta_star: np.ndarray  # Pa·s
    path: Path
    doi: str = DOI
    cite: str = CITE
    quality_ok: bool = True
    quality_note: str = ""

    @property
    def n_freq(self) -> int:
        return int(self.omega.size)

    def sorted(self) -> "UmnSAOSSample":
        order = np.argsort(self.omega)
        return UmnSAOSSample(
            sample_id=self.sample_id,
            nbb=self.nbb,
            nsc=self.nsc,
            temperature_c=self.temperature_c,
            omega=self.omega[order].copy(),
            g_prime=self.g_prime[order].copy(),
            g_double_prime=self.g_double_prime[order].copy(),
            eta_star=self.eta_star[order].copy(),
            path=self.path,
            doi=self.doi,
            cite=self.cite,
            quality_ok=self.quality_ok,
            quality_note=self.quality_note,
        )


def _parse_saos_meta(path: Path) -> tuple[int, int, str]:
    m = _SAOS_NAME_RE.search(path.stem)
    if not m:
        raise ValueError(f"cannot parse Nbb from SAOS filename: {path.name}")
    nbb = int(m.group("nbb"))
    nsc = int(m.group("nsc")) if m.group("nsc") else DEFAULT_NSC
    sid = f"Nbb{nbb}_Nsc{nsc}"
    return nbb, nsc, sid


def list_umn_saos_files(archival_dir: Optional[Path | str] = None) -> list[Path]:
    root = Path(archival_dir) if archival_dir is not None else DEFAULT_ARCHIVAL
    files = [p for p in root.glob("SAOS Data_Nbb*.csv") if p.is_file()]
    return sorted(files, key=lambda p: _parse_saos_meta(p)[0])


def load_umn_saos_file(path: Path | str) -> UmnSAOSSample:
    path = Path(path)
    nbb, nsc, sid = _parse_saos_meta(path)
    # Header: names; units; then data. Some rows pad empty log(aT)/ΔT cells.
    raw = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=2,
        usecols=(0, 1, 2, 3),
        dtype=np.float64,
        invalid_raise=False,
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    mask = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1]) & np.isfinite(raw[:, 2])
    raw = raw[mask]
    if raw.size == 0:
        raise ValueError(f"no SAOS rows in {path}")
    omega = raw[:, 0]
    g_mpa = raw[:, 1]
    gd_mpa = raw[:, 2]
    g_prime = g_mpa * 1e6  # MPa → Pa
    g_double = gd_mpa * 1e6
    eta = raw[:, 3]
    eta = np.where(np.isfinite(eta), eta, np.nan)
    # Source deposit: Nbb=210 sheet has G' values ~1e2–1e5 MPa (physically absurd for this melt).
    quality_ok = True
    quality_note = ""
    if float(np.nanmax(g_mpa)) > 500.0:
        quality_ok = False
        quality_note = "suspect G' units/columns in deposit (max G' > 500 MPa)"
    return UmnSAOSSample(
        sample_id=sid,
        nbb=nbb,
        nsc=nsc,
        temperature_c=T_REF_C,
        omega=omega,
        g_prime=g_prime,
        g_double_prime=g_double,
        eta_star=eta,
        path=path,
        quality_ok=quality_ok,
        quality_note=quality_note,
    ).sorted()


def load_umn_saos_all(archival_dir: Optional[Path | str] = None) -> list[UmnSAOSSample]:
    return [load_umn_saos_file(p) for p in list_umn_saos_files(archival_dir)]


def load_umn_sec_chromatograms(
    archival_dir: Optional[Path | str] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {sample_id: (time_min, dRI)} from SEC Data_All Data.csv.

    Not a calibrated MWD; for QC / peak shape only until Mw calibration is applied.
    """
    root = Path(archival_dir) if archival_dir is not None else DEFAULT_ARCHIVAL
    path = root / "SEC Data_All Data.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        next(reader, None)  # units / Time,dRI row
        rows = [row for row in reader if any(c.strip() for c in row)]

    labels: list[str] = []
    for i in range(0, len(header), 2):
        lab = (header[i] or "").strip()
        if not lab:
            continue
        m = re.search(r"Nbb\s*=\s*(\d+)", lab, re.I)
        if not m:
            continue
        nbb = int(m.group(1))
        m2 = re.search(r"Nsc\s*=\s*(\d+)", lab, re.I)
        nsc = int(m2.group(1)) if m2 else DEFAULT_NSC
        labels.append(f"Nbb{nbb}_Nsc{nsc}")

    ncol = max((len(r) for r in rows), default=0)
    data = np.full((len(rows), ncol), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue
            try:
                data[i, j] = float(cell)
            except ValueError:
                continue

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for j, sid in enumerate(labels):
        c0, c1 = 2 * j, 2 * j + 1
        if c1 >= data.shape[1]:
            break
        t = data[:, c0]
        dri = data[:, c1]
        mask = np.isfinite(t) & np.isfinite(dri)
        out[sid] = (t[mask].copy(), dri[mask].copy())
    return out
