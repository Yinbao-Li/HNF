# -*- coding: utf-8 -*-
"""Leeds PS SAOS loader (Elliott et al. 2025, DOI 10.5518/1689).

Each ``*_rheo.dat`` file:
  line 1  — measurement / reference temperature (°C)
  lines 2+ — tab-delimited ``ω  G'  G''`` (rad/s, Pa, Pa)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "external_data" / "rheo_leeds_ps" / "Rheo_Data"

# Literature sources (Elliott et al. Data_Sources.txt)
SAMPLE_META: dict[str, dict[str, str]] = {
    "PS1": {"source": "BASF / Reptate", "cite": "Boudara et al. J. Rheol. 2020"},
    "PS2": {"source": "BASF / Reptate", "cite": "Boudara et al. J. Rheol. 2020"},
    "PS3": {"source": "BASF / Reptate", "cite": "Boudara et al. J. Rheol. 2020"},
    "M1": {"source": "Wasserman & Graessley", "cite": "J. Rheol. 1992"},
    "M2": {"source": "Wasserman & Graessley", "cite": "J. Rheol. 1992"},
    "PScom": {"source": "Wasserman & Graessley", "cite": "J. Rheol. 1992"},
    "PSA": {"source": "Sugimoto et al.", "cite": "J. Polym. Sci. B 2009"},
    "A1PS": {"source": "Ferri & Lomellini", "cite": "J. Rheol. 1999"},
    "PS8": {"source": "Montfort et al.", "cite": "Rheol. Acta 1979"},
}


@dataclass
class LeedsSAOSSample:
    sample_id: str
    temperature_c: float
    omega: np.ndarray  # (N,) rad/s
    g_prime: np.ndarray  # (N,) Pa
    g_double_prime: np.ndarray  # (N,) Pa
    path: Path
    source: str = ""
    cite: str = ""

    @property
    def n_freq(self) -> int:
        return int(self.omega.size)

    def sorted(self) -> "LeedsSAOSSample":
        order = np.argsort(self.omega)
        return LeedsSAOSSample(
            sample_id=self.sample_id,
            temperature_c=self.temperature_c,
            omega=self.omega[order].copy(),
            g_prime=self.g_prime[order].copy(),
            g_double_prime=self.g_double_prime[order].copy(),
            path=self.path,
            source=self.source,
            cite=self.cite,
        )


def _parse_sample_id(path: Path) -> str:
    name = path.stem
    if name.endswith("_rheo"):
        name = name[: -len("_rheo")]
    return name


def load_leeds_saos_file(path: Path | str) -> LeedsSAOSSample:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty rheology file: {path}")
    temperature_c = float(lines[0].strip().replace(",", "."))
    rows: list[list[float]] = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"no ω/G'/G'' rows in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    sid = _parse_sample_id(path)
    meta = SAMPLE_META.get(sid, {})
    return LeedsSAOSSample(
        sample_id=sid,
        temperature_c=temperature_c,
        omega=arr[:, 0],
        g_prime=arr[:, 1],
        g_double_prime=arr[:, 2],
        path=path,
        source=str(meta.get("source", "")),
        cite=str(meta.get("cite", "")),
    ).sorted()


def list_leeds_saos_files(data_dir: Optional[Path | str] = None) -> list[Path]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return sorted(root.glob("*_rheo.dat"))


def load_leeds_saos_all(data_dir: Optional[Path | str] = None) -> list[LeedsSAOSSample]:
    return [load_leeds_saos_file(p) for p in list_leeds_saos_files(data_dir)]
