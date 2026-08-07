# -*- coding: utf-8 -*-
"""PolyWeight experimental PS pair (Laun et al. via Minotto PolyWeight).

Only ``data/experimental/`` counts for plan A. Synthetic datasets are model-generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "external_data"
    / "rheo_extra_pairs"
    / "PolyWeight"
)
CITE = (
    "PolyWeight experimental PS (Laun et al. 2005 data; material params "
    "Nobile & Cocchini 2008 via Minotto PolyWeight)"
)
# materials/Experimental line "105" is used as SAOS reference temperature (°C)
T_REF_C = 105.0


@dataclass
class PolyWeightPair:
    sample_id: str
    temperature_c: float
    omega: np.ndarray
    g_prime: np.ndarray
    g_double_prime: np.ndarray
    M: np.ndarray
    w: np.ndarray  # dW/dlogM-like; negatives clipped to 0
    path_g12: Path
    path_mwd: Path
    cite: str = CITE


def load_polyweight_experimental(
    root: Optional[Path | str] = None,
) -> PolyWeightPair:
    root = Path(root) if root is not None else DEFAULT_ROOT
    g12 = root / "data" / "experimental" / "G12_experimental.txt"
    mwd = root / "data" / "experimental" / "MWD_experimental.txt"
    if not g12.is_file() or not mwd.is_file():
        raise FileNotFoundError(f"missing PolyWeight experimental files under {root}")

    g = np.loadtxt(g12, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] < 3:
        raise ValueError(f"bad G12 format: {g12}")
    order = np.argsort(g[:, 0])
    omega = g[order, 0]
    gp = g[order, 1]
    gpp = g[order, 2]

    mw = np.loadtxt(mwd, dtype=np.float64)
    order_m = np.argsort(mw[:, 0])
    M = mw[order_m, 0]
    w = np.maximum(mw[order_m, 1], 0.0)

    return PolyWeightPair(
        sample_id="PW_LaunPS",
        temperature_c=T_REF_C,
        omega=omega,
        g_prime=gp,
        g_double_prime=gpp,
        M=M,
        w=w,
        path_g12=g12,
        path_mwd=mwd,
    )


def export_leeds_format(pair: PolyWeightPair, out_dir: Path | str) -> tuple[Path, Path]:
    """Write Leeds-compatible ``*_rheo.dat`` / ``*_GPC.dat`` for unified pipelines."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rheo = out / f"{pair.sample_id}_rheo.dat"
    gpc = out / f"{pair.sample_id}_GPC.dat"
    with rheo.open("w", encoding="utf-8") as f:
        f.write(f"{pair.temperature_c:g}\n")
        for w, a, b in zip(pair.omega, pair.g_prime, pair.g_double_prime):
            f.write(f"{w:.10g}\t{a:.10g}\t{b:.10g}\n")
    with gpc.open("w", encoding="utf-8") as f:
        for M, wi in zip(pair.M, pair.w):
            f.write(f"{M:.10g}\t{wi:.10g}\n")
    (out / "SOURCE.txt").write_text(
        f"{pair.cite}\nT_ref_C={pair.temperature_c}\nsource_files=\n  {pair.path_g12}\n  {pair.path_mwd}\n",
        encoding="utf-8",
    )
    return rheo, gpc
