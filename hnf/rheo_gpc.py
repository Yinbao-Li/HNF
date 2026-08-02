# -*- coding: utf-8 -*-
"""Leeds GPC / MWD loader + moments (Elliott et al. 2025 companion data)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_GPC_DIR = (
    Path(__file__).resolve().parents[1] / "external_data" / "rheo_leeds_ps" / "GPC_Data"
)


@dataclass
class LeedsGPCSample:
    sample_id: str
    M: np.ndarray  # g/mol
    w: np.ndarray  # differential weight intensity (arbitrary units)
    path: Path

    @property
    def logM(self) -> np.ndarray:
        return np.log10(np.maximum(self.M, 1.0))


def _parse_sample_id(path: Path) -> str:
    name = path.stem
    if name.endswith("_GPC"):
        name = name[: -len("_GPC")]
    return name


def load_leeds_gpc_file(path: Path | str) -> LeedsGPCSample:
    path = Path(path)
    rows: list[list[float]] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        rows.append([float(parts[0]), float(parts[1])])
    if len(rows) < 3:
        raise ValueError(f"too few GPC rows in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    M = arr[order, 0]
    w = np.maximum(arr[order, 1], 0.0)
    return LeedsGPCSample(sample_id=_parse_sample_id(path), M=M, w=w, path=path)


def load_leeds_gpc_all(data_dir: Optional[Path | str] = None) -> list[LeedsGPCSample]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_GPC_DIR
    return [load_leeds_gpc_file(p) for p in sorted(root.glob("*_GPC.dat"))]


def mwd_moments(sample: LeedsGPCSample, *, m_min: float = 500.0) -> dict[str, float]:
    """Weight-average moments treating w as dw/dlog10(M) intensity.

    Uses trapezoidal integration on log10(M). Masks M < m_min to drop GPC tails.
    """
    M = sample.M.astype(np.float64)
    w = sample.w.astype(np.float64)
    mask = M >= float(m_min)
    M, w = M[mask], w[mask]
    if M.size < 3:
        raise ValueError(f"{sample.sample_id}: insufficient points after mask")
    logM = np.log10(M)
    # normalize density on logM
    area = float(np.trapz(w, logM))
    if area <= 0:
        raise ValueError(f"{sample.sample_id}: non-positive GPC area")
    w_n = w / area

    # Mw = ∫ M w dlogM / ∫ w dlogM  (∫w=1)
    Mw = float(np.trapz(M * w_n, logM))
    # Mn = 1 / ∫ (1/M) w dlogM
    inv_Mn = float(np.trapz(w_n / M, logM))
    Mn = 1.0 / max(inv_Mn, 1e-30)
    # Mz = ∫ M^2 w / ∫ M w
    M2 = float(np.trapz((M ** 2) * w_n, logM))
    Mz = M2 / max(Mw, 1e-30)
    D = Mw / max(Mn, 1e-30)

    # shape descriptors on normalized density (cumulative trapz)
    c = np.zeros_like(w_n)
    for i in range(1, len(w_n)):
        c[i] = c[i - 1] + 0.5 * (w_n[i] + w_n[i - 1]) * (logM[i] - logM[i - 1])
    c = c / max(c[-1], 1e-30)

    def _pct(p: float) -> float:
        return float(np.interp(p, c, logM))

    logM_peak = float(logM[int(np.argmax(w_n))])
    return {
        "Mn": Mn,
        "Mw": Mw,
        "Mz": Mz,
        "D": D,
        "log10_Mn": float(np.log10(Mn)),
        "log10_Mw": float(np.log10(Mw)),
        "log10_Mz": float(np.log10(Mz)),
        "logM_peak": logM_peak,
        "logM_p10": _pct(0.10),
        "logM_p50": _pct(0.50),
        "logM_p90": _pct(0.90),
        "logM_width90": _pct(0.90) - _pct(0.10),
        "n_points": float(M.size),
    }


def mwd_on_log_grid(
    sample: LeedsGPCSample,
    logM_grid: np.ndarray,
    *,
    m_min: float = 500.0,
) -> np.ndarray:
    """Interpolate normalized dw/dlog10M onto a common logM grid."""
    M = sample.M
    w = sample.w
    mask = M >= float(m_min)
    M, w = M[mask], w[mask]
    logM = np.log10(M)
    area = float(np.trapz(w, logM))
    w_n = w / max(area, 1e-30)
    # interp; outside → 0
    out = np.interp(logM_grid, logM, w_n, left=0.0, right=0.0)
    # renormalize on grid
    a = float(np.trapz(out, logM_grid))
    if a > 0:
        out = out / a
    return out
