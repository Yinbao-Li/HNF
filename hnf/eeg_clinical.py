# -*- coding: utf-8 -*-
"""Clinical-breakthrough helpers for Domain II AD/FTD EEG.

Target: evidence that *materially helps* differential diagnosis / staging /
interpretable markers — not Stage-1 smoke-test metrics alone.

Checklist (see `.cursor/rules/eeg-clinical-standards.mdc`):
  1. True HC / FTD / AD taxonomy (class-1 = FTD, never hide as MCI)
  2. Align Age / Gender / MMSE; show incremental value beyond demographics
  3. Subject-level primary endpoints
  4. FDR-controlled interpretable marker mining
  5. Decision-relevant operating points + AD↔FTD confusion
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hnf.eeg_dataset import CLINICAL_ID_TO_LABEL, CLINICAL_LABEL_TO_ID


FEATURE_BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (rejected, q_values) for BH-FDR at level ``alpha``."""
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    n = p.size
    if n == 0:
        return np.asarray([], dtype=bool), np.asarray([], dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1, dtype=np.float64))
    # enforce monotonicity from the back
    q_rev = np.minimum.accumulate(q[::-1])[::-1]
    q_full = np.empty_like(q_rev)
    q_full[order] = np.clip(q_rev, 0.0, 1.0)
    rejected = q_full <= alpha
    return rejected, q_full


def welch_band_powers(
    x: np.ndarray,
    sample_rate: float,
    *,
    bands: tuple[tuple[str, float, float], ...] = FEATURE_BANDS_HZ,
) -> dict[str, float]:
    """Mean log-band-power over channels for a single epoch ``(C, T)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected (C,T), got {x.shape}")
    c, t = x.shape
    # Hann-windowed periodogram (no SciPy dependency)
    window = np.hanning(t)
    spec = np.fft.rfft(x * window[None, :], axis=1)
    power = (np.abs(spec) ** 2) / max(float(np.sum(window**2)), 1e-12)
    freqs = np.fft.rfftfreq(t, d=1.0 / float(sample_rate))
    out: dict[str, float] = {}
    for name, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            out[f"bp_{name}"] = float("nan")
            continue
        # geometric mean across channels of band-integrated power
        band = power[:, mask].sum(axis=1)
        out[f"bp_{name}"] = float(np.log10(np.maximum(band.mean(), 1e-20)))
    # clinically common ratios
    if np.isfinite(out.get("bp_theta", np.nan)) and np.isfinite(out.get("bp_alpha", np.nan)):
        out["theta_alpha_ratio"] = float(out["bp_theta"] - out["bp_alpha"])
    else:
        out["theta_alpha_ratio"] = float("nan")
    return out


def rho_summaries(rho: np.ndarray) -> dict[str, float]:
    """Summaries of ``rho`` shaped ``(T,)`` or ``(T,1)``."""
    r = np.asarray(rho, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return {
            "rho_mean": float("nan"),
            "rho_std": float("nan"),
            "rho_p90": float("nan"),
            "rho_cv": float("nan"),
        }
    mean = float(r.mean())
    std = float(r.std())
    return {
        "rho_mean": mean,
        "rho_std": std,
        "rho_p90": float(np.quantile(r, 0.9)),
        "rho_cv": float(std / (abs(mean) + 1e-8)),
    }


def epoch_feature_vector(
    x: np.ndarray,
    rho: Optional[np.ndarray],
    *,
    sample_rate: float,
    mean_omega: float = 0.0,
) -> dict[str, float]:
    """Interpretable per-epoch features (band power + ρ summaries)."""
    feats = welch_band_powers(x, sample_rate)
    if rho is not None:
        feats.update(rho_summaries(rho))
        feats["omega_rho"] = float(mean_omega) * float(feats.get("rho_mean", 0.0))
    else:
        feats.update(
            {
                "rho_mean": float("nan"),
                "rho_std": float("nan"),
                "rho_p90": float("nan"),
                "rho_cv": float("nan"),
                "omega_rho": float("nan"),
            }
        )
    return feats


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yi, pi in zip(y.tolist(), pred.tolist()):
        if 0 <= yi < n_classes and 0 <= pi < n_classes:
            cm[yi, pi] += 1
    return cm


def binary_operating_points(
    y_pos: np.ndarray,
    score: np.ndarray,
    *,
    target_sens: tuple[float, ...] = (0.70, 0.80, 0.90),
) -> dict:
    """Sensitivity / specificity curve + Youden and fixed-sensitivity points."""
    y = np.asarray(y_pos, dtype=np.int64).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.min() == y.max():
        return {"n": int(y.size), "auc": float("nan"), "points": []}
    order = np.argsort(-s)
    y_ord = y[order]
    # thresholds = unique scores midpoints
    thr = s[order]
    tps = np.cumsum(y_ord == 1)
    fps = np.cumsum(y_ord == 0)
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    sens = tps / P
    spec = 1.0 - fps / N
    youden = sens + spec - 1.0
    i_best = int(np.argmax(youden))
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, s))
    except Exception:
        auc = float("nan")

    def _pt(i: int, name: str) -> dict:
        return {
            "name": name,
            "threshold": float(thr[i]),
            "sensitivity": float(sens[i]),
            "specificity": float(spec[i]),
            "youden": float(youden[i]),
        }

    points = [_pt(i_best, "youden")]
    for ts in target_sens:
        ok = np.where(sens >= ts)[0]
        if ok.size == 0:
            points.append(
                {
                    "name": f"sens>={ts:.2f}",
                    "threshold": float("nan"),
                    "sensitivity": float("nan"),
                    "specificity": float("nan"),
                    "youden": float("nan"),
                }
            )
        else:
            # among those meeting sens, pick highest specificity
            j = ok[int(np.argmax(spec[ok]))]
            points.append(_pt(int(j), f"sens>={ts:.2f}"))
    return {"n": int(y.size), "n_pos": int(P), "n_neg": int(N), "auc": auc, "points": points}


def one_way_anova_pvalue(groups: list[np.ndarray]) -> float:
    groups = [np.asarray(g, dtype=np.float64).reshape(-1) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size > 0]
    if len(groups) < 2:
        return float("nan")
    all_v = np.concatenate(groups)
    grand = all_v.mean()
    ss_b = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b = len(groups) - 1
    df_w = int(all_v.size - len(groups))
    if df_w <= 0 or ss_w <= 0:
        return float("nan")
    f = (ss_b / df_b) / (ss_w / df_w)
    try:
        from scipy.stats import f as f_dist

        return float(f_dist.sf(f, df_b, df_w))
    except Exception:
        return float("nan")


def ridge_r2(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0) -> float:
    """Leave-in R² for a ridge model (small-N descriptive; not CV)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if y.size < 5:
        return float("nan")
    Xn = X - X.mean(axis=0, keepdims=True)
    sd = Xn.std(axis=0, keepdims=True) + 1e-8
    Xn = Xn / sd
    A = Xn.T @ Xn + l2 * np.eye(Xn.shape[1])
    b = Xn.T @ (y - y.mean())
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("nan")
    pred = Xn @ w + y.mean()
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def logistic_acc_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    n_steps: int = 400,
    lr: float = 0.1,
) -> dict[str, float]:
    """Simple L2-logistic (GD) accuracy + AUC for binary y∈{0,1}."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if y.size < 8 or y.min() == y.max():
        return {"n": float(y.size), "acc": float("nan"), "auc": float("nan")}
    Xn = X - X.mean(axis=0, keepdims=True)
    Xn = Xn / (Xn.std(axis=0, keepdims=True) + 1e-8)
    Xb = np.c_[np.ones(len(Xn)), Xn]
    w = np.zeros(Xb.shape[1], dtype=np.float64)
    for _ in range(n_steps):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= lr * grad
    p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
    pred = (p >= 0.5).astype(np.float64)
    acc = float((pred == y).mean())
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = float("nan")
    return {"n": float(y.size), "acc": acc, "auc": auc}


def clinical_label_name(label_id: int) -> str:
    return CLINICAL_ID_TO_LABEL.get(int(label_id), f"cls{label_id}")


def ad_vs_ftd_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask keeping only AD and FTD subjects."""
    y = np.asarray(labels, dtype=np.int64)
    return (y == CLINICAL_LABEL_TO_ID["AD"]) | (y == CLINICAL_LABEL_TO_ID["FTD"])
