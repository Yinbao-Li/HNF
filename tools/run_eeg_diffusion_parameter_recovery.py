#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEG diffusion-kernel parameter recovery on synthetic epoch correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
import sys

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.eeg_geometry import electrode_xyz
from hnf.eeg_subject_diffusion import diffusion_kernel_from_vec, fit_subject_diffusion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/eeg/diffusion_parameter_recovery")
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--n-epochs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    xyz = electrode_xyz().astype(np.float64)

    rows = []
    for i in range(int(args.n_trials)):
        # Cholesky-ish vector for SPD D (matches diffusion_kernel_from_vec layout).
        vec_true = np.array(
            [
                rng.uniform(0.05, 0.5),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-2.0, -0.3),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-2.0, -0.3),
                rng.uniform(-1.5, -0.2),
            ],
            dtype=np.float64,
        )
        K, D_true, _ = diffusion_kernel_from_vec(xyz, vec_true)
        C = 0.5 * (K + K.T)
        np.fill_diagonal(C, 1.0)
        w, v = np.linalg.eigh(C + 0.05 * np.eye(19))
        w = np.clip(w, 1e-4, None)
        A = v @ np.diag(np.sqrt(w))
        epochs = np.stack([A @ rng.standard_normal((19, 256)) for _ in range(args.n_epochs)])
        fit = fit_subject_diffusion(epochs, xyz=xyz, maxiter=120)
        eig_true = np.sort(np.linalg.eigvalsh(D_true))
        eig_hat = np.array([fit["D_eig0"], fit["D_eig1"], fit["D_eig2"]], float)
        # Align by sorting
        eig_hat = np.sort(eig_hat)
        rel = np.abs(eig_hat - eig_true) / np.maximum(np.abs(eig_true), 1e-6)
        rows.append(
            {
                "trial": i,
                "fit_r": float(fit["fit_r"]),
                "D_aniso_true": float(eig_true[-1] / max(eig_true[0], 1e-6)),
                "D_aniso_hat": float(fit["D_aniso"]),
                "eig_rel_err_mean": float(rel.mean()),
                "eig_rel_err_max": float(rel.max()),
            }
        )

    arr = lambda k: np.asarray([r[k] for r in rows], float)
    report = {
        "n_trials": int(args.n_trials),
        "summary": {
            "fit_r": {"mean": float(arr("fit_r").mean()), "median": float(np.median(arr("fit_r")))},
            "eig_rel_err_mean": {
                "mean": float(arr("eig_rel_err_mean").mean()),
                "median": float(np.median(arr("eig_rel_err_mean"))),
                "p90": float(np.percentile(arr("eig_rel_err_mean"), 90)),
            },
            "D_aniso_spearman": float(
                __import__("scipy").stats.spearmanr(arr("D_aniso_true"), arr("D_aniso_hat")).statistic
            ),
        },
        "rows": rows,
        "note": "Recovers anisotropic diffusion eigenvalues / anisotropy ratio from synthetic epoch correlations.",
    }
    (out / "RECOVERY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Wrote", out / "RECOVERY.json")


if __name__ == "__main__":
    main()
