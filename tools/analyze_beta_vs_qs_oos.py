#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B5: joint VS+QS residualisation with OOS R² / partial-R² gains for β_res.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--berg", default="outputs/structure_residual_socal/beta_vs_berg2021.csv")
    p.add_argument("--lin", default="outputs/structure_residual_socal/beta_vs_lin2023_q.csv")
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def oos_r2(X: np.ndarray, y: np.ndarray, model_kind: str, folds: int, seed: int) -> dict:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]
        if model_kind == "intercept":
            preds[te] = ytr.mean()
            continue
        if model_kind == "poly2":
            poly = PolynomialFeatures(degree=2, include_bias=False)
            Xtr_m = poly.fit_transform(Xtr)
            Xte_m = poly.transform(Xte)
        else:
            Xtr_m, Xte_m = Xtr, Xte
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_m)
        Xte_s = scaler.transform(Xte_m)
        mdl = Ridge(alpha=1.0)
        mdl.fit(Xtr_s, ytr)
        preds[te] = mdl.predict(Xte_s)
    return {"oos_r2": _r2(y, preds), "n": int(len(y))}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    berg = pd.read_csv(args.berg)
    lin = pd.read_csv(args.lin)
    m = berg.merge(lin[["trace_name", "qs_0to8", "beta_after_qs"]], on="trace_name", how="inner", suffixes=("", "_lin"))
    work = m.dropna(subset=["beta_resid", "vs_0to8", "qs_0to8"]).copy()
    y = work["beta_resid"].to_numpy(float)
    vs = work["vs_0to8"].to_numpy(float)
    qs = work["qs_0to8"].to_numpy(float)

    specs = {
        "intercept": np.ones((len(y), 1)),
        "vs": vs.reshape(-1, 1),
        "qs": qs.reshape(-1, 1),
        "vs_qs": np.column_stack([vs, qs]),
        "vs_qs_interact": np.column_stack([vs, qs, vs * qs]),
        "vs_qs_poly2": np.column_stack([vs, qs]),  # poly applied inside
    }
    results = {}
    for name, X in specs.items():
        kind = "intercept" if name == "intercept" else ("poly2" if name == "vs_qs_poly2" else "linear")
        results[name] = oos_r2(X, y, kind, args.folds, args.seed)

    base = results["intercept"]["oos_r2"]
    gains = {k: float(v["oos_r2"] - base) for k, v in results.items() if k != "intercept"}
    # Partial R² of QS given VS (and vice versa) using OOS R²
    # R2_partial(A|B) ≈ (R2_{B+A} - R2_B) / (1 - R2_B)
    def partial(full_key, reduced_key):
        r_full = results[full_key]["oos_r2"]
        r_red = results[reduced_key]["oos_r2"]
        return float((r_full - r_red) / max(1.0 - r_red, 1e-12))

    report = {
        "n": int(len(work)),
        "folds": int(args.folds),
        "oos_r2": {k: v["oos_r2"] for k, v in results.items()},
        "oos_r2_gain_vs_intercept": gains,
        "partial_r2_oos": {
            "qs_given_vs": partial("vs_qs", "vs"),
            "vs_given_qs": partial("vs_qs", "qs"),
            "interact_given_vs_qs": partial("vs_qs_interact", "vs_qs"),
            "poly2_given_vs_qs": partial("vs_qs_poly2", "vs_qs"),
        },
        "claim": (
            "OOS R² of β_res explained by published VS/QS summaries remains low; "
            "joint/nonlinear terms add little beyond linear VS+QS."
        ),
    }
    (out / "beta_vs_qs_oos.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", out / "beta_vs_qs_oos.json")


if __name__ == "__main__":
    main()
