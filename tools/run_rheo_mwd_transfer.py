#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpretable spectrum→MWD transfer vs Elliott NN (Leeds PS).

1. Fit Elliott-compatible Maxwell modes (WLF→180°C, 81 τ).
2. Black-box: ensemble of 9 published Keras models.
3. Interpretable zero-shot: tube projection G_k → MWD via M∝τ^{1/3.4}.
4. Interpretable pretrain: synthetic lognormal-mix MWDs → approximate G via
   entanglement scaling; Ridge logG→MWD weights; zero-shot on real samples.
5. Few-shot: leave-one-out Ridge on real logG→moments (optional sanity).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.integrate import simpson
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from hnf.rheo_elliott_compat import (
    MWD_MEANS,
    MWD_SIGMA,
    MWD_X,
    NUM_MWD_PARAMS,
    TAU_VALUES,
    fit_sample_elliott,
    lognormal_pdf,
    mwd_from_weights,
    mwd_moments_on_grid,
    normalize_mwd_on_x,
)
from hnf.rheo_gpc import load_leeds_gpc_all
from hnf.rheo_leeds import load_leeds_saos_all

DEFAULT_NN = _REPO_ROOT / "external_data" / "rheo_leeds_ps" / "NN_Models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--nn-dir", default=str(DEFAULT_NN))
    p.add_argument("--output-dir", default="outputs/rheo/mwd_transfer")
    p.add_argument("--n-synth", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-nn", action="store_true")
    return p.parse_args()


def rmse_mwd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def tau_of_M(M: np.ndarray, *, Me: float = 12870.0, tau_e: float = 2.2e-4, alpha: float = 3.4) -> np.ndarray:
    """Simplified entanglement time τ ≈ τ_e (M/Me)^α (linear tube heuristic)."""
    return tau_e * (np.maximum(M, Me) / Me) ** alpha


def tube_project_mwd(g: np.ndarray, tau: np.ndarray = TAU_VALUES, x: np.ndarray = MWD_X) -> np.ndarray:
    """Zero-shot: deposit entangled modal mass at M(τ) onto MWD grid."""
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    tau = np.asarray(tau, dtype=np.float64)
    Me, tau_e, alpha = 12870.0, 2.2e-4, 3.4
    # Keep only entangled (and slower) modes — glassy short-τ dump mass at Me
    ent = tau >= tau_e
    if not np.any(ent):
        ent = np.ones_like(tau, dtype=bool)
    g = g.copy()
    g[~ent] = 0.0
    w = g / max(g.sum(), 1e-30)
    M_of_tau = Me * (np.maximum(tau, tau_e) / tau_e) ** (1.0 / alpha)
    y = np.zeros_like(x, dtype=np.float64)
    sigma = 0.45
    for wi, Mi in zip(w, M_of_tau):
        if wi <= 0:
            continue
        y += wi * lognormal_pdf(x, np.log(Mi), sigma)
    area = simpson(y, x=np.log(x))
    if area > 0:
        y /= area
    return y


def sample_synth_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random sparse-ish positive weights on Elliott's 28 lognormals."""
    W = np.zeros((n, NUM_MWD_PARAMS), dtype=np.float64)
    for i in range(n):
        n_comp = int(rng.integers(2, 8))
        idx = rng.choice(NUM_MWD_PARAMS, size=n_comp, replace=False)
        amps = rng.lognormal(mean=0.0, sigma=0.8, size=n_comp)
        center = NUM_MWD_PARAMS // 2
        amps *= np.exp(-0.02 * (idx - center) ** 2)
        W[i, idx] = amps
        W[i] /= max(W[i].sum(), 1e-30)
    return W


def forward_G_from_weights(weights: np.ndarray, tau: np.ndarray = TAU_VALUES) -> np.ndarray:
    """Approximate Maxwell amplitudes from MWD mixture weights via tube bins."""
    x = MWD_X
    y = mwd_from_weights(weights, x)
    logx = np.log(x)
    taus = tau_of_M(x)
    g = np.zeros(len(tau), dtype=np.float64)
    dlog = np.empty_like(logx)
    dlog[1:-1] = 0.5 * (logx[2:] - logx[:-2])
    dlog[0] = logx[1] - logx[0]
    dlog[-1] = logx[-1] - logx[-2]
    for j in range(len(x)):
        if taus[j] < 2.2e-4:
            continue
        k = int(np.argmin(np.abs(np.log(tau) - np.log(taus[j]))))
        g[k] += y[j] * dlog[j]
    g = np.maximum(g, 0.0)
    if g.sum() <= 0:
        g[:] = 1e-8
    g = g / g.sum() * 1e5
    return g


def train_synth_ridge(
    n_synth: int,
    seed: int,
    noise_std: float = 0.15,
) -> Ridge:
    rng = np.random.default_rng(seed)
    W = sample_synth_weights(rng, n_synth)
    X = np.zeros((n_synth, len(TAU_VALUES)), dtype=np.float64)
    for i in range(n_synth):
        g = forward_G_from_weights(W[i])
        logg = np.log10(np.maximum(g, 1e-12))
        logg = logg + rng.normal(0.0, noise_std, size=logg.shape)
        X[i] = logg
    # Multi-output ridge: predict 28 weights
    model = MultiOutputRegressor(Ridge(alpha=10.0, fit_intercept=True))
    model.fit(X, W)
    return model


def predict_weights_ridge(model, log_g: np.ndarray) -> np.ndarray:
    x = (log_g / np.log(10.0)).reshape(1, -1)  # log10 g
    w = model.predict(x)[0]
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def load_elliott_ensemble(nn_dir: Path):
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    models = []
    for i in range(1, 10):
        path = nn_dir / f"PS_Model_{i}.keras"
        models.append(tf.keras.models.load_model(str(path)))
    return models


def predict_elliott_ensemble(models, nn_input: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    preds = []
    for m in models:
        w = np.abs(m.predict(nn_input, verbose=0)[0])
        preds.append(mwd_from_weights(w))
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), preds


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saos = {s.sample_id: s for s in load_leeds_saos_all(args.rheo_dir)}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all(args.gpc_dir)}
    ids = sorted(set(saos) & set(gpc))

    print("Fitting Elliott-compatible Maxwell modes...")
    fits = {}
    for sid in ids:
        s = saos[sid]
        fits[sid] = fit_sample_elliott(sid, s.temperature_c, s.omega, s.g_prime, s.g_double_prime)
        print(f"  {sid}: T={s.temperature_c:.0f}C  Nω={s.n_freq}  Gsum={fits[sid].g.sum():.3g}")

    print(f"Training synthetic Ridge on {args.n_synth} MWDs...")
    ridge = train_synth_ridge(args.n_synth, args.seed)

    models = None
    if not args.skip_nn:
        print("Loading Elliott NN ensemble...")
        models = load_elliott_ensemble(Path(args.nn_dir))

    rows = []
    for sid in ids:
        fit = fits[sid]
        gpc_y = normalize_mwd_on_x(gpc[sid].M, gpc[sid].w, MWD_X)
        gt_mom = mwd_moments_on_grid(MWD_X, gpc_y)

        # tube zero-shot
        y_tube = tube_project_mwd(fit.g, fit.tau)
        mom_tube = mwd_moments_on_grid(MWD_X, y_tube)

        # synth ridge zero-shot
        w_ridge = predict_weights_ridge(ridge, fit.log_g)
        y_ridge = mwd_from_weights(w_ridge)
        mom_ridge = mwd_moments_on_grid(MWD_X, y_ridge)

        row = {
            "sample": sid,
            "gt": gt_mom,
            "tube": {
                "rmse": rmse_mwd(gpc_y, y_tube),
                **{f"pred_{k}": v for k, v in mom_tube.items()},
                "abs_err_logMw": abs(mom_tube["log10_Mw"] - gt_mom["log10_Mw"]),
                "abs_err_D": abs(mom_tube["D"] - gt_mom["D"]),
            },
            "synth_ridge": {
                "rmse": rmse_mwd(gpc_y, y_ridge),
                **{f"pred_{k}": v for k, v in mom_ridge.items()},
                "abs_err_logMw": abs(mom_ridge["log10_Mw"] - gt_mom["log10_Mw"]),
                "abs_err_D": abs(mom_ridge["D"] - gt_mom["D"]),
            },
            "mwd_gt": gpc_y.tolist(),
            "mwd_tube": y_tube.tolist(),
            "mwd_ridge": y_ridge.tolist(),
        }

        if models is not None:
            y_nn, nn_members = predict_elliott_ensemble(models, fit.nn_input)
            mom_nn = mwd_moments_on_grid(MWD_X, y_nn)
            member_rmse = [rmse_mwd(gpc_y, p) for p in nn_members]
            row["elliott_nn"] = {
                "rmse": rmse_mwd(gpc_y, y_nn),
                "rmse_best_member": float(np.min(member_rmse)),
                "rmse_mean_member": float(np.mean(member_rmse)),
                **{f"pred_{k}": v for k, v in mom_nn.items()},
                "abs_err_logMw": abs(mom_nn["log10_Mw"] - gt_mom["log10_Mw"]),
                "abs_err_D": abs(mom_nn["D"] - gt_mom["D"]),
            }
            row["mwd_nn"] = y_nn.tolist()
            print(
                f"{sid}: NN RMSE={row['elliott_nn']['rmse']:.4f}  "
                f"tube={row['tube']['rmse']:.4f}  ridge={row['synth_ridge']['rmse']:.4f}  "
                f"|ΔlogMw| NN={row['elliott_nn']['abs_err_logMw']:.3f} "
                f"tube={row['tube']['abs_err_logMw']:.3f}"
            )
        else:
            print(f"{sid}: tube={row['tube']['rmse']:.4f} ridge={row['synth_ridge']['rmse']:.4f}")

        rows.append(row)

    def _agg(method: str, key: str) -> float:
        return float(np.mean([r[method][key] for r in rows if method in r]))

    summary = {
        "n_samples": len(rows),
        "methods": {},
    }
    for method in ["elliott_nn", "tube", "synth_ridge"]:
        if method not in rows[0]:
            continue
        summary["methods"][method] = {
            "mean_rmse": _agg(method, "rmse"),
            "mean_abs_err_logMw": _agg(method, "abs_err_logMw"),
            "mean_abs_err_D": _agg(method, "abs_err_D"),
        }

    board = {"summary": summary, "rows": rows, "mwd_x": MWD_X.tolist()}
    (out / "TRANSFER.json").write_text(json.dumps(board, indent=2), encoding="utf-8")

    lines = [
        "# Spectrum → MWD transfer: interpretable vs Elliott NN",
        "",
        "Same Maxwell features (WLF→180°C, 81 τ) as Elliott et al. 2025. "
        "Black-box: 9 published Keras models (ensemble mean). "
        "Interpretable: (i) tube projection zero-shot; (ii) synthetic-pretrained Ridge on tube-forward pairs.",
        "",
        "## Summary (mean over 9 PS samples)",
        "",
        "| Method | MWD RMSE ↓ | |Δ log10 Mw| ↓ | |Δ Ð| ↓ |",
        "|--------|-----------:|---------------:|--------:|",
    ]
    labels = {
        "elliott_nn": "Elliott NN ensemble",
        "tube": "Tube projection (zero-shot)",
        "synth_ridge": "Synth-pretrained Ridge",
    }
    for m, lab in labels.items():
        if m not in summary["methods"]:
            continue
        s = summary["methods"][m]
        lines.append(
            f"| {lab} | {s['mean_rmse']:.4f} | {s['mean_abs_err_logMw']:.3f} | {s['mean_abs_err_D']:.3f} |"
        )

    lines += [
        "",
        "## Per-sample MWD RMSE",
        "",
        "| Sample | NN | Tube | Synth-Ridge |",
        "|--------|---:|-----:|------------:|",
    ]
    for r in rows:
        nn = r.get("elliott_nn", {}).get("rmse", float("nan"))
        lines.append(f"| {r['sample']} | {nn:.4f} | {r['tube']['rmse']:.4f} | {r['synth_ridge']['rmse']:.4f} |")

    lines += [
        "",
        "## Takeaways",
        "",
        "1. Elliott NN is the accuracy ceiling on this exact task (trained on large tube-model data).",
        "2. Tube projection tests whether **readable Maxwell weights alone** carry MWD shape under classical scaling.",
        "3. Synth-Ridge asks whether a **linear map** on log G_k, pretrained with the same scaling prior, "
        "closes the gap without a deep net.",
        "4. If tube/ridge approach NN on Mw but lag on full MWD RMSE → interpretable statistics suffice for "
        "moments; shape needs nonlinear tube inversion (Elliott's claim).",
        "",
        "Artifacts: `TRANSFER.json`. Figure: `tools/plot_rheo_mwd_transfer_figure.py`.",
        "",
    ]
    (out / "TRANSFER.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSummary:", json.dumps(summary, indent=2))
    print("Wrote", out / "TRANSFER.md")


if __name__ == "__main__":
    main()
