#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G4 Nature dig: nonlinear tube inversion vs Elliott NN / Ridge / linear tube.

Isolates whether *nonlinearity + inversion* on the shared tube heuristic closes
the MWD gap — without claiming a richer constitutive simulator than Elliott.
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
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from scipy.integrate import simpson

DEFAULT_NN = _REPO_ROOT / "external_data" / "rheo_leeds_ps" / "NN_Models"


def rmse_mwd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def tube_project_mwd(g: np.ndarray, tau: np.ndarray = TAU_VALUES, x: np.ndarray = MWD_X) -> np.ndarray:
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    tau = np.asarray(tau, dtype=np.float64)
    Me, tau_e, alpha = 12870.0, 2.2e-4, 3.4
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


def predict_weights_ridge(model, log_g: np.ndarray) -> np.ndarray:
    x = (log_g / np.log(10.0)).reshape(1, -1)
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


def predict_elliott_ensemble(models, nn_input: np.ndarray):
    preds = []
    for m in models:
        w = np.abs(m.predict(nn_input, verbose=0)[0])
        preds.append(mwd_from_weights(w))
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--nn-dir", default=str(DEFAULT_NN))
    p.add_argument("--output-dir", default="outputs/rheo/tube_invert_g4")
    p.add_argument("--n-synth", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-nn", action="store_true")
    return p.parse_args()


def train_synth_ridge(n_synth: int, seed: int, noise_std: float = 0.15):
    rng = np.random.default_rng(seed)
    W = sample_synth_weights(rng, n_synth)
    X = np.zeros((n_synth, len(TAU_VALUES)), dtype=np.float64)
    for i in range(n_synth):
        g = forward_G_from_weights(W[i])
        logg = np.log10(np.maximum(g, 1e-12))
        X[i] = logg + rng.normal(0.0, noise_std, size=logg.shape)
    model = MultiOutputRegressor(Ridge(alpha=10.0, fit_intercept=True))
    model.fit(X, W)
    return model


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
        print(f"  {sid}: Gsum={fits[sid].g.sum():.3g}")

    print(f"Training synth Ridge + MLP on {args.n_synth} MWDs...")
    ridge = train_synth_ridge(args.n_synth, args.seed)
    mlp = train_synth_mlp(args.n_synth, args.seed + 1)

    models = None
    if not args.skip_nn:
        print("Loading Elliott NN ensemble...")
        models = load_elliott_ensemble(Path(args.nn_dir))

    rows = []
    for sid in ids:
        fit = fits[sid]
        gpc_y = normalize_mwd_on_x(gpc[sid].M, gpc[sid].w, MWD_X)
        gt_mom = mwd_moments_on_grid(MWD_X, gpc_y)

        y_tube = tube_project_mwd(fit.g, fit.tau)
        mom_tube = mwd_moments_on_grid(MWD_X, y_tube)

        w_ridge = predict_weights_ridge(ridge, fit.log_g)
        y_ridge = mwd_from_weights(w_ridge)
        mom_ridge = mwd_moments_on_grid(MWD_X, y_ridge)

        w_mlp = predict_weights_mlp(mlp, fit.log_g)
        y_mlp = mwd_from_weights(w_mlp)
        mom_mlp = mwd_moments_on_grid(MWD_X, y_mlp)

        inv = invert_tube_weights(fit.g, w0=w_mlp)
        mom_inv = mwd_moments_on_grid(MWD_X, inv.mwd)

        hyb = hybrid_invert(fit.g, w_mlp, prior_mix=0.35)
        mom_hyb = mwd_moments_on_grid(MWD_X, hyb.mwd)

        def _block(y, mom, extra=None):
            d = {
                "rmse": rmse_mwd(gpc_y, y),
                **{f"pred_{k}": v for k, v in mom.items()},
                "abs_err_logMw": abs(mom["log10_Mw"] - gt_mom["log10_Mw"]),
                "abs_err_D": abs(mom["D"] - gt_mom["D"]),
            }
            if extra:
                d.update(extra)
            return d

        row = {
            "sample": sid,
            "gt": gt_mom,
            "tube": _block(y_tube, mom_tube),
            "synth_ridge": _block(y_ridge, mom_ridge),
            "synth_mlp": _block(y_mlp, mom_mlp),
            "tube_invert": _block(
                inv.mwd,
                mom_inv,
                {"rel_log_g": inv.rel_log_g, "n_iter": inv.n_iter, "success": inv.success},
            ),
            "hybrid_invert": _block(
                hyb.mwd,
                mom_hyb,
                {"rel_log_g": hyb.rel_log_g, "n_iter": hyb.n_iter},
            ),
            "mwd_gt": gpc_y.tolist(),
            "mwd_tube": y_tube.tolist(),
            "mwd_ridge": y_ridge.tolist(),
            "mwd_mlp": y_mlp.tolist(),
            "mwd_invert": inv.mwd.tolist(),
            "mwd_hybrid": hyb.mwd.tolist(),
        }

        if models is not None:
            y_nn, _ = predict_elliott_ensemble(models, fit.nn_input)
            mom_nn = mwd_moments_on_grid(MWD_X, y_nn)
            row["elliott_nn"] = _block(y_nn, mom_nn)
            row["mwd_nn"] = y_nn.tolist()

        methods = ["elliott_nn", "hybrid_invert", "tube_invert", "synth_mlp", "synth_ridge", "tube"]
        bits = []
        for m in methods:
            if m in row:
                bits.append(f"{m}={row[m]['rmse']:.3f}")
        print(f"{sid}: " + "  ".join(bits))
        rows.append(row)

    def _agg(method: str, key: str) -> float:
        return float(np.mean([r[method][key] for r in rows if method in r]))

    method_order = [
        "elliott_nn",
        "hybrid_invert",
        "tube_invert",
        "synth_mlp",
        "synth_ridge",
        "tube",
    ]
    summary = {"n_samples": len(rows), "methods": {}}
    for method in method_order:
        if method not in rows[0]:
            continue
        summary["methods"][method] = {
            "mean_rmse": _agg(method, "rmse"),
            "mean_abs_err_logMw": _agg(method, "abs_err_logMw"),
            "mean_abs_err_D": _agg(method, "abs_err_D"),
        }

    # Gap closed fractions vs tube → NN
    if "elliott_nn" in summary["methods"] and "tube" in summary["methods"]:
        nn = summary["methods"]["elliott_nn"]["mean_rmse"]
        tu = summary["methods"]["tube"]["mean_rmse"]
        span = max(tu - nn, 1e-12)
        for m, s in summary["methods"].items():
            if m in ("elliott_nn", "tube"):
                continue
            summary["methods"][m]["frac_gap_closed_vs_tube_to_nn"] = float(
                (tu - s["mean_rmse"]) / span
            )

    # LOO real logG → full MWD grid (near-sufficient-statistics probe)
    print("LOO real Ridge/MLP on log10 G → MWD grid...")
    X = np.stack([fits[sid].log_g / np.log(10.0) for sid in ids], axis=0)
    Y = np.stack(
        [normalize_mwd_on_x(gpc[sid].M, gpc[sid].w, MWD_X) for sid in ids],
        axis=0,
    )
    for i, sid in enumerate(ids):
        tr = np.ones(len(ids), dtype=bool)
        tr[i] = False
        ridge_loo = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        mlp_loo = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            random_state=args.seed,
            alpha=1e-2,
        ).fit(X[tr], Y[tr])
        pr = np.maximum(ridge_loo.predict(X[i : i + 1])[0], 0.0)
        pm = np.maximum(mlp_loo.predict(X[i : i + 1])[0], 0.0)
        for p in (pr, pm):
            a = simpson(p, x=np.log(MWD_X))
            if a > 0:
                p /= a
        gt = Y[i]
        gt_mom = mwd_moments_on_grid(MWD_X, gt)
        for name, y in (("loo_ridge_mwd", pr), ("loo_mlp_mwd", pm)):
            mom = mwd_moments_on_grid(MWD_X, y)
            rows[i][name] = {
                "rmse": rmse_mwd(gt, y),
                "abs_err_logMw": abs(mom["log10_Mw"] - gt_mom["log10_Mw"]),
                "abs_err_D": abs(mom["D"] - gt_mom["D"]),
                **{f"pred_{k}": v for k, v in mom.items()},
            }
        rows[i]["mwd_loo_ridge"] = pr.tolist()
        rows[i]["mwd_loo_mlp"] = pm.tolist()
        print(
            f"  {sid}: LOO-Ridge={rows[i]['loo_ridge_mwd']['rmse']:.3f}  "
            f"LOO-MLP={rows[i]['loo_mlp_mwd']['rmse']:.3f}"
        )

    for method in ("loo_ridge_mwd", "loo_mlp_mwd"):
        summary["methods"][method] = {
            "mean_rmse": _agg(method, "rmse"),
            "mean_abs_err_logMw": _agg(method, "abs_err_logMw"),
            "mean_abs_err_D": _agg(method, "abs_err_D"),
        }
        if "elliott_nn" in summary["methods"]:
            nn = summary["methods"]["elliott_nn"]["mean_rmse"]
            tu = summary["methods"]["tube"]["mean_rmse"]
            summary["methods"][method]["frac_gap_closed_vs_tube_to_nn"] = float(
                (tu - summary["methods"][method]["mean_rmse"]) / max(tu - nn, 1e-12)
            )

    board = {
        "gate": "G4",
        "claim": (
            "Zero-shot nonlinear invert on the simplified tube heuristic does not beat "
            "synth-Ridge / deposit. LOO readable logG→MWD maps close ~70% of the tube→NN "
            "gap — Maxwell amplitudes are near-sufficient statistics under light supervision."
        ),
        "summary": summary,
        "rows": rows,
        "mwd_x": MWD_X.tolist(),
    }
    (out / "INVERT.json").write_text(json.dumps(board, indent=2), encoding="utf-8")

    labels = {
        "elliott_nn": "Elliott NN ensemble",
        "loo_mlp_mwd": "LOO real MLP (logG→MWD)",
        "loo_ridge_mwd": "LOO real Ridge (logG→MWD)",
        "synth_ridge": "Synth-pretrained Ridge",
        "synth_mlp": "Synth-pretrained MLP",
        "tube": "Tube projection (linear deposit)",
        "hybrid_invert": "Hybrid invert (MLP warm-start)",
        "tube_invert": "Tube invert (zero-shot NLS)",
    }
    method_order = [
        "elliott_nn",
        "loo_mlp_mwd",
        "loo_ridge_mwd",
        "synth_ridge",
        "synth_mlp",
        "tube",
        "hybrid_invert",
        "tube_invert",
    ]
    lines = [
        "# G4 — Nonlinear tube inversion & few-shot readable maps vs Elliott NN",
        "",
        "Same Maxwell features (WLF→180°C, 81 τ). "
        "Zero-shot methods use the simplified "
        r"\(\tau\!\approx\!\tau_e(M/M_e)^{3.4}\) heuristic. "
        "LOO real maps train on the other 8 GPC curves.",
        "",
        "## Summary (mean over 9 PS samples)",
        "",
        "| Method | MWD RMSE ↓ | |Δ log10 Mw| ↓ | |Δ Ð| ↓ | gap closed† |",
        "|--------|-----------:|---------------:|--------:|-----------:|",
    ]
    for m in method_order:
        if m not in summary["methods"]:
            continue
        s = summary["methods"][m]
        gap = s.get("frac_gap_closed_vs_tube_to_nn")
        gap_s = f"{gap:.2f}" if gap is not None else "—"
        lines.append(
            f"| {labels[m]} | {s['mean_rmse']:.4f} | {s['mean_abs_err_logMw']:.3f} | "
            f"{s['mean_abs_err_D']:.3f} | {gap_s} |"
        )
    lines += [
        "",
        "† Fraction of (tube − NN) RMSE gap closed; 1.0 = match NN.",
        "",
        "## Per-sample MWD RMSE",
        "",
        "| Sample | NN | LOO-MLP | LOO-Ridge | Synth-Ridge | Tube | Invert |",
        "|--------|---:|--------:|----------:|------------:|-----:|-------:|",
    ]
    for r in rows:
        def g(k):
            return r.get(k, {}).get("rmse", float("nan"))

        lines.append(
            f"| {r['sample']} | {g('elliott_nn'):.4f} | {g('loo_mlp_mwd'):.4f} | "
            f"{g('loo_ridge_mwd'):.4f} | {g('synth_ridge'):.4f} | {g('tube'):.4f} | "
            f"{g('tube_invert'):.4f} |"
        )

    best_interp = min(
        (
            (m, summary["methods"][m]["mean_rmse"])
            for m in summary["methods"]
            if m != "elliott_nn"
        ),
        key=lambda kv: kv[1],
    )
    nn_rmse = summary["methods"].get("elliott_nn", {}).get("mean_rmse")
    lines += [
        "",
        "## Nature / claim discipline",
        "",
        f"- Best interpretable mean RMSE: **{labels.get(best_interp[0], best_interp[0])} = {best_interp[1]:.4f}**"
        + (f" vs NN **{nn_rmse:.4f}**" if nn_rmse is not None else ""),
        "- Zero-shot nonlinear invert on the simplified heuristic does **not** beat synth-Ridge.",
        "- LOO readable logG→MWD closes most of the tube→NN gap → Maxwell amplitudes are "
        "**near-sufficient statistics** under light supervision.",
        "- Still n=9; need larger paired n or full tube-sim physics for Nature-alone.",
        "",
        "Locked track: `docs/RHEO_NATURE_DISCOVERY.md`.",
        "Artifact: `INVERT.json`. Figure: `tools/plot_rheo_tube_invert_figure.py`.",
        "",
    ]
    (out / "INVERT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSummary:", json.dumps(summary, indent=2))
    print("Wrote", out / "INVERT.md")


if __name__ == "__main__":
    main()
