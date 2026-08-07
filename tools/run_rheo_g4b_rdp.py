#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G4b: RDP+CLF tube forward pretrain vs heuristic / Elliott NN (zero-shot).

Does **not** claim Elliott's unpublished BoB training engine; uses RepTate-grade
Rolie–Double–Poly LVE + dilution/CLF as the nearest open, headless tube theory.
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

from hnf.rheo_elliott_compat import (
    MWD_X,
    TAU_VALUES,
    fit_sample_elliott,
    mwd_from_weights,
    mwd_moments_on_grid,
    normalize_mwd_on_x,
)
from hnf.rheo_gpc import load_leeds_gpc_all
from hnf.rheo_leeds import load_leeds_saos_all
from hnf.rheo_rdp_forward import forward_G_from_mwd_weights, sample_synth_weights
from hnf.rheo_tube_invert import forward_G_from_weights as forward_heuristic

DEFAULT_NN = _REPO_ROOT / "external_data" / "rheo_leeds_ps" / "NN_Models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--nn-dir", default=str(DEFAULT_NN))
    p.add_argument("--output-dir", default="outputs/rheo/tube_g4b_rdp")
    p.add_argument("--n-synth", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-nn", action="store_true")
    return p.parse_args()


def rmse_mwd(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_map(forward_fn, n_synth: int, seed: int, noise_std: float = 0.12):
    rng = np.random.default_rng(seed)
    W = sample_synth_weights(rng, n_synth)
    X = []
    Y = []
    skipped = 0
    for i in range(n_synth):
        g = forward_fn(W[i])
        if g is None:
            skipped += 1
            continue
        logg = np.log10(np.maximum(g, 1e-12))
        logg = logg + rng.normal(0.0, noise_std, size=logg.shape)
        X.append(logg)
        Y.append(W[i])
    X = np.asarray(X)
    Y = np.asarray(Y)
    ridge = MultiOutputRegressor(Ridge(alpha=10.0)).fit(X, Y)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-3,
        max_iter=400,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    ).fit(X, Y)
    return ridge, mlp, {"n_ok": int(len(X)), "n_skip": skipped}


def predict_w(model, log_g_nat: np.ndarray) -> np.ndarray:
    x = (log_g_nat / np.log(10.0)).reshape(1, -1)
    w = np.maximum(model.predict(x)[0], 0.0)
    s = w.sum()
    return w / s if s > 0 else w


def load_nn(nn_dir: Path):
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    return [tf.keras.models.load_model(str(nn_dir / f"PS_Model_{i}.keras")) for i in range(1, 10)]


def pred_nn(models, nn_input):
    preds = [mwd_from_weights(np.abs(m.predict(nn_input, verbose=0)[0])) for m in models]
    return np.mean(np.stack(preds, 0), 0)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training heuristic (α=3.4 deposit) maps on {args.n_synth} synth...")
    ridge_h, mlp_h, meta_h = train_map(forward_heuristic, args.n_synth, args.seed)
    print(f"  heuristic ok={meta_h['n_ok']} skip={meta_h['n_skip']}")

    print(f"Training RDP+CLF maps on {args.n_synth} synth...")
    ridge_r, mlp_r, meta_r = train_map(
        lambda w: forward_G_from_mwd_weights(w, with_gcorr=True),
        args.n_synth,
        args.seed + 7,
    )
    print(f"  RDP ok={meta_r['n_ok']} skip={meta_r['n_skip']}")

    saos = {s.sample_id: s for s in load_leeds_saos_all(args.rheo_dir)}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all(args.gpc_dir)}
    ids = sorted(set(saos) & set(gpc))

    print("Fitting experimental Elliott Maxwell modes...")
    fits = {}
    for sid in ids:
        s = saos[sid]
        fits[sid] = fit_sample_elliott(sid, s.temperature_c, s.omega, s.g_prime, s.g_double_prime)

    models = None if args.skip_nn else load_nn(Path(args.nn_dir))

    rows = []
    for sid in ids:
        fit = fits[sid]
        gt = normalize_mwd_on_x(gpc[sid].M, gpc[sid].w, MWD_X)
        gt_mom = mwd_moments_on_grid(MWD_X, gt)

        def block(y):
            mom = mwd_moments_on_grid(MWD_X, y)
            return {
                "rmse": rmse_mwd(gt, y),
                "abs_err_logMw": abs(mom["log10_Mw"] - gt_mom["log10_Mw"]),
                "abs_err_D": abs(mom["D"] - gt_mom["D"]),
                **{f"pred_{k}": v for k, v in mom.items()},
            }

        y_hr = mwd_from_weights(predict_w(ridge_h, fit.log_g))
        y_hm = mwd_from_weights(predict_w(mlp_h, fit.log_g))
        y_rr = mwd_from_weights(predict_w(ridge_r, fit.log_g))
        y_rm = mwd_from_weights(predict_w(mlp_r, fit.log_g))

        row = {
            "sample": sid,
            "heur_ridge": block(y_hr),
            "heur_mlp": block(y_hm),
            "rdp_ridge": block(y_rr),
            "rdp_mlp": block(y_rm),
            "mwd_gt": gt.tolist(),
            "mwd_heur_ridge": y_hr.tolist(),
            "mwd_rdp_ridge": y_rr.tolist(),
            "mwd_rdp_mlp": y_rm.tolist(),
        }
        if models is not None:
            y_nn = pred_nn(models, fit.nn_input)
            row["elliott_nn"] = block(y_nn)
            row["mwd_nn"] = y_nn.tolist()
        bits = [
            f"rdpR={row['rdp_ridge']['rmse']:.3f}",
            f"rdpM={row['rdp_mlp']['rmse']:.3f}",
            f"heurR={row['heur_ridge']['rmse']:.3f}",
        ]
        if "elliott_nn" in row:
            bits.insert(0, f"NN={row['elliott_nn']['rmse']:.3f}")
        print(f"{sid}: " + "  ".join(bits))
        rows.append(row)

    def agg(method, key):
        return float(np.mean([r[method][key] for r in rows if method in r]))

    methods = [m for m in ["elliott_nn", "rdp_mlp", "rdp_ridge", "heur_mlp", "heur_ridge"] if m in rows[0]]
    summary = {"n_samples": len(rows), "methods": {}, "synth_meta": {"heuristic": meta_h, "rdp": meta_r}}
    for m in methods:
        summary["methods"][m] = {
            "mean_rmse": agg(m, "rmse"),
            "mean_abs_err_logMw": agg(m, "abs_err_logMw"),
            "mean_abs_err_D": agg(m, "abs_err_D"),
        }
    if "elliott_nn" in summary["methods"] and "heur_ridge" in summary["methods"]:
        nn = summary["methods"]["elliott_nn"]["mean_rmse"]
        base = summary["methods"]["heur_ridge"]["mean_rmse"]
        span = max(base - nn, 1e-12)
        for m, s in summary["methods"].items():
            if m in ("elliott_nn", "heur_ridge"):
                continue
            s["frac_gap_closed_vs_heur_to_nn"] = float((base - s["mean_rmse"]) / span)

    board = {
        "gate": "G4b",
        "forward": "RepTate-style RDP LVE + dilution/CLF (headless); not Elliott BoB training dump",
        "claim": (
            "If RDP-pretrained readable maps beat α=3.4 heuristic toward NN, physics fidelity "
            "of the forward is the bottleneck. If not, Elliott's remaining edge is BoB-scale "
            "sim + deep net capacity / 8e5 data."
        ),
        "summary": summary,
        "rows": rows,
        "mwd_x": MWD_X.tolist(),
        "n_tau": len(TAU_VALUES),
    }
    (out / "G4B.json").write_text(json.dumps(board, indent=2), encoding="utf-8")

    labels = {
        "elliott_nn": "Elliott NN",
        "rdp_mlp": "RDP+CLF synth MLP",
        "rdp_ridge": "RDP+CLF synth Ridge",
        "heur_mlp": "Heuristic synth MLP",
        "heur_ridge": "Heuristic synth Ridge",
    }
    lines = [
        "# G4b — RDP+CLF tube forward vs heuristic / Elliott NN",
        "",
        "Zero-shot on Leeds experimental Maxwell features (WLF→180°C, 81 τ).",
        "Synth pretrain: same 28-weight MWD prior; forward = RDP+dilution+CLF "
        "or α=3.4 mass deposit.",
        "",
        "| Method | MWD RMSE ↓ | |Δ log10 Mw| ↓ | gap vs heur→NN |",
        "|--------|-----------:|---------------:|---------------:|",
    ]
    for m in methods:
        s = summary["methods"][m]
        gap = s.get("frac_gap_closed_vs_heur_to_nn")
        gap_s = f"{gap:.2f}" if gap is not None else "—"
        lines.append(
            f"| {labels[m]} | {s['mean_rmse']:.4f} | {s['mean_abs_err_logMw']:.3f} | {gap_s} |"
        )
    best_rdp = min(
        (summary["methods"][m]["mean_rmse"] for m in ("rdp_ridge", "rdp_mlp") if m in summary["methods"]),
        default=float("nan"),
    )
    heur = summary["methods"]["heur_ridge"]["mean_rmse"]
    nn = summary["methods"].get("elliott_nn", {}).get("mean_rmse")
    lines += [
        "",
        "## Verdict",
        "",
        f"- Heuristic Ridge RMSE = **{heur:.4f}**; best RDP = **{best_rdp:.4f}**"
        + (f"; NN = **{nn:.4f}**" if nn is not None else ""),
        "- If RDP ≪ heuristic toward NN → open tube theory closes part of G4b.",
        "- If RDP ≈ heuristic ≫ NN → need Elliott-grade BoB + large data (not open here).",
        "",
        "Locked track: `docs/RHEO_NATURE_DISCOVERY.md`. Artifact: `G4B.json`.",
        "",
    ]
    (out / "G4B.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSummary:", json.dumps(summary, indent=2))
    print("Wrote", out / "G4B.md")


if __name__ == "__main__":
    main()
