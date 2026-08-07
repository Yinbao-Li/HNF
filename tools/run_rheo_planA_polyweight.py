#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan A next step with PolyWeight: Leeds-train → external holdout (n=+1 real).

Honest scope: PolyWeight adds **one** experimental SAOS∩GPC pair (not a large n jump).
Protocol: fit Elliott Maxwell on all samples; train Ridge/MLP on Leeds logG→MWD;
evaluate zero-shot on PW_LaunPS. Also report n=10 LOO for completeness.
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
from sklearn.neural_network import MLPRegressor

from hnf.rheo_elliott_compat import (
    MWD_X,
    fit_sample_elliott,
    mwd_moments_on_grid,
    normalize_mwd_on_x,
)
from hnf.rheo_gpc import LeedsGPCSample, load_leeds_gpc_all, mwd_moments
from hnf.rheo_leeds import LeedsSAOSSample, load_leeds_saos_all
from hnf.rheo_polyweight import export_leeds_format, load_polyweight_experimental

DEFAULT_NN = _REPO_ROOT / "external_data" / "rheo_leeds_ps" / "NN_Models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--nn-dir", default=str(DEFAULT_NN))
    p.add_argument("--output-dir", default="outputs/rheo/planA_polyweight")
    p.add_argument("--export-dir", default="external_data/rheo_extra_pairs/PW_LaunPS")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-nn", action="store_true")
    return p.parse_args()


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def renorm(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 0.0)
    a = simpson(y, x=np.log(MWD_X))
    return y / a if a > 0 else y


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pw = load_polyweight_experimental()
    rheo_p, gpc_p = export_leeds_format(pw, args.export_dir)
    print(f"Exported Leeds-format pair → {rheo_p.parent}")

    # moments sanity
    gpc_s = LeedsGPCSample(sample_id=pw.sample_id, M=pw.M, w=pw.w, path=gpc_p)
    mom_pw = mwd_moments(gpc_s)
    print(f"{pw.sample_id}: Mw={mom_pw['Mw']:.3g} D={mom_pw['D']:.2f} Nω={pw.omega.size}")

    saos = {s.sample_id: s for s in load_leeds_saos_all(args.rheo_dir)}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all(args.gpc_dir)}
    leeds_ids = sorted(set(saos) & set(gpc))

    # add PW as SAOS/GPC-like objects
    saos[pw.sample_id] = LeedsSAOSSample(
        sample_id=pw.sample_id,
        temperature_c=pw.temperature_c,
        omega=pw.omega,
        g_prime=pw.g_prime,
        g_double_prime=pw.g_double_prime,
        path=rheo_p,
        source="PolyWeight",
        cite=pw.cite,
    )
    gpc[pw.sample_id] = gpc_s
    all_ids = leeds_ids + [pw.sample_id]

    print("Fitting Elliott Maxwell modes...")
    fits = {}
    Y = {}
    for sid in all_ids:
        s = saos[sid]
        fits[sid] = fit_sample_elliott(sid, s.temperature_c, s.omega, s.g_prime, s.g_double_prime)
        Y[sid] = normalize_mwd_on_x(gpc[sid].M, gpc[sid].w, MWD_X)

    X = {sid: fits[sid].log_g / np.log(10.0) for sid in all_ids}

    # --- Holdout: train Leeds only, test PW ---
    Xtr = np.stack([X[i] for i in leeds_ids])
    Ytr = np.stack([Y[i] for i in leeds_ids])
    ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=800,
        random_state=args.seed,
        alpha=1e-2,
    ).fit(Xtr, Ytr)

    y_true = Y[pw.sample_id]
    y_r = renorm(ridge.predict(X[pw.sample_id].reshape(1, -1))[0])
    y_m = renorm(mlp.predict(X[pw.sample_id].reshape(1, -1))[0])
    gt_mom = mwd_moments_on_grid(MWD_X, y_true)
    hold = {
        "train": "Leeds n=9",
        "test": pw.sample_id,
        "ridge": {
            "rmse": rmse(y_true, y_r),
            "abs_err_logMw": abs(mwd_moments_on_grid(MWD_X, y_r)["log10_Mw"] - gt_mom["log10_Mw"]),
        },
        "mlp": {
            "rmse": rmse(y_true, y_m),
            "abs_err_logMw": abs(mwd_moments_on_grid(MWD_X, y_m)["log10_Mw"] - gt_mom["log10_Mw"]),
        },
    }
    print(
        f"Holdout {pw.sample_id}: Ridge RMSE={hold['ridge']['rmse']:.4f}  "
        f"MLP RMSE={hold['mlp']['rmse']:.4f}"
    )

    # Elliott NN on PW if available
    if not args.skip_nn and Path(args.nn_dir).is_dir():
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        from hnf.rheo_elliott_compat import mwd_from_weights

        models = [
            tf.keras.models.load_model(str(Path(args.nn_dir) / f"PS_Model_{i}.keras"))
            for i in range(1, 10)
        ]
        preds = [
            mwd_from_weights(np.abs(m.predict(fits[pw.sample_id].nn_input, verbose=0)[0]))
            for m in models
        ]
        y_nn = np.mean(np.stack(preds, 0), 0)
        hold["elliott_nn"] = {
            "rmse": rmse(y_true, y_nn),
            "abs_err_logMw": abs(
                mwd_moments_on_grid(MWD_X, y_nn)["log10_Mw"] - gt_mom["log10_Mw"]
            ),
        }
        print(f"Holdout NN RMSE={hold['elliott_nn']['rmse']:.4f}")
    else:
        y_nn = None

    # --- LOO n=10 (Leeds+PW) ---
    loo_rows = []
    for i, sid in enumerate(all_ids):
        tr = [j for j in all_ids if j != sid]
        Xa = np.stack([X[j] for j in tr])
        Ya = np.stack([Y[j] for j in tr])
        rr = Ridge(alpha=1.0).fit(Xa, Ya)
        mm = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=800, random_state=args.seed, alpha=1e-2
        ).fit(Xa, Ya)
        pr = renorm(rr.predict(X[sid].reshape(1, -1))[0])
        pm = renorm(mm.predict(X[sid].reshape(1, -1))[0])
        gt = Y[sid]
        loo_rows.append(
            {
                "sample": sid,
                "is_polyweight": sid == pw.sample_id,
                "ridge_rmse": rmse(gt, pr),
                "mlp_rmse": rmse(gt, pm),
            }
        )
    loo_mean = {
        "ridge": float(np.mean([r["ridge_rmse"] for r in loo_rows])),
        "mlp": float(np.mean([r["mlp_rmse"] for r in loo_rows])),
        "ridge_leeds_only": float(
            np.mean([r["ridge_rmse"] for r in loo_rows if not r["is_polyweight"]])
        ),
        "mlp_leeds_only": float(
            np.mean([r["mlp_rmse"] for r in loo_rows if not r["is_polyweight"]])
        ),
    }

    board = {
        "plan": "A",
        "note": (
            "PolyWeight adds 1 experimental pair only; synthetic PolyWeight sets are not counted. "
            "Primary evidence = Leeds→PW external holdout."
        ),
        "polyweight": {
            "sample_id": pw.sample_id,
            "cite": pw.cite,
            "T_ref_C": pw.temperature_c,
            "moments": mom_pw,
            "export_dir": str(Path(args.export_dir)),
        },
        "holdout_leeds_to_pw": hold,
        "loo_n10": {"rows": loo_rows, "mean": loo_mean},
        "mwd_x": MWD_X.tolist(),
        "mwd_gt": y_true.tolist(),
        "mwd_ridge": y_r.tolist(),
        "mwd_mlp": y_m.tolist(),
        "mwd_nn": None if y_nn is None else y_nn.tolist(),
    }
    (out / "PLAN_A.json").write_text(json.dumps(board, indent=2), encoding="utf-8")

    lines = [
        "# Plan A — PolyWeight external holdout",
        "",
        f"**New real pair:** `{pw.sample_id}` (T={pw.temperature_c:g}°C). "
        f"Mw≈{mom_pw['Mw']:.3g}, Ð≈{mom_pw['D']:.2f}.",
        "",
        "PolyWeight ships **1** experimental SAOS∩GPC pair (+3 synthetic, not counted for A).",
        "",
        "## Primary: train Leeds n=9 → test PW",
        "",
        f"- Readable Ridge RMSE = **{hold['ridge']['rmse']:.4f}** "
        f"(|ΔlogMw|={hold['ridge']['abs_err_logMw']:.3f})",
        f"- Readable MLP RMSE = **{hold['mlp']['rmse']:.4f}** "
        f"(|ΔlogMw|={hold['mlp']['abs_err_logMw']:.3f})",
    ]
    if "elliott_nn" in hold:
        lines.append(
            f"- Elliott NN RMSE = **{hold['elliott_nn']['rmse']:.4f}** "
            f"(|ΔlogMw|={hold['elliott_nn']['abs_err_logMw']:.3f})"
        )
    lines += [
        "",
        "## Secondary: LOO mean RMSE on n=10 (Leeds+PW)",
        "",
        f"- All-10 Ridge/MLP = {loo_mean['ridge']:.4f} / {loo_mean['mlp']:.4f}",
        f"- Leeds-only rows in that LOO = {loo_mean['ridge_leeds_only']:.4f} / "
        f"{loo_mean['mlp_leeds_only']:.4f}",
        "",
        "## Discipline",
        "",
        "- This is **+1 real sample**, not a Nature-scale n jump.",
        "- Without letters/digitization of more literature pairs, A is data-blocked.",
        "- Next without letters: digitize tables/figures from Leeds source papers, or accept A ceiling.",
        "",
        f"Export: `{args.export_dir}`. Artifact: `PLAN_A.json`.",
        "",
    ]
    (out / "PLAN_A.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out / "PLAN_A.md")


if __name__ == "__main__":
    main()
