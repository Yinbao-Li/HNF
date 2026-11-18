#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Branch A/B residual filter probe on frozen imp_RSP.

A: train small MLP  r̂ = fθ(SNR, distance, PS-gap, depth) → |ref_at_gt| and refine shift
B: leftover  r̃ = r − r̂; test whether leftover still predicts timing improvement
   on STEAD held-out and transfers to ETHZ.

  python tools/phase2_residual_meta_filter.py --device cuda \
    --output-dir results/phase2/2_4_residual_filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tools.phase2_probe_refine_residual_knowledge import (  # noqa: E402
    collect_ethz,
    collect_stead,
    load_model,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--n-events", type=int, default=4000)
    p.add_argument("--n-noise", type=int, default=1000)
    p.add_argument("--ethz-events", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("results/phase2/2_4_residual_filter"))
    return p.parse_args()


META_KEYS = ("snr_db", "distance_km", "ps_gap_s", "depth_km")
# targets: magnitude of refine at GT, and signed refine-induced shift
TGT_KEYS = ("abs_ref", "shift_full_prop_s")


def phase_table(ph_data, meta_list):
    valid = ph_data["valid"].astype(bool)
    meta = {k: np.asarray([m[k] for m in meta_list], dtype=np.float64) for k in meta_list[0]}
    n = len(meta_list)
    assert n == len(valid)
    abs_ref = np.abs(ph_data["ref_at_gt"])
    improve = ph_data["abs_err_prop_s"] - ph_data["abs_err_full_s"]
    X = np.stack([meta[k] for k in META_KEYS], axis=1)
    Y = np.stack([abs_ref, ph_data["shift_full_prop_s"]], axis=1)
    # keep rows with finite meta+targets; depth may be nan on ETHZ → fill with train median later
    return {
        "X": X,
        "Y": Y,
        "improve": improve,
        "abs_err_prop": ph_data["abs_err_prop_s"],
        "abs_err_full": ph_data["abs_err_full_s"],
        "valid": valid,
        "abs_ref": abs_ref,
        "shift": ph_data["shift_full_prop_s"],
        "ref_at_gt": ph_data["ref_at_gt"],
    }


class MetaFilter(nn.Module):
    def __init__(self, d_in=4, d_out=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def fill_nan_cols(X, fill):
    out = X.copy()
    for j in range(out.shape[1]):
        m = ~np.isfinite(out[:, j])
        out[m, j] = fill[j]
    return out


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return {"rho": None, "p": None, "n": int(m.sum())}
    rho, p = stats.spearmanr(a[m], b[m])
    return {"rho": float(rho), "p": float(p), "n": int(m.sum())}


def ridge_r2(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return None
    x = x[m].reshape(-1, 1)
    y = y[m]
    # simple 80/20
    n = len(y)
    n_te = max(20, n // 5)
    xtr, ytr, xte, yte = x[:-n_te], y[:-n_te], x[-n_te:], y[-n_te:]
    reg = Ridge(1.0).fit(xtr, ytr)
    return float(r2_score(yte, reg.predict(xte)))


def eval_branch_b(y_true_shift, y_hat_shift, improve, abs_err_prop, abs_err_full, name):
    """Leftover shift = true − predicted; does it still track improvement / full vs prop?"""
    leftover = y_true_shift - y_hat_shift
    # |leftover| vs improvement
    # Also: predicted part vs improvement
    return {
        "split": name,
        "spearman_|pred|_vs_improve": spearman(np.abs(y_hat_shift), improve),
        "spearman_|leftover|_vs_improve": spearman(np.abs(leftover), improve),
        "spearman_pred_vs_true_shift": spearman(y_hat_shift, y_true_shift),
        "spearman_leftover_vs_true_shift": spearman(leftover, y_true_shift),
        "r2_pred_explains_true_shift": ridge_r2(y_hat_shift, y_true_shift),
        "frac_var_pred": float(
            1.0 - np.nanvar(leftover) / max(np.nanvar(y_true_shift), 1e-12)
        ),
        # oracle: full already includes all residual; compare MAE stats for context
        "mean_abs_err_prop": float(np.nanmean(abs_err_prop)),
        "mean_abs_err_full": float(np.nanmean(abs_err_full)),
        "mean_improve": float(np.nanmean(improve)),
    }


def train_filter(Xtr, Ytr, Xva, Yva, epochs, seed, device):
    torch.manual_seed(seed)
    model = MetaFilter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xtr = torch.from_numpy(Xtr.astype(np.float32)).to(device)
    ytr = torch.from_numpy(Ytr.astype(np.float32)).to(device)
    xva = torch.from_numpy(Xva.astype(np.float32)).to(device)
    yva = torch.from_numpy(Yva.astype(np.float32)).to(device)
    best = {"val": 1e9, "state": None}
    for ep in range(epochs):
        model.train()
        pred = model(xtr)
        # weighted: abs_ref + shift (shift in seconds, smaller scale)
        loss = ((pred[:, 0] - ytr[:, 0]) ** 2).mean() + 4.0 * ((pred[:, 1] - ytr[:, 1]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(xva)
            vloss = float(
                ((pv[:, 0] - yva[:, 0]) ** 2).mean()
                + 4.0 * ((pv[:, 1] - yva[:, 1]) ** 2).mean()
            )
        if vloss < best["val"]:
            best = {"val": vloss, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "ep": ep}
    model.load_state_dict(best["state"])
    return model, best


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    x = torch.from_numpy(X.astype(np.float32)).to(device)
    return model(x).cpu().numpy()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_model, _, _ = load_model(args.checkpoint, device)

    print("[collect] STEAD…", flush=True)
    stead_ph, stead_meta, _ = collect_stead(
        ckpt_model, device, args.n_events, args.n_noise, args.seed, 32
    )
    print("[collect] ETHZ…", flush=True)
    ethz_ph, ethz_meta = collect_ethz(ckpt_model, device, args.ethz_events, 32)

    # Primary phase P
    st = phase_table(stead_ph["p"], stead_meta)
    et = phase_table(ethz_ph["p"], ethz_meta)

    # usable STEAD rows: need finite snr, distance, ps_gap; depth optional
    core = np.isfinite(st["X"][:, 0]) & np.isfinite(st["X"][:, 1]) & np.isfinite(st["X"][:, 2]) & st["valid"]
    X_raw, Y_raw = st["X"][core], st["Y"][core]
    improve = st["improve"][core]
    abs_err_prop = st["abs_err_prop"][core]
    abs_err_full = st["abs_err_full"][core]
    # fill depth nan with median
    depth_med = np.nanmedian(X_raw[:, 3])
    fill = np.array([0.0, 0.0, 0.0, depth_med], dtype=np.float64)
    X_raw = fill_nan_cols(X_raw, fill)

    Xtr, Xte, Ytr, Yte, imp_tr, imp_te, pep_tr, pep_te, pef_tr, pef_te = train_test_split(
        X_raw, Y_raw, improve, abs_err_prop, abs_err_full,
        test_size=0.25, random_state=args.seed,
    )
    sx, sy = StandardScaler(), StandardScaler()
    Xtr_s, Xte_s = sx.fit_transform(Xtr), sx.transform(Xte)
    Ytr_s, Yte_s = sy.fit_transform(Ytr), sy.transform(Yte)

    print("[train] meta filter…", flush=True)
    filt, best = train_filter(Xtr_s, Ytr_s, Xte_s, Yte_s, args.epochs, args.seed, device)
    # predictions in original Y scale
    Yhat_te_s = predict(filt, Xte_s, device)
    Yhat_te = sy.inverse_transform(Yhat_te_s)
    Yhat_tr = sy.inverse_transform(predict(filt, Xtr_s, device))

    # Branch A metrics on STEAD test
    branch_a = {
        "val_loss_standardized": best["val"],
        "best_epoch": best["ep"],
        "stead_test": {
            "r2_abs_ref": float(r2_score(Yte[:, 0], Yhat_te[:, 0])),
            "r2_shift": float(r2_score(Yte[:, 1], Yhat_te[:, 1])),
            "spearman_abs_ref": spearman(Yte[:, 0], Yhat_te[:, 0]),
            "spearman_shift": spearman(Yte[:, 1], Yhat_te[:, 1]),
        },
    }

    # Branch B on STEAD test
    branch_b_stead = eval_branch_b(
        Yte[:, 1], Yhat_te[:, 1], imp_te, pep_te, pef_te, "stead_test"
    )
    # also leftover of abs_ref
    leftover_abs = Yte[:, 0] - Yhat_te[:, 0]
    branch_b_stead["spearman_|leftover_abs_ref|_vs_improve"] = spearman(np.abs(leftover_abs), imp_te)
    branch_b_stead["spearman_pred_abs_ref_vs_improve"] = spearman(Yhat_te[:, 0], imp_te)

    # ETHZ transfer: same scaler/filter; fill missing distance/depth
    e_core = et["valid"] & np.isfinite(et["X"][:, 0])  # at least SNR
    Xe = fill_nan_cols(et["X"][e_core], fill)
    # if distance mostly nan on ethz, fill with stead mean
    if not np.isfinite(et["X"][e_core][:, 1]).mean() > 0.5:
        Xe[:, 1] = sx.mean_[1] if hasattr(sx, "mean_") else np.nanmean(X_raw[:, 1])
        # actually after fill_nan with fill[1]=0 — better use train mean in raw space
        Xe[:, 1] = np.where(np.isfinite(et["X"][e_core][:, 1]), et["X"][e_core][:, 1], np.mean(X_raw[:, 1]))
    Xe[:, 3] = np.where(np.isfinite(et["X"][e_core][:, 3]), et["X"][e_core][:, 3], depth_med)
    # ps_gap may be nan for no S
    Xe[:, 2] = np.where(np.isfinite(et["X"][e_core][:, 2]), et["X"][e_core][:, 2], np.mean(X_raw[:, 2]))
    Ye = et["Y"][e_core]
    Ye_hat = sy.inverse_transform(predict(filt, sx.transform(Xe), device))
    branch_a["ethz"] = {
        "r2_abs_ref": float(r2_score(Ye[:, 0], Ye_hat[:, 0])) if np.isfinite(Ye[:, 0]).all() else None,
        "r2_shift": float(r2_score(Ye[:, 1], Ye_hat[:, 1])),
        "spearman_abs_ref": spearman(Ye[:, 0], Ye_hat[:, 0]),
        "spearman_shift": spearman(Ye[:, 1], Ye_hat[:, 1]),
        "n": int(e_core.sum()),
        "note": "distance/depth often imputed from STEAD stats when missing",
    }
    branch_b_ethz = eval_branch_b(
        Ye[:, 1],
        Ye_hat[:, 1],
        et["improve"][e_core],
        et["abs_err_prop"][e_core],
        et["abs_err_full"][e_core],
        "ethz",
    )
    leftover_abs_e = Ye[:, 0] - Ye_hat[:, 0]
    branch_b_ethz["spearman_|leftover_abs_ref|_vs_improve"] = spearman(
        np.abs(leftover_abs_e), et["improve"][e_core]
    )
    branch_b_ethz["spearman_pred_abs_ref_vs_improve"] = spearman(Ye_hat[:, 0], et["improve"][e_core])

    # Pre-registered gates for this probe
    # A pass: STEAD test R2_abs_ref>0.05 and ETHZ spearman_abs_ref same sign p<0.05
    a_stead = branch_a["stead_test"]["r2_abs_ref"] > 0.05
    a_ethz = (
        branch_a["ethz"]["spearman_abs_ref"]["rho"] is not None
        and branch_a["ethz"]["spearman_abs_ref"]["rho"] > 0
        and branch_a["ethz"]["spearman_abs_ref"]["p"] < 0.05
    )
    # B pass (leftover still useful): |leftover| vs improve Spearman p<0.01 and |rho|>0.05 on STEAD
    # AND same on ETHZ — otherwise leftover is STEAD-only noise
    def leftover_useful(blk):
        s = blk["spearman_|leftover|_vs_improve"]
        return s["rho"] is not None and abs(s["rho"]) > 0.05 and s["p"] < 0.01

    b_stead = leftover_useful(branch_b_stead)
    b_ethz = leftover_useful(branch_b_ethz)
    # Also: predicted part should explain a chunk of variance
    a_explains = branch_b_stead["frac_var_pred"] > 0.05

    verdict = {
        "branch_A_meta_predicts_residual": bool(a_stead and a_ethz),
        "branch_B_leftover_still_useful_cross_domain": bool(b_stead and b_ethz),
        "checks": {
            "A_stead_r2_abs_ref": a_stead,
            "A_ethz_spearman_abs_ref": a_ethz,
            "A_frac_var_shift_explained": a_explains,
            "B_leftover_stead": b_stead,
            "B_leftover_ethz": b_ethz,
        },
        "interpretation": None,
    }
    if verdict["branch_A_meta_predicts_residual"] and not verdict["branch_B_leftover_still_useful_cross_domain"]:
        verdict["interpretation"] = (
            "Filter recovers meta phenomenology; leftover does not transfer as useful signal. "
            "No evidence for a clean hidden law beyond SNR/geometry proxies."
        )
    elif verdict["branch_A_meta_predicts_residual"] and verdict["branch_B_leftover_still_useful_cross_domain"]:
        verdict["interpretation"] = (
            "Part of residual is meta-predictable; a transferable leftover remains — "
            "unnamed extra information survives filtering (still not a named physical law)."
        )
    elif not verdict["branch_A_meta_predicts_residual"]:
        verdict["interpretation"] = (
            "Meta filter failed to transfer; residual not well summarized by SNR/distance/PS-gap."
        )
    else:
        verdict["interpretation"] = "Mixed / inconclusive."

    report = {
        "checkpoint": str(args.checkpoint),
        "meta_features": list(META_KEYS),
        "targets": list(TGT_KEYS),
        "n_stead_train": int(len(Xtr)),
        "n_stead_test": int(len(Xte)),
        "branch_A": branch_a,
        "branch_B_stead": branch_b_stead,
        "branch_B_ethz": branch_b_ethz,
        "verdict": verdict,
    }
    (args.output_dir / "filter_probe.json").write_text(json.dumps(report, indent=2))
    # save filter weights
    torch.save({"state_dict": filt.state_dict(), "sx_mean": sx.mean_, "sx_scale": sx.scale_,
                "sy_mean": sy.mean_, "sy_scale": sy.scale_, "meta_keys": META_KEYS},
               args.output_dir / "meta_filter.pt")

    lines = [
        "# Residual meta-filter probe (Branch A/B)",
        "",
        f"**Branch A (meta predicts residual):** {'PASS' if verdict['branch_A_meta_predicts_residual'] else 'FAIL'}",
        f"**Branch B (leftover useful cross-domain):** {'PASS' if verdict['branch_B_leftover_still_useful_cross_domain'] else 'FAIL'}",
        "",
        verdict["interpretation"],
        "",
        "## Branch A",
        f"- STEAD test R² |ref|={branch_a['stead_test']['r2_abs_ref']:.3f}, shift={branch_a['stead_test']['r2_shift']:.3f}",
        f"- ETHZ Spearman |ref| pred vs true: {branch_a['ethz']['spearman_abs_ref']}",
        "",
        "## Branch B",
        f"- STEAD |leftover shift| vs improve: {branch_b_stead['spearman_|leftover|_vs_improve']}",
        f"- ETHZ  |leftover shift| vs improve: {branch_b_ethz['spearman_|leftover|_vs_improve']}",
        f"- STEAD frac var(shift) explained by meta pred: {branch_b_stead['frac_var_pred']:.3f}",
        "",
        "No main.tex edits.",
    ]
    (args.output_dir / "CONCLUSION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
