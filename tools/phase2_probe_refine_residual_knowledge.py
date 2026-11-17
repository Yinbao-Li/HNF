#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preliminary test: does the propose-refine residual carry usable "new information"?

Pre-registered pass bar (all required for exploratory "yes"):
  (1) permutation: residual features predict timing-error sign/magnitude above chance
  (2) incremental: residual features reduce physics-only residual error beyond physics features
  (3) exogenous: residual correlates with ≥1 held-out physical covariate (distance / proxy-SNR / PS gap)
  (4) transfer: same-signed exogenous correlation on ETHZ dev (or incremental R² > 0)

This is NOT a claim of new physics — only of transferable predictive information
not explained by the physics proposal path alone.

  python tools/phase2_probe_refine_residual_knowledge.py --device cuda \
    --output-dir results/phase2/2_4_residual_knowledge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402

T = 6000
FS = 100.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--n-events", type=int, default=4000)
    p.add_argument("--n-noise", type=int, default=1000)
    p.add_argument("--ethz-events", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("results/phase2/2_4_residual_knowledge"))
    return p.parse_args()


def proxy_snr_db(x: np.ndarray, p_idx: int) -> float:
    """Rough SNR: power in 2 s after P vs 2 s before P (trace mean over channels)."""
    p = int(p_idx)
    pre0, pre1 = max(0, p - 200), p
    post0, post1 = p, min(T, p + 200)
    if pre1 <= pre0 or post1 <= post0:
        return float("nan")
    pre = x[pre0:pre1].astype(np.float64)
    post = x[post0:post1].astype(np.float64)
    n = float((pre**2).mean()) + 1e-12
    s = float((post**2).mean()) + 1e-12
    return 10.0 * np.log10(s / n)


@torch.no_grad()
def forward_pack(model, x_btc: torch.Tensor):
    """x: (B,T,C) already demean/std."""
    t = torch.linspace(0.0, 60.0, T, device=x_btc.device).view(1, T, 1).expand(x_btc.size(0), -1, -1)
    out = model(x_btc, t)
    det = torch.sigmoid(out["det"].float())
    pw, sw = out["p_wave"].float(), out["s_wave"].float()
    pr, sr = out["p_ref"].float(), out["s_ref"].float()
    r_max = float(model.refine_max)
    eps = float(model.gate_floor)
    k = int(model.refine_pool_bins)
    p_base, s_base = pw, sw
    if k > 0:
        p_base = F.max_pool1d(pw.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)
        s_base = F.max_pool1d(sw.unsqueeze(1), 2 * k + 1, 1, k).squeeze(1)
    p_full = p_base + (eps + (1 - eps) * torch.sigmoid(p_base)) * r_max * torch.tanh(pr / r_max)
    s_full = s_base + (eps + (1 - eps) * torch.sigmoid(s_base)) * r_max * torch.tanh(sr / r_max)
    return {
        "det": det,
        "p_wave": pw,
        "s_wave": sw,
        "p_base": p_base,
        "s_base": s_base,
        "p_ref": pr,
        "s_ref": sr,
        "p_full": p_full,
        "s_full": s_full,
    }


def peak_idx(logit: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logit).argmax(-1)


def gather(logit: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return logit.gather(1, idx.view(-1, 1).clamp(0, T - 1)).squeeze(1)


def phase_rows(pack, gt, valid, phase: str):
    wave = pack[f"{phase}_wave"]
    base = pack[f"{phase}_base"]
    ref = pack[f"{phase}_ref"]
    full = pack[f"{phase}_full"]
    gt = gt.long()
    prop_i = peak_idx(base)
    full_i = peak_idx(full)
    wave_i = peak_idx(wave)
    # residual features at GT and at proposal peak
    ref_at_gt = gather(ref, gt)
    ref_at_prop = gather(ref, prop_i)
    tanh_ref_gt = torch.tanh(ref_at_gt / float(4.0))  # R=4 default scale
    shift_full_prop = (full_i - prop_i).float() / FS
    shift_full_gt = (full_i - gt).float() / FS
    shift_prop_gt = (prop_i - gt).float() / FS
    shift_wave_gt = (wave_i - gt).float() / FS
    abs_err_full = shift_full_gt.abs()
    abs_err_prop = shift_prop_gt.abs()
    abs_err_wave = shift_wave_gt.abs()
    # physics-only local confidence
    prop_logit_gt = gather(base, gt)
    prop_prob_gt = torch.sigmoid(prop_logit_gt)
    full_prob_gt = torch.sigmoid(gather(full, gt))
    return {
        "valid": valid.bool().cpu().numpy(),
        "gt": gt.cpu().numpy(),
        "ref_at_gt": ref_at_gt.cpu().numpy(),
        "ref_at_prop": ref_at_prop.cpu().numpy(),
        "tanh_ref_gt": tanh_ref_gt.cpu().numpy(),
        "shift_full_prop_s": shift_full_prop.cpu().numpy(),
        "abs_err_full_s": abs_err_full.cpu().numpy(),
        "abs_err_prop_s": abs_err_prop.cpu().numpy(),
        "abs_err_wave_s": abs_err_wave.cpu().numpy(),
        "prop_prob_gt": prop_prob_gt.cpu().numpy(),
        "full_prob_gt": full_prob_gt.cpu().numpy(),
        "signed_err_prop_s": shift_prop_gt.cpu().numpy(),
        "signed_err_full_s": shift_full_gt.cpu().numpy(),
    }


@torch.no_grad()
def collect_stead(model, device, n_ev, n_nz, seed, bs):
    ds = STEADPickingDataset(
        "val",
        seq_len=T,
        max_event_traces=n_ev,
        max_noise_traces=n_nz,
        seed=seed,
        load_geometry=True,
    )
    rows = {ph: [] for ph in "ps"}
    meta = []
    noise_ref = []
    buf_x, buf_y = [], []

    def flush():
        nonlocal buf_x, buf_y
        if not buf_x:
            return
        x = torch.stack(buf_x).to(device)
        x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
        pack = forward_pack(model, x)
        for j, y in enumerate(buf_y):
            is_ev = bool(y["is_event"])
            if not is_ev:
                # noise: residual energy stats
                pr = pack["p_ref"][j].cpu().numpy()
                noise_ref.append(
                    {
                        "p_ref_abs_mean": float(np.abs(pr).mean()),
                        "p_ref_abs_max": float(np.abs(pr).max()),
                        "p_ref_std": float(pr.std()),
                    }
                )
                continue
            xi = buf_x[j].cpu().numpy()
            p_i, s_i = int(y["p_idx"]), int(y["s_idx"])
            snr = proxy_snr_db(xi, p_i)
            dist = float(y["dist"])
            depth = float(y["depth"])
            ps_gap = (s_i - p_i) / FS if y["s_valid"] else float("nan")
            meta.append(
                {
                    "snr_db": snr,
                    "distance_km": dist,
                    "depth_km": depth,
                    "ps_gap_s": ps_gap,
                    "p_manual": float(y["p_manual"]),
                }
            )
            one = {
                "p": phase_rows(
                    {k: v[j : j + 1] for k, v in pack.items()},
                    torch.tensor([p_i], device=device),
                    torch.tensor([1.0], device=device),
                    "p",
                ),
                "s": phase_rows(
                    {k: v[j : j + 1] for k, v in pack.items()},
                    torch.tensor([s_i], device=device),
                    torch.tensor([1.0 if y["s_valid"] else 0.0], device=device),
                    "s",
                ),
            }
            for ph in "ps":
                rows[ph].append({k: v[0] for k, v in one[ph].items()})
        buf_x, buf_y = [], []

    for i in range(len(ds)):
        b = ds[i]
        x = b["x"]  # (T,C)
        y = {
            "is_event": bool(b["det"] > 0.5),
            "p_idx": int(b["p_idx"]),
            "s_idx": int(b["s_idx"]),
            "s_valid": bool(b["s_valid"] > 0),
            "dist": float(b["source_distance_km"]),
            "depth": float(b["source_depth_km"]),
            "p_manual": float(b["p_manual"]),
        }
        buf_x.append(x)
        buf_y.append(y)
        if len(buf_x) >= bs:
            flush()
    flush()

    def stack(lst):
        if not lst:
            return {}
        keys = lst[0].keys()
        return {k: np.asarray([r[k] for r in lst]) for k in keys}

    return {ph: stack(rows[ph]) for ph in "ps"}, meta, noise_ref


@torch.no_grad()
def collect_ethz(model, device, n_ev, bs):
    from scipy.signal import butter, sosfiltfilt

    from tools.eval_ethz_cross import P_COLS, S_COLS, cut_window, first_arrival, load_dataset

    data = load_dataset("ethz", "dev")
    rng = np.random.default_rng(0)
    sos = butter(4, [1, 45], btype="bandpass", fs=100.0, output="sos")
    rows = {ph: [] for ph in "ps"}
    meta = []
    buf_x, buf_y = [], []

    def flush():
        nonlocal buf_x, buf_y
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x)).to(device)
        x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
        pack = forward_pack(model, x)
        for j, y in enumerate(buf_y):
            meta.append(y["meta"])
            for ph in "ps":
                rows[ph].append(
                    {
                        k: v[0]
                        for k, v in phase_rows(
                            {k2: v2[j : j + 1] for k2, v2 in pack.items()},
                            torch.tensor([y[f"{ph}_idx"]], device=device),
                            torch.tensor([1.0 if y[f"{ph}_valid"] else 0.0], device=device),
                            ph,
                        ).items()
                    }
                )
        buf_x, buf_y = [], []

    for i in rng.permutation(len(data)):
        if len(meta) + len(buf_x) >= n_ev:
            break
        w, m = data.get_sample(int(i))
        p = first_arrival(m, P_COLS)
        if p is None or w.shape[0] != 3:
            continue
        s = first_arrival(m, S_COLS)
        tp = int(round(rng.uniform(5, 30) * 100))
        win, _, _ = cut_window(np.nan_to_num(np.asarray(w, dtype=np.float32)), int(round(p)), tp)
        win = sosfiltfilt(sos, win, axis=-1).astype(np.float32)
        s_rel = None if s is None else int(round(s)) - int(round(p)) + tp
        s_ok = s_rel is not None and 50 <= s_rel <= T - 50 and s_rel > tp
        # ETHZ meta often has source_distance_km
        dist = float(m.get("source_distance_km", np.nan)) if isinstance(m, dict) else float("nan")
        if not np.isfinite(dist):
            try:
                dist = float(m["source_distance_km"])
            except Exception:
                dist = float("nan")
        snr = proxy_snr_db(win.T, tp)  # win is (C,T) from cut? check
        # cut_window returns (3,T)
        if win.shape[0] == 3:
            x_tc = win.T
            snr = proxy_snr_db(x_tc, tp)
        else:
            x_tc = win
            snr = proxy_snr_db(x_tc, tp)
        buf_x.append(x_tc.astype(np.float32))
        buf_y.append(
            {
                "p_idx": tp,
                "s_idx": int(s_rel) if s_ok else 0,
                "p_valid": True,
                "s_valid": bool(s_ok),
                "meta": {
                    "snr_db": snr,
                    "distance_km": dist,
                    "depth_km": float("nan"),
                    "ps_gap_s": (float(s_rel - tp) / FS) if s_ok else float("nan"),
                    "p_manual": float("nan"),
                },
            }
        )
        if len(buf_x) >= bs:
            flush()
    flush()

    def stack(lst):
        keys = lst[0].keys()
        return {k: np.asarray([r[k] for r in lst]) for k in keys}

    return {ph: stack(rows[ph]) for ph in "ps"}, meta


def spearman_safe(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return {"rho": None, "p": None, "n": int(m.sum())}
    rho, p = stats.spearmanr(a[m], b[m])
    return {"rho": float(rho), "p": float(p), "n": int(m.sum())}


def ridge_cv_r2(X, y, seed=0, n_splits=5):
    m = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[m], y[m]
    if len(y) < 100:
        return {"r2_mean": None, "r2_std": None, "n": int(len(y))}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, y[tr])
        pred = reg.predict(Xte)
        scores.append(r2_score(y[te], pred))
    return {"r2_mean": float(np.mean(scores)), "r2_std": float(np.std(scores)), "n": int(len(y))}


def perm_test_r2(X, y, n_perm, seed):
    base = ridge_cv_r2(X, y, seed=seed)
    if base["r2_mean"] is None:
        return {**base, "perm_p": None, "perm_null_mean": None}
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        yp = rng.permutation(y)
        null.append(ridge_cv_r2(X, yp, seed=seed + 1 + i)["r2_mean"])
    null = np.asarray(null, dtype=np.float64)
    p = float((null >= base["r2_mean"]).mean())
    return {**base, "perm_p": p, "perm_null_mean": float(null.mean())}


def analyze_phase(ph, data, meta_list, noise_ref, n_perm, seed):
    valid = data["valid"].astype(bool)
    # align meta (events only, same order)
    meta = {k: np.asarray([m[k] for m in meta_list], dtype=np.float64) for k in meta_list[0]}
    assert len(meta_list) == len(valid)

    # residual feature matrix
    feat_names = ["ref_at_gt", "ref_at_prop", "tanh_ref_gt", "shift_full_prop_s"]
    X_res = np.stack([data[n] for n in feat_names], axis=1)
    # physics features
    phys_names = ["abs_err_prop_s", "prop_prob_gt", "signed_err_prop_s"]
    X_phys = np.stack([data[n] for n in phys_names], axis=1)

    y_abs = data["abs_err_full_s"]  # should be partly explained by residual by construction
    # Better target: physics-only absolute error (what residual is supposed to fix)
    y_phys_err = data["abs_err_prop_s"]
    # And whether residual shift reduces error
    y_improve = data["abs_err_prop_s"] - data["abs_err_full_s"]

    m = valid & np.isfinite(X_res).all(1) & np.isfinite(X_phys).all(1)
    X_res, X_phys = X_res[m], X_phys[m]
    y_phys_err, y_improve = y_phys_err[m], y_improve[m]
    meta_m = {k: v[m] for k, v in meta.items()}
    data_m = {k: v[m] for k, v in data.items()}

    # (1) permutation: residual predicts physics error & improvement
    perm_err = perm_test_r2(X_res, y_phys_err, n_perm, seed)
    perm_imp = perm_test_r2(X_res, y_improve, n_perm, seed + 17)

    # (2) incremental R²: physics-only vs physics+residual on FULL abs error
    y_full = data_m["abs_err_full_s"]
    r2_phys = ridge_cv_r2(X_phys, y_full, seed)
    X_both = np.concatenate([X_phys, X_res], 1)
    r2_both = ridge_cv_r2(X_both, y_full, seed + 3)
    # also on predicting physics error from residual alone already in perm_err
    delta_r2 = None
    if r2_phys["r2_mean"] is not None and r2_both["r2_mean"] is not None:
        delta_r2 = float(r2_both["r2_mean"] - r2_phys["r2_mean"])

    # (3) exogenous correlations with |ref| / shift
    abs_ref = np.abs(data_m["ref_at_gt"])
    exo = {}
    for name in ("snr_db", "distance_km", "depth_km", "ps_gap_s"):
        exo[name] = {
            "vs_abs_ref_at_gt": spearman_safe(abs_ref, meta_m[name]),
            "vs_shift_full_prop": spearman_safe(data_m["shift_full_prop_s"], meta_m[name]),
            "vs_improve": spearman_safe(y_improve, meta_m[name]),
        }

    # noise contrast
    noise = {}
    if noise_ref:
        ev_abs = float(np.mean(np.abs(data_m["ref_at_gt"])))
        nz_abs = float(np.mean([r["p_ref_abs_mean"] for r in noise_ref]))
        noise = {
            "event_mean_abs_ref_at_gt": ev_abs,
            "noise_mean_abs_ref": nz_abs,
            "event_over_noise_ratio": (ev_abs / max(nz_abs, 1e-8)),
        }

    return {
        "n_valid": int(m.sum()),
        "feat_names": feat_names,
        "perm_predict_physics_abs_err": perm_err,
        "perm_predict_improvement": perm_imp,
        "incremental": {
            "r2_physics_features_on_full_err": r2_phys,
            "r2_physics_plus_residual_on_full_err": r2_both,
            "delta_r2": delta_r2,
        },
        "exogenous_spearman": exo,
        "noise_contrast": noise,
        "summary_stats": {
            "mean_abs_err_prop_s": float(np.mean(data_m["abs_err_prop_s"])),
            "mean_abs_err_full_s": float(np.mean(data_m["abs_err_full_s"])),
            "mean_improvement_s": float(np.mean(y_improve)),
            "frac_improved": float(np.mean(y_improve > 1e-6)),
        },
    }


def gate_verdict(stead_p, ethz_p):
    """Pre-registered exploratory pass criteria."""
    checks = {}
    # (1) perm
    p1 = stead_p["perm_predict_improvement"]["perm_p"]
    r1 = stead_p["perm_predict_improvement"]["r2_mean"]
    checks["perm_improvement"] = bool(p1 is not None and p1 < 0.05 and r1 is not None and r1 > 0.02)

    # (2) incremental
    d = stead_p["incremental"]["delta_r2"]
    checks["incremental_delta_r2"] = bool(d is not None and d > 0.02)

    # (3) exogenous: any |rho|>=0.05 with p<0.01 on STEAD for abs_ref or shift
    exo_hit = None
    for name, blk in stead_p["exogenous_spearman"].items():
        for key in ("vs_abs_ref_at_gt", "vs_shift_full_prop"):
            rho, p = blk[key]["rho"], blk[key]["p"]
            if rho is not None and p is not None and abs(rho) >= 0.05 and p < 0.01:
                exo_hit = {"covariate": name, "key": key, "rho": rho, "p": p}
                break
        if exo_hit:
            break
    checks["exogenous"] = exo_hit is not None

    # (4) transfer: same covariate same key same sign on ETHZ, or ETHZ incremental delta_r2>0
    transfer_ok = False
    transfer_detail = None
    if exo_hit and ethz_p is not None:
        eblk = ethz_p["exogenous_spearman"][exo_hit["covariate"]][exo_hit["key"]]
        if eblk["rho"] is not None and np.sign(eblk["rho"]) == np.sign(exo_hit["rho"]) and eblk["p"] is not None and eblk["p"] < 0.05:
            transfer_ok = True
            transfer_detail = {"type": "same_sign_exo", **eblk, "covariate": exo_hit["covariate"], "key": exo_hit["key"]}
    if not transfer_ok and ethz_p is not None:
        ed = ethz_p["incremental"]["delta_r2"]
        if ed is not None and ed > 0.01:
            transfer_ok = True
            transfer_detail = {"type": "incremental_r2", "delta_r2": ed}
    checks["transfer"] = transfer_ok

    passed = all(checks[k] if k != "exogenous" else checks["exogenous"] for k in ("perm_improvement", "incremental_delta_r2", "exogenous", "transfer"))
    # exogenous stored as bool already
    return {
        "exploratory_pass": bool(passed),
        "checks": {
            "perm_improvement": checks["perm_improvement"],
            "incremental_delta_r2": checks["incremental_delta_r2"],
            "exogenous": checks["exogenous"],
            "transfer": checks["transfer"],
        },
        "exogenous_hit": exo_hit,
        "transfer_detail": transfer_detail,
        "claim_language_if_pass": (
            "Refine residual carries transferable predictive information not explained by "
            "physics-proposal features alone (exploratory; not a discovery of new physics)."
        ),
        "claim_language_if_fail": (
            "Residual knowledge probe did not meet pre-registered bar; do not add a "
            "'new knowledge in residual' claim to TGRS."
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, _, _ = load_model(args.checkpoint, device)
    assert model.fusion == "propose_refine"

    print("[stead] collecting…", flush=True)
    stead_ph, stead_meta, noise_ref = collect_stead(
        model, device, args.n_events, args.n_noise, args.seed, args.batch_size
    )
    print(f"[stead] events={len(stead_meta)} noise={len(noise_ref)}", flush=True)

    print("[ethz] collecting…", flush=True)
    ethz_ph, ethz_meta = collect_ethz(model, device, args.ethz_events, args.batch_size)
    print(f"[ethz] events={len(ethz_meta)}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "n_perm": args.n_perm,
        "pre_registered_bar": {
            "perm_improvement": "perm_p<0.05 and R2>0.02",
            "incremental_delta_r2": ">0.02 on STEAD",
            "exogenous": "|Spearman|>=0.05 and p<0.01 vs distance/snr/depth/ps_gap",
            "transfer": "same-sign exo on ETHZ (p<0.05) OR ETHZ delta_r2>0.01",
        },
        "stead_val": {},
        "ethz_dev": {},
    }
    for ph in "ps":
        print(f"[analyze] STEAD {ph.upper()}…", flush=True)
        report["stead_val"][ph] = analyze_phase(ph, stead_ph[ph], stead_meta, noise_ref if ph == "p" else [], args.n_perm, args.seed)
        print(f"[analyze] ETHZ {ph.upper()}…", flush=True)
        report["ethz_dev"][ph] = analyze_phase(ph, ethz_ph[ph], ethz_meta, [], args.n_perm, args.seed + 99)

    # Primary gate on P (as with low-SNR primary)
    verdict = gate_verdict(report["stead_val"]["p"], report["ethz_dev"]["p"])
    report["verdict"] = verdict
    # also S for info
    report["verdict_s_informational"] = gate_verdict(report["stead_val"]["s"], report["ethz_dev"]["s"])

    out = args.output_dir / "residual_knowledge_probe.json"
    out.write_text(json.dumps(report, indent=2))
    # one-pager
    lines = [
        "# Residual knowledge probe (exploratory)",
        "",
        f"**Verdict (primary P): {'PASS' if verdict['exploratory_pass'] else 'FAIL'}**",
        "",
        "## Pre-registered checks (P)",
    ]
    for k, v in verdict["checks"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines += ["", f"Exogenous hit: `{verdict['exogenous_hit']}`", f"Transfer: `{verdict['transfer_detail']}`", ""]
    sp = report["stead_val"]["p"]
    lines += [
        "## Key numbers (STEAD P)",
        f"- perm improvement R²={sp['perm_predict_improvement']['r2_mean']} p={sp['perm_predict_improvement']['perm_p']}",
        f"- ΔR² physics→+residual on full err: {sp['incremental']['delta_r2']}",
        f"- mean improvement prop→full: {sp['summary_stats']['mean_improvement_s']:.4f}s "
        f"(frac improved {sp['summary_stats']['frac_improved']:.3f})",
        "",
        "## Claim guidance",
        verdict["claim_language_if_pass"] if verdict["exploratory_pass"] else verdict["claim_language_if_fail"],
        "",
        "Do **not** edit main.tex from this probe alone without user confirmation.",
    ]
    (args.output_dir / "CONCLUSION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
