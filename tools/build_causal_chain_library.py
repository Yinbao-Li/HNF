#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Induce causal-chain modes from HNF picking forwards and read them physically.

Pipeline:
  1. dense forward per trace -> causal observables (rho, P/S field envelopes)
  2. causal reference frame (P at tau=0, S at tau=1) -> distance-normalised
  3. cluster the *shape* (distance & amplitude excluded from the vector)
  4. for each mode report the physical stats it correlates with, and cross-tab
     against the old scalar-summary clusters to show what each summary cluster
     maps to in causal-chain space.

CPU-friendly. Example:
  PYTHONPATH=. python tools/build_causal_chain_library.py \\
    --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt \\
    --split val --max-event 500 --max-noise 150 --k 5 --device cpu \\
    --output-dir outputs/causal_chain_run28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.causal_chain import (
    CAUSAL_SHAPE_NAMES,
    causal_chain_features,
    extract_causal_observables,
    features_to_vector,
    has_valid_chain,
    mean_trajectory,
)
from hnf.pattern_library import _kmeans, extract_pattern_features
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model
from tools.train_stead_picking import move_batch_to_device


def _merge_small_clusters(
    z: np.ndarray, labels: np.ndarray, *, min_frac: float
) -> np.ndarray:
    """Fold clusters below min_frac of N into the nearest surviving centroid."""
    n = z.shape[0]
    min_n = max(3, int(round(min_frac * n)))
    labels = labels.copy()
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) <= 1:
            break
        small = uniq[counts < min_n]
        if small.size == 0:
            break
        # centroids of currently surviving big clusters
        big = uniq[counts >= min_n]
        if big.size == 0:  # degenerate: keep the single largest
            keep = uniq[counts.argmax()]
            labels[:] = keep
            break
        cents = {int(c): z[labels == c].mean(axis=0) for c in big}
        victim = int(small[np.argmin([counts[list(uniq).index(s)] for s in small])])
        for i in np.where(labels == victim)[0]:
            d = {c: np.linalg.norm(z[i] - v) for c, v in cents.items()}
            labels[i] = min(d, key=d.get)
    # relabel to a compact 0..K-1 range
    remap = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    return np.asarray([remap[int(v)] for v in labels], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Induce causal-chain modes")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/causal_chain_run28")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max-event", type=int, default=500)
    p.add_argument("--max-noise", type=int, default=150)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    ds = STEADPickingDataset(
        args.split,
        seq_len=args.seq_len,
        max_event_traces=args.max_event,
        max_noise_traces=args.max_noise,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    obs_list = []
    feat_rows = []
    gap_list = []
    summary_gap = []      # scalar-summary P-S gap, for cross-tab
    summary_det = []
    kept_idx = []
    dist_km = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            x = batch["x"]
            t = batch["t"][0] if batch["t"].dim() == 3 else batch["t"]
            ev = bool(float(batch["det"][0].item()) > 0.5)
            obs = extract_causal_observables(
                model, x, t, pick_threshold=args.pick_threshold, is_event=ev
            )
            obs_list.append(obs)
            if has_valid_chain(obs):
                feat_rows.append(features_to_vector(causal_chain_features(obs)))
                gap_list.append(obs.ps_gap_sec)
                kept_idx.append(bi)
                summ = extract_pattern_features(
                    model, x, t, pick_threshold=args.pick_threshold
                )
                summary_gap.append(summ["ps_gap_sec"])
                summary_det.append(summ["det"])
                d = float(batch.get("source_distance_km", torch.tensor(float("nan")))[0]
                          if "source_distance_km" in batch else float("nan"))
                dist_km.append(d)
            if (bi + 1) % 100 == 0:
                print(f"[causal] {bi+1}/{len(ds)} chains={len(feat_rows)}", flush=True)

    n_chain = len(feat_rows)
    if n_chain < args.k:
        print(f"[causal] only {n_chain} valid chains; need >= k={args.k}. Abort.")
        return

    feats = np.stack(feat_rows)
    gaps = np.asarray(gap_list)
    # Robust standardisation (median / MAD) + clip, so a few spiky traces cannot
    # seed singleton clusters the way mean/std did.
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0)
    scale = np.where(mad < 1e-9, 1.0, 1.4826 * mad)
    z = np.clip((feats - med) / scale, -5.0, 5.0)
    mu, sd = med, scale
    labels, _ = _kmeans(z, k=args.k, seed=args.seed)
    labels = _merge_small_clusters(z, labels, min_frac=0.03)

    # ---- physical read-out per mode ----
    name_idx = {n: i for i, n in enumerate(CAUSAL_SHAPE_NAMES)}
    uniq_labels = sorted(int(v) for v in np.unique(labels))
    modes = []
    kept_obs = [obs_list[i] for i in kept_idx]
    for j in uniq_labels:
        m = labels == j
        if not m.any():
            continue
        sub = feats[m]
        modes.append(
            {
                "mode": j,
                "count": int(m.sum()),
                "ps_gap_sec": {
                    "mean": round(float(gaps[m].mean()), 2),
                    "std": round(float(gaps[m].std()), 2),
                },
                "coda_slope": round(float(sub[:, name_idx["coda_slope"]].mean()), 3),
                "n_rho_peaks": round(float(sub[:, name_idx["n_rho_peaks"]].mean()), 2),
                "onset_sharp": round(float(sub[:, name_idx["onset_sharp"]].mean()), 3),
                "log_sp_amp_ratio": round(
                    float(sub[:, name_idx["log_sp_amp_ratio"]].mean()), 3
                ),
                "pre_p_energy_frac": round(
                    float(sub[:, name_idx["pre_p_energy_frac"]].mean()), 4
                ),
            }
        )

    # ---- cross-tab: scalar-summary distance bins x causal mode ----
    sg = np.asarray(summary_gap)
    summ_bin = np.where(sg < 4.0, 0, np.where(sg < 8.0, 1, 2))  # near / mid / far
    summ_names = {0: "summary_near(<4s)", 1: "summary_mid(4-8s)", 2: "summary_far(>=8s)"}
    crosstab = {}
    for b in (0, 1, 2):
        row = {}
        for j in uniq_labels:
            row[f"mode{j}"] = int(((summ_bin == b) & (labels == j)).sum())
        crosstab[summ_names[b]] = row

    report = {
        "checkpoint": str(args.checkpoint),
        "n_traces": len(obs_list),
        "n_valid_chains": n_chain,
        "k": args.k,
        "elapsed_sec": round(time.time() - t0, 1),
        "feature_note": "clustered on causal-frame SHAPE only; ps_gap & amplitude "
        "excluded so modes are distance-independent mechanism modes",
        "modes": sorted(modes, key=lambda d: d["ps_gap_sec"]["mean"]),
        "crosstab_summary_x_causalmode": crosstab,
        "physical_reading": {
            "coda_slope": "more negative = faster coda decay = higher attenuation / lower scattering-Q contribution",
            "n_rho_peaks": "secondary rho peaks between P and S -> multipathing / reflectors / converted phases",
            "onset_sharp": "high = impulsive onset; low = emergent (distance or source complexity)",
            "log_sp_amp_ratio": "S/P envelope amplitude ratio -> radiation-pattern / focal-mechanism hint",
            "pre_p_energy_frac": "energy before P -> emergent onset / precursory noise",
        },
    }
    (out / "causal_chain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(
        out / "causal_features.npz",
        feats=feats,
        labels=labels,
        gaps=gaps,
        mean=mu,
        std=sd,
        names=np.asarray(list(CAUSAL_SHAPE_NAMES)),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            k = len(uniq_labels)
            fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.4), sharey=True)
            if k == 1:
                axes = [axes]
            for col, j in enumerate(uniq_labels):
                mobs = [kept_obs[i] for i in range(n_chain) if labels[i] == j]
                traj = mean_trajectory(mobs)
                ax = axes[col]
                if traj:
                    ax.plot(traj["tau"], traj["rho"], label="rho", lw=2)
                    ax.plot(traj["tau"], traj["p_env"], label="P env", lw=1.5)
                    ax.plot(traj["tau"], traj["s_env"], label="S env", lw=1.5)
                    ax.axvline(0.0, color="k", ls=":", lw=0.8)
                    ax.axvline(1.0, color="grey", ls=":", lw=0.8)
                gmean = float(gaps[labels == j].mean()) if (labels == j).any() else 0.0
                ax.set_title(f"mode {j}  n={(labels==j).sum()}\n<gap>={gmean:.1f}s")
                ax.set_xlabel("tau  (P=0, S=1)")
                if col == 0:
                    ax.set_ylabel("normalised amplitude")
                    ax.legend(fontsize=7)
            fig.suptitle("Causal-chain modes in the P->S normalised frame", y=1.02)
            fig.tight_layout()
            plot_path = out / "causal_chain_modes.png"
            fig.savefig(plot_path, dpi=130, bbox_inches="tight")
            print(f"[causal] wrote {plot_path}", flush=True)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"[causal] plot skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
