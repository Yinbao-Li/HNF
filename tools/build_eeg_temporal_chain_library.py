#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Induce EEG temporal-chain modes (early→late + θ→α propagation) and cluster by shape."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.eeg_dataset import EEGDataset
from hnf.eeg_temporal_chain import (
    EEG_CHAIN_SHAPE_NAMES,
    eeg_temporal_chain_features,
    extract_eeg_temporal_observables,
    features_to_vector,
    mean_trajectory,
)
from hnf.pattern_library import _kmeans, _zscore_fit
from tools.run_eeg_clinical_suite import _load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/eeg/adftd_hnf_native_v3/best.pt")
    p.add_argument("--output-dir", default="outputs/eeg/temporal_chain_native_v3")
    p.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=0, help="0 = all")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


@torch.no_grad()
def _collect_obs(
    model,
    split: str,
    *,
    data_dir: str,
    seed: int,
    sample_rate: int,
    epoch_sec: float,
    device: torch.device,
    synthetic_if_missing: bool,
    max_epochs: int,
) -> list:
    ds = EEGDataset(
        data_dir=data_dir,
        split=split,
        seed=seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,
        synthetic_if_missing=synthetic_if_missing,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    out = []
    n = 0
    for batch in loader:
        for i in range(batch["x"].size(0)):
            x = batch["x"][i : i + 1].to(device)
            obs = extract_eeg_temporal_observables(
                model,
                x,
                epoch_sec=epoch_sec,
                clinical_group=str(batch["clinical_group"][i]),
                subject_id=str(batch["subject_id"][i]),
            )
            out.append((obs, int(batch["label"][i])))
            n += 1
            if max_epochs and n >= max_epochs:
                return out
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt_args, arch = _load_model(Path(args.checkpoint), device)
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    obs_list = []
    labels_y = []
    for sp in splits:
        chunk = _collect_obs(
            model,
            sp,
            data_dir=args.data_dir,
            seed=args.seed,
            sample_rate=sample_rate,
            epoch_sec=epoch_sec,
            device=device,
            synthetic_if_missing=not args.no_synthetic,
            max_epochs=args.max_epochs,
        )
        for obs, lab in chunk:
            obs_list.append(obs)
            labels_y.append(lab)
    print(f"[eeg-chain] collected {len(obs_list)} epochs from {splits}", flush=True)

    feats = [eeg_temporal_chain_features(o) for o in obs_list]
    X = np.stack([features_to_vector(f) for f in feats], axis=0)
    mu, sd = _zscore_fit(X)
    z = (X - mu) / sd
    labels, centers_z = _kmeans(z, k=args.k, seed=args.seed)
    centers = centers_z * sd + mu

    # per-mode stats
    modes = []
    for j in range(args.k):
        mask = labels == j
        if not mask.any():
            continue
        mode_obs = [obs_list[i] for i in range(len(obs_list)) if labels[i] == j]
        groups = Counter(o.clinical_group for o in mode_obs)
        traj = mean_trajectory(mode_obs)
        modes.append(
            {
                "mode_id": j,
                "count": int(mask.sum()),
                "clinical_groups": dict(groups),
                "mean_theta_alpha_lag": float(
                    np.mean([feats[i]["theta_alpha_lag_norm"] for i in range(len(feats)) if labels[i] == j])
                ),
                "mean_rho_drift": float(
                    np.mean([feats[i]["rho_early_late_drift"] for i in range(len(feats)) if labels[i] == j])
                ),
                "mean_coupling": float(
                    np.mean([feats[i]["theta_alpha_coupling"] for i in range(len(feats)) if labels[i] == j])
                ),
                "trajectories": {
                    "tau_grid": traj["tau_grid"].tolist(),
                    "rho_tau": traj["rho_tau"].tolist(),
                    "theta_tau": traj["theta_tau"].tolist(),
                    "alpha_tau": traj["alpha_tau"].tolist(),
                },
            }
        )

    # plot mode trajectories
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, args.k))
    for m in modes:
        j = m["mode_id"]
        tg = np.asarray(m["trajectories"]["tau_grid"])
        axes[0].plot(tg, m["trajectories"]["rho_tau"], color=cmap[j], label=f"M{j}")
        axes[1].plot(tg, m["trajectories"]["theta_tau"], color=cmap[j], label=f"M{j}")
        axes[2].plot(tg, m["trajectories"]["alpha_tau"], color=cmap[j], label=f"M{j}")
    axes[0].set_title("ρ(t) in epoch frame")
    axes[1].set_title("θ envelope")
    axes[2].set_title("α envelope")
    for ax in axes:
        ax.set_xlabel("τ (early→late)")
        ax.legend(fontsize=7)
    fig.suptitle("EEG temporal-chain modes (shape clusters)")
    fig.tight_layout()
    fig.savefig(out / "temporal_chain_modes.png", dpi=150)
    plt.close(fig)

    # clinical cross-tab
    crosstab: dict[str, Counter] = defaultdict(Counter)
    for i, o in enumerate(obs_list):
        crosstab[o.clinical_group][int(labels[i])] += 1

    report = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "protocol": {
            "frame": "epoch tau ∈ [0,1] (early→late); θ→α lag + ratio trajectory",
            "n_shape_dims": len(EEG_CHAIN_SHAPE_NAMES),
            "k": args.k,
            "n_epochs": len(obs_list),
            "splits": splits,
        },
        "modes": modes,
        "clinical_crosstab": {g: dict(c) for g, c in crosstab.items()},
    }
    (out / "temporal_chain_library.json").write_text(json.dumps(report, indent=2))

    md = [
        "# EEG temporal-chain modes",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- epochs: {len(obs_list)}, K={args.k}",
        "- frame: **early→late** epoch τ + **θ→α propagation** (lag, ratio curve)",
        "",
        "## Clinical cross-tab (epoch counts per mode)",
        "",
        "| group | " + " | ".join(f"M{j}" for j in range(args.k)) + " |",
        "|-------|" + "|".join(["---:"] * args.k) + "|",
    ]
    for g in ("HC", "FTD", "AD"):
        row = crosstab.get(g, Counter())
        md.append("| " + g + " | " + " | ".join(str(row.get(j, 0)) for j in range(args.k)) + " |")
    md += ["", "## Mode summaries", ""]
    for m in modes:
        md.append(
            f"- **M{m['mode_id']}** n={m['count']} groups={m['clinical_groups']} "
            f"θ→α lag={m['mean_theta_alpha_lag']:.3f} ρ drift={m['mean_rho_drift']:.3f} "
            f"coupling={m['mean_coupling']:.3f}"
        )
    (out / "TEMPORAL_CHAIN_REPORT.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[eeg-chain] wrote {out / 'temporal_chain_library.json'}", flush=True)


if __name__ == "__main__":
    main()
