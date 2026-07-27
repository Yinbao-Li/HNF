#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subject-level feature clustering (NOT causal-chain) for EEG clinical patterns."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from hnf.eeg_pattern_library import EEG_ROUTER_FEATURES, EEGPatternLibrary, subject_router_features
from hnf.pattern_library import features_to_vector
from tools.run_eeg_clinical_suite import _aggregate_subjects, _collect_split, _load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/eeg/adftd_hnf_native_v3/best.pt")
    p.add_argument("--output-dir", default="outputs/eeg/subject_cluster_native_v3")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt_args, arch = _load_model(Path(args.checkpoint), device)
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    kparams = model.collect_kernel_params()
    mean_omega = float(np.mean([v["omega"] for v in kparams.values() if "omega" in v])) if kparams else 0.0

    all_subj = []
    for split in ("train", "val", "test"):
        pack = _collect_split(
            model,
            split,
            data_dir=args.data_dir,
            seed=args.seed,
            sample_rate=sample_rate,
            epoch_sec=epoch_sec,
            batch_size=16,
            device=device,
            synthetic_if_missing=not args.no_synthetic,
            max_epochs=0,
            mean_omega=mean_omega,
        )
        for s in _aggregate_subjects(pack["epochs"]):
            s["split"] = split
            all_subj.append(s)

    lib = EEGPatternLibrary.build_from_subjects(
        all_subj, k=args.k, seed=args.seed, checkpoint=str(args.checkpoint)
    )
    lib.save(out / "subject_clusters.json")

    # 2D PCA scatter for visualization
    labels = []
    groups = []
    vecs = np.stack(
        [features_to_vector(subject_router_features(s), EEG_ROUTER_FEATURES) for s in all_subj],
        axis=0,
    )
    for s in all_subj:
        labels.append(int(lib.route_subject(s).pattern_id))
        groups.append(s["clinical_group"])
    X = np.asarray(vecs, dtype=np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    pc = X @ vt[:2].T

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = {"HC": "#2a9d8f", "FTD": "#e9c46a", "AD": "#e76f51"}
    for g in ("HC", "FTD", "AD"):
        m = [i for i, gg in enumerate(groups) if gg == g]
        ax.scatter(pc[m, 0], pc[m, 1], c=cmap[g], label=g, alpha=0.75, s=40)
    ax.set_title(f"Subject feature clusters (K={args.k}, router features)\nNOT causal-chain")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "subject_cluster_pca.png", dpi=150)
    plt.close(fig)

    summary = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "k": args.k,
        "n_subjects": len(all_subj),
        "note": "K-means on cheap scalar router features; STEAD-style P→S causal-chain clustering not defined for EEG.",
        "clusters": [
            {
                "id": p.pattern_id,
                "name": p.name,
                "count": p.count,
                "HC": p.n_hc,
                "FTD": p.n_ftd,
                "AD": p.n_ad,
                "policy": p.policy.name,
            }
            for p in lib.prototypes
        ],
    }
    (out / "cluster_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
