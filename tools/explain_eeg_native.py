#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize EEG-native HNF internals: ρ(t), rhythm envelopes, band proxy, predictions."""

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
from torch.utils.data import DataLoader

from hnf.eeg_dataset import CLINICAL_ID_TO_LABEL, EEGDataset
from tools.run_eeg_clinical_suite import _load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain EEG-native HNF on sample epochs")
    p.add_argument("--checkpoint", default="outputs/eeg/adftd_hnf_native_v3/best.pt")
    p.add_argument("--output-dir", default="outputs/eeg/explain_native_v3")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-per-group", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default="external_data/eeg_adftd")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-synthetic", action="store_true")
    return p.parse_args()


@torch.no_grad()
def _pick_examples(loader, num_per_group: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list] = {"HC": [], "FTD": [], "AD": []}
    for batch in loader:
        for i in range(batch["x"].size(0)):
            g = str(batch["clinical_group"][i])
            if g not in buckets:
                continue
            buckets[g].append(
                {
                    "x": batch["x"][i : i + 1],
                    "subject_id": str(batch["subject_id"][i]),
                    "clinical_group": g,
                    "label": int(batch["label"][i]),
                    "mmse": float(batch["mmse"][i]),
                }
            )
    picked = []
    for g in ("HC", "FTD", "AD"):
        rows = buckets[g]
        if not rows:
            continue
        idx = rng.choice(len(rows), size=min(num_per_group, len(rows)), replace=False)
        picked.extend(rows[i] for i in idx)
    return picked


def _plot_example(
    out_path: Path,
    sample: dict,
    logits: torch.Tensor,
    aux: dict,
    kparams: dict,
    sample_rate: int,
) -> dict:
    x = sample["x"][0].cpu().numpy()  # (C, T)
    t = np.arange(x.shape[1], dtype=np.float64) / float(sample_rate)
    rho = aux["rho"][0, :, 0].detach().cpu().numpy()
    th = aux["theta_env"][0].detach().cpu().numpy()
    al = aux["alpha_env"][0].detach().cpu().numpy()
    de = aux.get("delta_env")
    de_np = de[0].detach().cpu().numpy() if de is not None else None
    band = aux["band_proxy"][0].detach().cpu().numpy()
    probs = torch.softmax(logits.detach(), dim=-1)[0].cpu().numpy()
    pred = int(probs.argmax())

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 0.8])
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, x.mean(axis=0), color="#333", lw=0.8)
    ax0.set_title(
        f"{sample['subject_id']}  true={sample['clinical_group']}  "
        f"pred={CLINICAL_ID_TO_LABEL[pred]}  MMSE={sample['mmse']:.0f}  "
        f"P=[{probs[0]:.2f},{probs[1]:.2f},{probs[2]:.2f}]"
    )
    ax0.set_ylabel("mean EEG")
    ax0.set_xlim(t[0], t[-1])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(np.linspace(t[0], t[-1], len(rho)), rho, color="#e76f51")
    ax1.set_title("ρ(t) medium density")
    ax1.set_ylabel("ρ")

    ax2 = fig.add_subplot(gs[1, 1])
    band_names = ["δ", "θ", "α", "β"]
    ax2.bar(band_names, band, color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261"])
    ax2.set_title("Band proxy (head input)")
    ax2.set_ylim(0, max(0.5, float(band.max()) * 1.2))

    ax3 = fig.add_subplot(gs[2, 0])
    tt = np.linspace(t[0], t[-1], th.shape[0])
    ax3.plot(tt, th.mean(axis=-1), label="θ env", color="#2a9d8f")
    ax3.plot(tt, al.mean(axis=-1), label="α env", color="#e9c46a")
    if de_np is not None:
        ax3.plot(tt, de_np.mean(axis=-1), label="δ env", color="#264653")
    ax3.legend(fontsize=8)
    ax3.set_title("Rhythm branch envelopes (channel-mean)")
    ax3.set_ylabel("envelope")

    ax4 = fig.add_subplot(gs[2, 1])
    if "region_energy" in aux:
        reg = aux["region_energy"][0].detach().cpu().numpy()
        reg_names = ["frontal", "temporal", "central", "posterior", "FT ctr", "PF ctr"]
        ax4.bar(reg_names, reg, color="#457b9d")
        ax4.set_title("Regional energy / contrast")
        ax4.tick_params(axis="x", rotation=30, labelsize=7)
    else:
        ax4.axis("off")
        ax4.text(0.1, 0.5, "No regional head in checkpoint", va="center")

    ax5 = fig.add_subplot(gs[3, :])
    txt = json.dumps(kparams, indent=2)
    ax5.axis("off")
    ax5.text(0.01, 0.95, "Learned kernel params:\n" + txt, va="top", family="monospace", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "subject_id": sample["subject_id"],
        "clinical_group": sample["clinical_group"],
        "pred": CLINICAL_ID_TO_LABEL[pred],
        "probs": probs.tolist(),
        "rho_mean": float(rho.mean()),
        "theta_energy": float(th.mean()),
        "alpha_energy": float(al.mean()),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt_args, arch = _load_model(Path(args.checkpoint), device)
    sample_rate = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    kparams = model.collect_kernel_params()

    ds = EEGDataset(
        data_dir=args.data_dir,
        split=args.split,
        seed=args.seed,
        sample_rate=sample_rate,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,
        synthetic_if_missing=not args.no_synthetic,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    examples = _pick_examples(loader, args.num_per_group, args.seed)

    summaries = []
    for i, sample in enumerate(examples):
        x = sample["x"].to(device)
        logits, aux = model(x, return_aux=True)
        png = out / f"explain_{sample['clinical_group']}_{sample['subject_id']}.png"
        summaries.append(_plot_example(png, sample, logits, aux, kparams, sample_rate))
        print(f"[explain-eeg] wrote {png}", flush=True)

    report = {"checkpoint": args.checkpoint, "arch": arch, "kernel_params": kparams, "examples": summaries}
    (out / "explain_report.json").write_text(json.dumps(report, indent=2))
    print(f"[explain-eeg] done → {out / 'explain_report.json'}", flush=True)


if __name__ == "__main__":
    main()
