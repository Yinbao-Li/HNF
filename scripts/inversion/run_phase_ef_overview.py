#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a README/report-ready overview panel combining:
  - Phase E synthetic closed-loop evidence
  - Phase F real-data pseudo-2D trusted profile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase E/F overview figure")
    p.add_argument("--phase-e-report", default="outputs/phase_e_formal/report.json")
    p.add_argument("--phase-f-report", default="outputs/phase_f_qc/report.json")
    p.add_argument("--output-dir", default="outputs/phase_ef_overview")
    return p.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_image(path_str: str):
    return mpimg.imread(path_str)


def _panel(ax, image, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_e = _read_json(Path(args.phase_e_report))
    phase_f = _read_json(Path(args.phase_f_report))

    fig_e_summary = _load_image(phase_e["figures"]["summary_panel"])
    fig_e_coverage = _load_image(phase_e["figures"]["coverage"])
    fig_f_vpvs = _load_image(phase_f["outputs"]["vpvs_masked"])
    fig_f_mask = _load_image(phase_f["outputs"]["trust_mask"])
    fig_f_support = _load_image(phase_f["outputs"]["support"])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    fig.suptitle("Phase E/F Imaging Overview", fontsize=16)

    _panel(axes[0, 0], fig_e_summary, "Phase E: Synthetic Closed Loop")
    _panel(axes[0, 1], fig_e_coverage, "Phase E: Coverage / Illumination")
    _panel(axes[0, 2], fig_f_vpvs, "Phase F: Trusted Vp/Vs Profile")
    _panel(axes[1, 0], fig_f_mask, "Phase F: Trust Mask")
    _panel(axes[1, 1], fig_f_support, "Phase F: Event Support")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        "\n".join(
            [
                "Synthetic closed loop",
                f"- model: {phase_e['model_type']}",
                f"- mean Vp RMSE: {phase_e['mean_vp_rmse']:.3f}",
                f"- mean Vs RMSE: {phase_e['mean_vs_rmse']:.3f}",
                f"- max Vp uncertainty: {phase_e['max_vp_uncertainty']:.3f}",
                f"- coverage frac: {phase_e['coverage_nonzero_frac']:.3f}",
                "",
                "Real-data pseudo-2D",
                f"- events used: {phase_f['n_events_used']}",
                f"- QC kept: {phase_f['n_events_qc_kept']} ({phase_f['qc_keep_frac']:.1%})",
                f"- mean refined TT: {phase_f['mean_refined_tt']:.3f}",
                f"- trusted-bin frac: {phase_f['trusted_bin_frac']:.1%}",
                f"- P pick MAE: {phase_f['pick_mae_p']:.3f}s",
                f"- S pick MAE: {phase_f['pick_mae_s']:.3f}s",
                "",
                "Trust rules",
                f"- P err <= {phase_f['qc_rules']['pick_err_p_max']:.2f}s",
                f"- S err <= {phase_f['qc_rules']['pick_err_s_max']:.2f}s",
                f"- TT <= {phase_f['qc_rules']['refined_tt_max']:.1f}",
                f"- support >= {phase_f['qc_rules']['min_events_per_bin']}",
                f"- vp_std <= {phase_f['qc_rules']['max_vp_std']:.1f}",
                f"- vs_std <= {phase_f['qc_rules']['max_vs_std']:.1f}",
            ]
        ),
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    out_path = out_dir / "phase_ef_overview.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    summary = {
        "phase_e_report": str(Path(args.phase_e_report)),
        "phase_f_report": str(Path(args.phase_f_report)),
        "output": str(out_path),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
