#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run EEG Nature-track suite and write a single board."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-transfer", action="store_true")
    p.add_argument("--skip-diffusion", action="store_true")
    p.add_argument("--skip-residual", action="store_true")
    p.add_argument("--device", default="")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = dict(**{**__import__("os").environ, "PYTHONPATH": str(_REPO)})
    subprocess.run(cmd, cwd=_REPO, check=True, env=env)


def main() -> None:
    args = parse_args()
    py = [sys.executable]
    if not args.skip_residual:
        run(py + ["tools/analyze_eeg_structure_residual.py"])
    if not args.skip_diffusion:
        run(py + ["tools/fit_eeg_subject_diffusion.py"])
    if not args.skip_transfer:
        cmd = py + ["tools/transfer_eeg_rho_ds005385.py"]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd)

    residual = json.loads((_REPO / "outputs/eeg/structure_residual/RESIDUAL_BOARD.json").read_text())
    diff_path = _REPO / "outputs/eeg/subject_diffusion/DIFFUSION_BOARD.json"
    xfer_path = _REPO / "outputs/eeg/longitudinal_ds005385_rho/TRANSFER.json"
    diff = json.loads(diff_path.read_text()) if diff_path.is_file() else {}
    xfer = json.loads(xfer_path.read_text()) if xfer_path.is_file() else {}

    md = [
        "# EEG Nature-track board",
        "",
        "Three locked checks, same bar as seismic β_res:",
        "1. leftover after age+sex+θ/α",
        "2. LOSO jackknife",
        "3. independent healthy-aging transfer + subject-level D topography",
        "",
    ]
    for model in residual.get("models", []):
        de = model["markers"].get("D_eff_residual", {})
        jk = model.get("jackknife", {}).get("D_eff_res_spearman", {})
        mm = ((model.get("mmse") or {}).get("patients") or {}).get("increments", {}).get("D_eff", {})
        md += [
            f"## Residual — {model['name']}",
            "",
            f"- leftover vs stage r={de.get('leftover_vs_stage_r', float('nan')):.3f} "
            f"p={de.get('leftover_vs_stage_p', float('nan')):.2e}  "
            f"(structure R²={de.get('structure_R2', float('nan')):.3f})",
            f"- FTD vs AD leftover p={de.get('mw_ftd_ad_p', float('nan')):.2e}",
            f"- LOSO median r={jk.get('median_r', float('nan')):.3f}  "
            f"frac(r>0)={jk.get('frac_positive', float('nan')):.2f}",
            f"- patients MMSE ΔR²(+D_eff)={mm.get('delta_R2', float('nan')):+.3f}",
            "",
        ]
    if diff:
        t = (diff.get("markers") or {}).get("tmpl_ftd_minus_ad", {})
        md += [
            "## Subject-level D / atrophy templates",
            "",
            f"- n={diff.get('n')} {diff.get('counts')}",
            f"- `tmpl_ftd_minus_ad` leftover r={t.get('leftover_r', float('nan')):.3f} "
            f"p={t.get('leftover_p', float('nan')):.2e}  "
            f"FTD vs AD p={t.get('mw_ftd_ad_p', float('nan')):.2e}",
            "",
        ]
    models = ((xfer.get("board") or {}).get("models")) or {}
    if models:
        md += ["## ds005385 ρ transfer", ""]
        for tag, summary in models.items():
            m = summary.get("markers", {})
            rho = m.get("rho_std", {})
            deff = m.get("D_eff", {})
            md.append(
                f"- **{tag}** ρ_std ICC={rho.get('icc', float('nan')):.3f}  "
                f"age r={rho.get('age_corr_ses1', float('nan')):.3f} ({rho.get('age_vs_disease_note')});  "
                f"D_eff ICC={deff.get('icc', float('nan')):.3f}  "
                f"age r={deff.get('age_corr_ses1', float('nan')):.3f} ({deff.get('age_vs_disease_note')})"
            )
        md.append("")
    md += [
        "## Honest ceiling (this run)",
        "",
        "- Scalar leftover only → NeuroImage / Brain Communications, not NC.",
        "- NC requires: sign-stable leftover **and** ds005385 direction match **and**",
        "  FTD/AD split on subject-D / template delta (or same-subject MRI later).",
        "",
        "Details: `outputs/eeg/structure_residual/BOARD.md`, "
        "`outputs/eeg/subject_diffusion/BOARD.md`, "
        "`outputs/eeg/longitudinal_ds005385_rho/BOARD.md`.",
    ]
    out = _REPO / "outputs/eeg/NATURE_TRACK_BOARD.md"
    out.write_text("\n".join(md))
    print("\n".join(md), flush=True)
    print(f"[nature-track] → {out}", flush=True)


if __name__ == "__main__":
    main()
