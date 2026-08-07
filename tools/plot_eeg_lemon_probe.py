#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LEMON probe figure: EC–EO reliability, age, leftover vs GM/ICV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from hnf.eeg_subject_diffusion import pearson_r, residualize, sex_to_float, spearman_r


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="outputs/eeg/lemon_probe/subjects_probe.csv")
    p.add_argument("--out", default="docs/figures/eeg/eeg_lemon_probe.png")
    return p.parse_args()


def _col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan")))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)


def _scatter(ax, x, y, *, c, xlabel, ylabel, title):
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=18, c=c[m] if c is not None else "#3b6ea5", alpha=0.75, edgecolors="none")
    r_s, p_s = spearman_r(x, y)
    r_p = pearson_r(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nρ={r_s:.2f} p={p_s:.3g}  r={r_p:.2f} n={int(m.sum())}", fontsize=9)
    ax.grid(True, alpha=0.25)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    with Path(args.csv).open() as fh:
        rows = list(csv.DictReader(fh))
    age = _col(rows, "age")
    sex = sex_to_float([r.get("sex", "") for r in rows])
    demo = np.column_stack([age, sex])
    gm = _col(rows, "gm_icv")
    deff = _col(rows, "native_v3_ec_D_eff")
    deff_eo = _col(rows, "native_v3_eo_D_eff")
    deff_res = _col(rows, "native_v3_ec_D_eff_res")
    rstd = _col(rows, "native_v3_ec_rho_std")
    tmtb = _col(rows, "tmt_b")
    gm_res, _, _ = residualize(gm, demo)
    tmt_res, _, _ = residualize(tmtb, demo)
    grp = np.array([0.2 if r.get("age_group") == "young" else 0.85 for r in rows])
    cmap = plt.cm.coolwarm

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    _scatter(axes[0, 0], deff, deff_eo, c=cmap(grp), xlabel="EC $D_\\mathrm{eff}$", ylabel="EO $D_\\mathrm{eff}$", title="H3 reliability (v3)")
    _scatter(axes[0, 1], age, deff, c=cmap(grp), xlabel="age (bin midpoint)", ylabel="EC $D_\\mathrm{eff}$", title="H4 age (v3; predicted +)")
    _scatter(axes[0, 2], age, rstd, c=cmap(grp), xlabel="age (bin midpoint)", ylabel=r"EC $\rho_\mathrm{std}$", title=r"H4b age $\rho_\mathrm{std}$ (predicted −)")
    _scatter(axes[1, 0], gm_res, deff_res, c=cmap(grp), xlabel="GM/ICV | age+sex", ylabel=r"leftover $D_\mathrm{eff}$", title="H1 structure (predicted −)")
    _scatter(axes[1, 1], tmt_res, deff_res, c=cmap(grp), xlabel="TMT-B | age+sex", ylabel=r"leftover $D_\mathrm{eff}$", title="H5 cognition (predicted +)")
    _scatter(axes[1, 2], age, deff_res, c=cmap(grp), xlabel="age", ylabel=r"leftover $D_\mathrm{eff}$", title="control: leftover ⊥ age")
    fig.suptitle("LEMON · frozen AHEPA v3 probe  (cool=young, warm=old)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] → {out}", flush=True)


if __name__ == "__main__":
    main()
