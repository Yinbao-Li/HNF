#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBS P–S tradeoff grid under Step-4 fairness protocol.

Fairness rules (enforced here):
  1. Same disjoint split: ``obs_matched_adapt_split_randoffset``
  2. Same treatment tables: ZS vs ZS, adapt vs adapt — never mixed
  3. Same OBS train budget / epochs for light-adapt peers (default 8)
  4. Primary metric = pick-only F1 on **holdout only** (never train-pool)
  5. EQT/PhaseNet light-adapt use the same split + epoch budget
  6. Disclose HNF ``seq_len`` (resampling of the same 60 s window); SB models
     keep native SeisBench length — physical window is identical

Grid: tune ∈ {heads+onset, trunk-tail} × seq_len ∈ {800, 1200, 1600}
Select by score = 0.6·P + 0.4·S with soft floor S≥0.60.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "obs_cmp", _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
)
obs_cmp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(obs_cmp)

from tools.analyze_stead_picking import load_model
from tools.obs_matched_split import load_split_samples
from tools.train_stead_picking import set_seed


SPLIT = "outputs/obs_matched_adapt_split_randoffset/split.json"
STEAD_CKPT = "outputs/run28/28_ms_fresnel_phys_20ep/best.pt"

# Known prior runs (will be re-evaluated on holdout for a single consistent board)
PRIOR = {
    ("heads+onset", 800): "outputs/obs_light_adapt_run28_randoff_onset/best.pt",
    ("heads+onset", 1600): "outputs/obs_light_adapt_run28_randoff_onset1600/best.pt",
    ("trunk-tail", 1600): "outputs/obs_light_adapt_run28_randoff_trunktail1600/best.pt",
    ("heads", 800): "outputs/obs_light_adapt_run28_randoff_heads/best.pt",
}

GRID = [
    ("heads+onset", 800),
    ("heads+onset", 1200),
    ("heads+onset", 1600),
    ("trunk-tail", 800),
    ("trunk-tail", 1200),
    ("trunk-tail", 1600),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair OBS P–S tradeoff grid")
    p.add_argument("--output-dir", default="outputs/obs_ps_tradeoff_grid")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--p-weight", type=float, default=0.6, help="Score = p_w*P + (1-p_w)*S")
    p.add_argument("--s-floor", type=float, default=0.60)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="Only eval existing ckpts")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def cell_dir(root: Path, tune: str, seq_len: int) -> Path:
    tag = tune.replace("+", "").replace("-", "")
    return root / f"hnf_{tag}_L{seq_len}"


def train_hnf_cell(tune: str, seq_len: int, out: Path, args: argparse.Namespace) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    best = out / "best.pt"
    if best.exists() and not args.force_retrain:
        print(f"[grid] reuse {best}", flush=True)
        return best
    prior = PRIOR.get((tune, seq_len))
    if prior and Path(prior).exists() and not args.force_retrain:
        # Symlink / copy prior into cell dir for uniform layout
        import shutil

        shutil.copy2(prior, best)
        print(f"[grid] adopt prior {prior} → {best}", flush=True)
        return best
    if args.skip_train:
        raise FileNotFoundError(f"missing ckpt for {tune}@{seq_len}")

    cmd = [
        sys.executable,
        "tools/train_obs_light_adapt.py",
        "--checkpoint",
        STEAD_CKPT,
        "--split-json",
        SPLIT,
        "--tune",
        tune,
        "--seq-len",
        str(seq_len),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "2" if seq_len >= 1200 else str(args.batch_size),
        "--det-loss-weight",
        "0",
        "--p-loss-weight",
        "2.0" if "trunk" in tune else "1.5",
        "--s-loss-weight",
        "1.0" if "trunk" in tune else "1.2",
        "--output-dir",
        str(out),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if seq_len >= 1200:
        cmd += ["--force-sparse-band", "--cap-local-window-sec", "8.0"]
    print("[grid] TRAIN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0 or not best.exists():
        raise RuntimeError(f"train failed for {tune}@{seq_len} rc={rc}")
    return best


def train_sb(model: str, out: Path, args: argparse.Namespace) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    best = out / "best.pt"
    if best.exists() and not args.force_retrain:
        return best
    # Prefer prior EQT randoff
    if model == "eqt":
        prior = Path("outputs/obs_light_adapt_eqt_randoff/best.pt")
        if prior.exists() and not args.force_retrain:
            import shutil

            shutil.copy2(prior, best)
            return best
    if args.skip_train and not best.exists():
        raise FileNotFoundError(model)
    cmd = [
        sys.executable,
        "tools/train_obs_sb_light_adapt.py",
        "--model",
        model,
        "--weights",
        "stead",
        "--split-json",
        SPLIT,
        "--tune",
        "heads",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(out),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    print("[grid] TRAIN-SB", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0 or not best.exists():
        raise RuntimeError(f"SB train failed {model} rc={rc}")
    return best


@torch.no_grad()
def eval_hnf_holdout(ckpt: Path, seq_len: int, samples, device) -> dict:
    model, _ = load_model(ckpt, device)
    t0 = time.time()
    try:
        m = obs_cmp.eval_hnf(
            model, samples, device, seq_len, 60.0, 0.3, 0.5, 0.5, batch_size=4 if seq_len >= 1200 else 8
        )["pick_only"]
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    m["sec_per_trace"] = (time.time() - t0) / max(len(samples), 1)
    m["seq_len"] = seq_len
    return m


@torch.no_grad()
def eval_sb_holdout(ckpt: Path, model_kind: str, samples, device) -> dict:
    """Load adapted SB weights saved by train_obs_sb_light_adapt."""
    import seisbench.models as sbm

    name = "EQTransformer" if model_kind == "eqt" else "PhaseNet"
    # Rebuild architecture then load state
    model = getattr(sbm, name).from_pretrained("stead")
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    kind = "eqt" if model_kind == "eqt" else "phasenet"
    n_ch = 3
    norm = "peak"
    t0 = time.time()
    try:
        m = obs_cmp.eval_seisbench(
            model, samples, device, 0.3, 0.5, 0.5, 8, kind, n_ch, norm
        )["pick_only"]
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    m["sec_per_trace"] = (time.time() - t0) / max(len(samples), 1)
    return m


def score(m: dict, p_w: float, s_floor: float) -> float:
    p, s = float(m["p_f1"]), float(m["s_f1"])
    base = p_w * p + (1.0 - p_w) * s
    if s < s_floor:
        base -= 0.15 * (s_floor - s)
    return base


def write_outputs(out: Path, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid_report.json").write_text(json.dumps(report, indent=2))

    # Pareto scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in report["hnf_grid"]:
        ax.scatter(row["p_f1"], row["s_f1"], s=80, label=row["label"])
        ax.annotate(row["label"], (row["p_f1"], row["s_f1"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    for row in report["adapt_baselines"]:
        ax.scatter(row["p_f1"], row["s_f1"], marker="*", s=160, label=row["label"])
    ax.set_xlabel("P-F1 (holdout pick-only)")
    ax.set_ylabel("S-F1 (holdout pick-only)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title("OBS adapt P–S (same split; holdout only)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig_path = out / "obs_ps_pareto.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    docs = _REPO_ROOT / "docs" / "figures" / "obs_ps_pareto.png"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_bytes(fig_path.read_bytes())

    lines = [
        "# OBS P–S Tradeoff Grid (fair protocol)",
        "",
        "## Fairness",
        f"- split: `{SPLIT}` (disjoint holdout n={report['n_holdout']})",
        "- tables separated: **A zero-shot** vs **B matched light-adapt**",
        f"- light-adapt epochs: **{report['epochs']}** (HNF and EQT/PN)",
        "- metrics: pick-only F1 on **holdout only**",
        "- physical window: 60 s; HNF `seq_len` is resampling (disclosed per row)",
        "",
        "## A. Zero-shot (train=STEAD → eval=OBS holdout)",
        "",
        "| Model | P-F1 | S-F1 |",
        "|-------|-----:|-----:|",
    ]
    for r in report["zs_baselines"]:
        lines.append(f"| `{r['label']}` | {r['p_f1']:.3f} | {r['s_f1']:.3f} |")
    lines += [
        "",
        "## B. Matched light-adapt (same split/epochs → eval=OBS holdout)",
        "",
        "| Model | seq_len | P-F1 | S-F1 | score |",
        "|-------|--------:|-----:|-----:|------:|",
    ]
    for r in report["hnf_grid"]:
        lines.append(
            f"| `{r['label']}` | {r['seq_len']} | {r['p_f1']:.3f} | {r['s_f1']:.3f} | {r['score']:.3f} |"
        )
    for r in report["adapt_baselines"]:
        lines.append(
            f"| `{r['label']}` | — | {r['p_f1']:.3f} | {r['s_f1']:.3f} | {r['score']:.3f} |"
        )
    best = report["selected"]
    lines += [
        "",
        f"**Selected HNF (score={best['score']:.3f}, S-floor={report['s_floor']}):** "
        f"`{best['label']}` P={best['p_f1']:.3f} S={best['s_f1']:.3f}",
        "",
        "- figure: `docs/figures/obs_ps_pareto.png`",
        "",
        "## C. Full OBS-pretrained reference (not same budget — do not mix into B)",
        "",
    ]
    for r in report.get("full_refs", []):
        lines.append(f"- `{r['label']}`: P={r['p_f1']:.3f} S={r['s_f1']:.3f} ({r['note']})")
    (out / "grid_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("[grid] loading holdout …", flush=True)
    holdout, _, _ = load_split_samples(SPLIT, "holdout")
    print(f"[grid] holdout n={len(holdout)}", flush=True)

    hnf_rows = []
    for tune, seq_len in GRID:
        cdir = cell_dir(out, tune, seq_len)
        try:
            ckpt = train_hnf_cell(tune, seq_len, cdir, args)
        except Exception as e:
            print(f"[grid] SKIP {tune}@{seq_len}: {e}", flush=True)
            continue
        print(f"[grid] eval holdout {tune}@{seq_len} …", flush=True)
        m = eval_hnf_holdout(Path(ckpt), seq_len, holdout, device)
        sc = score(m, args.p_weight, args.s_floor)
        row = {
            "label": f"HNF({tune}/L{seq_len})",
            "tune": tune,
            "seq_len": seq_len,
            "ckpt": str(ckpt),
            "p_f1": m["p_f1"],
            "s_f1": m["s_f1"],
            "p_mae_sec": m["p_mae_sec"],
            "s_mae_sec": m["s_mae_sec"],
            "score": sc,
            "sec_per_trace": m["sec_per_trace"],
        }
        hnf_rows.append(row)
        print(f"  P={m['p_f1']:.3f} S={m['s_f1']:.3f} score={sc:.3f}", flush=True)

    adapt_base = []
    if not args.skip_baselines:
        for model in ("eqt", "phasenet"):
            bdir = out / f"sb_{model}_heads"
            try:
                ckpt = train_sb(model, bdir, args)
                print(f"[grid] eval holdout {model} adapt …", flush=True)
                m = eval_sb_holdout(Path(ckpt), model, holdout, device)
                sc = score(m, args.p_weight, args.s_floor)
                label = "EQT(STEAD+OBS-adapt)" if model == "eqt" else "PhaseNet(STEAD+OBS-adapt)"
                adapt_base.append({
                    "label": label,
                    "p_f1": m["p_f1"],
                    "s_f1": m["s_f1"],
                    "p_mae_sec": m["p_mae_sec"],
                    "s_mae_sec": m["s_mae_sec"],
                    "score": sc,
                    "ckpt": str(ckpt),
                })
                print(f"  {label} P={m['p_f1']:.3f} S={m['s_f1']:.3f}", flush=True)
            except Exception as e:
                print(f"[grid] SKIP {model}: {e}", flush=True)

    # ZS baselines (reuse protocol; quick eval HNF STEAD + SB STEAD)
    zs = []
    print("[grid] ZS baselines …", flush=True)
    m = eval_hnf_holdout(Path(STEAD_CKPT), 800, holdout, device)
    zs.append({"label": "HNF(run28/STEAD)", "p_f1": m["p_f1"], "s_f1": m["s_f1"]})
    if not args.skip_baselines:
        for model, label in (("eqt", "EQT(STEAD)"), ("phasenet", "PhaseNet(STEAD)")):
            try:
                name = "EQTransformer" if model == "eqt" else "PhaseNet"
                import seisbench.models as sbm

                sb = getattr(sbm, name).from_pretrained("stead").to(device)
                mm = obs_cmp.eval_seisbench(
                    sb, holdout, device, 0.3, 0.5, 0.5, 8,
                    "eqt" if model == "eqt" else "phasenet", 3, "peak",
                )["pick_only"]
                zs.append({"label": label, "p_f1": mm["p_f1"], "s_f1": mm["s_f1"]})
            except Exception as e:
                print(f"[grid] ZS skip {label}: {e}", flush=True)

    selected = max(hnf_rows, key=lambda r: r["score"]) if hnf_rows else {}
    # Soft-copy best into canonical path
    if selected:
        canon = out / "selected_hnf"
        canon.mkdir(exist_ok=True)
        import shutil

        shutil.copy2(selected["ckpt"], canon / "best.pt")
        (canon / "meta.json").write_text(json.dumps(selected, indent=2))

    report = {
        "split": SPLIT,
        "n_holdout": len(holdout),
        "epochs": args.epochs,
        "p_weight": args.p_weight,
        "s_floor": args.s_floor,
        "fairness": [
            "same disjoint holdout",
            "ZS and adapt tables separated",
            "same epoch budget for light-adapt peers",
            "holdout-only primary metrics",
            "disclose HNF seq_len; same 60s physical window",
        ],
        "hnf_grid": hnf_rows,
        "adapt_baselines": adapt_base,
        "zs_baselines": zs,
        "selected": selected,
        "full_refs": [
            {
                "label": "EQT(OBS) / PhaseNet(OBS) full-pretrained",
                "p_f1": 0.79,
                "s_f1": 0.49,
                "note": "different budget/channels — Table C only",
            }
        ],
    }
    write_outputs(out, report)
    print(open(out / "grid_report.md").read())


if __name__ == "__main__":
    main()
