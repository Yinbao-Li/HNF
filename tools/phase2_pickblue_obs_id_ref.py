#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.3 Path B — PickBlue 4C OBS in-distribution reference (separate table).

NOT a STEAD→OBS cross-domain baseline. Same SeisBench OBS event-holdout test split,
but inputs are 4C Z12H (PickBlue) vs the paper's 3C 21Z transfer rows.

  python tools/phase2_pickblue_obs_id_ref.py --device cuda \
      --output-dir results/phase2/2_3b_pickblue_obs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

_REPO = Path(__file__).resolve().parents[1]
for sub in ("", "scripts/paper"):
    if str(_REPO / sub) not in sys.path:
        sys.path.insert(0, str(_REPO / sub))

import seisbench.data as sbd  # noqa: E402
import seisbench.models as sbm  # noqa: E402
from run_paper_obs_picking_compare import normalize_wave  # noqa: E402
from tools.eval_ethz_cross import P_COLS, S_COLS, T, first_arrival  # noqa: E402
from tools.sweep_seisbench_threshold import metrics_at  # noqa: E402

GROUP_COL = "trace_instrument_type"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bases", default="phasenet,eqtransformer")
    p.add_argument("--split", default="test")
    p.add_argument("--max-traces", type=int, default=0)
    p.add_argument("--p-min", type=float, default=5.0)
    p.add_argument("--p-max", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tolerances", default="0.1,0.2,0.5")
    p.add_argument("--bandpass", default="1,45", help="applied to Z12; H gets 0.5 Hz high-pass per PickBlue docs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=Path("results/phase2/2_3b_pickblue_obs"))
    return p.parse_args()


def cut_window_c(w: np.ndarray, p: int, target_p: int) -> tuple[np.ndarray, int, int]:
    """(C, T) window with P at target_p; pad with pre-P noise."""
    c = w.shape[0]
    start = p - target_p
    noise = w[:, : max(p - 50, 100)]
    if noise.shape[1] < 10:
        noise = np.zeros((c, 100), dtype=w.dtype)

    def pad(n):
        reps = int(np.ceil(n / noise.shape[1]))
        return np.tile(noise, reps)[:, :n]

    left = max(0, -start)
    seg = w[:, max(0, start) : max(0, start) + T - left]
    right = T - left - seg.shape[1]
    out = np.concatenate([pad(left), seg, pad(right)], axis=1) if (left or right) else seg
    return out.astype(np.float32), left, right


def load_obs_4c(split: str):
    data = sbd.OBS(sampling_rate=100, component_order="Z12H", download_if_missing=False)
    data = getattr(data, split)()
    order = data.metadata["trace_component_order"].astype(str)
    data.filter(
        order.str.contains("Z")
        & order.str.contains("1")
        & order.str.contains("2")
        & order.str.contains("H")
    )
    return data


def preprocess_4c(waves: np.ndarray, bandpass: str) -> np.ndarray:
    """waves (N,4,T) Z12H. Bandpass Z12; high-pass H at 0.5 Hz (PickBlue doc)."""
    out = waves.copy()
    if bandpass:
        lo, hi = (float(x) for x in bandpass.split(","))
        sos = butter(4, [lo, hi], btype="bandpass", fs=100.0, output="sos")
        out[:, :3] = sosfiltfilt(sos, out[:, :3], axis=-1).astype(np.float32)
    sos_h = butter(4, 0.5, btype="highpass", fs=100.0, output="sos")
    out[:, 3:4] = sosfiltfilt(sos_h, out[:, 3:4], axis=-1).astype(np.float32)
    return out


@torch.no_grad()
def run_pickblue(model, kind: str, waves: np.ndarray, device, bs: int):
    """waves (N,4,T) already Z12H order matching model.component_order."""
    labels = "".join(getattr(model, "labels", "PSN") or "PSN").upper()
    dets, ps, ss = [], [], []
    for a in range(0, len(waves), bs):
        chunk = waves[a : a + bs]
        x = torch.from_numpy(np.stack([normalize_wave(w, "peak") for w in chunk])).to(device)
        out = model(x)
        if kind == "eqt":
            det, p, s = (t.squeeze(1) if t.dim() == 3 else t for t in out)
            dets.append(det.amax(-1).float().cpu())
            ps.append(p.half().cpu())
            ss.append(s.half().cpu())
        else:
            # PhaseNet: (B,3,T) — labels typically N,P,S or P,S,N
            if labels.startswith("NP"):
                p, s = out[:, 1], out[:, 2]
            else:
                p, s = out[:, 0], out[:, 1]
            ps.append(p.half().cpu())
            ss.append(s.half().cpu())
    return (torch.cat(dets) if dets else None), torch.cat(ps), torch.cat(ss)


def write_summary(report: dict, path: Path):
    lines = [
        "# Phase 2.3 Path B — PickBlue 4C OBS in-distribution reference",
        "",
        "**Not a cross-domain baseline.** Separate from STEAD→OBS transfer tables.",
        "",
        f"- Split: SeisBench OBS `{report['split']}` (event-holdout; ID for PickBlue)",
        f"- Input: **4C Z12H** (H high-pass 0.5 Hz); paper transfer rows use **3C 21Z**",
        f"- Windows: {report['n_windows']} (S-valid {report['n_with_s']}); dead-ch dropped {report['n_dead_channel_dropped']}",
        f"- Protocol: P in [{report['p_range_sec'][0]}, {report['p_range_sec'][1]}] s; bandpass Z12={report['bandpass']}; th={report['pick_threshold']}/det={report['det_threshold']}",
        "",
        "| Model | Tol | P F1 | S F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["results"].items():
        for tol in report["tolerances"]:
            key = f"{tol:g}"
            lines.append(
                f"| {name} | {tol:g} | {row[key]['p_f1']:.4f} | {row[key]['s_f1']:.4f} |"
            )
    lines.append("")
    lines.append("## Disclosures (must travel with any table)")
    lines.append("1. In-distribution for PickBlue (OBS-trained, same corpus holdout).")
    lines.append("2. 4C vs paper 3C — not matched input dimensionality.")
    lines.append("3. Do not fold into STEAD-only transfer comparison.")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tols = [float(x) for x in args.tolerances.split(",")]
    rng = np.random.default_rng(args.seed)

    data = load_obs_4c(args.split)
    n = len(data) if args.max_traces <= 0 else min(args.max_traces, len(data))
    order = rng.permutation(len(data))[:n]

    waves, y, groups = [], {"p_idx": [], "s_idx": [], "p_valid": [], "s_valid": [], "is_event": []}, []
    n_pad = n_dead = 0
    for i in order:
        w, meta = data.get_sample(int(i))
        p = first_arrival(meta, P_COLS)
        if p is None or w.shape[0] != 4:
            continue
        w = np.nan_to_num(np.asarray(w, dtype=np.float32))
        if not (w.std(axis=1) > 0).all():
            n_dead += 1
            continue
        g = meta.get(GROUP_COL)
        groups.append(str(g) if isinstance(g, str) and g else "other")
        s = first_arrival(meta, S_COLS)
        p = int(round(p))
        tp = int(round(rng.uniform(args.p_min, args.p_max) * 100))
        win, left, right = cut_window_c(w, p, tp)
        n_pad += int(left > 0 or right > 0)
        s_rel = None if s is None else int(round(s)) - p + tp
        s_ok = s_rel is not None and 50 <= s_rel <= T - 50 and s_rel > tp
        waves.append(win)
        y["p_idx"].append(tp)
        y["s_idx"].append(s_rel if s_ok else 0)
        y["p_valid"].append(True)
        y["s_valid"].append(bool(s_ok))
        y["is_event"].append(True)

    waves = preprocess_4c(np.stack(waves), args.bandpass)
    y = {k: torch.tensor(v) for k, v in y.items()}
    print(
        f"[obs4c] {len(waves)} windows (S={int(y['s_valid'].sum())}), dead={n_dead}, pad={n_pad}",
        flush=True,
    )

    report = {
        "role": "OBS_in_distribution_reference_NOT_cross_domain",
        "dataset": "SeisBench OBS",
        "split": args.split,
        "component_order": "Z12H",
        "n_channels": 4,
        "paper_transfer_rows": "3C 21Z STEAD→OBS zero-shot — do not merge tables",
        "n_windows": len(waves),
        "n_with_s": int(y["s_valid"].sum()),
        "n_padded": n_pad,
        "n_dead_channel_dropped": n_dead,
        "bandpass": args.bandpass,
        "hydrophone_highpass_hz": 0.5,
        "p_range_sec": [args.p_min, args.p_max],
        "tolerances": tols,
        "pick_threshold": args.pick_threshold,
        "det_threshold": args.det_threshold,
        "results": {},
    }

    for base in [b for b in args.bases.split(",") if b]:
        model = sbm.PickBlue(base=base).to(device).eval()
        kind = "eqt" if "eqt" in base.lower() else "phasenet"
        name = f"PickBlue({base})-4C-Z12H"
        print(f"[eval] {name} ...", flush=True)
        det, p, s = run_pickblue(model, kind, waves, device, args.batch_size)
        row = {}
        for tol in tols:
            m = metrics_at(
                det,
                p,
                s,
                y,
                kind,
                args.pick_threshold,
                args.det_threshold,
                max(1, int(round(tol * 100))),
                device,
            )
            row[f"{tol:g}"] = {k: m[k] for k in m if k.startswith(("p_", "s_", "det_"))}
        by_group = {}
        for g in sorted(set(groups)):
            idx = torch.tensor([j for j, gg in enumerate(groups) if gg == g])
            if len(idx) < 50:
                continue
            m = metrics_at(
                None if det is None else det[idx],
                p[idx],
                s[idx],
                {k: v[idx] for k, v in y.items()},
                kind,
                args.pick_threshold,
                args.det_threshold,
                50,
                device,
            )
            by_group[g] = {"n": len(idx), "p_f1": m["p_f1"], "s_f1": m["s_f1"]}
        row["by_group_tol0.5"] = by_group
        report["results"][name] = row
        print(
            "  "
            + "  ".join(f"tol{t:g} P={row[f'{t:g}']['p_f1']:.4f} S={row[f'{t:g}']['s_f1']:.4f}" for t in tols),
            flush=True,
        )
        (args.output_dir / "pickblue_obs_id_ref.json").write_text(json.dumps(report, indent=2))

    write_summary(report, args.output_dir / "SUMMARY.md")
    print(f"[done] -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
