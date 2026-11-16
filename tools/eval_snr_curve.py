#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-SNR dev curve: STEAD *val* events with real STEAD val noise injected at fixed SNRs.

SNR is defined as in training (power of the 4 s window after P / power of the noise, hnf.robust_aug).
The event set, noise traces and their pairing are fixed by --seed, so every model sees identical inputs.
A fixed set of noise-only val traces is appended to every SNR level so precision includes false alarms.
Thresholds follow EQT (det 0.5, P/S 0.3).

  python tools/eval_snr_curve.py --c2f base=outputs/stead_c2f_imp_RSP/last.pt,kal=outputs/x/last.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from hnf.robust_aug import add_real_noise  # noqa: E402
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.eval_shift_invariance import load_sbc, run_c2f, run_sb, run_sbc  # noqa: E402
from run_paper_stead_triple_compare import load_sb  # noqa: E402  (path set by eval_shift_invariance)
from tools.sweep_seisbench_threshold import MODEL_MAP, metrics_at  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402

T = 6000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c2f", default="", help="name=ckpt,...")
    p.add_argument("--sbc", default="", help="name=ckpt,... (tools/train_sb_stead_aug.py checkpoints)")
    p.add_argument("--seisbench", default="", help="official SeisBench STEAD weights: eqt,phasenet")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--snrs", default="clean,20,10,5,0,-5")
    p.add_argument("--n-events", type=int, default=3000)
    p.add_argument("--n-noise", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tolerances", default="0.1,0.5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("outputs/arch_budget/snr_curve.json"))
    return p.parse_args()


def load_sets(n_ev, n_nz, seed, split="val"):
    ds = STEADPickingDataset(split, seq_len=T, load_geometry=False)
    refs = [ds.refs[i] for i in range(len(ds))]
    ev = [r for r in refs if r.is_event == 1 and r.p_sample is not None and r.s_sample is not None
          and 0 < r.p_sample < T - 400 and r.s_sample < T]
    nz = [r for r in refs if r.is_event == 0]
    rng = np.random.default_rng(seed)
    ev = [ev[i] for i in rng.permutation(len(ev))[:n_ev]]
    nz = [nz[i] for i in rng.permutation(len(nz))]
    nz_only, nz_pool = nz[:n_nz], nz[n_nz:n_nz + n_ev]

    def wave(r):
        return np.asarray(ds._get_handle(r.chunk)["data"][r.trace_name][()], dtype=np.float32).T

    ev_w = np.stack([wave(r) for r in ev])
    pool_w = np.stack([wave(r) for r in nz_pool])
    only_w = np.stack([wave(r) for r in nz_only])
    y_ev = {"p_idx": torch.tensor([r.p_sample for r in ev]), "s_idx": torch.tensor([r.s_sample for r in ev])}
    return ev_w, pool_w, only_w, y_ev


def main():
    args = parse_args()
    device = torch.device(args.device)
    ev_w, pool_w, only_w, y_ev = load_sets(args.n_events, args.n_noise, args.seed, args.split)
    n_ev, n_nz = len(ev_w), len(only_w)
    print(f"[snr] events={n_ev} noise-only={n_nz} injection pool={len(pool_w)}", flush=True)
    y = {
        "p_idx": torch.cat([y_ev["p_idx"], torch.full((n_nz,), -1)]),
        "s_idx": torch.cat([y_ev["s_idx"], torch.full((n_nz,), -1)]),
        "p_valid": torch.cat([torch.ones(n_ev, dtype=torch.bool), torch.zeros(n_nz, dtype=torch.bool)]),
        "s_valid": torch.cat([torch.ones(n_ev, dtype=torch.bool), torch.zeros(n_nz, dtype=torch.bool)]),
        "is_event": torch.cat([torch.ones(n_ev, dtype=torch.bool), torch.zeros(n_nz, dtype=torch.bool)]),
    }
    tols = [float(t) for t in args.tolerances.split(",")]
    snrs = [s.strip() for s in args.snrs.split(",")]
    report = {"split": args.split, "n_events": n_ev, "n_noise_only": n_nz, "seed": args.seed, "tolerances": tols,
              "results": {}}

    inputs = {}
    for s in snrs:
        if s == "clean":
            w = ev_w
        else:
            w = np.stack([add_real_noise(ev_w[i], int(y_ev["p_idx"][i]), pool_w[i], float(s)) for i in range(n_ev)])
        inputs[s] = np.concatenate([w, only_w]).astype(np.float32)

    specs = [(n, "c2f", ck) for n, ck in (i.split("=", 1) for i in args.c2f.split(",") if i)]
    specs += [(n, "sbc", ck) for n, ck in (i.split("=", 1) for i in args.sbc.split(",") if i)]
    specs += [(MODEL_MAP[k][0], "sb", k) for k in args.seisbench.split(",") if k]
    for name, kind, ck in specs:
        if kind == "c2f":
            model = load_model(Path(ck), device)[0]
            mk = "eqt"
            run = lambda w: run_c2f(model, w, device, args.batch_size, args.det_threshold)  # noqa: E731
        elif kind == "sbc":
            mk, model = load_sbc(Path(ck), device)
            run = lambda w: run_sbc(model, mk, w, device, args.batch_size)  # noqa: E731
        else:
            _, cls, weights, mk = MODEL_MAP[ck]
            model = load_sb(cls, weights).to(device).eval()
            run = lambda w: run_sb(model, mk, w, device, args.batch_size)  # noqa: E731
        row = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=kind == "c2f"):
            for s in snrs:
                det, p, sp = run(inputs[s])
                row[s] = {}
                for tol in tols:
                    m = metrics_at(det, p, sp, y, mk, args.pick_threshold, args.det_threshold,
                                   max(1, int(round(tol * 100))), device)
                    row[s][f"{tol:g}"] = {k: m[k] for k in m if k.startswith(("p_", "s_", "det_"))}
        report["results"][name] = row
        print(f"  {name:10s} " + "  ".join(
            f"{s}: P={row[s]['0.5']['p_f1']:.3f}/{row[s]['0.1']['p_f1']:.3f} S={row[s]['0.5']['s_f1']:.3f}/{row[s]['0.1']['s_f1']:.3f}"
            for s in snrs), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output}", flush=True)


if __name__ == "__main__":
    main()
