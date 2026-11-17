#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Threshold sweep for coarse-to-fine STEAD picker (val select → test report).

  python tools/sweep_stead_c2f_threshold.py \
    --checkpoint outputs/stead_c2f_750_v3/best.pt \
    --output-dir outputs/stead_c2f_750_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.picking_coarse_to_fine import build_c2f_picker
from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    det_pred_from_logits,
    finalize_metrics,
    picking_score,
    tolerance_bins,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.train_stead_picking import move_batch_to_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-val", type=int, default=8000)
    p.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--also-test-th", default="", help="extra test thresholds, comma-separated")
    return p.parse_args()


@torch.no_grad()
def cache_logits(model, loader, device):
    model.eval()
    packs = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        packs.append(
            {
                "det": out["det"].detach().cpu(),
                "p": out["p"].detach().cpu(),
                "s": out["s"].detach().cpu(),
                "det_true": batch["det"].detach().cpu(),
                "p_idx": batch["p_idx"].detach().cpu(),
                "s_idx": batch["s_idx"].detach().cpu(),
                "p_valid": batch["p_valid"].detach().cpu(),
                "s_valid": batch["s_valid"].detach().cpu(),
            }
        )
    return packs


def metrics_from_cache(packs, *, seq_len: int, pick_th: float, tol_sec: float):
    acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, tol_sec)
    for pack in packs:
        det_pred = det_pred_from_logits(pack["det"])
        det_true = pack["det_true"] > 0.5
        update_detection_counts(acc, det_pred, det_true)
        p_probs = torch.sigmoid(pack["p"])
        s_probs = torch.sigmoid(pack["s"])
        p_probs, s_probs = apply_p_before_s_constraint(p_probs, s_probs, pick_th)
        for probs, idx_key, valid_key, counts in [
            (p_probs, "p_idx", "p_valid", acc.p),
            (s_probs, "s_idx", "s_valid", acc.s),
        ]:
            update_picking_counts(
                counts, probs, det_pred, det_true,
                pack[valid_key] > 0, pack[idx_key], pick_th, tol, seq_len,
            )
    m = finalize_metrics(acc)
    m["pick_focus"] = picking_score(m, mode="pick_focus")
    m["score"] = picking_score(m, mode="mean")
    return m


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})
    # Path objects may be strings in json-roundtrip; tolerate
    model = build_c2f_picker(
        fine_len=int(_cfg_get(cfg, "fine_len", 6000)),
        coarse_len=int(_cfg_get(cfg, "coarse_len", 750)),
        embed_dim=int(_cfg_get(cfg, "embed_dim", 96)),
        skip_dim=int(_cfg_get(cfg, "skip_dim", 64)),
        mid_dim=int(_cfg_get(cfg, "mid_dim", 64)),
        num_shared_layers=int(_cfg_get(cfg, "num_shared_layers", 3)),
        num_branch_layers=int(_cfg_get(cfg, "num_branch_layers", 3)),
        gamma=float(_cfg_get(cfg, "gamma", 0.5)),
        omega=float(_cfg_get(cfg, "omega", 0.3)),
        vp=float(_cfg_get(cfg, "vp", 8.0)),
        vs=float(_cfg_get(cfg, "vs", 4.5)),
        local_window_sec=float(_cfg_get(cfg, "local_window_sec", 15.0)),
        dropout=float(_cfg_get(cfg, "dropout", 0.1)),
        sparse_band=bool(_cfg_get(cfg, "sparse_band", True)),
        principle=str(_cfg_get(cfg, "principle", "huygens_fresnel")),
        fusion=str(_cfg_get(cfg, "fusion", "sum")),
        wave_input=str(_cfg_get(cfg, "wave_input", "fused")),
        refine_max=float(_cfg_get(cfg, "refine_max", 4.0)),
        time_head_drop=float(_cfg_get(cfg, "time_head_drop", 0.0)),
        skip_bottleneck=int(_cfg_get(cfg, "skip_bottleneck", 0)),
        refine_pool_sec=float(_cfg_get(cfg, "refine_pool_sec", 0.0)),
        propagation=str(_cfg_get(cfg, "propagation", "huygens")),
        gate_floor=float(_cfg_get(cfg, "gate_floor", 0.0)),
        kalman=str(_cfg_get(cfg, "kalman", "none")),
        kalman_groups=int(_cfg_get(cfg, "kalman_groups", 8)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, cfg, ckpt


def main():
    args = parse_args()
    out_dir = args.output_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    model, cfg, ckpt = load_model(args.checkpoint, device)
    seq_len = int(cfg.get("fine_len", 6000))
    print(
        f"[sweep] ckpt={args.checkpoint} device={device} seq_len={seq_len} "
        f"thresholds={thresholds}",
        flush=True,
    )

    val_ds = STEADPickingDataset(
        "val",
        seq_len=seq_len,
        max_event_traces=args.max_val // 2,
        max_noise_traces=args.max_val // 2,
        label_sigma_sec=float(cfg.get("label_sigma_sec", 0.35)),
        seed=int(cfg.get("seed", 42)),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(f"[sweep] caching val logits n={len(val_ds)}…", flush=True)
    val_cache = cache_logits(model, val_loader, device)

    rows = []
    for th in thresholds:
        m = metrics_from_cache(val_cache, seq_len=seq_len, pick_th=th, tol_sec=args.tol_sec)
        rows.append({"split": "val", "pick_threshold": th, **m})
        print(
            f"  val th={th:.2f} det={m['det_f1']:.4f} P={m['p_f1']:.4f} "
            f"S={m['s_f1']:.4f} focus={m['pick_focus']:.4f} "
            f"P_MAE={m['p_mae_sec']:.4f}s S_MAE={m['s_mae_sec']:.4f}s",
            flush=True,
        )

    best = max(rows, key=lambda r: r["pick_focus"])
    best_th = float(best["pick_threshold"])
    print(
        f"[sweep] best_val th={best_th:.2f} pick_focus={best['pick_focus']:.4f} "
        f"P={best['p_f1']:.4f} S={best['s_f1']:.4f}",
        flush=True,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "best_epoch": ckpt.get("epoch") or ckpt.get("best", {}).get("epoch"),
        "n_params": sum(p.numel() for p in model.parameters()),
        "seq_len": seq_len,
        "tol_sec": args.tol_sec,
        "val_sweep": rows,
        "best_val": best,
        "selected_threshold": best_th,
        "baselines_manuscript": {
            "EQTransformer": {
                "n_params": 377_000,
                "det_f1": 0.9989,
                "p_f1": 0.9897,
                "s_f1": 0.9731,
                "p_mae_sec": 0.0462,
                "s_mae_sec": 0.0874,
            },
            "PhaseNet": {
                "n_params": 269_000,
                "det_f1": 0.9969,
                "p_f1": 0.9512,
                "s_f1": 0.9618,
                "p_mae_sec": 0.0723,
                "s_mae_sec": 0.0801,
            },
            "PNF_run28_800bin": {
                "n_params": 192_000,
                "det_f1": 0.9989,
                "p_f1": 0.9852,
                "s_f1": 0.9770,
                "p_mae_sec": 0.0169,
                "s_mae_sec": 0.0878,
                "note": "seq_len=800 resampled; not protocol-matched to c2f@6000",
            },
        },
    }

    if not args.skip_test:
        from tools.train_stead_c2f_picking import evaluate as eval_loader

        test_ds = STEADPickingDataset(
            "test",
            seq_len=seq_len,
            label_sigma_sec=float(cfg.get("label_sigma_sec", 0.35)),
            seed=int(cfg.get("seed", 42)),
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        # Streaming test at val-selected threshold only (+ optional refs via --also-test-th)
        also = []
        if hasattr(args, "also_test_th") and args.also_test_th:
            also = [float(x) for x in args.also_test_th.split(",") if x.strip()]
        test_ths = sorted(set([best_th] + also))
        for th in test_ths:
            print(f"[sweep] test eval th={th:.2f} n={len(test_ds)}…", flush=True)
            te = eval_loader(
                model, test_loader, device,
                seq_len=seq_len, pick_th=th, tol_sec=args.tol_sec,
            )
            key = "test_selected" if abs(th - best_th) < 1e-9 else f"test_th_{th:.2f}"
            report[key] = {"pick_threshold": th, **te}
            print(
                f"  test th={th:.2f} det={te['det_f1']:.4f} P={te['p_f1']:.4f} "
                f"S={te['s_f1']:.4f} focus={te['pick_focus']:.4f} "
                f"P_MAE={te['p_mae_sec']:.4f}s S_MAE={te['s_mae_sec']:.4f}s",
                flush=True,
            )

    (out_dir / "threshold_sweep.json").write_text(json.dumps(report, indent=2))

    # markdown snapshot for Paper B / NUMBERS
    te = report.get("test_selected") or report.get("best_val")
    md = []
    md.append("# STEAD c2f v3 threshold sweep\n")
    md.append(f"- checkpoint: `{args.checkpoint}` (best ep {report.get('best_epoch')})")
    md.append(f"- selected pick_threshold (val pick_focus): **{best_th:.2f}**")
    md.append(f"- params: **{report['n_params']:,}**; seq_len=**{seq_len}** @ 100 Hz; tol=0.5s\n")
    md.append("| Model | params | det F1 | P F1 | S F1 | P MAE | S MAE |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    if "test_selected" in report:
        t = report["test_selected"]
        md.append(
            f"| **HNF c2f v3** | {report['n_params']//1000}k | "
            f"{t['det_f1']:.4f} | {t['p_f1']:.4f} | {t['s_f1']:.4f} | "
            f"{t['p_mae_sec']:.4f} | {t['s_mae_sec']:.4f} |"
        )
    for name, b in report["baselines_manuscript"].items():
        md.append(
            f"| {name} | {b['n_params']//1000}k | {b['det_f1']:.4f} | "
            f"{b['p_f1']:.4f} | {b['s_f1']:.4f} | {b['p_mae_sec']:.4f} | {b['s_mae_sec']:.4f} |"
        )
    md.append("\nVal sweep (pick_focus):\n")
    for r in rows:
        md.append(
            f"- th={r['pick_threshold']:.2f}: focus={r['pick_focus']:.4f} "
            f"P={r['p_f1']:.4f} S={r['s_f1']:.4f}"
        )
    (out_dir / "NUMBERS_c2f_v3.md").write_text("\n".join(md) + "\n")
    print(f"[done] → {out_dir / 'threshold_sweep.json'}", flush=True)
    print(f"[done] → {out_dir / 'NUMBERS_c2f_v3.md'}", flush=True)


if __name__ == "__main__":
    main()
