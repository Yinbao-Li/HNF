#!/usr/bin/env python3
"""Quick threshold sweep for run14 checkpoints."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval_stead_picking import evaluate_checkpoint

OUT = ROOT / "outputs" / "run14" / "threshold_results.json"

CHECKPOINTS = {
    "01_seq800": ROOT / "outputs/ablation/01_seq800/best.pt",
    "14_main": ROOT / "outputs/run14/14_main/best.pt",
    "14_sharp": ROOT / "outputs/run14/14_sharp/best.pt",
}
THRESHOLDS = [0.1, 0.3]

results = {}
for name, ckpt in CHECKPOINTS.items():
    if not ckpt.is_file():
        continue
    results[name] = {}
    for th in THRESHOLDS:
        print(f"[sweep] {name} threshold={th}", flush=True)
        m = evaluate_checkpoint(
            str(ckpt),
            {"pick_threshold": th, "seq_len": 800},
            post_process_p_before_s=True,
        )
        results[name][str(th)] = {
            "det_f1": m["det_f1"],
            "p_f1": m["p_f1"],
            "s_f1": m["s_f1"],
            "p_mae_sec": m["p_mae_sec"],
            "s_mae_sec": m["s_mae_sec"],
        }
        print(
            f"  det={m['det_f1']:.4f} P={m['p_f1']:.4f} S={m['s_f1']:.4f}",
            flush=True,
        )

OUT.write_text(json.dumps(results, indent=2))
print(f"[sweep] saved {OUT}")
