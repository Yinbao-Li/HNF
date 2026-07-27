#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build official Braindecode SOTA + HNF comparison table."""

from __future__ import annotations

import json
from pathlib import Path

from hnf.eeg_braindecode_models import SOTA_CITATIONS, display_name

_REPO = Path(__file__).resolve().parents[2]


def _ad_ftd_from_per_subject(per_subj: dict) -> float:
    yt, yp = [], []
    for row in per_subj.values():
        lab = int(row["label"])
        if lab not in (1, 2):
            continue
        yt.append(lab)
        yp.append(int(row["pred"]))
    return sum(a == b for a, b in zip(yt, yp)) / len(yt) if yt else float("nan")


def main() -> None:
    out = Path("outputs/eeg/adftd_braindecode_sota")
    sweep = json.loads((out / "sweep_report.json").read_text()) if (out / "sweep_report.json").is_file() else {"models": {}}

    stage1 = json.loads(Path("outputs/eeg/adftd_hnf_stage1/test_metrics.json").read_text())
    native = json.loads(Path("outputs/eeg/adftd_hnf_native_v3/test_metrics.json").read_text())
    clinical = json.loads(Path("outputs/eeg/clinical_breakthrough_native_v3/clinical_report.json").read_text())

    rows = [
        {
            "model": "HNF Stage-1",
            "source": "STEAD transfer",
            "subject_accuracy": stage1["subject_accuracy"],
            "ad_ftd_subject_accuracy": _ad_ftd_from_per_subject(stage1["per_subject"]),
            "auc_macro": stage1["auc_macro"],
            "accuracy": stage1["accuracy"],
            "macro_f1": stage1["macro_f1"],
            "n_params": 89442,
        },
        {
            "model": "HNF native v3",
            "source": "EEG-native HNF",
            "subject_accuracy": native["test_subject_accuracy"],
            "ad_ftd_subject_accuracy": clinical["test_metrics"]["ad_ftd_differential"]["accuracy"],
            "auc_macro": native["test_epoch_auc"],
            "accuracy": native["test_epoch_acc"],
            "macro_f1": None,
            "n_params": native["n_params"],
        },
    ]

    for key, block in sweep.get("models", {}).items():
        tm = block["test_metrics"]
        hp = block["best_hp"]
        rows.append(
            {
                "model": display_name(key),
                "source": SOTA_CITATIONS.get(key, "Braindecode"),
                "tuning": hp,
                "sweep_best_val_auc": block.get("sweep_best_val_auc"),
                "subject_accuracy": tm["subject_accuracy"],
                "ad_ftd_subject_accuracy": tm["ad_ftd_subject_accuracy"],
                "auc_macro": tm["auc_macro"],
                "accuracy": tm["accuracy"],
                "macro_f1": tm["macro_f1"],
                "n_params": tm["n_params"],
            }
        )

    summary = {
        "protocol": {
            "dataset": "OpenNeuro ds004504",
            "n_test_subjects": 18,
            "split_seed": 42,
            "input": "19ch x 10s @ 128Hz",
            "sota_library": "Braindecode >= 0.8 (official architectures)",
            "baseline_tuning": "val macro-AUC grid sweep (30ep) + best config retrain (50ep)",
            "selection": "best val macro-AUC checkpoint",
        },
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# EEG 分类对比（Braindecode 官方 SOTA + HNF）",
        "",
        "- 数据：ds004504，test N=18，split seed=42",
        "- SOTA：Braindecode 0.8 官方实现（EEGNetv4 / ShallowFBCSPNet / Deep4Net / EEGConformer）",
        "- 调参：各模型 val macro-AUC 网格搜索（30 epoch）→ 最优配置再训 50 epoch",
        "",
        "| 模型 | subject acc | AD↔FTD acc | macro-AUC | epoch acc | macro-F1 | 参数量 |",
        "|------|------------:|-----------:|----------:|----------:|---------:|-------:|",
    ]
    for r in rows:
        mf1 = r.get("macro_f1")
        mf1_s = f"{mf1:.3f}" if mf1 is not None else "—"
        lines.append(
            f"| {r['model']} | {r['subject_accuracy']:.3f} | {r['ad_ftd_subject_accuracy']:.3f} | "
            f"{r['auc_macro']:.3f} | {r['accuracy']:.3f} | {mf1_s} | {r['n_params']} |"
        )
    lines += ["", "## 文献来源", ""]
    for k, cite in SOTA_CITATIONS.items():
        if k in sweep.get("models", {}):
            lines.append(f"- **{display_name(k)}**: {cite}")
    md = "\n".join(lines)
    (out / "compare_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
