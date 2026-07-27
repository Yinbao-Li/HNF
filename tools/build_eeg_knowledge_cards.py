#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn FDR-confirmed EEG markers into readable clinical knowledge cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTERPRETATION: dict[str, str] = {
    "rho_mean": "HNF 介质密度 ρ 的时间均值：反映核场在整段 epoch 上的平均“介质响应”强度。",
    "rho_std": "ρ 的时间波动：HC 上 ρ 动态更大，疾病组更平抑。",
    "rho_p90": "ρ 的 90 分位：捕捉 epoch 内高强度介质响应峰值。",
    "rho_cv": "ρ 变异系数 std/|mean|：归一化后的 ρ 动态范围。",
    "omega_rho": "mean(ω)·mean(ρ)：把学到的节律 ω 与 ρ 幅度耦合后的标量。",
    "hnf_alpha_energy": "HNF α 分支（~10 Hz 先验）包络能量：与经典 α 功率相关但经 Huygens 分支提取。",
    "hnf_theta_energy": "HNF θ 分支（~6 Hz 先验）包络能量。",
    "hnf_theta_alpha_ratio": "HNF θ/α 能量比（log 域）：AD 相关 slowing / α 相对变化。",
    "hnf_delta_energy": "HNF δ 分支（~2.5 Hz 先验）包络能量。",
    "theta_alpha_ratio": "经典 Welch θ−α 对数功率差。",
    "bp_alpha": "经典 α 带 log 功率。",
    "bp_theta": "经典 θ 带 log 功率。",
    "region_ft_contrast": "额颞能量对比：FTD 相关区域几何特征。",
    "region_pf_contrast": "顶额能量对比：AD/FTD 空间分布差异 probe。",
}

CONTRAST_READABLE = {
    "HC_vs_AD": "HC vs AD",
    "HC_vs_FTD": "HC vs FTD",
    "AD_vs_FTD": "AD vs FTD",
    "HC_vs_disease": "HC vs 疾病组（AD∪FTD）",
}


def _direction(means: list[float], groups: list[str]) -> str:
    if len(means) != len(groups):
        return ""
    pairs = list(zip(groups, means))
    hi = max(pairs, key=lambda x: x[1])
    lo = min(pairs, key=lambda x: x[1])
    return f"{hi[0]} > {lo[0]} ({hi[1]:.3f} vs {lo[1]:.3f})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clinical-report",
        default="outputs/eeg/clinical_breakthrough_native_v3/clinical_report.json",
    )
    p.add_argument("--output-dir", default="outputs/eeg/knowledge_cards_native_v3")
    p.add_argument("--include-train-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.clinical_report).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool = report.get("marker_mining_test_confirmation", [])
    if not args.include_train_only:
        pool = [r for r in pool if r.get("reject_fdr")]
    else:
        pool = [r for r in report.get("marker_mining_train", []) if r.get("reject_fdr")]

    cards = []
    for i, r in enumerate(pool, start=1):
        feat = r["feature"]
        card = {
            "id": i,
            "feature": feat,
            "contrast": r["contrast"],
            "contrast_readable": CONTRAST_READABLE.get(r["contrast"], r["contrast"]),
            "p": r["p"],
            "q": r["q"],
            "direction": _direction(r.get("means", []), r.get("groups", [])),
            "interpretation": INTERPRETATION.get(feat, "可解释 HNF / 频带特征（见 clinical suite）。"),
            "confirmed_on_test": bool(r.get("confirmed_from_train") and r.get("reject_fdr")),
        }
        cards.append(card)

    (out / "knowledge_cards.json").write_text(json.dumps(cards, indent=2))

    md_lines = [
        "# EEG 临床知识卡片（FDR 确认 marker）",
        "",
        f"来源：`{args.clinical_report}`",
        f"卡片数：**{len(cards)}**（test 确认）",
        "",
    ]
    for c in cards:
        md_lines += [
            f"## {c['id']}. `{c['feature']}` × {c['contrast_readable']}",
            "",
            f"- **方向**：{c['direction']}",
            f"- **统计**：p={c['p']:.2e}, q={c['q']:.2e}",
            f"- **含义**：{c['interpretation']}",
            f"- **test 复现**：{'是' if c['confirmed_on_test'] else '否（仅 train 发现）'}",
            "",
        ]
    md = "\n".join(md_lines)
    (out / "KNOWLEDGE_CARDS.md").write_text(md)
    print(md[:2000], flush=True)
    print(f"[knowledge-cards] wrote {out / 'KNOWLEDGE_CARDS.md'} ({len(cards)} cards)", flush=True)


if __name__ == "__main__":
    main()
