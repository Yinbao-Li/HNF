#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare STEAD and OBS in interpretability and physical-discovery outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STEAD vs OBS interpretability comparison")
    p.add_argument(
        "--stead-explain",
        default="outputs/stead_hnf_picking_run7/explain/explain_summary.json",
    )
    p.add_argument(
        "--obs-explain",
        default="outputs/run28_obs_full_800/explain_obs/explain_summary.json",
    )
    p.add_argument(
        "--obs-compare",
        default="outputs/obs_step4_run28_multichunk/obs_picking_compare_report.json",
    )
    p.add_argument(
        "--geo-confirm",
        default="outputs/paper_geo_confirm/geo_confirm_report.json",
    )
    p.add_argument("--output-dir", default="outputs/paper_stead_obs_interpret_compare")
    return p.parse_args()


def _load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _flatten_kernel_params(kernel_params: dict) -> dict[str, float]:
    out = {}
    for name, vals in kernel_params.items():
        for k, v in vals.items():
            if isinstance(v, (int, float)):
                out[f"{name}.{k}"] = float(v)
    return out


def _extract_example_stats(summary: dict | None) -> dict[str, float | None]:
    if not summary:
        return {}
    agg = dict(summary.get("aggregate") or {})
    if "mean_p_abs_err_sec" in agg or "mean_s_abs_err_sec" in agg:
        return agg
    p_err = [x.get("p_abs_err_sec") for x in summary.get("examples", []) if x.get("p_abs_err_sec") is not None]
    s_err = [x.get("s_abs_err_sec") for x in summary.get("examples", []) if x.get("s_abs_err_sec") is not None]
    return {
        "n_examples": len(summary.get("examples", [])),
        "mean_p_abs_err_sec": float(np.mean(p_err)) if p_err else None,
        "mean_s_abs_err_sec": float(np.mean(s_err)) if s_err else None,
    }


def _physical_claim_digest(geo_confirm: dict | None) -> list[dict]:
    if not geo_confirm:
        return []
    verdicts = []
    for item in geo_confirm.get("verdicts", []):
        verdicts.append(
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "note": item.get("note"),
            }
        )
    return verdicts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stead = _load_json(args.stead_explain)
    obs = _load_json(args.obs_explain)
    obs_compare = _load_json(args.obs_compare)
    geo_confirm = _load_json(args.geo_confirm)

    stead_stats = _extract_example_stats(stead)
    obs_stats = _extract_example_stats(obs)
    stead_k = _flatten_kernel_params((stead or {}).get("kernel_params", {}))
    obs_k = _flatten_kernel_params((obs or {}).get("kernel_params", {}))
    shared_keys = sorted(set(stead_k) & set(obs_k))
    kernel_delta = {
        k: {
            "stead": stead_k[k],
            "obs": obs_k[k],
            "delta_obs_minus_stead": obs_k[k] - stead_k[k],
        }
        for k in shared_keys
    }

    obs_rank = {}
    if obs_compare:
        for name, vals in (obs_compare.get("results") or {}).items():
            if "pick_only" in vals:
                obs_rank[name] = {
                    "p_f1": vals["pick_only"].get("p_f1"),
                    "s_f1": vals["pick_only"].get("s_f1"),
                    "p_mae_sec": vals["pick_only"].get("p_mae_sec"),
                    "s_mae_sec": vals["pick_only"].get("s_mae_sec"),
                }

    narrative = {
        "domain_shift_hypotheses": [
            "OBS 通常拥有更强站点噪声、海洋微震与仪器耦合差异，因此解释图中的 rho(t) 与 kernel_contrib 往往更弥散，局部因果支持更宽。",
            "STEAD 的事件窗更规则，P/S 相位与背景噪声分离度更高，因此解释性更接近单峰因果传播；OBS 更容易出现多峰或长尾支持。",
            "若 OBS 训练后 gamma / omega / wave_speed 等有效参数相对 STEAD 发生系统漂移，可视为模型对海底介质与传播复杂度的自适应重参数化。",
        ],
        "physical_discovery_focus": [
            "STEAD 侧更适合验证跨样本稳定的弱物理规律，例如 noise_ratio→pick_err_p、rho_p_lag→init_tt。",
            "OBS 侧更值得关注的是这些规律是否仍保留方向一致性，以及是否出现更强的 site/noise 依赖或更宽的不确定性区间。",
            "若 STEAD 的强规律在 OBS 上显著减弱，优先解释为域噪声、台站分布与观测系统差异，而不是立即否定规律本身。",
        ],
    }

    report = {
        "stead_explain": args.stead_explain,
        "obs_explain": args.obs_explain,
        "obs_compare": args.obs_compare,
        "geo_confirm": args.geo_confirm,
        "stead_example_stats": stead_stats,
        "obs_example_stats": obs_stats,
        "kernel_delta": kernel_delta,
        "obs_model_compare": obs_rank,
        "stead_physical_claims": _physical_claim_digest(geo_confirm),
        "narrative": narrative,
    }
    (out_dir / "stead_obs_interpret_compare.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# STEAD vs OBS: 可解释性与物理发现对比",
        "",
        "## 1. 解释性差异",
        "",
    ]
    if stead_stats:
        lines.append(
            f"- STEAD 示例: n={stead_stats.get('n_examples')} "
            f"P误差={stead_stats.get('mean_p_abs_err_sec')} s, "
            f"S误差={stead_stats.get('mean_s_abs_err_sec')} s"
        )
    if obs_stats:
        lines.append(
            f"- OBS 示例: n={obs_stats.get('n_examples')} "
            f"P误差={obs_stats.get('mean_p_abs_err_sec')} s, "
            f"S误差={obs_stats.get('mean_s_abs_err_sec')} s"
        )
    if kernel_delta:
        lines += ["", "### 核参数漂移", ""]
        for k in sorted(kernel_delta)[:20]:
            v = kernel_delta[k]
            lines.append(
                f"- `{k}`: STEAD={v['stead']:.4f}, OBS={v['obs']:.4f}, "
                f"Δ={v['delta_obs_minus_stead']:+.4f}"
            )
    lines += ["", "## 2. OBS 上的模型对比", ""]
    if obs_rank:
        for name, vals in obs_rank.items():
            lines.append(
                f"- `{name}`: P-F1={vals['p_f1']:.3f}, S-F1={vals['s_f1']:.3f}, "
                f"P-MAE={vals['p_mae_sec']:.3f}, S-MAE={vals['s_mae_sec']:.3f}"
            )
    else:
        lines.append("- 未提供 OBS 对比报告。")
    lines += ["", "## 3. 物理发现视角", ""]
    if geo_confirm:
        for item in report["stead_physical_claims"]:
            lines.append(f"- `{item['id']}`: {item['label']} - {item['note']}")
    else:
        lines.append("- 未提供 STEAD 物理发现确认报告。")
    lines += ["", "## 4. 结论模板", ""]
    for s in narrative["domain_shift_hypotheses"]:
        lines.append(f"- {s}")
    for s in narrative["physical_discovery_focus"]:
        lines.append(f"- {s}")
    (out_dir / "stead_obs_interpret_compare.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
