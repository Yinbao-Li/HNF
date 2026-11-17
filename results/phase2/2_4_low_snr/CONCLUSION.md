# Phase B / 2.4 — 一页结论

**裁决：未修复**（预注册标准；不改 main.tex；阶段 C 不自动开始）

## Cond1（动手前）

冻结 `imp_RSP`，−5 dB P：`miss_no_proposal=0.681` ≫ `miss_with_proposal=0.036` → **proposal 主导**，放行动手。

## 试验（预注册后执行）

- 杠杆：MAD 归一化 proposal 分数 + soft BOOST（`MAD_FLOOR=0.05`, `Z_FLOOR=2`, `BOOST=logit(0.3)−logit(0.15)`）+ **proposal-head-only** 6 ep（backbone/refine 冻结）。
- 预算：约 **0.5 GPU·h**（3080 Ti），远低于 12 h 上限。

## 主指标 vs 手稿

| | −5 dB P F1 (tol 0.5, STEAD test) |
|--|--:|
| 手稿 / `imp_RSP` 重跑 | **0.4532** |
| 2.4 试验 ckpt | 0.4693 |
| Δ | **+0.016** |

预注册：Δ≥+0.10 才算修复成功；Δ&lt;+0.05 → **未修复**。本试验落在失败带。

增益来自 recall（0.294→0.307），precision 仍高（≈0.992）；**不是**假阳性买来的，但也远不够。

## 头部 / clean 回归

- Test clean P F1：ours−imp ≈ **−0.0005**（≤0.01，未伤头）
- Val clean 同量级；高 SNR（20 dB）无回退

## Proposition 1

| 套件 | 窗口 | created (P/S) |
|--|--:|--|
| STEAD val | 4000 | 0 / 0 |
| ETHZ dev | 2846 | **1** / 0 |
| 基线 `imp_RSP` 同协议 | 同左 | **0 / 0** |

**任一 created → 按设计稿回滚。** 已从 `hnf/picking_coarse_to_fine.py` 移除 SNR-aware proposal；**不发货** `outputs/stead_c2f_2_4_proposal_head/`。生产仍为 `outputs/stead_c2f_imp_RSP/last.pt`。

## 对照（−5 dB P F1）

| 模型 | P F1 | P prec / rec |
|--|--:|--|
| ours 试验 | 0.469 | 0.993 / 0.307 |
| imp_RSP | 0.453 | 0.992 / 0.294 |
| eqt_aug | 0.709 | 0.986 / 0.553 |
| pn_aug | 0.762 | 0.987 / 0.621 |

## 建议交给阶段 C（等确认）

专门小节 + claim 收窄：低 SNR 仍为 documented failure；Abstract/Limits 保持软表述；不宣称已修复。完整 P/R 曲线见 `RESULTS.json` / `snr_curve_test.json`。

git SHA at report: `f071773`（回滚后工作树另有未提交改动）。
