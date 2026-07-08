# Huygens Neural Field (HNF)

基于**惠更斯原理**的神经场框架：用可学习复值核

\[
K(x_i,x_j)=\frac{1}{r^2+\varepsilon}\exp(-\gamma r^2)\exp(i\,\omega r)
\]

建模波传播 / 稀疏场重建，并延伸到 STEAD 震相拾取与 1D 速度结构反演。

当前仓库主线（终点）：

```
惠更斯核骨架
  → STEAD 拾取（run20 冻结）
  → 智子反演桥（macro Physics Head）
  → 短步波形/走时精修（Route A2）
  → 证明包（真实几何 STEAD + 全基线 + 中间变量可视化）
```

| 阶段 | 最佳资产 | 关键数字 |
|------|----------|----------|
| 拾取 | `outputs/run20/20_wrongpeak_sharp/best.pt` | det F1 **0.994** / P **0.959** / S **0.949**；约 **139k** 参数 |
| 反演初值 | `outputs/zhizi_inversion_bridge_macro/best_physics_head.pt` | 合成波形精修胜率 **93.8%**；STEAD 几何精修胜率 **77.1%** |

详细反演结论见 [`README_ZHIZI_INVERSION.md`](README_ZHIZI_INVERSION.md)。本文件覆盖 **HNF 从头到尾**。

---

## 目录

1. [环境与数据](#1-环境与数据)
2. [核心框架](#2-核心框架)
3. [合成 / 场重建 / SST](#3-合成--场重建--sst)
4. [STEAD 分类](#4-stead-分类)
5. [STEAD 震相拾取（run11→run20）](#5-stead-震相拾取run11run20)
6. [经典 1D 反演（inv01→inv06）](#6-经典-1d-反演inv01inv06)
7. [智子反演桥与证明包](#7-智子反演桥与证明包)
8. [可视化与解释](#8-可视化与解释)
9. [仓库地图](#9-仓库地图)
10. [测试与约定](#10-测试与约定)

---

## 1. 环境与数据

```bash
cd HNF
pip install -r requirements.txt
# torch>=2.0, numpy, matplotlib, pytest, tqdm, openpyxl
```

**STEAD**（约 90GB）放在 `STEAD/`（已 `.gitignore`，勿 push）。拾取数据集脚本默认读本地 CSV/波形。

**不推送**：`STEAD/`、`outputs/`、`*.pt`、HDF5。本地实验结果均在 `outputs/`。

建议 GPU ≥12GB。拾取训/评用 `seq_len=800`；智子桥推理常用 `infer_seq_len=600` 控显存。

---

## 2. 核心框架

包入口：`hnf/`（`__version__ = 2.0.0`）。

| 模块 | 角色 |
|------|------|
| `kernel.py` | `HuygensKernel`：因果/可学习 γ、ω、波速；feature / time / hybrid 距离；可选稀疏带 |
| `density.py` | `DensityNet`：空间密度 ρ(x)，Softplus 正定 |
| `layers.py` | `HuygensWaveLayer` / `HuygensAttention` / WaveBlock |
| `fmm.py` | 快多极 / 直接传播加速 |
| `deep.py` | `DeepHuygensKernel` 叠层核 |
| `bayesian.py` | 贝叶斯 HNF |
| `field.py` | `HuygensNeuralField`：稀疏观测 → 核回归 → 稠密场 |
| `multiscale.py` | 多尺度编码器 |
| `noise_cancel.py` | 惠更斯三步消噪支路 |
| `picking_model.py` | `STEADHNFPickingModel`：三分量次波源 + 时域 ρ(t) + 波场拾取头 |
| `demos.py` | 因果核 / 分类 / 长序列 / 贝叶斯 / FMM 基准 |

设计要点：

- 核是**物理启发算子**，不是黑盒注意力替身。
- 拾取分支中的 **`rho(t)`**、**kernel wave_speed** 是软先验 / 调制量，**不是**地壳真密度或绝对波速。
- 后续反演明确禁止把 ρ 硬映射成 `vp`（inv06 式硬映射已判负）。

快速 smoke：

```bash
python -c "from hnf import HuygensKernel, HuygensNeuralField, STEADHNFPickingModel; print('ok')"
pytest hnf/tests -q
```

---

## 3. 合成 / 场重建 / SST

### 2D 稀疏场重建

```bash
python example_2d_reconstruction.py
python example_2d_reconstruction.py --field-type vortex --n-obs 200 --train-steps 300
```

流程：`data_generator` 合成平面波/径向波/涡 → `HuygensNeuralField` 拟合 → `visualize` 出图。

### NOAA SST（稀疏→稠密）

```bash
python train_sst.py --epochs 300 --use-density --device cuda
python eval_sst.py   # 按脚本内路径改 checkpoint
```

相关：`hnf/sst_dataset.py`、`hnf/field.py`。

### 内建 demo

```python
from hnf import demo_causality, demo_fmm_benchmark
demo_causality()
```

---

## 4. STEAD 分类

地震 vs 噪声二分类（早期验证惠更斯注意力在波形上的可用性）：

```bash
python train_stead.py --device cuda
```

数据：`hnf/stead_dataset.py`。这是拾取工作之前的分类基线，不是当前主交付物。

---

## 5. STEAD 震相拾取（run11→run20）

目标：在**小参数量**下达到高 det / P / S，且**不借用** EQTransformer / PhaseNet。

统一训练入口：

```bash
python train_stead_picking.py --help
# 评估 / 分析 / 消融 / 实时对比
python eval_stead_picking.py --checkpoint outputs/run20/20_wrongpeak_sharp/best.pt
python analyze_stead_picking.py ...
python ablation_stead_picking.py ...
python explain_stead_picking.py --checkpoint outputs/run20/20_wrongpeak_sharp/best.pt
python benchmark_realtime_picking.py ...
```

各 run 脚本是**可复现实验编排**（内部调 `train_stead_picking.py`）：

| Run | 意图 | 关键取舍 |
|-----|------|----------|
| run11 | 增广 / 多尺度 DeepHuygens 消融 | 摸底座 |
| run12 | 稀疏核 / anchor 算力消融 | 仅当加速且不降指标才采纳 |
| run13 | 固定算力策略 + run11 式消融 | multi-scale 伤 det → 后续弃 |
| run14 | det≥0.99，推 P/S→0.95；深膨胀拾取头 | 全时间分辨率、det_guard |
| run15 | 冻结 backbone+det，只炼 P/S | 防联合微调毁掉 det |
| run16 | 两阶段：onset det → pick refine | 联合 refine 曾伤 P/S |
| run17 | 三步消噪 warmup + joint | 消噪助 det；joint 仍危险 |
| run18 | 冻结消噪+det，只炼 pick | 固定表征上的 pick refine |
| run19 | det 走消噪路径；P/S 走**原始波形**+消噪提示 | 避免消噪输入抬高 wrong_peak |
| **run20** | run19 上短程 sharp + wrong-peak 抑制 | **当前冻结拾取主干** |

```bash
python run20_stead_picking.py
# → outputs/run20/20_wrongpeak_sharp/best.pt
#    det_f1≈0.994, p_f1≈0.959, s_f1≈0.949, n_params≈139402
```

数据：`hnf/stead_picking_dataset.py`（含 `source_distance_km` / `source_depth_km` 几何字段，供真实反演）。

指标与失败模式：`hnf/picking_metrics.py`；常见残差是 S 的 **wrong_peak**（run20 针对性压制）。

**冻结规则（后续反演必须遵守）**：不解冻 det / P / S 头；只用特征与（可选）拾取作观测。

---

## 6. 经典 1D 反演（inv01→inv06）

在拾取上游打通之前，先有一套 **分层地球 + 射线走时 + 基线优化器** 的独立实验链：

| 脚本 | 内容 |
|------|------|
| `run_inv01_synth_1d.py` | 合成 vp/vs，Adam 反演 |
| `run_inv02_synth_1d.py` / `run_inv03_synth_1d.py` | 扩展合成场景 |
| `run_inv04_ambon.py` | Ambon 类真实几何尝试 |
| `run_inv05_pick_to_inversion.py` | 拾取 → 走时观测 串联 |
| `run_inv06_picking_prior.py` | run20 特征作 prior（后续确认硬 ρ→vp **不可用**） |
| `run_inv_compare_baselines.py` / `run_inv_full_compare.py` | GN / L-BFGS / Adam 等对比 |
| `run_inv_fwi_lite.py` | 声学 FWI-lite |

核心库：

- `hnf/inversion_1d.py` — `LayeredEarth1D`、走时
- `hnf/inversion_baselines.py` — Gauss–Newton / L-BFGS / Adam
- `hnf/acoustic_fwi_1d.py` — 直接波前向、可微精修、unrolled
- `hnf/synth_waveforms_1d.py` — 合成波形
- `hnf/ray_paths.py` — 1D 射线可视化
- `hnf/picking_prior.py` — 批量拾取封装

这些实验确立了：**oracle 走时求解器很强；智子的价值应定义为更好的波形反演初值，而不是取代 GN。**

---

## 7. 智子反演桥与证明包

### 思想

```
冻结 run20 波形 → 站台特征（rho、envelope、kernel、拾取）
  → Physics Head（推荐 macro：scale / contrast / Vs ratio）
  → vp0/vs0（相对标准分层的宏观形变，零初始化≈参考模型）
  → 短步波形 FWI-lite / 走时精修 → m*
```

### 训练与分项评估

```bash
# 推荐短训（勿盲目加 epoch；30ep 未超过 8ep best≈0.277 Val Vp RMSE）
python train_zhizi_inversion.py \
  --head-mode macro --epochs 8 --n-train 96 --n-val 16 \
  --unrolled-weight 0.5 --unrolled-steps 5 \
  --vp-sup-weight 0.05 --lr 3e-3 \
  --output-dir outputs/zhizi_inversion_bridge_macro

# 合成：智子初值 vs 扰动 → 波形反演（Route A2）
python run_route_a2_waveform.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --n-test 32 --fwi-steps 60 --device cuda

# STEAD 入口
python run_zhizi_inv05_real.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --obs-fallback --max-events 24 --device cuda

bash scripts/reproduce_macro_route.sh
```

### 证明包（几何校准 + 全对比 + 可视化）

```bash
python run_proof_suite.py --device cuda --max-events 48 --n-synth 32 \
  --output-dir outputs/proof_suite
```

| 关卡 | 结果 |
|------|------|
| STEAD 真实距离/深度精修 | **PASS** — 胜率 77.1%，TT 3.08 vs 扰动 11.22 |
| Route A2 波形初值 | **PASS** — 胜率 93.8%，VpRMSE 0.924 vs 0.982 |
| 走时 oracle（GN 等） | GN 仍最低绝对误差 — 预期内 |

产出：`outputs/proof_suite/proof_report.json`，以及训练曲线、STEAD 散点、基线柱状图、`latent/`（rho/envelope/拾取）、`analysis/`（几何条件与宏参诊断）。

### 已验证失败 / 弱路径（勿回退）

- 绝对速度头；inv06 硬 `rho→vp`
- Route A：智子作 GN 初值（合成未过）
- residual / waveform-aware / pairwise：不稳定或退化
- unrolled5：32 事件边缘、64 事件回落
- 名义固定几何（50 km / 10 km）真实关失败 → 必须用 CSV 几何

---

## 8. 可视化与解释

| 目的 | 入口 |
|------|------|
| 核矩阵 / 场重建 | `hnf/visualize.py`、`example_2d_reconstruction.py` |
| 拾取可解释性 | `explain_stead_picking.py`（波形、ρ、包络、P/S） |
| 反演速度剖面 / misfit | `hnf/inv_plot.py` |
| 射线路径 | `hnf/ray_paths.py`；证明包 `example_paths.png` |
| 训练与反演诊断 | `outputs/proof_suite/training_curves.png`、`latent/`、`analysis/` |

**中间变量读法**

| 量 | 正确理解 | 错误理解 |
|----|----------|----------|
| `rho(t)` | 潜空间权重，常随强能量（尤其 S）抬升 | 地壳密度 ρ |
| envelope | 复波场能量对齐相位 | 完整介质模型 |
| kernel_vp/vs | 无量纲软尺度 | 绝对波速真值 |
| macro 三参 | 相对参考分层的整体形变 | 无关层噪声 |

---

## 9. 仓库地图

```
HNF/
├── hnf/                      # 框架与物理模块
│   ├── kernel / density / layers / fmm / deep / bayesian / field
│   ├── picking_model / noise_cancel / multiscale
│   ├── stead_* / sst_dataset / data_generator
│   ├── inversion_* / acoustic_fwi_1d / ray_paths / picking_prior
│   ├── zhizi_*               # 反演桥、数据集、损失、物理头
│   └── tests/
├── train_stead_picking.py    # 拾取训练总入口
├── run11…run20_stead_picking.py
├── train_zhizi_inversion.py / run_route_a*.py / run_zhizi_*.py
├── run_inv*.py               # 经典反演链
├── run_proof_suite.py        # 端到端证明 + 可视化
├── example_2d_reconstruction.py / train_sst.py / train_stead.py
├── scripts/reproduce_macro_route.sh
├── README_ZHIZI_INVERSION.md  # 反演冻结文档（细）
├── STEAD/                    # 本地数据（忽略）
└── outputs/                  # 实验产物（忽略）
```

---

## 10. 测试与约定

```bash
pytest hnf/tests -q
```

覆盖核、FMM、拾取指标、消噪、反演、智子桥、射线、FWI 等。

**工程约定**

- 参数量保持可控（拾取主干 ~1.4e5）
- 不解冻已验证好的 det/P/S
- `rho` / kernel = 软先验，非硬物性
- 不以堆 epoch 替代结构或数据升级
- 真实反演必须使用事件几何，禁止默认名义距离糊弄过关

---

## 推荐阅读顺序

1. `hnf/kernel.py` + `example_2d_reconstruction.py` — 理解惠更斯场  
2. `hnf/picking_model.py` + `run20_stead_picking.py` — 理解拾取终点  
3. `hnf/zhizi_physics_head.py` + `train_zhizi_inversion.py` — 理解 macro 桥  
4. `run_proof_suite.py` 产物 — 理解「可用初值」主张与可视化证据  
5. [`README_ZHIZI_INVERSION.md`](README_ZHIZI_INVERSION.md) — 反演冻结细节与复现命令  
