# Huygens Neural Field (HNF)

基于惠更斯原理的神经场：用可学习复值核建模波传播与稀疏重建，并落地到 STEAD 震相拾取与 1D 速度反演。

```
模型设计（Huygens Kernel + ρ + 波场层）
  → 框架与场重建 / SST
  → STEAD 分类验证
  → STEAD 震相拾取（→ run20 冻结）
  → 1D 走时 / FWI-lite 反演基线
  → 智子反演桥（macro Head）
  → 证明包（真实几何 + 全对比 + 可视化）
```

| 环节 | 结论资产 | 结论 |
|------|----------|------|
| 拾取 | `outputs/run20/20_wrongpeak_sharp/best.pt` | det F1 **0.994** / P **0.959** / S **0.949**；约 **139k** 参数 |
| 智子→反演 | `outputs/zhizi_inversion_bridge_macro/best_physics_head.pt` | 合成波形精修胜率 **93.8%**；STEAD 几何精修胜率 **77.1%** |

反演复现细节另见 [`README_ZHIZI_INVERSION.md`](README_ZHIZI_INVERSION.md)。

---

## 1. 环境

```bash
cd HNF
pip install -r requirements.txt
```

- 依赖：`torch>=2.0`、`numpy`、`matplotlib`、`pytest`、`tqdm`、`openpyxl`
- 数据：`STEAD/`（本地约 90GB，已 gitignore）
- 产物：`outputs/`（大文件与 checkpoint 不入库）
- 建议 GPU ≥12GB；拾取 `seq_len=800`；桥推理可用 `infer_seq_len=600`

```bash
python -c "from hnf import HuygensKernel, HuygensNeuralField, STEADHNFPickingModel; print('ok')"
pytest hnf/tests -q
```

---

## 2. 模型设计

惠更斯核（`hnf/kernel.py`）：

\[
K(x_i,x_j)=\frac{1}{r^2+\varepsilon}\exp(-\gamma r^2)\exp(i\,\omega r)
\]

| 设计 | 作用 |
|------|------|
| 复值传播相位 `exp(i ω r)` | 表达波动干涉与走时结构 |
| 高斯衰减 `exp(-γ r²)` | 局部次波权重 |
| 因果 / 波速 | 时间上有向传播 |
| 可学习 γ、ω、wave_speed | 数据适配软物理参数 |
| distance：feature / time / hybrid | 适配场坐标或波形时间轴 |

配套：

- **`DensityNet`（`density.py`）**：空间密度 ρ(x)，Softplus 保证正定，调制介质响应  
- **`HuygensWaveLayer` / `HuygensAttention`（`layers.py`）**：把核算入 Transformer 式堆叠  
- **`FastMultipoleMethod`（`fmm.py`）**：加速远程传播  
- **`DeepHuygensKernel`（`deep.py`）**、**`BayesianHNF`（`bayesian.py`）**：深层与不确定性扩展  

**约定**：ρ 与 kernel 波速是**软条件**，用于表征与调制；物理速度剖面由其后的反演头 / 优化器给出。

---

## 3. 框架能力：场重建与 SST

### 稀疏→稠密场（`HuygensNeuralField`）

```
观测 (x_obs, y) → K_obs = Re(K(obs,obs))
             → w = (K_obs + αI)^{-1} y
             → 场 = Re(K(target,obs)) @ w
```

```bash
python example_2d_reconstruction.py
python example_2d_reconstruction.py --field-type vortex --n-obs 200 --train-steps 300
```

可视化：`hnf/visualize.py`（观测分布、重建场对比）。演示：`demo_causality`、`demo_fmm_benchmark` 等（`hnf/demos.py`）。

### NOAA SST

```bash
python train_sst.py --epochs 300 --use-density --device cuda
python eval_sst.py
```

`hnf/sst_dataset.py` + 同上场重建管线，验证 HNF 在真实稀疏地球场上的可用性。

---

## 4. STEAD：分类 → 震相拾取

### 4.1 分类基线

```bash
python train_stead.py --device cuda
```

`HuygensAttention` 在 STEAD 上做地震 / 噪声分类，确认波形任务适配。

### 4.2 拾取模型（`STEADHNFPickingModel`）

三分量次波源耦合 → 时域 `rho(t)` 调制 → 惠更斯波场块（可选消噪支路）→ det / P / S 曲线头（包络残差拾取头）。

训练总入口：`train_stead_picking.py`。实验编排：`run11`…`run20_stead_picking.py`。

**迭代上沉淀的有效结论（已写入最终模型）：**

- 保持全时间分辨率与稳定 det，再推 P/S  
- 消噪路径服务 **det**；P/S 以**原始波形**为主、辅以消噪提示  
- 分阶段冻结 backbone / det，专门 refine 拾取头  
- 短程 low-LR sharp + wrong-peak 抑制（run20）

**冻结结果（run20）**

```text
outputs/run20/20_wrongpeak_sharp/best.pt
  det_f1 ≈ 0.994
  p_f1   ≈ 0.959
  s_f1   ≈ 0.949
  n_params ≈ 139402
```

```bash
python run20_stead_picking.py
python eval_stead_picking.py --checkpoint outputs/run20/20_wrongpeak_sharp/best.pt
python explain_stead_picking.py --checkpoint outputs/run20/20_wrongpeak_sharp/best.pt
```

**拾取可视化（已完成）**

| 图 | 含义 |
|----|------|
| `explain_stead_picking.py` 输出 | 波形、ρ(t)、包络、P/S 拾取曲线与真值 |
| `outputs/run20/.../threshold_sweep_curve.png` | 拾取阈值–指标曲线 |
| `outputs/run20/.../det_threshold_sweep_curve.png` | 检测阈值扫描 |

数据：`hnf/stead_picking_dataset.py`（含 `source_distance_km` / `source_depth_km`，供真实反演几何）。

---

## 5. 1D 反演基线

在智子桥之前，已完成分层模型上的标准反演栈，并与拾取串联验证。

| 组件 | 文件 |
|------|------|
| 分层地球 + P/S 走时 | `hnf/inversion_1d.py` |
| GN / L-BFGS / Adam | `hnf/inversion_baselines.py` |
| 声学 FWI-lite | `hnf/acoustic_fwi_1d.py` |
| 合成波形 | `hnf/synth_waveforms_1d.py` |
| 射线路径 | `hnf/ray_paths.py` |
| 剖面 / misfit 图 | `hnf/inv_plot.py` |

代表入口：

```bash
python run_inv01_synth_1d.py
python run_inv_full_compare.py
python run_inv_fwi_lite.py
python run_inv05_pick_to_inversion.py
```

**结论**：在合成走时 oracle 下，GN / L-BFGS 等经典求解器达到更低 Vp RMSE；波形 FWI-lite 可从初值继续下降。因此后续智子路线的目标定为——**提供更好的波形反演初值**，并与扰动初值做配对对比。

可视化示例：`outputs/inv_full_compare/full_comparison.png`（方法对比总览）。

---

## 6. 智子反演桥

### 设计

```
冻结 run20
  → 站台特征：rho(t)、envelope、kernel 软尺度、P/S
  → macro Physics Head：scale / contrast / Vs ratio
  → 相对标准分层的 vp0 / vs0（零初值 ≈ 参考模型）
  → 短步可微波形精修（Route A2）或走时精修
```

关键代码：`zhizi_physics_head.py`、`zhizi_inversion_bridge.py`、`zhizi_inversion_dataset.py`、`zhizi_inversion_loss.py`。

### 训练（已收敛配方）

短训即可；验证集 best Val Vp RMSE ≈ **0.277**（约 epoch 4）。

```bash
python train_zhizi_inversion.py \
  --head-mode macro --epochs 8 --n-train 96 --n-val 16 \
  --unrolled-weight 0.5 --unrolled-steps 5 \
  --vp-sup-weight 0.05 --lr 3e-3 \
  --output-dir outputs/zhizi_inversion_bridge_macro
```

检查点：`outputs/zhizi_inversion_bridge_macro/best_physics_head.pt`。

**训练曲线**（证明包已导出）：

`outputs/proof_suite/training_curves.png`  
— Val Vp RMSE / 总 loss / unrolled Vp MSE 随 epoch 变化；最佳落在前半程。

### 合成 Route A2（已结论）

```bash
python run_route_a2_waveform.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --n-test 32 --fwi-steps 60 --device cuda
```

| 设定 | 智子+wave VpRMSE | 扰动+wave | 智子更好比例 |
|------|------------------|-----------|--------------|
| 32 事件 | **0.924** | 0.982 | **93.8%** |
| 64 事件 | **0.935** | 0.977 | **87.5%** |

说明：一次前向的绝对初值不必强于扰动；宏参数形变更常把后续 FWI-lite 带到更好的解。

---

## 7. 证明包：真实几何 + 全对比 + 可视化

一键端到端：

```bash
python run_proof_suite.py --device cuda --max-events 48 --n-synth 32 \
  --output-dir outputs/proof_suite
```

### 7.1 定量结论

**STEAD（真实 `source_distance_km` / `source_depth_km`，n=48）**

| 指标 | 智子精修 | 扰动精修 |
|------|----------|----------|
| 平均 TT misfit | **3.08** | 11.22 |
| 胜率 | **77.1%** | — |
| Wilcoxon（近似） | p ≈ 3×10⁻⁵ | — |

**合成全对比（n=32，均值 Vp RMSE）**

| 方法 | Vp RMSE |
|------|---------|
| zhizi_wave | **0.924** |
| perturb_wave | 0.982 |
| gn_tt（走时 oracle） | 0.136 |
| lbfgs_tt | 0.201 |
| adam_tt | 1.597 |

智子相对扰动：Wilcoxon p ≈ 6×10⁻⁷。走时 oracle 绝对值更低，与第 5 节定位一致。

总报告：`outputs/proof_suite/proof_report.json`。

### 7.2 可视化图册

| 图 | 读法 |
|----|------|
| `training_curves.png` | 宏头短训收敛；best 约在 epoch 4 |
| `stead_refine_scatter.png` | 对角线下侧 = 智子精修 TT 更小 |
| `analysis/stead_geom_conditioning.png` | 距离 / 深度条件下的胜负差；远震胜率更高 |
| `synth_full_compare_bars.png` | 各方法平均 Vp RMSE |
| `analysis/synth_method_box.png` | 方法分布箱线图 |
| `analysis/synth_wave_delta_hist.png` | 智子−扰动 VpRMSE 差分（负值为主） |
| `example_paths.png` | True / 智子初值 / 智子+wave 射线路径 |
| `latent/latent_case_*.png` | 波形 ∥ **ρ(t)** ∥ envelope ∥ P·S 拾取（与能量段对齐） |
| `latent/rho_vs_distance.png` | 样本级 mean ρ 与震中距 |
| `analysis/macro_latent_diagnostics.png` | scale / contrast / Vs·Vp、kernel 与几何关系 |

**中间变量（已在 `latent/` 中验证可读性）**

| 量 | 观察到的含义 |
|----|----------------|
| `rho(t)` | 潜空间权重；强能量（尤其 S）段抬升，与相位同步 |
| envelope | 复波场包络，随 P/S 到时结构变化 |
| kernel_vp / vs | 无量纲软尺度，进入头网络作条件 |
| macro (scale, contrast, ratio) | 相对标准分层的整体形变控制量 |

复现脚本：`scripts/reproduce_macro_route.sh`。

---

## 8. 仓库结构

```
HNF/
├── hnf/                         # 核、层、场、拾取、反演、智子桥
│   ├── kernel.py density.py layers.py fmm.py field.py ...
│   ├── picking_model.py noise_cancel.py multiscale.py
│   ├── inversion_1d.py inversion_baselines.py acoustic_fwi_1d.py ray_paths.py
│   ├── zhizi_*.py
│   └── tests/
├── train_stead_picking.py       # 拾取训练
├── run11 … run20_stead_picking.py
├── train_zhizi_inversion.py
├── run_route_a2_waveform.py / run_zhizi_inv05_real.py / run_proof_suite.py
├── run_inv*.py                  # 1D 反演基线
├── example_2d_reconstruction.py / train_sst.py / train_stead.py
├── explain_stead_picking.py
├── scripts/reproduce_macro_route.sh
├── README_ZHIZI_INVERSION.md
└── outputs/                     # 本地实验与图（gitignore）
```

---

## 9. 端到端复现最短路径

```bash
# 拾取（若已有 run20 checkpoint 可跳过）
python run20_stead_picking.py

# 智子 macro 头（若已有 best_physics_head.pt 可跳过）
python train_zhizi_inversion.py --head-mode macro --epochs 8 ...

# 证明包：定量 + 全套图
python run_proof_suite.py --device cuda --max-events 48 --n-synth 32
```

看结论：打开 `outputs/proof_suite/proof_report.json`，并浏览同目录下训练曲线、STEAD 散点、基线柱状图与 `latent/` 面板。
