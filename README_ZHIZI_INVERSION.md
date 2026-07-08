# Zhizi Inversion Bridge（宏参数路线）

冻结 **run20 拾取主干** + **宏观 Physics Head**（`scale` / `contrast` / `Vs ratio`）→ 1D `vp/vs` 初值，再经短步波形 / 走时精修。主张是：**更好的 FWI-lite 初值**，不是取代走时 oracle（GN / L-BFGS）。

## 证明包结论（`run_proof_suite.py`）

| 关卡 | 结果 | 要点 |
|------|------|------|
| STEAD 真实几何精修 | **PASS** | n=48，智子 vs 扰动胜率 **77.1%**；TT misfit 均值 3.08 vs 11.22；Wilcoxon p≈3e-5 |
| 合成 Route A2（波形精修） | **PASS** | n=32，胜率 **93.8%**；VpRMSE 0.924 vs 扰动 0.982；p≈6e-7 |
| 合成走时 oracle | 参考 | GN ≈0.136 仍最低 — 预期内；智子主张是波形初值，不是 TT 闭式求解 |

补充：

- 一次前向初值 VpRMSE 仍可能弱于标准扰动（约 0.30 vs 0.18）；价值体现在**后续波形反演**。
- 30 epoch 长训未超过短训 best（Val Vp RMSE **0.277** @ ~epoch 4）。
- 名义几何（固定 50 km / 10 km）时真实关未过；接入 CSV `source_distance_km` / `source_depth_km` 后翻盘。

报告与图：`outputs/proof_suite/proof_report.json`。

## 管线

```
波形 U_obs
  → 冻结智子（run20）→ rho(t) / envelope / kernel 软先验 / P·S 拾取
  → 宏参数头 → (scale, contrast, vs_ratio) → vp0/vs0
  → 短步可微波形 / 走时精修 → m*
```

约束：`rho`、kernel 仅为软条件，**不是**硬物性；不解冻 det/P/S；不借用 EQTransformer / PhaseNet。

## 冻结资产

| 资产 | 路径 |
|------|------|
| 拾取主干 | `outputs/run20/20_wrongpeak_sharp/best.pt` |
| 物理头 | `outputs/zhizi_inversion_bridge_macro/best_physics_head.pt` |
| 头模式 | `--head-mode macro` |
| 训练曲线 | `outputs/zhizi_inversion_bridge_macro/history.json` |

核心代码：

- `hnf/zhizi_physics_head.py` — residual / **macro** 头  
- `hnf/zhizi_inversion_bridge.py` — 冻结 backbone + head  
- `hnf/acoustic_fwi_1d.py` — 直接波前向 / unrolled 精修  
- `hnf/stead_picking_dataset.py` — STEAD + 真实几何字段  
- `train_zhizi_inversion.py` — 桥训练  
- `run_route_a2_waveform.py` — 合成智子初值 vs 扰动 → 波形反演  
- `run_zhizi_inv05_real.py` — STEAD 单对比入口  
- `run_proof_suite.py` — 几何感知 STEAD + 全基线 + 中间变量可视化  

## 一键：证明包（推荐）

```bash
python run_proof_suite.py --device cuda --max-events 48 --n-synth 32 \
  --output-dir outputs/proof_suite
```

产出结构：

```
outputs/proof_suite/
  proof_report.json              # 总摘要 + verdict
  stead_geom_report.json         # 真实几何事件级明细
  synth_full_compare.json        # 全基线 + Wilcoxon
  training_curves.png            # Val Vp / loss / unrolled
  stead_refine_scatter.png       # 智子 vs 扰动精修散点
  synth_full_compare_bars.png    # 方法均值柱状图
  example_paths.png              # 射线路径示例
  latent/                        # rho / envelope / 拾取面板
    latent_case_*.png
    rho_vs_distance.png
  analysis/                      # 条件分析与宏参诊断
    stead_geom_conditioning.png
    synth_wave_delta_hist.png
    synth_method_box.png
    macro_latent_diagnostics.png
    analysis_summary.json
```

可选跳过：`--skip-stead` / `--skip-synth` / `--skip-latent`。

## 分项复现

```bash
# 合成 Route A2
python run_route_a2_waveform.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --n-test 32 --fwi-steps 60 --device cuda \
  --output-dir outputs/route_a2_waveform_macro_32

# STEAD（建议用证明包几何版；本脚本为专用入口）
python run_zhizi_inv05_real.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --obs-fallback --max-events 24 --device cuda \
  --output-dir outputs/zhizi_inv05_real_macro

bash scripts/reproduce_macro_route.sh
```

## 重训配方（短训）

```bash
python train_zhizi_inversion.py \
  --head-mode macro \
  --epochs 8 --n-train 96 --n-val 16 \
  --unrolled-weight 0.5 --unrolled-steps 5 \
  --vp-sup-weight 0.05 --lr 3e-3 \
  --output-dir outputs/zhizi_inversion_bridge_macro
```

推理侧桥用 `infer_seq_len=600` 控显存；拾取评估尽量全长 800。GPU ≈12GB 时注意 run20@800 的 OOM。

## 中间变量怎么读

| 量 | 含义 | 勿误解为 |
|----|------|----------|
| `rho(t)` | 潜空间“密度/权重”，常在强能量（尤其 S）段抬升 | 地壳真实密度 ρ |
| envelope | 波场复特征包络，对齐相位能量 | 完整介质模型 |
| kernel_vp / kernel_vs | 无量纲软先验尺度 | 绝对波速真值 |
| macro (scale, contrast, ratio) | 相对标准分层的宏观形变 | 层间逐层无关噪声 |

潜变量面板与宏参直方图见 `outputs/proof_suite/latent/`、`analysis/macro_latent_diagnostics.png`。

## 明确不做

- 不解冻 det / P / S  
- 不用 inv06 硬 `rho→vp`  
- 不把拾取核当作完整 FWI 内核  
- 不以堆 epoch 替代架构或数据升级  
- 不把 `STEAD/`、大 checkpoint 推入 git（已 ignore）

## 失败路径（勿回退）

绝对速度头、inv06 硬映射、Route A「智子作 GN 初值」、盲目 unrolled5 扩事件、30 epoch 长训宏观头 — 均已验证弱于当前宏参数短训 + Route A2。
