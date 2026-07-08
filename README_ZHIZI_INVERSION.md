# Zhizi Inversion Bridge（宏参数路线）

冻结 run20 拾取主干 + macro Physics Head（scale / contrast / Vs ratio）→ `vp0/vs0`，再经短步波形精修。定位：**更好的 FWI-lite 初值**。

## 结论

| 实验 | 结果 |
|------|------|
| Route A2，32 事件 | 反演后胜率 **≈0.94**；VpRMSE 智子 **0.924** vs 扰动 **0.982** |
| Route A2，64 事件 | 胜率 **≈0.875**；智子 **0.935** vs **0.977** |
| STEAD 真实几何精修，48 事件 | 胜率 **77.1%**；TT 均值 **3.08** vs 扰动 **11.22** |
| 宏头短训 | best Val Vp RMSE **≈0.277**（约 epoch 4） |

一次前向初值不必强于扰动；宏参数形变更常引导后续波形反演到更好的解。全链路说明见 [`README.md`](README.md)。

## 资产

- 拾取：`outputs/run20/20_wrongpeak_sharp/best.pt`
- 物理头：`outputs/zhizi_inversion_bridge_macro/best_physics_head.pt`
- 模式：`--head-mode macro`

## 复现

```bash
python train_zhizi_inversion.py \
  --head-mode macro --epochs 8 --n-train 96 --n-val 16 \
  --unrolled-weight 0.5 --unrolled-steps 5 \
  --vp-sup-weight 0.05 --lr 3e-3 \
  --output-dir outputs/zhizi_inversion_bridge_macro

python run_route_a2_waveform.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --n-test 32 --fwi-steps 60 --device cuda

python run_proof_suite.py --device cuda --max-events 48 --n-synth 32 \
  --output-dir outputs/proof_suite

bash scripts/reproduce_macro_route.sh
```

## 管线

```
波形 → 冻结智子特征（rho / envelope / kernel / picks）
     → macro 头 → vp0/vs0
     → 短步波形 / 走时精修 → m*
```

## 主要可视化

| 路径 | 内容 |
|------|------|
| `outputs/proof_suite/training_curves.png` | 训练曲线 |
| `outputs/proof_suite/stead_refine_scatter.png` | STEAD 精修对比 |
| `outputs/proof_suite/synth_full_compare_bars.png` | 合成全方法柱状图 |
| `outputs/proof_suite/example_paths.png` | 射线路径 |
| `outputs/proof_suite/latent/` | ρ(t)、envelope、P/S 面板 |
| `outputs/proof_suite/analysis/` | 几何条件与宏参诊断 |
