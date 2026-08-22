# 整合稿大纲（方案一 · 顶刊常规结构）

**Status:** revised · 2026-08-22 · **4 Figures + 2 Tables = 6**（完整 EQT/PhaseNet 表进主文）  
**结构:** Abstract → **Introduction** → **Results** → **Discussion** → **Methods**（文末）  
**英文骨架稿:** `docs/manuscripts/UNIFIED_PROPAGATION_DRAFT.md`  
**Claim 纪律:** `docs/PROPAGATION_DYNAMICS_PAPER_CLAIMS.md`  
**地震底稿:** `docs/manuscripts/nc_coda_facies_path_domains.md` · `docs/NATURE_SUBMISSION.md`

---

## 0. 标题与一句话

**Working title (EN):**  
*Local propagation as a unified physical computational primitive: from Huygens seismic facies to cross-domain mechanism coordinates*

**一句话:**  
统一算子 \(P(X;G,\Theta,\alpha)\)；地震落在 Huygens 角点并给出五种 coda 传播模式与路径域；脑电/海温为其他 \(\Theta^\star\)；α 经反事实与回灌验证。

**不进摘要主句:** 临床 biomarker；全面 SOTA；多机制 mixture。

---

## 1. 正文章节顺序（顶刊常规）

| 顺序 | 章节 | 写什么 |
|--|--|--|
| — | **Abstract** | 问题 → 思路 → 三块结果 → 含义（≤150–200 words） |
| **1** | **Introduction** | 面临的问题、缺口、我们的思路与贡献（概念可在此用 Fig. 1 首现） |
| **2** | **Results** | 2.1 地震厚结果 → 2.2 跨域 \(\Theta^\star\) 与连续空间 → 2.3 α |
| **3** | **Discussion** | 合成含义、局限、前瞻；诚实不声称 |
| **4** | **Methods** | 落地协议：算子、facies ceiling、消融预算、CF/回灌、统计 |
| — | Data / Code availability | — |
| — | Extended Data / SI | 溢出图、表、协议细节 |

**方法放哪里讲清「思路」vs「操作」:**
- **Introduction:** 概念层思路（什么是统一原语、为何要 \(\Theta\)/α、预报只是探针）— 读者先懂故事。  
- **Methods（文末）:** 公式、超参、ceiling、bootstrap、冻结规则 — 可复现落地。  
- **Results 开口一句:** “在统一算子下……”即可，不在 Results 展开实现细节。

---

## 2. Introduction（问题 → 思路 → 贡献）

### 面临的问题
1. 跨系统时空动力学常被建成黑盒预报器，机制不可辨识。  
2. 地震学有成熟速度/\(Q\) 图，但 coda「如何响铃」常被压成标量 \(Q_c\)。  
3. 脑电/海温等是否共享同一传播语言，缺乏匹配预算下的可证伪比较。  
4. 即便拟合偏差场 α，也缺少反事实与冻结回灌，难以称为机制。

### 我们的思路（写清，但不取代 Methods）
- 递归局部传播：state → response → neighborhood \((A,\tau)\) → secondary sources。  
- 统一原语 \(\hat X=P(X;G,\Theta,\alpha)\)；wave / instantaneous / diffusion 等是 \(\Theta\) 坐标，而非互不相关模型。  
- 预报误差是 **probe**；科学对象是 \(\Theta^\star\) 与 α。  
- 地震：Huygens 角点 + 冻结五类 facies + 路径残差。  
- 跨域：同一协议选 \(\Theta^\star\)；\(\Theta(\lambda)\) 看 wave→damped-wave→diffusion。  
- α：相对域内 \(\Theta^\star\) 的局部偏差，门控验证。

### 贡献（三条，对应 Results 三块）
1. **地震厚度:** Huygens 型局部传播下的五种可解释 coda 模式 + 不可约路径域（+ 到时竞争力作旁证）。  
2. **跨域坐标:** EEG/SST 等为不同 \(\Theta^\star\)；连续动力学空间存在 damped-wave 中间相。  
3. **α 闭环:** 几何绑定（EEG）与冻结回灌（SST 满；EEG RDG 部分对比）。

---

## 3. Results 结构

### 3.1 Seismic depth（厚 · 主实证）

因果链（正文必须按此写）:
1. STEAD 上 wave/Huygens 为优选传播角点（机制门）。  
2. 冻结五类 facies（非 k-means）：impulsive fast-\(Q\) · emergent · multipath · slow coda · standard。  
3. SoCal \(\beta_{\mathrm{res}}\) 两极 + same-station；相对 \(Q_c/V_S/Q_S\) 不可约。  
4. Cascadia 复制（St Helens / Seattle 异号）。  
5. **一句旁证:** 同骨架 STEAD 到时具竞争力（细节进 Table 或 ED，勿写成全文 SOTA 文）。

**Claim:** *Huygens-type local propagation yields five frozen coda facies that encode path structure beyond scalar \(Q_c\).*

### 3.2 Cross-domain \(\Theta^\star\) and continuum

1. EEG→instantaneous；SST→graph_diffusion；与地震 wave 组成坐标三角。  
2. Unified 角点恢复 legacy；empirical instances parity。  
3. \(\Theta(\lambda)\): lag 单调塌缩 + mid-λ damped-wave。

**Claim:** *Regimes are coordinates of one family; wave and diffusion are linked by a continuous damped-wave path.*

### 3.3 α in mechanism space

1. EEG harden + LEMON（几何 CF）。  
2. SST RDG 冻结回灌闭包。  
3. EEG RDG Partial–Weak 对比（regime-dependent closure depth）。

**Claim:** *α is a geometry-bound deviation from domain \(\Theta^\star\), gated by counterfactuals and reinjection where full.*

---

## 4. Discussion（短而硬）

- 厚度如何合成：同一原语 → 地震最深 → 跨域坐标 → α 验证。  
- 惠更斯：地震角点正确可解释，**非**宇宙唯一。  
- 五模式：传播/响铃分类，不是地震时钟或前兆。  
- 局限：T2–T4 taxonomy；临床未闭环；OBS 迁移另表。  
- 前瞻：reaction / advection 等扩展位。

---

## 5. Methods（文末 · 落地操作）

分小节（与 Results 镜像，便于审稿人跳转）:
1. **Unified operator** — \(A(\Theta),\tau(\Theta),R,\alpha\)；角点与 \(\Theta(\lambda)\) 定义。  
2. **Matched-budget regime ablation** — 步数/α 类/response 冻结。  
3. **Seismic facies** — run28 picker；ceiling 表；冻结纪律。  
4. **Structure residual** — ridge、same-station、\(q_c\)、Berg/Lin 采样。  
5. **Cascadia / jackknife** — 复制与稳健性。  
6. **α gates** — CF、bootstrap、RDG mine→freeze（禁止回灌重拟合）。  
7. **STEAD timing benchmark** — tol、split、与 EQT/PhaseNet 协议。  
8. **Statistics** — 阈值、效应量、多重比较策略（若有）。

---

## 6. 图表预算 — **已锁定 4 Figures + 2 Tables = 6**

| ID | 类型 | 内容 |
|--|--|--|
| **Fig. 1** | Figure | 概念：统一原语 + \(\Theta\) + 三域落点 + 微型 \(\Theta(\lambda)\) |
| **Fig. 2** | Figure | 地震厚合并：五类 facies + SoCal \(\beta_{\mathrm{res}}\) + same-station + 包络栈 |
| **Fig. 3** | Figure | 跨域 ranking + \(\Theta(\lambda)\) continuum |
| **Fig. 4** | Figure | α：EEG CF/LEMON + SST RDG closure + EEG RDG 对比 |
| **Table 1** | Table | **完整 STEAD vs EQTransformer / PhaseNet**（full test；见 DRAFT） |
| **Table 2** | Table | 跨域 \(\Theta^\star\) + α 闭包等级总表 |

**原「不可约 + Cascadia」大图 → Extended Data**（保厚度，不占主文 6 格）。  
**Table 1 数字:** README canonical · `outputs/paper_stead_full_test_compare/` · 已写入 `UNIFIED_PROPAGATION_DRAFT.md`。

### 视觉规范

- 双栏主图；panel **a,b,c,d**；语义色板固定。  
- 地图发散残差；旧 facies 资产统一皮肤进 Fig. 2。  

### 图–文挂载

```
Introduction     → Fig. 1
Results 地震     → Fig. 2 + Table 1
Results 跨域     → Fig. 3 + Table 2
Results α        → Fig. 4 + Table 2
Methods          → Table 1 协议细节
ED               → Qc/Vs/QS 不可约、Cascadia、scaling、临床探索等
```

---

## 7. 下一步

1. ~~锁定 4+2；Table 1 完整进主文；英文骨架~~ → `UNIFIED_PROPAGATION_DRAFT.md`  
2. 扩写 Intro/Results 地震段（从 NC 底稿粘贴数字）。  
3. 拼版 Fig. 2；脚本 Fig. 3–4。  
4. LaTeX 化 Table 1/2。

**本文件:** `docs/manuscripts/UNIFIED_PROPAGATION_MANUSCRIPT_OUTLINE.md`
