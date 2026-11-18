# Residual meta-filter probe (Branch A/B) — honest readout

Small MLP: \((\mathrm{SNR},\mathrm{distance},\mathrm{P–S},\mathrm{depth})\rightarrow(|r|,\Delta t_{\mathrm{refine}})\), frozen `imp_RSP`.

## Headline

| Branch | Gate (literal) | Honest meaning |
|--------|----------------|----------------|
| **A** meta→residual | PASS on \|r\| | Meta explains **amplitude** of refine (\|ref\|), STEAD R²≈0.24; ETHZ rank ρ≈0.18 |
| **A** meta→shift | **FAIL** | Signed timing correction almost **not** predictable from meta (R²≈0; variance explained ≈0.03%) |
| **B** leftover useful | PASS (literal) | **Mostly tautological**: leftover≈full shift because pred(shift)≈0; leftover inherits known useful signal |

**Bottom line:** 滤波网络能滤出的「干净规律」主要是 **\|refine\| ~ 难度元数据**；**不能**把真正改善到时的纠偏位移滤成更干净的可解释律。去掉 meta 之后，有用信息几乎原样留在 leftover 里——不是新发现，而是「纠偏不在 meta 里」。

## Numbers

- STEAD test: R²(\|ref\|)=**0.238**, R²(shift)=**−0.001**
- ETHZ: Spearman(\|ref\| pred, true)=**0.182** (p≪1e-19); shift pred fails
- \|leftover shift\| vs improve: STEAD ρ=−0.36, ETHZ ρ=−0.60 — but leftover≈true shift

## TGRS implication

- Optional: residual **magnitude** is partly a function of SNR / geometry (filterable phenomenology).
- Do **not** claim a filter recovered a clean new physical law behind the timing residual.
- Unnamed useful information remains the **corrective shift**, which is not meta-filterable here.

Artifacts: `filter_probe.json`, `meta_filter.pt`, `tools/phase2_residual_meta_filter.py`.
