# NMI Paper B — Cross-domain Huygens Protocol

**Venue:** Nature Machine Intelligence (primary)  
**Status:** planning lock for Paper B (protocol transfer)  
**Not this paper:** temporal-necessity / regime diagram (Paper A, separate venue)  
**Fork note:** This is **distinct** from the locked `NMI_DUAL_DOMAIN_ROADMAP.md` (STEAD×DANDI, small-n×readable θ). Only one manuscript should be the NMI primary; Paper B replaces that dual-domain thesis if we commit here.

---

## One-sentence claim (NMI-safe)

> A single **Huygens-style learning protocol** — secondary sources, a distance kernel, and scramble/occlusion gates — transfers across seismic time series, mmWave frame sequences, and static molecular graphs, with domain-appropriate distance (inter-frame lag vs Euclidean geometry); within a frame, spatial sites are treated as **simultaneous**.

**Forbidden in title/abstract:** universal wave backbone; cross-domain light-cone validation; SOTA-on-all-tasks.

---

## Protocol table (methods spine — Fig. 1)

| Slot | STEAD / seismic | RadHAR mmWave | QM9 molecules |
|------|-----------------|---------------|---------------|
| Task | phase picking / event readout (existing PNF stack) | 5-class activity (official split) | property regression (`gap`) |
| Secondary sources | station / channel embeds | range-bin × cue channels over frames | atoms \(Z_i\) |
| **Huygens axis** | **inter-frame / lag time** | **inter-frame (slow-time)** | none (static) → **geometric variant** |
| Same-frame space | simultaneous snapshot; geometry may modulate kernel or readout | same: points in one frame co-temporal | N/A (single graph) |
| Distance in kernel | time lag (+ optional station geometry) | frame lag (+ spatial pattern readout) | \(r_{ij}\) (Å), cutoff |
| Positive control | matched temporal baseline (PhaseNet / EQT / capacity-matched) | BiGRU / WaveGRU @ T=120 | SchNet-style CFConv (M1) |
| Null A — break structure | time shuffle / anti-causal window | frame shuffle / shorten history | geometry scramble (H1-scr) |
| Null B — locality | — | pattern occlusion (doppler/spatial) | neighbor-shell occlusion |
| Null C — wrong distance | — | — | feature-space distance (H1-feat) |
| Pass rule | true temporal structure ≻ null | acc competitive + null hurts | MAE ≈ M1; scr/shell/feat gates |

---

## Numbers ready for tables (fill manuscript; cite artifact paths)

### Domain — QM9 `gap` (full, public)

Artifact: `outputs/qm9/huygens_full_gap/REPORT.md`

| Model | test MAE | Gate |
|-------|----------|------|
| M0 (no geom) | 0.0295 | — |
| M1 (CFConv) | **0.0052** | baseline |
| H1 (geom Huygens v2) | **0.0063** | competitive (+22% vs M1) |
| H1-scr | 0.090 (Δ=+0.084) | **PASS** |
| H1-shell | s1=0.092 / s2=0.013 | **PASS** (1st shell dominates) |
| H1-feat | 0.0387 | **PASS** (geom ≫ feature dist) |

**U0 (negative control, SI only):** full U0 H1=2.69 vs M1=0.70; shell gate WEAK — shows protocol is not label-agnostic.

### Domain — RadHAR (official split)

Artifact: `outputs/mmwave/RADHAR_FINAL_REPORT.md`

| Model | Acc | Role |
|-------|-----|------|
| Wave+Pattern T=120 | **0.9789** | claim model (wave + spatial pattern head) |
| WaveGRU T=120 | 0.9795 | pure wave temporal |
| BiGRU T=120 | 0.9786 | matched temporal baseline |
| Pattern-only | ~0.91–0.95 | interpretability tradeoff |

**Wording for NMI:** Huygens/wave frontend operates on **frame sequences**; within-frame cloud is simultaneous; spatial **energy-distribution patterns** (e.g. Doppler width for jump) are a geometric readout, not intra-frame EM delay.

**Still recommended before submit:** one explicit **frame-shuffle / history-window** null on the claim model (aligns protocol table Null A).

### Domain — STEAD (protocol origin)

Artifact: `outputs/stead/temporal_null_b2v2/REPORT.md` (v2; first run failed due to wrong axis).

| Condition | mean P/S F1 |
|-----------|-------------|
| clean | **0.9797** |
| time_shuffle | 0.0006 |
| block_shuffle | 0.0086 |
| circular_shift | 0.0014 |
| time_reverse | 0.0545 |

Gate: **PASS**. Full-data PhaseNet/EQT leaderboard comparison is optional context, not the Paper B knife.

---

## Main figures (NMI layout)

| Fig | Content |
|-----|---------|
| **1** | Protocol schematic (table above as visual): sources → kernel distance → gather → head; three columns |
| **2** | Three-domain gate panel: STEAD time-null; RadHAR frame-null + pattern occlusion; QM9 scr/shell/feat |
| **3** | RadHAR: accuracy vs BiGRU + named pattern explanation (jack/jump) |
| **4** | QM9: MAE bars M0/M1/H1 + scramble Δ + shell bars (full `gap`) |
| **ED/SI** | U0 failure; RadHAR saturation note; EEG demoted or absent |

**Not in Paper B:** regime diagram (τ_process / Δt_frame vs gain) — that is Paper A.

---

## Manuscript skeleton (NMI Methods-heavy)

1. **Intro** — transferable inductive *protocol*, not universal PDE solver  
2. **Huygens protocol** — definitions; simultaneous frame; temporal vs geometric instantiation  
3. **Instantiations** — STEAD, RadHAR, QM9  
4. **Experiments** — gates + matched baselines  
5. **Discussion** — what transfers (slots), what does not (U0; no regime claim)  
6. **Limitations** — Paper A deferred; no claim of EM/neural intra-frame delay  

Working TeX can fork from `scripts/sn-article-nmi.tex` **or** new `scripts/nmi-paper-b-protocol.tex` to avoid colliding with dual-domain draft.

---

## Execution checklist (Paper B → NMI)

| # | Item | Priority |
|---|------|----------|
| B1 | Freeze claim sentence + forbidden phrases | **done** |
| B2 | STEAD time-shuffle null (v2) | **PASS** → `temporal_null_b2v2` |
| B3 | RadHAR frame-shuffle / history | **PASS** → `radhar_temporal_null_b3` |
| B4 | Fig.1–4 | **done** → `outputs/nmi_paper_b/figures/` |
| B5 | Methods + TeX draft | **done** → `docs/manuscripts/NMI_PAPER_B_METHODS.md`, `scripts/nmi-paper-b-protocol.tex` |
| B6 | SI: U0 negative; ED tables; forbidden-claim checklist | **done** in TeX Extended Data |
| B7 | Decide: archive dual-domain NMI draft as alt track / SI only | editorial |
| B8 | Compile PDF (`pdflatex` not on agent host — run locally) | user |

---

## B2 / B3 — run commands & pass criteria

### B3 RadHAR (eval-only)

```bash
cd /home/bob/TRELLIS/HNF
nvidia-smi
python -u tools/eval_radhar_temporal_null.py --device cuda \
  --checkpoint outputs/mmwave/radhar_wave_pattern_t120/best.pt \
  --output-dir outputs/mmwave/radhar_temporal_null_b3
```

| Gate | Rule |
|------|------|
| frame_shuffle | `clean_acc - shuffle_acc ≥ 0.05` |
| history | `clean ≥ T60 ≥ T30` and `clean - T30 ≥ 0.02` |

Artifacts: `outputs/mmwave/radhar_temporal_null_b3/{SUMMARY.json,REPORT.md}`

### B2 STEAD (eval-only) — **v2 required**

First run (`temporal_null_b2`) **FAIL’d because of a bug**: STEAD `x` is `(B,T,C)` but the script shuffled channels. **Re-run v2:**

```bash
cd /home/bob/TRELLIS/HNF
python -u tools/eval_stead_temporal_null.py --device cuda \
  --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt \
  --output-dir outputs/stead/temporal_null_b2v2
# smoke: add --max-batches 20
```

| Gate | Rule |
|------|------|
| time_shuffle | Δ(mean P/S F1) ≥ 0.10 **or** relative drop ≥ 25% |
| secondary | block_shuffle / circular_shift / reverse also reported |

Artifacts: `outputs/stead/temporal_null_b2v2/{SUMMARY.json,REPORT.md}`

**Note:** Within-frame spatial sites stay simultaneous; nulls only permute/reverse the **time** axis (`dim=1`).

---

## Honest NMI bar

Paper B is publishable at NMI **if** the contribution reads as a **clean, gated protocol** with three instantiations — not a scoreboard. Weakest link today: **STEAD/RadHAR temporal nulls not yet logged**. Run B2/B3 above before promising a submit date.
