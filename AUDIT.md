# Phase 0 Audit — HNF TGRS manuscript (`paper/tgrs/main.tex`, “v14” claims)

**Date:** 2026-11-16  
**Repo:** `github.com/Yinbao-Li/HNF` @ `871b581d015aed1f50c844f67f39a839e2703fa9`  
**Manuscript audited:** `/home/bob/TRELLIS/HNF/paper/tgrs/main.tex` (synced with `paper/tgrs/tgrs_overleaf.zip`)  
**No manuscript edits in this phase.**

---

## Executive verdict

| Question | Answer |
|---|---|
| Does code for the **v14** architecture exist in-repo? | **YES** — `hnf/picking_coarse_to_fine.py` + `outputs/stead_c2f_imp_RSP/` (0.601 M, `propose_refine`) |
| Is that the **public README canonical** picker? | **NO** — README / `outputs/CURRENT.md` still point to **run28** (`seq_len=800`, ~192 k) |
| Do Abstract/tables numbers mostly trace to result files? | **YES** (almost all EXACT under `outputs/paper_stead_final_compare/`, `paper_tgrs/`, `ae/`, `ridgecrest/`, `mars/`) |
| Ready for TGRS submission after Phase 0 alone? | **NO** — Phase 1 reproducibility package missing; Phase 2 multi-seed / matched budget / modern baseline / CIs incomplete; scope/author formalities fail |
| Single most dangerous claim risk | Abstract “highest share of sub-0.1 s picks … on ETHZ, OBS **and Ridgecrest**” is **false for Ridgecrest** (and OBS S) vs Table `conditional.tex` |

**Primary paper checkpoint (not named “v14”):**  
`outputs/stead_c2f_imp_RSP/last.pt` — `n_params=600916`, `fine_len=6000`, `fusion=propose_refine`, `refine_max=4`, `refine_pool_sec=0.2`, `skip_bottleneck=8`, `propagation=huygens`, seed 42.  
(“v14” in this repo mainly labels Mars catalogue files, not the picker.)

---

## 0a. Code–manuscript discrepancy audit

### A. Two pickers in one repo

| | **Manuscript v14 (c2f)** | **Canonical run28 (public README)** |
|---|---|---|
| Module | `hnf/picking_coarse_to_fine.py` → `CoarseToFineHuygensPicker` | `hnf/picking_model.py` → `STEADHNFPickingModel` |
| Train entry | `tools/train_stead_c2f_picking.py` | `scripts/experiments/run28_stead_ms_fresnel_phys.py` |
| Input | Native **6000 @ 100 Hz** | Resampled **`seq_len=800`** / 60 s |
| Params | **600 916 (0.601 M)** | **192 493 (~192 k)** |
| Layers | **3 shared + 3 P + 3 S** | **2 shared + 2 branch** |
| Pick fusion | Eq. (4) gated: `ℓ = w' + σ(w') R tanh(r/R)`, `R=4`, pool ±0.2 s | `WaveFieldPickingHead`: envelope + **unbounded** CNN `delta` |
| Paper weight | `outputs/stead_c2f_imp_RSP/` | `outputs/run28/28_ms_fresnel_phys_*` |
| Status vs manuscript | **Matches architecture claims** when flags set | **Does not match** manuscript |

**Risk:** A reviewer who clones the GitHub repo and follows `README.md` will evaluate **run28**, not the paper model.

### B. Architectural claims → code status

| Manuscript claim | Code status | Evidence | Notes |
|---|---|---|---|
| 60 s ENZ @ 100 Hz → 6000×3 | **EXISTS** | `picking_coarse_to_fine.py` `fine_len=6000` | |
| Encoder: skip 6000×64 (bn 8), mid 1500×64, coarse 750×96 | **EXISTS (PARTIAL default)** | `MultiScaleEncoder`; bn via `--skip-bottleneck 8` | Class default `skip_bottleneck=0`; paper ckpt uses 8 |
| 3 shared + 3 P + 3 S HF layers | **EXISTS** | defaults `num_shared/branch_layers=3` | |
| Kernel Eq. (2)–(3), 15 s window, Fresnel obliquity | **EXISTS** | `principle=huygens_fresnel`, `local_window_sec=15` | Kernel formula in code uses lag `r` / `c`; matches spirit |
| “36 physical scalars” (4×9) | **MISMATCH** | `hnf/kernel.py`: per kernel `γ, ω, wave_speed, c_log_scale, obliquity_scale` | **5 learnable scalars/layer × 9 = 45** (always includes `c_log_scale`). Manuscript undercounts |
| Eq. (4) proposal-gated bounded fusion, `R=4` | **EXISTS (not CLI default)** | `fusion=="propose_refine"` lines ~360–378 | CLI/model default is `fusion="sum"`; paper ckpt enables it. Optional `gate_floor=ε`; paper uses ε=0 → matches Eq. (4) |
| Max-pool ±0.2 s on proposal | **EXISTS** | `refine_pool_sec=0.2` → `max_pool1d` | |
| Proposition 1 bound | **EXISTS (math + empiric)** | Proof in tex; empiric `cnn_role.json` | |
| Detection from coarse field energy | **EXISTS** | `ScalarDetHead` / `det_head` | |
| Sparse-band kernel apply | **EXISTS in c2f** | `sparse_band=True` default; `kernel.forward_apply` | Already on v14 path (Phase 2 item 5 “port” may be partly done; **re-benchmark** still needed) |

**Explicit checklist (user-requested):**

| Item | Exists? |
|---|---|
| Gated fusion (Eq. 4) | **YES** (c2f; enabled in `imp_RSP`) |
| Free-causal-kernel ablation | **YES** — `propagation=free` / `FreeKernelWaveBlock`; runs `stead_c2f_pair_{huygens,free}/`; table from controls JSON + `CONTROLS_RESULTS.md` |
| ETHZ evaluation | **YES** — `tools/eval_ethz_cross.py` (c2f); results under `paper_stead_final_compare/final/` |
| OBS evaluation | **YES** (c2f via same tool `--dataset obs`); **also** separate run28 OBS scripts (do not confuse) |
| Ridgecrest evaluation | **YES** — `tools/ridgecrest_pick.py`, `ridgecrest_eval_picks.py`, HypoDD under `outputs/ridgecrest/` |
| AE (AEsNet) evaluation | **YES** — `tools/train_ae_picking.py` / discovery; `outputs/ae/zs_seeds/` |
| Mars / InSight evaluation | **YES** — `tools/eval_mars.py`, `mars_fewshot.py`; `outputs/mars/fix/` |

### C. Training claims → code / run status

| Manuscript claim | Status | Evidence |
|---|---|---|
| STEAD train; BCE Gaussian `σ=0.35` s; late half ×0.4 | **EXISTS in paper recipe** | `imp_RSP` args: `label_sigma_sec=0.35`, `late_sigma_ratio=0.4` (CLI default late ratio is 1.0) |
| Det loss; noise-window penalty 0.15 | **EXISTS** | defaults + ckpt |
| Proposal aux 0.5; refine aux `σ=0.1`, weight 1.0 | **EXISTS in paper recipe** | `wave_sup_weight=0.5`, `refine_sup_weight=1.0` (CLI defaults 0) |
| Manual-reviewed weight 2 | **EXISTS in paper recipe** | `manual_weight=2.0` (CLI default 1) |
| AdamW, cosine, bf16 | **EXISTS in paper recipe** | `amp=true`, `amp_dtype=bf16` |
| 200 000 windows / epoch | **EXISTS** | `samples_per_epoch=200000` |
| “80 epochs from scratch with light augmentation, then FT stages…” | **PARTIAL / NEEDS PRECISE WORDING** | Lineage to `imp_RSP`: `750_v5` **80 ep** → `shift_v5` 10 → `rft_RSP` 4 → `lft_RSP` 56 → `rob_RSP` 20 → `imp_RSP` 12 ≈ **182 epochs** cumulative, seed **42 only**. “Light aug” for v5 is true relative to later EQT-style aug, but total budget ≫ “80+FT” narrative if read as short |
| Same seed multi-run for main / free ablation | **MISSING** | Only single seed-42 chain; pair ablation = **one run each** (caption admits this) |

### D. Evaluation / baseline claims → code status

| Claim area | Status | Notes |
|---|---|---|
| Official EQT / PN (SeisBench STEAD) | **EXISTS** | `final/official_fix/` |
| EQT-aug / PN-aug, 20 epochs, our aug | **EXISTS** | `outputs/sb_aug_eqt/`, `sb_aug_phasenet/`; **budget not matched** to ~182 ep Ours (disclosed in §Data, still a fairness issue) |
| Modern picker beyond EQT/PN | **MISSING / BLOCKED** | See **CORRECTION** below: PickBlue ≠ cross-domain OBS baseline (4C ID on same corpus); DiTingMotion ≠ picker and not in SeisBench |
| SNR curve @ −5 dB | **EXISTS** | `outputs/paper_tgrs/snr_curve_test.json` |
| Compute table (windows/s, MiB) | **EXISTS** | `outputs/paper_tgrs/compute.json` |
| Continuous-data throughput (not just windows/s) | **MISSING** | Table V is batch windows/s only |
| Cluster bootstrap CIs (AE by event/sensor) | **MISSING** | AE uses **5 eval seeds** + **paired trace bootstrap**; paper itself warns CIs may be optimistic |
| Headline table CIs / multi-seed mean±std | **MISSING** | Single-run point estimates |
| ρ-scrubbing (n=24) / waveform editing (n=50) | **CODE on run28 interpret path; NOT in manuscript** | `scripts/interpret/run_probing_suite.py` (n=24); waveform edits in `run_interpret_suite.py` (run28). **Not v14 claims in tex.** Phase 2 item 6 applies if you keep them as evidence elsewhere |

### E. Reproducibility surface (Phase 1 preview)

| Artifact | Status |
|---|---|
| Tagged release `tgrs-v14` | **MISSING** |
| One config per results table | **MISSING** (scattered `args.json` + launch scripts) |
| Pinned `requirements.txt` (exact versions) | **FAIL** — only lower bounds (`torch>=2.0`, …) |
| Top-level `LICENSE` | **MISSING** |
| `REPRODUCE.md` one-command / table | **MISSING** |
| Manuscript code URL + tag + commit | **FAIL** — still “will be released upon publication” |
| Public README points to paper model | **FAIL** — advertises run28 |

---

## 0b. Claim–evidence matrix

Rating key: **SUPPORTED** = number/claim matches a traced result file; **WEAK** = true but fragile (single run, unmatched budget, optimistic CI, over-worded); **UNSUPPORTED** = contradicts evidence or evidence absent.

### Abstract

| # | Claim (compressed) | Support | Table/Fig / file | Rating |
|---|---|---|---|---|
| A1 | Proposal–refine; CNN cannot invent unsupported picks | Prop 1 + 0 creates / 7000 wins | Tab. `cnn_role`; `.../interp/imp/cnn_role.json` | **SUPPORTED** |
| A2 | Memory truncation double dissociation; S longest memory | Fig. `interp`; win_*.json | **SUPPORTED** |
| A3 | In-domain P/S F1 0.988/0.975 @0.5 s | Full STEAD test | `ps_decode_last.json` / `eval_tol_last.log` | **SUPPORTED** |
| A4 | “Highest share of sub-0.1 s picks among its hits on ETHZ, OBS **and Ridgecrest**” | Tab. `conditional` | ETHZ/ETHZ-raw: Ours best. **OBS S: PN 0.553 > Ours 0.520. Ridgecrest P/S: PN 0.928/0.760 > Ours 0.914/0.752** | **UNSUPPORTED** as written (overclaim). Body text is more careful |
| A5 | AE lead glass F1 +4.3, AP +5.5 @3.2 µs; CI excludes 0 | Fig. `ae`; `ae/zs_seeds/stats.json` | **SUPPORTED** numbers; **WEAK** CI design (not cluster-by-event) |
| A6 | Limits: aug baselines higher recall ETHZ/OBS; worse low-SNR; slower; free kernel ≈ Huygens on quake data | Tabs. cross, kernel, compute; Fig. snr | **SUPPORTED** (kernel = single-run → **WEAK** for “equals”) |

### Introduction contributions

| # | Contribution | Support | Rating |
|---|---|---|---|
| C1 | Provable proposal–refine bound + empiric CNN role | §Method + Tabs. readout/cnn_role | **SUPPORTED** |
| C2 | Interventions beyond visualization (truncation / memory) | §interp + Fig. interp | **SUPPORTED** (not ρ-scrub/edit — those are not in this draft) |
| C3 | Multi-domain eval vs official + same-aug EQT/PN at 0.1 & 0.5 s | Tabs. stead/cross/cond/obs; Figs. snr/rc/ae | **SUPPORTED** results; **WEAK** vs 2026 TGRS bar (no fair modern **phase-picking** baseline yet; see CORRECTION) |
| C4 | Honest attribution: aug > physics prior for cross-region; free kernel | Tab. kernel; CONTROLS_RESULTS | **SUPPORTED** direction; **WEAK** (n=1 train run each) |
| — | Phrase “research agenda” / “expressive boundary” | Not present in current `main.tex` | N/A (Phase 3 rewrite item already satisfied in this draft) |

### Conclusion

| # | Claim | Rating |
|---|---|---|
| K1 | Decisions by explicit propagation path; CNN confined to sub-second timing | **SUPPORTED** (with Prop 1 caveats: bound is on probability, pool ±0.2 s) |
| K2 | Most precise onsets among compared models + strongest AE zero-shot | **WEAK** — “most precise” ≈ conditional timing; **false if read as universal** (Ridgecrest/OBS S conditional: PN ≥ Ours). AE lead **SUPPORTED** |
| K3 | Price: coverage, low-SNR P, compute; cross-region from aug not prior | **SUPPORTED** |

### Key numeric claims (traceability summary)

Almost all floats in Results/Abstract **EXACT-match** traced JSON (see `outputs/paper_stead_final_compare/PAPER_EVIDENCE.md` and `paper/tgrs/make_figures.py`). Notable nuances:

| Number | File | Note |
|---|---|---|
| Full-test S@0.5 = **0.975** vs Tab. stead S = **0.976** | full `ps_decode_last` vs 6 k `shift.json` | Different sets — both real; Abstract uses full-test |
| Kernel AE AP cells | Hardcoded in `make_figures.py` from `CONTROLS_RESULTS.md` | Traceable but not regenerated from a dedicated JSON in the figure script |
| Mars ~14 / 5 F1 pts; 7.7 vs 3.9 s | `outputs/mars/fix/RESULTS.md` | **SUPPORTED**; Discussion only |
| Template match 165 vs 262 | `outputs/ridgecrest/template_match/` | **SUPPORTED**; Discussion only |
| “36 physical scalars” | code = **45** | **UNSUPPORTED** count |

---

## 0c. TGRS compliance check (Phase 3 preview)

| Item | Status | Detail |
|---|---|---|
| **Scope framing** (sensing Earth / interpreting sensed waveforms on page 1) | **FAIL / RISK** | Intro opens on catalogs/early warning; EikoNet is cited in Related Work (`smith2021eikonet` — real TGRS paper) but **first page does not preempt “send to seismology journal”** |
| **Novelty vs SeisBench-era pickers** | **WEAK** | Method novelty clear; baselines stop at EQT/PN. Citing PickBlue’s **dataset** paper does not license treating PickBlue as a STEAD→OBS peer (see CORRECTION) |
| **Length ≤10 IEEE two-column pages** | **UNKNOWN / LIKELY FAIL** | No compiled PDF page count in tree; content density (6+ figs, 8 tables, Mars+QTM+HypoDD) matches user “~12 p” concern. Mars / template-matching / secondary tables still in main text |
| **Author names / affiliations / corresponding contact** | **FAIL** | `Author One/Two/Three`, `XX` placeholders in `main.tex` **and** Overleaf zip |
| **Bios + photos** | **MISSING** | Recommended by TGRS |
| **Funding / acknowledgments / conflict of interest** | **MISSING** | |
| **IEEE template** | **PARTIAL** | `IEEEtran` journal class OK; formal front/back matter incomplete |
| **Data & code availability** | **FAIL for submission hygiene** | “released upon publication”; no LICENSE; README ≠ paper model |
| **Citation hygiene** | **WEAK** | `refs.bib` has `TODO: verify` on `bornstein2024obs`, `aesnet2025`, `majstorovic2021interpreting`, `mqs2023`; `insight2020` uses “and others” |
| **Claim discipline (Abstract structure)** | **PARTIAL** | Abstract already mixes problem/method/results/**limits** (limits belong in Discussion per your Phase 3 brief) |
| **Statistics** | **FAIL vs Phase 2 bar** | No multi-seed train; no cluster bootstrap; no CIs on headline F1 tables |
| **Experimental fairness** | **DISCLOSED but inadequate** | §Data admits unmatched epochs & missing final aug on baselines — disclosure ≠ matched budget |
| **Low-SNR (−5 dB P F1 0.453 vs PN-aug 0.762)** | **DOCUMENTED failure; no fix** | Diagnosis present; no proposal-mechanism fix evaluated |

---

## Phase 1–2 blockers (do not start manuscript rewrite until addressed or claims downgraded)

### Phase 1 — P0 reproducibility (all currently **FAIL**)

1. Tag release with **c2f** code + `imp_RSP` (or frozen) checkpoint + per-table configs  
2. `LICENSE` + replace “upon publication” with URL/tag/commit  
3. Pin exact dependency versions; write & dry-run `REPRODUCE.md`  
4. Point README “canonical picker for this paper” at c2f **without deleting run28**

### Phase 2 — experiments / downgrades

| # | Item | Current | Options |
|---|---|---|---|
| 1 | Multi-seed (≥3, target 5) main + free-kernel | **n=1** | Run or downgrade all “equals / leads” to single-run |
| 2 | Matched budget EQT-aug/PN-aug | 20 vs ~182 ep | Retrain / budget-sensitivity **or** remove superiority language anywhere residual |
| 3 | Modern **phase-picking** baseline (fair protocol) | **Absent; prior PickBlue/DiTingMotion plan invalid** | Path A: land-trained modern picker with verifiable weights (e.g. SeisT dpk) under STEAD protocol; Path B: PickBlue **4C** only as OBS **ID reference** (not cross-domain); Path C: claim downgrade. **Reject DiTingMotion.** Details in CORRECTION |
| 4 | Low-SNR / proposal failure | Diagnosed only | Fix+eval **or** dedicated subsection + claim scope (Abstract already soft; keep) |
| 5 | Efficiency: sparse-band on v14 + same-GPU Table V + continuous throughput | sparse-band **already on**; continuous throughput **missing** | Re-bench + add stream metric |
| 6 | Scale ρ-scrub / waveform edit | run28 small-n; **not in tex** | Dataset-scale **or** omit / supplement as case studies if used |
| 7 | Cluster bootstrap AE; CIs on headline tables | **Missing** | Implement **or** remove/soften CI language |

---

## CORRECTION (2026-11-16) — Modern baselines: PickBlue / DiTingMotion

Phase 0 initially treated “PickBlue and/or DiTingMotion via SeisBench” as a drop-in Phase 2 item. That was **wrong on both names**. Corrected constraints below replace the Phase 2 row for item 3.

### PickBlue — not a STEAD→OBS cross-domain baseline

| Check | Finding |
|---|---|
| What SeisBench `PickBlue()` is | Thin wrapper: `PhaseNet.from_pretrained("obs")` or `EQTransformer.from_pretrained("obs")` (`seisbench/models/pickblue.py`) |
| Training data | Same SeisBench **OBS** corpus as this paper’s OBS eval (Bornstein et al.); splits are **by entire event within deployments** (train/dev/holdout) |
| Our OBS protocol | `tools/eval_ethz_cross.py`: `sbd.OBS(..., component_order="21Z")` on **`split=test`**, **3 channels only** (Z/1/2); hydrophone omitted |
| Official PickBlue weights | Local cache `obs.json.v1`: **`in_channels: 4`**, **`component_order: "Z12H"`**, hydrophone high-pass 0.5 Hz. Docstring: “three seismometer and one hydrophone” |
| 3C OBS-trained PickBlue | Paper discusses 3C OBS ablations; **SeisBench `PickBlue()` does not expose a separate 3C weight name** — only the 4C `obs` weights are what `from_pretrained("obs")` loads |
| Deployment/event overlap vs our table | Evaluating PickBlue on our OBS **test** split is evaluating it on **its own holdout** → **in-distribution** for PickBlue. Our / EQT-STEAD / PN-STEAD rows are **STEAD→OBS zero-shot**. Mixing PickBlue into the same “all models trained only on STEAD” table would be **protocol fraud** |

**Allowed use:** optional **OBS in-distribution ceiling / reference** (separate panel or clearly labelled column), stating (i) same SeisBench OBS event-holdout test, (ii) **4C Z12H** vs our **3C 21Z** input mismatch, (iii) not comparable as cross-domain transfer.  
**Disallowed use:** “modern cross-domain baseline on OBS” without the above disclosures.

### DiTingMotion — wrong task; no SeisBench one-liner

| Check | Finding |
|---|---|
| In local SeisBench 0.4.1? | **`hasattr(sbm, "DiTingMotion") == False`**. Model list has PickBlue, EQT, PhaseNet, GPD, … — no DiTingMotion |
| Task | Zhao et al. (2023) *Front. Earth Sci.*: **P-wave first-motion polarity** classifier (Up/Down/Unknown), **not** P/S onset picking. Needs an arrival time already |
| “Weights” people find | HuggingFace `HeXingChen/SeisPolarity-Model` (`DITINGMOTION_*.pth`): polarity, 128 samples, 2 channels — **unusable as a picker baseline** |
| Verdict | **Reject as Phase 2 picker baseline.** Do not “skip quietly”; replace with a real modern **phase picker** or downgrade the modernity claim |

### Replacement policy (must pick one path before Phase 2 closes)

**Path A — Prefer (fair cross-domain, land-trained modern picker):**  
Obtain a **phase-picking** model newer than EQT/PN with **documented weights**, trained on land/STEAD-like data (not OBS-only), and run it under the **same STEAD-only / same-aug protocol** on ETHZ, Ridgecrest, AE, and (3C) OBS. Candidates to **verify weight availability before committing**:

1. **SeisT** detection+picking checkpoints (senli1073/SeisT, DiTing-trained `seist_*_dpk`) — **not** a SeisBench one-liner; requires external download + adapter. Confirm license and exact ckpt hash.  
2. Newer SeisBench-integrated pickers **if** this env upgrades past 0.4.1 and `list_pretrained()` returns usable land weights (docs list SeisT / Skynet / EQCCTP / etc. on latest docs — **must re-check after upgrade**; do not assume).  
3. Published SeisBench weights on **other land corpora** (e.g. PhaseNet/EQT `instance` / `geofon`) as an extra row — architecture may not be “2024-new”, but protocol is clean and weights are one-call **if** present after upgrade.

**Path B — OBS-only ID reference (does not satisfy “modern cross-domain baseline” alone):**  
Add **PickBlue 4C** (`base="phasenet"` and/or `"eqtransformer"`) and optionally **OBSTransformer** `from_pretrained("obst2024")` (needs SeisBench ≥ ~0.7; also OBS-domain ID) as a **separate OBS ceiling panel**, never folded into STEAD-only transfer claims.

**Path C — Claim downgrade if A fails:**  
Keep Discussion “only EQT/PN”; strengthen to: no modern picker with redistributable picking weights could be run under matched protocol before submission. **Do not** invent a DiTingMotion picking number.

**Phase 2 item 3 revised status:** **BLOCKED pending Path A or C** (Path B is optional extra, not a substitute for a cross-domain modern picker).

---

## Recommended claim edits (for Phase 3 — **not applied yet**)

1. **Abstract A4:** Replace universal “highest share … ETHZ, OBS and Ridgecrest” with what Table `conditional` actually supports (e.g. ETHZ and OBS **P**; note PN on Ridgecrest / OBS S).  
2. **“36 physical scalars” → 45** (or redefine what is counted).  
3. **Training narrative:** state cumulative epoch/stage schedule and seed; do not imply a short 80-ep story if total is ~182.  
4. **Public/repo identity:** never let run28 be discoverable as “the paper model.”  
5. **Baselines wording:** never imply PickBlue is a STEAD→OBS transfer peer; if shown, label **4C in-distribution OBS reference**. Never cite DiTingMotion as a picking baseline.

---

## GO / NO-GO (Phase 0 only)

**NO-GO for submission.**  
Blocking: dual-model public confusion; no LICENSE/REPRODUCE/tag; unmatched budget & multi-seed gaps; **modern picker baseline still unresolved under corrected PickBlue/DiTingMotion constraints**; Abstract Ridgecrest overclaim; author placeholders; scope framing; page budget uncleared.

**Phase 0 complete (+ baseline correction).** Awaiting your approval before any manuscript edits or Phase 1 packaging.

---

## ERRATA (2026-11-16, post Phase-0 approval)

1. **Obliquity ablation:** the manuscript discusses the Fresnel obliquity factor in the kernel but reports **no quantitative obliquity on/off ablation table, no n, and no effect size**. Do not imply an obliquity ablation was run unless new evidence is added.
2. **Learned \(v_P/v_S\approx 0.9\):** already flagged in §interp as non-physical (`c` enters only through the obliquity factor). **Phase 3 rewrite must not describe learned `c` as velocity inversion / recoverable \(v_P,v_S\).** Keep the cautionary sentence; do not strengthen it into a physics claim.

## Phase 2 cost estimates (no runs authorized yet)

GPU reference: **NVIDIA RTX 3080 Ti** (same class as Table V). Epoch times from existing `history.json` / `run.log` on this machine. All items below need explicit approval before launch; anything >1 h must be re-confirmed.

| Item | What would be run | Est. GPU-hours (3080 Ti) | Primary output |
|------|-------------------|---------------------------|----------------|
| **2.1 Multi-seed main** | Replicate full c2f lineage (~182 ep) × (3–5 seeds). At ~0.25–0.46 h/ep (950–1660 s). | **~45–85 h per seed** → **~140–420 h** for 3–5 seeds | mean±std on STEAD / ETHZ / OBS headline F1 |
| **2.1 Multi-seed free-kernel** | Matched **20 ep** pair protocol (`pair_huygens`/`pair_free` style) × 3–5 seeds each | **~9.3 h × 2 × N_seeds** → **~56–93 h** for N=3–5 | Table kernel ablation with mean±std |
| **2.1 (cheaper alt.)** | Only multi-seed the **20 ep** recipe for both Huygens and free (skip full 182 ep chain) | **~56–93 h** total | Weaker “main model” claim; disclose short schedule |
| **2.2 Matched budget EQT-aug / PN-aug** | Extend from 20 ep to ~182 ep (or sensitivity at 20/40/80/182) | EQT ~4–6 min/ep → **~12–18 h** to 182 ep; PN ~2.6 min/ep → **~8–10 h**; sensitivity grid ~**1.5×** that | Fairness vs Table cross / Discussion budget note |
| **2.3 Path A SeisT** | After weight verify: adapter + STEAD-only eval (no long train if using published ckpt) | **~2–8 h** eval across ETHZ/OBS/Ridgecrest/AE (mostly forward) | Modern land picker row |
| **2.3 Path B PickBlue 4C** | Forward on OBS test 4C | **~1–3 h** | Separate OBS ID reference table |
| **2.4 Low-SNR proposal fix** | Design unknown; if FT 6–12 ep from `imp_RSP` + SNR curve | **~3–8 h** FT + **~2–4 h** SNR sweep | Fig. snr / claim scope |
| **2.5 Efficiency re-bench** | Table V + continuous throughput, same GPU all models | **~1–2 h** | Table V + stream metric |
| **2.6 ρ-scrub / waveform edit scale-up** | Dataset-level interventions | **CPU-heavy; ~4–20 h** wall depending on n | Supplement or drop |
| **2.7 Cluster bootstrap + CIs** | Resample existing logits / picks | **CPU ~2–10 h**; little GPU | AE CIs; headline table CIs |

## Phase 2 execution log (2026-11-16)

| Item | Status | Artifacts | Key finding vs manuscript |
|------|--------|-----------|---------------------------|
| Spec | DONE | `docs/TGRS_TARGET_SPEC.md` | Verbatim; no main.tex |
| **2.7** cluster CI | DONE | `results/phase2/2_7_ci/` | AE: Minusil F1@50 reverse excl0; headline dumps+CIs in; ETHZ/OBS dump abs F1 suspect vs paper |
| **2.5** controlled compute | DONE | `results/phase2/2_5_compute/` | Throughput ~−9–12% vs Table V; MiB match; continuous win/s added |
| **2.3B** PickBlue 4C | DONE | `results/phase2/2_3b_pickblue_obs/` | Separate ID table; n=6932 4C vs paper 11256 3C |
| **2.3A** SeisT | FAIL → **Path C** | `results/phase2/2_3a_seist_verify/PATH_C.md` | No download/eval; downgrade modern STEAD-only land-picker claim |
| **2.2** matched budget | **REVISED→20 ep** | existing `outputs/sb_aug_*` | 182-ep run stopped; align at 20 ep only |
| **2.1** 20ep×3seeds | NOT STARTED | — | — |
| **2.4** low-SNR design | NOT STARTED | — | design-only |
| **2.6** ρ-scrub scale | REJECTED | (this AUDIT) | Not a manuscript claim; supplement or drop — **do not scale** |


**Not started.** Awaiting per-item approval after Phase 1 acceptance.

## Phase 1 reproduce verification (local, 2026-11-16)

- **Table:** manuscript Table V (`compute`) — **Params** column only  
- **Command:** `python tools/reproduce_table_compute.py --checkpoint outputs/stead_c2f_imp_RSP/last.pt --expected paper/tgrs/expected/compute.json --device cpu`  
- **Result:** `n_params=600916`, `params_ok=True` (matches `Ours & 0.601` in `compute.tex` / `expected/compute.json`)  
- **Not re-run here:** fp32/bf16 traces/s and MiB (no CUDA in this environment; hardware-dependent)

## Phase 1 GPU re-verify (2026-11-16, RTX 3080 Ti)

- Params smoke: still `600916` / `params_ok=True`.
- Full `tools/paper_compute_bench.py` re-run → `outputs/paper_tgrs/compute_rerun_2026-11-16.json`.
- **Exact match:** params; peak MiB (ours 1517/1314; EQT 69/61; PN 86/52).
- **Throughput lower than frozen Table V** (same GPU class, load-dependent), e.g. ours fp32 **814** vs paper **898** tr/s (−9%); bf16 **1082** vs **1185** (−9%). Do not overwrite manuscript Table V with this rerun without a controlled re-bench protocol.
