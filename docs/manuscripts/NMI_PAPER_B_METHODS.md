# NMI Paper B — Methods (expanded)

**Manuscript:** `scripts/nmi-paper-b-protocol.tex`  
**Numbers:** `outputs/nmi_paper_b/NUMBERS.json`  
**Figures:** `outputs/nmi_paper_b/figures/`

## Huygens learning protocol

We define a **protocol**, not a single network: five slots that every domain must fill and test.

### Slots

1. **Secondary sources** \(s_i\): learnable embeddings of entities (waveform channels, range–cue bins, atoms).
2. **Distance** \(d_{ij}\): the quantity the kernel is allowed to see.
3. **Kernel** \(K(d_{ij})\): Huygens-style complex phase/envelope and/or a radial filter; optional medium density \(\rho_i\).
4. **Gather + head**: message aggregation and task readout.
5. **Gates**: interventions that break the intended \(d_{ij}\). Pass = clean performance ≫ gated performance (or gated error ≫ clean error).

### Temporal vs geometric distance (locked wording)

- **Within one frame / one time index:** spatial sites are a **simultaneous snapshot**. We do **not** treat intra-frame point-cloud or electrode geometry as resolvable propagation delay.
- **Across frames / samples:** when a sequence exists, the Huygens axis is **inter-frame (inter-sample) time**.
- **Static molecules:** there is no frame axis; the protocol’s geometric variant uses Euclidean \(r_{ij}\) (Å) with a cutoff.

### Domain fillings

| Slot | STEAD | RadHAR | QM9 (`gap`) |
|------|-------|--------|-------------|
| Sources | 3-C waveform embeds | range×cue channels over frames | atom-number embeds |
| \(d_{ij}\) | time lag along \(T\) | frame lag along slow-time | \(\|R_i-R_j\|_2\) |
| Kernel | Huygens–Fresnel wave blocks | Huygens wave frontend (+ spatial pattern readout) | Huygens phase/envelope × RBF amplitude (CFConv-matched) |
| Gate A | time shuffle / block shuffle / reverse | frame shuffle / history truncate | geometry scramble |
| Gate B | — | (optional) pattern occlusion | neighbor-shell occlusion |
| Gate C | — | — | feature-space distance |

### Pass criteria (pre-registered) — status

| Domain | Criterion | Result |
|--------|-----------|--------|
| STEAD | Δ mean P/S F1 ≥ 0.10 or ≥25% rel. on time shuffle | **PASS** (0.980 → 0.001) |
| RadHAR | frame shuffle ≥5 pp; history monotonic + ≥2 pp | **PASS** (0.979 → 0.532; T30=0.821) |
| QM9 | H1≈M1; scr/shell/feat gates | **PASS** on `gap`; **U0 SI negative** |

### Negative control (SI)

Full QM9 **U0**: M1 MAE 0.70 vs H1 2.69 — same protocol slots, label not solved. Use only as Extended Data.

### What we explicitly do not claim

- Universal spacetime light-cone validation across domains.
- Intra-frame EM delay in RadHAR.
- That every molecular target works.
- Temporal-necessity regime maps (Paper A).

## Implementation pointers

- STEAD nulls: `tools/eval_stead_temporal_null.py` → `outputs/stead/temporal_null_b2v2/`
- RadHAR nulls: `tools/eval_radhar_temporal_null.py` → `outputs/mmwave/radhar_temporal_null_b3/`
- QM9: `tools/train_qm9_huygens.py` → `outputs/qm9/huygens_full_{gap,u0}/`
- Figures: `tools/plot_nmi_paper_b_figures.py` (+ SI U0 panel in same folder)
- Compile: `cd scripts && pdflatex nmi-paper-b-protocol.tex`
