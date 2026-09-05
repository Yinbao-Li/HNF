# Domain IV: Molecular Geometric Huygens (QM9 wedge)

**Status:** design + scaffold (not trained).  
**Principle:** keep Huygens as *secondary sources + geometric kernel + optional medium ρ*.  
**Do not claim** spacetime light-cone causality on QM9 (static graphs; no resolvable wave delay).

## Goal

Cross-domain transfer of HNF after RadHAR closure:

> Treat atoms as Huygens secondary sources; propagate with a distance kernel
> \(K(r_{ij})\); read out molecular properties. Show geometry dependence via
> scramble nulls and shell occlusion — same *protocol language* as STEAD/EEG/mmWave.

## Why QM9 (not lifetime / Domain III)

| Option | Verdict |
|--------|---------|
| Domain III real rheology | Too few real samples for a learning claim |
| Molecular lifetime / half-life | Labels far from local interactions; hard to attribute Huygens |
| **QM9 property regression** | Large public graph data; \(r_{ij}\) is native geometry |

## Data

- **QM9** (Ramakrishnan et al.): ~134k small molecules, DFT labels.  
- Start with a **subset** (e.g. 10k train / 1k val / 1k test) for smoke + gates.  
- Targets (phase-0): `U0` (or `mu`, `homo`, `lumo`, `gap`) — pick **one** primary to avoid claim dilution.  
- Recommended primary: **`gap`** or **`U0`** (standard in SchNet-style papers).

Download later via `torch_geometric.datasets.QM9` or official release; do not invent labels.

## Model interface (minimal)

Reuse `HuygensKernel` with **`distance_mode="feature"`** where `x` is **3D atomic coordinates** (Å), not latent features.

```
atoms Z_i, positions R_i
  → atom embed e_i = Emb(Z)                 # secondary-source amplitude
  → optional ρ_i = MediumNet(e_i, neighbors)
  → for L layers:
        H ← H + Proj( Re/Im( K(R) @ H_complex ) )   # geometric Huygens gather
  → pool (mean / sum / energy) → MLP → ŷ
```

### Module sketch (`hnf/qm9_huygens.py`)

| Symbol | Role |
|--------|------|
| `AtomSecondarySources` | `Z → embed_dim` (cf. `ComponentSecondarySources`) |
| `AtomicMediumDensity` | local ρ from embed (+ optional neighbor summary) |
| `GeometricHuygensBlock` | `HuygensKernel(distance_mode="feature", causal=False, principle="huygens")` on coords |
| `MoleculePoolHead` | mean + energy readout → property |

**Defaults:** `causal=False` (static molecule), `learnable_kernel_params=True` or fixed for ablation, cutoff radius ~5 Å (mask \(r>r_c\)).

### Explicit non-goals

- No slow-time / `distance_mode="time"` on QM9.  
- No RadHAR-style near→far radar causality.  
- No lifetime labels in phase-0.

## Experiment matrix (gates)

All runs: same split, same embed/layers budget, same optimizer budget.

| ID | Model | What it tests | Pass criterion |
|----|--------|---------------|----------------|
| **M0** | MLP on summed atom embeds (no geometry) | Need geometry at all? | Huygens ≪ M0 error |
| **M1** | Distance GNN / SchNet-style continuous filter (baseline) | Strong domain baseline | — |
| **H1** | Geometric Huygens (full) | Main method | Competitive with M1 (not necessarily beat) |
| **H1-scr** | H1 + **geometry scramble** (permute coords / jitter \(R\)) | Kernel uses real \(r_{ij}\)? | Error **worse** than H1 (rel. gap ≥ agreed %) |
| **H1-feat** | Same as H1 but distance from latent feats, not \(R\) | Geometric vs feature distance | Prefer H1 over H1-feat |
| **H1-fixedK** | Fixed γ,ω | Learnable kernel needed? | Report either way |
| **H1-shell** | Occlude 1st / 2nd neighbor shells at test | Interpretable locality | Shell-1 occlusion > random atom drop |

### Reporting (always)

- MAE / RMSE on primary target (and 1–2 secondary optional).  
- Param count.  
- Geometry scramble ΔMAE.  
- One figure: neighbor-shell occlusion bar chart (cf. RadHAR pattern occlusion).

### Claim language (allowed)

> On QM9, a Huygens-style geometric kernel on atomic coordinates reaches competitive property error versus a matched-budget continuous-filter baseline, and degrades under geometry scramble; neighbor-shell occlusion shows localized sensitivity.

### Claim language (forbidden until proven)

- “Validated wave propagation / light-cone in molecules.”  
- “Discovered new chemistry.”  
- “Beats all SOTA GNNs” (phase-0 is matched-budget, not SOTA hunt).

## Relation to other domains

| Domain | Huygens object | Distance | Claim type |
|--------|----------------|----------|------------|
| STEAD | stations / time | km + time | spacetime + facies |
| EEG | electrodes | scalp chord | often diffusion / spectral |
| RadHAR | range bins / cues | spatial energy patterns | **geometric patterns** (propagation FAIL) |
| **QM9** | atoms | Å | **geometric Huygens** |

Narrative arc: same inductive bias, domain-appropriate distance; RadHAR taught us not to force unresolved delays.

## Suggested execution order

1. Scaffold loader + `GeometricHuygensMolecule` forward smoke (CPU).  
2. Download QM9 subset; freeze split seed.  
3. Run M0 / M1 / H1 / H1-scr (primary target only).  
4. Shell occlusion panel + short REPORT.  
5. Only then expand targets or full QM9.

## Scaffold paths

- Design: this file  
- Code (stub): `hnf/qm9_huygens.py`  
- Train entry (later): `tools/train_qm9_huygens.py`
