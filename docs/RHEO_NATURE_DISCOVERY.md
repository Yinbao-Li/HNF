# Rheology — top-end discovery track (locked)

**Date locked:** 2026-08-06 (G4 update same day)  
**Primary data:** Leeds PS SAOS∩GPC n=9 ([DOI 10.5518/1689](https://doi.org/10.5518/1689))  
**Diversity data:** UMN PLA graft SAOS ([DOI 10.13020/y7as-3w53](https://doi.org/10.13020/y7as-3w53)) — **not** tube–MWD  

Artifacts: `outputs/rheo/spectrum_mwd_harden/`, `outputs/rheo/umn_bottlebrush/`,  
`outputs/rheo/tube_invert_g4/`, figures `rheo_nature_dig.*`, `rheo_nature_g4_invert.*`.

---

## Choice: nearest Nature gate = **G4** (done)

| Gate | Why not first |
|------|----------------|
| G1 mechanism uniqueness | Absolute tube-\(r\) partly structural; hard to uniquely prove α=3.4 at n=9 |
| G2 larger paired n | Blocked on public SAOS∩\(w(M)\) |
| G3 cross-chemistry | UMN lacks calibrated MWD |
| **G4 competitive / hybrid map** | **Nearest:** data + Elliott NN ceiling already in-repo |

---

## Locked discovery claim A — tube-aligned mode mass

Mode mass \(G_k/\sum G\) aligns with GPC under tube-style maps, **≫ shuffle-\(G_k\) null**  
(mean \(r=0.911\) vs \(0.565\), Δ≈0.35, 8/9 significant; α-robust).  
Publishable increment vs null / readable modes — not unique proof of α=3.4 alone.

## Locked discovery claim B — G4 map ladder (new)

Same Elliott Maxwell features (WLF→180°C, 81 τ):

| Method | mean MWD RMSE | gap closed (tube→NN) |
|--------|---------------:|---------------------:|
| Elliott NN ensemble | **0.046** | — |
| **LOO real MLP** (log G→MWD) | **0.090** | **~71%** |
| LOO real Ridge | 0.091 | ~70% |
| Synth-pretrained Ridge (α=3.4 heuristic) | 0.155 | ~27% |
| Tube deposit (zero-shot) | 0.195 | 0% |
| Zero-shot nonlinear tube invert | **0.269** | **worse** |

**Verdict:**
1. Nonlinear invert / MLP on the **simplified** \(\tau\!\propto\!M^{3.4}\) heuristic **cannot** replace Elliott’s full tube-sim pretraining (invert is worse than linear deposit).
2. **Nearest Nature increment:** readable \(\log G_k\) + a handful of real GPC labels (LOO) recovers most of the MWD accuracy gap and nearly matches NN on \(\log M_w\). Maxwell amplitudes act as **near-sufficient statistics** under light supervision.

**Method (not discovery):** PNF ≡ Prony NLS on PS and PLA grafts.

---

## Locked discovery claim C — G4b open tube forward (2026-08-06)

Elliott's ~8×10⁵ training pairs used a **state-of-the-art tube model (BoB-class; Das/Read)** that is **not** in the public Leeds dump (only NN weights).

**What we implemented instead (open, headless):** RepTate-grade **Rolie–Double–Poly LVE + dynamic dilution + Likhtman–McLeish CLF**, synth-pretrain Ridge/MLP, zero-shot on Leeds Maxwell features.

| Method (zero-shot) | mean MWD RMSE | gap closed (heur→NN) |
|--|--:|--:|
| Elliott NN | **0.046** | — |
| **RDP+CLF synth MLP** | **0.161** | **~15%** |
| Heuristic α=3.4 synth Ridge | 0.182 | 0% |
| RDP+CLF synth Ridge | 0.273 | worse |

**Verdict:** Better open physics helps a little; **does not** replace Elliott. Remaining zero-shot gap is BoB-scale fidelity + large sim corpus + deep net — not merely “use MLP instead of Ridge.”

Artifacts: `outputs/rheo/tube_g4b_rdp/`, `docs/figures/rheo/rheo_nature_g4b_rdp.*`.

---

## Locked ceilings


1. Absolute tube-\(r\) partly structural (SAOS-bin ≈0.86).
2. Descriptor-only LOO for Mw/Ð (few scalars) still fails — full \(\log G_k\) vector is what works.
3. LOO n=8 is fragile; PS1 remains hard for every method; Ð can blow up on some maps.
4. UMN ≠ tube–MWD \(n\); no Nbb–spectrum biomarker.
5. Not yet Nature alone without larger paired \(n\) or true tube-sim physics / external validation.

---

## Remaining Nature gates

| Gate | Next action |
|------|-------------|
| G2 | More paired SAOS∩\(w(M)\) to harden LOO claim beyond n=9 |
| G4c | Headless **BoB** (RepTate `bob2p5_lib_linux.so`) if linear-MWD→LVE API is usable without GUI |
| G3 | Second chemistry with calibrated MWD |

## Reproduce

```bash
PYTHONPATH=. python tools/run_rheo_leeds_tube_harden.py
PYTHONPATH=. python tools/run_rheo_umn_board.py
PYTHONPATH=. python tools/run_rheo_tube_invert.py
PYTHONPATH=. python tools/run_rheo_g4b_rdp.py
PYTHONPATH=. python tools/plot_rheo_nature_dig_figure.py
PYTHONPATH=. python tools/plot_rheo_tube_invert_figure.py
PYTHONPATH=. python tools/plot_rheo_g4b_figure.py
```
