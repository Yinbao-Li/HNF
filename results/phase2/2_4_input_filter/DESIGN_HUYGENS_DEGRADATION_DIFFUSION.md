# Huygens physical-degradation diffusion (design sketch)

Status: **new paper (HDD)** — minimal code landed (`hnf/huygens_degradation.py`, `tools/train_huygens_degradation_diffusion.py`); **not** part of TGRS.  
Motivation: replace Gaussian blur forward process with a **Huygens propagator** so each step is “how the wavefield evolves,” then learn the reverse for denoise / pick support.

## 1. Idea in one line

Standard DDPM: \(x_t = \sqrt{\bar\alpha_t}\,x_0 + \sigma_t\varepsilon\) (information destroyed by isotropic noise).  
**Ours:** \(x_t\) is obtained by **causal Huygens propagation + light stochasticity** from \(x_{t-1}\) (or from \(x_0\) in one multi-step schedule), so degradation is **physically structured**; reverse learns to invert that structure.

## 2. Forward process (physical degradation)

Let \(\mathcal{H}_{\Delta\tau,\theta}\) be one (or a stack of) frozen/lightly-learned `HuygensWaveBlock`-style operators on a wavefield representation \(h\) (real/imag or 3C waveform lifted to field).

**Markov form (preferred for “next from previous”):**

\[
h_t = \mathcal{H}_{\Delta\tau,\theta}(h_{t-1}) + \sigma_t\,\varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,I)
\]

with \(h_0 = E(x)\) (encode 3C window → field) and \(x_t = R(h_t)\) (readout back to waveform if needed).

**Schedule:** \(t=1..T\) increases **propagation depth / attenuation / obliquity scrambling**, not just \(\beta_t\) Gaussian. Practical knobs already in-repo: \(\gamma,\omega,c\), window, optional medium \(\rho\).

**Identity at \(t=0\):** \(h_0\) must reconstruct \(x\) well (small encoder–readout error) so “undo degradation” targets true onsets.

## 3. Reverse process (what we train)

Predict one of:

- \(\hat\varepsilon_t\) given \((h_t,t)\), or  
- \(\hat h_0\) (or \(\hat x_0\)) given \((h_t,t)\), or  
- residual field correction \(\Delta h\) inside a **Huygens-backed** reverse net.

Losses (pre-register before runs):

1. **Reconstruction:** \(\|\hat x_0 - x_0\|\) (waveform / envelope).  
2. **Pick consistency (optional joint):** freeze or FT c2f on \(\hat x_0\); det/P/S loss.  
3. **Onset fidelity:** penalize shift of energy near labeled P/S.  
4. **Do not** optimize only cosmetic SNR.

## 4. Why this might beat Gaussian diffusion for seismology

- Degradation **respects causality / travel-time geometry** → reverse may preserve arrivals better than isotropic blur.  
- Domain gap: different noise ≠ different physics; keep \(\sigma_t\varepsilon\) for sensor noise, keep \(\mathcal{H}\) for wave physics — **split what OBS/Mars change** (noise stats) from what is shared (propagation).  
- Ties narrative to existing Prop 1 / Huygens story (not a bolted-on U-Net denoise).

## 5. Hard problems (do not hide)

| Risk | Mitigation |
|------|------------|
| \(\mathcal{H}\) nearly irreversible for large \(T\) | Short \(T\) (e.g. 4–16); residual \(\sigma_t\); learn reverse with Huygens + CNN |
| Collapse to “just run Huygens then Gaussian” | Ablation: Gaussian-only vs \(\mathcal{H}\)+noise vs \(\mathcal{H}\)-only |
| Train–test mismatch if \(\theta\) trained per domain | Freeze \(\theta\) from c2f picker; only train reverse + \(\sigma\) schedule |
| Compute | Coarse field (750) for \(\mathcal{H}\), fine readout only at end |
| Overclaim “new physics” | Claim: structured prior for denoise; validate by pick F1 OOD |

## 6. Minimal experiment ladder (when approved)

1. **Smoke:** \(T=4\), freeze Huygens from `imp_RSP`, train reverse \(\hat x_0\) on STEAD; compare SNR/pick vs DeepDenoiser cascade & Gaussian DDPM small U-Net.  
2. **Joint:** feed \(\hat x_0\) into frozen c2f; then light FT.  
3. **OOD:** OBS / Mars zero-shot and small FT of noise scale only.  
4. Stop if −5 dB / OBS pick Δ trivial or onsets shift systematically.

## 7. Non-goals (this design)

- Not a drop-in Diffusers config change.  
- Not “Huygens replaces \(\varepsilon\) in the DDPM algebra” without redefining the forward kernel.  
- No `main.tex` edits until metrics exist.

## 8. Suggested next implementation slice

`hnf/huygens_degradation.py`: `forward_degrade(h0, t)`, `reverse_step(h_t, t; net)`.  
Reuse `HuygensWaveBlock` / c2f coarse path; waveform encode/decode as thin convs.  
Pilot script under `tools/train_huygens_degradation_diffusion.py` — **only after explicit budget OK**.
