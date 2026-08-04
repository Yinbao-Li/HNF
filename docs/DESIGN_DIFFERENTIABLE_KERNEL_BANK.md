# Differentiable Kernel Bank — Design

Status: **scaffold** (`hnf/kernel_bank.py`). Not yet wired into run28 picking.

## Motivation

Current HNF: each `HuygensWaveBlock` owns **one** scalar kernel \((\gamma,\omega,c)\).
Multi-scale + P/S branches give ~10 independent kernels with **fixed topology**.

Goal: start from \(N\) identical Huygens kernels and let them **differentiate /
merge** into \(M\ll N\) meta-kernels that form a readable physical dictionary
(Meta-P / Meta-S / Meta-N / …), cutting effective kernel cost toward \(O(N_{\mathrm{time}}^2\cdot M)\)
(or sparse \(O(N_{\mathrm{time}} W M)\)) and enabling cross-domain role transfer.

## Parameter space

Each bank member \(k\) lives in
\[
\theta_k = (\gamma_k,\,\omega_k,\,c_k) \in \mathbb{R}^3_{+}
\]
(optional Fresnel \(\alpha_k\)). Effective values use the same softplus / \(c\)-scale
as `HuygensKernel`.

## Soft path (Phase-0, always on)

Fully differentiable; preferred training default.

| Mechanism | Form |
|-----------|------|
| Soft assignment | \(A\in\Delta^{N-1}\) from features (or uniform); entropy regularizer |
| Diversity | maximize mean pairwise \(\|\theta_i-\theta_j\|_2\) over alive pairs |
| Soft merge gate | \(g_{ij}=\sigma\!\big(\tau(\delta-\mathrm{EMA}\|\theta_i-\theta_j\|)\big)\); high \(g\) → shared forward mix |
| Forward | \(\sum_k \pi_k\,K_{\theta_k}@h\) (alive / top-\(M\) sparsified) |

## Discrete path (Phase-1, optional)

Matches the narrative split/merge; use sparingly (unstable if frequent).

| Event | Trigger | Action |
|-------|---------|--------|
| Split | Assigned batch stats are bimodal (e.g. Hartigan dip / 2-Gaussian BIC) on proxies (SNR, Δ, P/S cues) | Clone kernel; push children toward high-\((\gamma,\omega)\) and low-\((\gamma,\omega)\); split mass \(1/2\) |
| Merge | \(\mathrm{EMA}\|\theta_i-\theta_j\|<\delta\) **and** assignment overlap high | Soft-gate → EMA-blend params; freeze new meta for \(E_{\mathrm{ft}}\) epochs; **rollback** if val loss rises |

## Three-phase schedule

Fraction of total epochs \(p=e/E\):

1. **Differentiate** \(p\in[0,0.2)\): diversity ↑, assignment entropy ↑; merge rate \(=0\).
2. **Progressive merge** \(p\in[0.2,0.7)\): merge rate \(\propto e^{-\lambda p}\); soft gates active.
3. **Role lock** \(p\in[0.7,1]\): no merge; **role-anchor** pulls \(\theta_k\) toward running role centers; if val loss\(_k\) rises \(>20\%\), allow slow drift (reduced LR).

## Role anchoring

After mid-training, cluster alive \(\theta_k\) (or use Hungarian match to templates
P/S/N/M). Penalty:
\[
\mathcal{L}_{\mathrm{role}}=\sum_k w_k\,\|\theta_k-\theta_k^{\star}\|_2^2
\]
opposite-direction drift extra-penalized. Cross-domain: keep \(\theta^{\star}\) soft,
learn new mix weights first.

## Insertion into HNF

```
HuygensWaveBlock
  └── DifferentiableKernelBank (N members)
        └── ModuleList[HuygensKernel] × N   # reuse sparse_band / Fresnel
```

Factory: `build_huygens_kernel(..., bank_size=N)` or `HuygensWaveBlock(..., kernel_bank_size=N)`.

Picking: prefer **shared bank across shared encoder layers**, branch-specific mix
heads for P/S (same dictionary, different \(\pi\)).

## Complexity

| Mode | Cost vs single kernel |
|------|------------------------|
| Full soft mix | \(\times N\) |
| Top-\(M\) / alive mask | \(\times M\) |
| Soft param-mix (single effective \(\bar\theta\)) | \(\times 1\) (weaker) |

Default: top-\(M\) with \(M\le 4\) after mid-merge.

## Evaluation plan (after wire-up)

1. Toy: \(N=8\) on STEAD subset — visualize \(\theta_k(t)\) trajectories → expect P/S/N separation.
2. Compare run28 vs Bank-\(M\) on full STEAD: F1 / MAE / #alive kernels.
3. Transfer: freeze meta-\(\theta\), adapt mix on EEG / rheology.

## Non-goals (v0)

- Replacing multi-scale topology (Bank complements MS, does not replace it).
- Bayesian MC × Bank (compose later).
- Automatic discrete split every epoch (too noisy); start soft-only.
