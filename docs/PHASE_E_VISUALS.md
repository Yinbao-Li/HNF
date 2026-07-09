# Phase E Visual Outputs

This note reserves the synthetic imaging figures that should later be surfaced in `README.md`.

## Synthetic Closed Loop

- `outputs/phase_e_smoke/figures/acquisition_geometry.png`
  - Synthetic source/receiver geometry along the profile.
- `outputs/phase_e_smoke/figures/true_vp_2d.png`
  - Ground-truth Vp section.
- `outputs/phase_e_smoke/figures/recovered_vp_2d.png`
  - Recovered Vp section from local 1D inversions assembled into a 2D image.
- `outputs/phase_e_smoke/figures/vp_error_2d.png`
  - Recovery error map (`recovered - true`) for Vp.
- `outputs/phase_e_smoke/figures/true_vs_2d.png`
  - Ground-truth Vs section.
- `outputs/phase_e_smoke/figures/recovered_vs_2d.png`
  - Recovered Vs section.
- `outputs/phase_e_smoke/figures/vs_error_2d.png`
  - Recovery error map for Vs.
- `outputs/phase_e_smoke/figures/ray_coverage_2d.png`
  - Illumination / ray-hit coverage map.
- `outputs/phase_e_smoke/figures/vp_uncertainty_2d.png`
  - Bootstrap uncertainty map for Vp.
- `outputs/phase_e_smoke/figures/vs_uncertainty_2d.png`
  - Bootstrap uncertainty map for Vs.

## Local Evidence Panels

- `outputs/phase_e_smoke/figures/local_profile_col00.png`
- `outputs/phase_e_smoke/figures/local_profile_col02.png`
- `outputs/phase_e_smoke/figures/local_profile_col04.png`

These provide column-wise evidence that the assembled 2D image is grounded in explicit 1D recoveries rather than pure interpolation.

## Intended README Placement

1. `Synthetic Closed Loop`
2. `Trust / Coverage Maps`
3. `Local 1D Evidence`
4. `Transition to Real-Data Structural Imaging`
