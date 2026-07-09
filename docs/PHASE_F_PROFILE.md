# Phase F STEAD Pseudo-2D Profile

## Why Pseudo-2D

The current STEAD integration reliably exposes:

- `source_distance_km`
- `source_depth_km`
- waveform
- P/S catalog picks

but does not yet provide a stable survey-line geometry with station coordinates
for every trace in the current workflow. Therefore the first real-data imaging
product is a **pseudo-2D section along epicentral distance**.

This still gives a valid structural-imaging bridge:

1. waveform / picks
2. local 1D vp-vs model
3. distance-binned section
4. support + uncertainty maps

## Outputs

`run_phase_f_stead_profile.py` produces:

- `stead_profile_vp.png`
- `stead_profile_vs.png`
- `stead_profile_vpvs.png`
- `stead_profile_support.png`
- `stead_profile_vp_std.png`
- `stead_profile_vs_std.png`
- `stead_profile_trust_mask.png`
- `stead_profile_vp_masked.png`
- `stead_profile_vs_masked.png`
- `stead_profile_vpvs_masked.png`
- `report.json`

## Quality Control And Trust Mask

To avoid over-interpreting unstable real-data bins, Phase F now supports a
lightweight QC stage before bin stacking plus a final trust mask after section
assembly.

Event-level QC keeps only rows that satisfy:

- `pick_err_p <= qc_pick_err_p_max`
- `pick_err_s <= qc_pick_err_s_max`
- `refined_tt <= qc_refined_tt_max`

The default thresholds are intentionally moderate so the profile remains
readable while obvious outliers are removed:

- `qc_pick_err_p_max = 0.35 s`
- `qc_pick_err_s_max = 0.25 s`
- `qc_refined_tt_max = 6.0`

Bin-level trust is then defined from three practical constraints:

- enough support: `event_count >= qc_min_events_per_bin`
- limited Vp spread: `vp_std <= qc_max_vp_std`
- limited Vs spread: `vs_std <= qc_max_vs_std`

The resulting mask is rendered as `stead_profile_trust_mask.png`, and the
masked Vp / Vs / VpVs panels only show the trusted region for README-quality
visual interpretation.

## Shared Profile Sample Format

Each distance bin is exported as:

```json
{
  "x_km": 90.2,
  "depths_km": [0.0, 2.0, 6.0, 12.0, 20.0, 35.0],
  "vp_km_s": [...],
  "vs_km_s": [...],
  "vp_std": [...],
  "vs_std": [...],
  "coverage_score": 8.0,
  "event_count": 8,
  "station_count": 1
}
```

This is intentionally aligned with the synthetic `Phase E` structure so later
README figures and any future 3D voxelization can reuse the same contract.

## Interpretation Use

The most meaningful first-layer interpretations should use:

- `Vp` for broad velocity structure
- `Vs` for shear sensitivity
- `Vp/Vs` for possible fluid / fracture / weakness hints
- `support` to down-weight poorly constrained bins
- `std` to distinguish stable vs fragile anomalies
- `trust_mask` / `*_masked` panels as the default presentation view

## Next Upgrade Path

After the pseudo-2D section is stable, the next real-data upgrade is:

1. introduce explicit line projection from station/event coordinates
2. replace distance bins with true profile distance
3. stack multiple profiles into sparse 3D slices
