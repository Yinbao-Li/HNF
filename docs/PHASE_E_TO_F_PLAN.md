# Phase E to Phase F Bridge

This note defines how the synthetic imaging loop (`Phase E`) transitions into
real-data 2D structural imaging (`Phase F`).

## What Phase E Proves

`run_phase_e_synth_imaging.py` establishes a closed loop:

1. build a known quasi-2D Earth
2. generate sparse observations
3. recover local 1D models
4. assemble a 2D section
5. attach trust maps:
   - ray-hit coverage
   - bootstrap uncertainty

This is the minimal research backbone for "data input -> geologic image output".

## Shared Data Structure for Phase F

Each local inversion result should be exportable as:

```json
{
  "x_km": 12.5,
  "depths_km": [0.0, 2.0, 6.0, 12.0, 20.0, 35.0],
  "vp_km_s": [...],
  "vs_km_s": [...],
  "vp_std": [...],
  "vs_std": [...],
  "coverage_score": 0.42,
  "event_count": 18,
  "station_count": 6
}
```

This structure is enough to:

- build a 2D section
- build a 3D sparse voxel set later
- filter low-confidence regions
- preserve provenance for each profile sample

## Phase F Requirements

To move from synthetic to real-data sections, the next implementation should:

1. define one profile direction or station-line projection
2. gather local inversion outputs along that profile
3. project each local 1D model to profile distance
4. interpolate only where coverage is sufficient
5. render:
   - `Vp`
   - `Vs`
   - `Vp/Vs`
   - coverage / illumination
   - uncertainty

## README Figure Mapping

- `Phase E`: method validation on known truth
- `Phase F`: real-data structural image
- `Phase G`: scientific interpretation and anomaly evidence

## Combined Overview Artifact

For README or slide-level communication, `run_phase_ef_overview.py` combines the
formal `Phase E` report and the QC-filtered `Phase F` report into one overview
panel:

- synthetic summary / coverage
- trusted real-data `Vp/Vs` panel
- trust mask + support map
- key metrics and QC thresholds

Default generated output:

- `outputs/phase_ef_overview/phase_ef_overview.png`

Stable README asset:

- `docs/figures/phase_ef_overview.png`
