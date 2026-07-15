# HNF tools

Day-to-day CLIs previously at the repo root. Run from **repo root**:

```bash
cd /path/to/HNF
python tools/train_stead_picking.py --help
python tools/eval_stead_picking.py --checkpoint outputs/run28/.../best.pt
python tools/train_zhizi_inversion.py --help
python tools/download_obs_chunks.py --help
```

| Group | Scripts |
|-------|---------|
| STEAD picking | `train_stead_picking`, `eval_stead_picking`, `analyze_stead_picking`, `explain_stead_picking`, `ablation_stead_picking`, `benchmark_realtime_picking` |
| Physics Decoder | `train_zhizi_inversion` |
| Classic / field | `train_stead`, `example_2d_reconstruction` |
| Domain II/III data | `train_eeg`, `eval_eeg`, `transfer_eeg`, `train_sst`, `eval_sst` |
| Downloads | `download_obs_chunks`, `download_eeg_adftd`, `download_raclette` |

Experiment / suite drivers live under `scripts/` (see `scripts/README.md`).
