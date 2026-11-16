# Training lineage for `stead_c2f_imp_RSP` (paper “Ours”)

All stages: STEAD, `fine_len=6000`, seed **42**, `samples_per_epoch=200000` unless noted.
Wall times measured on the original training GPU (see histories under `outputs/stead_c2f_*`).

| Stage dir | Epochs | Role (approx.) | Resume from |
|-----------|--------|----------------|-------------|
| `stead_c2f_750_v5` | 80 | scratch + light aug | — |
| `stead_c2f_shift_v5` | 10 | large random shifts | `750_v5/best.pt` |
| `stead_c2f_rft_RSP` | 4 | EQT-style robust aug start | `shift_v5/last.pt` |
| `stead_c2f_lft_RSP` | 56 | continue robust / long FT | `rft_RSP/last.pt` |
| `stead_c2f_rob_RSP` | 20 | robust recipe | `lft_RSP/best.pt` |
| `stead_c2f_imp_RSP` | 12 | LF + reverb improve FT | `rob_RSP/last.pt` |

Cumulative epochs ≈ **182**. Paper Table kernel ablation uses a **separate** matched 20-epoch pair (`stead_c2f_pair_{huygens,free}`), not this long chain.
