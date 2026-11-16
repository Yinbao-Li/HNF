# Phase 2.5 controlled compute vs manuscript Table V

GPU: NVIDIA GeForce RTX 3080 Ti; batch=64; warmup=20; runs=5

| Model | Prec | MS tr/s | Ctrl median | Rel% | MS MiB | Ctrl MiB | Cont win/s | ×realtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ours | fp32 | 898 | 804 | -10.4% | 1517 | 1517 | — | — |
| ours | bf16 | 1185 | 1082 | -8.7% | 1314 | 1312 | 682.3 | 20640 |
| EQTransformer | fp32 | 10535 | 9430 | -10.5% | 69 | 69 | — | — |
| EQTransformer | bf16 | 5184 | 4682 | -9.7% | 61 | 61 | 2413.9 | 73025 |
| PhaseNet | fp32 | 33499 | 29599 | -11.6% | 86 | 86 | — | — |
| PhaseNet | bf16 | 32123 | 28097 | -12.5% | 52 | 52 | 4257.7 | 128804 |
