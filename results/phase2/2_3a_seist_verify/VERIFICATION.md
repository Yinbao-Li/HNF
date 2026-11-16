# Path A — SeisT dpk weight verification (STOP for approval)

Date: 2026-11-16  
Status: **VERIFIED for availability / license; SHA256 of file bytes not computed (download deferred per instruction).**  
**Do not download / write adapter / evaluate until user confirms.**

## Recommended checkpoint (detection & phase picking)

| Field | Value |
|-------|--------|
| Model | `seist_m_dpk` (medium; paper’s mid-size picking backbone) |
| Train set (as published) | **DiTing** (not STEAD) |
| URL | https://raw.githubusercontent.com/senli1073/SeisT/main/pretrained/seist_m_dpk_diting.pth |
| HTTP | 200 |
| Content-Length | 1848817 bytes (~1.8 MiB) |
| GitHub raw ETag | `fc907f501426f603d9f257e25f603c8fd20711263bba2722098fc5143f696b47` (treat as content id; confirm with `sha256sum` after approved download) |
| Alternatives | `seist_s_dpk_diting.pth` (734599 B), `seist_l_dpk_diting.pth` (3140091 B) — same base URL dir |

## License

- Repo: https://github.com/senli1073/SeisT  
- `LICENSE`: **MIT** (Copyright 2023 Sen Li) — redistributable with notice.

## Protocol conflict to resolve before eval

User asked for evaluation under **STEAD-only same protocol**. Published `*_dpk_diting.pth` weights are **DiTing-trained**, not STEAD-trained.

Options after your confirmation:

1. **Zero-shot published DiTing weights** on ETHZ/OBS/Ridgecrest/AE under our eval protocol (fair as “modern published picker”, **not** STEAD-matched training).  
2. **Retrain SeisT on STEAD** under our aug/budget (true STEAD-only; much more GPU; needs separate approval).  
3. If neither acceptable → **Path C** claim downgrade (no modern land picker row).

## Verdict

- Weight **source URL**: OK (reachable).  
- **License**: OK (MIT).  
- **File hash**: provisional ETag recorded; byte SHA256 after you approve download.  
- **Pass/Fail for “STEAD-only trained modern baseline”**: **FAIL as published** (DiTing train set). Needs your call among options 1–3 above.
