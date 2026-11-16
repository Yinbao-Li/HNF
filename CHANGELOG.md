# CHANGELOG

## 2026-11-16 — Phase 0 audit (no manuscript edits)

- **Added** `AUDIT.md`: code–manuscript discrepancy audit (run28 vs c2f/v14), claim–evidence matrix for Abstract/Intro/Conclusion, TGRS compliance preview, and Phase 1–2 blockers.
- **Reason:** User-requested Phase 0 deliverable before any manuscript changes; establish evidence-backed GO/NO-GO baseline for TGRS preparation.
- **Not changed:** `paper/tgrs/main.tex`, run28 code path, checkpoints, or experiment outputs.

## 2026-11-16 — Audit correction: PickBlue / DiTingMotion baselines

- **Updated** `AUDIT.md` with CORRECTION section and revised Phase 2 item 3.
- **Reason:** User correction — (1) PickBlue cannot be used as STEAD→OBS cross-domain baseline without deployment/event-overlap analysis and 3C vs 4C disclosure; at best an in-distribution OBS reference. (2) DiTingMotion has no SeisBench picking weights and is a polarity classifier, not a picker; must verify weight source or replace, not skip.
- **Verified:** SeisBench 0.4.1 `PickBlue()` → 4C `Z12H` `from_pretrained("obs")`; paper OBS eval is 3C `21Z` on same corpus test split; `DiTingMotion` absent from SeisBench; Zhao 2023 task = FMP polarity.
- **Not changed:** manuscript, code, or experiment outputs.

## 2026-11-16 — Phase 1 packaging (in progress)

- **Added** `LICENSE` (MIT), pinned `requirements.txt`, `REPRODUCE.md`, `configs/tgrs_v14/*`, `tools/reproduce_table_compute.py`, `paper/tgrs/expected/compute.{json,tex}`.
- **Reason:** Approved Phase 1 reproducibility package for TGRS v14 (c2f); checkpoint stays on Release, not git.
- **AUDIT:** appended ERRATA (obliquity ablation; non-physical vP/vS) and Phase 2 GPU-hour cost table (no training launched).

## 2026-11-16 — Phase 1 commit ready; push needs your GitHub token

- **Local commit:** `80a3041` — c2f code, configs, LICENSE, pinned requirements, REPRODUCE.md, README paper-model box.
- **Local tag:** `tgrs-v14` → same commit.
- **Reproduce smoke (local):** Table V Params → `600916` / `0.601` OK.
- **Push blocked in agent environment** (HTTPS auth). Run the commands in the Phase 1 report.

## 2026-11-16 — Phase 1 GPU smoke (Table V)

- Re-ran `reproduce_table_compute.py` on CUDA and `paper_compute_bench.py` → `outputs/paper_tgrs/compute_rerun_2026-11-16.json`.
- Params/MiB match expected; throughput ~9% below frozen paper numbers on same 3080 Ti (not used to edit manuscript).

## 2026-11-16 — tgrs-v14 Release checkpoint upload BLOCKED (auth)

- Local `outputs/stead_c2f_imp_RSP/last.pt` SHA256 verified OK:
  `b79743f8fbd524f25a763c30137de8b873585e6851bfb1d64edc912e9c0a60b1` (matches REPRODUCE.md).
- `gh` installed to `~/.local/bin/gh` (v2.62.0) but **not logged in** (`gh auth status`: not logged into any GitHub hosts).
- **Did not upload**; did not put checkpoint into git. Manual upload required by owner.

## 2026-11-16 — tgrs-v14 Release checkpoint upload COMPLETE

- Created GitHub Release on existing tag `tgrs-v14` (tag was present; prior “release page” had no Release/assets — API listed 0 releases).
- Asset: `stead_c2f_imp_RSP_last.pt` (7.1 MiB / 7411145 bytes).
- SHA256 (local + `gh release download` + public curl): `b79743f8fbd524f25a763c30137de8b873585e6851bfb1d64edc912e9c0a60b1`.
- Public URL verified: https://github.com/Yinbao-Li/HNF/releases/download/tgrs-v14/stead_c2f_imp_RSP_last.pt
- Note: first minutes after upload returned public 404 (CDN); later closed-loop OK. Wrong temporary asset name `last.pt` was removed.
