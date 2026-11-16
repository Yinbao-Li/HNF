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
