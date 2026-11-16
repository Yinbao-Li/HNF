# Path A → Path C (auto): SeisT STEAD-only protocol FAIL

Date: 2026-11-16

## Verification outcome

| Check | Result |
|-------|--------|
| Source URL reachable | PASS (`seist_{s,m,l}_dpk_diting.pth`) |
| License | PASS (MIT) |
| Content id (ETag) | Recorded; byte SHA256 deferred (no download) |
| **STEAD-only trained** | **FAIL** — published dpk weights are **DiTing**-trained |

Per Phase 2 order: *核实不通过 → 记录原因，自动转 Path C；不要换未核实的模型顶替.*

**No download. No adapter. No eval.**

## Path C — claim downgrade recommendations (Phase 3 wording)

1. **Do not** claim a “post-2022 / modern land picker baseline under STEAD-only protocol” in the main comparison tables.
2. Keep novelty on **inspectable propose–refine structure** (Proposition 1 + interventions), not on beating a SeisT row that does not exist under our protocol.
3. Optional discussion colour (not a table peer): note that SeisT public dpk checkpoints are DiTing-trained; a fair STEAD-matched SeisT would require a separate approved retrain (out of Phase 2 scope after Path C).
4. Path B PickBlue 4C remains the only modern OBS **in-distribution reference** (separate table); it does **not** fill the modern STEAD-trained land-picker gap.

## Status

Path A closed as **FAIL → Path C**. Awaiting no further SeisT action in Phase 2.
