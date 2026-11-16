# Quote only — 80-epoch trunk fork multi-seed (NOT approved / NOT run)

Context: alternative to full-chain multi-seed for the **main** c2f model
(~182 ep lineage). Phase 2 approved item is **20-ep pair × 3 seeds only**;
this quote answers the side question.

## Protocol imagined

1. Train **one** shared trunk to epoch 80 (single seed).
2. Fork **3 seeds** for epochs 81–182 only (re-seed optimizer/data order;
   optionally re-init late heads — disclose whichever).
3. Report mean±std on headline metrics from the three late forks.

## GPU-hour estimate (RTX 3080 Ti, ~1655 s/ep ≈ 0.46 h/ep)

| Stage | Compute |
|-------|---------|
| Shared trunk 0→80 | **~36.8 h** (once) |
| 3 seeds × 102 ep (81→182) | **~140.8 h** |
| **Total** | **~178 h** |
| vs full 3-seed 0→182 | ~252 h (3×84 h) — fork saves ~30% but |

## What variance this measures

- **Measures:** late-training / optimizer-noise variance after a fixed early
  trajectory (and whatever is re-seeded at the fork).
- **Does not measure:** sensitivity of the full recipe to initialization and
  early-epoch path; understates total seed variance if early dynamics matter.
- **Does not replace** the approved 20-ep Huygens↔free pair × 3 seeds (that
  isolates kernel factor under matched short budget).

## Decision

**Not run in Phase 2.** Main full-chain multi-seed remains unapproved.
