# Experiment plan (post-run28)

Documentation frame = README Parts I–IV.
**Next open item: Step 4 OBS.**

Primary ckpt: `outputs/run28/28_ms_fresnel_phys_20ep/best.pt`
Preferred Decoder: `outputs/physics_decoder_run28_macro/`
Outputs index: [`../outputs/CURRENT.md`](../outputs/CURRENT.md),
[`../outputs/SUMMARY.json`](../outputs/SUMMARY.json).

| Step | Status |
|------|--------|
| 0 Promote run28 metrics | DONE |
| 1 README Part I metrics | DONE (refreshed 2026-07-15) |
| 2 Interpret + probing | DONE first pass; fold figures ongoing |
| 3 Decoder upgrade + large-N | DONE; **claim = init** |
| **4 OBS multi-chunk** | **NEXT** |
| 5 Mining / reparam | pending |
| 6 EEG | pending |
| 7 Fluid | after EEG |

### Step 3 evidence (keep)

- A2 n=256 init: run28 macro **0.173** vs run20 macro 0.304 vs perturb 0.146
- A2 wave-win: run28 ~53–56% vs run20-macro **91%** → do not overclaim wave-win
- Proof n=500 STEAD refine win **69.6%**

### Step 4 checklist

- [ ] Multi-chunk OBS download
- [ ] Zero-shot HNF(run28) / EQT(STEAD) / PhaseNet(STEAD)
- [ ] Light-adapt / threshold sweep
- [ ] Protocol: PSN, sigmoid, drop incomplete 3C
