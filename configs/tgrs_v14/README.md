# TGRS v14 paper model configuration

Canonical inference / evaluation checkpoint (not stored in git):

- Release asset on tag `tgrs-v14`: `stead_c2f_imp_RSP_last.pt`
- Local path used while developing: `outputs/stead_c2f_imp_RSP/last.pt`
- SHA256 (`last.pt`): `b79743f8fbd524f25a763c30137de8b873585e6851bfb1d64edc912e9c0a60b1`
- Parameters: **600916** (0.601M)
- Architecture: native 6000 @ 100 Hz, 3 shared + 3 P + 3 S, `fusion=propose_refine`, `refine_max=4`, `refine_pool_sec=0.2`, `skip_bottleneck=8`

Files in this directory:

- `imp_RSP.args.json` — full CLI args of the final fine-tune stage (seed 42)
- `imp_RSP.SUMMARY.json` — validation summary + `n_params`
- `TRAINING_LINEAGE.md` — staged training chain that produced `last.pt`

This is **not** the legacy run28 picker (`seq_len=800`, ~192k params).
