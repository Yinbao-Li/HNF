# Reproduce TGRS v14 (c2f) tables

**Paper model = coarse-to-fine Huygens picker (`hnf/picking_coarse_to_fine.py`), not run28.**

Tag: `tgrs-v14`  
Config: `configs/tgrs_v14/`  
Checkpoint (Release asset, not in git): `stead_c2f_imp_RSP_last.pt`  
SHA256: `b79743f8fbd524f25a763c30137de8b873585e6851bfb1d64edc912e9c0a60b1`

## 0. Clean environment

```bash
git clone https://github.com/Yinbao-Li/HNF.git
cd HNF
git checkout tgrs-v14
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Release checkpoint (requires `gh` or a browser):

```bash
mkdir -p outputs/stead_c2f_imp_RSP
# Option A: GitHub CLI
gh release download tgrs-v14 -p stead_c2f_imp_RSP_last.pt -D outputs/stead_c2f_imp_RSP/
mv outputs/stead_c2f_imp_RSP/stead_c2f_imp_RSP_last.pt outputs/stead_c2f_imp_RSP/last.pt
# Option B: browser — open the tgrs-v14 Release, download the asset, save as
#   outputs/stead_c2f_imp_RSP/last.pt
sha256sum outputs/stead_c2f_imp_RSP/last.pt
# expect: b79743f8fbd524f25a763c30137de8b873585e6851bfb1d64edc912e9c0a60b1
```

## 1. Smallest table (required smoke test): Table V — **Params** column

Manuscript Table V (`paper/tgrs/tables/compute.tex`) reports **Ours params = 0.601M**.
This command loads the Release checkpoint on **CPU or GPU** and checks the exact count **600916**.

```bash
python tools/reproduce_table_compute.py \
  --checkpoint outputs/stead_c2f_imp_RSP/last.pt \
  --expected paper/tgrs/expected/compute.json
```

**Pass criterion:** printed `params_ok=True` and `n_params=600916` (matches `Ours & 0.601` in `compute.tex`).

Throughput / MiB columns need a CUDA GPU comparable to the original RTX 3080 Ti and are **not** part of the required smoke test (hardware-dependent). To re-bench throughput when a GPU is available:

```bash
python tools/paper_compute_bench.py \
  --ours outputs/stead_c2f_imp_RSP/last.pt \
  --output outputs/paper_tgrs/compute.json
```

## 2. Other tables (optional; need STEAD / SeisBench data)

| Manuscript table | Command (after STEAD under `STEAD/` and/or SeisBench caches) |
|------------------|--------------------------------------------------------------|
| STEAD subset / full | `python tools/eval_c2f_ps_decode.py --checkpoint outputs/stead_c2f_imp_RSP/last.pt ...` |
| ETHZ / OBS | `python tools/eval_ethz_cross.py --dataset ethz\|obs --c2f ours=outputs/stead_c2f_imp_RSP/last.pt` |
| Read-out ablation | `python tools/eval_c2f_physics_suite.py --checkpoint outputs/stead_c2f_imp_RSP/last.pt` |

Exact CLI flags used for the paper JSONs are recorded next to those runs under `outputs/paper_stead_final_compare/` (developer machine; not required for the smoke test).

## 3. Expected artifacts in-repo

- `paper/tgrs/expected/compute.json` — frozen Table V numbers from the paper GPU run  
- `paper/tgrs/expected/compute.tex` — manuscript table body  
- `configs/tgrs_v14/imp_RSP.args.json` — final-stage training args  
