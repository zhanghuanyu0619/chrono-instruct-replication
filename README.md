# chrono-instruct

A clean, reproducible replication of the supervised fine-tuning (SFT) pipeline
from **"Instruction Tuning Chronologically Consistent Language Models"**
(He, Lv, Manela, Wu, 2025 — [SSRN 5348747](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5348747),
arXiv 2510.11677).

The authors release the ChronoGPT-Instruct **weights** and the **SFT data**, but
not the training code. This repo reconstructs the training pipeline and exposes a
small, unified API to load any ChronoGPT vintage for **text generation** or
**embedding extraction** — usable both to reproduce the paper and as
infrastructure for downstream, lookahead-bias-free prediction work.

## What it does

- Reproduces the 3-stage curriculum SFT (scratch → GPT-3 self-instruct → Tulu-3),
  masked cross-entropy on response tokens, started from a released `chrono-gpt-v1`
  base.
- Config-driven: one run = one vintage on one GPU. Sweep vintages by changing one
  line; fan out across GPUs or SLURM with the scripts in `scripts/`.
- Unified inference (`generate` / `embed`) for any vintage, base or instruct.
- Reproduces the president-prediction consistency test (Table 2). AlpacaEval
  win-rate (Figure 3) is stubbed pending an LLM-judge harness.

## Layout

```
src/chrono_instruct/   model.py  data.py  train.py  infer.py  eval.py  cli.py
configs/               train.yaml  eval.yaml
scripts/               lambda_setup.sh  launch_local.sh  slurm_array.sbatch
tests/                 test_smoke.py        # tiny CPU end-to-end, no download
docs/brainstorms/      requirements doc
```

## Quickstart

For a GPU box (Lambda H100), see `docs/env-setup.md` for the proven environment
(stable `cu126` torch, not the upstream nightly) or just run
`bash scripts/lambda_setup.sh`. Locally:

```bash
pip install -e .
pytest -q                                   # smoke test: no GPU, no download

chrono inspect                              # see dataset `source` values + counts
chrono train  --config configs/train.yaml   # one-vintage curriculum SFT (GPU)
chrono infer  --repo runs/chrono-instruct-2020/final --mode generate --text "Explain inflation."
chrono eval   --repo manelalab/chrono-gpt-instruct-v1-20201231 --cutoff 2020
```

## Attribution & licenses

`src/chrono_instruct/model.py` is adapted from `ChronoGPT_inference.py`
(manelalab, MIT) with `@torch.inference_mode()` removed so the model can train.
The base models are MIT; `ChronoInstruct-SFT` is ODC-BY and derives from three
upstream datasets with their own terms. Research/educational use.
