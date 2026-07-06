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
- Training robustness: per-stage cosine schedule with a `min_lr` floor and warmup,
  gradient checkpointing, and optional **early stopping** that restores the
  best-val weights and carries them into the next stage.
- Unified inference (`generate` / `embed`) for any vintage, base or instruct.
- Reproduces the president-prediction consistency test (Table 2), the SFT loss
  curves (Figures 1-2, from `metrics.csv`), and the AlpacaEval length-controlled
  win-rate vs Qwen-1.5-1.8B-Chat (Figure 3, judged by the `alpaca_eval` package).
- Config-toggled Weights & Biases logging and Hugging Face Hub push (both default
  on, each degrades gracefully). After a run, loss logs + figures auto-save to
  `results/<name>/`; the sequential sweep publishes each vintage to GitHub + HF and
  can **email you** as each one finishes.

## Layout

```
src/chrono_instruct/   model.py  data.py  train.py  infer.py  eval.py  cli.py
                       tracking.py  hub.py  figures.py   # logging / HF push / plots
configs/               train.yaml  eval.yaml
scripts/               lambda_setup.sh  train_all_vintages.sh  make_vintage_config.py
                       publish_results.sh  notify_email.py  launch_local.sh  slurm_array.sbatch
tests/                 test_smoke.py        # tiny CPU end-to-end, no download
docs/                  running-guide.md  walkthrough/   # end-to-end guide + line-by-line walkthrough
```

Extras: `pip install -e '.[viz]'` for figures + W&B, `pip install -e '.[eval]'`
for the AlpacaEval (Figure 3) judge + the Qwen reference model.

## Quickstart

For a GPU box (Lambda H100), run `bash scripts/lambda_setup.sh` for the proven
environment (stable `cu126` torch, not the upstream nightly); see `docs/running-guide.md`
for the full end-to-end workflow. Locally:

```bash
pip install -e .
pytest -q                                   # smoke test: no GPU, no download

chrono inspect                              # see dataset `source` values + counts
chrono train  --config configs/train.yaml   # one-vintage curriculum SFT (GPU)
chrono infer  --repo runs/chrono-instruct-2020/final --mode generate --text "Explain inflation."
chrono eval   --repo manelalab/chrono-gpt-instruct-v1-20201231 --cutoff 2020   # Table 2
```

Figures and publishing (see `configs/eval.yaml` for the full Figure 3 pipeline):

```bash
chrono figure  --kind 1 --run runs/chrono-instruct-2020          # Fig 1: one run's loss curves
chrono figure  --kind 2 --runs runs/chrono-instruct-*            # Fig 2: val loss across vintages
chrono push    --repo runs/chrono-instruct-2020/final --to <user>/chrono-instruct-v1-20201231
```

Sequential vintage sweep on one GPU (trains, publishes, optionally emails per vintage):

```bash
bash scripts/train_all_vintages.sh 1999 2005 2010 2015 2024      # omit years already trained
```

## Attribution & licenses

`src/chrono_instruct/model.py` is adapted from `ChronoGPT_inference.py`
(manelalab, MIT) with `@torch.inference_mode()` removed so the model can train.
The base models are MIT; `ChronoInstruct-SFT` is ODC-BY and derives from three
upstream datasets with their own terms. Research/educational use.
