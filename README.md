# chrono-instruct

A clean, reproducible replication of the supervised fine-tuning (SFT) pipeline
from **"Instruction Tuning Chronologically Consistent Language Models"**
(He, Lv, Manela, Wu, 2025 — [SSRN 5348747](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5348747),
[arXiv 2510.11677](https://arxiv.org/abs/2510.11677)).

The authors release the ChronoGPT-Instruct **weights** and the **SFT data**, but
not the training code. This repo reconstructs the training pipeline and exposes a
small, unified API to load any ChronoGPT vintage for **text generation** or
**embedding extraction** — usable both to reproduce the paper and as infrastructure
for downstream, lookahead-bias-free prediction work.

**Status.** All six headline vintages (τ ∈ {1999, 2005, 2010, 2015, 2020, 2024})
are fine-tuned and pushed to the Hub as `HZ0619/chrono-instruct-v1-{τ}1231`. See the
[**replication report**](results/replication-report/README.md) for results, and
[`docs/guides/`](docs/guides/) for the full end-to-end workflow.

> **Chronological consistency (the point).** Each vintage only ever saw text
> available before its cutoff τ, so it is safe to run *as of* that date — no
> look-ahead bias in any text-conditioned backtest.

## What it does

- **Reproduces the 3-stage curriculum SFT** (scratch → GPT-3 self-instruct → Tulu-3),
  masked cross-entropy on response tokens, from a released `chrono-gpt-v1` base.
  The data screen matches the paper (647,944 → **425,119** pairs).
- **Config-driven:** one run = one vintage on one GPU. Sweep all vintages with one
  script; fan out across GPUs or SLURM (`archive/slurm_array.sbatch`).
- **Training robustness:** per-stage cosine schedule with `min_lr` floor + warmup,
  gradient checkpointing, and optional early stopping that restores best-val weights
  and carries them into the next stage.
- **Unified inference** (`generate` / `embed`) for any vintage, base or instruct.
- **Full evaluation cycle:** SFT loss curves (Figures 1–2 + a combined sweep figure),
  the chronological-consistency tests (Tables 2–3), and AlpacaEval win-rate vs
  Qwen-1.5-1.8B-Chat (Figure 3) — with a sweep that runs every vintage and collects
  the results automatically.
- **Publishing:** loss logs + figures auto-save to `results/<name>/` and publish to
  GitHub; checkpoints push to the HF Hub; a script renders HF model cards. Optional
  Weights & Biases logging (live curves + run-finished emails). Each integration
  degrades gracefully.

## Layout

```
src/chrono_instruct/   model.py  data.py  train.py  infer.py  eval.py  cli.py
                       tracking.py  hub.py  figures.py   # logging / HF push / plots
configs/               train.yaml  eval.yaml
scripts/               lambda_setup.sh                       # env setup
                       train_all_vintages.sh  make_vintage_config.py  publish_results.sh
                       eval_all_vintages.sh  run_eval.py  aggregate_eval.py   # eval sweep
                       full_eval.py  plot_sweep_combined.py                   # uncapped val + figure
                       push_model_cards.py  model_card_template.md            # HF model cards
                       inference_demo.py  launch_local.sh
archive/               slurm_array.sbatch   # off-main-path SLURM launcher, kept for reference
results/               chrono-instruct-<τ>/ (metrics + figures)  combined/  replication-report/
tests/                 test_smoke.py        # tiny CPU end-to-end, no download
docs/                  guides/              # detailed step-by-step replication guides
```

Install extras as needed: `pip install -e '.[viz]'` (figures + W&B),
`pip install -e '.[eval]'` (AlpacaEval judge + Qwen reference).

## Quickstart

Full box setup and every step is in [`docs/guides/`](docs/guides/) — start with
[01-environment-setup](docs/guides/01-environment-setup.md). The essentials:

```bash
pip install -e .
pytest -q                                    # smoke test: no GPU, no download
chrono inspect                               # dataset `source` values + counts (screen -> 425,119)
```

**Train** (one vintage, or the whole sweep — see [02-training](docs/guides/02-training-the-models.md)):

```bash
chrono train --config configs/train.yaml                    # one vintage (GPU)
bash scripts/train_all_vintages.sh                          # all 6, trains + publishes each
```

**Evaluate** every vintage and collect results (defaults to the HF-published models —
see [04](docs/guides/04-table-2-president-consistency.md)–[06](docs/guides/06-figure-3-alpacaeval.md)):

```bash
bash scripts/eval_all_vintages.sh            # Tables 2-3, saved per vintage + aggregated for the report
ALPACA=1 bash scripts/eval_all_vintages.sh   # also Figure 3 (needs OPENAI_API_KEY)
```

**Generate / embed / figures / publish:**

```bash
chrono infer  --repo HZ0619/chrono-instruct-v1-20201231 --mode generate --text "Explain inflation."
chrono figure --kind 1 --run runs/chrono-instruct-2020    # Fig 1: one run's loss curves
python scripts/plot_sweep_combined.py                     # combined figure -> results/combined/
chrono push   --repo runs/chrono-instruct-2020/final --to <hf-user>/chrono-instruct-v1-20201231
python scripts/push_model_cards.py                        # render + upload HF model cards, make repos public
```

## Attribution & licenses

`src/chrono_instruct/model.py` is adapted from `ChronoGPT_inference.py`
(manelalab, MIT) with `@torch.inference_mode()` removed so the model can train.
The base models are MIT; `ChronoInstruct-SFT` is ODC-BY and derives from three
upstream datasets with their own terms. The fine-tuned vintages here are an
**independent replication by Huanyu Zhang**, not an official manelalab release.
Research/educational use.
