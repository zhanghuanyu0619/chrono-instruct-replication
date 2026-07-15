# 06 — Figure 3: AlpacaEval length-controlled win-rate

Reproduce the AlpacaEval length-controlled (LC) win-rate of each vintage vs
**Qwen-1.5-1.8B-Chat**. Needs a trained or released model (see
[02-training-the-models.md](02-training-the-models.md)) and the `eval` extras.

## Judge setup

Step 3 below calls the canonical `alpaca_eval` package, whose judge needs an
**annotator key** (e.g. `OPENAI_API_KEY`). Install the extras once:
```bash
pip install -e '.[eval]'      # AlpacaEval judge + the Qwen reference model
export OPENAI_API_KEY=...      # annotator key for the alpaca_eval judge
```

## The 3-step pipeline

From `configs/eval.yaml`:

**1. Generate the model's outputs** (per vintage). `--repo` takes an HF id or a local
run dir; `--name` is recorded in the outputs.
```bash
chrono alpaca --backend chrono --repo /home/ubuntu/persist/runs/chrono-instruct-2020/final \
    --name chrono-2020 --out out/chrono-2020.json
```

**2. Generate the reference outputs** (Qwen — once, reused for every vintage):
```bash
chrono alpaca --backend hf --repo Qwen/Qwen1.5-1.8B-Chat --name qwen --out results/qwen/qwen.json
```

**3. Score the LC win-rate** of model vs reference:
```bash
chrono winrate --model out/chrono-2020.json --reference results/qwen/qwen.json
```
Both models decode **greedily** (matching the authors' released
`ChronoGPT_instruct.py`: temperature 0, argmax), so the win-rate reflects model
quality, not decoding strategy. Prints `LC win rate: <pct>%`.

> **Judge model.** AlpacaEval's default annotator calls OpenAI's **retired**
> `gpt-4-1106-preview` (now 404s). `eval.py` defaults instead to a live annotator
> (`weighted_alpaca_eval_gpt-4o-mini-2024-07-18`); override with
> `--annotator <cfg>` or `ALPACA_ANNOTATOR=<cfg>` (e.g.
> `weighted_alpaca_eval_gpt4_turbo_new` for the closest match to the paper's
> gpt-4-turbo-family judge, at higher cost). Expect win rates around the paper's
> Figure 3 range (**12.6–16.8%**), not higher — Qwen saw ~31× more pretraining data.
> To re-score *already-generated* outputs without any GPU work, use
> `scripts/score_alpaca.py`.

## Plot Figure 3

Collect `{year: winrate}` across vintages into a JSON, then:
```bash
chrono figure --kind 3 --results <winrate-json>
```
Default output `figure3.png`; override with `--out`.

## All vintages at once

To run Figure 3 across every vintage and collect win-rates automatically:

```bash
ALPACA=1 bash scripts/eval_all_vintages.sh
```

It generates the Qwen reference once, judges each vintage (needs `OPENAI_API_KEY`),
and writes the win-rates into `results/replication-report/eval_summary.md` alongside
Tables 2-3. Per-vintage detail lands in `results/chrono-instruct-<Y>/eval.json`.
