# 04 — Table 2: U.S. president prediction consistency

Reproduce the lookahead-bias consistency check on U.S. presidents. Needs a trained
or released model (see [02-training-the-models.md](02-training-the-models.md)).

## Run it

```bash
chrono eval --repo manelalab/chrono-gpt-instruct-v1-20201231 --cutoff 2020
```
`--repo` takes an HF repo id **or a local run dir** (e.g.
`/home/ubuntu/persist/runs/chrono-instruct-2020/final`); eval, inference, and
AlpacaEval generation all accept either. `--cutoff` is the model's knowledge-cutoff
year. The command prints the president test (Table 2) *and* the major-events test
(Table 3, see [05](05-table-3-major-events.md)).

## The test (`eval.py` `president_test`)

For each presidential transition, the prompt lists the **three prior presidents** in
chronological order plus the target's **actual inauguration year**, then greedy-decodes
**2 tokens** (`top_k=1`, deterministic, as in the paper) and checks whether the target's
name appears:

```
U.S. Presidents in chronological order:
Took office in 2001: President George W. Bush
Took office in 2009: President Barack Obama
Took office in 2017: President Donald Trump
Took office in 2021: President
```

The query year is the target's *real* inauguration year, not previous+4 — two-term
presidents make the gap 8 years, so arithmetic would mis-date the blank.

**Consistency logic.** A chronologically consistent model should be **CORRECT** for
transitions on or before its cutoff and **WRONG** for transitions after it. Each result
row carries a `past_cutoff` flag (`target_year > cutoff_year`); the honest pattern is
`correct` for pre-cutoff rows and *not* `correct` for post-cutoff rows. Output marks
each row `OK`/`x` with a `(past cutoff)` tag.

## Alongside the full val loss

`scripts/full_eval.py --consistency` runs this same test (and Table 3) next to the
uncapped full-validation loss, and reports how many rows behave as expected (correct
pre-cutoff, wrong post-cutoff):
```bash
python scripts/full_eval.py --config configs/train.yaml \
    --repo /home/ubuntu/persist/runs/chrono-instruct-2020/final --cutoff 2020 --consistency
```
See [03 §Full uncapped validation eval](03-figures-1-2-loss-curves.md#full-uncapped-validation-eval).

## All vintages at once (saved + collected)

To evaluate every vintage and collect results without running them one by one:

```bash
bash scripts/eval_all_vintages.sh            # Tables 2-3, all vintages, from the HF repos
bash scripts/eval_all_vintages.sh 2020 2024  # a subset
REPO_DIR=/home/ubuntu/persist/runs bash scripts/eval_all_vintages.sh   # use local checkpoints
```

By default it evaluates the HF-published models `HZ0619/chrono-instruct-v1-<Y>1231`.
Each vintage's results are saved to `results/chrono-instruct-<Y>/eval.json` (like
training saves `metrics.csv`), then aggregated into
`results/replication-report/{eval_results.json, eval_summary.md}` for the report.
Add `ALPACA=1` to also run Figure 3 (see [06](06-figure-3-alpacaeval.md)).
