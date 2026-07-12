# 05 — Table 3: dated world-events completion

Reproduce the companion consistency check on dated world events. Needs a trained or
released model (see [02-training-the-models.md](02-training-the-models.md)).

## Run it

Same command as Table 2 — `chrono eval` prints both tests:
```bash
chrono eval --repo manelalab/chrono-gpt-instruct-v1-20201231 --cutoff 2020
```
`--repo` accepts an HF repo id or a local run dir; `--cutoff` is the model's
knowledge-cutoff year. See [04](04-table-2-president-consistency.md) for the president
half of the output.

## The test (`eval.py` `major_events_test`)

Each item is a sentence about a dated event with the key term left blank (e.g. the 2001
Enron *scandal*, 2008 subprime mortgage *crisis*, 2020 *COVID*, 2022 *ChatGPT*). The
model greedy-decodes **3 tokens** (`top_k=1`, deterministic, as in the paper) and the
completion is checked for the accepted term(s), matched case-insensitively.

**Consistency logic.** Same as Table 2: a chronologically consistent model should be
**CORRECT** for events on or before its cutoff and **WRONG** for events after it. Each
row carries a `past_cutoff` flag (`event_year > cutoff_year`) — right before the
cutoff, wrong after. Output marks each row `OK`/`x` with a `(past cutoff)` tag.

The event prompts/terms are transcribed from Table 3, Panel A; verify exact wording
against the PDF if you need a byte-faithful reproduction.

## Alongside the full val loss

`scripts/full_eval.py --consistency` runs this test (and Table 2) next to the uncapped
full-validation loss — see
[04 §Alongside the full val loss](04-table-2-president-consistency.md#alongside-the-full-val-loss)
and [03 §Full uncapped validation eval](03-figures-1-2-loss-curves.md#full-uncapped-validation-eval).

## All vintages at once

The major-events test runs together with Table 2 in the same commands — see
[04 §All vintages at once](04-table-2-president-consistency.md#all-vintages-at-once-saved--collected).
`bash scripts/eval_all_vintages.sh` saves and collects both tables for every vintage.
