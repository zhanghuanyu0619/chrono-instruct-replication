# Running Guides — ChronoGPT-Instruct replication

The old monolithic `docs/running-guide.md` is split into focused guides. Each one
is self-contained for a single task (Lambda box, `$PERSIST`, one 1.55B model per
GPU).

## Guides

- [01-environment-setup.md](01-environment-setup.md) — provision a GPU box, install
  the stable `cu126` torch stack, authenticate HF + GitHub, verify the pipeline, and
  keep long runs alive with tmux. **Do this first.**
- [02-training-the-models.md](02-training-the-models.md) — configure and run the
  3-stage curriculum SFT for one vintage or the whole sweep, then publish logs to
  GitHub and checkpoints to HF. **Produces the vintage models every exhibit needs.**
- [03-figures-1-2-loss-curves.md](03-figures-1-2-loss-curves.md) — Figure 1 (one
  run's 3-stage loss curve), Figure 2 (overlaid across vintages), the combined sweep
  figure, and the uncapped full-validation eval.
- [04-table-2-president-consistency.md](04-table-2-president-consistency.md) —
  Table 2: U.S. president prediction consistency across the knowledge cutoff.
- [05-table-3-major-events.md](05-table-3-major-events.md) — Table 3: dated
  world-events completion across the cutoff.
- [06-figure-3-alpacaeval.md](06-figure-3-alpacaeval.md) — Figure 3: AlpacaEval
  length-controlled win-rate vs Qwen-1.5-1.8B-Chat.

## Prerequisites chain

```
01-environment-setup  ->  02-training-the-models  ->  any exhibit (03 / 04 / 05 / 06)
```

Every exhibit needs a trained (or released) model, so set up the box, then train,
then reproduce whichever figure/table you want. The exhibit guides are independent
of each other.

## Where artifacts live

- **Model weights** → Hugging Face Hub (`HZ0619/chrono-instruct-v1-*`, one repo per
  vintage). Large; never committed to git.
- **Loss logs + figures** → `results/` in this repo (small; committed via
  `publish_results.sh`). Live curves also stream to the `chrono-instruct` wandb
  project.
