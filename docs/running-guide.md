# Running Guide — moved

This monolithic guide has been split into focused, single-task documents under
[`docs/guides/`](guides/). Start with the index at
[`docs/guides/README.md`](guides/README.md), then follow the prerequisites chain:
**environment setup → train → any exhibit**.

## Guides

- [01-environment-setup.md](guides/01-environment-setup.md) — provision a GPU box,
  install the stable `cu126` torch stack, authenticate HF + GitHub, verify, and tmux.
  **Do this first.**
- [02-training-the-models.md](guides/02-training-the-models.md) — configure and run
  the 3-stage curriculum SFT (one vintage or the whole sweep), then publish logs to
  GitHub and checkpoints to HF. **Produces the vintage models every exhibit needs.**
- [03-figures-1-2-loss-curves.md](guides/03-figures-1-2-loss-curves.md) — Figures 1–2
  loss curves, the combined sweep figure, and the uncapped full-validation eval.
- [04-table-2-president-consistency.md](guides/04-table-2-president-consistency.md) —
  Table 2: U.S. president prediction consistency.
- [05-table-3-major-events.md](guides/05-table-3-major-events.md) — Table 3: dated
  world-events completion.
- [06-figure-3-alpacaeval.md](guides/06-figure-3-alpacaeval.md) — Figure 3: AlpacaEval
  win-rate vs Qwen-1.5-1.8B-Chat.

Model weights live on Hugging Face (`HZ0619/chrono-instruct-v1-*`); loss logs and
figures live in `results/`.
