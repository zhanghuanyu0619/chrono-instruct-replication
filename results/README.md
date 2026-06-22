# Results

Published training artifacts, one subfolder per run (e.g. `chrono-instruct-2020/`):
`metrics.csv` (the loss curves behind Figures 1–2) and the figure PNGs.

Populated by `scripts/publish_results.sh <run_dir> <name>`, which copies the
metrics, renders Figure 1, and pushes them to GitHub.

Model **weights are not here** — checkpoints go to the Hugging Face Hub
(`chrono push`, or the `push_to_hub` block in `configs/train.yaml`). Git stays
small; only logs and figures live in the repo.
