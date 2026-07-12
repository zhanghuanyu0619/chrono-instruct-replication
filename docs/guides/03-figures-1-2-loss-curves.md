# 03 — Figures 1–2: SFT loss curves

Reproduce the training-loss exhibits from `metrics.csv`. Needs trained runs (see
[02-training-the-models.md](02-training-the-models.md)); each run's `metrics.csv`
lives in its `output_dir` and is mirrored to `results/<name>/`.

## Figure 1 — one run's 3-stage loss curve

```bash
chrono figure --kind 1 --run /home/ubuntu/persist/runs/chrono-instruct-2020
```
Plots the per-stage train/val loss for a single vintage from its `metrics.csv`. The
image defaults **into the run dir** (`<run>/figure1.png`) so it never litters the
project root; override with `--out <path>`. Training already renders this
automatically (the `save_results` step in
[02 §5](02-training-the-models.md#5-configure-the-run)).

## Figure 2 — overlaid across vintages

```bash
chrono figure --kind 2 --runs /home/ubuntu/persist/runs/chrono-instruct-*
```
Overlays the validation-loss curves for every vintage. For the curves to be
apples-to-apples, all runs must share the same `configs/train.yaml` hyperparameters —
see the "Comparable curves" note in
[02 §Sweep](02-training-the-models.md#10-sweep-across-vintages). Default output
`figure2.png`; override with `--out`.

## Combined sweep figure (all 6 vintages)

A richer, dependency-free view built straight from the `results/` tree (pure stdlib,
emits a self-contained SVG — no matplotlib/numpy):
```bash
python scripts/plot_sweep_combined.py                    # -> results/combined/sweep_combined.svg
python scripts/plot_sweep_combined.py [results_dir] [out.svg] [out.html]   # optional overrides
```
Two panel rows over all six vintages (1999 2005 2010 2015 2020 2024), color-coded on a
sequential blue ramp (later cutoff = darker):
- **Row 1** — one panel per SFT stage: validation loss vs optimizer step, one line
  per vintage.
- **Row 2** — headline panel: **best validation loss vs cutoff year**, direct-labeled,
  showing the monotone improvement as the knowledge cutoff moves forward.

It reads each run's `metrics.csv` (val series) and `summary.json` (`final_val_loss`)
under `results/chrono-instruct-<year>/`, so run the sweep + publish first.

## Full uncapped validation eval

During training, periodic eval caps the held-out set at `val_max_blocks: 500` to stay
cheap — that cap only truncates how many val **blocks** are scored, never which
**examples** are held out (the split is seeded, applied before packing). To get the
honest full-val loss, re-score the **same** seeded holdout with **no cap**:
```bash
python scripts/full_eval.py --config configs/train.yaml \
    --repo /home/ubuntu/persist/runs/chrono-instruct-2020/final --cutoff 2020 \
    [--stages stage3_tulu] [--consistency] --out results/full_eval/2020.json
```
- Reuses `data.py`'s split logic (same `dataset`, `seed`, `val_fraction`, screen), so
  the val examples are byte-identical to training's holdout. Nothing is re-trained —
  it only measures.
- Scores every stage in the config by default; limit with `--stages`.
- `--repo` accepts an HF id or a local run dir.
- `--consistency` also runs the president (Table 2) + major-events (Table 3) tests
  alongside the loss — see [04](04-table-2-president-consistency.md) /
  [05](05-table-3-major-events.md).
- Writes a JSON result dict (per-stage `loss`/`ppl`/`val_examples`/`val_blocks`) to
  `--out`.
