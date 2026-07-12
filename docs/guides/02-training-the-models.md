# 02 — Training the models

Configure and run the 3-stage curriculum SFT that produces the vintage models, then
publish results. **This is the prerequisite for every exhibit
(03/04/05/06).** Assumes the box is set up per
[01-environment-setup.md](01-environment-setup.md).

> **Division of artifacts:** loss logs + figures → **GitHub** (small, in
> `results/`); checkpoints → **Hugging Face Hub** (large). Weights are never
> committed to git.

## 5. Configure the run

Edit `configs/train.yaml`:
- `model_repo` — the base vintage (e.g. `manelalab/chrono-gpt-v1-20201231`).
- `output_dir` — an **absolute path on the persistent FS**, e.g.
  `/home/ubuntu/persist/runs/chrono-instruct-2020`.
- `min_confidence` — `10` (paper's strict screen) or `null` (keep all label-0).
- `wandb.enabled` — **default `true`**; mirrors loss curves live. Needs
  `wandb login` on the box; if not logged in it silently falls back to CSV-only
  (never crashes the run).
- `push_to_hub.enabled` — **default `true`**; pushes `final/` to your `repo_id`
  when a **complete** run finishes. Partial/smoke runs are skipped (no 7.4 GB
  upload per tuning run). Needs a WRITE token (see
  [01 §HF auth](01-environment-setup.md#3-authenticate-with-hugging-face)).
- `save_results` — **default `true`**; after training, copies `metrics.csv` /
  `config.yaml` / `summary.json` and renders `figure1.png` into `results/<name>/`
  automatically (git-friendly; commit/push is still §8).

Per-stage training knobs (already tuned in the config, worth understanding):
- `early_stop_patience: 3` — stop a stage after 3 evals with no val improvement,
  restore the **best-val** weights, and continue the next stage from them. `min_delta`
  (default `null`) sets how much of a drop counts as improvement.
- `min_lr_ratio: 0.1` — cosine floor at 10 % of each stage's `lr` (not decay-to-0).
- Per-stage `lr` / `eval_every` / `log_every` — Stage 1 uses a high `lr` (tiny stage,
  few steps) and `eval_every: 1`; the big stages use standard LRs and coarser logging.
  `eval_every` is in **optimizer steps**, so keep it well below a stage's step count.

## 6. Train

**Dry-run first** (validate the full loop in minutes): copy the config, keep only
the `stage1_scratch` stage (~1,097 examples), and run:
```bash
cp configs/train.yaml configs/_dryrun.yaml   # then delete stage2/stage3 from `stages:`
chrono train --config configs/_dryrun.yaml
```
✅ loss decreases, `metrics.csv` appears in `output_dir`, `stage1_scratch/` + `final/` saved.

Then the full curriculum:
```bash
chrono train --config configs/train.yaml
```

### 6b. Stage-by-stage (train Stage 1, diagnose, then resume Stages 2–3)
Because each stage continues from the previous stage's weights, you can stop after
Stage 1, inspect it, and resume. Use **two configs that share the same `output_dir`**
(so `metrics.csv` accumulates and the figure shows all stages):

**Phase 1 — Stage 1 only.** Copy `train.yaml` → `train_s1.yaml`, keep only the
`stage1_scratch` entry under `stages:`, keep `model_repo` = the base vintage:
```bash
chrono train --config configs/train_s1.yaml          # saves <output_dir>/stage1_scratch + final
chrono figure --kind 1 --run <output_dir>            # inspect the Stage-1 curve
```
Diagnose: is Stage 1's val still dropping at the last epoch? If so, raise its
`epochs` and re-run (delete `<output_dir>/metrics.csv` first to start the curve clean).

**Phase 2 — resume Stages 2–3.** Copy `train.yaml` → `train_s23.yaml`, keep only
`stage2_self_instruct` + `stage3_tulu`, set **`model_repo` to the Stage-1 checkpoint**
and the **same `output_dir`**:
```yaml
model_repo: /home/ubuntu/persist/runs/chrono-instruct-2020/stage1_scratch   # local dir = resume point
output_dir: /home/ubuntu/persist/runs/chrono-instruct-2020                  # same -> metrics append
```
```bash
chrono train --config configs/train_s23.yaml
```
`from_pretrained` now accepts a local directory, so `model_repo` can be any saved
checkpoint. The combined `metrics.csv` then holds all three stages for one Fig 1.

### Memory (verified)

Full FT is activation-heavy, not weight-heavy (~25 GB is just Adam states).
`grad_checkpoint: true` (config default) recomputes each transformer block in the
backward pass instead of storing its activations: ~10× less block-activation memory
for ~20% slower steps (one extra forward over the checkpointed blocks). Active only
during training (`model.py`, `grad_checkpoint and self.training`), per-block, via
`torch.utils.checkpoint(..., use_reentrant=False)`; paired with `return_hidden=False`,
this fits `batch_size: 8` on one **80GB** card. Without it, batch 8 OOMs even on
80GB; a **40GB** card OOMs at batch 1 regardless. If you still OOM: lower
`batch_size`, raise `grad_accum` to keep the effective batch (e.g. `2`/`16` = 32).

## 10. Sweep across vintages

One run = one `(vintage, config)`. The filtered+packed data is cached and reused,
so only `model_repo`/`output_dir`/`repo_id` change.

**Sequential, one GPU** (the common case — fine-tune all vintages back-to-back,
each auto-published to GitHub + HF):
```bash
bash scripts/train_all_vintages.sh                       # 1999 2005 2010 2015 2020 2024
bash scripts/train_all_vintages.sh 1999 2005 2010 2015 2024   # skip an already-trained vintage
```
It derives a per-vintage config from `configs/train.yaml` (overriding `model_repo`,
`output_dir`, and the HF `repo_id`), trains, and runs `publish_results.sh`. A single
vintage failing is logged and skipped — the sweep continues and prints a `trained:` /
`failed:` summary at the end. Set `HF_USER` / `PERSIST` as env vars if they differ
from the defaults. For a run-finished notification, enable wandb's run-finished
emails (§12). Rough budget: ~3 h/vintage on an 80GB H100 → ~15–24 h for all six
(cache built once).

**Prerequisites for an unattended sweep:** `hf auth login`
([01 §3](01-environment-setup.md#3-authenticate-with-hugging-face)), the GitHub
identity + cached PAT
([01 §3b](01-environment-setup.md#3b-github-identity--credentials-for-publishing-results),
or the per-vintage GitHub publish hangs on a password prompt), and `wandb login`
(config default; also flip on **run-finished emails** in your wandb user settings to
be pinged as each vintage completes). Run it detached (see below).

**Comparable curves:** every vintage must use the *same* `configs/train.yaml`
hyperparameters, or the overlaid Figure 2 isn't apples-to-apples. If you retuned the
config after training an earlier vintage, retrain that vintage too (the data cache is
shared, so it's just GPU time). Verify with
`diff <run_dir>/config.yaml configs/train.yaml` — only `model_repo`/`output_dir`/
`repo_id` should differ.

**Parallel, multi-GPU / cluster:** `scripts/launch_local.sh` (one vintage per GPU)
or `archive/slurm_array.sbatch` (SLURM array — archived, off the main path).

**Detach for a hands-off sweep (Lambda browser terminal / JupyterLab).** These
persist server-side, so a long job survives you closing the browser tab. For a fully
hands-off sweep, still detach it with `nohup` so nothing — not even the terminal
dying — can kill it, and tee the output to a log you can reattach to:
```bash
nohup bash scripts/train_all_vintages.sh 1999 2005 2010 2015 2024 > sweep.log 2>&1 &
tail -f sweep.log     # watch live; Ctrl-C stops watching, NOT the job
```
Check on it later with `tail -f sweep.log`, `nvidia-smi`, or `jobs`. Because the job
is detached, you can close the browser; enable wandb run-finished emails (§12) to be
pinged as each vintage completes. (Over local SSH, use tmux instead — see
[01 §7](01-environment-setup.md#7-tmux-for-long-runs-over-local-ssh).)

## 12. Run-finished notifications (wandb)

wandb is on by default (`configs/train.yaml`), so the simplest way to be told when a
run/vintage finishes is wandb's own notification — no extra code or credentials:

1. `wandb login` on the box (once).
2. In your wandb **user settings → notifications**, enable **run-finished** emails
   (and/or Slack). You'll get an email with a link to the run's curves as each vintage
   completes; crashes are reported too.

Loss curves stream live to the wandb project (`chrono-instruct`) during the run, and
still land in `output_dir/metrics.csv` (+ `results/<name>/`) regardless of wandb.

## 8. Publish results to GitHub + checkpoints to Hugging Face

**Logs + figures → GitHub** (small, tracked under `results/`). Training already
*saves* `results/<name>/` locally (the `save_results` step); this script commits +
pushes it (and renders the figure if training didn't):
```bash
bash scripts/publish_results.sh /home/ubuntu/persist/runs/chrono-instruct-2020 chrono-instruct-2020
```
Needs the GitHub identity + cached PAT from
[01 §3b](01-environment-setup.md#3b-github-identity--credentials-for-publishing-results),
or the `git commit`/`git push` step fails. It degrades gracefully — a missing
`chrono`/figure or push-credential failure warns instead of aborting.

**Checkpoints → Hugging Face Hub** (large weights). `<hf-user>` must be your HF
namespace from `hf auth whoami` (NOT your GitHub username), with a **Write** token:
```bash
chrono push --repo /home/ubuntu/persist/runs/chrono-instruct-2020/final \
            --to <hf-user>/chrono-instruct-v1-20201231 --private
```
(or set `push_to_hub.enabled: true` in the config to do this automatically at the
end of training.)

## Inference + clearing GPU memory

To generate/embed from any vintage (base or instruct), local dir or HF id:
```bash
python scripts/inference_demo.py --repo manelalab/chrono-gpt-instruct-v1-19991231
# or your own: --repo /home/ubuntu/persist/runs/chrono-instruct-2020/final
```
Generates, embeds, then `del model; free_memory()` and reports VRAM. To free the
GPU between model loads in a notebook/REPL:
```python
from chrono_instruct.infer import free_memory
del model
free_memory()   # gc.collect() + torch.cuda.empty_cache()
```

## Gotchas (training / memory)

- **Shut down notebook kernels before training:** a live JupyterLab kernel keeps
  its models on the GPU (tens of GB), so `chrono train` then OOMs even on an 80GB
  card. `nvidia-smi` to spot the stale PID; Kernel → Shut Down All, or `kill <pid>`.
- **`batch_size` on one card:** full FT of 1.55B is memory-heavy; if you OOM on a
  clean 80GB card, lower `batch_size` and raise `grad_accum` to keep the effective
  batch (e.g. `2`/`16` = 32). Watch `nvidia-smi` for the real peak.

---

**Models are done.** The trained checkpoints are now on HF
(`HZ0619/chrono-instruct-v1-*`) and/or in `runs/…/final`, with logs in `results/`.
Proceed to the exhibit guides:
[03 (Figures 1–2)](03-figures-1-2-loss-curves.md) ·
[04 (Table 2)](04-table-2-president-consistency.md) ·
[05 (Table 3)](05-table-3-major-events.md) ·
[06 (Figure 3)](06-figure-3-alpacaeval.md).
