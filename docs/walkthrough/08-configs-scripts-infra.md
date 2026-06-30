# 08 — Configs, Scripts, Packaging, and Running It on a GPU Box

**What this doc is / who it's for.** This is the *operations* chapter: it explains the two YAML config files, every run script under `scripts/`, the Python packaging in `pyproject.toml`, the smoke test, the `.gitignore` policy, and what a finished result looks like in `results/`. The audience is a quantitatively strong reader who is comfortable scripting but has little ML/MLOps background and is not a systems/DevOps person — so every cloud-GPU term (virtual environment, CUDA, persistent filesystem, SLURM, console-script, editable install) is defined from scratch the first time it appears. By the end you should understand the whole operational picture well enough to run a vintage on Lambda Labs and to discuss the pipeline with Manela. Earlier docs (`01-ml-primer.md`, `05-train.md`, `04-data.md`, `06-infer-and-eval.md`) cover *what* the code computes; this one covers *how you drive it on a machine*. The full step-by-step runbook lives in `docs/running-guide.md`; this chapter summarizes and cross-links it rather than repeating it.

---

## Table of contents

1. [`configs/train.yaml` — every field](#1-configstrainyaml--every-field)
2. [`configs/eval.yaml` — the Figure-3 pipeline](#2-configsevalyaml--the-figure-3-pipeline)
3. [The scripts](#3-the-scripts)
   - [3.1 `lambda_setup.sh` — provisioning a Lambda box](#31-lambda_setupsh--provisioning-a-lambda-box)
   - [3.2 `launch_local.sh` — one vintage per GPU](#32-launch_localsh--one-vintage-per-gpu)
   - [3.3 `slurm_array.sbatch` — array jobs on a cluster](#33-slurm_arraysbatch--array-jobs-on-a-cluster)
   - [3.4 `inference_demo.py` — generate + embed + free memory](#34-inference_demopy--generate--embed--free-memory)
   - [3.5 `make_vintage_config.py` — derive a per-vintage config](#35-make_vintage_configpy--derive-a-per-vintage-config)
   - [3.6 `publish_results.sh` — logs + figures to GitHub](#36-publish_resultssh--logs--figures-to-github)
   - [3.7 `train_all_vintages.sh` — the sequential sweep](#37-train_all_vintagessh--the-sequential-sweep)
4. [`pyproject.toml` — packaging, deps, the `chrono` command](#4-pyprojecttoml--packaging-deps-the-chrono-command)
5. [`tests/test_smoke.py` — the CPU end-to-end test](#5-teststest_smokepy--the-cpu-end-to-end-test)
6. [`.gitignore` — the GitHub-vs-HuggingFace artifact split](#6-gitignore--the-github-vs-huggingface-artifact-split)
7. [`results/` — what a published run looks like](#7-results--what-a-published-run-looks-like)
8. [From zero to a trained vintage — operational checklist](#8-from-zero-to-a-trained-vintage--operational-checklist)
9. [Mini-FAQ](#9-mini-faq)

---

## 1. `configs/train.yaml` — every field

A **config file** here is a plain-text YAML file that the training entry point (`chrono train --config configs/train.yaml`) reads to learn everything about the run. The design philosophy is *one vintage, one config*: you never edit Python to change a run; you edit (or generate) a YAML. The file is short and every field matters, so we walk through all of them. The mapping from these fields to the actual training loop is in `05-train.md` (loss, optimizer, scheduler) and `04-data.md` (filtering, packing); here we explain what to *set* and *why*.

### The model / data / location block

```yaml
model_repo: manelalab/chrono-gpt-v1-20201231
dataset: manelalab/ChronoInstruct-SFT
output_dir: runs/chrono-instruct-2020
cache_dir: cache            # packed blocks cached here; shared across vintage runs
min_confidence: 10          # confidence gate for label-0 retention (set null to skip)
```

- **`model_repo`** — the *base vintage* you fine-tune. ChronoGPT ships one pretrained checkpoint per knowledge-cutoff date (the "vintage"); `chrono-gpt-v1-20201231` is the model that has only seen text up to 2020-12-31. This is the single most important field to change when sweeping vintages. It accepts either a Hugging Face repo id (downloaded and cached) *or* a local directory — that local-dir capability is what makes the stage-by-stage resume in §8 work (you point `model_repo` at a saved checkpoint).
- **`dataset`** — the Hugging Face dataset id of the raw ChronoInstruct SFT pairs (instruction → response). Downloaded once and cached.
- **`output_dir`** — where this run writes everything: checkpoints, `metrics.csv`, `config.yaml`, `summary.json`. **This must live on a persistent filesystem** on a cloud box (see §3.1) or you lose the run when the instance is torn down. Use an absolute path like `/home/ubuntu/persist/runs/chrono-instruct-2020` in practice; the relative `runs/...` default is for local testing.
- **`cache_dir`** — where the *filtered and packed* training blocks are stored after the expensive data-prep step. The key point: this cache is **shared across all vintages**. The temporal screen (keep only GPT-4.1-labeled "pre-2000" pairs at confidence 10) is a single conservative screen, not a per-vintage one — pre-2000 data is before the cutoff τ for *every* vintage τ ≥ 1999 — so the packed corpus is built once and reused. That is why a six-vintage sweep only pays the data-prep cost once (see `04-data.md`).
- **`min_confidence: 10`** — the gate for retaining a pair. The GPT-4.1 classifier labels each pair 0 ("pre-2000") with a confidence score; only label-0, confidence-10 pairs survive. This is the strict screen from the paper (§2.2.1) and drops the raw release from 647,944 rows to ~425,119. Set to `null` to skip the screen entirely (keep all label-0 pairs) — useful only for experiments.

### The data-shape and reproducibility block

```yaml
block_size: 1792
seed: 123                    # single global seed: train/val split, shuffle, and sampling all derive from it
```

- **`block_size: 1792`** — the fixed sequence length (in tokens) that examples are packed into for training. Multiple short instruction/response pairs are concatenated into one 1,792-token block so the GPU never trains on mostly-padding. This is the "context length" the model sees per step; see `04-data.md` for the packing logic and `01-ml-primer.md` §2 for what a token is.
- **`seed: 123`** — a single global random seed. Everything stochastic — the train/validation split, the data shuffle, and any sampling — derives from this one number, so a run is reproducible: same seed + same config + same code ⇒ same batches.

### The optimization block

```yaml
batch_size: 8
grad_accum: 4
grad_checkpoint: true        # recompute blocks in backward: ~10x less activation memory, ~20% slower.
warmup_ratio: 0.03
weight_decay: 0.0
grad_clip: null              # null = no clipping (grad norm still logged); set e.g. 1.0 to clip
val_fraction: 0.05
```

- **`batch_size: 8`** — how many 1,792-token blocks go through the GPU in one forward/backward pass. Bigger = more stable gradients but more memory. 8 is what fits one 80GB card with gradient checkpointing on.
- **`grad_accum: 4`** — gradient accumulation steps. Instead of updating the weights after every batch, the loop sums gradients over 4 batches and updates once. The **effective batch size** the optimizer actually "sees" is `batch_size * grad_accum = 8 * 4 = 32` blocks. This is the standard trick to get a large effective batch on a single card whose memory only holds 8 blocks at a time. If you must shrink `batch_size` to avoid running out of memory, raise `grad_accum` to keep the product at 32 (e.g. `2`×`16`).
- **`grad_checkpoint: true`** — gradient (activation) checkpointing. During the backward pass, instead of storing every intermediate activation from the forward pass (memory-expensive), it *recomputes* each transformer block on the fly. Cost: ~20% slower steps. Benefit: ~10× less activation memory, which is what lets `batch_size: 8` fit on one 80GB card. This is the difference between fitting and an out-of-memory (OOM) crash here. See the verified memory note in `docs/running-guide.md` §6.
- **`warmup_ratio: 0.03`** — the first 3% of training steps ramp the learning rate up from ~0 to its target value, then it decays. Warmup avoids destabilizing the pretrained weights with a large step on the very first, noisy batches. See the scheduler in `05-train.md`.
- **`weight_decay: 0.0`** — L2-style regularization on the weights. Set to 0 here (SFT on a strong pretrained model rarely needs it); raise it if you see overfitting.
- **`grad_clip: null`** — gradient clipping. `null` means no clipping (the gradient norm is still *logged* so you can watch for spikes). Set e.g. `1.0` to cap the gradient norm and tame occasional huge updates.
- **`val_fraction: 0.05`** — fraction of the packed blocks held out as a validation set to track generalization. 5% held out, 95% trained on.

### The logging / checkpointing cadence block

```yaml
log_every: 20                # train metrics (loss, lr, grad_norm, tok/s, VRAM) logged every N steps
eval_every: 200              # periodic in-stage validation -> a real val curve (like the paper's Fig 1)
val_max_blocks: 500          # cap the random held-out set so EACH eval covers the whole val set cheaply
save_every: 500
```

- **`log_every: 20`** — write a row of *training* metrics (loss, perplexity, learning rate, gradient norm, tokens/sec, GPU memory) to `metrics.csv` every 20 steps. These rows are the raw material for Figures 1–2.
- **`eval_every: 200`** — every 200 steps, run a validation pass and log a *val* row. This is what produces a genuine validation curve over the course of a stage, matching the paper's Figure 1 (loss vs. step).
- **`val_max_blocks: 500`** — cap the validation set at 500 random blocks per eval. Without a cap, each eval would scan the entire held-out set and slow training; 500 blocks gives a cheap, stable estimate that still covers the val distribution.
- **`save_every: 500`** — checkpoint the model to `output_dir` every 500 steps, so a crash loses at most 500 steps.

### Optional integrations: W&B and the Hub

```yaml
wandb:
  enabled: false
  project: chrono-instruct
  name: null                # defaults to the output_dir basename
push_to_hub:
  enabled: false            # push output_dir/final to the Hub when training ends
  repo_id: HZ0619/chrono-instruct-v1-20201231   # must be your HF namespace (hf auth whoami), NOT GitHub
  final_stage: stage3_tulu  # only a run ENDING here pushes to repo_id; partial runs get a "-<stage>" suffix
  private: true             # needs a WRITE token (hf auth login / HF_TOKEN)
```

- **`wandb`** — Weights & Biases is an optional cloud dashboard that mirrors your loss curves live in a browser. **Off by default**, and importantly: loss curves *always* go to `output_dir/metrics.csv` regardless, so W&B is pure convenience, never a dependency. `project` is the W&B project name; `name: null` defaults the run name to the `output_dir` basename. To use it: set `enabled: true` and install the `viz` extra (§4).
- **`push_to_hub`** — automatically upload the finished weights to the Hugging Face Hub when training ends.
  - `enabled: false` — off by default; flip to `true` to auto-push.
  - `repo_id` — the destination Hub repo. **This must be your HF namespace** (check with `hf auth whoami`), not your GitHub username — a common mix-up. Weights go to HF, never to GitHub (§6).
  - `final_stage: stage3_tulu` — only a run that *ends* at this stage pushes to the clean `repo_id`. A partial run that stops earlier gets a `-<stage>` suffix appended, so a Stage-1-only run can't masquerade as the finished model.
  - `private: true` — make the Hub repo private; requires a **Write** token (`hf auth login` or the `HF_TOKEN` env var).

### The 3-stage curriculum

```yaml
stages:
  - name: stage1_scratch
    sources: ["scratch"]
    epochs: 3
    lr: 3.0e-5
  - name: stage2_self_instruct
    sources: ["self-instruct", "self-generated", "gpt-3"]
    epochs: 2          # SFT standard is 1-3 epochs; Tulu-3 used 2. Watch the val curve.
    lr: 3.0e-5
  - name: stage3_tulu
    sources: ["tulu"]
    epochs: 2          # ~2x our 1-epoch step count, closer to the paper's Fig 1 (~14k steps)
    lr: 2.0e-5
```

This is a **curriculum**: three stages trained in order, each one continuing from the previous stage's weights (not from scratch each time). Each stage selects its training data by `sources` — a list of case-insensitive substrings matched against the dataset's `source` column. Run `chrono inspect` first to see the exact source values and counts, because the substrings must match real values.

- **`name`** — the stage label; also the name of the per-stage checkpoint subfolder in `output_dir`.
- **`sources`** — which slices of the corpus feed this stage. Stage 1 uses hand-written "scratch" instructions; Stage 2 uses self-instruct / GPT-3-generated data; Stage 3 uses the Tulu mixture. The curriculum goes from cleaner/simpler toward broader/messier data.
- **`epochs`** — how many full passes over that stage's data. An **epoch** is one complete pass over the training examples.
- **`lr`** — the peak learning rate for that stage (after warmup). Note Stage 3 uses a lower LR (`2.0e-5`) to fine-tune gently on the final mixture.

**On the epoch counts and under-training.** The current settings (3 / 2 / 2 epochs) are an initial guess to be tuned against the paper's Figure 1 curves, and they **under-train relative to the paper**. The very first real run (the one published in `results/`, §7) used 3 / 1 / 1 epochs and the validation loss was still falling — meaning the model had not converged. The comments in the file capture the reasoning: SFT standard is 1–3 epochs, Tulu-3 used 2, and 2 epochs on Stage 3 roughly doubles the step count toward the paper's ~14k steps. The practical workflow (§8, and `docs/running-guide.md` §6b) is to train Stage 1, look at whether its val curve is still dropping, raise `epochs` if so, and only then proceed. Expect to increase these numbers as you match Figure 1.

---

## 2. `configs/eval.yaml` — the Figure-3 pipeline

`eval.yaml` is much smaller because most evaluation is driven by CLI flags, not config. Its job is to document the AlpacaEval / Figure-3 win-rate pipeline and pin the one setting that must stay fixed across all vintages (the reference model).

```yaml
alpaca_eval:
  reference_model: Qwen/Qwen1.5-1.8B-Chat
  length_controlled: true
```

The header comments lay out two evaluations:

- **Table 2 (the "president test")** is a one-liner: `chrono eval --repo <model> --cutoff <year>`. It checks that a vintage doesn't "know" facts from after its cutoff. No config needed.
- **Figure 3 (AlpacaEval length-controlled win-rate)** is a 3-step pipeline, glued together by the `chrono` CLI (see `06-infer-and-eval.md` for the flow):
  1. Generate your model's answers to a fixed instruction set:
     `chrono alpaca --backend chrono --repo runs/chrono-instruct-2020/final --name chrono-2020 --out out/chrono-2020.json`
  2. Generate the **reference** model's answers once (the comparison baseline):
     `chrono alpaca --backend hf --repo Qwen/Qwen1.5-1.8B-Chat --name qwen --out out/qwen.json`
  3. Compute the win-rate of your model vs. the reference:
     `chrono winrate --model out/chrono-2020.json --reference out/qwen.json`
  Then collect `{year: winrate}` across vintages and render the figure: `chrono figure --kind 3 --results ...`.

The two YAML fields:

- **`reference_model: Qwen/Qwen1.5-1.8B-Chat`** — the fixed opponent every vintage is scored against. Keeping this constant is what makes win-rates comparable across vintages.
- **`length_controlled: true`** — use AlpacaEval's *length-controlled* win-rate, which corrects for the known bias where LLM judges prefer longer answers. This matches the paper's Figure 3 metric.

One operational note from the comments: step 3 calls the canonical `alpaca_eval` package, whose LLM judge needs an annotator API key (e.g. `OPENAI_API_KEY`), and the package only installs with the `eval` extra: `pip install -e '.[eval]'` (§4).

---

## 3. The scripts

Everything in `scripts/` is a thin convenience wrapper around the same `chrono` CLI. None of them contain training logic — they set up the environment, derive configs, fan out across GPUs, or publish artifacts. We go through each one.

A few shell conventions you'll see at the top of every bash script:

- `set -euo pipefail` — fail fast and loudly: `-e` exit on any error, `-u` error on use of an unset variable, `-o pipefail` make a pipeline fail if *any* stage fails. This prevents a script from silently continuing after a broken step.
- `"${VAR:-default}"` — "use `$VAR` if it's set, otherwise this default." This is how the scripts let you override `PERSIST`, `HF_USER`, etc. via environment variables without editing the file.

### 3.1 `lambda_setup.sh` — provisioning a Lambda box

**Context: what a Lambda box is and why setup is fiddly.** Lambda Labs rents you a cloud machine with one or more GPUs. You SSH in, and it starts as a near-blank Linux box. Crucially, its *local disk is ephemeral* — when you terminate the instance, that disk is wiped. Lambda offers a separate **persistent filesystem** (a network volume that survives instance termination); you mount it at a path and put anything you want to keep there. `lambda_setup.sh` provisions a fresh box: it installs the right PyTorch, installs this package, and routes all large downloads to the persistent volume. Run it once per new instance.

```bash
PERSIST="${PERSIST:-$HOME/persist}"     # mount your persistent filesystem here
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"  # use cu124 if driver is CUDA 12.4
mkdir -p "$PERSIST/hf-cache" "$PERSIST/runs"
export HF_HOME="$PERSIST/hf-cache"      # large downloads land on the persistent FS
```

- `PERSIST` is the persistent-volume path. Everything that must survive — caches, runs, the environment itself — goes under it.
- `HF_HOME` tells Hugging Face where to cache downloaded models and datasets. Pointing it at `$PERSIST` means you download a multi-GB vintage once, and it survives instance restarts.
- **CUDA** is NVIDIA's GPU compute platform; PyTorch must be built against the CUDA version your box's GPU driver supports. `cu126` = "CUDA 12.6". The `CUDA_INDEX` variable lets you switch to `cu124` if your driver is 12.4.

```bash
python3 -m venv "$PERSIST/venv"
source "$PERSIST/venv/bin/activate"
python -m pip install --upgrade pip
```

A **virtual environment (venv)** is an isolated Python installation in a folder — its own interpreter and its own set of installed packages — so this project's dependencies don't collide with the system Python or other projects. `python3 -m venv $PERSIST/venv` creates it *on the persistent volume* (so it survives restarts); `source .../activate` switches your shell to use it; from then on `python` and `pip` mean the venv's.

```bash
pip install torch==2.7.0 --index-url "$CUDA_INDEX"
pip install -e .
```

This ordering is the whole point of the script. **PyTorch is installed first, explicitly, from the cu126 index**, *before* the package's own dependencies. Why:

- The upstream ChronoGPT repo pins a *nightly* (unreleased, daily-build) torch, which is unstable. This script instead pins the stable `torch==2.7.0` cu126 wheel, which has been tested on H100 80GB.
- If you let `pip install -e .` resolve torch from PyPI's default index, you'd get a generic build that may not match your CUDA. Installing the cu126 wheel *first* means it's already present, so the later install sees the requirement satisfied and doesn't override it. The version (`2.7.0`) matches the pin in `pyproject.toml`; the only thing this script adds is the *index URL* that selects the cu126 build.

`pip install -e .` is the **editable install** of this project (explained in §4): it installs `chrono-instruct` and all its non-torch dependencies (datasets, tiktoken, huggingface-hub, pyyaml, numpy, tqdm).

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "Setup done. Activate with: source $PERSIST/venv/bin/activate"
echo "Then: pytest -q   (smoke test, no download)   and   chrono inspect"
```

A one-line sanity check that torch sees the GPU (`cuda.is_available()` should print `True` and a device name), then a reminder of the next two commands.

### 3.2 `launch_local.sh` — one vintage per GPU

If your box has *several* GPUs, you can train several vintages at once — one per GPU. That's all this script does.

```bash
gpu=0
for year in "$@"; do
  cfg="configs/_vintage_${year}.yaml"
  out="$PERSIST/runs/chrono-instruct-${year}"
  python scripts/make_vintage_config.py --base "$BASE_CFG" --out "$cfg" \
    --cutoff "${year}1231" --output-dir "$out" --hf-user "$HF_USER"
  echo "GPU $gpu -> vintage $year ($out)"
  CUDA_VISIBLE_DEVICES=$gpu chrono train --config "$cfg" > "logs_${year}.txt" 2>&1 &
  gpu=$((gpu + 1))
done
wait
```

- `for year in "$@"` loops over the years you pass on the command line (`bash scripts/launch_local.sh 1999 2005 2010`).
- For each year it derives a per-vintage config (§3.5), then launches training.
- **`CUDA_VISIBLE_DEVICES=$gpu`** is the key line: this environment variable tells PyTorch *which physical GPU to use*. Setting it to `0`, then `1`, then `2`, ... pins each vintage to a distinct card so they don't fight over memory.
- `> "logs_${year}.txt" 2>&1` redirects both stdout and stderr to a per-vintage log file. The trailing **`&`** backgrounds the job so the loop immediately starts the next one. `wait` at the end blocks until all backgrounded jobs finish.

### 3.3 `slurm_array.sbatch` — array jobs on a cluster

**What SLURM is.** On a university HPC cluster you don't run jobs interactively; you *submit* them to a **scheduler** that queues your job and runs it when a GPU frees up. SLURM is the most common such scheduler. You submit a "batch script" (`sbatch slurm_array.sbatch`); the `#SBATCH` comment lines at the top are directives telling SLURM what resources you need.

```bash
#SBATCH --job-name=chrono-instruct
#SBATCH --array=0-5            # one task per vintage
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a.out
```

- `--job-name` — a label for the job in the queue.
- **`--array=0-5`** — this is an **array job**: SLURM launches *six* near-identical tasks, numbered 0 through 5, each getting its own GPU as one becomes available. The number is exposed inside each task as `$SLURM_ARRAY_TASK_ID`.
- `--gpus=1` — each task gets one GPU (the 1.55B model is one process on one card).
- `--cpus-per-task=8` — 8 CPU cores per task (for data loading).
- `--time=24:00:00` — kill the task if it runs past 24 hours (a guardrail).
- `--output=logs/%x_%a.out` — write logs to `logs/<jobname>_<arrayindex>.out`.

```bash
VINTAGES=(1999 2005 2010 2015 2020 2024)
year="${VINTAGES[$SLURM_ARRAY_TASK_ID]}"
...
srun chrono train --config "$cfg"
```

Each array task indexes into the `VINTAGES` list by its array id to pick *its* vintage (task 0 → 1999, task 5 → 2024), derives the per-vintage config, then runs training. `srun` is SLURM's way of launching the actual command inside the allocated resources. The comment says it best: *same entrypoint as everything else; SLURM just picks the vintage by array index.* This is the parallel-cluster analog of `launch_local.sh`.

### 3.4 `inference_demo.py` — generate + embed + free memory

A tiny script to confirm a trained or released model actually works, and to demonstrate releasing GPU memory between model loads. It ties directly to `06-infer-and-eval.md`.

```python
model, device = load(args.repo)
...
prompt = PROMPT_NO_INPUT.format(instruction=args.instruction)
print(generate(model, device, prompt, max_new_tokens=60))
...
v = embed(model, device, "Inflation is a sustained rise in the general price level.", layer=-1)
print("embedding shape:", tuple(v.shape))
...
del model
free_memory()
```

- `load(args.repo)` — load a model from an HF repo id *or* a local run dir (e.g. `runs/chrono-instruct-2020/final`).
- It prints VRAM after load, then runs two of the three things the model is for: **generation** (text completion of an instruction prompt, formatted with `PROMPT_NO_INPUT` so the prompt template matches training) and **embedding** (a hidden-state vector for downstream finance use, `layer=-1` = last layer).
- `del model; free_memory()` — the demonstration payoff: delete the model and call `free_memory()` (which runs `gc.collect()` + `torch.cuda.empty_cache()`), then print VRAM again to show it dropped. This is the pattern to use in a notebook/REPL when you want to load several vintages in sequence without OOMing.

Run it with `python scripts/inference_demo.py --repo manelalab/chrono-gpt-instruct-v1-19991231`.

### 3.5 `make_vintage_config.py` — derive a per-vintage config

Every sweep script calls this. Given the base `train.yaml`, it writes a new config with three fields overridden for a specific vintage: `model_repo`, `output_dir`, and `push_to_hub.repo_id`.

```python
cfg["model_repo"] = f"manelalab/chrono-gpt-v1-{args.cutoff}"
cfg["output_dir"] = args.output_dir
cfg.setdefault("push_to_hub", {})
cfg["push_to_hub"]["repo_id"] = f"{args.hf_user}/chrono-instruct-v1-{args.cutoff}"
```

The important design choice, called out in the docstring: this is a **structure-aware YAML edit, not a text substitution.** It loads the YAML into a dict, changes specific keys, and writes it back. A naive `sed` find/replace on the year would be fragile — e.g. the repo id `chrono-instruct-v1-20201231` contains a `v1-` infix that a year-only replace can miss, which would silently leave every vintage pushing to the *same* HF repo (overwriting each other). Editing by key avoids that entire class of bug. Usage:

```bash
python scripts/make_vintage_config.py --base configs/train.yaml \
  --out configs/_vintage_1999.yaml --cutoff 19991231 \
  --output-dir /home/ubuntu/persist/runs/chrono-instruct-1999 --hf-user HZ0619
```

The generated `configs/_vintage_*.yaml` files are gitignored (§6) — they're disposable, regenerated on demand.

### 3.6 `publish_results.sh` — logs + figures to GitHub

After a run finishes, this copies the *small* artifacts into `results/<name>/` and commits them to GitHub. Weights are deliberately **not** touched here — they go to HF (§6).

```bash
DEST="results/$NAME"
mkdir -p "$DEST"
cp "$RUN_DIR/metrics.csv" "$DEST/metrics.csv"
cp "$RUN_DIR/config.yaml"  "$DEST/" 2>/dev/null || true   # resolved run config (reproducibility)
cp "$RUN_DIR/summary.json" "$DEST/" 2>/dev/null || true   # final val loss, peak VRAM, throughput
chrono figure --kind 1 --run "$RUN_DIR" --out "$DEST/figure1.png"

git add "$DEST"
git commit -m "results: $NAME (loss curves + metrics + summary)"
git push origin main
```

- Copies `metrics.csv` (the loss curves), and best-effort copies `config.yaml` (the *resolved* config the run actually used — for reproducibility) and `summary.json` (final val loss, peak VRAM, throughput). The `2>/dev/null || true` means "don't fail if these optional files are missing."
- Renders Figure 1 (the loss curve) into the results folder with `chrono figure --kind 1`.
- Adds, commits, and pushes to GitHub. The result: GitHub holds a compact, browsable record of every run's curves and final numbers, with the heavy weights living elsewhere.

Usage: `bash scripts/publish_results.sh /home/ubuntu/persist/runs/chrono-instruct-2020 chrono-instruct-2020`.

### 3.7 `train_all_vintages.sh` — the sequential sweep

The common case: fine-tune several vintages back-to-back on **one** GPU, auto-publishing each. The cache makes this efficient — the filtered/packed data is built during the first vintage and every later vintage skips data prep.

```bash
YEARS=("$@")
[ ${#YEARS[@]} -eq 0 ] && YEARS=(1999 2005 2010 2015 2020 2024)

for Y in "${YEARS[@]}"; do
    CUTOFF="${Y}1231"
    NAME="chrono-instruct-${Y}"
    CFG="configs/_vintage_${Y}.yaml"          # gitignored
    OUT="$PERSIST/runs/$NAME"
    ...
    python scripts/make_vintage_config.py --base "$BASE_CFG" --out "$CFG" \
        --cutoff "$CUTOFF" --output-dir "$OUT" --hf-user "$HF_USER"
    chrono train --config "$CFG"
    bash scripts/publish_results.sh "$OUT" "$NAME" || echo "WARN: publish failed for $NAME (continuing)"
done
```

- Defaults to the six paper vintages if you pass none; otherwise trains exactly the years you list (`bash scripts/train_all_vintages.sh 1999 2020`).
- For each: derive the per-vintage config (§3.5), train, then publish (§3.6). The `|| echo "WARN..."` means a failed publish logs a warning but doesn't abort the rest of the sweep.
- Run it inside `tmux` (so a dropped SSH connection doesn't kill a multi-hour job — see `docs/running-guide.md` §11). Whether checkpoints go to HF depends on `push_to_hub.enabled` in the base config; the `repo_id` is overridden per vintage. Rough budget: ~3 h/vintage on an 80GB H100, so ~15–24 h for all six (cache built once).

---

## 4. `pyproject.toml` — packaging, deps, the `chrono` command

`pyproject.toml` is the standard Python project manifest: it declares the package name, version, what it depends on, and how to install it. Understanding it demystifies the `pip install -e '.[dev,viz,eval,nb]'` line you'll type during setup.

```toml
[project]
name = "chrono-instruct"
version = "0.0.1"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch==2.7.0",   # version pinned here; lambda_setup.sh adds the cu126 index at install time
    "datasets>=2.18",
    "tiktoken>=0.6",
    "huggingface-hub>=0.23",
    "pyyaml>=6.0",
    "numpy",
    "tqdm",
]
```

- **`requires-python = ">=3.10,<3.13"`** — works on Python 3.10–3.12.
- **`dependencies`** — the *core* libraries always installed: `torch` (the deep-learning engine, version-pinned here — `lambda_setup.sh` only adds the cu126 *index* so you get the GPU build), `datasets` + `huggingface-hub` (download data/models from HF), `tiktoken` (the GPT-2 tokenizer), `pyyaml` (read the configs), `numpy`, `tqdm` (progress bars). Note what's *not* here: matplotlib, transformers, pytest — those are optional, below.

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.5", "mypy>=1.10"]
viz = ["matplotlib>=3.7", "wandb>=0.16"]                       # figures + optional W&B
eval = ["alpaca-eval>=0.6", "transformers>=4.40", "accelerate>=0.30"]  # Figure 3
nb = ["jupyterlab>=4", "ipykernel>=6"]                         # verification notebook (§4)
```

**Optional-dependency extras** are named bundles of extra packages you install only if you need that capability. You select them in square brackets:

- **`dev`** — the test/lint/typecheck toolchain: `pytest` (run the smoke test), `ruff` (linter/formatter), `mypy` (type checker). You need this just to run `pytest -q`.
- **`viz`** — `matplotlib` (render Figures 1–3) and `wandb` (the optional live dashboard, §1).
- **`eval`** — the AlpacaEval / Figure-3 stack: `alpaca-eval` (the judge), `transformers` + `accelerate` (to run the Qwen reference model via the HF backend).
- **`nb`** — `jupyterlab` + `ipykernel` for the verification notebook (`notebooks/verify_pipeline.ipynb`).

`pip install -e '.[dev,viz,eval,nb]'` installs the package **plus all four extras at once** (the quotes protect the brackets from the shell). That's the one command in `docs/running-guide.md` §2 that gives you everything: tests, figures, AlpacaEval, and notebook tooling.

```toml
[project.scripts]
chrono = "chrono_instruct.cli:main"
```

This is the **console-script entry point**. It says: when this package is installed, create a command-line program named `chrono` that runs the `main()` function in `chrono_instruct/cli.py`. That's why you can type `chrono train`, `chrono figure`, `chrono inspect`, etc. anywhere in the activated venv — there's no `python some_long_path.py`; the install registers `chrono` on your PATH.

```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 110

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- `packages.find where = ["src"]` — the source code lives under `src/` (a `src`-layout package). This is what makes `-e` (editable) installs and tests find `chrono_instruct`.
- `[tool.ruff] line-length = 110` — the linter allows lines up to 110 chars.
- `[tool.pytest.ini_options] pythonpath = ["src"]` — tells pytest to look in `src/` for the package, so `pytest -q` works without installing.

**Editable install (`-e`), explained.** A normal `pip install` copies the package into the venv's site-packages; if you then edit the source, the installed copy is stale. `pip install -e .` ("editable") instead *links* to your working directory, so any edit to the source takes effect immediately with no reinstall. That's ideal for a project you're actively developing or tweaking — exactly the replication situation here.

---

## 5. `tests/test_smoke.py` — the CPU end-to-end test

This is the **first thing to run on any new machine**, before downloading a single model. It's a "smoke test": a fast, minimal check that the whole training path is wired up and nothing is on fire. It needs no GPU and no network.

```python
def test_training_step_decreases_loss():
    torch.manual_seed(0)
    vocab, T, B = 512, 32, 2
    model = build_tiny(vocab_size=vocab)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ids = torch.randint(0, vocab, (B, T))
    labels = ids.clone()
    labels[:, : T // 2] = -100  # mask the "prompt" half
    ...
    for _ in range(5):
        logits, layer_outputs = model(ids)
        assert logits.shape == (B, T, vocab)
        assert len(layer_outputs) == len(model.blocks)
        loss = masked_lm_loss(logits, labels)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        assert model.embed.weight.grad is not None  # gradients flow
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]  # the loop actually learns on a fixed batch
```

What it actually verifies, all on a tiny randomly-initialized ChronoGPT (`build_tiny`) trained on fake random token data:

- **The model runs forward** and returns logits of the right shape `(B, T, vocab)` and one hidden-state output per transformer block.
- **The loss masking works** — the "prompt" half of the labels is set to `-100` (the ignore index), mirroring how real SFT masks the instruction so only the response contributes to the loss (`05-train.md`). The loss is finite (no NaN/Inf).
- **Gradients flow** — after `loss.backward()`, the embedding weight has a gradient (`grad is not None`), proving backprop reaches the parameters.
- **The optimizer learns** — over 5 steps on a *fixed* batch the loss strictly decreases (`losses[-1] < losses[0]`). If the loss didn't drop, something in the model/loss/optimizer chain is broken.

Because it's pure CPU and instantaneous (~5s, no download), it catches a broken install or environment *before* you spend money on a GPU pulling multi-GB checkpoints. Run it with `pytest -q` (needs the `dev` extra).

---

## 6. `.gitignore` — the GitHub-vs-HuggingFace artifact split

`.gitignore` tells git which files to *never* track. Here it encodes a deliberate policy: **code, logs, and figures live in GitHub; model weights live in Hugging Face.** Git stays small and fast; the multi-GB weights go to the platform built for them.

```
# Data, model caches, run outputs — never commit weights
cache/
runs/
logs*/
out/
wandb/
*.bin
*.pt
*.png
configs/_vintage_*.yaml

# published results ARE tracked (small: loss logs + figures); weights never are
!results/
!results/**/*.png
!results/**/*.csv

# superseded standalone runbook — the "why" now lives in lambda_setup.sh comments
docs/env-setup.md
```

- **Never committed:** `cache/` and `runs/` (training outputs and the packed-data cache), `logs*/`, `out/` (AlpacaEval generations), `wandb/`, and — most importantly — weight files `*.bin` / `*.pt`. Also `*.png` globally and the disposable generated `configs/_vintage_*.yaml`.
- **The exception, re-included with `!`:** `results/` *is* tracked, and within it `*.png` and `*.csv` are re-allowed (the `!` un-ignores them, overriding the global `*.png` ignore). So the small loss logs and figures under `results/` are versioned in GitHub even though PNGs are ignored everywhere else.
- `docs/env-setup.md` is ignored because it was a superseded standalone runbook — its "why" now lives in the `lambda_setup.sh` comments.

**Why split GitHub vs HF.** Git is terrible at large binary files (every version is stored forever, bloating the repo). A single vintage's weights are gigabytes; six vintages would make the repo unusable. So weights go to the Hugging Face Hub (designed for model storage, via `chrono push` or the `push_to_hub` config block), and GitHub keeps only the human-readable record: code, the loss curves (`metrics.csv`), the figures, and the run summaries. This is the same split enforced by `publish_results.sh` (§3.6) and `results/README.md`.

---

## 7. `results/` — what a published run looks like

`results/` is the GitHub-tracked record of finished runs, one subfolder per run. The repo currently contains one real run, `results/chrono-instruct-2020/`, produced by `publish_results.sh`. It holds four small files:

- **`metrics.csv`** — the per-step log (224 rows for this run). The columns are `elapsed_s, stage, epoch, step, split, loss, ppl, lr, grad_norm, tokens_per_sec, gpu_mem_gb`. `split` is `train` or `val`; train rows carry throughput/LR/grad-norm, val rows carry perplexity (`ppl`). This is the exact source data behind Figures 1–2. The first few rows:
  ```
  elapsed_s,stage,epoch,step,split,loss,ppl,lr,grad_norm,tokens_per_sec,gpu_mem_gb
  277.4,stage1_scratch,0,0,train,1.8561,,3e-05,3.713,12879,41.9
  277.4,stage1_scratch,0,1,val,1.5329,4.63,,,,
  ```
- **`config.yaml`** — the *resolved* config the run actually used. Note this published run used `epochs: 3 / 1 / 1` (the under-trained first run discussed in §1), and W&B + push_to_hub were enabled. Keeping the resolved config makes the run reproducible.
- **`summary.json`** — the headline numbers:
  ```json
  {
    "final_val_loss": {"stage1_scratch": 1.4099, "stage2_self_instruct": 1.3344, "stage3_tulu": 1.0306},
    "peak_gpu_gb": 47.5,
    "seed": 123, "block_size": 1792, "batch_size": 8, "grad_accum": 4, "grad_checkpoint": true,
    "elapsed_s": 21610.0
  }
  ```
  So: final validation loss fell stage-over-stage to ~1.03, peak GPU memory was 47.5 GB (comfortably inside 80 GB — confirming the gradient-checkpointing memory plan), and the whole run took ~21,610 s ≈ 6 hours. The downward but still-improving val loss is the empirical evidence that more epochs are warranted (§1).
- **`figure1.png`** — the rendered loss curve (train + val vs. step), the visual analog of `metrics.csv`.

This is the template every future vintage follows: a compact, versioned, browsable summary you could show Manela without sending him gigabytes of weights.

---

## 8. From zero to a trained vintage — operational checklist

A condensed stitch-together of the configs and scripts above. The authoritative, fully-explained version (with tmux, OOM notes, and the verification notebook) is `docs/running-guide.md`; this is the map.

1. **Provision** a Lambda box with one 80GB GPU (A100/H100). One process, one card.
2. **Clone + set `PERSIST`:** `git clone ...`, `cd`, `export PERSIST=/home/ubuntu/persist`.
3. **Setup:** `bash scripts/lambda_setup.sh` (stable cu126 torch + core deps, HF cache on `$PERSIST`), then `source $PERSIST/venv/bin/activate`, then `pip install -e '.[dev,viz,eval,nb]'` for all extras.
4. **Authenticate HF:** `hf auth login` (paste a **Write** token — needed to push weights later). The token caches under `$HF_HOME` on the persistent FS.
5. **Smoke test:** `pytest -q` — the CPU end-to-end test (§5). Must pass before spending GPU time. Optionally run the verification notebook to confirm the ~425,119 screen count and logit parity.
6. **Configure:** edit `configs/train.yaml` — set `model_repo` (the base vintage), `output_dir` (absolute path on `$PERSIST`), `min_confidence: 10`, and optionally `wandb.enabled` / `push_to_hub`.
7. **Dry-run first (cheap):** copy `train.yaml`, delete Stage 2 and Stage 3 from `stages:` (keep only `stage1_scratch`, ~1,097 examples), and `chrono train --config configs/_dryrun.yaml`. Confirm loss drops, `metrics.csv` appears, `stage1_scratch/` + `final/` are saved — this validates the whole loop in minutes.
8. **Train (stage-by-stage recommended):** because each stage resumes from the previous stage's weights, you can train Stage 1 alone, inspect it, then resume Stages 2–3:
   - *Phase 1* — a config with only `stage1_scratch`; run it, then `chrono figure --kind 1 --run <output_dir>` and ask: is val still dropping at the last epoch? If yes, raise Stage 1's `epochs` and re-run (delete `metrics.csv` first to start the curve clean).
   - *Phase 2* — a config with only Stages 2–3, with `model_repo` pointed at the saved `.../stage1_scratch` checkpoint (local dir = resume point) and the **same `output_dir`** so `metrics.csv` accumulates into one combined Figure 1.
   Or, once tuned, just run the full curriculum: `chrono train --config configs/train.yaml`.
9. **Figures + Table 2:** `chrono figure --kind 1 --run <output_dir>`; `chrono eval --repo <model> --cutoff <year>`. For Figure 3, follow the `configs/eval.yaml` pipeline (§2).
10. **Publish:** logs + figures to GitHub via `bash scripts/publish_results.sh <run_dir> <name>`; weights to HF via `chrono push ... --private` (or `push_to_hub.enabled: true`). For the whole sweep, `bash scripts/train_all_vintages.sh` does steps 6–10 per vintage automatically (cache built once).

---

## 9. Mini-FAQ

**Q: Why pin torch to the cu126 *stable* build instead of the upstream nightly?**
The upstream ChronoGPT repo pins a nightly (daily, unreleased) torch, which is unstable. `lambda_setup.sh` installs stable `torch==2.7.0` from the cu126 index, a wheel tested on H100 80GB. It's installed *first* (before `pip install -e .`) so the GPU build wins over PyPI's default resolution. The version matches the pin in `pyproject.toml`; the script only supplies the cu126 *index URL* (switch to `cu124` if your driver is CUDA 12.4).

**Q: What is the "effective batch size" and how do I keep it constant?**
The optimizer updates the weights once per `grad_accum` batches, so effective batch = `batch_size × grad_accum` = `8 × 4 = 32` blocks. If you must lower `batch_size` to avoid an out-of-memory crash, raise `grad_accum` to keep the product at 32 (e.g. `batch_size: 2`, `grad_accum: 16`). The training dynamics depend on the *effective* batch, not on either factor alone.

**Q: What goes to GitHub vs. Hugging Face?**
GitHub holds code + small artifacts (`metrics.csv`, figures, `summary.json`, `config.yaml`) under `results/`, pushed by `publish_results.sh`. Hugging Face holds the multi-GB weights, pushed by `chrono push` or `push_to_hub`. `.gitignore` enforces this: `*.bin`/`*.pt`/`runs/` are never committed; `results/**/*.csv` and `*.png` are explicitly re-included. Git stays small; binaries live where they belong.

**Q: Persistent vs. ephemeral disk — what's the difference and what must I put where?**
A cloud instance's *local* disk is **ephemeral**: wiped when the instance is terminated. Lambda's **persistent** filesystem (mounted at `$PERSIST`) survives. Put the venv, `$HF_HOME` (model/dataset cache), and every run's `output_dir` on `$PERSIST` — otherwise a teardown loses hours of downloads and training. The `running-guide.md` "Ephemeral disk" gotcha is the one most likely to bite you.

**Q: How is the data cache reused across vintages?**
The temporal screen is a single conservative filter — keep only GPT-4.1-labeled "pre-2000" pairs at confidence 10 — and pre-2000 data is before the cutoff for *every* vintage τ ≥ 1999. So the filtered+packed corpus is built once into `cache_dir` and reused by every vintage; only `model_repo`/`output_dir`/`repo_id` change per run. That's why a six-vintage sweep pays the data-prep cost just once, and why `train_all_vintages.sh` is efficient.

**Q: How do I do a cheap dry-run before committing to a multi-hour run?**
Copy `configs/train.yaml`, delete Stage 2 and Stage 3 from `stages:` so only `stage1_scratch` (~1,097 examples) runs, and `chrono train --config configs/_dryrun.yaml`. In a few minutes you confirm loss decreases, `metrics.csv` is written to `output_dir`, and both `stage1_scratch/` and `final/` checkpoints save — i.e. the full pipeline (data → train → save) works end to end before you spend on the real run. Pair this with `pytest -q` (CPU, ~5s) as the even-cheaper first check.
