# Running Guide — Lambda Labs, end to end

Full workflow for one vintage: provision → setup → verify → train → figures →
publish results to GitHub → push checkpoints to Hugging Face → inference. §2 sets
up the environment via `scripts/lambda_setup.sh`; `env-setup.md` is the background
runbook explaining *why* (stable `cu126` torch, not the upstream nightly).

> **Division of artifacts:** loss logs + figures go to **GitHub** (small, in
> `results/`); model checkpoints go to the **Hugging Face Hub** (large). Weights
> are never committed to git.

---

## 1. Provision

A single GPU is enough — the 1.55B model is one process on one card.
- **80GB (A100/H100 80GB):** trains at the config's `batch_size: 8`.
- **40GB (A100 40GB):** full fine-tuning is tight; see the OOM note in §6.

Attach a **persistent filesystem** and mount it (e.g. `/home/ubuntu/persist`).
Lambda's instance disk is wiped on termination — checkpoints and the HF cache
must live on the persistent mount.

## 2. One-time setup

**Start tmux FIRST**, then do everything inside it — so the venv activation and
env vars live in the persistent session (a dropped SSH connection won't kill a
multi-hour run, and re-attaching keeps the venv active):
```bash
tmux new -s chrono     # detach: Ctrl-b d   |   reattach: tmux attach -t chrono
```

Then, **inside tmux**:
```bash
git clone https://github.com/zhanghuanyu0619/chrono-instruct-replication.git
cd chrono-instruct-replication
export PERSIST=/home/ubuntu/persist
bash scripts/lambda_setup.sh                 # stable cu126 torch + deps, HF cache on $PERSIST
source $PERSIST/venv/bin/activate            # literal path; $PERSIST must be exported in THIS shell
pip install -e '.[dev,viz,eval]'             # tests, figures/W&B, AlpacaEval
pip install jupyterlab ipykernel             # for the verification notebook — INTO the venv
```

> Do NOT activate the venv *before* `tmux new` — a fresh tmux shell won't inherit
> it. Activate inside tmux (as above); after `tmux attach` the activation persists.

## 3. Authenticate with Hugging Face

Needed for higher dataset download rate limits and for pushing checkpoints:
```bash
hf auth login          # paste a WRITE token (write = required for pushing models)
```
The token is cached under `$HF_HOME` (on the persistent FS). Never hardcode it.

## 4. Verify before training

```bash
pytest -q                                    # CPU smoke test, ~5s

# Register the venv as a Jupyter kernel (jupyterlab + ipykernel installed in §2).
# --user makes it visible to ANY JupyterLab, including Lambda's hosted one.
python -m ipykernel install --user --name chrono --display-name "Python (chrono)"
jupyter lab notebooks/verify_pipeline.ipynb  # or use Lambda's hosted JupyterLab
```
Select the **Python (chrono)** kernel (top-right / Kernel → Change Kernel) before
running. Verify you're on the right one with a quick cell:
```python
import sys; print(sys.executable)            # must be /home/ubuntu/persist/venv/bin/python
```
If it shows a different path (or `tiktoken` ImportError), the kernel points at the
wrong Python — re-run the `ipykernel install` line above *with the venv active*.

Confirm in the notebook: screen total ≈ 425,119 (§4/§4b), param dtype (§9),
logit parity ≈ 0 vs the official file (§10), and peak VRAM (§13).

## 5. Configure the run

Edit `configs/train.yaml`:
- `model_repo` — the base vintage (e.g. `manelalab/chrono-gpt-v1-20201231`).
- `output_dir` — an **absolute path on the persistent FS**, e.g.
  `/home/ubuntu/persist/runs/chrono-instruct-2020`.
- `min_confidence` — `10` (paper's strict screen) or `null` (keep all label-0).
- `wandb.enabled` — `true` to mirror loss curves live (optional).
- `push_to_hub` — `enabled: true` + your `repo_id` to auto-push `final/` to the
  Hub when training ends (or push manually later, §8).

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

> **Memory (verified).** Full FT is activation-heavy, not weight-heavy (~25 GB is
> just Adam states). With `grad_checkpoint: true` (default in the config) plus
> `return_hidden=False` during training, `batch_size: 8` fits one **80GB** card.
> Without checkpointing, batch 8 OOMs even on 80GB; a **40GB** card OOMs at batch 1
> regardless. If you still OOM: lower `batch_size`, raise `grad_accum` to keep the
> effective batch (e.g. `2`/`16` = 32). Gradient checkpointing costs ~20% step time.

## 7. Figures + Table 2

```bash
chrono figure --kind 1 --run /home/ubuntu/persist/runs/chrono-instruct-2020   # Fig 1
chrono eval --repo manelalab/chrono-gpt-instruct-v1-20201231 --cutoff 2020     # Table 2
```
For the vintage sweep: `chrono figure --kind 2 --runs /home/ubuntu/persist/runs/chrono-instruct-*`.
For Figure 3 (AlpacaEval), see `configs/eval.yaml`.

## 8. Publish results to GitHub + checkpoints to Hugging Face

**Logs + figures → GitHub** (small, tracked under `results/`):
```bash
bash scripts/publish_results.sh /home/ubuntu/persist/runs/chrono-instruct-2020 chrono-instruct-2020
```
This copies `metrics.csv`, renders `results/chrono-instruct-2020/figure1.png`, and pushes.

**Checkpoints → Hugging Face Hub** (large weights). `<hf-user>` must be your HF
namespace from `hf auth whoami` (NOT your GitHub username), with a **Write** token:
```bash
chrono push --repo /home/ubuntu/persist/runs/chrono-instruct-2020/final \
            --to <hf-user>/chrono-instruct-v1-20201231 --private
```
(or set `push_to_hub.enabled: true` in the config to do this automatically at the
end of training.)

## 9. Inference + clearing GPU memory

```bash
python scripts/inference_demo.py --repo manelalab/chrono-gpt-instruct-v1-19991231
# or your own: --repo /home/ubuntu/persist/runs/chrono-instruct-2020/final
```
It generates, embeds, then `del model; free_memory()` and reports VRAM. In a
notebook/REPL, free the GPU between model loads with:
```python
from chrono_instruct.infer import free_memory
del model
free_memory()   # gc.collect() + torch.cuda.empty_cache()
```

## 10. Sweep across vintages

One run = one `(vintage, config)`. The filtered+packed data is cached and reused,
so only `model_repo`/`output_dir`/`repo_id` change.

**Sequential, one GPU** (the common case — fine-tune all vintages back-to-back,
each auto-published to GitHub + HF):
```bash
bash scripts/train_all_vintages.sh                 # 1999 2005 2010 2015 2020 2024
bash scripts/train_all_vintages.sh 1999 2020       # or a subset
```
It derives a per-vintage config from `configs/train.yaml` (overriding `model_repo`,
`output_dir`, and the HF `repo_id`), trains, and runs `publish_results.sh`. Set
`HF_USER` / `PERSIST` as env vars if they differ from the defaults. Rough budget:
~3 h/vintage on an 80GB H100 → ~15–24 h for all six (cache built once).

**Parallel, multi-GPU / cluster:** `scripts/launch_local.sh` (one vintage per GPU)
or `scripts/slurm_array.sbatch` (SLURM array).

## Gotchas
- **Shut down notebook kernels before training:** a live JupyterLab kernel keeps
  its models on the GPU (tens of GB), so `chrono train` then OOMs even on an 80GB
  card. `nvidia-smi` to spot the stale PID; Kernel → Shut Down All, or `kill <pid>`.
- **`batch_size` on one card:** full FT of 1.55B is memory-heavy; if you OOM on a
  clean 80GB card, lower `batch_size` and raise `grad_accum` to keep the effective
  batch (e.g. `2`/`16` = 32). Watch `nvidia-smi` for the real peak.
- **tmux loses your venv:** activate the venv *inside* tmux, not before — a fresh
  tmux shell doesn't inherit activation. Symptom: `No module named ipykernel` /
  wrong-Python kernel / `tiktoken` not found.
- **Wrong notebook kernel:** if `sys.executable` isn't the venv path, re-run
  `python -m ipykernel install --user --name chrono` with the venv active.
- **Ephemeral disk:** keep `output_dir`, `$HF_HOME`, and the venv on `$PERSIST`.
- **`$PERSIST` not set:** `source $PERSIST/...` fails silently if it expanded to
  empty — `export PERSIST=...` first, or use the literal path.
- **No nightly torch:** install stable `torch==2.7.0+cu126` (lambda_setup does this).
- **`pytest` missing:** it's in the `dev` extra (`pip install -e '.[dev]'`).
