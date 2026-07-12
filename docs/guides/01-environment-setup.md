# 01 — Environment setup (before training)

Everything needed to get a working Lambda box **before** you train: provision,
one-time setup, HF + GitHub auth, verify, and tmux for long runs. Next:
[02-training-the-models.md](02-training-the-models.md).

## 1. Provision

A single GPU is enough — the 1.55B model is one process on one card.
- **80GB (A100/H100 80GB):** trains at the config's `batch_size: 8`.
- **40GB (A100 40GB):** full fine-tuning is tight; see the OOM note in
  [02-training-the-models.md §Memory](02-training-the-models.md#memory-verified).

## 2. One-time setup

For multi-hour runs over a local SSH session, start inside tmux first (see
[§7 tmux](#7-tmux-for-long-runs-over-local-ssh)); Lambda's web terminal / hosted
JupyterLab keep the session alive server-side, so tmux is optional there.

```bash
git clone https://github.com/zhanghuanyu0619/chrono-instruct-replication.git
cd chrono-instruct-replication
export PERSIST=/home/ubuntu/persist
bash scripts/lambda_setup.sh                 # stable cu126 torch + deps, HF cache on $PERSIST
source $PERSIST/venv/bin/activate            # literal path; $PERSIST must be exported in THIS shell
pip install -e '.[dev,viz,eval,nb]'          # tests, figures/W&B, AlpacaEval, notebook tooling
```

`lambda_setup.sh` installs `torch==2.7.0` from the cu126 index first (so that build
wins over PyPI's default), then `pip install -e .` (core deps from `pyproject.toml`).
All optional tooling lives in extras: `dev,viz,eval,nb`.

## 3. Authenticate with Hugging Face

For higher download rate limits and for pushing checkpoints:
```bash
hf auth login          # paste a WRITE token (required for pushing models)
```
Cached under `$HF_HOME` (persistent FS). Never hardcode it.

## 3b. GitHub identity + credentials (for publishing results)

Publishing loss logs/figures back to GitHub (see
[02 §Publish](02-training-the-models.md#publish-results-to-github--checkpoints-to-hugging-face),
and automatically during the sweep) needs a git **identity** and
**non-interactive push credentials** on the box — set these once or the first
`git commit`/`git push` will fail:

```bash
git config --global user.name  "Huanyu Zhang"           # both name AND email, or commits fail
git config --global user.email "zhanghuanyu0619@gmail.com"
git config --global credential.helper store             # cache the PAT after the next push
```

Then do one `git push` and enter your username + a **fine-grained PAT** (Settings →
Developer settings → Fine-grained tokens, scoped to this repo with **Contents:
read/write**) as the password. `credential.helper store` caches it to
`~/.git-credentials`, so every later push — including the sweep's per-vintage
publishes — runs silently. **This matters for the sweep:** an interactive password
prompt inside an unattended run would hang it.

If the box's `main` has fallen behind `origin/main` (e.g. code was pushed while you
trained), integrate before pushing: `git pull --rebase origin main`.

## 4. Verify before training

```bash
pytest -q                                    # CPU smoke test, ~5s

# Register the venv as a Jupyter kernel. --user makes it visible to ANY
# JupyterLab, including Lambda's hosted one.
pip install ipykernel
python -m ipykernel install --user --name chrono --display-name "Python (chrono)"
jupyter lab notebooks/verify_pipeline.ipynb  # or Lambda's hosted JupyterLab
```
Select the **Python (chrono)** kernel before running, and confirm it points at the
venv:
```python
import sys; print(sys.executable)            # must be /home/ubuntu/persist/venv/bin/python
```
Wrong path (or `tiktoken` ImportError) → re-run the `ipykernel install` line *with
the venv active*.

Confirm in the notebook: screen total ≈ 425,119, param dtype, logit parity ≈ 0 vs
the official file, and peak VRAM.

## 7. tmux (for long runs over local SSH)

Only needed when you drive the box from a **local terminal over SSH** — a dropped
connection would otherwise kill the job. Skip it if you use Lambda's web terminal
or hosted JupyterLab (those persist server-side). tmux runs on the **remote**
instance; if missing, `sudo apt-get install -y tmux`.

```bash
tmux new -s chrono     # detach: Ctrl-b d   |   reattach: tmux attach -t chrono
```
Run the §2 setup (and everything after) **inside** the tmux session. Activate the
venv *inside* tmux, not before `tmux new` — a fresh tmux shell won't inherit it.
After `tmux attach`, the activation persists.

For a fully hands-off sweep on Lambda's browser terminal / JupyterLab (no tmux),
detach with `nohup` — see
[02 §Sweep](02-training-the-models.md#sweep-across-vintages).

## Gotchas (environment)

- **tmux loses your venv:** activate the venv *inside* tmux, not before (§7).
  Symptom: `No module named ipykernel` / wrong-Python kernel / `tiktoken` not found.
- **Wrong notebook kernel:** if `sys.executable` isn't the venv path, re-run
  `python -m ipykernel install --user --name chrono` with the venv active.
- **Ephemeral disk:** keep `output_dir`, `$HF_HOME`, and the venv on `$PERSIST`.
- **`$PERSIST` not set:** `source $PERSIST/...` fails silently if it expanded to
  empty — `export PERSIST=...` first, or use the literal path.
- **No nightly torch:** install stable `torch==2.7.0+cu126` (lambda_setup does this).
- **`pytest` missing:** it's in the `dev` extra (`pip install -e '.[dev]'`).
