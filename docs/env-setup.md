# ChronoGPT Environment Setup (Lambda / H100)

A reproducible runbook for running ChronoGPT inference (text generation + embedding
extraction) in an isolated `venv`, with a JupyterLab kernel.

Tested on: Lambda Cloud instance, **NVIDIA H100 PCIe 80GB**, Ubuntu, driver
`580.105.08` (CUDA 13.0 capable), system **Python 3.10**.

---

## TL;DR

```bash
# 1. Isolated env from the system Python (3.10 is fine; no need for 3.11)
python3 -m venv ~/chronogpt-env
source ~/chronogpt-env/bin/activate
python -m pip install --upgrade pip

# 2. STABLE torch built for CUDA 12.6 (NOT the nightly the repo pins)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# 3. Everything else the tutorial actually needs
pip install "huggingface_hub>=0.24,<2" "tiktoken>=0.7" numpy ipykernel

# 4. Register as a Jupyter kernel
python -m ipykernel install --user --name chronogpt --display-name "Python (chronogpt)"

# 5. Lock exact versions for replication
pip freeze > ~/chronogpt-env-requirements.lock.txt
```

Then in JupyterLab pick the **Python (chronogpt)** kernel and run the notebook.

---

## Why not `pip install -r requirements.txt`?

The project's `requirements.txt` pins a **nightly** torch:

```
torch==2.7.0.dev20250110+cu126
```

PyTorch only keeps nightlies on its index for a few months and then deletes them, so
that exact build no longer exists anywhere — the install can never succeed as written.
That pin is a leftover from the **training** environment (ChronoGPT's training is built
on modded-nanoGPT, which used nightlies for FlexAttention/Muon).

**Inference needs none of that.** `ChronoGPT_inference.py` only uses
`F.scaled_dot_product_attention` (stable since torch 2.0) plus a few `.bfloat16()` casts.
So any stable torch >= 2.1 with a matching CUDA build runs the tutorial. We pin
**`torch==2.7.0+cu126`** for a clean, reproducible env.

---

## Step-by-step

### 0. Check the GPU / driver

```bash
nvidia-smi
```

Confirm the GPU and note the **CUDA Version** (top-right). On the test box this was
`13.0`, which is forward-compatible with all CUDA 12.x wheels — so the `cu126` wheels
work. If your driver only reports CUDA 12.4, swap the install index below to
`.../whl/cu124`. The torch *version* matters far less than matching the CUDA major line.

### 1. Create and activate the venv

```bash
python3 -m venv ~/chronogpt-env
source ~/chronogpt-env/bin/activate     # prompt should now show (chronogpt-env)
python -m pip install --upgrade pip
```

If `python3 -m venv` errors about `ensurepip` / `python3-venv`, install it once
(you have sudo on Lambda) and rerun:

```bash
sudo apt install -y python3-venv
```

> A venv disables user site-packages by default, so anything you previously installed
> with `pip install --user` is intentionally invisible here. That's what keeps the env
> clean and reproducible.

### 2. Install stable torch for CUDA 12.6

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

This pulls `torch-2.7.0+cu126` (~867 MB) plus the bundled `nvidia-*-cu12` 12.6.x
runtime and `triton`. The `+cu126` suffix is your proof you're on the chosen wheel,
not a system-provided torch.

### 3. Install the remaining dependencies

```bash
pip install "huggingface_hub>=0.24,<2" "tiktoken>=0.7" numpy ipykernel
```

- `huggingface_hub` — model download + `PyTorchModelHubMixin`
- `tiktoken` — GPT-2 tokenizer used by the tutorial
- `numpy` — not used directly by the tutorial, but torch prints a warning on every
  import without it, and you'll need it downstream (pandas/matplotlib/`.numpy()`).
  torch 2.7 is fine with numpy 2.x.
- `ipykernel` — required to expose this venv as a Jupyter kernel

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name chronogpt --display-name "Python (chronogpt)"
```

In JupyterLab: open the notebook → kernel name (top-right) or **Kernel → Change Kernel**
→ **Python (chronogpt)**. Refresh the browser tab if it doesn't appear.

### 5. (Optional) Authenticate with the HF Hub

The ChronoGPT weights are public, so this is **not required** — but it removes anonymous
rate-limit throttling and speeds up the large `pytorch_model.bin` download.

```bash
hf auth login        # or: huggingface-cli login  (older huggingface_hub)
```

Paste a **read** token from <https://huggingface.co/settings/tokens>. It's cached to
`~/.cache/huggingface/` and picked up automatically by the running kernel on the next
download — just re-run the cell.

> **Do not** hardcode the token in a notebook cell (`login(token="hf_...")`). Since
> notebooks go into git, a committed token leaks and gets auto-revoked. The CLI login
> keeps it out of your code. If you need it inline, read from `os.environ["HF_TOKEN"]`.

---

## Verifying the environment

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected:

```
2.7.0+cu126 True NVIDIA H100 PCIe
```

`True` + the device name means the GPU path is live. (If you skipped numpy in step 3
you'll see a `Failed to initialize NumPy` warning here — harmless, but installing numpy
clears it.)

---

## Running the tutorial — two gotchas

1. **Pick the model cutoff.** The tutorial hardcodes
   `repo_id = "manelalab/chrono-gpt-v1-20241231"`. Change it to the cutoff you want,
   e.g. `"manelalab/chrono-gpt-v1-20201231"` (data through 2020-12-31). Same
   architecture, different weights.

2. **`max_length` in the embeddings cell.** That cell references `max_length` but only
   the *text-generation* cell defines it. Run the generation cell first, or add
   `max_length = 30` at the top of the embeddings cell so it stands alone — otherwise
   you get a `NameError`.

---

## Reproducing the environment later

After a successful run you froze exact versions:

```bash
pip freeze > ~/chronogpt-env-requirements.lock.txt
```

To rebuild from scratch (note the extra index so the `+cu126` torch resolves):

```bash
python3 -m venv ~/chronogpt-env && source ~/chronogpt-env/bin/activate
pip install --upgrade pip
pip install -r ~/chronogpt-env-requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

---

## Daily use

```bash
source ~/chronogpt-env/bin/activate   # start working
deactivate                            # leave the env
```

Anything you `pip install` while the env is active is immediately available to the
**Python (chronogpt)** Jupyter kernel.

---

## Troubleshooting (issues actually hit during setup)

| Symptom | Cause | Fix |
|---|---|---|
| `Command 'python3.11' not found` | Box only has Python 3.10 | Use `python3` — 3.10 runs ChronoGPT fine |
| `Defaulting to user installation...` / `Requirement already satisfied: torch` in `/usr/lib/...` | No venv active; pip hit system/user space and saw Lambda's pre-installed torch | Create + activate the venv first; reinstall torch inside it |
| `torch` version has no `+cu126` suffix | You're on the system torch, not your wheel | Install inside the activated venv from the cu126 index |
| `Failed to initialize NumPy: No module named 'numpy'` | numpy not in venv | `pip install numpy` |
| `Unauthenticated requests to the HF Hub... set a HF_TOKEN` | Anonymous rate limits | Optional: `hf auth login` with a read token |
| `NameError: max_length` | Variable defined only in the generation cell | Add `max_length = 30` to the embeddings cell |

---

## Notes

- **venv vs conda:** venv wins here. torch's `+cu126` wheels bundle their own CUDA
  runtime, so conda's CUDA management buys nothing; and as of PyTorch 2.6 the `pytorch`
  conda channel is deprecated (you install via pip even inside conda). `uv` is a faster
  drop-in alternative to venv if you want it (`uv venv --python 3.12`).
- **Hardware headroom:** the H100's 80 GB VRAM dwarfs this GPT-2-scale model, so you can
  raise batch sizes substantially when extracting embeddings over a corpus. For bulk
  fp32 matmuls, `torch.set_float32_matmul_precision("high")` lets Hopper use TF32 tensor
  cores for free throughput.
