#!/usr/bin/env bash
# Bootstrap a fresh Lambda Labs instance for training.
# Mirrors the proven inference runbook (docs/env-setup.md): stable cu126 torch,
# NOT the nightly the upstream repo pins. Run once per new instance.
#
# Put data/cache/checkpoints on a PERSISTENT filesystem so they survive instance
# termination (Lambda's local disk does not).
set -euo pipefail

PERSIST="${PERSIST:-$HOME/persist}"     # mount your persistent filesystem here
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"  # use cu124 if driver is CUDA 12.4
mkdir -p "$PERSIST/hf-cache" "$PERSIST/runs"
export HF_HOME="$PERSIST/hf-cache"      # large downloads land on the persistent FS

# Python 3.10 is fine; no need for 3.11.
python3 -m venv "$PERSIST/venv"
# shellcheck disable=SC1091
source "$PERSIST/venv/bin/activate"
python -m pip install --upgrade pip

# Stable torch built for CUDA 12.6 (proven on H100 PCIe 80GB), installed BEFORE
# the package so this wheel wins over PyPI's default resolution.
pip install torch==2.7.0 --index-url "$CUDA_INDEX"

# Everything else (datasets, tiktoken, huggingface-hub, pyyaml, numpy, ...).
pip install -e .

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "Setup done. Activate with: source $PERSIST/venv/bin/activate"
echo "Then: pytest -q   (smoke test, no download)   and   chrono inspect"
