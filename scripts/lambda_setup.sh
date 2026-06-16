#!/usr/bin/env bash
# Bootstrap a fresh Lambda Labs instance. Run once per new instance.
# Put data/cache/checkpoints on a PERSISTENT filesystem so they survive
# instance termination (Lambda's local disk does not).
set -euo pipefail

PERSIST="${PERSIST:-/home/ubuntu/persist}"   # mount your persistent filesystem here
mkdir -p "$PERSIST/hf-cache" "$PERSIST/runs"

# Caches on the persistent FS.
export HF_HOME="$PERSIST/hf-cache"

python3 -m venv "$PERSIST/venv"
# shellcheck disable=SC1091
source "$PERSIST/venv/bin/activate"
pip install --upgrade pip
pip install -e .            # installs torch, datasets, tiktoken, etc. from pyproject

echo "Setup done. Activate with: source $PERSIST/venv/bin/activate"
echo "Then run the smoke test:   pytest -q"
