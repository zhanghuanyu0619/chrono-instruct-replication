#!/usr/bin/env bash
# Fan out one job per local GPU: one vintage per GPU on a single multi-GPU box.
# Usage: bash scripts/launch_local.sh 1999 2005 2010 ...
# Set HF_USER / PERSIST / BASE_CFG as env vars if they differ from the defaults.
set -euo pipefail

PERSIST="${PERSIST:-$HOME/persist}"
HF_USER="${HF_USER:-HZ0619}"
BASE_CFG="${BASE_CFG:-configs/train.yaml}"

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
echo "All vintages done."
