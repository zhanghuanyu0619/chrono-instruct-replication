#!/usr/bin/env bash
# Fan out one job per local GPU: one vintage per GPU on a single multi-GPU box.
# Usage: bash scripts/launch_local.sh 1999 2005 2010 ...
set -euo pipefail

gpu=0
for year in "$@"; do
  cfg="configs/_vintage_${year}.yaml"
  sed "s|chrono-gpt-v1-2020|chrono-gpt-v1-${year}|g; s|chrono-instruct-2020|chrono-instruct-${year}|g" \
    configs/train.yaml > "$cfg"
  echo "GPU $gpu -> vintage $year"
  CUDA_VISIBLE_DEVICES=$gpu chrono train --config "$cfg" > "logs_${year}.txt" 2>&1 &
  gpu=$((gpu + 1))
done
wait
echo "All vintages done."
