#!/usr/bin/env bash
#SBATCH --job-name=chrono-instruct
#SBATCH --array=0-5            # one task per vintage
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a.out
# Same entrypoint as everything else; SLURM just picks the vintage by array index.
set -euo pipefail

PERSIST="${PERSIST:-$HOME/persist}"
HF_USER="${HF_USER:-HZ0619}"
BASE_CFG="${BASE_CFG:-configs/train.yaml}"

VINTAGES=(1999 2005 2010 2015 2020 2024)
year="${VINTAGES[$SLURM_ARRAY_TASK_ID]}"

cfg="configs/_vintage_${year}.yaml"
out="$PERSIST/runs/chrono-instruct-${year}"
python scripts/make_vintage_config.py --base "$BASE_CFG" --out "$cfg" \
  --cutoff "${year}1231" --output-dir "$out" --hf-user "$HF_USER"

srun chrono train --config "$cfg"
