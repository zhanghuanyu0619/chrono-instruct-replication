#!/usr/bin/env bash
#SBATCH --job-name=chrono-instruct
#SBATCH --array=0-5            # one task per vintage
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a.out
# Same entrypoint as everything else; SLURM just picks the vintage by array index.
set -euo pipefail

VINTAGES=(1999 2005 2010 2015 2020 2024)
year="${VINTAGES[$SLURM_ARRAY_TASK_ID]}"

cfg="configs/_vintage_${year}.yaml"
sed "s|chrono-gpt-v1-2020|chrono-gpt-v1-${year}|g; s|chrono-instruct-2020|chrono-instruct-${year}|g" \
  configs/train.yaml > "$cfg"

srun chrono train --config "$cfg"
