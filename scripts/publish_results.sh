#!/usr/bin/env bash
# Publish a run's loss logs + figures to GitHub, under results/<name>/.
# Weights are NOT published here — those go to the HF Hub (chrono push, or the
# push_to_hub block in configs/train.yaml). This keeps git small.
#
#   bash scripts/publish_results.sh /home/ubuntu/persist/runs/chrono-instruct-2020 chrono-instruct-2020
set -euo pipefail

RUN_DIR="$1"        # the training output_dir (contains metrics.csv + checkpoints)
NAME="$2"           # a short results label, e.g. chrono-instruct-2020
DEST="results/$NAME"

mkdir -p "$DEST"
cp "$RUN_DIR/metrics.csv" "$DEST/metrics.csv"
cp "$RUN_DIR/config.yaml"  "$DEST/" 2>/dev/null || true   # resolved run config (reproducibility)
cp "$RUN_DIR/summary.json" "$DEST/" 2>/dev/null || true   # final val loss, peak VRAM, throughput
chrono figure --kind 1 --run "$RUN_DIR" --out "$DEST/figure1.png"

git add "$DEST"
git commit -m "results: $NAME (loss curves + metrics + summary)"
git push origin main
echo "published $DEST -> GitHub"
