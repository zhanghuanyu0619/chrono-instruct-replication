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

[ -f "$RUN_DIR/metrics.csv" ] || { echo "ERROR: $RUN_DIR/metrics.csv not found — is RUN_DIR correct?"; exit 1; }

mkdir -p "$DEST"
cp "$RUN_DIR/metrics.csv" "$DEST/metrics.csv"
cp "$RUN_DIR/config.yaml"  "$DEST/" 2>/dev/null || true   # resolved run config (reproducibility)
cp "$RUN_DIR/summary.json" "$DEST/" 2>/dev/null || true   # final val loss, peak VRAM, throughput

# Figure: reuse the one training already saved (results/<name>/figure1.png via
# save_results), else render now. Non-fatal — a missing `chrono` (venv not active)
# or a matplotlib error must not block publishing the metrics.
if [ -f "$RUN_DIR/figure1.png" ]; then
    cp "$RUN_DIR/figure1.png" "$DEST/figure1.png"
elif [ ! -f "$DEST/figure1.png" ]; then
    chrono figure --kind 1 --run "$RUN_DIR" --out "$DEST/figure1.png" \
        || echo "WARN: figure render failed (activate the venv so 'chrono' is on PATH); publishing metrics anyway"
fi

git add "$DEST"
# Non-fatal: re-publishing an unchanged run, or a box without push creds, shouldn't error out.
git commit -m "results: $NAME (loss curves + metrics + summary)" || echo "note: nothing new to commit for $NAME"
git push origin main || { echo "WARN: git push failed — set up a GitHub PAT on this box, then: git push origin main"; exit 0; }
echo "published $DEST -> GitHub"
