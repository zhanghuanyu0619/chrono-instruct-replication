#!/usr/bin/env bash
# Sequentially fine-tune ChronoGPT-Instruct for several vintages on ONE GPU.
# The filtered+packed data is built once and cached, so every vintage after the
# first skips data prep; only model_repo / output_dir / repo_id change.
#
# Run inside the venv (and tmux). Examples:
#   bash scripts/train_all_vintages.sh                       # default 6 paper vintages
#   bash scripts/train_all_vintages.sh 1999 2020             # just these two
#   HF_USER=HZ0619 PERSIST=/home/ubuntu/persist bash scripts/train_all_vintages.sh
#
# Checkpoints go to the HF Hub if push_to_hub.enabled is true in the base config
# (repo_id is overridden per vintage). Loss logs + figures go to GitHub via
# publish_results.sh after each run. Enable wandb (config default) for live curves
# and per-run finished notifications.
set -euo pipefail

PERSIST="${PERSIST:-$HOME/persist}"
HF_USER="${HF_USER:-HZ0619}"
BASE_CFG="${BASE_CFG:-configs/train.yaml}"
YEARS=("$@")
[ ${#YEARS[@]} -eq 0 ] && YEARS=(1999 2005 2010 2015 2020 2024)

OK=() ; FAILED=()
for Y in "${YEARS[@]}"; do
    CUTOFF="${Y}1231"
    NAME="chrono-instruct-${Y}"
    CFG="configs/_vintage_${Y}.yaml"          # gitignored (configs/_vintage_*.yaml)
    OUT="$PERSIST/runs/$NAME"
    echo "==================  $NAME  (base manelalab/chrono-gpt-v1-$CUTOFF)  =================="

    # Derive a per-vintage config from the base (robust YAML edit, not sed). A
    # failure here (or in training) logs and moves to the next vintage rather than
    # aborting the whole unattended sweep.
    if python scripts/make_vintage_config.py --base "$BASE_CFG" --out "$CFG" \
            --cutoff "$CUTOFF" --output-dir "$OUT" --hf-user "$HF_USER" \
       && chrono train --config "$CFG"; then
        bash scripts/publish_results.sh "$OUT" "$NAME" || echo "WARN: publish failed for $NAME (continuing)"
        OK+=("$Y")
    else
        echo "ERROR: $NAME failed (base repo missing? OOM? auth?) — continuing to next vintage"
        FAILED+=("$Y")
    fi
done

echo "==================  sweep done  =================="
echo "  trained: ${OK[*]:-none}"
echo "  failed:  ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]   # exit non-zero if any vintage failed, so CI/`&&` chains notice
