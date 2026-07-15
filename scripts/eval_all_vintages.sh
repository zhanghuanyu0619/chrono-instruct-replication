#!/usr/bin/env bash
# Run the evaluation exhibits (Table 2, Table 3, optionally Figure 3) across ALL
# vintages and collect the results — the eval counterpart of train_all_vintages.sh.
# Results are saved per-vintage (results/chrono-instruct-<Y>/eval.json) and then
# aggregated into results/replication-report/{eval_results.json, eval_summary.md},
# so you never collect anything by hand.
#
# By DEFAULT it evaluates the HF-published models (HZ0619/chrono-instruct-v1-<Y>1231);
# pass --repo-dir to use local run dirs instead.
#
# Examples:
#   bash scripts/eval_all_vintages.sh                     # Tables 2-3, all 6 vintages, from HF
#   bash scripts/eval_all_vintages.sh 2020 2024           # just these two
#   ALPACA=1 bash scripts/eval_all_vintages.sh            # also Figure 3 (needs OPENAI_API_KEY)
#   REPO_DIR=/home/ubuntu/persist/runs bash scripts/eval_all_vintages.sh   # eval local checkpoints
set -euo pipefail

HF_USER="${HF_USER:-HZ0619}"
ALPACA="${ALPACA:-0}"                 # set ALPACA=1 to also run Figure 3 (AlpacaEval)
REPO_DIR="${REPO_DIR:-}"              # if set, eval <REPO_DIR>/chrono-instruct-<Y>/final instead of HF
REF="${REF:-results/qwen/qwen.json}" # AlpacaEval reference (Qwen), generated once (tracked)
YEARS=("$@")
[ ${#YEARS[@]} -eq 0 ] && YEARS=(1999 2005 2010 2015 2020 2024)

# Figure 3 needs a single shared reference (Qwen outputs) — generate it once.
if [ "$ALPACA" = "1" ] && [ ! -f "$REF" ]; then
    echo "generating AlpacaEval reference (Qwen-1.5-1.8B-Chat) once -> $REF"
    mkdir -p "$(dirname "$REF")"
    chrono alpaca --backend hf --repo Qwen/Qwen1.5-1.8B-Chat --name qwen --out "$REF"
fi

OK=() ; FAILED=()
for Y in "${YEARS[@]}"; do
    echo "==================  eval chrono-instruct-${Y}  =================="
    ARGS=(--vintage "$Y" --hf-user "$HF_USER")
    [ -n "$REPO_DIR" ] && ARGS+=(--repo "$REPO_DIR/chrono-instruct-${Y}/final")
    [ "$ALPACA" = "1" ] && ARGS+=(--alpaca --alpaca-reference "$REF")
    if python scripts/run_eval.py "${ARGS[@]}"; then
        OK+=("$Y")
    else
        echo "ERROR: eval chrono-instruct-${Y} failed (repo missing? auth? OOM?) — continuing"
        FAILED+=("$Y")
    fi
done

# Collect everything the report needs, then publish to GitHub (best-effort).
python scripts/aggregate_eval.py || echo "WARN: aggregate_eval failed (continuing)"
git add "results/chrono-instruct-"*/eval.json \
        "results/chrono-instruct-"*/alpaca_*.json \
        results/qwen/qwen.json \
        results/replication-report/eval_results.json \
        results/replication-report/eval_summary.md 2>/dev/null \
    && git commit -q -m "results: evaluation sweep (Tables 2-3$([ "$ALPACA" = "1" ] && echo ', Figure 3'))" 2>/dev/null \
    && git push -q 2>/dev/null && echo "published eval results" \
    || echo "WARN: eval results saved but not pushed (continuing)"

echo "==================  eval sweep done  =================="
echo "  evaluated: ${OK[*]:-none}"
echo "  failed:    ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
