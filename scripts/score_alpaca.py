#!/usr/bin/env python3
"""Score an ALREADY-GENERATED AlpacaEval outputs file — no GPU, no re-generation.

Figure 3's expensive part (generating each vintage's 805 completions) is done and
saved to results/chrono-instruct-<Y>/alpaca_<Y>.json; only the LLM-judge step failed
in the sweep because AlpacaEval's default annotator calls the retired
gpt-4-1106-preview (404 model_not_found). This re-runs JUST the judging with a
currently-available annotator and writes the win-rate back into that vintage's
eval.json (replacing the stale `winrate_error`).

Needs (wherever an OpenAI-family judge key lives — e.g. the training box, not this Mac):
  pip install -e '.[eval]'            # alpaca_eval + openai
  export OPENAI_API_KEY=...           # the judge key
  # a reference outputs JSON (Qwen-1.5-1.8B-Chat), as produced by the sweep:
  #   chrono alpaca --backend hf --repo Qwen/Qwen1.5-1.8B-Chat --name qwen --out out/qwen.json

Examples:
  python scripts/score_alpaca.py                                  # 2015 vs out/qwen.json, default judge
  python scripts/score_alpaca.py --model results/chrono-instruct-2020/alpaca_2020.json
  python scripts/score_alpaca.py --annotator weighted_alpaca_eval_gpt4_turbo_new   # stronger judge
  ALPACA_ANNOTATOR=... python scripts/score_alpaca.py            # env override also honored
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chrono_instruct.eval import DEFAULT_ALPACA_ANNOTATOR  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="results/chrono-instruct-2015/alpaca_2015.json",
                    help="already-generated model outputs JSON to score")
    ap.add_argument("--reference", default="out/qwen.json",
                    help="reference outputs JSON (Qwen-1.5-1.8B-Chat), as the sweep generates")
    ap.add_argument("--annotator", default=None,
                    help=f"alpaca_eval annotators_config (default: $ALPACA_ANNOTATOR or {DEFAULT_ALPACA_ANNOTATOR})")
    ap.add_argument("--eval-json", default=None,
                    help="vintage eval.json to update in place (default: sibling eval.json of --model)")
    ap.add_argument("--no-write", action="store_true", help="print the win-rate but don't touch eval.json")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model outputs not found: {args.model}")
    if not os.path.exists(args.reference):
        sys.exit(f"reference outputs not found: {args.reference}\n"
                 "  generate once: chrono alpaca --backend hf --repo Qwen/Qwen1.5-1.8B-Chat "
                 "--name qwen --out out/qwen.json")
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("WARN: no OPENAI_API_KEY in env — alpaca_eval's judge will fail to authenticate.",
              file=sys.stderr)

    annotator = args.annotator or os.environ.get("ALPACA_ANNOTATOR", DEFAULT_ALPACA_ANNOTATOR)
    from alpaca_eval import evaluate as alpaca_evaluate
    with open(args.model) as f:
        model_outputs = json.load(f)
    with open(args.reference) as f:
        reference_outputs = json.load(f)

    print(f"[score] judging {os.path.basename(args.model)} ({len(model_outputs)} outputs) "
          f"vs {os.path.basename(args.reference)} with annotator={annotator}")
    leaderboard, _ = alpaca_evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        annotators_config=annotator,
        is_return_instead_of_print=True,
    )
    row = leaderboard.iloc[0]
    lc = row.get("length_controlled_winrate")
    wr = row.get("win_rate")
    lc = float(lc) if lc is not None else None
    wr = float(wr) if wr is not None else None
    print(f"[score]   length-controlled win-rate: {lc:.2f}%" if lc is not None else "[score]   (no LC column)")
    print(f"[score]   raw win-rate:               {wr:.2f}%" if wr is not None else "[score]   (no win_rate column)")

    if args.no_write:
        return
    eval_json = args.eval_json or os.path.join(os.path.dirname(os.path.abspath(args.model)), "eval.json")
    if not os.path.exists(eval_json):
        print(f"[score] no eval.json at {eval_json} to update; skipping write")
        return
    with open(eval_json) as f:
        data = json.load(f)
    fig3 = data.setdefault("figure3_alpacaeval", {})
    fig3["outputs"] = os.path.relpath(os.path.abspath(args.model), REPO_ROOT)
    fig3["n"] = len(model_outputs)
    fig3["annotator"] = annotator
    if lc is not None:
        fig3["lc_winrate"] = round(lc, 2)
    if wr is not None:
        fig3["winrate"] = round(wr, 2)
    fig3.pop("winrate_error", None)  # clear the stale gpt-4-1106-preview 404
    with open(eval_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[score] updated {eval_json} (figure3_alpacaeval.lc_winrate)")


if __name__ == "__main__":
    main()
