#!/usr/bin/env python3
"""Evaluate ONE ChronoGPT-Instruct vintage and SAVE the results (Tables 2-3, opt. Figure 3).

Default target is the HF-published repo `HZ0619/chrono-instruct-v1-{cutoff}1231`
(all vintages are on the Hub). Writes a structured `eval.json` under
`results/chrono-instruct-{vintage}/` — the same place training saves its metrics —
so `eval_all_vintages.sh` collects every vintage without manual copying.

- Table 2 (U.S. president prediction) + Table 3 (major world events) run by default:
  cheap, greedy-decoded, no dataset needed — just the model.
- Figure 3 (AlpacaEval length-controlled win-rate vs Qwen-1.5-1.8B-Chat) is opt-in
  via --alpaca; it generates this model's outputs and, if given a reference outputs
  JSON (+ an OPENAI_API_KEY judge), computes the win-rate.

Punchline each exhibit should show (paper): a chronologically consistent model is
CORRECT on items before its cutoff and BLIND (wrong) on items after it — that is
the no-look-ahead guarantee (Tables 2-3); and despite that constraint it still
follows instructions competitively (~54-62% LC win-rate vs Qwen — Figure 3).

Usage:
  python scripts/run_eval.py --vintage 2024
  python scripts/run_eval.py --vintage 2024 --repo runs/chrono-instruct-2024/final
  python scripts/run_eval.py --vintage 2024 --alpaca --alpaca-reference out/qwen.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chrono_instruct.infer import load, free_memory  # noqa: E402
from chrono_instruct.eval import president_test, major_events_test  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def consistency_summary(rows):
    """Score a consistency test: a chronologically consistent model is correct on
    pre-cutoff items and wrong on post-cutoff items (`correct != past_cutoff`)."""
    pre = [r for r in rows if not r["past_cutoff"]]
    post = [r for r in rows if r["past_cutoff"]]
    return {
        "n": len(rows),
        "pre_cutoff_correct": f"{sum(r['correct'] for r in pre)}/{len(pre)}",
        "post_cutoff_correct": f"{sum(r['correct'] for r in post)}/{len(post)}",  # want 0/N
        "consistent_rows": f"{sum(1 for r in rows if r['correct'] != r['past_cutoff'])}/{len(rows)}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintage", type=int, required=True, help="cutoff year, e.g. 2024")
    ap.add_argument("--repo", default=None,
                    help="model repo id or local dir (default: HF HZ0619/chrono-instruct-v1-{vintage}1231)")
    ap.add_argument("--hf-user", default="HZ0619")
    ap.add_argument("--out", default=None, help="output JSON (default results/chrono-instruct-{vintage}/eval.json)")
    ap.add_argument("--alpaca", action="store_true", help="also run Figure 3 (AlpacaEval)")
    ap.add_argument("--alpaca-reference", default=None, help="reference outputs JSON (e.g. Qwen) to judge against")
    ap.add_argument("--alpaca-n", type=int, default=None, help="limit #AlpacaEval instructions (debug)")
    args = ap.parse_args()

    vintage = args.vintage
    cutoff = vintage  # the model's knowledge-cutoff year
    repo = args.repo or f"{args.hf_user}/chrono-instruct-v1-{vintage}1231"
    out = args.out or os.path.join(REPO_ROOT, "results", f"chrono-instruct-{vintage}", "eval.json")

    model, device = load(repo)
    print(f"[eval] {repo} on {device} (cutoff {cutoff})")

    table2 = president_test(model, device, cutoff)
    table3 = major_events_test(model, device, cutoff)
    result = {
        "vintage": vintage, "cutoff": cutoff, "repo": repo,
        "table2_president": {"rows": table2, "summary": consistency_summary(table2)},
        "table3_major_events": {"rows": table3, "summary": consistency_summary(table3)},
    }
    print(f"[eval]   Table 2 (president):     {result['table2_president']['summary']}")
    print(f"[eval]   Table 3 (major events):  {result['table3_major_events']['summary']}")

    if args.alpaca:
        from chrono_instruct.eval import alpaca_instructions, alpaca_outputs, alpaca_winrate
        outs = alpaca_outputs(repo, alpaca_instructions(args.alpaca_n), f"chrono-{vintage}", backend="chrono")
        model_json = os.path.join(os.path.dirname(out), f"alpaca_{vintage}.json")
        os.makedirs(os.path.dirname(model_json) or ".", exist_ok=True)
        with open(model_json, "w") as f:
            json.dump(outs, f)
        fig3 = {"outputs": os.path.relpath(model_json, REPO_ROOT), "n": len(outs)}
        if args.alpaca_reference and os.path.exists(args.alpaca_reference):
            try:
                wr = alpaca_winrate(model_json, args.alpaca_reference)
                fig3["lc_winrate"] = round(wr, 2)
                print(f"[eval]   Figure 3 LC win-rate vs reference: {wr:.2f}%")
            except Exception as e:  # judging needs OPENAI_API_KEY; don't lose the generated outputs
                fig3["winrate_error"] = str(e)
                print(f"[eval]   Figure 3: outputs saved, winrate not computed ({e})")
        else:
            print("[eval]   Figure 3: outputs saved; pass --alpaca-reference + set OPENAI_API_KEY to judge")
        result["figure3_alpacaeval"] = fig3

    del model
    free_memory()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
