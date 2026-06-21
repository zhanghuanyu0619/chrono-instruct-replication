"""Evaluation.

president_test  -- the cheap, public, lookahead-bias consistency check (Table 2):
                   prompt the model to name the most recent U.S. president and
                   verify it cannot predict presidents past its knowledge cutoff.
AlpacaEval (Figure 3) -- length-controlled win-rate vs Qwen-1.5-1.8B-Chat. The
                   flow is: generate outputs for each model (`alpaca_outputs`),
                   then score model-vs-reference with the canonical `alpaca_eval`
                   package (`alpaca_winrate`). Judging needs an annotator key
                   (e.g. OPENAI_API_KEY) configured for alpaca_eval.
"""
import json

import torch

from .infer import generate, ENC

# (took_office_year, name) in chronological order — public knowledge.
PRESIDENTS = [
    (1993, "Bill Clinton"),
    (2001, "George W. Bush"),
    (2009, "Barack Obama"),
    (2017, "Donald Trump"),
    (2021, "Joe Biden"),
    (2025, "Donald Trump"),
]


def president_prompt(history):
    lines = ["U.S. Presidents in chronological order:"]
    for year, name in history:
        lines.append(f"Took office in {year}: President {name}")
    lines.append("Took office in {year}: President".format(year=history[-1][0] + 4))
    return "\n".join(lines)


@torch.no_grad()
def president_test(model, device, cutoff_year):
    """For each election, prompt with the 4 prior presidents and check the prediction.

    Returns a list of dicts; `past_cutoff` flags rows the model should NOT get
    right if it is chronologically consistent.
    """
    results = []
    for i in range(3, len(PRESIDENTS)):
        history = PRESIDENTS[i - 3 : i]
        target_year, target_name = PRESIDENTS[i]
        out = generate(model, device, president_prompt(history), max_new_tokens=4, top_k=1)
        completion = out[len(president_prompt(history)):].strip()
        results.append({
            "target_year": target_year,
            "target": target_name,
            "prediction": completion,
            "correct": target_name.split()[0] in completion,
            "past_cutoff": target_year > cutoff_year,
        })
    return results


def alpaca_instructions(n=None):
    """The AlpacaEval instruction set (805 prompts); `n` limits it for quick tests."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)
    items = [{"instruction": r["instruction"]} for r in ds]
    return items[:n] if n else items


def alpaca_outputs(repo, instructions, generator, backend="chrono", max_new_tokens=256):
    """Generate AlpacaEval-format outputs ({instruction, output, generator}).

    backend="chrono": our ChronoGPT, prompted with the same Alpaca template used
    in training. backend="hf": any HF chat model (e.g. the Qwen reference), via
    its chat template.
    """
    if backend == "chrono":
        from .infer import load
        from .data import PROMPT_NO_INPUT
        model, device = load(repo)
        outs = []
        for item in instructions:
            prompt = PROMPT_NO_INPUT.format(instruction=item["instruction"])
            text = generate(model, device, prompt, max_new_tokens=max_new_tokens)
            outs.append({"instruction": item["instruction"],
                         "output": text[len(prompt):].strip(), "generator": generator})
        return outs

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")
    outs = []
    for item in instructions:
        chat = tok.apply_chat_template([{"role": "user", "content": item["instruction"]}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        completion = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append({"instruction": item["instruction"],
                     "output": completion.strip(), "generator": generator})
    return outs


def alpaca_winrate(model_outputs_json, reference_outputs_json):
    """Length-controlled win-rate (%) of model vs reference, via the alpaca_eval package.

    Delegates judging + the length-controlled regression to the canonical tool so
    we don't re-implement (and mis-implement) it. The exact return column may vary
    by alpaca_eval version; the saved output JSONs are the stable artifacts.
    """
    from alpaca_eval import evaluate as alpaca_evaluate
    leaderboard, _ = alpaca_evaluate(
        model_outputs=json.load(open(model_outputs_json)),
        reference_outputs=json.load(open(reference_outputs_json)),
        is_return_instead_of_print=True,
    )
    row = leaderboard.iloc[0]
    return float(row.get("length_controlled_winrate", row.get("win_rate")))
