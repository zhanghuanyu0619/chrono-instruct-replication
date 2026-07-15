"""Evaluation.

president_test  -- the lookahead-bias consistency check on U.S. presidents (Table 2):
                   prompt with the three prior presidents and the target's actual
                   inauguration year, and verify the model cannot name presidents
                   past its knowledge cutoff.
major_events_test -- the companion check on dated world events (Table 3): complete a
                   sentence about an event and verify the model cannot recall events
                   past its cutoff. Both decode deterministically (greedy), as in the paper.
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


def president_prompt(history, query_year):
    """Build the Table 2 prompt: three prior presidents, then `query_year` to fill.

    `query_year` is the TARGET's actual inauguration year (not previous+4): two-term
    presidents make the gap 8 years, so deriving it arithmetically would mis-date the
    blank (e.g. asking about 2013, mid-Obama, when the target took office in 2017).
    """
    lines = ["U.S. Presidents in chronological order:"]
    for year, name in history:
        lines.append(f"Took office in {year}: President {name}")
    lines.append(f"Took office in {query_year}: President")
    return "\n".join(lines)


@torch.no_grad()
def president_test(model, device, cutoff_year):
    """For each transition, prompt with the three prior presidents and check the prediction.

    Reads exactly two tokens by greedy decoding, as in the paper. Returns a list of
    dicts; `past_cutoff` flags rows the model should NOT get right if it is
    chronologically consistent.
    """
    results = []
    for i in range(0, len(PRESIDENTS)):
        history = PRESIDENTS[i - 3 : i]
        target_year, target_name = PRESIDENTS[i]
        completion = generate(model, device, president_prompt(history, target_year),
                              max_new_tokens=2, top_k=1, return_completion=True).strip()
        results.append({
            "target_year": target_year,
            "target": target_name,
            "prediction": completion,
            "correct": target_name.split()[0] in completion,
            "past_cutoff": target_year > cutoff_year,
        })
    return results


# (event_year, prompt prefix, accepted answer substrings) transcribed from Table 3,
# Panel A. The model completes the blank; verify exact wording against the PDF if you
# need a byte-faithful reproduction. Accepted terms are matched case-insensitively.
MAJOR_EVENTS = [
    (2001, "The Sarbanes-Oxley Act was introduced in response to the 2001 Enron", ["scandal"]),
    (2003, "In 2003, a major public health crisis was the outbreak of the virus known as", ["SARS"]),
    (2008, "In 2008, the global economy was dominated by the subprime mortgage", ["crisis"]),
    (2016, "In 2016, market volatility increased surrounding the general vote known as the Brexit",
     ["referendum"]),
    (2020, "In 2020, the global economy was devastated by the health crisis known as the",
     ["COVID", "coronavirus", "corona"]),
    (2022, "In 2022, a major milestone for generative AI was marked by the release of the AI chatbot "
           "known as", ["ChatGPT", "GPT"]),
]


@torch.no_grad()
def major_events_test(model, device, cutoff_year):
    """Complete a dated-event sentence and check the term (Table 3).

    Reads three tokens by greedy decoding, as in the paper. `past_cutoff` flags events
    after the model's knowledge cutoff — a chronologically consistent model should fail
    those.
    """
    results = []
    for event_year, prompt, answers in MAJOR_EVENTS:
        completion = generate(model, device, prompt,
                              max_new_tokens=3, top_k=1, return_completion=True).strip()
        low = completion.lower()
        results.append({
            "event_year": event_year,
            "answer": answers[0],
            "prediction": completion,
            "correct": any(a.lower() in low for a in answers),
            "past_cutoff": event_year > cutoff_year,
        })
    return results


def alpaca_instructions(n=None):
    """The AlpacaEval instruction set (805 prompts); `n` limits it for quick tests.

    Loads the raw JSON straight from the Hub. The old
    load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval",
    trust_remote_code=True) relied on a dataset *loading script*, which
    `datasets` >= 3.0 refuses to run ("Dataset scripts are no longer supported")
    and which no longer honors trust_remote_code. Downloading alpaca_eval.json
    directly is version-proof and needs no `datasets`.
    """
    import json
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    items = [{"instruction": r["instruction"]} for r in data]
    return items[:n] if n else items


def alpaca_outputs(repo, instructions, generator, backend="chrono", max_new_tokens=256):
    """Generate AlpacaEval-format outputs ({instruction, output, generator}).

    backend="chrono": our ChronoGPT, prompted with the same Alpaca template used
    in training. backend="hf": any HF chat model (e.g. the Qwen reference), via
    its chat template.

    Decoding is pure GREEDY (top_k=1 / do_sample=False), matching the authors'
    released `ChronoGPT_instruct.py generate()` (temperature=0.0, argmax, no
    repetition penalty). A 1.55B model under greedy does degenerate into repetition
    loops on some prompts — but that is exactly what the authors' own model does, so
    the paper's low AlpacaEval win rates (Fig 3: 12.6-16.8% vs Qwen) already reflect
    it. Adding anti-repetition here would make our model look *better* than the paper's
    actual generation, i.e. an unfaithful replication; we deliberately don't.
    """
    if backend == "chrono":
        from .infer import load
        from .data import PROMPT_NO_INPUT
        model, device = load(repo)
        outs = []
        for item in instructions:
            prompt = PROMPT_NO_INPUT.format(instruction=item["instruction"])
            # Greedy (top_k=1) to match the HF reference's do_sample=False below: the
            # win-rate must not confound decoding strategy with model quality.
            completion = generate(model, device, prompt, max_new_tokens=max_new_tokens,
                                  top_k=1, return_completion=True)
            outs.append({"instruction": item["instruction"],
                         "output": completion.strip(), "generator": generator})
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


# AlpacaEval's default annotator (`weighted_alpaca_eval_gpt4_turbo`) judges with
# gpt-4-1106-preview, which OpenAI has retired — it now 404s ("model_not_found").
# Default instead to a shipped annotator config backed by a currently-available
# model; override via the ALPACA_ANNOTATOR env var (any evaluators_config name in
# the installed alpaca_eval, e.g. a gpt-4-turbo-family config for higher human
# agreement at higher cost).
DEFAULT_ALPACA_ANNOTATOR = "weighted_alpaca_eval_gpt-4o-mini-2024-07-18"


def alpaca_winrate(model_outputs_json, reference_outputs_json, annotators_config=None):
    """Length-controlled win-rate (%) of model vs reference, via the alpaca_eval package.

    Delegates judging + the length-controlled regression to the canonical tool so
    we don't re-implement (and mis-implement) it. The exact return column may vary
    by alpaca_eval version; the saved output JSONs are the stable artifacts.
    `annotators_config` selects the judge; defaults to ALPACA_ANNOTATOR or
    DEFAULT_ALPACA_ANNOTATOR (avoiding the retired gpt-4-1106-preview default).

    `output_path=None` disables alpaca_eval's own file dump: by default it saves the
    per-prompt annotations + a leaderboard to `results/{generator}/{annotator}/`
    (relative to cwd), which spawned stray `results/chrono-<τ>/` dirs (~3 MB each of
    annotations) parallel to our `results/chrono-instruct-<τ>/`. We already write the
    win-rate into each vintage's eval.json, so the dump is pure clutter — off by default.
    """
    import os
    from alpaca_eval import evaluate as alpaca_evaluate
    annotator = annotators_config or os.environ.get("ALPACA_ANNOTATOR", DEFAULT_ALPACA_ANNOTATOR)
    with open(model_outputs_json) as f:
        model_outputs = json.load(f)
    with open(reference_outputs_json) as f:
        reference_outputs = json.load(f)
    leaderboard, _ = alpaca_evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        annotators_config=annotator,
        is_return_instead_of_print=True,
        output_path=None,
    )
    row = leaderboard.iloc[0]
    return float(row.get("length_controlled_winrate", row.get("win_rate")))
