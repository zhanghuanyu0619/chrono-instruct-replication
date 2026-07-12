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

# (election_year, name) — the six presidential elections in the paper's Table 2.
# The prompt DISPLAYS the inauguration year (election_year + 1, e.g. 2001 for the
# 2000 election), matching the paper's "Took office in ..." template; the knowledge-
# cutoff comparison keys on the ELECTION year, since who won is only knowable after
# the November vote. Paper elections: 1992, 2000, 2008, 2016, 2020, 2024.
PRESIDENTS = [
    (1992, "Bill Clinton"),
    (2000, "George W. Bush"),
    (2008, "Barack Obama"),
    (2016, "Donald Trump"),
    (2020, "Joe Biden"),
    (2024, "Donald Trump"),
]


def president_prompt(history, target_election_year):
    """Build the Table 2 prompt: prior presidents, then the target blank to fill.

    Presidents are stored by ELECTION year; each is shown by its INAUGURATION year
    (election + 1 — the January after the November vote, exact for one- and two-term
    presidents alike), matching the paper's "Took office in ..." wording.
    """
    lines = ["U.S. Presidents in chronological order:"]
    for election_year, name in history:
        lines.append(f"Took office in {election_year + 1}: President {name}")
    lines.append(f"Took office in {target_election_year + 1}: President")
    return "\n".join(lines)


@torch.no_grad()
def president_test(model, device, cutoff_year):
    """For each election, prompt with up to three prior presidents and check the prediction.

    Reads exactly two tokens by greedy decoding, as in the paper. `past_cutoff` keys on
    the ELECTION year (who won is only knowable after the November vote), so a
    chronologically consistent model should be correct only for elections at/before its
    cutoff. History is clamped to the priors that exist — with only the six paper-tested
    presidents, the earliest targets get fewer than three priors (the paper's full list
    carries older presidents so every target has three).
    """
    results = []
    for i in range(0, len(PRESIDENTS)):
        history = PRESIDENTS[max(0, i - 3) : i]          # up to 3 priors; clamp, never negative-wrap
        election_year, target_name = PRESIDENTS[i]
        completion = generate(model, device, president_prompt(history, election_year),
                              max_new_tokens=2, top_k=1, return_completion=True).strip()
        results.append({
            "election_year": election_year,
            "took_office_year": election_year + 1,
            "target": target_name,
            "prediction": completion,
            "correct": target_name.split()[0] in completion,
            "past_cutoff": election_year > cutoff_year,
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


def alpaca_winrate(model_outputs_json, reference_outputs_json):
    """Length-controlled win-rate (%) of model vs reference, via the alpaca_eval package.

    Delegates judging + the length-controlled regression to the canonical tool so
    we don't re-implement (and mis-implement) it. The exact return column may vary
    by alpaca_eval version; the saved output JSONs are the stable artifacts.
    """
    from alpaca_eval import evaluate as alpaca_evaluate
    with open(model_outputs_json) as f:
        model_outputs = json.load(f)
    with open(reference_outputs_json) as f:
        reference_outputs = json.load(f)
    leaderboard, _ = alpaca_evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        is_return_instead_of_print=True,
    )
    row = leaderboard.iloc[0]
    return float(row.get("length_controlled_winrate", row.get("win_rate")))
