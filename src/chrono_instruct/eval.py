"""Evaluation.

president_test  -- the cheap, public, lookahead-bias consistency check (Table 2):
                   prompt the model to name the most recent U.S. president and
                   verify it cannot predict presidents past its knowledge cutoff.
alpaca_winrate  -- length-controlled win-rate vs Qwen-1.5-1.8B-Chat (Figure 3).
                   Deferred: needs an LLM-judge API + the AlpacaEval harness.
"""
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


def alpaca_winrate(*args, **kwargs):
    raise NotImplementedError(
        "AlpacaEval length-controlled win-rate is deferred — wire up the AlpacaEval "
        "harness + an LLM judge here (configs/eval.yaml)."
    )
