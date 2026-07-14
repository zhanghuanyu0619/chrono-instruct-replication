---
license: mit
language:
- en
library_name: pytorch
tags:
- finance
- chronological-consistency
- instruction-tuning
- chronologically-consistent
- modded-nanogpt
- point-in-time
- llm
datasets:
- manelalab/ChronoInstruct-SFT
base_model: {{BASE_MODEL}}
pipeline_tag: text-generation
inference: false
---

# ChronoGPT-Instruct (replication) — vintage {{VINTAGE}} (cutoff {{CUTOFF_DATE}})

> **Independent replication — not an official release.** This model was fine-tuned
> by **Huanyu Zhang** as a from-scratch reconstruction of the training pipeline in
> He, Lv, Manela & Wu (2025). It is **not** produced or endorsed by the paper's
> authors (manelalab). The base weights and the SFT dataset are theirs; the
> instruction-tuning **code and these fine-tuned weights** are this replication's.
> Absolute numbers are not expected to match the paper to the decimal (see
> [Limitations](#limitations--bias)).

## Model description

`chrono-instruct-v1-{{VINTAGE}}1231` is an instruction-tuned language model built by
supervised fine-tuning (SFT) of the base model
[`{{BASE_MODEL}}`](https://huggingface.co/{{BASE_MODEL}}) — a ~1.55B-parameter,
52-layer modded-nanoGPT U-net (26 encoder + 26 decoder layers with skip
connections and value embeddings; `model_dim` 1536, 12 heads, vocab 50304, context
1792; RMSNorm, rotary position embeddings, QK-norm, ReLU² MLP, and logit softcap).
The tokenizer is `tiktoken` GPT-2.

**Chronological-consistency guarantee (the point).** The base ChronoGPT vintage was
pretrained **only** on timestamped text available before its knowledge cutoff
**τ = {{CUTOFF_DATE}}**, and this replication fine-tunes it on an instruction set
that is screened to contain **only pre-2000** instruction/response pairs (see
[Training data](#training-data)). Because pre-2000 ⊆ pre-τ for every vintage
τ ≥ 1999, the fine-tuned model **never saw any text created after {{CUTOFF_DATE}}**.
This eliminates look-ahead bias / training leakage: the model is a point-in-time
artifact that "knows" only what was knowable at its cutoff.

**Why that matters.** A model contaminated with future information silently inflates
backtests of any text-conditioned prediction (return prediction, sentiment,
event studies). A chronologically consistent model is safe to run *as of* its cutoff
date, so a signal it produces on a {{CUTOFF_DATE}}-or-earlier document could genuinely
have been produced at that time.

## Intended use

- **Point-in-time (PIT) backtesting.** Use the vintage whose cutoff **precedes** the
  timestamp of each document you score, so no evaluation ever uses a model that saw
  the future. The six released vintages (1999, 2005, 2010, 2015, 2020, 2024) let you
  assemble a leakage-free panel across time.
- **Instruction following on pre-cutoff tasks** (Alpaca-style single-turn
  instruction/response), and hidden-state **embedding extraction** for downstream
  text-as-data / asset-pricing research.
- Research into chronological consistency, temporal knowledge cutoffs, and
  instruction tuning of small time-stamped LLMs.

### Out of scope

- **Current-events or factual QA about anything after {{CUTOFF_DATE}}.** The cutoff
  blindness is *intentional*; the model cannot know post-cutoff facts and will answer
  as if they do not exist.
- Production chat / assistant deployment. This is a small (~1.55B) research model,
  not safety-tuned or RLHF-aligned; it can hallucinate and loop under greedy decoding.
- Multilingual, multi-turn, or long-context (>1792 token) use.
- Any high-stakes decision without independent validation.

## How to load and generate

These repos store `pytorch_model.bin` + `config.pt`/`config.json` written by this
replication's `ChronoGPT.save_pretrained`. Loading therefore uses the replication
package's own `ChronoGPT` class (a lightly adapted, numerically bit-identical vendor
of the authors' `ChronoGPT_inference.py`) — **not** `transformers.AutoModel`, and
`trust_remote_code` is not required.

```bash
# Install the replication package (provides the ChronoGPT class + infer helpers)
pip install "git+https://github.com/zhanghuanyu0619/chrono-instruct-replication.git"
# or: git clone ... && cd chrono-instruct-replication && pip install -e .
```

```python
import torch
from chrono_instruct import infer

# Loads weights from the Hub, moves to CUDA if available, sets eval mode.
model, device = infer.load("HZ0619/chrono-instruct-v1-{{VINTAGE}}1231")

# The models were fine-tuned on the Stanford Alpaca prompt template
# (with a trailing newline after "### Response:"). Match it at inference time:
def alpaca_prompt(instruction: str, inp: str = "") -> str:
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n"
    )

prompt = alpaca_prompt("Name the President of the United States.")

# Greedy decoding by default (temperature=0.0, top_k=None) — matches the authors'
# ChronoGPT_instruct.py. A ~1.5B model can loop under pure greedy; for readable
# text turn on the anti-repetition guards (they change decoding, so leave them OFF
# to reproduce the paper's exact greedy output).
completion = infer.generate(
    model, device, prompt,
    max_new_tokens=128,
    temperature=0.0,          # 0.0 == greedy (argmax); set >0 to sample
    top_k=None,               # e.g. 50 to restrict sampling
    return_completion=True,   # return only the newly generated tokens
    # repetition_penalty=1.3, no_repeat_ngram_size=3,  # optional readability guards
)
print(completion)
```

For hidden-state embeddings (e.g. mean-pooled last-layer features for text-as-data):

```python
vec = infer.embed(model, device, "Some pre-cutoff document text.", layer=-1, pool="mean")
print(vec.shape)  # (1536,)
```

## Training data

- **Dataset:** [`manelalab/ChronoInstruct-SFT`](https://huggingface.co/datasets/manelalab/ChronoInstruct-SFT)
  — Alpaca-format instruction/response triples from three sources
  (LLMs-from-scratch simple tasks → GPT-3 self-instruct → AllenAI Tulu-3 mixture),
  647,944 rows as released.
- **Temporal screen (the leakage guard).** Only pairs the authors' GPT-4.1 temporal
  classifier labeled `0` ("knowledge available pre-2000") with `confidence == 10`
  are kept — a deliberately strict double filter. This replication reproduces the
  paper's counts exactly: **647,944 → 425,119** retained pairs (1,097 / 67,136 /
  356,886 across the three curriculum stages). A single pre-2000 screen is reused
  across all vintages because pre-2000 ⊆ pre-τ for every τ ≥ 1999.
- **Loss masking.** Response-only masked cross-entropy: prompt tokens are set to
  `-100` so the loss scores only the response tokens.

## Training procedure

- **Full fine-tuning** of all ~1.55B parameters with AdamW (no LoRA/PEFT).
- **3-stage curriculum**, each stage resuming from the previous stage's weights:
  1. `stage1_scratch` (LLMs-from-scratch simple tasks) — 3 epochs
  2. `stage2_self_instruct` (GPT-3 self-instruct) — 2 epochs
  3. `stage3_tulu` (Tulu-3 mixture) — 2 epochs
- **Hyperparameters:** learning rate **3e-4** (all stages, per-stage cosine schedule,
  warmup 0.03, floor `0.1·lr`); block size **1792**; batch size **8** × grad-accum
  **4** (effective 32); gradient checkpointing on; global seed **123** (drives the
  train/val split, shuffle, and sampling).
- **Compute:** ~**19.4 h** wall-clock on a single 80 GB card (peak ~50.4 GB GPU) per
  vintage.
- **Fidelity note:** the paper releases the SFT **data** and **weights** but not the
  **training code**; the LR schedule and epoch counts here are this replication's
  tuning against the *shape* of the paper's Figure 1, not paper-exact values.

## Evaluation

The intended evaluations follow the paper and are documented in this replication's
report — see `results/replication-report/README.md` in the repository:

- **SFT loss curves (Figures 1–2).** ✅ Reproduced. Stage-3 validation cross-entropy
  falls monotonically as the cutoff advances (0.8691 at 1999 → 0.7855 at 2024),
  the paper's "later vintage = better language model, not leakage" structure.
- **Chronological-consistency tests (Tables 2–3).** ✅ Reproduced. U.S. president
  prediction and major-world-events tests: correct on pre-cutoff items, blind (0/N)
  on post-cutoff items — zero look-ahead leakage across all vintages. Run per vintage
  with `chrono eval --repo HZ0619/chrono-instruct-v1-{{VINTAGE}}1231 --cutoff {{VINTAGE}}`.
- **AlpacaEval LC win-rate (Figure 3)** vs Qwen-1.5-1.8B-Chat: the paper reports a
  modest **12.59–16.79%** (Qwen saw ~31× more pretraining data). Generated; needs a
  judge API key to score.

## Limitations & bias

- **Small model (~1.55B).** Limited reasoning/factual coverage; greedy decoding can
  loop (use the anti-repetition guards for readable output).
- **Cutoff blindness is intentional, not a bug.** The model has zero knowledge of
  anything after {{CUTOFF_DATE}} — that is the whole design. Do not use it as a
  general knowledge base.
- **Not paper-exact.** LR schedule and epoch counts are tuned to the *shape* of the
  paper's curves; absolute losses differ from the paper. The replication claim is
  qualitative structure (data-count match, clean curriculum chaining, monotone
  improvement with cutoff), not decimal parity.
- **Packing.** Examples are packed into 1792-token blocks; ~5.1% of Tulu examples are
  split across a block boundary (no response tokens dropped, but such examples are
  never seen whole in one forward).
- Inherits any biases of the base ChronoGPT pretraining corpus and the
  ChronoInstruct-SFT data. Not safety-tuned.

## Citation

The paper this replicates:

```bibtex
@article{he2025instructchronogpt,
  title   = {Instruction Tuning Chronologically Consistent Language Models},
  author  = {He, Songrun and Lv, Linying and Manela, Asaf and Wu, Jimmy},
  year    = {2025},
  note    = {arXiv:2510.11677; SSRN 5348747}
}
```

Base models and SFT data (please cite the authors for both):

```bibtex
@article{he2025chronogpt,
  title   = {Chronologically Consistent Large Language Models},
  author  = {He, Songrun and Lv, Linying and Manela, Asaf and Wu, Jimmy},
  journal = {Working Paper},
  year    = {2025}
}
```

- **Base model:** [`{{BASE_MODEL}}`](https://huggingface.co/{{BASE_MODEL}}) (MIT license) — © manelalab.
- **SFT dataset:** [`manelalab/ChronoInstruct-SFT`](https://huggingface.co/datasets/manelalab/ChronoInstruct-SFT) — © manelalab.

## Model card authors

Replication and card by **Huanyu Zhang**. License **MIT**, inherited from the base
model; the ChronoInstruct-SFT dataset is subject to its own terms — consult the
dataset page before redistribution. This is an independent replication and is **not**
an official manelalab release.
