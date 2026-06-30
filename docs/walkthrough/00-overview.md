# Code Walkthrough — Overview & Reading Guide

**What this is.** A complete, self-paced walkthrough of the `chrono-instruct-replication`
project, written for *me* (Huanyu) to read offline and understand the codebase
**completely** — every module, every non-obvious line, and how each piece maps to
the paper. The target reader is a finance PhD who is strong on econometrics and
optimization but new to deep learning, so every ML concept is defined from scratch.

**Why it exists.** I want to be able to discuss this replication credibly with
**Asaf Manela** (co-author of the paper, and a hoped-for future advisor). That
means understanding not just *what* the code does but *why* each engineering
choice was made and where it departs from or fills gaps in the paper. This set is
for my own review — it is **not** meant to be pushed to GitHub.

---

## The project in one paragraph

The paper — *Instruction Tuning Chronologically Consistent Language Models* (He,
Lv, Manela, Wu 2025; SSRN 5348747 / arXiv 2510.11677) — takes **ChronoGPT**, a
family of language models each trained only on text up to a cutoff year τ (so they
carry **no lookahead bias** for analysis "as of" τ), and teaches them to **follow
instructions** without reintroducing future knowledge. It does this with a 3-stage
supervised fine-tuning (SFT) curriculum on instruction data that has itself been
**temporally screened** to remove post-cutoff content. The authors released the
**weights** and the **SFT data** but **not the training code**. This repo
reconstructs that training pipeline from the released base weights as **reusable
research infrastructure** — so any ChronoGPT vintage can be fine-tuned, then used
for text generation *or* embedding extraction in downstream, lookahead-free
prediction work.

If you read nothing else, read `02-paper-and-research-framing.md` first — it is the
bridge from the research idea to the code.

---

## How to read this set

The docs are ordered so each builds on the last. Two ways through:

**Linear (recommended for full understanding):** 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08.

**Fast path (if I just want the research story + the hard parts):**
02 (paper) → 03 (architecture) → 04 (data/no-leakage screen) → 06 (the eval tests
that *are* the paper's results).

| # | Doc | What it covers | Hardest / most important |
|---|-----|----------------|--------------------------|
| 00 | `00-overview.md` (this file) | Map, reading order, glossary pointers | — |
| 01 | `01-ml-primer.md` | All the deep-learning concepts I need, with finance analogies | Foundation for everything else |
| 02 | `02-paper-and-research-framing.md` | The research contribution; each table/figure → repo function | ⭐ The "why it matters" doc |
| 03 | `03-model.md` | `model.py` line-by-line — the modded-nanoGPT architecture | ⭐⭐ The flagship; most likely to be quizzed on |
| 04 | `04-data.md` | `data.py` — temporal screen, prompt format, masking, packing | ⭐ The no-lookahead contract lives here |
| 05 | `05-train.md` | `train.py` — loss, optimizer, curriculum, resume, memory | ⭐ The optimization + MLOps core |
| 06 | `06-infer-and-eval.md` | `infer.py` + `eval.py` — generation, embeddings, Tables 2-3, Fig 3 | The headline results in code |
| 07 | `07-cli-tracking-hub-figures.md` | The glue: `cli.py`, `tracking.py`, `hub.py`, `figures.py` | How runs are driven & plotted |
| 08 | `08-configs-scripts-infra.md` | configs, scripts, packaging, tests, running on Lambda GPUs | The operational picture |

Companion docs already in `docs/` (referenced throughout, not part of this set):
- `implementation-notes.md` — the design-decision log (the *why* behind every gap-filling choice). Read alongside 03/04/05.
- `running-guide.md` — the end-to-end Lambda workflow (provision → train → figures → publish).

---

## The codebase at a glance

About 1,200 lines of Python across 10 small modules — deliberately kept simple and
auditable. The data flows left to right:

```
 raw SFT data ──► data.py ──► train.py ──► checkpoints ──► infer.py ──► eval.py
 (HF dataset)     screen +     curriculum    (HF Hub)      generate/    Tables 2-3,
                  tokenize +   SFT loop                     embed        Figure 3
                  mask + pack
                                  │
                                  ├─ tracking.py  → metrics.csv (+ optional W&B)
                                  ├─ hub.py       → push weights to Hugging Face
                                  └─ figures.py   → Figures 1-2-3 from metrics.csv
            cli.py  = the `chrono ...` command that drives all of the above
            configs/, scripts/ = how a run is configured and launched on a GPU box
```

| Module | Lines | Role | Walkthrough |
|--------|-------|------|-------------|
| `model.py` | 222 | The 1.55B causal LM (modded-nanoGPT lineage), vendored from the authors' inference file | 03 |
| `data.py` | 223 | Screen → tokenize → mask → pack → split → cache the SFT data | 04 |
| `train.py` | 188 | The curriculum SFT loop: loss, AdamW, eval, checkpoint, resume, HF push | 05 |
| `eval.py` | 173 | President/events consistency tests + AlpacaEval pipeline | 06 |
| `infer.py` | 70 | `generate` (text) and `embed` (vectors), plus GPU memory cleanup | 06 |
| `cli.py` | 142 | The `chrono` subcommands (`inspect/train/infer/eval/push/alpaca/winrate/figure`) | 07 |
| `tracking.py` | 67 | `RunLogger` → metrics.csv + summary.json (+ optional W&B) | 07 |
| `figures.py` | 73 | Plot Figures 1/2/3 from metrics.csv | 07 |
| `hub.py` | 17 | Push a checkpoint folder to the Hugging Face Hub | 07 |
| `__init__.py` | 2 | Package marker | 07 |

---

## Five things to keep in mind while reading

These are the recurring threads that tie the whole project together — and the
points most worth being fluent in for a conversation with Asaf.

1. **Two different "no future" mechanisms — don't conflate them.** The Transformer's
   *causal mask* stops a token from attending to *later tokens in the same
   sequence* (a within-text mechanism, in `model.py`). The paper's *chronological
   consistency* stops the model from knowing *anything that happened after year τ*
   (a data/training-time mechanism, enforced by the **temporal screen** in
   `data.py`). Both say "no looking ahead," but at completely different levels. See
   `01-ml-primer.md` §4 and `02-paper-and-research-framing.md`.

2. **The temporal screen *is* the contribution, in code form.** `keep_row` /
   `_parse_label` in `data.py` are short but load-bearing: they enforce the
   no-leakage contract on the *instruction* data, not just the pretraining data.
   A subtle parsing bug there once silently dropped ~90% of the Tulu stage — proof
   that this filter is where correctness is won or lost. See `04-data.md`.

3. **The architecture is not a textbook GPT-2.** It is the modded-nanoGPT lineage:
   U-net depth skips, value embeddings, x0 skips, RoPE, QK-norm, RMSNorm, squared-
   ReLU MLP, logit softcap, zero-init projections — and notably **no padding mask
   and no pad token**, which is *why* the data is packed rather than padded. See
   `03-model.md` and `04-data.md`.

4. **It's full fine-tuning, and memory is dominated by activations, not weights.**
   All 1.55B params are updated with AdamW. The ~25 GB of optimizer state is fixed;
   what actually blows up the GPU is *activations* across 52 layers × 1792 tokens —
   which is why gradient checkpointing exists. See `05-train.md` §7.

5. **This is a faithful-but-honest replication, with known open items.** Where the
   paper is silent (the *how*), the repo makes documented choices (Alpaca prompt
   format, simple packing, AdamW not Muon). Known gaps: simple packing splits ~5%
   of long Tulu examples, and the current run is **under-trained** (loss above the
   paper's — it needs more epochs). Being candid about these is a strength, not a
   weakness, in a conversation with the author. See `implementation-notes.md`.

---

## Glossary

A 25-term ML glossary lives at the end of `01-ml-primer.md`. The design-decision
rationale (every gap-filling choice, with status flags) lives in
`docs/implementation-notes.md`. When a term shows up mid-walkthrough without
definition, it's defined in one of those two places.

---

*Reading next: `01-ml-primer.md` for the concepts, or jump to
`02-paper-and-research-framing.md` for the research story.*
