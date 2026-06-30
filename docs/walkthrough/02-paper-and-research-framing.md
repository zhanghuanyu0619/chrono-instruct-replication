# 02 — The Paper and the Research Framing

**What this doc is.** This is the bridge between the academic paper —
He, Lv, Manela & Wu (2025), *"Instruction Tuning Chronologically Consistent
Language Models"* (SSRN 5348747 / arXiv 2510.11677) — and the code in this repo.
By the end you should be able to (a) state the paper's core contribution
precisely, in language a finance audience understands, and (b) point to the exact
repo function or CLI command that reproduces each table and figure. It assumes
you've skimmed `01-ml-primer.md` for the ML vocabulary (tokens, cross-entropy,
SFT); it does *not* assume you've read the code yet — siblings like
`06-infer-and-eval.md` cover that.

## Table of contents

1. [The research problem: lookahead bias in LLM-based finance](#1-the-research-problem-lookahead-bias-in-llm-based-finance)
2. [ChronoGPT and the "vintage" idea](#2-chronogpt-and-the-vintage-idea)
3. [What *this* paper adds: instruction-following without re-introducing leakage](#3-what-this-paper-adds-instruction-following-without-re-introducing-leakage)
4. [The 3-stage curriculum in detail (Table 1)](#4-the-3-stage-curriculum-in-detail-table-1)
5. [The temporal screen (eq. 7, the no-leakage contract)](#5-the-temporal-screen-eq-7-the-no-leakage-contract)
6. [Results, tables, figures — and how the repo reproduces each](#6-results-tables-figures--and-how-the-repo-reproduces-each)
7. [What this replication repo is, and is not](#7-what-this-replication-repo-is-and-is-not)
8. [Talking to Asaf: six understanding-checks](#8-talking-to-asaf-six-understanding-checks)

---

## 1. The research problem: lookahead bias in LLM-based finance

### The one-line version

If you use a 2024-trained LLM to "predict" a 2010 stock return or to score the
sentiment of a 2010 news headline, the model may already *know how 2010 turned
out*. Any predictive power you measure is then partly memory, not forecasting.
The paper calls this **lookahead bias / training leakage**; a finance reader can
read it as a particularly insidious form of **look-ahead bias / data snooping**.

### Why this is exactly look-ahead bias (and where the analogy is approximate)

You already police look-ahead bias constantly. If you compute a trading signal at
the close of day *t* but accidentally use day *t+1*'s closing price in the
formula, your backtest is contaminated — the Sharpe ratio is fake. Standard
defenses: lag your features, use point-in-time databases (Compustat PIT), respect
the announcement-vs-period-end gap in fundamentals.

An LLM creates a subtler version of the same problem, one **no lag can fix**. The
contamination is not in your feature matrix; it is baked into the model's
*parameters* during pretraining. A model trained on text through 2024 has read
news articles, Wikipedia edits, and analyst reports written *after* 2010 that
discuss what happened to firms in 2010. When you prompt it about a 2010 headline,
it can draw on that future text. There is no column to lag — the leak is inside
the weights.

Where the finance analogy is **approximate**: classic look-ahead bias is usually
a discrete, locatable mistake (a misaligned timestamp). LLM leakage is *diffuse
and unobservable* — you cannot point to the offending row, and for a closed
commercial model (GPT-4, Claude) you cannot even inspect the training set to know
whether 2010 was leaked. That unobservability is the whole reason the paper needs
a model that is leakage-free **by construction** rather than leakage-checked after
the fact.

### The formal statement (paper §2.1)

The paper adopts the **no-training-leakage contract** of Ludwig, Mullainathan &
Rambachan (2025). It writes expected evaluation loss as a clean "true
out-of-sample loss" *minus* a **leakage term** (paper eq. 4). The leakage term
vanishes if and only if membership in the training set is *statistically
independent* of membership in the evaluation set (eq. 6):

> for every text *r*: `q_{T|D}(t_r) / q_T(t_r) = 1`.

In words: knowing a document is in your test set must not make it any more likely
to have been in the training set. (For the careful probability mechanics — why
this is an ex-ante / superpopulation statement, and why it's `Pr(t|D)` not
`Pr(D|t)` — see the reading Q&A in the Obsidian note
`he-lv-manela-wu-2025-chronogpt-instruct.md`; it's worth reading once.)

This is the **core intellectual contribution's foundation**: reframe "did the
model cheat?" as a precise independence condition, then engineer a model that
satisfies it.

---

## 2. ChronoGPT and the "vintage" idea

### The base model (the prerequisite paper)

ChronoGPT comes from a **prior** paper — He, Lv, Manela & Wu (2025),
*"Chronologically consistent large language models"* (arXiv 2502.21206). The idea:
instead of patching leakage after training, **pretrain from scratch on a corpus
that contains only text dated on or before a cutoff τ.** Every document carries a
verifiable publication timestamp; anything dated after τ is mechanically
discarded. So for any evaluation item dated after τ, the probability it was in
training is exactly zero — the independence condition holds *by construction*
(paper eq. 8). The corpus is built from historical web snapshots, archived news,
and scientific literature (paper §2.1).

### "Vintages"

They don't build one model; they build a **family**, one per cutoff:
τ ∈ {1999, 2000, 2001, …, 2024}. Think of these like **point-in-time database
snapshots**, or like CRSP/Compustat *vintages* — each ChronoGPT-τ is "the
language model as the world knew it at year-end τ." To run any analysis as of
time τ with zero lookahead, you pick the vintage-τ model. The current paper uses
six headline vintages — {1999, 2005, 2010, 2015, 2020, 2024} — and the finer
annual grid for robustness (paper §3.1, and Fig. 4 in the full paper).

Crucial nuance for the "realtime" trading test (§6 below): a properly run live
backtest uses, *at each date*, the vintage trained only through the **prior**
year. That keeps the trading exercise itself genuinely zero-lookahead, not just
the model.

### Base vs. Instruct

- **Base ChronoGPT** (prior paper): a pretrained, *non-chat* decoder-only
  language model. It completes text and produces good embeddings, but it does not
  reliably follow instructions like "classify this headline as FAVORABLE /
  UNFAVORABLE."
- **Instruct ChronoGPT** (*this* paper): takes each base vintage and adds an
  instruction-following layer via supervised fine-tuning (SFT) — so you can
  *prompt* it conversationally — **while keeping the chronological-consistency
  guarantee intact.** That last clause is the entire challenge.

---

## 3. What *this* paper adds: instruction-following without re-introducing leakage

The base paper solved leakage for **pretraining**. But a base LM is awkward to
use for social-science research — you can't just ask it questions. Modern usable
LLMs are *instruction-tuned* (the "Instruct" / "Chat" suffix). Instruction tuning
means a second training stage on (instruction, response) examples.

Here is the trap, and the paper's contribution in one sentence: **instruction-
tuning data is itself a leakage backdoor**, because Q&A pairs and dialogues
usually carry *no reliable publication timestamp*, so you cannot filter them "by
date" the way you filter dated news. A 2023-authored instruction that mentions
COVID-19 or ChatGPT, fed to the 2010 vintage during SFT, silently teaches the
"2010" model about the future.

The paper's three deliverables on top of base ChronoGPT:

1. **A 3-stage instruction-tuning curriculum** (§4 below) that turns each base
   vintage into a chat model.
2. **A temporal screen of the SFT data** (§5 below) so the instruction data also
   contains no post-cutoff knowledge — closing the backdoor. This is formalized
   as *stage-wise sufficiency* (eq. 7): because "in training" is the union of two
   *disjoint* events (pretraining ∪ IFT), the overall independence condition holds
   once it holds *separately* for each stage. Pretraining is already clean (eq. 8);
   they just need to make IFT clean too.
3. **Empirical proof** that the result follows instructions (Fig. 1–3) *and*
   stays chronologically consistent (Tables 2–3) — i.e., you can buy instruction-
   following without re-buying lookahead.

The payoff application: a **conservative lower bound** on how much of the famous
"LLMs predict stock returns from news" result survives once leakage is fully
removed — roughly **54–62%** (paper §3.3; abstract). The logic is "stack the deck
against yourself": ChronoGPT-Instruct is a *weaker* model (its base saw ~70B
tokens vs. Qwen-1.5-1.8B-Chat's ~2.2T, ≈31× less; paper p.2), so whatever
predictability it *retains* is a floor, not a point estimate.

---

## 4. The 3-stage curriculum in detail (Table 1)

The SFT corpus is ~425,000 (instruction, response) pairs **after screening**,
ordered as a curriculum that *grows in cognitive load and sequence length* —
easy/short tasks first, hard/long conversations last. This is the standard
"curriculum learning" intuition: don't open with the hardest material.

Paper **Table 1** (verified against the PDF, p.7):

| Stage | SFT data source | N examples (post-screen) | Avg. conversation length |
|------:|-----------------|-------------------------:|-------------------------:|
| 1 | LLMs-from-scratch simple tasks — spelling, basic math (Raschka 2024) | 1,097 | 102 |
| 2 | GPT-3 self-generated data via self-instruct (Wang et al. 2022) | 67,136 | 183 |
| 3 | AllenAI Tulu-3 SFT mixture (Lambert et al. 2024) | 356,886 | 2,513 |

Total ≈ **425,119** examples (1,097 + 67,136 + 356,886). The "average length"
column is the paper's reported conversation length, in its own units.

Why a curriculum (easy → hard)? Stage 1 teaches the *format* — what an
"### Instruction / ### Response" turn even looks like — on trivially short tasks.
Stage 2 adds genuine but still-short instructions. Stage 3 (Tulu-3) is the broad,
long, complex chat mixture that does most of the heavy lifting toward general
instruction-following. Figure 1 (below) shows the loss restarting and dropping at
each stage transition.

> **A discrepancy worth holding in your head.** The paper's Table 1 lists Stage-3
> average length as **2,513**, but when this repo tokenizes Tulu-3 with ChronoGPT's
> GPT-2 tokenizer it measures a **704-token mean** (implementation-notes §7; only
> 5.1% of Tulu exceeds the 1,792 block size). The two numbers use different
> definitions/units (the paper's "conversation length" vs. the repo's GPT-2 token
> count), so they're not directly comparable — but the gap is exactly the kind of
> thing to confirm before quoting "2,513 tokens" to anyone. See §8.

---

## 5. The temporal screen (eq. 7, the no-leakage contract)

This is the paper's *novel* operational step. Since IFT examples have no
timestamp, the authors **classify** them instead.

**The classifier.** An LLM judge — **GPT-4.1** — reads each (instruction,
response) pair and returns a JSON verdict: `{"label": 0/1, "confidence": 0–10,
"suspected term": "..."}`. The full prompt is reproduced on paper p.5. It asks:
does this conversation reference *any* concept, company, product, technology,
event, or terminology that was created, or became economically salient, *after
1999*? It is deliberately **conservative**:

- Ambiguous or uncertain cases → label **1** (exclude).
- Low-quality conversations → label **1**.
- Explicit post-1999 entities ("GPT-3", "Kubernetes", "TikTok", "blockchain",
  "COVID-19", "Tesla") → strong indicators of label **1**.

**The admission rule.** Keep an example only if `label == 0` **and**
`confidence == 10` — i.e., "knowledge available pre-2000" with *maximal*
certainty. A strict double filter that trades data volume for cleanliness. In the
repo this is `keep_row` / `_parse_label` in `data.py` (implementation-notes §3).

**Why a *single* pre-2000 screen for all vintages.** This is the elegant
shortcut. The screen is run once at the pre-2000 boundary and reused for every
vintage τ ≥ 1999. It's valid because *pre-2000 ⊆ pre-τ* for all those τ: anything
safe for the 1999 model is automatically safe for the 2010 or 2024 model. So the
filtered corpus is **model-independent** — built and cached once, reused by every
vintage run (implementation-notes §3; the repo only varies `model_repo`). This is
precisely the second equality of **eq. 7** ( `q_{T|D}(t^ift_r)/q_T(t^ift_r) = 1` )
being enforced: every post-cutoff evaluation item gets `t^ift_r = 0`, so the IFT
leakage channel is shut.

The honest cost (authors acknowledge, paper §3.2; note §critical-analysis): one
classifier, one prompt, one binary label/threshold — no inter-judge or
prompt-robustness check is reported. And the single pre-2000 screen "anchors"
*every* vintage to the same historical register, which the authors hypothesize
explains why their cross-vintage Sharpe "envelope" (Fig. 4) is more muted than in
the base ChronoGPT paper.

---

## 6. Results, tables, figures — and how the repo reproduces each

The model is trained with **standard masked next-token cross-entropy** (paper
eq. 9) through the three stages. Each result below maps to a specific repo entry
point. (Code-level detail lives in `06-infer-and-eval.md` and
`07-cli-tracking-hub-figures.md`; here we cover *what each test proves* and *which function runs
it*.)

### Figures 1–2 — SFT loss curves → `figures.py` / `chrono figure --kind 1|2`

- **Figure 1** plots training vs. validation cross-entropy across the three stages
  for one vintage (the paper shows ChronoGPT-1999). You see the characteristic
  steep early drop (rapid adaptation to the instruction format) then gradual
  improvement, and a fresh drop at each stage boundary.
- **Figure 2** overlays *validation* loss across all six vintages per stage. Later
  vintages sit lower (more pretraining data → better LM), consistent with the base
  paper.
- **Repo:** every stage logs token-level cross-entropy on a 5% held-out split to
  `output_dir/metrics.csv` (independent of W&B; implementation-notes §9–10).
  `figures.figure1(run_dir)` and `figures.figure2(run_dirs)` read that CSV;
  CLI: `chrono figure --kind 1 --run …` and `chrono figure --kind 2 --runs …`
  (`figures.py`, `cli.py`).

### Table 2 — U.S. president consistency test → `eval.py:president_test` / `chrono eval`

- **What it shows:** the model is prompted with "U.S. Presidents in chronological
  order … Took office in {year}: President ___" and must name the next president.
  The smoking gun: each ChronoGPT-Instruct-τ gets the *majority of pre-cutoff*
  presidents right (paper reports **67 / 83** across the family) but **0 / 73**
  post-cutoff. It knows real, temporally-appropriate history and is blind to the
  future — exactly what chronological consistency predicts. Baselines (GPT-2,
  Llama-3.2-3B-Instruct, Qwen-1.5-1.8B-Chat) answer post-cutoff years correctly,
  exposing *their* leakage.
- **Repo:** `eval.py:president_test(model, device, cutoff_year)`, with the prompt
  builder `president_prompt`. CLI: `chrono eval --repo … --cutoff 2020`.

### Table 3 — dated world-events test → `eval.py:major_events_test`

- **What it shows:** the companion test, completing dated event descriptions
  (e.g., "In 2020, the global economy was devastated by the disease known as the
  ___" → COVID/coronavirus). Same pattern: strong pre-cutoff, **0 / 76**
  post-cutoff. Paper p.12 text reports **76 of 80** events correct in the
  pre-cutoff window. (Minor caveat: the Table 3 footer totals and the p.12 text
  phrasing differ slightly in the denominator — worth not over-precise quoting;
  the *qualitative* result, zero post-cutoff hits, is unambiguous.)
- **Repo:** `eval.py:major_events_test(model, device, cutoff_year)`. Note that
  the `chrono eval` command runs **both** `president_test` and `major_events_test`
  (`cli.py` imports both), so one command produces the data behind Tables 2 *and* 3.

### Figure 3 — AlpacaEval length-controlled win-rate vs. Qwen-1.5-1.8B-Chat → `eval.py:alpaca_outputs` + `alpaca_winrate`

- **What it shows:** head-to-head instruction-following quality. For each
  AlpacaEval instruction, generate an answer from ChronoGPT-Instruct and from the
  Qwen reference; an automatic judge picks a winner; report the **length-controlled
  (LC) win rate**. Win rates *rise monotonically* with vintage — **12.59%** (1999)
  → 13.19% (2005) → 16.21% (2010) → 16.36% (2015) → 16.56% (2020) → **16.79%**
  (2024). The reading is important: later vintages are simply *better language
  models* (more pretraining text up to their cutoff), **not** evidence of
  leakage — this cleanly disentangles "knowledge recency" from "knowledge of the
  future." The absolute level is modest (~12–17%) because the base saw ~31× fewer
  tokens than Qwen.
- **Repo:** `eval.py:alpaca_instructions` loads the prompts; `alpaca_outputs(repo,
  …, backend="chrono"|"hf")` generates from both the ChronoGPT and Hugging-Face
  reference backends; `alpaca_winrate(model_outputs_json, reference_outputs_json)`
  calls the canonical `alpaca_eval` package to judge; `figures.figure3` draws the
  bar chart. CLI path: `chrono alpaca …` → `chrono winrate …` → `chrono figure
  --kind 3` (`cli.py`). The judge needs an annotator key (e.g. `OPENAI_API_KEY`),
  and `alpaca_eval`'s output column name can vary by version, so the saved output
  JSONs are the stable artifacts (implementation-notes §10).

### (Not reproduced here) Table 4 / Figure 4 — the trading application

The paper's headline economic result — the Lopez-Lira & Tang (2023)-style
FAVORABLE/UNFAVORABLE/UNCLEAR long-short portfolio, realtime Sharpe **0.95** vs.
Llama-3.2-3B-Instruct **1.76** (→ ≈54%) and Qwen-1.5-1.8B-Chat **1.53** (→ ≈62%),
plus the cross-vintage Sharpe "envelope" (Fig. 4) — requires the Dow Jones
Newswire + CRSP data (Jan 2007 – Jul 2023), which is **not** part of this repo.
This repo reproduces the *model and its validation* (Figs 1–3, Tables 2–3); the
trading backtest is a downstream use of the trained vintages. That is by design:
the repo is the *infrastructure* (load any vintage, generate/embed), not the
asset-pricing study itself.

---

## 7. What this replication repo is, and is not

**The gap it fills.** The authors released the base + instruct **weights** (on
`huggingface.co/manelalab`) and the **SFT data** (`ChronoInstruct-SFT`), but **not
the training code**. The paper specifies the *what* (3-stage curriculum, Alpaca
format, masked cross-entropy, pre-2000 screen) and leaves most of the *how*
unspecified. This repo **reconstructs the training pipeline** from the released
base weights, as reusable infrastructure — a clean SFT loop plus a unified
`generate`/`embed` API for any vintage (README).

**Faithfully matched to the paper:**

- 3-stage curriculum, started from a released `chrono-gpt-v1` base
  (README; implementation-notes §6).
- Masked cross-entropy on **response tokens only** (`labels = -100` on the prompt;
  implementation-notes §2) — matches "standard masked cross-entropy," eq. 9.
- The pre-2000 / confidence-10 screen, reproducing the paper's ~425,119 total
  across the three stages (implementation-notes §3). A real bug was found and
  fixed here: Tulu rows store the classifier label as single-quoted Python dicts,
  not JSON, so a `json.loads`-only parser silently dropped them and collapsed
  Stage 3 to ~32k; the `ast.literal_eval` fallback restores the full 356,886.
- Alpaca-style prompt templates, model code vendored from the authors'
  `ChronoGPT_inference.py` (numerically bit-identical, max logit diff 0.0 —
  implementation-notes §6, §1).

**Engineering choices made to fill gaps (be honest about these):**

- **Packing, not padding.** ChronoGPT's `forward` takes only `input_ids` (no
  attention mask), and the GPT-2/tiktoken vocab has *no pad token* — so padded
  examples couldn't be masked out. The repo therefore concatenates examples into
  fixed 1,792-token blocks (the pretraining/TRL convention) instead of padding
  (implementation-notes §4–5). Open item: this **splits ~5.1% of Stage-3 (Tulu)
  examples** across block boundaries (Stages 1–2: 0%); the loss mask is carried so
  no response tokens are dropped, but the model never sees those examples whole.
  Deemed low-priority; a no-split "best-fit" packing refinement is deferred.
- **AdamW, full fine-tuning, gradient checkpointing.** All 1.55B params updated
  (no LoRA); needs ≥80GB (a 40GB card OOMs at batch 1) — implementation-notes §7.
- **"masked" cross-entropy is mildly ambiguous in the paper** — it could mean only
  the causal mask, but the near-universal SFT reading is response-masking, which
  the repo adopts (implementation-notes §2).

**Honest open item on results:** the replication's loss is currently **higher
than the paper's** (the model is under-trained relative to the published run) —
treat the reproduced Figs 1–2 as pipeline-correct but not yet converged to
paper-level numbers. (See `docs/running-guide.md` / implementation-notes for
status.)

---

## 8. Talking to Asaf: six understanding-checks

Substantive, non-sycophantic things you could raise — each demonstrates you read
both the paper and the engineering. Frame them as genuine questions, not praise.

1. **The "masked cross-entropy" reading (eq. 9).** "I read eq. 9's *masked*
   cross-entropy as masking the prompt tokens (loss on responses only), since
   that's the standard SFT recipe — but the word could also just mean the causal
   mask. Did you in fact compute loss only on response tokens?" — This is a real
   ambiguity the replication had to resolve (implementation-notes §2).

2. **The single pre-2000 screen vs. per-vintage screens.** "The one-screen-for-
   all-vintages design is clean because pre-2000 ⊆ pre-τ — but you flag in the
   paper that it 'anchors' every vintage to the same register and may be why the
   Fig. 4 Sharpe envelope is more muted than in the base ChronoGPT paper. Have you
   considered constructing vintage-specific IFT corpora (each screened to its own
   τ) to test that anchoring hypothesis directly?" — This is the natural extension
   the Obsidian note also identifies.

3. **Packing vs. padding given no pad token.** "ChronoGPT has no pad token and the
   forward pass takes no attention mask, so naively one has to pack examples into
   fixed blocks rather than pad. Did your training pipeline pack as well, and if
   so how did you handle Tulu examples longer than the context window — split, or
   best-fit without splitting?" — Shows you understand a real architectural
   constraint (implementation-notes §4–5).

4. **Classifier robustness.** "Everything downstream rides on one GPT-4.1 pass
   with one prompt and a binary label at confidence 10. Did you do any inter-judge
   or prompt-perturbation robustness on the screen, or check how sensitive the
   final trading comparison is to the filter?" — Acknowledged as not-flawless in
   §3.2; no robustness check is reported.

5. **Optimizer choice.** "The base ChronoGPT lineage is modded-nanoGPT, which
   famously uses the Muon optimizer for pretraining. For the SFT stage did you stay
   on Muon or switch to AdamW? (The replication uses AdamW full-FT.)" — Signals you
   know the architecture lineage (implementation-notes §6–7).

6. **Textual vs. informational leakage.** "The formal contract defines membership
   `t_r` at the level of literal corpus membership, but a paraphrase or
   re-reported version of a post-cutoff event could carry the same information
   without any string overlap. Your GPT-4.1 screen actually targets *informational*
   recency ('references concepts salient after τ'), which seems stronger than the
   formal contract — is that gap between the stated guarantee and the operational
   filter intentional?" — A genuinely sharp point from the reading Q&A; it shows
   you engaged with the identification, not just the results.

(One more you could fold into #4 or keep in reserve: the **Table 1 length /
2,513-vs-704-token discrepancy** from §4 — "your Table 1 lists Stage-3 average
length as 2,513; with the GPT-2 tokenizer I get ~700-token means. What unit is the
2,513 in — characters, a different tokenizer, full multi-turn conversations?" A
small, concrete, real question.)
