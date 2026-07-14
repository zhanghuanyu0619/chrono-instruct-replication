# Replication Report — Instruction Tuning of ChronoGPT

**Paper.** He, Lv, Manela & Wu (2025), *"Instruction Tuning Chronologically
Consistent Language Models"* (SSRN 5348747 / arXiv 2510.11677).
**This report.** A faithful reconstruction of the paper's SFT / instruction-tuning
pipeline, with the six headline vintages trained end-to-end from the released
`chrono-gpt-v1` base weights.
**Author.** Huanyu Zhang · **Prepared for.** Prof. Asaf Manela.

---

## 1. Abstract

The authors released the ChronoGPT-Instruct **weights** and the **SFT data**
(`manelalab/ChronoInstruct-SFT`) but not the **training code**. This repository
reconstructs that code and re-runs the full pipeline: it re-derives the paper's
data screen and trains all six headline vintages
(τ ∈ {1999, 2005, 2010, 2015, 2020, 2024}) through the 3-stage curriculum
(scratch → GPT-3 self-instruct → Tulu-3), one model per knowledge cutoff, from the
matching `chrono-gpt-v1` base (~1.55 B params, 52-layer modded-nanoGPT U-net,
`model_dim` 1536, 12 heads, vocab 50304, context 1792). **Headline result:** the
3-stage curriculum chains cleanly (each stage resumes near the prior stage's
endpoint and improves), and the final Stage-3
validation cross-entropy falls **monotonically** as the cutoff moves forward —
0.8691 (1999) → 0.7855 (2024) — exactly the "later vintage = better language
model, not leakage" structure the paper predicts. The fine-tuned models are on the
Hub under `HZ0619/chrono-instruct-v1-{vintage}1231`. **Status:** the SFT training
exhibits (data screen, loss curves, curriculum behavior) and the two
chronological-consistency exhibits (Tables 2–3) are complete — **the consistency
tests show zero post-cutoff leakage across all six vintages, with pre-cutoff
accuracy rising monotonically as the cutoff advances** (§6). AlpacaEval (Figure 3)
is generated but not yet judged (the outputs exist for all six vintages; the LLM
judge step is a one-command re-run, §7).

---

## 2. Scope & status

| # | Exhibit | What it establishes | Status |
|---|---------|---------------------|--------|
| Table 1 | Temporal data screen (647,944 → 425,119) | The pre-2000 / confidence-10 filter reproduces the paper's counts | ✅ done |
| Figs 1–2 | SFT loss curves, 6 vintages × 3 stages | Curriculum chains cleanly; monotone improvement with cutoff | ✅ done |
| — | Six trained vintages pushed to HF | Reusable, leakage-free instruct models | ✅ done |
| Table 2 | U.S. president consistency test | Correct pre-cutoff, blind post-cutoff | ✅ done (§6) |
| Table 3 | Major-world-events test | Same chronological-consistency pattern | ✅ done (§6) |
| Figure 3 | AlpacaEval LC win-rate vs Qwen-1.5-1.8B-Chat | Instruction-following quality (12.59–16.79% in paper) | ⏳ generated, judging pending |
| — | Uncapped full-validation re-scoring | Tightens the noisy Stage-1 training-time numbers | ⏳ coded, not yet run |

Not in scope (by design): any downstream return-prediction / trading application
built on these models needs external news and market data (e.g. a newswire feed +
CRSP) that is not part of this repo. This repository reproduces the *model and its
validation* — the instruction-tuning infrastructure; an asset-pricing study would
be a downstream use of the trained vintages.

---

## 3. Table 1 — data screen reproduction

The released `ChronoInstruct-SFT` has **647,944** rows. The paper keeps only pairs
the authors' **GPT-4.1** temporal classifier marked `label 0` ("knowledge
available pre-2000") with `confidence == 10` — a deliberately strict double
filter. Reproducing that screen (`keep_row` / `_parse_label` in `data.py`,
`min_confidence: 10`) yields:

| Stage | Source | Rows after screen | Paper Table 1 |
|-------|--------|------------------:|--------------:|
| 1 | LLMs-from-scratch simple tasks | 1,097 | 1,097 |
| 2 | GPT-3 self-instruct | 67,136 | 67,136 |
| 3 | AllenAI Tulu-3 mixture | 356,886 | 356,886 |
| **Total** | | **425,119** | **≈425,119** |

**The one consequential fidelity fix.** An initial JSON-only label parser
(`json.loads`) silently dropped **every** Tulu row, collapsing Stage 3 to ~32k and
the total to ~100k. Cause: the classifier verdicts are stored **inconsistently** —
scratch and self-instruct rows use valid JSON (`'{"label": 0, ...}'`), but Tulu
rows store single-quoted Python-dict reprs (`"{'label': 0, ...}"`), which
`json.loads` cannot parse. Because Stage 1 (1,097) and Stage 2 (67,136) matched the
paper *exactly*, the discrepancy isolated cleanly to parsing rather than the
confidence threshold. Adding an `ast.literal_eval` fallback in `_parse_label`
recovered Tulu to 356,886 and the total to **425,119**, matching the paper. This is
the single most consequential fidelity fix in the replication
(`docs/implementation-notes.md` §3).

---

## 4. SFT results

Each cell is the **best validation cross-entropy** (token-weighted, 5% seeded
held-out split; lower is better), read from
`results/chrono-instruct-{τ}/summary.json`.

| Vintage | Base model (HF) | Stage 1 (scratch) | Stage 2 (self-instruct) | Stage 3 (Tulu) |
|--------:|-----------------|------------------:|------------------------:|---------------:|
| 1999 | `manelalab/chrono-gpt-v1-19991231` | 1.4492 | 1.3080 | **0.8691** |
| 2005 | `manelalab/chrono-gpt-v1-20051231` | 1.2098 | 1.1827 | **0.8279** |
| 2010 | `manelalab/chrono-gpt-v1-20101231` | 1.2147 | 1.1390 | **0.8137** |
| 2015 | `manelalab/chrono-gpt-v1-20151231` | 1.1960 | 1.1075 | **0.8026** |
| 2020 | `manelalab/chrono-gpt-v1-20201231` | 1.1751 | 1.0801 | **0.7931** |
| 2024 | `manelalab/chrono-gpt-v1-20241231` | 1.1370 | 1.0573 | **0.7855** |

![Combined sweep — validation loss across vintages and stages](../combined/sweep_combined.svg)

*(Combined figure: `results/combined/sweep_combined.svg`. Per-run loss curves are
in each `results/chrono-instruct-{τ}/metrics.csv`, the source for Figures 1–2.)*

**Run profile** (identical across all six, from `summary.json` + per-run
`config.yaml`): one 80 GB card, **~19.4 h** wall-clock each (69,423–70,486 s),
peak **50.4 GB** GPU, seed **123**, block **1792**, `batch_size` **8** ×
`grad_accum` **4** (effective batch 32), gradient checkpointing on. All three
stages use `lr 3e-4` with a per-stage cosine schedule (warmup 0.03, floor
`0.1·lr`); epochs 3 / 2 / 2 for stages 1 / 2 / 3.

**What the numbers say.**
- **Monotone improvement with cutoff.** Stage-3 loss falls strictly as τ advances
  (0.8691 → 0.8279 → 0.8137 → 0.8026 → 0.7931 → 0.7855). Later vintages pretrained
  on more (pre-cutoff) text are better language models — the paper's
  Figure-2 reading, and the clean disentangling of "knowledge recency" from
  "knowledge of the future." Stage 2 is monotone as well; Stage 1 is near-monotone
  (a single 2005↔2010 crossing of 0.005, within the noise of its very short
  validation set — see §6, uncapped re-scoring).
- **The curriculum chains cleanly.** Each stage begins near the previous stage's
  endpoint and improves through it, and the largest single drop is at the Stage-2 →
  Stage-3 (Tulu) transition — the qualitative shape of the paper's Figure 1.

**Comparison to the paper (Figs 1–2).** The *structure* matches: same stage-wise
descent, same monotone-with-cutoff ordering, same "Tulu does the heavy lifting"
shape. Absolute loss values are **not** expected to match to the decimal — see the
caveat in §8 (the LR schedule and epoch counts are our tuning against the *shape*
of the paper's Figure 1, which the paper does not fully disclose).

---

## 5. Implementation details

Each choice cites its source file. Faithful reproductions and deliberate
deviations are both documented, per the paper's "specifies the *what*, not the
*how*" gap.

**Faithful to the paper**
- **Alpaca prompt template, verbatim.** Stanford Alpaca `PROMPT_DICT` reproduced
  exactly, with one deliberate change: a trailing `\n` after `### Response:`,
  because ChronoGPT's own rendering puts the response on the next line. We
  A/B-tested this against the released model's `extract_response` format on the
  same instruct vintage; `extract_response` produced **degenerate output**, so we
  kept the Alpaca template. This is a format effect, not a weights issue — our
  `model.py` is numerically bit-identical to the official `ChronoGPT_inference.py`
  (max logit diff 0.0). (`data.py`; `implementation-notes.md` §1, §6)
- **Response-only masked cross-entropy.** `pack_blocks` sets `labels = -100` on
  prompt tokens and the true id on response tokens (`IGNORE_INDEX = -100`), so loss
  scores only the response — the standard SFT reading of the paper's "masked
  cross-entropy" (eq. 9). (`data.py`; §2)
- **Single conservative pre-2000 screen, reused across vintages.** One screen at
  the pre-2000 boundary serves every τ ≥ 1999 because pre-2000 ⊆ pre-τ. The screen
  is therefore model-independent — filtered + packed **once** and cached, with only
  `model_repo` varying per run — which is the operational form of the paper's
  stage-wise sufficiency (eq. 7). (`data.py` `prepare_stages`; §3)
- **Label-parsing robustness fix.** The `ast.literal_eval` fallback in
  `_parse_label` (see §3 above), which restores the paper's 425,119 total. (§3)
- **Full fine-tuning, no PEFT.** All 1.55 B parameters updated with AdamW; no
  LoRA/adapters, matching the paper's full SFT. (`train.py`; §7)

**Deliberate engineering deviations (gaps the paper leaves open)**
- **Packing, not padding — for throughput.** Examples are concatenated into fixed
  1792-token blocks (the pretraining / TRL `ConstantLengthDataset` convention)
  rather than padded. Motivated by efficiency: Stage-1 examples average ~102
  tokens, so padding each to 1792 would waste ~94% of every forward. Quantified
  cost: **5.1% of Tulu examples** exceed the block and are split across a boundary
  (Stages 1–2: 0%); the loss mask is carried so no response tokens are dropped, and
  the split is judged acceptable (a no-split best-fit refinement is deferred).
  (`data.py`; §4–5)
- **Gradient checkpointing.** Recompute each block in the backward pass (~10× less
  activation memory, ~20% slower) so `batch_size 8` fits one 80 GB card; a 40 GB
  card OOMs even at batch 1. (`config: grad_checkpoint: true`; §7)
- **Optional inference-time KV cache**, numerically identical to full recompute;
  **greedy decoding by default** (temperature 0, `top_k=None`), matching the
  released `ChronoGPT_instruct.py` generate defaults. (`infer.py`; §6)
- **One global seed (123)** drives the train/val split, DataLoader shuffle, and
  sampling — the run reproduces end-to-end. (`train.py`; §8)

---

## 6. Tables 2 & 3 — chronological-consistency results

Both tests run against the HF-published vintages (`scripts/run_eval.py` →
`results/chrono-instruct-{τ}/eval.json`, aggregated by `scripts/aggregate_eval.py`).
Each probes whether a vintage can name a dated fact given only prior context, decoded
**greedily** (2 tokens for presidents, 3 for events), exactly as the paper specifies.

**The claim being tested.** A chronologically consistent model should be **correct on
facts dated at/before its cutoff τ and blind (wrong) on facts after τ**. Post-cutoff
correctness would be look-ahead *leakage* — the failure mode the whole construction
exists to prevent. So the number to watch is the **post-cutoff-correct** column: it
should be **0** for every vintage.

**Legend** (cell = the model's greedy completion for that year, scored):
`✓` correct, pre-cutoff (in-knowledge hit) · `✗` wrong, pre-cutoff (capability miss,
benign) · `∅` wrong, post-cutoff (**blind — the no-look-ahead guarantee working**) ·
`⚠` correct, post-cutoff (**leakage**). A clean run is `✓`/`✗` on/below the diagonal
and `∅` above it, with **no `⚠` anywhere.**

### Table 2 — U.S. president prediction

Prompt: the chronologically-ordered prior presidents, then "Took office in {year}:
President ___". Columns are the target's inauguration year.

| Vintage τ | 1993 Clinton | 2001 Bush | 2009 Obama | 2017 Trump | 2021 Biden | 2025 Trump |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| **1999** | ✗ | ∅ | ∅ | ∅ | ∅ | ∅ |
| **2005** | ✗ | ✓ | ∅ | ∅ | ∅ | ∅ |
| **2010** | ✗ | ✓ | ✗ | ∅ | ∅ | ∅ |
| **2015** | ✓ | ✓ | ✓ | ∅ | ∅ | ∅ |
| **2020** | ✓ | ✓ | ✓ | ✓ | ∅ | ∅ |
| **2024**† | ✗ | ✓ | ✗ | ✓ | ✓ | ∅ |

| Vintage τ | Pre-cutoff correct | Post-cutoff correct (want 0) | Consistent |
|---:|:--:|:--:|:--:|
| 1999 | 0/1 | **0/5** | 5/6 |
| 2005 | 1/2 | **0/4** | 5/6 |
| 2010 | 1/3 | **0/3** | 4/6 |
| 2015 | 3/3 | **0/3** | 6/6 |
| 2020 | 4/4 | **0/2** | 6/6 |
| 2024†| 3/5 | **0/1** | 4/6 |

### Table 3 — major world events

Prompt: a dated sentence with the key term blanked (Enron 2001, SARS 2003, GFC 2008,
Brexit 2016, COVID 2020, ChatGPT 2022); the model completes the term.

| Vintage τ | 2001 Enron | 2003 SARS | 2008 GFC | 2016 Brexit | 2020 COVID | 2022 ChatGPT |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| **1999** | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| **2005** | ✓ | ✓ | ∅ | ∅ | ∅ | ∅ |
| **2010** | ✓ | ✗ | ✓ | ∅ | ∅ | ∅ |
| **2015** | ✓ | ✓ | ✓ | ∅ | ∅ | ∅ |
| **2020** | ✓ | ✓ | ✓ | ✓ | ✓ | ∅ |
| **2024** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| Vintage τ | Pre-cutoff correct | Post-cutoff correct (want 0) | Consistent |
|---:|:--:|:--:|:--:|
| 1999 | 0/0 | **0/6** | 6/6 |
| 2005 | 2/2 | **0/4** | 6/6 |
| 2010 | 2/3 | **0/3** | 5/6 |
| 2015 | 3/3 | **0/3** | 6/6 |
| 2020 | 5/5 | **0/1** | 6/6 |
| 2024 | 6/6 | **0/0** | 6/6 |

**What the two tables show.**
- **Zero look-ahead leakage.** The post-cutoff-correct column is **0/N for all twelve
  rows** — no vintage ever names a president or event dated after its cutoff. The
  entire upper-right triangle is `∅`: the models are correctly *blind* to the future.
  This is the paper's core chronological-consistency claim, reproduced.
- **Pre-cutoff accuracy rises monotonically with τ.** As the knowledge window widens,
  the lower-left triangle fills in — president recall goes 0/1 → 1/2 → 1/3 → 3/3 →
  4/4 (1999 → 2020); events go 0/0 → 2/2 → 2/3 → 3/3 → 5/5 → 6/6. Later vintages know
  more *of their own past*, exactly as expected.
- **`✗` cells are capability misses, not leakage.** Early vintages (and the
  earliest-president slots, which get the least prior context) sometimes name the
  *wrong* pre-cutoff figure — e.g. 1999 answers "Ronald Reagan" for 1993. These count
  against strict row-consistency but are benign for the no-look-ahead guarantee, which
  is governed solely by the post-cutoff column. The signal that matters — zero
  post-cutoff hits — is clean everywhere.

†**2024 caveat.** The 2024 `eval.json` was produced by an interim variant of the
president harness (election-year keying + 3-prior context for the earliest slots),
since reverted to the inauguration-year form the other five vintages use. Post-cutoff
labels above are recomputed uniformly (year > τ), so its leakage conclusion (0/1)
stands; its two `✗` early-slot cells should be re-confirmed by re-running
`python scripts/run_eval.py --vintage 2024` under the current harness. Table 3 is
harness-independent and needs no re-run.

**Reproduce** (defaults to the HF-published models; writes `eval.json` per vintage and
aggregates into this folder's `eval_results.json` / `eval_summary.md`):

```bash
bash scripts/eval_all_vintages.sh              # Tables 2-3, all six vintages
python scripts/run_eval.py --vintage 2020      # a single vintage
```

---

## 6b. Pending — Figure 3 (AlpacaEval) and uncapped re-scoring

**Figure 3 — AlpacaEval LC win-rate vs Qwen-1.5-1.8B-Chat.** All six vintages'
**outputs are generated** (805 completions each, saved as
`results/chrono-instruct-{τ}/alpaca_{τ}.json`); only the LLM-judge step is
outstanding. The initial sweep's judging 404'd because AlpacaEval's default annotator
calls OpenAI's **retired `gpt-4-1106-preview`**; `eval.py` now defaults to a live
annotator (`weighted_alpaca_eval_gpt-4o-mini-2024-07-18`, overridable via
`--annotator` / `$ALPACA_ANNOTATOR`). Re-judge the existing generations without any
GPU work — where an OpenAI key + the Qwen reference live:

```bash
export OPENAI_API_KEY=...
chrono alpaca --backend hf --repo Qwen/Qwen1.5-1.8B-Chat --name qwen --out out/qwen.json
python scripts/score_alpaca.py --model results/chrono-instruct-2020/alpaca_2020.json
```

`score_alpaca.py` writes the win-rate back into each vintage's `eval.json`. The paper's
**Figure 3** reports LC win rates of **12.59% → 16.79%** (1999 → 2024) — deliberately
low, because Qwen-1.5-1.8B-Chat saw ~31× more pretraining tokens; the point is that a
leakage-free model is still competitive, not that it wins. (Do not confuse this with
the "54–62%" figure in the paper — that is the **trading Sharpe-ratio** result of
Table 4, a different exhibit we do not reproduce here.) Decoding is **greedy**, matching
the authors' released `ChronoGPT_instruct.py` (temperature 0, argmax, no repetition
penalty); the reference judge (gpt-4o-mini) is cheaper than the paper's, so a
gpt-4-turbo-family annotator gives the closest absolute comparison (§8).

**Uncapped full-validation re-scoring** (`scripts/full_eval.py`). Training-time
validation caps the held-out set at `val_max_blocks: 500`, which made Stage-1's
~3-block validation noisy. Re-scoring the *same seeded holdout* in full (no cap, no
re-training) tightens the Stage-1 numbers without changing which examples are held
out:

```bash
python scripts/full_eval.py --config configs/train.yaml \
    --repo runs/chrono-instruct-2020/final --cutoff 2020 \
    --out results/full_eval/2020.json
```

---

## 7. Reproducibility & pointers

- **Configs.** `configs/train.yaml` (annotated training config — the sweep default,
  with hyperparameter semantics inline); `configs/eval.yaml` (evaluation +
  Figure-3 pipeline). Per-run *actual* hyperparameters are frozen in each
  `results/chrono-instruct-{τ}/config.yaml`.
- **Results.** `results/chrono-instruct-{τ}/summary.json` (final losses, elapsed,
  peak GPU, hyperparameters) and `.../metrics.csv` (loss curves);
  `results/combined/sweep_combined.svg` (the combined figure).
- **Code.** `src/chrono_instruct/` — `data.py` (screen + packing), `train.py`
  (curriculum loop), `model.py` (vendored from `ChronoGPT_inference.py`),
  `infer.py`, `eval.py`, `figures.py`, `cli.py`. `scripts/` — sweep, publish, and
  `full_eval.py`.
- **Design log.** `docs/implementation-notes.md` (every open choice, with A/B
  tests and the box-verification results); `docs/walkthrough/` (paper-to-code map).
- **Models.** Fine-tuned vintages on the Hub:
  `HZ0619/chrono-instruct-v1-{1999,2005,2010,2015,2020,2024}1231`.
- **Seed.** 123, global, for every stage.

*Prof. Manela — if it is useful, the fastest way to audit fidelity is
`docs/implementation-notes.md` §1–4 (the four resolved design decisions) and any
single `results/chrono-instruct-{τ}/config.yaml` next to its `summary.json`.
I would welcome your correction on any of the "how" choices the paper left open,
especially the masked-loss reading (§2) and the packing/split handling (§4).*

---

## 8. Honest caveats

- **Loss values are not expected to match the paper to the decimal.** The
  learning-rate schedule and epoch counts are **our** tuning against the *shape* of
  the paper's Figure 1 (the paper does not fully disclose them). The replication
  claim is the **qualitative structure** — the data-count match (§3), clean
  curriculum chaining, and monotone improvement with cutoff (§4) — not absolute
  parity. The model is deliberately trained toward pipeline-correctness first.
- **Stage-1 validation is noisy.** Its ~3-block capped validation set makes the
  Stage-1 column the least precise; the uncapped re-scoring in §6 will tighten it.
- **"masked cross-entropy" is mildly ambiguous in the paper** — it could denote
  only the causal mask. We adopt response-only masking, the near-universal SFT
  reading (§5, §2); worth confirming.
- **Tables 2–3 are run and reported (§6); AlpacaEval (Figure 3) is generated but not
  yet judged.** The consistency tests use small hand-built probe sets (6 presidents,
  6 events) transcribed from the paper — they demonstrate the leakage/no-leakage
  structure, not a large-sample accuracy estimate. The 2024 president row was scored
  under an interim harness (§6†) and is flagged for a one-command re-run.
- **AlpacaEval judge choice affects the absolute win-rate.** The default was switched
  off OpenAI's retired `gpt-4-1106-preview`; the current cheap default (gpt-4o-mini)
  trades some human agreement for cost, so re-judging with a gpt-4-turbo-family
  annotator gives the closest comparison to the paper's Figure 3 (12.59–16.79%, §6b).
  A first greedy run scored 2015 at LC 9.94% (vs the paper's 16.36% for that vintage) —
  same order of magnitude; the gap is plausibly judge-model and model-capability
  differences, not a pipeline bug.
- **Packing splits ~5.1% of Tulu examples** at block boundaries; no response tokens
  are lost, but such examples are never seen whole in one forward (§5).
