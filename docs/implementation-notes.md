# Implementation Notes & Design Decisions

Why the code does what it does, where each choice came from, and what is still
open. The paper specifies the *what* (3-stage curriculum, Alpaca format, masked
cross-entropy, pre-2000 screen) but not most of the *how* — the authors did not
release training code. This file records the engineering choices we made to fill
those gaps, so they are auditable later and defensible to the authors.

Status legend: **Settled** (matches paper or a cited standard) · **Pending**
(awaiting a decision) · **Verify** (to confirm on the box; see
`notebooks/verify_pipeline.ipynb`).

---

## 1. Prompt format — Stanford Alpaca templates · Settled (with one open variant)

`data.py` renders each example with the Stanford Alpaca `PROMPT_DICT` templates,
**verbatim**, with a single deliberate change: a trailing `\n` after
`### Response:`, because ChronoGPT's own rendering puts the response on the next
line (paper p.7 example, and `ChronoGPT_instruct.py:extract_response`).

- Source: <https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py> (`PROMPT_DICT`).
- **No-input case:** the paper does not specify it; the two-template split
  (`PROMPT_WITH_INPUT` / `PROMPT_NO_INPUT`) is the Alpaca convention — rows with
  an empty `input` field use the no-input template. The released data has both.

**Resolved (June 2026, notebook §12): keep the Alpaca template.** We A/B-tested
our Alpaca template against the released model's `extract_response` format (system
prompt inside `### Instruction:`, always-present `### Input:`) on the same instruct
vintage. Our Alpaca template produced a coherent answer; `extract_response`
produced **degenerate garbage**. So we do NOT switch — `data.py`/`infer.py`/
`eval.py` stay on the Alpaca format. (Our model is also bit-identical to the
official one — notebook §10, max logit diff 0.0 — so this is a format effect, not
a weights issue.)

## 2. Loss on the response only (response masking) · Settled

`pack_blocks` sets `labels = -100` on prompt tokens and the true token id on
response tokens (`data.py`), so cross-entropy (`ignore_index=-100`) scores only
the response. We want the model to learn to *produce answers*, not to model the
fixed user instruction.

- Source: the same Alpaca file (`IGNORE_INDEX = -100`, applied to `label[:source_len]`).
- Paper: "standard **masked** cross-entropy" (eq. 9) — consistent. Caveat:
  "masked" is not 100% unambiguous in the paper (could mean only the causal
  mask), but response-masking is the near-universal SFT reading.

## 3. Temporal screen — one conservative pre-2000 filter, reused across vintages · Settled

The released dataset is unfiltered (647,944 rows). We keep only rows the authors'
GPT-4.1 classifier marked `label 0` ("pre-2000") with `confidence 10`
(`keep_row`), reaching the paper's ~425,119.

- **Single screen for all vintages, not per-vintage.** pre-2000 ⊆ pre-τ for every
  vintage τ ≥ 1999, so one filtered corpus satisfies the no-leakage contract
  (paper §2.1, eq. 7) for the whole 1999–2024 family. We do **not** re-run the
  classifier per cutoff.
- Consequence: the filtered + packed corpus is **model-independent**, so it is
  built once and cached (`prepare_stages`, keyed on data not model) and reused by
  every vintage run — only `model_repo` varies.
- **Verified on box (June 2026):** the `label` field is stored **inconsistently** —
  scratch and self-instruct use valid JSON (`'{"label": 0, ...}'`), but Tulu rows
  use single-quoted Python-dict reprs (`"{'label': 0, ...}"`). The original
  `json.loads`-only parser silently dropped every Tulu row, collapsing it to ~32k
  vs the paper's ~357k (scratch 1,097 and self-instruct 67,136 matched exactly,
  which is what isolated the bug to parsing, not the confidence threshold).
  **Fix:** `_parse_label` now falls back to `ast.literal_eval`, so all three
  stages screen on their real verdict. This was the cause of the 100k-vs-425k gap.

## 4. Packing into fixed blocks — chosen for throughput, not correctness · Settled mechanism, Pending refinement

`pack_blocks` concatenates tokenized examples (each followed by `EOT`) into a
buffer and slices `block_size` (1792) chunks — the pretraining / TRL
`ConstantLengthDataset` convention, **not** Alpaca (which pads one example per
sequence).

- Why pack instead of pad: **efficiency, not a correctness requirement** (an
  earlier version of this note said padding "would corrupt attention" — that is
  overstated; see the correction below). The model's `forward` takes only
  `input_ids` with no attention mask, but with **right**-padding and a causal
  mask, real tokens at positions `< L` attend only to positions `<= their index`,
  which are all real — so trailing pad tokens never enter any real token's
  attention, and `-100` labels keep them out of the loss. Right-padding would
  therefore be *correct*; it is just **wasteful**: Stage-1 examples average ~102
  tokens, so padding each to 1792 spends ~94% of every forward on pad tokens.
  Packing fills each block with real tokens instead (near-100% useful compute) —
  a multi-fold training speedup. (The "no attention mask" point still matters for
  *left*-padding or bidirectional models, where pad **would** contaminate; we
  simply never pad.) See §5, §6.
- **Known costs (your review caught these):**
  1. Examples that exceed a block boundary are **split** across two blocks (the
     loss mask is carried, so no response tokens are dropped, but the model never
     sees such an example whole in one forward).
  2. This bites **Stage 3 (Tulu) hardest**: paper Table 1 average length is
     102 / 183 / **2513** tokens for stages 1/2/3 — Stage 3 averages *above* the
     1792 block, so many Tulu examples are split/truncated.
  3. Cross-example attention within a block is **not** masked (only the causal
     mask + the `EOT` soft boundary).
- **Pending refinement (recommended):** best-fit packing *without* mid-example
  splits — start a new block when the next example won't fit, pad the remainder
  with loss-masked `EOT`, and truncate examples longer than `block_size`
  (unavoidable at this context length). Removes cost #1, keeps efficiency.
- **Verified (June 2026, notebook §7): low priority.** Only **5.1% of Tulu**
  examples exceed 1792 (stages 1–2: 0%); Tulu mean length is 704 tokens (a few
  long outliers, max 44,808). So simple packing splits ~5% of Stage-3 examples —
  acceptable. We keep simple packing; the no-split refinement is deferred unless
  Fig 1/2 show artifacts.

## 5. No `[pad]` token — it's a property of GPT-2 / tiktoken, not a bug · Settled

ChronoGPT uses the GPT-2 `tiktoken` vocabulary, whose only special token is
`<|endoftext|>` (id 50256). There is **no pad token** — GPT-2 never had one.
Combined with the padding-mask-free `forward` (§4), this is *why* we pack instead
of pad.

- Note the easy confusion: the model config's `vocab_size = 50304` is **vocabulary
  padding** (rounding 50257 up to a GPU-friendly multiple) — extra unused
  embedding rows, **not** a pad token. The tokenizer only ever emits ids ≤ 50256.
- **Verify:** print `enc.n_vocab`, `enc.eot_token`, `enc._special_tokens`, and the
  model config `vocab_size` (notebook §2).

## 6. Model code — vendored from `ChronoGPT_inference.py`, two intentional changes · Settled

`model.py` is the authors' model, adapted: (a) every `@torch.inference_mode()`
removed so gradients flow (the released instruct file removes them too), (b) the
KV-cache generation branch dropped. We **keep** the `(logits, layer_outputs)`
return — the instruct file comments out `layer_outputs`, but we need it for
`embed()`, so our version is a strict superset.

- Architecture lineage: modded-nanoGPT (Keller Jordan, 2024) — U-net depth skips,
  value embeddings, x0 skips, RoPE, QK-norm, RMSNorm, ReLU² MLP, logit softcap,
  zero-init projections. It is still a causal decoder-only LM.
- No `transformers` / `AutoModel` integration — it uses `PyTorchModelHubMixin`
  for `save_/from_/push_to_hub`. That is why we hand-roll generation and pushing.
- **Verify:** numerical parity of our `model.py` vs the official
  `ChronoGPT_inference.py` on the same input (notebook §10) — confirms the
  inference_mode removal changed no numerics.

## 7. Full fine-tuning, no PEFT · Settled

All 1.55B parameters are updated (`AdamW(model.parameters())`). The paper does
full SFT ("standard masked cross-entropy"); no LoRA/adapters mentioned.
- **Verified (June 2026): memory is dominated by Adam states + activations, not
  weights.** Model states ≈ 25 GB fixed (params 6 + grads 6 + Adam m,v 12 — the
  16 bytes/param rule). Activations scale with batch×seq×layers and dominate: on a
  **40GB** card full FT OOMs even at batch 1; on an **80GB H100**, `batch_size 8`
  OOMs *in the forward* (~50 GB of activations across 52 layers × 1792 tokens ×
  the 4x squared-ReLU MLP). `batch_size 2` fits 80GB without help.
- **Fix wired (config `grad_checkpoint: true`, default on):** recompute each block
  in the backward pass (`torch.utils.checkpoint`, ~10x less activation memory,
  ~20% slower) **plus** skip retaining `layer_outputs` during training
  (`forward(return_hidden=False)`). With these, `batch_size 8` fits one 80GB card.

## 8. Reproducibility — one global seed · Settled

A single `seed` in `configs/train.yaml` drives all randomness: the train/val
split (`load_stage`), the `DataLoader` shuffle (explicit `Generator`), and
sampling. `run()` does `cfg.setdefault("seed", 123)` so there is one fallback
point; the same seed reproduces a run end-to-end. The same seed is used for every
stage (the split RNG differs per stage only because the data differs).

## 9. Optional integrations — off by default · Settled

- **Weights & Biases:** `wandb.enabled` (default false). Loss curves always go to
  `output_dir/metrics.csv` regardless, so Figures 1–2 never depend on W&B.
- **Hugging Face push:** `push_to_hub.enabled` (default false); also via
  `chrono push`. Needs a write token (`hf auth login` / `HF_TOKEN`), never
  hardcoded.

## 10. Figures · Settled

- **Fig 1/2** plotted from `metrics.csv` (`figures.py`).
- **Fig 3** (AlpacaEval LC win-rate vs Qwen-1.5-1.8B-Chat): generate outputs
  (`alpaca_outputs`, chrono + HF backends) → judge via the canonical
  `alpaca_eval` package (`alpaca_winrate`) → bar chart. The judge needs an
  annotator key (e.g. `OPENAI_API_KEY`); `alpaca_eval`'s return column can vary
  by version, so the saved output JSONs are the stable artifacts.

## 11. Checkpoint storage — must be on persistent FS · Settled (operational)

`train.py` saves to `output_dir/<stage>` and `output_dir/final`. On Lambda, set
`output_dir` to an **absolute path on the persistent filesystem** — the ephemeral
disk is wiped on instance termination. Weights are git-ignored (`*.bin`, `*.pt`,
`runs/`).

---

## Decisions — all resolved by the June 2026 box verification
1. **Prompt format** (§1): **keep Alpaca** — `extract_response` produced garbage (§12).
2. **Packing** (§4): **keep simple packing** — only 5.1% of Tulu splits (§7).
3. **Label parsing** (§3): **fixed** — `ast` fallback recovers Tulu to 356,886;
   all three stages now match the paper (total 425,119). `min_confidence: 10` stays.
4. **Compute** (§7): **needs ≥80GB** for full FT; 40GB OOMs at batch 1.

Remaining to confirm: param dtype (notebook §9 — add `next(model.parameters()).dtype`);
expected fp32, which is correct for AdamW. Not blocking.
