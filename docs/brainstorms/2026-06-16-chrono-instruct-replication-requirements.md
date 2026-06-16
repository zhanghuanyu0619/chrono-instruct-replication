---
date: 2026-06-16
topic: chrono-instruct-replication
---

# Instruct ChronoGPT Replication — Requirements

## Summary

Build a production-grade, reusable Python pipeline that fine-tunes any released
ChronoGPT vintage into an instruction-following model via a faithful replication
of the paper's three-stage curriculum SFT, and exposes a unified inference API
for both text generation and embedding extraction across vintages. Ship it as a
public repository — the training code (which the authors did not release) plus
reproduced evaluation (AlpacaEval win-rate curve and president-prediction
consistency) as proof of faithfulness. Validate one vintage end-to-end first,
then scale to the full sweep.

## Problem Frame

He, Lv, Manela, and Wu (2025) release the ChronoGPT-Instruct *weights* and the
*SFT data* on Hugging Face, but not the training code. Anyone wanting to
reproduce the instruction-tuning stage, vary it, or build on it must reconstruct
the pipeline from the paper's prose. That gap is the opportunity here.

The work serves two ends at once. First, it is durable research
infrastructure: the eventual goal is text-based stock-return prediction, which
needs any ChronoGPT vintage callable on demand for either generation or
embeddings, free of lookahead bias. Second, it is a credibility artifact — a
clean, verified reproduction of the training pipeline is the kind of concrete
contribution that opens a conversation with Asaf Manela (a former computer
engineer), where polite acknowledgement of already-public weights would not.

The financial application itself is out of reach for now (the Dow Jones Newswire
data is proprietary), so the deliverable is the pipeline and its verification,
not a trading result.

## Key Decisions

- **Reusable library, not one-off scripts.** The pipeline is built as an
  installable package with a stable model-loading and inference API, because it
  is meant to be reused for downstream prediction tasks, not discarded after the
  figures are reproduced.
- **Faithful reproduction is the credibility hook.** Success is measured against
  the paper's published results, not just "code that runs." The training code is
  the unreleased gap, so it is the centerpiece of what gets shown.
- **Full evaluation in scope (Fig 3 + Table 2).** Reproduce the AlpacaEval
  length-controlled win-rate curve across vintages and the president-prediction
  consistency test. This requires multi-vintage sweep machinery and an
  LLM-judge dependency for AlpacaEval — accepted because the sweep machinery is
  wanted regardless.
- **Unified inference across modalities.** The inference layer supports both
  generation (instruct models) and hidden-state/embedding extraction (base
  models) selected by config, so the downstream return-prediction direction is
  not boxed in.
- **Start from released bases; reconstruct hyperparameters.** Only the
  instruction-tuning stage is replicated, starting from released `chrono-gpt-v1`
  vintages. Training hyperparameters absent from the paper are reconstructed and
  documented.
- **One vintage end-to-end before the sweep.** The first milestone is a single
  vintage taken all the way through train → eval → demo, to de-risk the pipeline
  before spending compute on every vintage.

## Requirements

### Training pipeline

- R1. Reproduce the three-stage curriculum SFT in order: Stage 1 LLMs-from-scratch
  (~1,097 examples), Stage 2 GPT-3 self-instruct (~67,136), Stage 3 Tulu-3 SFT
  mixture (~356,886), sourced from the released `ChronoInstruct-SFT` dataset.
- R2. Apply Alpaca-style prompt formatting and the masked cross-entropy
  next-token objective (loss on response tokens), matching the paper.
- R3. Hold out a 5% validation split per stage and log token-level
  cross-entropy, so training dynamics can be compared to the paper's Figure 1.
- R4. Start each run from a released `chrono-gpt-v1-<cutoff>` base via
  `trust_remote_code` loading of the custom architecture.
- R5. Drive a run entirely from a single config (vintage, stage selection,
  hyperparameters, output location) with no code edits between runs.
- R6. Support sweeping multiple vintages (target set: 1999, 2005, 2010, 2015,
  2020, 2024) through the same config-driven entry point.
- R7. Checkpoint to durable storage at a configurable interval and resume cleanly
  from a checkpoint after interruption.

### Inference

- R8. Load any vintage (base or instruct) and run text generation from a prompt.
- R9. Extract hidden-state embeddings for arbitrary input text from any vintage.
- R10. Select vintage and modality (generate / embed) by configuration through one
  stable API, without bespoke per-model code.

### Evaluation and faithfulness

- R11. Reproduce the AlpacaEval length-controlled win-rate against
  Qwen-1.5-1.8B-Chat across the swept vintages (Figure 3).
- R12. Reproduce the president-prediction next-token consistency test across
  vintages (Table 2), demonstrating no lookahead bias past each cutoff.
- R13. Record reproduced metrics alongside the paper's published values so the
  gap is explicit and inspectable.

### Engineering standard

- R14. Distribute as a pip-installable package with a documented public API.
- R15. Pin dependencies and fix random seeds so a run is reproducible.
- R16. Integrate experiment logging (e.g. Weights & Biases) for losses and eval
  metrics.
- R17. Provide unit tests plus a fast end-to-end smoke test (tiny data, few
  steps) that runs without a large GPU.
- R18. Run automated checks in CI (lint, type-check, tests/smoke test).
- R19. Use type hints and a linter/formatter across the codebase.
- R20. Provide a README and a demo notebook covering setup, a training run, and
  inference in both modalities.

### Remote compute workflow

- R21. Document and script environment setup on a rented Lambda Labs A100 (80GB),
  with data, caches, and checkpoints on a persistent filesystem that survives
  instance termination.
- R22. Make long runs resilient to SSH disconnects (e.g. tmux) and resumable from
  checkpoints (see R7).

## Scope Boundaries

### Deferred for later

- Financial return-prediction / trading application, including the Dow Jones
  Newswire ingestion and CRSP merge. The inference API (R8–R10) is the seam this
  will plug into later.

### Outside this project

- Re-pretraining or otherwise reproducing the ChronoGPT *base* models — the
  released bases are taken as given.
- Distillation, RLHF/DPO, or any post-SFT alignment stage not in the paper.

## Dependencies / Assumptions

- Released assets remain available: `manelalab/chrono-gpt-v1-<cutoff>` bases,
  `manelalab/chrono-gpt-instruct-v1-<cutoff>` instruct models (for comparison),
  and the `manelalab/ChronoInstruct-SFT` dataset.
- The custom model architecture loads via `trust_remote_code`; the base may carry
  pretraining-specific code (e.g. a non-standard optimizer) to verify before
  assuming vanilla AdamW SFT works unmodified.
- AlpacaEval reproduction needs an external LLM-judge API and the Qwen-1.5-1.8B-Chat
  reference outputs.
- Access to a Lambda Labs account with A100 (80GB) availability and a persistent
  filesystem in the chosen region.
- A GitHub account for the public repository.

## Success Criteria

- A new vintage can be fine-tuned, evaluated, and queried end-to-end from config,
  by someone following the README, without editing source.
- Reproduced Figure 1 loss curves and Figure 3 win-rates are in the
  neighborhood of the paper's, with deviations explained (R13).
- The president-prediction test reproduces the paper's qualitative result: no
  correct predictions past each model's cutoff (R12).
- The repository reads as production-grade to an engineer: tests pass in CI, the
  API is documented, runs are reproducible.

## Outstanding Questions

### Resolve before planning

- Production-bar scope: confirm the R14–R20 set is the intended level — heavier
  (e.g. Docker, packaged release) or lighter — before the plan commits to it.

### Deferred to planning

- Training framework choice (e.g. Hugging Face `Trainer` vs TRL `SFTTrainer` vs a
  custom loop) and whether the base model's custom code constrains it.
- Reconstructed hyperparameters (LR, schedule, batch size, epochs per stage) and
  how to tune them against the Figure 1 curves.
- Whether to run vintages sequentially on one A100 or in parallel across
  instances, given cost and time.
- AlpacaEval harness specifics (judge model, length-control configuration) and
  its own cost.

## Sources / Research

- He, Lv, Manela, Wu (2025), *Instruction Tuning Chronologically Consistent
  Language Models* — local copy at `~/Downloads/ssrn-5348747.pdf`; arXiv 2510.11677.
- Model size confirmed ~1.55B params (52 layers, 1,536 dim, 1,792 context;
  modified modded-nanoGPT) from the instruct model card.
- Hugging Face org `manelalab` — base, instruct, and `ChronoInstruct-SFT`
  dataset.
- Paper specifics grounded: three-stage curriculum and counts (Table 1),
  masked cross-entropy objective (Eq. 9), 5% hold-out (§3.1), AlpacaEval vs
  Qwen-1.5-1.8B-Chat win-rates 12.59%–16.79% (Fig 3), president consistency
  (Table 2).
</content>
</invoke>
