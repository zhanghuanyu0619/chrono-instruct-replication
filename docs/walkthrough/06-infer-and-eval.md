# 06 — Inference and Evaluation (`infer.py` + `eval.py`)

This document is a block-by-block reading of the two files that *use* a trained
ChronoGPT: `infer.py` turns text into either generated continuations or
fixed-length embedding vectors, and `eval.py` runs the three headline tests of
the paper (the president consistency check = Table 2, the dated-events check =
Table 3, and the AlpacaEval length-controlled win-rate = Figure 3). The training
side is covered in the earlier docs; here we are strictly downstream of a saved
checkpoint, and the goal is that you can connect every line of code to the claim
it supports before you talk to Manela.

For the *research meaning* of each test (why no-lookahead matters for asset
pricing, what the paper concludes), see `02-paper-and-research-framing.md`; this
doc stays code-focused. For ML vocabulary (logits, greedy vs sampling, top-k,
embeddings), see `01-ml-primer.md` — terms are reintroduced briefly here but not
re-derived.

## Table of contents

- [Part A — `infer.py`](#part-a--inferpy)
  - [A0. Module docstring and imports](#a0-module-docstring-and-imports)
  - [A1. `load(repo_id, ...)` — get a model onto a device](#a1-loadrepo_id---get-a-model-onto-a-device)
  - [A2. `free_memory()` — releasing GPU RAM](#a2-free_memory--releasing-gpu-ram)
  - [A3. `generate(...)` — the autoregressive loop](#a3-generate--the-autoregressive-loop)
  - [A4. `embed(...)` — text → a fixed-length vector](#a4-embed--text--a-fixed-length-vector)
- [Part B — `eval.py`](#part-b--evalpy)
  - [B0. Module docstring and imports](#b0-module-docstring-and-imports)
  - [B1. `PRESIDENTS` + `president_prompt(...)`](#b1-presidents--president_prompt)
  - [B2. `president_test(...)` — Table 2](#b2-president_test--table-2)
  - [B3. `MAJOR_EVENTS` + `major_events_test(...)` — Table 3](#b3-major_events--major_events_test--table-3)
  - [B4. The AlpacaEval pipeline — Figure 3](#b4-the-alpacaeval-pipeline--figure-3)
- [Mini-FAQ](#mini-faq)

---

# Part A — `infer.py`

## A0. Module docstring and imports

```python
 1  """Inference: unified generation + embedding extraction for any vintage.
 2
 3  The model's forward returns (logits, layer_outputs), so both modalities share
 4  one load path. Generation recomputes the full sequence each step (no KV cache),
 5  matching the original model card's demo.
 6  """
 7  import torch
 8  import torch.nn.functional as F
 9  import tiktoken
10
11  from .model import ChronoGPT
12
13  ENC = tiktoken.get_encoding("gpt2")
```

Two ideas to anchor on before the functions:

- **"any vintage."** The paper trains one model per knowledge cutoff (a
  *vintage*: e.g. the 2019 vintage has seen nothing after 2019). `infer.py` is
  vintage-agnostic — you hand it a checkpoint, it runs. Nothing here knows or
  cares which cutoff produced the weights.
- **One forward, two outputs.** `ChronoGPT.forward` returns a tuple
  `(logits, layer_outputs)` (see `model.py:146` and `docs/implementation-notes.md`
  §6). `logits` drive *generation* (predict the next token); `layer_outputs` (the
  per-layer hidden states) drive *embeddings*. Because the same forward pass gives
  both, `generate` and `embed` can share one `load` path.

`ENC = tiktoken.get_encoding("gpt2")` (line 13) is the GPT-2 byte-pair tokenizer
— the *exact same* tokenizer used in training (`data.py:28`). A tokenizer maps
text ↔ integer token ids. It is critical that inference and training use the
identical encoding, or the model would receive token ids that mean something
different from what it learned. `ENC.eot_token` is id 50256, the
`<|endoftext|>` marker; it is the only special token GPT-2 has (there is no pad
token — see `implementation-notes.md` §5).

## A1. `load(repo_id, ...)` — get a model onto a device

```python
16  def load(repo_id, device=None, cache_dir="cache"):
17      device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
18      model = ChronoGPT.from_pretrained(repo_id, cache_dir=cache_dir).to(device)
19      model.eval()
20      return model, device
```

- **`repo_id` is overloaded on purpose.** `ChronoGPT.from_pretrained`
  (`model.py:204-217`) first checks `os.path.isdir(repo_id)`. So you can pass
  *either* a Hugging Face repo id like `"manelalab/chrono-gpt-v1-20201231"`
  (downloaded and cached under `cache/`) *or* a local run directory like
  `"runs/2026.../final"` produced by your own training. Same call site, both work
  — this is what lets `chrono infer --repo <dir>` point at your replication
  output.
- **Device selection (line 17).** A *device* is where the tensors live and the
  math runs: `"cuda"` is an NVIDIA GPU, `"cpu"` is the processor. `device or ...`
  means "use the caller's device if they passed one, otherwise auto-pick": GPU if
  one is visible, else CPU. On your Lambda A100 box this resolves to `cuda`; on
  your laptop it falls back to `cpu` (slow, but fine for the tiny smoke-test
  model).
- **`.to(device)` (line 18)** physically moves the model's weights onto that
  device. Inputs you feed it later must live on the same device (you'll see
  `device=device` in the tensor constructors below).
- **`model.eval()` (line 19)** flips the model into *evaluation mode*. In ML,
  some layers behave differently during training vs inference (dropout,
  batch-norm statistics). `eval()` selects the inference behavior. This model has
  no dropout, but calling `eval()` is the correct, defensive habit and signals
  intent. Note this is separate from *gradient* tracking — that is handled by the
  `@torch.no_grad()` decorators on the functions below.
- **Returns `(model, device)`** so callers can keep passing `device` to
  `generate`/`embed` without recomputing it.

There is no explicit dtype argument here; the model loads in its saved dtype
(fp32 weights), and the heavy compute is done under a `bfloat16` autocast inside
the forward passes (see A3). `bfloat16` is a 16-bit floating-point format that
halves memory and speeds up matrix multiplies on GPUs at a small precision cost —
standard for LLM inference.

## A2. `free_memory()` — releasing GPU RAM

```python
23  def free_memory():
24      """Release cached GPU memory. Call AFTER `del`-ing your model/tensor refs,
25      e.g. `del model; free_memory()` — useful between loading different vintages."""
26      import gc
27
28      gc.collect()
29      if torch.cuda.is_available():
30          torch.cuda.empty_cache()
```

The motivating problem: each ChronoGPT vintage is 1.55B parameters. If you load
the 2019 vintage, evaluate it, then load the 2020 vintage in the same Python
session (e.g. a notebook looping over vintages, or the AlpacaEval driver), the
first model's memory must actually be returned or the second load OOMs ("out of
memory").

- **`gc.collect()` (line 28).** Python frees objects when their *reference count*
  hits zero. `gc.collect()` forces the garbage collector to run *now*, sweeping
  up objects that are unreferenced (including reference cycles). This is why the
  docstring says call it *after* `del model` — `del` drops your reference, and
  `gc.collect()` then actually reclaims it.
- **`torch.cuda.empty_cache()` (line 30).** PyTorch does not hand GPU memory back
  to the driver the instant a tensor is freed; it keeps a private cache to reuse
  for future allocations (faster). That cached memory is invisible to the next
  `from_pretrained`, which asks the *driver* for fresh memory and can fail even
  though PyTorch is sitting on plenty. `empty_cache()` returns the cache to the
  driver. Guarded by `torch.cuda.is_available()` so it is a no-op on CPU.

**When to call it:** between loading different vintages, or after a big batch of
generation, when you are about to load another large model. You do *not* need it
in a normal single-model script — process exit frees everything. It is a
convenience for long-lived sessions.

## A3. `generate(...)` — the autoregressive loop

This is the heart of the file and the one piece worth reading slowly if
generation is new to you.

```python
33  @torch.no_grad()
34  def generate(model, device, prompt, max_new_tokens=128, top_k=50, temperature=1.0, seed=123,
35               return_completion=False):
36      """Sample a continuation of `prompt`. top_k=1 makes it greedy/deterministic.
37
38      Returns the full decoded text (prompt + completion) by default; with
39      return_completion=True it decodes only the newly generated tokens — sliced by
40      TOKEN count, not by prompt string length, so extraction is exact regardless of
41      tokenizer round-trip whitespace quirks.
42      """
43      ids = torch.tensor(ENC.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
44      n_prompt = ids.shape[1]
45      rng = torch.Generator(device=device).manual_seed(seed)
46      for _ in range(max_new_tokens):
47          with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
48              logits, _ = model(ids)
49          logits = logits[:, -1, :] / temperature
50          probs = F.softmax(logits, dim=-1)
51          topk_p, topk_i = torch.topk(probs, top_k, dim=-1)
52          nxt = torch.gather(topk_i, -1, torch.multinomial(topk_p, 1, generator=rng))
53          if nxt.item() == ENC.eot_token:
54              break
55          ids = torch.cat([ids, nxt], dim=1)
56      out_ids = ids[0, n_prompt:] if return_completion else ids[0]
57      return ENC.decode(out_ids.tolist())
```

**`@torch.no_grad()` (line 33).** Tells PyTorch not to build the autograd graph
(the bookkeeping needed to compute gradients for training). At inference we never
call `.backward()`, so this saves memory and time. Every function in these two
files that runs the model is wrapped this way.

**What "autoregressive" means.** A language model does not emit a whole sentence
in one shot. It predicts *one* next token given everything so far, you append
that token, and feed the longer sequence back in to predict the *next* one. Like
forecasting a time series one step ahead, then rolling the realized value into
the information set and forecasting again. The loop on line 46 is exactly that
roll-forward.

Now line by line:

- **Line 43 — tokenize the prompt.** `ENC.encode(prompt)` turns the prompt string
  into a list of token ids. `torch.tensor(..., dtype=torch.long, device=device)`
  makes it an integer tensor on the model's device. `.unsqueeze(0)` adds a leading
  *batch* dimension, turning shape `(T,)` into `(1, T)` — the model expects a batch
  of sequences, and here the batch is a single sequence. So `ids` has shape
  `(1, n_prompt)`.
- **Line 44 — `n_prompt`** records how many tokens the prompt was. Used at the end
  to slice off the prompt and keep only what the model generated.
- **Line 45 — the RNG.** `torch.Generator(...).manual_seed(seed)` creates a
  dedicated random-number generator with a fixed seed. Sampling (line 52) draws
  from it, so a given `(prompt, seed)` reproduces the same continuation. (With
  `top_k=1` sampling is deterministic anyway — see below — so the seed only
  matters when you actually sample.)
- **Line 46 — the loop** runs at most `max_new_tokens` times. Each iteration adds
  one token (or breaks early on end-of-text).
- **Lines 47-48 — one forward pass.** `with torch.autocast(..., dtype=bfloat16)`
  runs the matrix multiplies in 16-bit for speed/memory. `model(ids)` returns
  `(logits, layer_outputs)`; we keep `logits` and discard the hidden states with
  `_`. `logits` has shape `(1, T, vocab_size)` — for *every* position `t` in the
  sequence, a score for every possible next token. Note the model recomputes the
  *whole* sequence every step (no KV cache; see the module docstring and
  `implementation-notes.md` §6). That is simpler and matches the original model
  card, at the cost of speed — fine for short eval completions.
- **Line 49 — take the last position, apply temperature.** `logits[:, -1, :]`
  keeps only position `-1` (the last token), since that is the distribution over
  *the next* token. Shape becomes `(1, vocab_size)`. Dividing by `temperature`
  rescales the scores: `temperature < 1` sharpens (more confident),
  `> 1` flattens (more random); `1.0` (the default) leaves them unchanged.
  - A subtlety worth knowing for fidelity to the paper: the model already applies
    a *logit softcap* inside `forward` — `logits = 15 * torch.tanh(logits / 15)`
    (`model.py:187`). That squashes raw logits into roughly `(-15, 15)` to keep
    them numerically tame. So the `logits` you get here are already softcapped;
    temperature scales them after that.
- **Line 50 — softmax → probabilities.** `F.softmax` exponentiates and
  normalizes the logits so they sum to 1: now `probs[i]` is the model's
  probability that token `i` comes next.
- **Line 51 — top-k truncation.** `torch.topk(probs, top_k)` keeps only the `k`
  most-probable tokens, returning their probabilities `topk_p` and their vocab
  indices `topk_i`. This is *top-k sampling*: we will only ever pick from these
  `k` candidates, never the long tail of implausible tokens.
- **Line 52 — sample one token.**
  `torch.multinomial(topk_p, 1, generator=rng)` draws one index *into the
  top-k list*, with probability proportional to `topk_p` (so more likely tokens
  are picked more often). `torch.gather(topk_i, -1, ...)` translates that
  position back into the actual vocab id. Result `nxt` has shape `(1, 1)`.
  - **Greedy vs sampling — the `top_k=1` trick.** If `top_k=1`, `topk_p`/`topk_i`
    contain only the single highest-probability token, and `multinomial` over one
    option always returns it. So `top_k=1` is *greedy decoding*: deterministically
    take the argmax every step, no randomness. This is exactly how `eval.py`
    calls `generate` (all eval is greedy). With `top_k=50` (the default) you get
    varied, more natural samples — useful for demos, not for reproducible scoring.
- **Lines 53-54 — stop on end-of-text.** If the sampled token is `eot_token`
  (`<|endoftext|>`, id 50256), the model is signaling "I'm done" — break out of
  the loop without appending it. This is why a 2-token greedy call can return
  fewer than 2 tokens if the model emits EOT early.
- **Line 55 — append and roll forward.** `torch.cat([ids, nxt], dim=1)` glues the
  new token onto the end along the sequence dimension. Next iteration feeds this
  longer `ids` back in — the autoregressive step.
- **Lines 56-57 — decode the output.** Here `return_completion` matters:
  - `return_completion=True`: `ids[0, n_prompt:]` slices off the prompt tokens
    and keeps only the newly generated ones, **by token count** (`n_prompt`).
  - `return_completion=False` (default): `ids[0]` keeps the whole sequence
    (prompt + completion).
  - `ENC.decode(...)` converts token ids back to a string.

  **Why slice by token count, not string length** (the docstring's point):
  tokenizers are not perfectly invertible at the boundary — decoding the prompt
  alone can differ in leading/trailing whitespace from how the prompt's tokens
  sit inside the full sequence. If you tried to chop the completion by
  `len(prompt)` characters you could clip a character or leave a stray space.
  Slicing the *token* array at `n_prompt` is exact: those first `n_prompt` ids are
  precisely the prompt. This is what makes `president_test`/`major_events_test`
  able to match `"Trump"` reliably.

This `return_completion=True` slice is also why this project does **not** use the
released model's `extract_response` parser: per `implementation-notes.md` §1/§12,
`extract_response`'s format produced degenerate garbage, so we keep the Alpaca
template and take the completion by token slice instead. (More in the FAQ.)

## A4. `embed(...)` — text → a fixed-length vector

```python
60  @torch.no_grad()
61  def embed(model, device, text, layer=-1, max_length=1792, pool="mean"):
62      """Return a hidden-state embedding for `text` from the given layer."""
63      token_ids = ENC.encode(text)[:max_length]
64      if not token_ids:
65          raise ValueError("embed() received text that tokenized to zero tokens")
66      ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
67      with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
68          _, layer_outputs = model(ids)
69      h = layer_outputs[layer][0].float()  # (T, model_dim)
70      return h.mean(0) if pool == "mean" else h[-1]
```

**What an embedding is, in finance terms.** Generation uses the *logits* (the
first element of the forward tuple). Embedding uses the *hidden states* (the
second element). A hidden state is the model's internal numeric representation of
the text at each position — a vector in `model_dim`-dimensional space (here 1536).
`embed` collapses a whole document into a single fixed-length vector. Think of it
as a learned, dense analogue of a TF-IDF or LDA topic vector: a 10-K, an earnings
call transcript, or a news article becomes a length-1536 row of numbers you can
drop into a cross-sectional regression as a set of text-derived regressors. That
is precisely the input your downstream return-prediction work needs, and the
no-lookahead property of the vintage model means the vector for a 2018 filing
contains no information leaked from after the cutoff.

Line by line:

- **Line 63 — tokenize and cap length.** `ENC.encode(text)[:max_length]` truncates
  to at most `max_length=1792` tokens (the training block size; the model's usable
  context). Long documents are cut to fit.
- **Lines 64-65 — empty-token guard.** If `text` tokenized to *zero* tokens (empty
  or whitespace-only string), there is no last token to pool and `model(ids)` on
  an empty tensor would crash with an opaque error. The explicit `raise
  ValueError` fails loudly with a clear message — a small but real robustness
  point when you batch-embed thousands of documents and one row is blank.
- **Line 66 — to tensor, add batch dim** (same pattern as `generate`): shape
  `(1, T)`.
- **Lines 67-68 — one forward pass**, keeping the *second* return value this time:
  `layer_outputs`, a Python list with one entry per layer. We discard the logits
  with `_`. (Recall `model.py:177-183` only fills this list when
  `return_hidden=True`, which is the default and is what inference uses; training
  sets it False to save memory — `implementation-notes.md` §6/§7.)
- **Line 69 — pick a layer.** `layer_outputs[layer]` selects one layer's hidden
  states; default `layer=-1` is the last (deepest) layer. `[0]` drops the batch
  dimension (we have one sequence), giving shape `(T, model_dim)` — one vector per
  token. `.float()` casts from bfloat16 back to fp32 so the returned numbers are
  full precision for downstream stats. The comment documents the shape.
- **Line 70 — pooling: token vectors → one document vector.** A `(T, model_dim)`
  matrix has one row per token; we need a single vector for the whole text. Two
  options:
  - `pool="mean"` (default): `h.mean(0)` averages across the `T` token positions —
    every token contributes equally. Robust, the common default for document
    embeddings.
  - otherwise (`pool="last"`): `h[-1]` takes the *last* token's vector. Because
    the model is a causal decoder (each position attends to everything before it),
    the last position has "seen" the whole document, so its vector is a natural
    summary — the GPT-style convention.

  Both return a length-`model_dim` vector. Which pooling predicts returns better
  is an empirical question for your downstream work; the function exposes the knob
  so you can test both.

**Why this is the reusability payoff.** Once you have `embed`, the whole 1999–2024
family of vintage models becomes a set of *point-in-time text encoders*. For any
historical document you can produce a contemporaneously-valid embedding (using the
vintage whose cutoff matches the document's date), with a hard guarantee that the
representation could not have peeked at the future. For look-ahead-bias-sensitive
asset-pricing tests, that guarantee is the entire selling point — and it is four
lines of code on top of the same forward pass generation already uses.

---

# Part B — `eval.py`

## B0. Module docstring and imports

```python
16  import json
17
18  import torch
19
20  from .infer import generate, ENC
```

The docstring (lines 1-15) lays out the three tests and is worth reading as a
map: `president_test` (Table 2), `major_events_test` (Table 3), and the
AlpacaEval flow (Figure 3). Note line 20: `eval.py` imports the *same* `generate`
and `ENC` from `infer.py` — the tests do not re-implement decoding, they call the
one generation path, so anything true of `generate` (greedy when `top_k=1`,
token-exact completion slicing) is automatically true of the tests.

The two consistency tests share a single logic: build a prompt whose correct
completion is a fact with a known *date*, decode greedily, and check whether the
model produced it. The cleverness is in *which* facts and what a "pass" means.

## B1. `PRESIDENTS` + `president_prompt(...)`

```python
22  # (took_office_year, name) in chronological order — public knowledge.
23  PRESIDENTS = [
24      (1993, "Bill Clinton"),
25      (2001, "George W. Bush"),
26      (2009, "Barack Obama"),
27      (2017, "Donald Trump"),
28      (2021, "Joe Biden"),
29      (2025, "Donald Trump"),
30  ]
31
32
33  def president_prompt(history, query_year):
34      """Build the Table 2 prompt: three prior presidents, then `query_year` to fill.
35
36      `query_year` is the TARGET's actual inauguration year (not previous+4): two-term
37      presidents make the gap 8 years, so deriving it arithmetically would mis-date the
38      blank (e.g. asking about 2013, mid-Obama, when the target took office in 2017).
39      """
40      lines = ["U.S. Presidents in chronological order:"]
41      for year, name in history:
42          lines.append(f"Took office in {year}: President {name}")
43      lines.append(f"Took office in {query_year}: President")
44      return "\n".join(lines)
```

- **`PRESIDENTS` (lines 23-30)** is a list of `(took_office_year, name)` pairs in
  chronological order. These are inauguration years, and they are *uneven*: 1993 →
  2001 → 2009 → 2017 are 8 years apart (two-term presidents), but 2017 → 2021 →
  2025 are 4 apart.
- **`president_prompt` (lines 40-44)** renders a few-shot prompt: a header, then
  one line per president in `history` (`Took office in 1993: President Bill
  Clinton`, etc.), then a final line that stops right after `President` so the
  model must fill in the name for `query_year`. Building the string by joining
  lines is plain formatting; the only thing to internalize is the docstring's
  warning.
- **The `query_year` warning (lines 36-38) — read this carefully.** `query_year`
  is the target president's *actual* inauguration year, looked up from the table —
  **never** computed as "last shown year + 4." Because terms can be 8 years apart,
  arithmetic would point at a year with no transition (e.g. 2013, the middle of
  Obama's second term, when the next person to "take office" did so in 2017). The
  whole test depends on the prompt asking about a *real* inauguration date, so the
  year is always pulled from `PRESIDENTS`, not derived. `president_test` enforces
  this by indexing the table directly.

## B2. `president_test(...)` — Table 2

```python
47  @torch.no_grad()
48  def president_test(model, device, cutoff_year):
49      """For each transition, prompt with the three prior presidents and check the prediction.
50
51      Reads exactly two tokens by greedy decoding, as in the paper. Returns a list of
52      dicts; `past_cutoff` flags rows the model should NOT get right if it is
53      chronologically consistent.
54      """
55      results = []
56      for i in range(3, len(PRESIDENTS)):
57          history = PRESIDENTS[i - 3 : i]
58          target_year, target_name = PRESIDENTS[i]
59          completion = generate(model, device, president_prompt(history, target_year),
60                                max_new_tokens=2, top_k=1, return_completion=True).strip()
61          results.append({
62              "target_year": target_year,
63              "target": target_name,
64              "prediction": completion,
65              "correct": target_name.split()[0] in completion,
66              "past_cutoff": target_year > cutoff_year,
67          })
68      return results
```

- **The sliding window (lines 56-58).** For each index `i` from 3 onward,
  `history = PRESIDENTS[i-3:i]` is the *three* presidents immediately before
  position `i`, and `PRESIDENTS[i]` is the target the model must predict. So the
  model always sees three real prior transitions and must name the fourth. Index
  3 is the first target with three predecessors, hence `range(3, ...)`.
- **The generation call (lines 59-60).** Note the arguments:
  `max_new_tokens=2`, `top_k=1`, `return_completion=True`.
  - `top_k=1` → **greedy**: deterministic, reproducible, no seed dependence. The
    test must be greedy so the result reflects the model, not a lucky sample (FAQ).
  - `max_new_tokens=2` → enough tokens to capture a first name like "Donald" or
    "Barack" (GPT-2 BPE may split a name across tokens), since matching only needs
    the first name.
  - `return_completion=True` → returns only the generated tokens (not the long
    prompt), so `completion` is just the model's answer. `.strip()` trims
    whitespace.
- **The result dict (lines 61-67).**
  - `correct` (line 65): `target_name.split()[0] in completion` — does the
    completion contain the target's *first name* (e.g. `"Donald"`)? First-name
    matching is a deliberately lenient, robust check given 2-token greedy output.
  - `past_cutoff` (line 66): `target_year > cutoff_year` — is this transition
    *after* the model's knowledge cutoff?

**What a PASS looks like (this is the conceptual crux).** A chronologically
consistent vintage model should get the **pre-cutoff** rows *correct*
(`correct=True`, `past_cutoff=False`) — it knows the history it was allowed to see
— and should get the **post-cutoff** rows *wrong* (`correct=False`,
`past_cutoff=True`), because the answer lies in its future and it genuinely cannot
know it. So:

| | `past_cutoff=False` (in-sample) | `past_cutoff=True` (future) |
|---|---|---|
| desired outcome | `correct=True` (knows it) | `correct=False` (can't know it) |

A standard pretrained model that absorbed the whole internet would "correctly"
name the post-cutoff president — and that is exactly the *lookahead bias* the
paper warns about. The test passes precisely when the model **fails** the future
rows. Failure on those rows is the *positive* result: it demonstrates the model is
not peeking past its cutoff. `cutoff_year` is passed in by the caller (the vintage
the checkpoint represents), so the same function scores any vintage. The research
interpretation (why this is the right notion of no-lookahead) is in
`02-paper-and-research-framing.md`.

## B3. `MAJOR_EVENTS` + `major_events_test(...)` — Table 3

```python
71  # (event_year, prompt prefix, accepted answer substrings) transcribed from Table 3,
72  # Panel A. The model completes the blank; verify exact wording against the PDF if you
73  # need a byte-faithful reproduction. Accepted terms are matched case-insensitively.
74  MAJOR_EVENTS = [
75      (2001, "The Sarbanes-Oxley Act was introduced in response to the 2001 Enron", ["scandal"]),
76      (2003, "In 2003, a major public health crisis was the outbreak of the virus known as", ["SARS"]),
77      (2008, "In 2008, the global economy was dominated by the subprime mortgage", ["crisis"]),
78      (2016, "In 2016, market volatility increased surrounding the general vote known as the Brexit",
79       ["referendum"]),
80      (2020, "In 2020, the global economy was devastated by the health crisis known as the",
81       ["COVID", "coronavirus", "corona"]),
82      (2022, "In 2022, a major milestone for generative AI was marked by the release of the AI chatbot "
83             "known as", ["ChatGPT", "GPT"]),
84  ]
```

This is the dated-events analogue of the presidents table. Each entry is
`(event_year, prompt_prefix, accepted_answer_substrings)`: a sentence that stops
mid-phrase, plus a list of acceptable completions. Multiple accepted strings
(e.g. `["COVID", "coronavirus", "corona"]`) absorb phrasing variation. The comment
flags that the prompts are *transcribed* from the paper's Table 3 Panel A — verify
exact wording against the PDF if you need byte-faithful reproduction (small wording
drift would not change the pass/fail logic, but matters for an exact replication
claim).

```python
87  @torch.no_grad()
88  def major_events_test(model, device, cutoff_year):
89      """Complete a dated-event sentence and check the term (Table 3).
90
91      Reads three tokens by greedy decoding, as in the paper. `past_cutoff` flags events
92      after the model's knowledge cutoff — a chronologically consistent model should fail
93      those.
94      """
95      results = []
96      for event_year, prompt, answers in MAJOR_EVENTS:
97          completion = generate(model, device, prompt,
98                                max_new_tokens=3, top_k=1, return_completion=True).strip()
99          low = completion.lower()
100         results.append({
101             "event_year": event_year,
102             "answer": answers[0],
103             "prediction": completion,
104             "correct": any(a.lower() in low for a in answers),
105             "past_cutoff": event_year > cutoff_year,
106         })
107     return results
```

Same shape as `president_test`, with three differences:

- **`max_new_tokens=3`** (line 98) instead of 2 — event terms can be slightly
  longer (e.g. "subprime mortgage crisis"). Still greedy (`top_k=1`),
  completion-only.
- **Case-insensitive multi-answer match** (lines 99, 104): lowercase the
  completion, and `correct` is True if *any* accepted term appears as a substring.
- **`past_cutoff = event_year > cutoff_year`** (line 105) — identical logic. A
  consistent vintage should complete pre-cutoff events ("Enron … scandal") and
  fail post-cutoff ones (e.g. ChatGPT for a 2019 vintage).

The pass/fail interpretation is exactly Table 2's: correct before the cutoff,
incorrect after. Both tests together (presidents = people, events = facts)
triangulate the same no-lookahead claim across two fact types.

## B4. The AlpacaEval pipeline — Figure 3

The consistency tests show the model *forgot the future correctly*. Figure 3
shows the complementary thing: the model is still a *good instruction-follower*
on ordinary tasks. AlpacaEval measures general answer quality via head-to-head
comparison against a reference model, judged by an LLM.

### `alpaca_instructions(n)` — the 805-prompt set

```python
110 def alpaca_instructions(n=None):
111     """The AlpacaEval instruction set (805 prompts); `n` limits it for quick tests."""
112     from datasets import load_dataset
113     ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)
114     items = [{"instruction": r["instruction"]} for r in ds]
115     return items[:n] if n else items
```

Loads the canonical AlpacaEval evaluation set — 805 diverse instructions ("write
an email…", "explain X…") — and returns them as a list of `{"instruction": ...}`
dicts. `n` caps the count for a quick smoke test; `None` uses all 805. The import
is local (inside the function) so that merely importing `eval.py` does not require
`datasets` to be installed.

### `alpaca_outputs(...)` — generate one model's answers

```python
118 def alpaca_outputs(repo, instructions, generator, backend="chrono", max_new_tokens=256):
119     """Generate AlpacaEval-format outputs ({instruction, output, generator}).
120
121     backend="chrono": our ChronoGPT, prompted with the same Alpaca template used
122     in training. backend="hf": any HF chat model (e.g. the Qwen reference), via
123     its chat template.
124     """
125     if backend == "chrono":
126         from .infer import load
127         from .data import PROMPT_NO_INPUT
128         model, device = load(repo)
129         outs = []
130         for item in instructions:
131             prompt = PROMPT_NO_INPUT.format(instruction=item["instruction"])
132             # Greedy (top_k=1) to match the HF reference's do_sample=False below: the
133             # win-rate must not confound decoding strategy with model quality.
134             completion = generate(model, device, prompt, max_new_tokens=max_new_tokens,
135                                   top_k=1, return_completion=True)
136             outs.append({"instruction": item["instruction"],
137                          "output": completion.strip(), "generator": generator})
138         return outs
139
140     from transformers import AutoModelForCausalLM, AutoTokenizer
141     tok = AutoTokenizer.from_pretrained(repo)
142     model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")
143     outs = []
144     for item in instructions:
145         chat = tok.apply_chat_template([{"role": "user", "content": item["instruction"]}],
146                                        tokenize=False, add_generation_prompt=True)
147         enc = tok(chat, return_tensors="pt").to(model.device)
148         gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
149         completion = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
150         outs.append({"instruction": item["instruction"],
151                      "output": completion.strip(), "generator": generator})
152     return outs
```

One function, two backends, producing the *same* output schema (a list of
`{instruction, output, generator}` dicts — the format `alpaca_eval` expects).
`generator` is just a label string identifying which model produced the answers.

- **`backend="chrono"` (lines 125-138).** Uses our model.
  - Line 131 wraps the raw instruction in `PROMPT_NO_INPUT` — the **same Alpaca
    template used in training** (`data.py:36-39`). This matters: the model was
    fine-tuned to respond to that exact prompt scaffold, so evaluation must use it
    too. (Recall from `implementation-notes.md` §1 we deliberately did *not* switch
    to the released `extract_response` format — it produced garbage.)
  - Lines 134-135 call `generate` with `top_k=1` (**greedy**), `return_completion=True`
    (answer only), `max_new_tokens=256` (room for a full answer, vs the 2-3 tokens
    of the consistency tests). The inline comment is the key methodological note:
    greedy here is chosen to *match* the HF reference's `do_sample=False` below.
- **`backend="hf"` (lines 140-152).** Uses a standard Hugging Face chat model —
  here the Qwen-1.5-1.8B-Chat reference.
  - Line 141-142 load its tokenizer and model. `device_map="auto"` lets
    `transformers` place it on available GPUs; `torch_dtype="auto"` picks its
    native dtype.
  - Line 145-146: `apply_chat_template` wraps the instruction in *that model's*
    chat format (its own special role/turn markers) — different models expect
    different scaffolds, and using the wrong one cripples the model. `tokenize=False`
    returns the formatted string; `add_generation_prompt=True` appends the cue that
    it is the assistant's turn.
  - Line 148: `model.generate(..., do_sample=False)` — `do_sample=False` is
    `transformers`' name for **greedy decoding**. Same decoding strategy as the
    chrono branch, just a different API.
  - Line 149: decode only the newly generated tokens by slicing off the input
    length (`enc["input_ids"].shape[1]`) — the `transformers` equivalent of our
    `return_completion=True` token-slice. `skip_special_tokens=True` drops the
    chat markers.

**Why both backends must be greedy (the load-bearing design choice).** A win-rate
compares two models' answers. If one model sampled (random, possibly more
"creative") and the other was greedy, you could not tell whether a win came from
*model quality* or from the *decoding strategy*. Forcing both to greedy
(`top_k=1` ⟺ `do_sample=False`) removes that confound: any difference in win-rate
is attributable to the models, which is the whole point of the comparison. This is
why the comment on lines 132-133 exists.

### `alpaca_winrate(...)` — judge and report

```python
155 def alpaca_winrate(model_outputs_json, reference_outputs_json):
156     """Length-controlled win-rate (%) of model vs reference, via the alpaca_eval package.
157
158     Delegates judging + the length-controlled regression to the canonical tool so
159     we don't re-implement (and mis-implement) it. The exact return column may vary
160     by alpaca_eval version; the saved output JSONs are the stable artifacts.
161     """
162     from alpaca_eval import evaluate as alpaca_evaluate
163     with open(model_outputs_json) as f:
164         model_outputs = json.load(f)
165     with open(reference_outputs_json) as f:
166         reference_outputs = json.load(f)
167     leaderboard, _ = alpaca_evaluate(
168         model_outputs=model_outputs,
169         reference_outputs=reference_outputs,
170         is_return_instead_of_print=True,
171     )
172     row = leaderboard.iloc[0]
173     return float(row.get("length_controlled_winrate", row.get("win_rate")))
```

- **Delegation, on purpose (lines 158-160).** The judging and the length-control
  statistics are *not* re-implemented here. We read the two saved output JSONs
  (our model's answers and the reference's), hand them to the canonical
  `alpaca_eval.evaluate`, and let it do the work. Re-implementing the LLM-judge
  protocol risks subtly diverging from the published number; delegating keeps us
  comparable to the literature.
- **What `evaluate` does under the hood.** For each of the 805 instructions it
  shows an *annotator* (an LLM judge — needs an API key such as `OPENAI_API_KEY`
  configured for `alpaca_eval`; see `implementation-notes.md` §10) both answers and
  asks which is better. The raw win-rate is the fraction of comparisons our model
  wins.
- **`is_return_instead_of_print=True` (line 170)** makes it return a pandas
  leaderboard DataFrame instead of printing. Line 172-173 take the first row and
  read `length_controlled_winrate`, falling back to plain `win_rate` if that column
  name differs across `alpaca_eval` versions (the docstring's robustness caveat:
  the saved JSONs are the stable artifact, the column name is not). Returns a
  float percent.

**Length-controlled (LC) win-rate, conceptually.** LLM judges have a known bias:
they tend to prefer *longer* answers, somewhat independent of quality. A naive
win-rate therefore rewards verbosity. The length-controlled win-rate fits a
regression that separates the effect of answer length from the effect of the
model identity, and reports the win-rate *as if both models produced
equal-length answers* — i.e. it debiases for the length confound. For a finance
reader: it is like adding response length as a control in a regression so the
"model" coefficient is not contaminated by a length premium. This is the number
the paper reports in Figure 3 (LC win-rate vs Qwen-1.5-1.8B-Chat), and it is why
the function name and docstring emphasize "length-controlled."

The end-to-end Figure 3 flow: `alpaca_instructions(805)` → `alpaca_outputs(...,
backend="chrono")` for our model and `alpaca_outputs(..., backend="hf")` for the
Qwen reference (save both to JSON) → `alpaca_winrate(ours.json, qwen.json)` → bar
chart.

---

# Mini-FAQ

**1. Why is *everything* in eval greedy (`top_k=1` / `do_sample=False`)?**
Because evaluation must measure the *model*, not the dice. Greedy decoding is
deterministic, so the president/event answers and the AlpacaEval outputs are
reproducible and seed-independent, and the head-to-head win-rate cannot be skewed
by one side sampling more adventurously than the other. Recall `top_k=1` collapses
the top-k sampler in `generate` to a pure argmax (A3, line 52), which is exactly
the same strategy as `transformers`' `do_sample=False`.

**2. Why keep the Alpaca prompt format instead of the released `extract_response`?**
Per `implementation-notes.md` §1/§12, we A/B-tested our training-time Alpaca
template against the released model's `extract_response` format on the same
vintage: Alpaca produced a coherent answer, `extract_response` produced
degenerate garbage. Since our weights are bit-identical to the official model
(§6), that was a *format* effect, not a weights bug. So `data.py`, `infer.py`,
and `eval.py` all stay on the Alpaca template, and we extract the answer by token
slice (`return_completion=True`) rather than by parsing.

**3. What exactly does "length-controlled" correct for?**
LLM judges systematically favor longer responses. LC win-rate regresses out the
length effect and reports the win-rate at equalized length, so the score reflects
answer quality rather than verbosity — analogous to adding a length control in a
regression so it does not contaminate the coefficient of interest.

**4. How does `embed` differ from `generate`?**
They are two readouts of the *same* forward pass. `generate` consumes the
**logits** (next-token scores) and loops autoregressively to emit text.
`embed` consumes the **hidden states** (`layer_outputs`) from a *single* forward
pass and pools them into one fixed-length vector — no loop, no token emission. One
produces language; the other produces a regressor you can put in a return-prediction
model.

**5. How do the president/events tests prove *chronological consistency*, not just
ignorance?**
Because each test scores *both* sides of the cutoff. A merely ignorant model would
fail the post-cutoff rows but might also botch the pre-cutoff ones. A
chronologically consistent vintage shows the precise pattern: **correct before the
cutoff** (it learned the history it was allowed to see) and **incorrect after**
(the answer is in its future). The `past_cutoff` flag plus `correct` lets you read
off that 2×2 pattern. Getting the future rows *wrong* is the positive result — it
is the direct evidence of no lookahead bias.

**6. Why does generation recompute the whole sequence each step instead of caching?**
`model.py` deliberately dropped the KV-cache branch to keep the training path
clean (`implementation-notes.md` §6), so `generate` re-runs the full sequence
every token (A3, line 48). It is slower but simpler and matches the original model
card's demo. For eval this is irrelevant — completions are 2-256 tokens — and it
keeps one code path for both training and inference.
