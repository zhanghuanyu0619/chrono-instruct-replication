# 04 — The Data Pipeline (`data.py`)

This doc is a line-by-line walkthrough of `src/chrono_instruct/data.py`, the module that turns the released `ChronoInstruct-SFT` dataset into the fixed-length, loss-masked token blocks the trainer consumes. It is the most research-sensitive file in the repo: the **temporal screen** here *is* the paper's no-lookahead guarantee, and three other quiet decisions (response masking, example-level splitting, packing) determine whether the validation curve is honest. Read `01-ml-primer.md` first for tokens / tokenization / masking / cross-entropy; this doc assumes those and explains the data-specific mechanics in full.

> Finance framing up front: think of the temporal screen as a **point-in-time / no-look-ahead filter** on a backtest. The whole scientific claim of the paper is "the model never saw any text that postdates its knowledge cutoff τ." `keep_row` is the screen that enforces it. A parsing bug in that screen is exactly analogous to accidentally merging restated (forward-looking) accounting data into a backtest — it silently contaminates the experiment. We will spend real time on it.

## Table of contents
1. [What the released dataset looks like](#1-what-the-released-dataset-looks-like)
2. [Imports, the tokenizer, and constants](#2-imports-the-tokenizer-and-constants)
3. [The Alpaca prompt templates](#3-the-alpaca-prompt-templates-prompt_with_input--prompt_no_input)
4. [Rendering one example: `format_example`](#4-rendering-one-example-format_example)
5. [Tokenizing + masking one example: `encode_example`](#5-tokenizing--masking-one-example-encode_example)
6. [The temporal screen: `_parse_label` and `keep_row`](#6-the-temporal-screen-_parse_label-and-keep_row)
7. [Grouping into curriculum stages: `stage_examples`](#7-grouping-into-curriculum-stages-stage_examples)
8. [Packing: `pack_blocks` and `PackedDataset`](#8-packing-pack_blocks-and-packeddataset)
9. [The example-level split: `load_stage`](#9-the-example-level-split-load_stage)
10. [Loading + inspection helpers: `load_raw`, `source_counts`](#10-loading--inspection-helpers-load_raw-source_counts)
11. [Caching: `_cache_key` and `prepare_stages`](#11-caching-_cache_key-and-prepare_stages)
12. [End-to-end data flow (in prose)](#12-end-to-end-data-flow-in-prose)
13. [Mini-FAQ](#13-mini-faq)

---

## 1. What the released dataset looks like

The module docstring describes the schema of `manelalab/ChronoInstruct-SFT` (the dataset is loaded from the Hugging Face Hub):

```python
1  """Data: load ChronoInstruct-SFT, filter, reconstruct curriculum stages, pack.
2
3  The released dataset has three columns:
4    - `conversation`: a JSON object {instruction, input, output} (arrives as a
5      dict or as a JSON string depending on the loader; we handle both).
6    - `label`: the GPT-4.1 temporal-screen verdict. The paper keeps only pairs
7      classified label 0 ("knowledge available pre-2000") with confidence 10.
8    - `source`: which of the three upstreams the pair came from.
9
10  The temporal screen is a single conservative pre-2000 filter applied ONCE, not
11  per vintage: pre-2000 data is pre-tau for every vintage tau >= 1999, so one
12  filtered corpus is reused across all vintage runs (see `prepare_stages`). The
13  3-stage curriculum is reconstructed by grouping the filtered rows on `source`.
14  Each example is rendered Alpaca-style and the loss is masked to the response
15  span only; examples are packed into fixed-length blocks (the model has no
16  padding-mask support).
17  """
```

Three columns, three jobs:

- **`conversation`** — the actual training example. It is an instruction-tuning triple `{instruction, input, output}`. `instruction` is the task ("Summarize the following filing"), `input` is optional supporting context (the filing text), and `output` is the gold answer the model should learn to produce. This is the standard Stanford Alpaca shape. A subtlety worth noting now: depending on how the HF loader deserializes the column, this arrives either as a Python `dict` or as a JSON *string* — the code handles both (see `format_example`).

- **`label`** — the verdict of the authors' **GPT-4.1 temporal classifier**. For each pair, GPT-4.1 was asked "could the knowledge needed to answer this have existed before 2000?" and returned a small dict like `{"label": 0, "confidence": 10}`. `label 0` = "yes, knowledge available pre-2000"; `label 1` = "no / postdates 2000 / ambiguous." `confidence` runs 0–10, where 10 = certain. This column is what powers the no-lookahead screen.

- **`source`** — which of three upstream corpora the pair came from. This is how the **3-stage curriculum** is reconstructed: the released file is one big flat table, and `data.py` re-partitions it into stage 1 (scratch), stage 2 (self-instruct), stage 3 (Tulu) by matching this string.

Two structural facts from the docstring drive the rest of the file, and both are research decisions, not conveniences:

1. **One screen, reused across all vintages** (lines 10–13). The paper trains a *family* of models, one per cutoff year τ (1999, 2000, …, 2024). You might expect a per-vintage screen ("keep text before τ"). Instead there is a single conservative **pre-2000** screen. The logic: anything available before 2000 is also available before any τ ≥ 1999, so the pre-2000 corpus is a valid (if conservative) training set for *every* vintage. The huge practical payoff is that the filtered, tokenized, packed corpus is **model-independent** — build it once, cache it, reuse it for every vintage run (only the model checkpoint changes). See `prepare_stages` (§11) and `implementation-notes.md` §3.

2. **Response-only masked loss + packing, not padding** (lines 14–16). Each example is rendered in Alpaca format, the loss is computed only over the response span, and examples are concatenated and sliced into fixed-length blocks rather than padded. The "no padding-mask support" parenthetical is the *reason* for packing and is explained in §8 and cross-referenced in `03-model.md`.

---

## 2. Imports, the tokenizer, and constants

```python
18  import ast
19  import hashlib
20  import json
21  import os
22  from dataclasses import dataclass
23
24  import torch
25  import tiktoken
26  from datasets import load_dataset
27
28  ENC = tiktoken.get_encoding("gpt2")
29  EOT = ENC.eot_token  # 50256, used as end-of-response separator
```

The standard-library imports each map to one job below: `ast` for the label-parsing fallback (§6), `hashlib`/`json` for the cache key (§11), `os` for cache paths, `dataclass` for the dataset wrapper (§8).

The third-party imports:
- `torch` — tensors and the `Dataset` base class.
- `tiktoken` — OpenAI's fast tokenizer. We use the **`gpt2`** encoding because ChronoGPT is built on the GPT-2 byte-pair vocabulary (see `03-model.md`). This is not a cosmetic choice: the model's embedding table is indexed by exactly these token ids, so we *must* tokenize with the same scheme the model was pretrained on.
- `load_dataset` from Hugging Face `datasets` — the loader that pulls `ChronoInstruct-SFT` from the Hub.

Line 28, **`ENC`**, is the single shared tokenizer instance for the whole module. Line 29, **`EOT`**, is the *end-of-text* token, id **50256**. In the GPT-2 / tiktoken vocabulary this is the **only** special token — there is famously **no padding token** (see `implementation-notes.md` §5). That single fact is why the pipeline packs instead of pads (§8). Here `EOT` plays two roles that happen to coincide: it terminates each response (teaching the model when to stop generating) *and* it acts as a soft separator between packed examples. See `01-ml-primer.md` for what an EOT/EOS token is conceptually.

> Aside on a common confusion (from `implementation-notes.md` §5): the model config reports `vocab_size = 50304`. That is **vocabulary padding** — 50257 rounded up to a GPU-friendly multiple, adding unused embedding rows — **not** a pad token. The tokenizer never emits an id above 50256.

---

## 3. The Alpaca prompt templates (`PROMPT_WITH_INPUT` / `PROMPT_NO_INPUT`)

```python
31  PROMPT_WITH_INPUT = (
32      "Below is an instruction that describes a task, paired with an input that "
33      "provides further context. Write a response that appropriately completes "
34      "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
35  )
36  PROMPT_NO_INPUT = (
37      "Below is an instruction that describes a task. Write a response that "
38      "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
39  )
```

These are the **Stanford Alpaca** prompt templates, reproduced verbatim from the original `stanford_alpaca/train.py` `PROMPT_DICT` (`implementation-notes.md` §1). They are Python format strings — `{instruction}` and `{input}` get filled in later by `.str.format(...)`.

Why two templates? Many instruction-tuning examples have *no* supporting `input` (e.g. "Write a poem about autumn"). Alpaca's convention is to use a different preamble in that case and to omit the `### Input:` section entirely, rather than render an empty `### Input:\n\n`. The released ChronoInstruct data contains both kinds of rows, so both templates are needed. The choice between them is made per row in `format_example` (§4).

Three details that matter and are easy to skim past:

- **The trailing `\n` after `### Response:`** (end of lines 34 and 38). This is the *one deliberate departure* from the verbatim Alpaca templates. ChronoGPT's own rendering puts the response on the line *after* the `### Response:` marker (paper p.7 and the released `ChronoGPT_instruct.py:extract_response`). Matching that newline is what lets the model's learned "answer starts here" boundary line up at inference time. Get this wrong and you train on a slightly different prompt shape than you generate with.

- **The `\n\n` section separators.** The blank line between sections is part of the format the model learns. Tokenization is exact: these whitespace characters become real tokens, so the boundary between prompt and response is at a specific, reproducible token index — which is precisely what response masking (§5) relies on.

- **The format is locked across the codebase.** `implementation-notes.md` §1 records an A/B test (June 2026): the Alpaca template produced coherent answers while the released model's own `extract_response` format produced "degenerate garbage." So `data.py`, `infer.py`, and `eval.py` all stay on this exact Alpaca format. Treat the template strings as a contract shared between training and inference — do not edit one without the others.

---

## 4. Rendering one example: `format_example`

```python
42  def format_example(conv):
43      if isinstance(conv, str):  # released `conversation` may be a JSON string
44          conv = json.loads(conv)
45      instruction = (conv.get("instruction") or "").strip()
46      inp = (conv.get("input") or "").strip()
47      output = (conv.get("output") or "").strip()
48      prompt = (PROMPT_WITH_INPUT if inp else PROMPT_NO_INPUT).format(instruction=instruction, input=inp)
49      return prompt, output
```

This takes one `conversation` value and returns a `(prompt, output)` pair of plain strings — the fully rendered prompt the model reads, and the gold response it should produce.

- **Line 43–44** handle the dict-or-string ambiguity flagged in the docstring. If the HF loader handed back the column as a JSON string, parse it into a dict first; if it is already a dict, leave it. Defensive, and necessary — the loader's behavior depends on dataset version and config.

- **Lines 45–47** pull the three fields with `(conv.get(...) or "").strip()`. The `.get(...)` returns `None` for a missing key; the `... or ""` coerces both missing *and* explicitly-null fields to an empty string; `.strip()` trims whitespace. The `or ""` matters because a missing/empty `input` is exactly the signal used on the next line.

- **Line 48** is the template selection: `if inp` is truthy only when `input` is a non-empty string after stripping, so rows with real supporting context get `PROMPT_WITH_INPUT` and the rest get `PROMPT_NO_INPUT`. Then `.format(...)` fills the placeholders. (Note `input=inp` is passed even to the no-input template; `PROMPT_NO_INPUT` simply has no `{input}` placeholder, so the argument is harmlessly ignored.)

- **Line 49** returns prompt and output **separately** — not concatenated. That separation is the whole point: the caller (`encode_example`) needs to know exactly where the prompt ends and the response begins, in order to mask the loss correctly.

---

## 5. Tokenizing + masking one example: `encode_example`

```python
52  def encode_example(conv):
53      """Return (token_ids, target_mask) where target_mask is True on response tokens."""
54      prompt, output = format_example(conv)
55      p_ids = ENC.encode(prompt)
56      r_ids = ENC.encode(output) + [EOT]
57      ids = p_ids + r_ids
58      mask = [False] * len(p_ids) + [True] * len(r_ids)
59      return ids, mask
```

This is where text becomes integers and where the **response-only loss** is set up. It returns two equal-length lists: `ids` (the token ids) and `mask` (a boolean per token, `True` exactly where the model should be scored).

Step by step:
- **Line 54** renders the prompt/output strings.
- **Line 55** tokenizes the prompt into `p_ids`.
- **Line 56** tokenizes the output and **appends `EOT`**. The `+ [EOT]` is important on its own: it teaches the model to *emit* the end-of-text token after a complete answer, i.e. to stop. Without it the model would never learn where answers end. The `EOT` is counted as a response token (it is part of `r_ids`), so the model is trained to produce it.
- **Line 57** concatenates into the full sequence `ids = prompt ++ response ++ EOT`.
- **Line 58** builds the parallel mask: `False` for every prompt token, `True` for every response token (including the `EOT`). `len(mask) == len(ids)` by construction.

This mask is what makes the loss "masked cross-entropy" in the response-only sense. The model still *reads* the prompt (it is part of the input sequence and attended to), but it is never *scored* on predicting the prompt tokens — only on predicting the response. The research reason (`implementation-notes.md` §2): we want the model to learn to *produce answers given instructions*, not to memorize or model the fixed instruction text itself. Scoring the prompt would waste capacity on reconstructing inputs the user always supplies.

Note what this function does **not** do: it does not convert `mask` into the `-100` ignore-index labels yet, and it does not apply the next-token shift. Those happen in `pack_blocks` (§8) and in the training loss respectively. Keeping the mask as booleans here makes the packing logic (which slices across block boundaries) cleaner.

---

## 6. The temporal screen: `_parse_label` and `keep_row`

This is the scientific heart of the file. Read it carefully.

### 6a. `_parse_label` — robust verdict parsing

```python
62  def _parse_label(label):
63      """Parse the `label` verdict, tolerant of JSON *and* Python-dict-repr strings.
64
65      The GPT-4.1 verdict is stored inconsistently across sources (verified on the
66      box): scratch and self-instruct use valid JSON ('{"label": 0, ...}'), but
67      Tulu rows use single-quoted Python-dict reprs ("{'label': 0, ...}") that
68      json.loads rejects. Falling back to ast.literal_eval means those rows get
69      screened on their real verdict instead of being silently dropped — which was
70      the cause of Tulu collapsing to ~32k vs the paper's ~357k. Returns a dict, or
71      None if the verdict is genuinely unrecoverable.
72      """
73      if isinstance(label, dict):
74          return label
75      if not isinstance(label, str):
76          return None
77      for parse in (json.loads, ast.literal_eval):
78          try:
79              obj = parse(label)
80              if isinstance(obj, dict):
81                  return obj
82          except (ValueError, SyntaxError):  # JSONDecodeError subclasses ValueError; literal_eval raises both
83              continue
84      return None
```

The `label` column is supposed to be a small dict, but **it is stored inconsistently across the three upstream sources** (verified on the box, `implementation-notes.md` §3):

- `scratch` and `self-instruct` rows store **valid JSON**: `'{"label": 0, "confidence": 10}'` — double quotes, which `json.loads` accepts.
- `tulu` rows store **Python `dict` reprs**: `"{'label': 0, 'confidence': 10}"` — single quotes, which `json.loads` **rejects** (JSON requires double quotes).

Why this is a five-alarm bug and not a cosmetic one: the original screen used `json.loads` only. Every Tulu row therefore failed to parse and was silently dropped, collapsing Tulu from the paper's ~357k usable rows to ~32k. Because the screen *fails closed* (an unparseable verdict is treated as "drop"), the contamination was invisible — no error, just a much smaller corpus. The tell that isolated it to *parsing* and not the confidence threshold: scratch (1,097) and self-instruct (67,136) matched the paper exactly, so the gap was source-specific, which pointed straight at the single-quote difference.

How the function fixes it, line by line:
- **Lines 73–74**: if `label` is already a dict (some loaders deserialize it), return it as-is.
- **Lines 75–76**: if it is not a dict and not a string, it is unrecoverable — return `None`.
- **Lines 77–83**: the core. Try parsers **in order**: first `json.loads` (fast, strict), then `ast.literal_eval` (Python's safe literal evaluator, which accepts single-quoted dict reprs). The first one that yields a `dict` wins. `ast.literal_eval` is the right tool here — it only evaluates literals (dicts, strings, numbers), never executes arbitrary code, so it is safe on untrusted strings.
- **Line 82**: the `except` catches `(ValueError, SyntaxError)`. This is deliberate and load-bearing: `json.JSONDecodeError` is a subclass of `ValueError`, while `ast.literal_eval` can raise *either* `ValueError` *or* `SyntaxError` on malformed input. Catching both means a failure of the first parser cleanly falls through to the second, and a genuine double-failure falls through to `return None` (line 84).

The net effect: all three stages are now screened on their *real* GPT-4.1 verdict. `implementation-notes.md` §3 records the result — Tulu recovered to 356,886, total to **425,119**, matching the paper. This single `ast` fallback closed the "100k-vs-425k gap."

### 6b. `keep_row` — the keep/drop decision

```python
87  def keep_row(row, min_confidence=10):
88      """Temporal screen: keep pairs the GPT-4.1 classifier marked pre-2000.
89
90      Paper s2.2.1 keeps label 0 with confidence 10. Verdicts that can't be parsed
91      are dropped (the paper's "ambiguity -> label 1" stance). Set `min_confidence`
92      to null in the config to keep every label-0 row regardless of confidence.
93      """
94      obj = _parse_label(row.get("label"))
95      if obj is None or obj.get("label") != 0:
96          return False
97      conf = obj.get("confidence")
98      return min_confidence is None or conf is None or conf >= min_confidence
```

This is the predicate that implements the no-lookahead contract row by row.

- **Line 94** parses the verdict via `_parse_label`.
- **Line 95**: drop the row if the verdict is unparseable (`obj is None`) **or** the label is not 0. Dropping unparseable verdicts is the conservative, correct choice — it matches the paper's stance that ambiguity maps to "label 1" (not pre-2000), i.e. **when in doubt, exclude**. In backtest terms: if you cannot prove a data point was knowable at the time, you do not get to use it.
- **Lines 97–98**: the confidence gate. Keep the row only if `min_confidence is None` (gate disabled), **or** the verdict has no confidence field, **or** confidence ≥ the threshold. With the default `min_confidence=10`, only label-0 rows the classifier was *certain* about survive — the paper's §2.2.1 rule. Setting `min_confidence: null` in `configs/train.yaml` relaxes this to keep every label-0 row regardless of confidence.

Tie it back to the contract: this filter is the operationalization of paper §2.1 eq. 7 (no training text postdating the cutoff). Because pre-2000 ⊆ pre-τ for all τ ≥ 1999, one pass of `keep_row` produces a corpus valid for the entire vintage family — which is why it is applied once in `prepare_stages` (§11), not per model.

---

## 7. Grouping into curriculum stages: `stage_examples`

```python
101  def stage_examples(dataset, sources):
102      """Filter rows whose `source` matches any of `sources` (case-insensitive substring)."""
103      needles = [s.lower() for s in sources]
104      for row in dataset:
105          src = (row.get("source") or "").lower()
106          if any(n in src for n in needles):
107              yield row["conversation"]
```

This reconstructs one curriculum stage by selecting rows whose `source` matches. `sources` is a list of substrings from the config (e.g. stage 2 uses `["self-instruct", "self-generated", "gpt-3"]`).

- **Line 103** lowercases the needles once.
- **Lines 104–107** iterate the dataset, lowercase each row's `source`, and `yield` the `conversation` (only the conversation — the screen already happened upstream) if **any** needle is a substring of it. The substring + case-insensitive match is intentionally forgiving: the exact `source` strings in the release are messy, so matching on a substring like `"tulu"` is more robust than exact equality. The config comments advise running `chrono inspect` (which calls `source_counts`, §10) first to see the real values.

It is a **generator** (`yield`), so it streams rather than materializing a list — the caller (`load_stage`) decides when to realize it.

---

## 8. Packing: `pack_blocks` and `PackedDataset`

### 8a. `pack_blocks`

```python
110  def pack_blocks(examples, block_size):
111      """Concatenate encoded examples into fixed-length (input_ids, labels) blocks.
112
113      labels[t] = input_ids[t] on response tokens, else -100. The shift for
114      next-token prediction is applied in the training loss, not here.
115      """
116      buf_ids, buf_mask = [], []
117      blocks = []
118      for conv in examples:
119          ids, mask = encode_example(conv)
120          buf_ids.extend(ids)
121          buf_mask.extend(mask)
122          while len(buf_ids) >= block_size:
123              chunk_ids = buf_ids[:block_size]
124              chunk_mask = buf_mask[:block_size]
125              labels = [tid if m else -100 for tid, m in zip(chunk_ids, chunk_mask)]
126              blocks.append((chunk_ids, labels))
127              buf_ids = buf_ids[block_size:]
128              buf_mask = buf_mask[block_size:]
129      if buf_ids:  # sub-block tail can't fill a fixed-length block; report rather than drop silently
130          print(f"[data] pack_blocks: dropped {len(buf_ids)} trailing tokens (< block_size={block_size})")
131      return blocks
```

This is the **ConstantLengthDataset-style packing** used in pretraining and in TRL — *not* the one-example-per-sequence padding Alpaca uses.

Mechanism:
- **Lines 116–117**: two running buffers, `buf_ids` (token ids) and `buf_mask` (the response booleans), plus the output list `blocks`.
- **Lines 118–121**: for each example, encode it (which appends its terminating `EOT`, §5) and append both `ids` and `mask` to the buffers. After this the buffer is a long concatenation of `example1 ++ EOT ++ example2 ++ EOT ++ ...`.
- **Lines 122–128**: the draining loop. *While* the buffer holds at least one full block, slice off the first `block_size` (=1792, from config) tokens and the matching mask slice, convert the mask to **labels** (line 125: `tid` where the mask is `True`, else **`-100`**), append the `(chunk_ids, labels)` pair, and drop the consumed prefix from both buffers. The `while` (not `if`) matters: a single very long example can fill several blocks at once.
- **Lines 129–130**: after all examples, whatever is left in the buffer is shorter than a full block. Because the model needs *fixed-length* blocks (no padding mask, §8b), this tail cannot become a block and is dropped — but the count is **printed**, never dropped silently. The tail is at most `block_size - 1` tokens, i.e. negligible.

The `-100` on line 125 is **`IGNORE_INDEX`**. PyTorch's `cross_entropy` takes an `ignore_index` argument (set to `-100` in the training loss, see `05-train.md`); any label position equal to `-100` contributes **zero loss and zero gradient**. So `labels = -100` on prompt tokens means "do not score these," exactly realizing the response-only masking from §5. See `01-ml-primer.md` for the cross-entropy / ignore-index mechanics. The docstring (lines 113–114) also flags that the **next-token shift** is *not* applied here — the input/label alignment shift happens in the loss, so a block's `input_ids` and `labels` are index-aligned at this stage.

**Why pack instead of pad?** (`implementation-notes.md` §4–5, and `03-model.md`.) ChronoGPT's `forward` takes **only `input_ids` — there is no attention mask**, and GPT-2/tiktoken has **no pad token**. If we padded each example to a fixed length, the pad tokens would be attended to and would corrupt the computation, with no mask available to neutralize them. Packing sidesteps the problem: every token in a block is a real token, so the fixed-length requirement is met without any padding. The `EOT` between examples acts as a soft "new example" cue.

**The known cost — honest accounting** (`implementation-notes.md` §4, §7): because we slice on token boundaries, an example that straddles a block boundary is **split** across two blocks. No response tokens are lost (the loss mask is carried correctly), but the model never sees that example whole in a single forward pass, and cross-example attention within a block is **not** masked off (only the causal mask + the `EOT` soft boundary separate examples). This bites **Stage 3 (Tulu)** hardest — Tulu's longest examples exceed 1792 tokens. Measured impact: only **5.1% of Tulu** examples exceed the block size (stages 1–2: 0%); Tulu mean length is 704 tokens with a few long outliers (max 44,808). The authors judged 5% acceptable and kept simple packing; a no-split "best-fit" refinement is documented as deferred (§4 of the notes).

### 8b. `PackedDataset`

```python
134  @dataclass
135  class PackedDataset(torch.utils.data.Dataset):
136      blocks: list
137
138      def __len__(self):
139          return len(self.blocks)
140
141      def __getitem__(self, i):
142          ids, labels = self.blocks[i]
143          return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

A thin `torch.utils.data.Dataset` wrapper so the packed blocks can be fed to a PyTorch `DataLoader` (which handles batching and shuffling in `05-train.md`).

- A `@dataclass` with one field, `blocks` (the list of `(ids, labels)` tuples from `pack_blocks`).
- `__len__` (line 138–139): number of blocks — and since every block is the same length, this is a meaningful unit for steps/epochs.
- `__getitem__` (lines 141–143): converts one block's `ids` and `labels` to `torch.long` tensors on demand. `long` (64-bit int) is required because token ids are indices into the embedding table and labels are class indices for cross-entropy. Converting lazily here (rather than storing tensors) keeps the cached object small and pickle-friendly.

---

## 9. The example-level split: `load_stage`

```python
146  def load_stage(dataset, sources, block_size, val_fraction=0.05, seed=123, val_max_blocks=None):
147      # Split by EXAMPLE before packing, then pack each side separately. Splitting
148      # blocks *after* packing leaks: an example straddling a block boundary could
149      # land its head in train and tail in val, deflating val loss. This holds out
150      # whole examples — "never seen by the optimizer", as the paper requires.
151      examples = list(stage_examples(dataset, sources))
152      g = torch.Generator().manual_seed(seed)
153      perm = torch.randperm(len(examples), generator=g).tolist()
154      n_val = int(len(examples) * val_fraction)
155      val_ex = [examples[i] for i in perm[:n_val]]
156      train_ex = [examples[i] for i in perm[n_val:]]
157      val_blocks = pack_blocks(val_ex, block_size)
158      if val_max_blocks:                 # cap the (random) held-out set so a FULL eval stays cheap
159          val_blocks = val_blocks[:val_max_blocks]
160      return PackedDataset(pack_blocks(train_ex, block_size)), PackedDataset(val_blocks)
```

This builds the `(train, val)` datasets for one stage, and the comment on lines 147–150 is the most important design point in the whole file after the temporal screen.

**Split by example, *then* pack — never the other way around.** Lines 151–156: materialize the stage's examples, build a deterministic random permutation seeded by `seed` (line 152–153, an explicit `torch.Generator` so the split is reproducible), take the first `val_fraction` (default 5%) as validation and the rest as train, **at the example level**. Only *then* (lines 157, 160) does each side get packed independently.

Why the order matters — and why getting it backwards is a silent leak: if you packed first and split blocks afterward, an example straddling a block boundary could land its **head in a train block and its tail in a val block**. The optimizer would then have effectively seen part of every "held-out" example, deflating validation loss and producing a dishonestly optimistic val curve. In backtest terms this is a train/test contamination — the classic sin that makes out-of-sample results meaningless. Splitting whole examples guarantees each validation example is "never seen by the optimizer," which is what the paper's evaluation requires.

The `_cache_key` marker `"split": "example-v2"` (§11) exists precisely to invalidate any old cache that was built with the wrong (block-level) split.

**Lines 158–159, the `val_max_blocks` cap.** The held-out set is random and can be large; capping the number of val blocks keeps each *full* evaluation pass cheap. With `val_max_blocks: 500` (from config), every `eval_every` evaluation covers the *entire* (capped) val set rather than a moving sample, which gives a stable, comparable val curve across steps. Note the cap is applied **after** packing the val examples, so it caps blocks, not examples; this is fine because validation only needs a representative, fixed held-out sample, not a guarantee of whole-example integrity within the cap.

Line 160 returns the two `PackedDataset`s.

---

## 10. Loading + inspection helpers: `load_raw`, `source_counts`

```python
163  def load_raw(dataset_name):
164      return load_dataset(dataset_name, split="train")
```

A one-liner over HF `datasets`: load the named dataset's `train` split (the release ships everything under `train`). This is the single entry point for fetching raw rows; `prepare_stages` calls it.

```python
167  def source_counts(dataset, after_filter=False, min_confidence=10):
168      """Inspect helper: unique `source` values and their row counts.
169
170      With after_filter=True, count only rows passing the temporal screen.
171      """
172      counts = {}
173      for row in dataset:
174          if after_filter and not keep_row(row, min_confidence):
175              continue
176          src = row.get("source") or "<none>"
177          counts[src] = counts.get(src, 0) + 1
178      return counts
```

A diagnostic used by the `chrono inspect` command. It tallies rows per `source` value, optionally **after** applying the temporal screen (`after_filter=True`). This is how you confirm the exact `source` strings (to write correct stage `sources` in the config) and verify the screen reproduces the paper's per-stage counts (scratch 1,097 / self-instruct 67,136 / Tulu 356,886). It is the practical tool that surfaced the Tulu parsing bug in the first place — run it before and after the fix and the Tulu count jumps.

---

## 11. Caching: `_cache_key` and `prepare_stages`

### 11a. `_cache_key`

```python
181  def _cache_key(cfg):
182      payload = {
183          "split": "example-v2",  # bump to invalidate old block-level-split caches
184          "dataset": cfg["dataset"],
185          "block_size": cfg["block_size"],
186          "val_fraction": cfg.get("val_fraction", 0.05),
187          "val_max_blocks": cfg.get("val_max_blocks"),
188          "seed": cfg.get("seed", 123),
189          "min_confidence": cfg.get("min_confidence", 10),
190          "stages": [[s["name"], s["sources"]] for s in cfg["stages"]],
191      }
192      return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
```

This derives a deterministic 16-hex-char cache id from **exactly and only** the inputs that affect the packed data. The payload (lines 182–191) lists every such input: the dataset name, block size, val fraction, val cap, seed, confidence threshold, and the stage definitions. `json.dumps(..., sort_keys=True)` makes the serialization order-independent, then SHA-1 hashes it.

The point: if *any* of these change, the key changes, and the next run rebuilds rather than reusing a stale cache. Conversely, things that do **not** affect the data — learning rate, epochs, batch size, the model checkpoint — are deliberately **absent**, so changing them reuses the cache (the data is model- and hyperparameter-independent).

The `"split": "example-v2"` literal on line 183 is a manual cache-version marker. It is hardcoded so that the move from block-level to example-level splitting (§9) invalidates every previously built cache — a critical correctness flush, since an old cache could contain the leaky split. Bump this string whenever the packing/splitting logic changes in a way that should not silently reuse old artifacts.

### 11b. `prepare_stages`

```python
195  def prepare_stages(cfg):
196      """Build (or load from cache) packed train/val blocks, keyed by stage name.
197
198      Only the tokenized data is cached — never the model or hyperparameters like
199      lr/epochs (those stay live from the config). The data depends solely on
200      (dataset, screen, block_size, stages, seed), so it is built once and reused
201      across every vintage run. Returns {stage_name: (train_ds, val_ds)}.
202      """
203      cache_dir = cfg.get("cache_dir", "cache")
204      path = os.path.join(cache_dir, f"packed-{_cache_key(cfg)}.pt")
205      if os.path.exists(path):
206          print(f"[data] loading cached packed blocks: {path}")
207          return torch.load(path, weights_only=False)
208
209      print(f"[data] building packed cache (one-time, ~10-20 min on first full run) -> {path}")
210      print("[data] loading dataset + applying temporal screen ...")
211      rows = [r for r in load_raw(cfg["dataset"]) if keep_row(r, cfg.get("min_confidence", 10))]
212      print(f"[data] {len(rows):,} rows kept; tokenizing + packing stages ...")
213      stages = {}
214      for s in cfg["stages"]:
215          train_ds, val_ds = load_stage(rows, s["sources"], cfg["block_size"],
216                                        cfg.get("val_fraction", 0.05), cfg.get("seed", 123),
217                                        cfg.get("val_max_blocks"))
218          print(f"[data]   {s['name']}: {len(train_ds):,} train + {len(val_ds):,} val blocks")
219          stages[s["name"]] = (train_ds, val_ds)
220      os.makedirs(cache_dir, exist_ok=True)
221      torch.save(stages, path)
222      print(f"[data] cache saved: {path}")
223      return stages
```

The top-level orchestrator the trainer calls. It returns `{stage_name: (train_ds, val_ds)}` — everything `05-train.md` needs.

- **Lines 203–207, the cache fast path.** Compute the cache file path from `_cache_key`. If it exists, `torch.load` and return immediately. `weights_only=False` is required because the cached object is a dict of `PackedDataset` instances (arbitrary Python objects), not just tensors — `torch.load` defaults to `weights_only=True` for safety, so this opts back in. (Safe here because *we* wrote the file.)

- **Lines 209–212, the build path.** Print a one-time warning (the first full build is ~10–20 min, dominated by tokenizing ~425k examples). Then **apply the temporal screen once** (line 211): load all raw rows and keep only those passing `keep_row`. This is the single screen pass for the entire vintage family — the corpus-wide enforcement of the no-lookahead contract. Print the kept count (should read 425,119 after the parsing fix).

- **Lines 213–219, per-stage build.** For each stage in the config, call `load_stage` (which groups by `source`, splits by example, packs each side) and stash the `(train_ds, val_ds)` under the stage name. The per-stage print line is the human check that each stage's block counts look right.

- **Lines 220–222, persist.** Make the cache dir and `torch.save` the whole `stages` dict so the next run hits the fast path.

**The practical payoff** (`implementation-notes.md` §3): because the cache key excludes the model and hyperparameters, the *same* packed cache is reused across every vintage run — you sweep cutoff years by changing only `model_repo` and `output_dir`, paying the 10–20 min tokenization cost exactly **once**. This is the efficiency justification for the "single conservative screen" design from §1.

---

## 12. End-to-end data flow (in prose)

The full path from the Hub to a training batch, naming the function at each hop. **Important:** steps 1–8 below are the *first-build* path (a cache **miss**). The cache is a **gate in front of this whole pipeline**, not a step inside it — `prepare_stages` first computes `_cache_key(cfg)` and, if the matching `.pt` file already exists, it `torch.load`s the saved `PackedDataset`s and jumps **straight to step 9**, skipping 1–8 entirely. Because `_cache_key` excludes `model_repo`/`lr`/`epochs`, *every vintage run after the first* (and every hyperparameter re-run) takes that fast path — that is where the reuse happens. The cached object **is** the dict of `PackedDataset`s from steps 7–8; there is no separate "packed" vs "cached" data.

1. **Raw rows** — `load_raw(dataset)` pulls all 647,944 rows of `ChronoInstruct-SFT` from the Hub (HF `datasets`).
2. **Temporal screen** — `prepare_stages` filters with `keep_row` (which calls `_parse_label`), keeping the ~425,119 rows the GPT-4.1 classifier marked label 0 / confidence 10. *This is the no-lookahead guarantee.* Applied once for the whole vintage family.
3. **Stage grouping** — for each curriculum stage, `stage_examples` selects the kept rows whose `source` matches, yielding their `conversation` triples.
4. **Example-level split** — `load_stage` shuffles (seeded) and splits those examples into train/val *before* packing, so no example leaks across the boundary.
5. **Render + tokenize + mask** — each example flows through `format_example` (Alpaca prompt) → `encode_example` (tiktoken GPT-2 ids + response mask, terminating `EOT`).
6. **Pack** — `pack_blocks` concatenates the tokenized examples (each + `EOT`) and slices fixed `block_size=1792` chunks, converting the mask to `-100`/id labels; the trailing sub-block tail is reported and dropped.
7. **Blocks → Dataset** — the `(ids, labels)` tuples are wrapped in `PackedDataset` (one per train/val, per stage).
8. **Cache** — `prepare_stages` saves the whole `{stage: (train, val)}` dict, keyed by data-only inputs, for reuse across vintages.
9. **DataLoader** — `05-train.md` wraps each `PackedDataset` in a `DataLoader` for batching/shuffling, and feeds batches to the model (`03-model.md`), where cross-entropy with `ignore_index=-100` scores only the response tokens.

---

## 13. Mini-FAQ

**Q1. Why `-100` specifically for masked tokens?**
`-100` is PyTorch's default `ignore_index` for `cross_entropy`. Any label position equal to `-100` contributes zero loss and zero gradient. Setting prompt tokens to `-100` is therefore exactly how "score only the response" is implemented in code (`pack_blocks`, line 125). It is a convention, not a magic number — but it must match the `ignore_index` passed to the loss in `05-train.md` (it does). See `01-ml-primer.md`.

**Q2. Why pack examples into blocks instead of padding each to a fixed length (the Alpaca way)?**
Because ChronoGPT's `forward` takes only `input_ids` — there is **no attention mask** — and the GPT-2/tiktoken vocabulary has **no pad token**. Padding tokens could not be masked out and would corrupt attention. Packing fills every block with real tokens, meeting the fixed-length requirement without padding. The cost: ~5.1% of Tulu examples (and 0% of stages 1–2) get split across a block boundary, and cross-example attention within a block is not masked. The authors judged this acceptable (`implementation-notes.md` §4, §7); `03-model.md` covers the maskless `forward`.

**Q3. Why one pre-2000 screen for all vintages rather than a per-cutoff screen?**
Pre-2000 text is, by definition, available before *any* cutoff τ ≥ 1999, so a single conservative pre-2000 corpus satisfies the no-leakage contract for the entire 1999–2024 family. It is conservative (a 2010 vintage could legitimately use 2000–2009 data this discards) but **safe**, and it makes the filtered/packed corpus model-independent — built once, cached, reused across every vintage run with only `model_repo` changing. See `prepare_stages` (§11) and `implementation-notes.md` §3.

**Q4. What happens to an example longer than `block_size` (1792 tokens)?**
It is **split** across consecutive blocks by `pack_blocks` (the `while` loop drains the buffer one block at a time). No response tokens are lost — the loss mask is carried with the ids — but the model never sees that example whole in one forward pass. This affects only the long tail of Tulu (max observed length 44,808 tokens). A no-split "best-fit" packing refinement (start a new block, truncate over-length examples) is documented as deferred in `implementation-notes.md` §4.

**Q5. Why split into train/val *before* packing, not after?**
To avoid a silent train/val leak. If you packed first and split blocks afterward, an example straddling a block boundary could land its head in a train block and its tail in a val block — the optimizer would have partially seen "held-out" data, deflating val loss and making the validation curve dishonest. Example-level splitting (`load_stage`) holds out whole examples, the standard the paper requires. The `"split": "example-v2"` cache marker exists to invalidate any cache built with the old leaky scheme.

**Q6. The Tulu stage was collapsing from ~357k to ~32k. What was the root cause?**
A parsing bug, not a data or threshold problem. The `label` column is stored as valid JSON for scratch/self-instruct but as **single-quoted Python dict reprs** for Tulu, which `json.loads` rejects. The original JSON-only parser silently dropped every Tulu row (the screen fails closed). The fix in `_parse_label` is an `ast.literal_eval` fallback, with the `except` catching both `ValueError` and `SyntaxError`. After the fix, Tulu recovered to 356,886 and the total to 425,119, matching the paper. This is the file's most consequential bug — it is the difference between a faithful replication and a quietly broken one.

---

## Addendum (2026-06): why we pack — throughput, not "padding corrupts attention"

An earlier framing (and `implementation-notes.md` §4, now corrected) justified
packing by saying the model has no attention mask, so pad tokens "would corrupt
attention." For this model that reasoning is **overstated**, and it's worth getting
right because the distinction is exactly the kind of thing an expert would probe.

**You never *need* to fill the context window.** A transformer processes any length
`T ≤ block_size`. A 50-token example can be a single length-50 forward pass. Filling
to 1792 is never a *modeling* requirement.

**Padding exists only for *batching*.** To run `B` sequences in one forward pass you
must stack them into a rectangular `(B, T)` tensor; tensors can't have ragged rows,
so the short ones are padded up to a common `T`. With batch size 1 you'd need no
padding at all. Padding is a tensor-shape device, not a model requirement.

**`-100` labels handle the *loss*; the worry is *attention contamination* — and for
a causal, right-padded model it doesn't happen.** Put the filler at the **end**: a
real token at position `i` attends only to positions `≤ i` (causal mask), which are
all real tokens. The trailing pad sits at positions `> i`, so real tokens **never
attend to it**; only the pad positions' own outputs are polluted, and those are
discarded (`-100` + we never read their predictions). Every per-position op here
(RMSNorm, RoPE, value embeddings, U-net skips) preserves this. So right-padding
would be **correct**, even without an attention mask. (The contamination worry is
real for *left*-padding or *bidirectional* models like BERT — not here.)

**So why pack? Efficiency.** Right-padding short examples to 1792 wastes almost all
the compute on pad tokens — Stage-1 examples average ~102 tokens, so ~94% of every
forward would be filler. Packing concatenates real examples (separated by `EOT`) to
fill each block with real tokens, so ~100% of the compute is useful — often a
several-fold speedup. The price is the two costs in Q3/Q4: ~5% of long Tulu examples
are **split** across a boundary, and a packed block allows mild **cross-example
attention** across the `EOT` (a later example can attend to an earlier one —
something padding would avoid).

**The honest three-way summary:**

| Approach | Correct? | Compute waste | Downside |
|---|---|---|---|
| One example per forward (batch 1) | yes | none | tiny batches, poor GPU use, slow |
| Right-pad to `block_size` + `-100` | yes (causal) | high (pad tokens) | wasted compute |
| **Packing (what we do)** | yes | ~none | ~5% examples split; mild cross-example attention |

We pack purely for throughput. `-100` already handles the loss; padding is only
about batching; and for this causal model padding wouldn't even corrupt the real
tokens — packing simply wins on speed. See `03-model.md` (Addendum) for the
causal-mask mechanics and `implementation-notes.md` §4.
