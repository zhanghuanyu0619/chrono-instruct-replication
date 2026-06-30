# 03 — The Model (`model.py`)

**What this doc is.** A line-by-line, block-by-block reading of
`src/chrono_instruct/model.py`, the definition of the ChronoGPT neural network we
fine-tune. It assumes you have read `01-ml-primer.md` (tokens, embeddings,
attention/QKV, softmax, logits, cross-entropy, residuals, layer norm). Everything
here that is *not* in a textbook GPT — and there is a lot — is explained from
scratch, because this architecture is from the **modded-nanoGPT** lineage (Keller
Jordan, 2024) and has several unusual pieces an expert (Asaf) may quiz you on.

This is the hardest doc in the set. Take it slowly. By the end you should be able
to explain, in your own words, every parameter in the network and why it is there.

---

## Table of contents

1. [Where this file sits](#1-where-this-file-sits)
2. [The high-level shape (sizes and the U-net)](#2-the-high-level-shape)
3. [Imports and the module docstring](#3-imports-and-the-module-docstring)
4. [`norm` — parameter-free RMSNorm](#4-norm--parameter-free-rmsnorm)
5. [`CastedLinear` — fp32 weights, low-precision compute](#5-castedlinear)
6. [`Rotary` — rotary position embeddings (RoPE)](#6-rotary--rope)
7. [`CausalSelfAttention` — QK-norm, RoPE, value-embedding blend](#7-causalselfattention)
8. [`MLP` — squared-ReLU and the 4× expansion](#8-mlp--squared-relu)
9. [`Block` — the pre-norm residual block with x0 skips](#9-block)
10. [`ValueEmbedding` — the shared side-channel into V](#10-valueembedding)
11. [`ChronoGPT.__init__` — assembling the network](#11-chronogpt__init__)
12. [`ChronoGPT.forward` — the U-net pass](#12-chronogptforward)
13. [Save / load / hub](#13-save--load--hub)
14. [`build_tiny`](#14-build_tiny)
15. [How this differs from a vanilla GPT-2 (recap table)](#15-recap-table)
16. [Questions an expert might ask](#16-faq)

---

## 1. Where this file sits

`model.py` is the **model definition** — the pure description of the network's
layers and how a batch of token ids flows through them to produce next-token
logits. It does not train, load data, or generate text; those live in
`train.py` (`05-train.md`), `data.py` (`04-data.md`), and `infer.py`.

The file is **vendored** (copied, with minimal edits) from the authors' released
`ChronoGPT_inference.py` on the Hugging Face model card. Per
`docs/implementation-notes.md` §6 there are two intentional changes:

1. **Every `@torch.inference_mode()` decorator was removed.** `inference_mode`
   tells PyTorch "do not build the autograd graph" — it is a stronger `no_grad`.
   The released file is inference-only and *cannot train as published*: with those
   decorators, the forward pass records no graph, so there is nothing to
   backpropagate through. Removing them is what makes fine-tuning possible.
   (Numerical parity was verified — notebook §10 — so removing them changed no
   numerics, only whether gradients are tracked.)
2. **The KV-cache generation branch was reworked into an *optional* `past`
   argument to `forward`** (off by default, so the training path is untouched).
   The original had a separate caching code path; rather than drop it, `forward`
   now takes an optional per-block cache that `infer.generate` uses to decode one
   token at a time. With `past=None` (training and eval) the forward is identical
   to a clean full-sequence pass. See the **Addendum** at the end of this doc for
   the mechanism, and `06-infer-and-eval.md` for how `generate` drives it. (An
   earlier version of this walkthrough said the cache was simply *dropped* — that
   was true before the 2026-06 update.)

We **keep** the `(logits, layer_outputs)` return signature. The released instruct
file comments out `layer_outputs`; we need it for `embed()` (extracting hidden
states as features for return prediction), so our version is a strict superset.

It is, despite all the modifications below, **still a causal decoder-only language
model**: same job as GPT-2 — read tokens left to right, predict the next one.

> **Reading note (2026-06 update).** The per-method code quotes in the
> walkthrough below show the **no-cache / training-and-eval path** — i.e. the
> behavior when `forward` is called with `past=None`, which is exactly how
> training, `evaluate`, and `embed` call it. The KV-cache update added a few
> *optional* parameters that don't appear in these quotes: `Rotary.forward` gained
> an `offset`, `CausalSelfAttention.forward`/`Block.forward` gained `past` and
> `use_cache`, and `ChronoGPT.forward` gained `past`. They are inactive on the
> path shown here and are documented in full in the **Addendum** at the end of this
> doc. So if you diff these snippets against the current `model.py`, the only
> differences are those optional cache arguments.

---

## 2. The high-level shape

The model we actually fine-tune (`chrono-gpt-v1-*`) is configured as:

| Hyperparameter | Value | Meaning |
|---|---|---|
| `vocab_size` | 50304 | rows in the embedding table (see note below) |
| `num_layers` | 52 | total transformer blocks = **26 encoder + 26 decoder** |
| `num_heads` | 12 | attention heads per layer |
| `model_dim` | 1536 | width of the residual stream (`head_dim` = 1536/12 = **128**) |
| context | 1792 | max tokens per sequence (the data `block_size`) |
| parameters | ~1.55B | headline count from the model card |

**On `vocab_size = 50304`.** GPT-2's tokenizer (`tiktoken`) has 50257 tokens. 50304
is that rounded **up** to a GPU-friendly multiple (a multiple of 128). The extra ~47
rows are never emitted by the tokenizer — they are unused embedding slots, **not**
a pad token. This matters downstream: ChronoGPT has *no pad token at all*, which
is why the data pipeline packs instead of pads (see §16 and `04-data.md`).

**The U-net.** The 52 blocks are not a flat stack. They are split into a 26-layer
**encoder** half and a 26-layer **decoder** half, wired together with **skip
connections** like a U-net (the architecture from image segmentation). Encoder
layer *k*'s output is added back into the matching decoder layer. This, plus
**value embeddings** and **x0 skips**, is the signature of modded-nanoGPT. We work
through each piece below; for now just hold the picture: information flows down
through 26 layers, then back up through 26 layers, with "shortcut" wires across.

**Parameter budget intuition.** The 52 blocks dominate: each block is ~4 square
weight matrices for attention (`c_q,c_k,c_v,c_proj`, each 1536×1536 ≈ 2.36M) plus
the MLP (`c_fc` 1536×6144 and `c_proj` 6144×1536, ≈ 9.4M each), so ≈ 28M/block ×
52 ≈ **1.47B**. The token embedding, the three value-embedding tables, and the
output head add a few hundred million more on top. The exact total depends on how
you count the (untied) embeddings; the model card's headline is ~1.55B. Hold the
takeaway: **the blocks are where the parameters — and the compute and the
activation memory — live.**

---

## 3. Imports and the module docstring

```python
# L1-26
"""ChronoGPT model — training-enabled.
... (changes from the original documented above: inference_mode removed; KV cache
made an optional `past` arg) ...
"""
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
```

- `torch`, `torch.nn` (`nn`), `torch.nn.functional` (`F`) are the standard PyTorch
  trio: tensors, layer classes, and stateless math functions respectively.
- `from torch.utils.checkpoint import checkpoint` — **gradient checkpointing**, a
  memory-saving trick used in `forward`. Explained in §12.
- `PyTorchModelHubMixin, hf_hub_download` — Hugging Face Hub plumbing. The mixin
  grants `save_pretrained` / `from_pretrained` / `push_to_hub` behavior;
  `hf_hub_download` fetches a single file from a repo. Note: the model does **not**
  use `transformers` / `AutoModel` — it is a plain `nn.Module` with the hub mixin
  bolted on. That is why we hand-roll generation and pushing elsewhere.

---

## 4. `norm` — parameter-free RMSNorm

```python
# L28-29
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

This one tiny function is used *everywhere* in the model, so understand it well.

**What it does.** RMSNorm (Root-Mean-Square normalization) rescales a vector so
its root-mean-square magnitude is 1. For a vector $x = (x_1, \dots, x_d)$ over the
last dimension (here $d = $ `x.size(-1)`):

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}}$$

`F.rms_norm(x, (x.size(-1),))` normalizes over the **last** axis only, with the
shape tuple telling it which trailing dimensions to normalize. We pass **no
weight**, so this is *parameter-free*: there is no learnable per-feature gain
(no $\gamma$). Pure rescaling.

**RMSNorm vs LayerNorm.** Standard LayerNorm does two things: subtract the mean
(re-center to zero), then divide by the standard deviation (re-scale). RMSNorm
**drops the mean subtraction** and divides by the RMS instead of the std. So:

- LayerNorm: $\hat{x} = (x - \mu)/\sigma$, then scale + shift by learned $\gamma,\beta$.
- RMSNorm: $\hat{x} = x / \text{RMS}(x)$, (optionally scale by learned $\gamma$; here even that is omitted).

**Why no mean-centering?** Empirically the re-centering contributes little, and
dropping it makes the op cheaper and the architecture simpler. It is the now-common
choice (LLaMA, Gemma, modded-nanoGPT all use RMSNorm). Finance analogy (approximate):
LayerNorm is like de-meaning *and* standardizing a return series; RMSNorm is like
dividing by the L2 magnitude without de-meaning — you keep the "level," only kill
the scale.

**The causality consideration you asked about.** A natural worry: does normalizing
"leak" information across time and break the left-to-right (causal) property? **No.**
RMSNorm here normalizes over the **feature dimension** (`x.size(-1)`, the
`model_dim` or `head_dim` axis), **independently for each token position**. Token
*t*'s normalized value depends only on token *t*'s own features — never on tokens
*t+1, t+2, …*. So it is fully compatible with causal (autoregressive) modeling.
The thing that would break causality is normalizing *across positions* (the time
axis), which this never does. Same reasoning applies to QK-norm in §7.

**Where it's applied.** Pre-norm (before attention and before the MLP inside each
`Block`), as QK-norm on the queries/keys, on the token embedding (`x0`), on every
retained hidden state, and once more right before the output head. We will flag
each site as we reach it.

---

## 5. `CastedLinear`

```python
# L32-37
class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))
```

A `CastedLinear` is a normal linear (matrix-multiply) layer — $y = xW^\top$ — with
two twists:

- **`bias=False`.** No additive bias term, just the weight matrix. Common in modern
  LLMs; the bias is largely redundant given the normalization layers.
- **`self.weight.type_as(x)`.** Before multiplying, cast the weight to the *same
  dtype as the input* `x`. This is the linchpin of the model's mixed-precision
  scheme: the **weights are stored in float32** (so the AdamW optimizer can keep
  precise master copies — see `05-train.md`), but the **compute happens in
  bfloat16** because `x` arrives as bfloat16. So every matmul is fast/low-memory
  bf16 while the stored parameters stay fp32. Without this cast, a bf16 activation
  hitting an fp32 weight would error or silently upcast.

Every learned linear in the model (`c_q/c_k/c_v/c_proj`, the MLP, the LM head) is a
`CastedLinear`. The token/value embeddings are plain `nn.Embedding`.

---

## 6. `Rotary` — RoPE

```python
# L40-55
class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x):
        cos, sin = self.cos[None, : x.size(-3), None, :], self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)
```

**The problem RoPE solves.** Attention by itself is permutation-invariant — it has
no notion of word order. GPT-2 added a *learned absolute* position embedding (a
vector for "position 0", "position 1", …) to the token embeddings. **RoPE (Rotary
Position Embedding)** does it differently and better: instead of *adding* a
position vector, it **rotates** each query and key vector by an angle proportional
to its position. The beautiful consequence: when you later take the dot product
$q_m \cdot k_n$ inside attention, the result depends only on the **relative**
offset $m - n$, not on the absolute positions. Position information enters
multiplicatively, through geometry, and is inherently relative.

Here `dim` is the **per-head** dimension (128). Walk through `__init__`:

- **L43** `angular_freq = (1/1024) ** linspace(0, 1, steps=dim//4)`. Build `dim//4`
  = 32 base frequencies, geometrically spaced from $(1/1024)^0 = 1$ (fastest
  rotation) down to $(1/1024)^1 = 1/1024$ (slowest). Different feature pairs rotate
  at different rates — fast ones encode fine/local position, slow ones encode
  coarse/long-range position. (`1/1024` is this model's base; classic RoPE uses
  `1/10000`.)
- **L44** `cat([angular_freq, zeros(dim//4)])`. Append 32 **zeros**, giving a length-64
  (= `dim//2`) frequency vector where the **second half is all zero**. A zero
  frequency means zero rotation — those feature pairs are passed through
  **unrotated**. This is a modded-nanoGPT choice: **only half of each head's
  dimensions are position-rotated; the other half stays position-agnostic.** Worth
  flagging to Asaf — it is not standard RoPE, which rotates the whole head.
- **L45-46** Build a position grid `t = 0,1,…,max_seq_len-1` and the outer product
  `theta[i,j] = t_i * angular_freq_j`, shape `[max_seq_len, 64]`: the rotation angle
  for every (position, frequency) pair.
- **L47-48** Precompute and cache `cos(theta)` and `sin(theta)` as **buffers**
  (`register_buffer`). Buffers are non-trainable tensors that move with the model
  (`.to(device)`) but are not parameters. `persistent=False` keeps them out of the
  saved `state_dict` (they are cheap to recompute, so no need to ship them).

`forward` applies the rotation to an input `x` of shape `[B, T, num_heads,
head_dim]` (this is called on Q and K *before* the transpose to head-major layout):

- **L51** Slice the cached tables to the current sequence length. `x.size(-3)` is
  `T` (the layout is `[B, T, heads, dim]`, so axis −3 is time). The `[None, …, None, :]`
  indexing inserts singleton batch and head axes so `cos`/`sin` broadcast as
  `[1, T, 1, 64]`.
- **L52** `x1, x2 = x.float().chunk(2, dim=-1)`. Split the 128-dim head vector into
  two halves of 64. (Upcast to float32 first for a numerically clean rotation.)
- **L53-54** The 2-D rotation, applied pairwise to `(x1[i], x2[i])`:
  $$y_1 = x_1\cos\theta + x_2\sin\theta,\qquad y_2 = -x_1\sin\theta + x_2\cos\theta.$$
  For the 32 pairs with zero frequency, $\cos = 1, \sin = 0$, so $y_1 = x_1,\ y_2 =
  x_2$ — confirming those pass through untouched.
- **L55** Re-concatenate the rotated halves back to 128 dims and cast back to `x`'s
  original dtype.

**Why the context window matters.** RoPE generalizes to positions it was trained
on; far beyond the training context, the unseen large rotation angles degrade
quality. Our training context is **1792** tokens (the data `block_size`), so the
model is reliable up to ~1792 positions even though the `cos`/`sin` tables are
pre-built out to 65536. The data pipeline therefore packs sequences to exactly
1792 (`04-data.md`). This is also why a too-long prompt at inference can wander.

---

## 7. `CausalSelfAttention`

This is the attention module — where tokens look back at earlier tokens and mix
information. It is mostly textbook (see `01-ml-primer.md` for QKV/softmax), with
two non-standard additions: **QK-norm** and the **value-embedding blend**.

```python
# L58-69
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)
```

- `head_dim = dim // num_heads` = 1536/12 = **128**. The 1536-wide stream is split
  into 12 independent 128-dim heads, each attending separately.
- `c_q, c_k, c_v` project the input into **queries, keys, values** (each 1536→1536,
  reshaped into 12×128). `c_proj` mixes the heads' outputs back to 1536.
- `self.lambdas = nn.Parameter([0.5, 0.5])` — **two learnable scalars** controlling
  the value-embedding blend (below). Init at 0.5/0.5.
- `self.rotary = Rotary(self.head_dim)` — a per-head RoPE instance.

```python
# L71-86
    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)
```

- **L73-75** Project `x` to Q, K, V and reshape to `[B, T, 12, 128]`.
- **L76-79 — the value-embedding blend (study this).** `ve` is an optional extra
  value signal coming from the shared `ValueEmbedding` tables (§10), supplied only
  at the outermost layers. When present:
  $$v = \lambda_0\, v + \lambda_1\, \text{ve},\qquad \lambda \text{ learned, init } 0.5/0.5.$$
  So the values used in attention become a **learned mix** of (a) the values
  projected from the current residual stream and (b) a *direct* embedding of the
  token id. When `ve is None` (the inner layers), it is just $\lambda_0 v$ — note
  even then the scalar $\lambda_0$ still scales V, which the network can learn to
  use. `ve.view_as(v)` reshapes the embedding to match V's `[B,T,12,128]` layout.
  *Why* this exists rather than a bigger embedding is in §10 and §16.
- **L80 — QK-norm.** `q, k = norm(q), norm(k)` applies RMSNorm to queries and keys
  over the `head_dim` axis, **per token, per head**. This bounds the magnitude of Q
  and K so the dot products $q\cdot k$ cannot blow up, which stabilizes the softmax
  and training (attention logits stay in a sane range). Crucially, V is **not**
  normalized — only Q and K, because they are the ones multiplied together inside
  the softmax. Like RMSNorm in §4, this is per-position and does not touch
  causality.
- **L81** Apply RoPE to the (now normalized) Q and K. Order matters: **normalize,
  then rotate**.
- **L82-84** `F.scaled_dot_product_attention(..., is_causal=True)`. PyTorch's fused
  attention: compute $\text{softmax}(QK^\top/\sqrt{d})V$ with a **causal mask** so
  each position attends only to itself and earlier positions. The `transpose(1, 2)`
  puts tensors in `[B, heads, T, head_dim]` head-major layout the kernel expects.
  **Note what is *absent*: no `attn_mask` / no padding mask is passed** — only the
  built-in causal triangle. The model literally cannot mask out padding tokens.
  This is the architectural fact that forces the data pipeline to **pack** rather
  than pad (`04-data.md`, `implementation-notes.md` §4).
- **L85-86** Transpose back, flatten the 12 heads into one 1536 vector, and mix
  with `c_proj`. `.contiguous()` makes memory layout linear after the transpose so
  the `.view` is legal.

---

## 8. `MLP` — squared-ReLU

```python
# L89-99
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)
```

The position-wise feed-forward network, applied independently to every token.

- **L92-93 — the 4× expansion.** `c_fc` widens 1536 → **6144** (= 4×1536), and
  `c_proj` shrinks 6144 → 1536. The middle "hidden" layer is 4× wider than the
  stream. This is the standard transformer FFN ratio, and it is also where a large
  share of the **activation memory** lives: every token holds a 6144-wide vector
  inside every one of the 52 layers. Hold this thought — it is the direct reason
  full fine-tuning OOMs without gradient checkpointing (`implementation-notes.md`
  §7; §12 here).
- **L94 — zero-init of the output projection.** `c_proj.weight.data.zero_()` sets
  the down-projection to all zeros *at initialization*. So at the start of
  training **every MLP outputs exactly 0**, i.e. each block initially acts as the
  identity (`x = x + 0`). This makes a very deep (52-layer) residual network
  trainable from scratch: signal passes cleanly through the residual highway and
  blocks "switch on" gradually as their weights move off zero. See §16.
- **L98 — squared-ReLU (ReLU²).** `F.relu(x).square()` is the activation:
  $\text{ReLU}(x)^2 = (\max(0, x))^2$. Versus a plain ReLU it keeps the same
  "kill negatives" gate but grows **quadratically** for positives instead of
  linearly. modded-nanoGPT found this trains better than GELU/ReLU here. (It is
  unbounded above, which is part of *why* the logit softcap in §12 exists.)

---

## 9. `Block`

```python
# L102-114
class Block(nn.Module):
    def __init__(self, model_dim, num_heads, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads) if use_attn else None
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x
```

One transformer block = attention sub-layer + MLP sub-layer, each wrapped in a
**pre-norm residual** (normalize, transform, add back). Two non-standard bits:

- **L107 + L110 — the x0 lambda skip.** Each block carries `lambdas =
  [1.0, 0.0]` (learnable) and *begins* its forward with
  $$x \leftarrow \lambda_0\, x + \lambda_1\, x_0,$$
  where `x0` is the (normalized) **original token embedding** for the sequence (see
  §12). This injects a fresh copy of the raw token identity into **every block**,
  not just the bottom one. At init $\lambda = [1, 0]$, so it starts as a pure
  pass-through ($x$ unchanged) and the network *learns* how much embedding to
  re-inject at each depth. Intuition: deep layers can drift far from "what the
  token literally was"; this lets any layer re-anchor to the original token
  cheaply. (modded-nanoGPT calls these the "x0 / value-residual" skips.)
- **L111-113 — pre-norm residuals.** `x = x + self.attn(norm(x), ve)` and
  `x = x + self.mlp(norm(x))`. We normalize the **input** to each sub-layer
  (`norm(x)`), pass it through, and **add** the result onto the un-normalized
  residual stream. "Pre-norm" (normalize before, not after) is the stable modern
  choice for deep transformers. Note attention receives `norm(x)` *and* `ve`; QK-
  norm in §7 is an *additional* normalization inside attention, not a duplicate.
- **L105** `use_attn` can build an MLP-only block (`attn=None`), in which case the
  attention residual is skipped. In our configs every block has attention
  (`use_attn=True` for all), so the `if self.attn is not None` is always true here
  — but the option exists in the lineage (some modded-nanoGPT variants drop
  attention in a few layers).

---

## 10. `ValueEmbedding`

```python
# L117-128
class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers=52):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        base = [emb(inputs).bfloat16() for emb in self.embed]
        half = self.num_layers // 2  # encoder layer count; assumes num_layers even and >= 6
        encoder = [base[i] if i < 3 else None for i in range(half)]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder
```

This produces the `ve` tensors fed into attention's value path (§7).

- **L121 — three embedding tables.** Besides the main token embedding (in
  `ChronoGPT`), there are **3 extra** `nn.Embedding(vocab_size, model_dim)` tables.
  Each maps a token id directly to a 1536-vector.
- **L123** `base = [emb(inputs).bfloat16() for emb in self.embed]` — look up all 3
  tables for the input ids, cast to bf16. `base` is a list of 3 tensors.
- **L124-127 — which layers get a value embedding.** With `num_layers = 52`,
  `half = 26`. The function returns a length-52 list (one slot per block), but only
  **6 slots are non-`None`**:
  - **Encoder** layers 0, 1, 2 → tables `base[0], base[1], base[2]`; layers 3–25 → `None`.
  - **Decoder** layers 23, 24, 25 (i.e. global layers 49, 50, 51) → `base[0],
    base[1], base[2]` again; decoder layers 0–22 → `None`.

  So the **same three tables are shared** between the first three and last three
  layers (the "outer" layers of the U). Everything in the middle gets `None` and so
  uses only its own projected V (the `ve is None` branch in §7). The comment notes
  the layout assumes an even `num_layers ≥ 6`.

**Why value embeddings instead of just a bigger token embedding?** (Asaf bait — see
also §16.) Three reasons. (1) They feed a *different place*: the **value** path of
attention at specific depths, not the residual input — so they let particular
layers consult the raw token id directly when forming "what to retrieve." (2) They
are **shared across the symmetric encoder/decoder ends**, reinforcing the U-net
structure rather than just widening one table. (3) Empirically (modded-nanoGPT)
this gives a better quality-per-parameter trade than simply enlarging `model_dim`
or the main embedding. A bigger embedding would only change the block *inputs*;
these inject token identity straight into attention values mid-network.

---

## 11. `ChronoGPT.__init__`

```python
# L131-144
class ChronoGPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=True) for _ in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.grad_checkpoint = False  # set True (training only) to recompute blocks in backward
```

- **L132** Constructor takes the four shape hyperparameters; `**kwargs` swallows any
  extra config keys (e.g. `model_type` from a saved config) so loading never errors
  on unknown fields.
- **L136** `self.embed` — the main token embedding table, 50304 × 1536.
- **L137** `self.blocks` — a `ModuleList` of `num_layers` (52) `Block`s, all with
  attention. The encoder/decoder split is **not** stored in the block list; it is
  imposed by the `forward` loop (§12). The blocks are just a flat list of 52.
- **L138** `self.value_embeds` — the `ValueEmbedding` from §10.
- **L139-140** `self.lm_head` — the output projection, 1536 → 50304 (one logit per
  vocabulary entry), **also zero-initialized**. So at init the model outputs all-zero
  logits → a perfectly uniform softmax → loss ≈ ln(vocab). Combined with the zeroed
  MLP outputs (§8), the whole network starts as a near-identity that predicts
  uniform; training carves structure into it. Note: the LM head is a **separate**
  matrix from `self.embed` (weights are *not* tied).
- **L141-142** Compute the encoder/decoder split: 26 and 26 for 52. Using a
  subtraction (`num_layers - num_encoder_layers`) means an odd `num_layers` would
  put the extra block in the decoder — fine for the tiny smoke-test config.
- **L143** `self.skip_weights = Parameter(ones(num_decoder_layers))` — **26
  learnable scalars**, one per decoder layer, weighting each U-net skip connection
  (§12). Init at 1.0 (skip fully on).
- **L144** `self.grad_checkpoint = False` — a runtime flag (not a parameter). When
  `True` *and* training, blocks are recomputed in the backward pass to save memory
  (§12). Default off; `train.py` flips it on when `grad_checkpoint: true` (config
  default).

---

## 12. `ChronoGPT.forward` — the U-net pass

This is the heart of the file. It takes `inputs` (token ids) and returns
`(logits, layer_outputs)`.

```python
# L146-158
    def forward(self, inputs, return_hidden=True):
        """Returns (logits[B,T,V] float, layer_outputs). ..."""
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        B = inputs.size(0)
        x0 = norm(self.embed(inputs).bfloat16())
        x = x0
```

- **L146** Signature: **only `inputs`** (token ids), plus the `return_hidden`
  toggle. **There is no `attention_mask` / `labels` / `position_ids` argument.**
  Positions come from RoPE; there is no padding mask by design (§7). Loss is
  computed *outside* the model in the training loop (`05-train.md`).
- **L154-155** If a single 1-D sequence is passed, add a batch axis → `[1, T]`.
- **L157 — build `x0`.** Embed the ids (`self.embed(inputs)`), cast to bf16, and
  RMSNorm. This normalized embedding is both the **initial residual stream** and the
  `x0` re-injected into every block (§9). **L158** sets the working stream `x = x0`.

```python
# L160-165
        ve = [self.value_embeds(inputs[i].view(-1)) for i in range(B)]
        ve = [
            torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None
            for i in range(len(ve[0]))
        ]
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]
```

- **L160** Run `value_embeds` once per batch element (it takes a 1-D id sequence),
  giving for each batch element a length-52 list (mostly `None`, §10).
- **L161-164** "Transpose" that list-of-lists: for each of the 52 layer slots, if
  the slot is non-`None`, `torch.stack` the per-batch tensors back into one
  `[B, T, model_dim]` tensor; otherwise keep `None`. Result: a length-52 list, each
  entry either a batched value-embedding or `None`.
- **L165** Split into encoder (first 26) and decoder (last 26) value-embedding
  lists, matching the loops below.

```python
# L167-170
        ckpt = self.grad_checkpoint and self.training

        def run_block(blk, *args):
            return checkpoint(blk, *args, use_reentrant=False) if ckpt else blk(*args)
```

- **L167 — the gradient-checkpointing gate.** `ckpt` is true only if the flag is set
  **and** the module is in training mode (`self.training`, toggled by
  `model.train()`/`model.eval()`). So checkpointing never fires at inference.
- **L169-170 — `run_block`.** A wrapper that either calls the block normally
  (`blk(*args)`) or wraps it in `torch.utils.checkpoint`. **What checkpointing
  does:** normally PyTorch saves every layer's intermediate activations during the
  forward pass so they are available for the backward pass — across 52 layers ×
  1792 tokens × the 6144-wide MLP, that is tens of GB. Checkpointing instead
  **discards** a block's internal activations after the forward and **recomputes
  them on the fly during backward**. This trades ~20% extra compute for roughly 10×
  less activation memory (`implementation-notes.md` §7), which is what lets
  `batch_size 8` fit one 80GB card. `use_reentrant=False` selects the modern, more
  robust checkpoint implementation (correct gradients with the non-reentrant
  autograd path).

```python
# L172-183
        layer_outputs = []
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = run_block(self.blocks[i], x, ve_enc[i], x0)
            skip_connections.append(x)
            if return_hidden:
                layer_outputs.append(norm(x))
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = run_block(self.blocks[self.num_encoder_layers + i], x, ve_dec[i], x0)
            if return_hidden:
                layer_outputs.append(norm(x))
```

This is the **U-net**. Two phases:

- **Encoder (L174-178).** Run blocks 0…25 in order. After each, **push** its output
  onto a `skip_connections` stack. Each block gets the current stream `x`, its value
  embedding `ve_enc[i]` (non-`None` only for the first 3), and the global `x0`.
- **Decoder (L179-183).** Run blocks 26…51. Before each decoder block,
  **L180** does `x = x + skip_weights[i] * skip_connections.pop()`. Because a stack
  is LIFO, the **first** decoder layer pops the **last** encoder layer's output, the
  second decoder pops the second-to-last encoder, and so on — pairing encoder layer
  *k* with decoder layer *(25−k)*. That symmetric pairing is exactly the "U" shape:
  go down 26 levels, come back up 26 levels, wiring each descent level to its
  matching ascent level. `skip_weights[i]` (learnable, init 1.0) scales how strongly
  each skip is mixed in. The block index is offset by `num_encoder_layers` to reach
  the decoder half of the flat `blocks` list.
- **`layer_outputs` (L177-178, 182-183).** When `return_hidden=True`, append the
  RMSNormed hidden state after every layer — the per-layer hidden-state list that
  `embed()` uses to build features for return prediction. **During training this is
  set to `False`** (the loop skips the appends), because retaining 52 hidden states
  also costs memory; combined with checkpointing this is the second memory lever
  (`implementation-notes.md` §7). `return_hidden=True` is the default for inference
  and embedding extraction.

**Is it still causal with all these skips?** Yes — see §16. The skips move
information **across depth** (layer ↔ layer), never **across time within a layer**;
every position-mixing operation (only the attention) carries the causal mask.

```python
# L185-188
        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)  # logit softcap
        return logits.float(), layer_outputs
```

- **L185** Final RMSNorm of the stream before the head (standard pre-head norm).
- **L186** Project to vocabulary logits with the (zero-initialized) LM head →
  `[B, T, vocab_size]`.
- **L187 — logit softcap.** `15 * tanh(logits / 15)` squashes every logit smoothly
  into the range **(−15, +15)**. `tanh` is near-linear for small inputs (so normal
  logits pass through almost unchanged) but saturates for large ones, capping
  extreme values. This prevents any single logit from exploding — important given
  the unbounded squared-ReLU MLP (§8) — and improves training stability. (Same
  trick as Gemma.) The constant 15 sets the soft ceiling.
- **L188** Return logits cast back to **float32** (for a numerically stable
  cross-entropy / softmax in the loss, computed outside) plus `layer_outputs`.

---

## 13. Save / load / hub

```python
# L190-202
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        config = {
            "model_type": "ChronoGPT",
            "vocab_size": self.embed.num_embeddings,
            "num_layers": len(self.blocks),
            "num_heads": self.num_heads,
            "model_dim": self.embed.embedding_dim,
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
```

- Writes three files to a directory: **`pytorch_model.bin`** (the weights, i.e. the
  `state_dict`), **`config.pt`** (the shape hyperparameters as a pickled dict), and
  **`config.json`** (the same config, human-readable). The four config values are
  read back off the live modules (`num_embeddings`, `len(self.blocks)`,
  `embedding_dim`) so they always match the actual model. This overrides the mixin's
  default save so the config travels with the weights — necessary because the
  constructor needs those four numbers to rebuild the architecture.

```python
# L204-217
    @classmethod
    def from_pretrained(cls, repo_id, cache_dir=None, **kwargs):
        if os.path.isdir(repo_id):
            config_path = os.path.join(repo_id, "config.pt")
            bin_path = os.path.join(repo_id, "pytorch_model.bin")
        else:
            config_path = hf_hub_download(repo_id=repo_id, filename="config.pt", cache_dir=cache_dir)
            bin_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)
        config = torch.load(config_path, map_location="cpu", weights_only=False)
        model = cls(**config)
        model.load_state_dict(torch.load(bin_path, map_location="cpu", weights_only=False))
        return model
```

- **L208-213** Accept **either** a local directory **or** a Hugging Face repo id. If
  it is a local dir (e.g. `runs/.../stage1_scratch` for resume, or `chrono infer
  --repo <dir>`), read the files straight from disk; otherwise download them from
  the Hub. This dual path is what lets us resume from a local checkpoint and run
  the official released model with the same call.
- **L214-216** Load the config, **rebuild the model** via `cls(**config)` (the
  `**kwargs` in `__init__` absorbs `model_type`), then load the weights. `map_location="cpu"`
  loads onto CPU first (the caller moves it to GPU); `weights_only=False` permits
  the pickled config object.

There is **no `generate` method** in this file — generation was intentionally
dropped (§1) and lives in `infer.py`, which calls `forward` repeatedly. There is
also no `embed` method here; `embed()` (the feature extractor) consumes the
`layer_outputs` this `forward` returns and lives elsewhere in the package.

---

## 14. `build_tiny`

```python
# L220-222
def build_tiny(vocab_size=512, num_layers=8, num_heads=4, model_dim=64):
    """Small randomly-initialized model for the CPU smoke test (no download)."""
    return ChronoGPT(vocab_size=vocab_size, num_layers=num_layers, num_heads=num_heads, model_dim=model_dim)
```

A factory for a **toy** ChronoGPT — 8 layers, 64-wide, 512 vocab — small enough to
run on a CPU with no Hub download. Used by the smoke test to exercise the whole
forward/loss/backward path quickly (4 encoder + 4 decoder; the U-net logic still
holds since `num_layers` is even and ≥ 6, satisfying `ValueEmbedding`'s assumption).

---

## 15. Recap table — vs a vanilla GPT-2 / textbook decoder

| Component | Textbook GPT-2 | ChronoGPT (this file) |
|---|---|---|
| Normalization | LayerNorm (mean + var, learned γ/β) | **RMSNorm**, parameter-free (`norm`, L28) |
| Position info | learned absolute position embeddings | **RoPE** rotation of Q/K, only half the head dim rotated (L40) |
| Attention Q/K | raw projections | **QK-norm** (RMSNorm on Q,K before RoPE) (L80) |
| Linear bias | yes | **no bias** anywhere (`CastedLinear`, L32) |
| Precision | single dtype | **fp32 weights, bf16 compute** via per-matmul cast (L37) |
| MLP activation | GELU | **squared-ReLU** (ReLU²) (L98) |
| Depth structure | flat stack of N blocks | **U-net**: 26 encoder + 26 decoder with learned skip weights (L174-181) |
| Embedding use | one table → block 0 input | + **3 shared value-embedding tables** injected into attention V at the 6 outer layers (L117) + **x0 re-injected into every block** (L110) |
| Output logits | raw `Wx` | **zero-init head** + **softcap** `15·tanh(z/15)` (L186-187) |
| Init | random | **zero-init** of MLP and head output projections (L94, L140) |
| Forward inputs | ids + attention mask | **ids only** — no padding mask (causal-only) (L146) |
| Loss / generate | often inside model | **outside** (`05-train.md` / `infer.py`); model returns `(logits, layer_outputs)` |

Same skeleton (embed → attention+MLP blocks → logits, causal, residual), many
modernized internals, plus the genuinely distinctive U-net + value-embedding +
x0-skip topology of modded-nanoGPT.

---

## 16. Questions an expert might ask (FAQ)

**Q1. With U-net skips, x0 skips, and value embeddings, is the model still causal?**
Yes, strictly. Causality is a statement about the **time/position axis**: token *t*'s
output must not depend on tokens *> t*. The **only** operation that mixes across
positions is attention, and it enforces causality — `is_causal=True` on the
training/full-sequence path (and during KV-cached decoding a single new query only
attends to already-generated positions, which is the same guarantee). RMSNorm, QK-norm, the
MLP, RoPE, the softcap, x0 injection, and *all* the U-net/value-embedding skips act
**within a position** (mixing the feature dimension) or **across depth** (layer to
layer) — never across time. So no future token leaks into a past prediction.

**Q2. What is the U-net actually doing, conceptually?** It gives the network direct
"shortcut" wires from early (encoder) layers to late (decoder) layers, paired
symmetrically. The intuition (borrowed from image segmentation U-nets) is that
later layers can recover fine-grained, low-level information computed early that
would otherwise be washed out after 30+ transformations. The learned `skip_weights`
let the model dial each shortcut up or down. It is still a single autoregressive
decoder — "encoder/decoder" here means *depth halves of one stack*, **not** the
separate text-encoder/text-decoder of a translation transformer.

**Q3. Why value embeddings instead of just a wider model or bigger embedding?**
They inject the raw token identity into a *different* place (the attention **value**
path) at *specific* depths (the 6 outer layers), and the three tables are **shared**
across the symmetric encoder/decoder ends, reinforcing the U-net. A bigger token
embedding would only change block inputs; widening `model_dim` would cost quadratic
compute in every layer. modded-nanoGPT found this a better quality-per-parameter
lever. See §10.

**Q4. Why no padding token / no attention mask — isn't that limiting?** ChronoGPT
inherits GPT-2's `tiktoken` vocabulary, whose only special token is `<|endoftext|>`
(id 50256); GPT-2 never had a pad token (`implementation-notes.md` §5). And
`forward` accepts no attention mask, so even if you padded, you could not mask the
pad tokens out — they would corrupt attention. The clean solution is to **pack**
many short examples into full 1792-token blocks separated by `<|endoftext|>`, never
padding (`04-data.md`, `implementation-notes.md` §4). The `vocab_size = 50304` is
just GPU-friendly rounding of 50257, **not** a pad token.

**Q5. Why zero-init the MLP and LM-head output projections?** To make a 52-layer
residual network trainable from scratch. With those projections at zero, every
block starts as an identity (`x = x + 0`) and the head emits uniform logits, so
the initial forward is well-behaved and gradients flow cleanly down the residual
highway. Blocks then "turn on" gradually as weights move off zero. (For our
fine-tuning we load pretrained weights, so the zero-init mainly matters for the
authors' from-scratch pretraining and for the `build_tiny` smoke test — but it is
part of the architecture's design.)

**Q6. Why squared-ReLU and a logit softcap together?** ReLU² is unbounded above and
grows quadratically, which helps capacity/optimization but can produce large
activations and, downstream, large logits. The `15·tanh(z/15)` softcap bounds the
logits smoothly so the softmax/loss stays stable. They are complementary: an
aggressive activation paired with a safety valve at the output.

**Q7. Why RMSNorm without a learnable gain, and why does it not break causality?**
Parameter-free RMSNorm is cheaper and, with QK-norm and the residual structure
already present, the per-feature gain adds little. It normalizes each token over its
**feature** axis independently, so it cannot move information across positions —
fully causal (§4).

**Q8. What changed from the released file, and could that change the numbers?** Two
things (§1): the `@torch.inference_mode()` decorators were removed (so gradients can
flow) and the KV-cache generation branch was dropped. Neither alters the math of the
forward pass — `inference_mode` only governs whether the autograd graph is recorded,
and the KV cache is an efficiency path, not a different computation. Numerical parity
against the official `ChronoGPT_inference.py` was confirmed (notebook §10, max logit
diff 0.0). We additionally *keep* the `layer_outputs` return the instruct file
comments out, because `embed()` needs it — a strict superset, not a change to logits.

---

*Cross-references: `01-ml-primer.md` (transformer fundamentals), `04-data.md`
(packing and why there is no padding), `05-train.md` (training loop, mixed
precision, gradient checkpointing in context), `docs/implementation-notes.md`
§§4–7 (design rationale).*

---

## Addendum (2026-06): the KV cache, and why training can't use one

The model now exposes an **optional KV cache** via a `past` argument to `forward`
(off by default). This is purely an inference speedup; understanding it also
clarifies a common confusion about training.

**What a KV cache is.** During autoregressive generation you emit tokens one at a
time. Without a cache, generating the *t*-th token re-runs attention over the whole
growing prefix, so producing `T` tokens costs `O(T²)` work per layer. A KV cache
stores each past token's keys and values (`(k, v)` per block) so each new step only
computes the **one** new token's query against the cached keys — `O(T)` total. For
long generations this is commonly a 5–20× speedup.

**How it's wired here (three touch-points):**
- `Rotary.forward(x, offset=0)` — the new token sits at absolute position
  `past_len`, so RoPE is applied with that offset (otherwise a length-1 input would
  wrongly get position-0 angles).
- `CausalSelfAttention.forward(x, ve, past, use_cache)` — appends the new `k,v` to
  the cached ones and returns the updated cache. `is_causal=(past_len == 0)`: the
  initial *prefill* (whole prompt, no past) needs the causal mask; a single-token
  decode step has one query that should see **all** cached positions, so no mask.
- `ChronoGPT.forward(inputs, return_hidden=False, past=...)` — threads one cache
  slot per block. With `past=None` (training/eval) the code path is byte-identical
  to before; the U-net skip connections work unchanged because they are per-position
  (each step's new position gets its own encoder skip).

`tests/test_smoke.py::test_kv_cache_matches_full_forward` asserts the cached path
reproduces the full-sequence logits (and every greedy argmax) — a speedup, not a
behavior change.

**Why training fundamentally cannot use a KV cache.** A KV cache only helps when you
generate *sequentially* and would otherwise recompute the prefix. Training does no
such thing: under **teacher forcing**, the entire target sequence is fed in **one**
forward pass and the loss is computed at **all** positions at once (the causal mask
ensures position *t* only sees ≤ *t*). There are no sequential steps, so every
token's `k,v` is already computed exactly once — nothing to cache or reuse. (Two more
reasons it's moot: backprop needs the full activation graph anyway, and the weights
change every optimizer step, so any cached `k,v` would be instantly stale.) The cache
lives only in `infer.generate`; see `06-infer-and-eval.md`.

**A causal-masking subtlety worth knowing.** Because attention here is causal, the
"the model has no padding mask, so we must pack" story is more nuanced than it
sounds — *right*-padding would actually be safe (trailing pad never enters a real
token's attention). The real reason we pack is throughput, not correctness. That
argument lives in `04-data.md` (Addendum) and `docs/implementation-notes.md` §4.

---

## Addendum II (2026-06): normalization & backprop, `c_proj`, RoPE — with a worked example

Four follow-up questions, answered with the mechanics first and then one fully
worked numerical pass through an attention sublayer.

### A. How RMSNorm shapes the *backward* pass

`norm(x)` here is `F.rms_norm(x, (d,))` with no learnable gain:

$$y = \frac{x}{r}, \qquad r = \sqrt{\tfrac{1}{d}\textstyle\sum_j x_j^2} = \frac{\lVert x\rVert}{\sqrt d}.$$

Its Jacobian (what the chain rule multiplies the incoming gradient by) is

$$\frac{\partial y}{\partial x} = \frac{1}{r}\Big(I - \tfrac{1}{d}\, y\,y^{\top}\Big).$$

Two things fall out of that one expression, and they are *why* normalization helps
training:

1. **The `1/r` factor is an automatic gradient regulator.** Gradients flowing back
   through the norm are scaled by `1/r`. If a layer's activations grow large
   (`r` big), the backward signal is *damped*; if they shrink, it is *amplified*.
   Across 52 layers this keeps gradient magnitudes in a healthy band and is a major
   reason deep transformers don't explode/vanish during backprop.
2. **The `(I − yyᵀ/d)` term projects out the radial direction.** It removes the
   component of the incoming gradient that points along `y` (the current activation
   direction). The reason is exact: `y` is *invariant* to the length of `x`
   (scaling `x → cx` leaves `y` unchanged), so the loss genuinely *cannot* depend on
   `‖x‖` — there is no gradient in that direction. The model can only learn to move
   the **direction** of `x`, never its length. That deletes a redundant,
   ill-conditioned degree of freedom and makes the optimization landscape better
   conditioned. *Finance analogy:* it's like standardizing regressors before a
   regression — you strip out a nuisance scale so the optimizer isn't fighting an
   ill-conditioned Hessian.

Two placement facts compound this. **Pre-norm** (norm sits *inside* each block's
branch; the residual skip is added un-normalized — see `Block.forward`) gives the
gradient a clean identity "highway" straight down the residual stream, which is the
standard trick for training very deep transformers. And **QK-norm** (normalizing
`q` and `k` *before* the dot product) bounds the attention logits, so the softmax
can't saturate — a saturated softmax has near-zero gradient, so this protects the
backward pass *through attention* specifically.

### B. What `c_proj` is

`c_proj` ("channel projection") is the **output / down projection** — a bias-free
linear layer (`CastedLinear`, which just casts its weight to the input's dtype on
the fly). It appears in two places with the same role — *recombine and write back*:

- **In attention** it is the `W_O` of standard attention notation, shape
  `dim → dim`. Each head produces its own `head_dim` outputs in a disjoint
  subspace; `c_proj` linearly **mixes all heads together and maps the result back
  into the residual-stream coordinate system**. Without it, head outputs would be
  stranded in fixed, un-mixable coordinates.
- **In the MLP** it is the `4·dim → dim` down-projection that brings the widened
  hidden layer back to model width.

**Initialization matters here.** In this code the **MLP's `c_proj` and the
`lm_head` are zero-initialized** (`...weight.data.zero_()`), while the attention
`c_proj` uses standard init. Zero-init means that *at the start of training* each
block's MLP contributes exactly **0** and the logits are 0 (→ uniform
distribution): the network starts as a clean stack of identity residuals and each
block "turns on" gradually as `c_proj` learns. This is the "zero-init projections"
line in the architecture summary, and it is a stability trick, not a quirk.

### C. RoPE: a rotation of `q`/`k` per layer — not an added embedding

The direct answers to your question:

- **It is *not* added to the embeddings (unlike BERT / learned absolute positions).**
  Nothing is summed into the token embeddings and nothing is added to the residual
  stream. There is no "position vector."
- **Instead, inside *every* attention layer, the query and key vectors are
  *rotated* by an angle proportional to the token's position**, applied per head,
  per position, *after* the `c_q`/`c_k` projections and QK-norm, *before* the `q·k`
  score (see `CausalSelfAttention.forward`: `q, k = self.rotary(q), self.rotary(k)`).
  **Values `v` are NOT rotated**, and the residual stream is not rotated.

**Does the rotation scramble the word-embedding information? No — by construction:**

1. **It touches only `q` and `k`** — the *matching* vectors used to compute
   attention weights — not `v`, the *content* that gets aggregated and written back
   to the stream. So the information being transported forward is never rotated.
2. **A rotation is orthogonal**: it preserves vector norms and inner products and is
   perfectly invertible. Position is *encoded into the geometry*, not destroyed.
3. **Only half the dimensions actually rotate.** In `Rotary.__init__`,
   `angular_freq` is `dim//4` real frequencies *concatenated with* `dim//4` zeros,
   so half the rotation-pairs have angle 0 and pass through **unchanged** — pure,
   position-independent content channels.
4. The `c_q`/`c_k` projections are *learned in the presence of* the rotation, so the
   model allocates feature dimensions knowing rotation will be applied.

**The payoff — relative position.** Because the rotation `R` is orthogonal,
`⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩`: the attention score between a query at position
`m` and a key at position `n` depends only on their **relative** offset `n − m` and
the content — never on absolute position. Different frequencies rotate at different
rates, so they encode distance at different scales (a harmonic / Fourier code of
position).

### D. Worked example: one attention sublayer, by the numbers

A tiny model so every number is checkable: `model_dim = 4`, `1` head,
`head_dim = 4`, two tokens **A** (position 0) and **B** (position 1). With
`head_dim = 4`, `Rotary` builds `dim//4 = 1` real frequency `f = 1.0` plus one zero,
so dimension-pair **{0,2} rotates** and pair **{1,3} never rotates**. Attention
scale is `1/√head_dim = 0.5`. At init the block lambdas are `[1, 0]` and the
attention lambdas are `[0.5, 0.5]`; take this to be a middle layer so the value
embedding `ve` is `None`. (Real RoPE uses many tiny-angle frequencies; we use
`θ = 1` radian, `cos 1 = 0.540`, `sin 1 = 0.841`, so the rotation is visible.)

Two "orthogonal" input words on the residual stream:

```
x_A = [3, 0, 0, 0]      # token A, position 0
x_B = [0, 0, 3, 0]      # token B, position 1
```

**Step 1 — x0 mix (`Block.forward`).** `x = 1.0·x + 0.0·x0 = x` (lambdas `[1,0]` at
init), so unchanged.

**Step 2 — pre-attention RMSNorm.** `r_A = √(9/4) = 1.5` → `x̂_A = [2,0,0,0]`;
likewise `x̂_B = [0,0,2,0]`. (Norm rescaled the magnitude 3 → 2; RMS is now 1. This
is the forward side of section A.)

**Step 3 — q, k, v.** Use identity projections for illustration, so `q = k = v = x̂`.
The value blend with `ve = None` is `v ← lambdas[0]·v = 0.5·v`:

```
q_A = [2,0,0,0]   k_A = [2,0,0,0]   v_A = [1,0,0,0]
q_B = [0,0,2,0]   k_B = [0,0,2,0]   v_B = [0,0,1,0]
```

**Step 4 — QK-norm.** `norm([2,0,0,0]) = [2,0,0,0]` (already RMS 1) — unchanged here,
but in general this is what bounds the logits.

**Step 5 — RoPE on q, k (NOT v).** A is at position 0 → angle 0 → unchanged. B is at
position 1 → rotate pair {0,2} by `θ = 1`, pair {1,3} by 0:

```
dim0' = dim0·cosθ + dim2·sinθ = 0·0.540 + 2·0.841 = 1.683
dim2' = −dim0·sinθ + dim2·cosθ = −0      + 2·0.540 = 1.081
q_B = k_B = [1.683, 0, 1.081, 0]
```

Notice: *before* rotation `q_B = [0,0,2,0]` was **orthogonal** to `k_A = [2,0,0,0]`
(dot product 0) — B could not "see" A at all on content. The rotation tilted a
component of `q_B` into dimension 0, creating overlap with `k_A`. That overlap is
**pure position** (distance 1), injected by RoPE — and note dims {1,3} would have
carried position-independent *content* matching, untouched.

**Step 6 — attention scores (×0.5), causal.**
A (pos 0) sees only A: `score(A,A) = 0.5·(2·2) = 2` → softmax → `1.0` →
`out_A = v_A = [1,0,0,0]`.
B (pos 1) sees A and B:

```
score(B,A) = 0.5·(q_B · k_A) = 0.5·(1.683·2)               = 1.683
score(B,B) = 0.5·(q_B · k_B) = 0.5·(1.683² + 1.081²)        = 2.000
softmax([1.683, 2.000]) = [0.421, 0.579]
out_B = 0.421·v_A + 0.579·v_B = [0.421, 0, 0.579, 0]
```

**Step 7 — `c_proj` (the `W_O` output projection).** Take a simple content-mixing
matrix to see it recombine the channels:

```
        [0.5 0 0.5 0]
W_O  =  [ 0  1  0  0]            W_O · out_B = [0.5, 0, 0.5, 0]
        [0.5 0 0.5 0]
        [ 0  0  0  1]
```

`c_proj` blended the two content channels (dim 0 carried A's value, dim 2 carried
B's) and wrote the mixed result back into residual-stream coordinates.

**Step 8 — residual add (to the *un-normalized* `x_B`).**

```
x_B  ←  x_B + W_O·out_B = [0,0,3,0] + [0.5,0,0.5,0] = [0.5, 0, 3.5, 0]
```

**Step 9 — MLP sublayer (schematic).**
`x ← x + c_proj_mlp( relu(c_fc(norm(x)))² )`. Because the MLP's `c_proj` is
**zero-initialized** (section B), at the start of training this term is exactly
`[0,0,0,0]` — the block's only update early on is the attention term above. As the
MLP's `c_proj` learns nonzero weights, this contribution grows.

**What the numbers show, mapped back to the questions:**
- **Normalization** (steps 2, 4) rescaled magnitudes to a common ~1 RMS; its
  Jacobian feeds that back as the `1/r`-damped, radial-direction-removed gradient of
  section A.
- **RoPE** (step 5) left A (pos 0) alone, rotated B, and the rotation is precisely
  what let B attend to A *by distance*; the non-rotating pair and the un-rotated
  values mean content is never scrambled.
- **`c_proj`** (step 7) recombined the attention output and wrote it back into the
  residual stream — its whole job.

---

*Cross-references for this addendum: `01-ml-primer.md` (RMSNorm, attention, softmax
basics), `05-train.md` (how these gradients drive AdamW), and Sections 4–7 above for
the line-by-line of `CausalSelfAttention`, `MLP`, `Rotary`, and `Block`.*
