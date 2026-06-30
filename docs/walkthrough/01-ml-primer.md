# 01 — A Machine-Learning Primer for the Finance Reader

**What this doc is / who it's for.** This is a self-contained conceptual primer on the deep-learning and NLP ideas you need to understand the rest of this codebase — written for a quantitatively strong reader (econometrics, time series, MLE, optimization) who has had little exposure to neural networks or language models. The goal is correctness without hand-waving: every term is defined the first time it appears, and finance/econometrics analogies are used where they illuminate and flagged explicitly where they break. You should be able to read this offline, end to end, and then follow `03-model.md` (architecture), the data doc, and the training doc without getting lost.

A note on the analogies: a language model is *not* a linear factor model, and self-attention is *not* literally a kernel regression. The analogies below are scaffolding for intuition. Each time one is load-bearing I will say how far it actually goes.

---

## Table of contents

1. [What a language model is, mathematically](#1-what-a-language-model-is-mathematically)
2. [Tokens and tokenization](#2-tokens-and-tokenization)
3. [Embeddings](#3-embeddings)
4. [The Transformer (decoder-only, causal)](#4-the-transformer-decoder-only-causal)
5. [Logits, softmax, cross-entropy, perplexity](#5-logits-softmax-cross-entropy-perplexity)
6. [How training works](#6-how-training-works)
7. [Pretraining vs fine-tuning vs instruction tuning (SFT)](#7-pretraining-vs-fine-tuning-vs-instruction-tuning-sft)
8. [Practical training machinery](#8-practical-training-machinery)
9. [Generation and inference](#9-generation-and-inference)
10. [Glossary](#10-glossary)
11. [What to read next](#11-what-to-read-next)

---

## 1. What a language model is, mathematically

A language model (LM) is a **probability model over sequences of tokens** (think: words or word-pieces; defined precisely in §2). Given a sequence of tokens $x_1, x_2, \dots, x_T$, the model factorizes the joint probability of the sequence using the chain rule of probability:

$$
p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, \dots, x_{t-1}).
$$

So at its core a modern LM is a parametric estimator of the **conditional distribution of the next token given all previous tokens**, $p_\theta(x_t \mid x_{<t})$, where $\theta$ is a vector of parameters (here, ~hundreds of millions of them). A "large" language model (LLM) just means $\theta$ and the training corpus are large.

**The MLE framing you already know.** Training the model means choosing $\theta$ to maximize the log-likelihood of the observed text corpus:

$$
\hat\theta = \arg\max_\theta \sum_{\text{sequences}} \sum_{t} \log p_\theta(x_t \mid x_{<t}).
$$

This is *exactly* maximum likelihood estimation. The objective is the log-likelihood of an autoregressive (AR) model — conceptually the same object as the conditional log-likelihood of an AR(p) time-series model, where each observation is predicted from its past. Three differences from a finance AR model:

- The "observation" $x_t$ is **categorical** (one of ~50,000 possible tokens), not real-valued, so the conditional distribution is a categorical/multinomial, not a Gaussian. Maximizing its log-likelihood is the same as minimizing **cross-entropy** (§5).
- The conditioning function $p_\theta(\cdot \mid x_{<t})$ is a **highly nonlinear neural network** (a Transformer, §4), not a linear projection of lagged values. There is no closed-form estimator; $\hat\theta$ is found by gradient descent (§6).
- The "lags" can be very long (this project uses sequences of length 1,792 tokens; the config calls this `block_size`) and the dependence is learned, not assumed.

That is the whole conceptual core. Everything else — embeddings, attention, optimizers, mixed precision — is machinery for making this MLE tractable and accurate at scale. *Where the analogy breaks:* an AR model has a fixed, small parameter vector and a likelihood you can write down and differentiate by hand; here $\theta$ is millions of weights inside many composed nonlinear layers, and the "likelihood surface" is non-convex with no identification or asymptotic-normality guarantees. We do not interpret individual parameters.

---

## 2. Tokens and tokenization

The model does not see characters or words directly. Text is first chopped into **tokens** by a **tokenizer**, then each token is mapped to an integer ID in a fixed **vocabulary**.

A **token** is a sub-word unit. This project uses **BPE (Byte-Pair Encoding)** via OpenAI's `tiktoken` library with the GPT-2 vocabulary (you'll see `tiktoken.get_encoding("gpt2")` at the top of `data.py` and `infer.py`). BPE is built greedily: start from individual bytes, then repeatedly merge the most frequent adjacent pair into a new symbol, until the vocabulary reaches a target size. The result is that common words become a single token while rare words split into several pieces.

- `" the"` → one token (note the leading space is part of the token).
- `" Manela"` → likely several tokens (`" Man"`, `"ela"`, ...).
- A token is therefore **not** a word. As a rule of thumb, one English token ≈ 0.75 words, or ~4 characters.

The GPT-2 vocabulary has **50,257** entries (50,256 BPE tokens plus one special **end-of-text** marker, the `EOT` token, integer ID **50256**). In `data.py` you'll see `EOT = ENC.eot_token` used as a separator placed after each training response, so the model learns where an answer ends.

**Why `vocab_size = 50304` and not 50257?** The released ChronoGPT checkpoint declares a vocabulary of **50,304**, which is `50257` rounded up to the next multiple of 128 (it's $2^7 \times 393$). This is pure **GPU efficiency padding**: matrix-multiplication kernels (and the embedding/output tables, §3) run faster when the dimension is a multiple of 64 or 128, so the extra ~47 rows are unused "dead" tokens that never appear in data. They cost a little memory and buy aligned, faster kernels. There is no linguistic meaning to them.

*Finance analogy (mild):* tokenization is like deciding the unit of observation before estimation — daily vs monthly returns, or how you bucket a categorical variable. The choice is upstream of the model and affects everything downstream, but it is a fixed preprocessing convention, not something estimated jointly here.

---

## 3. Embeddings

A token ID is just an integer; it carries no notion of similarity (token 5012 is not "between" 5011 and 5013 in any meaningful sense). The first thing the model does is look up each token ID in an **embedding table**: a learned matrix $E \in \mathbb{R}^{V \times d}$ where $V$ is the vocabulary size and $d$ is the model's hidden dimension (`model_dim`). Row $i$ of $E$ is the **embedding vector** for token $i$ — a dense, real-valued vector that the model learns. In `model.py` this is `self.embed = nn.Embedding(vocab_size, model_dim)`.

So tokenization turns text into integers, and embedding turns integers into vectors. Everything inside the Transformer operates on these vectors.

*Finance analogy (genuinely useful here):* think of the embedding as a vector of **latent factor loadings** for each token. Just as a stock can be summarized by its loadings on a set of latent factors, a token is summarized by its coordinates in a $d$-dimensional learned space, and tokens that behave similarly (appear in similar contexts) end up with similar vectors. The classic result that `vec("king") − vec("man") + vec("woman") ≈ vec("queen")` is the analogue of factor arithmetic. *Where it breaks:* these factors are not identified, not orthogonal, not ordered by variance explained, and have no economic interpretation — they are whatever directions minimize prediction loss. Do not read them like PCA components.

This codebase has a second, related sense of "embedding" worth flagging now: at *inference* time you can extract a model's internal **hidden state** for a piece of text and use it as a feature vector for a downstream task (e.g. predicting returns from text). That is what `infer.py::embed` does — it returns a layer's hidden state, mean-pooled over the sequence. This is the artifact a finance researcher would actually feed into a regression. Distinguish it from the *input* embedding table above: the input embedding is the first layer's lookup; the *extracted* embedding is a deep, context-dependent representation from layer $\ell$. More in §9.

---

## 4. The Transformer (decoder-only, causal)

The **Transformer** is the neural-network architecture that maps the sequence of input embeddings to a prediction of the next token at every position. This project uses a **decoder-only, causal** Transformer (the GPT family), meaning it reads left-to-right and predicts the next token. (The specific variant here is a "modded-nanoGPT" with some extras — RoPE, RMSNorm, a U-net skip structure; those specifics live in `03-model.md`. This section covers the universal mechanics.)

The network is a stack of identical **blocks** (this model has 52 of them — `num_layers`). Each block has two sub-components: a **self-attention** layer and a **feed-forward MLP**, wrapped with **residual connections** and **normalization**. The hidden state — one $d$-dimensional vector per token position — flows up through the stack, refined at each block.

### 4.1 Self-attention

Attention is the mechanism that lets each token's representation **incorporate information from other tokens**. For each token position the model computes three vectors by multiplying the hidden state by learned weight matrices (`c_q`, `c_k`, `c_v` in `model.py`):

- a **query** $q$ — "what am I looking for?"
- a **key** $k$ — "what do I offer?"
- a **value** $v$ — "what information do I carry?"

Attention then computes, for each position, a **weighted average of the value vectors of other positions**, where the weights come from how well this position's query matches each other position's key. The canonical formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

Here $Q, K, V$ are the stacked query/key/value vectors, $QK^\top$ is the matrix of all pairwise query-key dot products (similarity scores), $\sqrt{d_k}$ rescales them so they don't blow up, and `softmax` (§5) turns each row of scores into non-negative weights summing to 1. The output for a position is $\sum_j w_j v_j$ — a convex combination of values. In the code this is the single call `F.scaled_dot_product_attention(q, k, v, is_causal=True)`.

*Finance/econometrics analogy (useful but approximate):* attention is a **data-dependent weighted average**, structurally close to a **kernel smoother / kernel regression** where the "kernel weights" between positions $i$ and $j$ are $\propto \exp(q_i \cdot k_j / \sqrt{d_k})$ instead of a fixed distance kernel like $\exp(-\|t_i - t_j\|^2/h)$. The crucial difference: in a Nadaraya–Watson kernel the weights depend only on a *fixed* distance and bandwidth $h$; in attention the "similarity" $q_i \cdot k_j$ is **learned and content-dependent** — two tokens are "close" if the model has learned their query/key vectors should align, not because they are near in position. So it's an adaptive, learned kernel, not a fixed one. (Positional information enters separately, via RoPE — see `03-model.md`.)

### 4.2 Multi-head attention

Rather than one attention computation, the model runs several in parallel — **heads** (`num_heads`) — each with its own smaller query/key/value projections, and concatenates the results. Intuitively, different heads specialize: one might track syntactic agreement, another long-range topic, another the most recent token. *Loose analogy:* estimating several regressions on different subspaces of the regressors and stacking the fitted pieces. The split is what `view(B, T, num_heads, head_dim)` does in `CausalSelfAttention`.

### 4.3 The causal mask — and a warning about "no lookahead"

Because this is a *next-token predictor*, when computing the representation at position $t$ the model must **not** be allowed to attend to positions $> t$ — otherwise it would peek at the answer it is trying to predict. The **causal mask** enforces this by setting the attention weights to future positions to zero (mechanically, $-\infty$ before the softmax). That's the `is_causal=True` flag.

**Be careful with terminology here.** This causal mask is sometimes called "no look-ahead," and the *paper this project replicates* is centrally about avoiding **lookahead bias**. These are two completely different things, and conflating them in front of Manela would be a real error:

- The **causal mask** is a *within-sequence* architectural constraint: at training time, position $t$ cannot see token $t+1$ inside the same text window. Every GPT has it. It is about not trivially copying the label.
- **Lookahead bias** (the paper's theme) is a *temporal data-contamination* problem: a model used to study, say, year 2000 must not have been trained on text that didn't exist until 2005, or it "knows the future" relative to the research timestamp. ChronoGPT addresses this with **chronologically consistent training data** (vintage models with a knowledge cutoff) and a temporal screen on the SFT data (`data.py::keep_row`), *not* with the attention mask.

Both involve "not seeing the future," but at different levels (architecture vs. corpus construction). Keep them separate.

### 4.4 Feed-forward MLP, residuals, normalization

After attention mixes information *across* positions, each position is passed independently through a small **feed-forward MLP** (multi-layer perceptron): a linear layer that expands the dimension (here to $4d$), a nonlinearity, and a linear layer back down. The nonlinearity in this model is **ReLU-squared** (`F.relu(x).square()`); the standard choice elsewhere is GELU. The MLP is where most of the parameters live and is loosely the model's "per-token nonlinear feature transform."

Two pieces of glue make deep stacks trainable:

- **Residual connections.** Each sub-layer computes `x = x + sublayer(x)` rather than `x = sublayer(x)`. The block learns a *correction* to the running representation, not a wholesale replacement. This keeps gradients flowing through dozens of layers (it gives the backward pass a "shortcut" path). *Analogy:* like modeling $y_t = y_{t-1} + \Delta_t$ and learning the increments — the identity path is preserved by construction.
- **Normalization.** Before each sub-layer the hidden vector is normalized to a controlled scale. This model uses **RMSNorm** (`norm()` calls `F.rms_norm`), which divides by the root-mean-square of the vector's entries. It keeps activations numerically well-behaved across layers — roughly analogous to standardizing regressors so the optimization is well-conditioned, though it's applied per-vector at every layer, not once to the data.

So one block, schematically: `x = x + attention(norm(x))`, then `x = x + mlp(norm(x))`. Stack 52 of these, normalize, and project to vocabulary logits (§5).

---

## 5. Logits, softmax, cross-entropy, perplexity

After the final block, the model projects each position's hidden vector through an output matrix — the **language-model head**, `lm_head` in `model.py`, of shape $d \times V$ — producing a vector of $V$ real numbers per position called **logits**. Logits are unnormalized scores, one per vocabulary token. (This model also applies a "logit softcap" $15\tanh(\cdot/15)$ to bound them — a stability detail.)

**Softmax** turns logits $z$ into a probability distribution over the vocabulary:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.
$$

This is exactly the **multinomial logit / softmax** you know from discrete-choice models: each token is an "alternative," the logit is its utility, and $p_i$ is the choice probability. The model's predicted $p(x_t \mid x_{<t})$ from §1 *is* this softmax over the logits at position $t-1$.

**Cross-entropy loss.** For a known correct next token $y$, the loss is the negative log-probability the model assigned to it:

$$
\ell = -\log p_y = -\log \frac{e^{z_y}}{\sum_j e^{z_j}}.
$$

Averaged over all predicted positions, cross-entropy **is the negative average log-likelihood** of the data under the model. Minimizing cross-entropy = maximizing the likelihood of §1. Full circle to MLE. In `train.py` this is `F.cross_entropy(...)` inside `masked_lm_loss`. The "masked" part (covered in §7 and the data doc) means the average is taken only over the tokens the model is supposed to be graded on (`ignore_index=-100` skips the rest).

**Perplexity** is just $\text{PPL} = \exp(\text{cross-entropy})$ (you'll see `math.exp(...)` applied to val loss in `train.py`). Because cross-entropy is in **nats** (natural-log units), exponentiating gives an interpretable number: the model's *effective branching factor* — roughly, "on average the model is as uncertain as if it were choosing uniformly among PPL tokens." Lower is better. A perplexity of 20 means the model's uncertainty equals a uniform 20-way guess. *Analogy:* perplexity is to a categorical AR likelihood what $e^{\text{(per-obs log-loss)}}$ is — a monotone, more legible transform of the same likelihood you're already optimizing.

---

## 6. How training works

Training is iterative numerical optimization of the cross-entropy objective. One iteration:

1. **Forward pass.** Run a batch of token sequences through the network to get logits, then the loss (§5). "Forward" because data flows input → output.
2. **Backpropagation.** Compute the gradient of the scalar loss with respect to *every* parameter, $\nabla_\theta \ell$. This is just the **chain rule**, applied mechanically and efficiently backward through the computational graph (output → input). You know the chain rule for a few composed functions; backprop is the same rule bookkept automatically (by PyTorch's autograd) across millions of parameters and dozens of layers, reusing intermediate results so the whole gradient costs about one extra forward pass. In code, the entire backward pass is the single call `loss.backward()`.
3. **Parameter update.** Take a step downhill: $\theta \leftarrow \theta - \eta \, \nabla_\theta \ell$, where $\eta$ is the **learning rate** (step size). This is **gradient descent**. Repeat.

*Econometrics analogy:* this is numerical MLE by gradient ascent on the log-likelihood — the same family as a Newton/BHHH/quasi-Newton optimizer maximizing a likelihood you can't solve in closed form. The differences are scale (millions of parameters) and that we never form a Hessian.

**Batches, steps, epochs.**
- We never use the whole corpus at once. A **batch** (or mini-batch) is a small set of sequences (here `batch_size: 8`) processed together. Using batches instead of the full dataset is **mini-batch stochastic gradient descent (SGD)** — each gradient is a *noisy estimate* of the full-data gradient, which is both cheaper and, empirically, helpful for generalization.
- A **step** (or iteration) is one parameter update — one forward/backward/update cycle (possibly aggregating several batches; see gradient accumulation in §8).
- An **epoch** is one full pass over the training data. This project trains each curriculum stage for a few epochs (`epochs: 3`, `2`, `2` in `configs/train.yaml`).

**Optimizers: SGD vs Adam/AdamW.** Plain SGD uses the same scalar learning rate for every parameter. **Adam** improves on this in two ways: (i) **momentum** — it averages recent gradients (an EWMA), smoothing the noisy mini-batch signal, like adding inertia to the descent; and (ii) **adaptive per-parameter step sizes** — it scales each parameter's step by a running estimate of that gradient's magnitude (its second moment), so parameters with consistently small gradients take relatively larger steps and vice-versa. *Analogy:* loosely like a diagonal preconditioner / a crude per-parameter standardization of the gradient, giving you something step-size-wise reminiscent of a diagonal-Hessian scaling without computing a Hessian. **AdamW** is Adam with **decoupled weight decay** — an explicit $\ell_2$ pull of the weights toward zero applied separately from the gradient step (a cleaner form of ridge-style regularization). This codebase uses `torch.optim.AdamW` (in `train.py`), the standard choice for training Transformers.

---

## 7. Pretraining vs fine-tuning vs instruction tuning (SFT)

These three terms name *stages*, not different algorithms — they all do the MLE-by-gradient-descent of §6. What changes is the data and the starting point.

- **Pretraining.** Train from random initialization on a massive, generic text corpus to predict the next token. The result is a **base model** (here, the released `chrono-gpt-v1-20201231`). A base model is a superb *text continuer* but it does not "follow instructions" — give it the literal text *"Write a poem about bonds."* and it may continue with more instructions, or a list, because that's what such text is often followed by in the wild. It models $p(\text{next token})$, full stop. **This project does not do pretraining**; it starts from the released base checkpoint.

- **Fine-tuning.** Continue training a pretrained model on a smaller, narrower dataset, so it adapts without re-learning language from scratch. Same loss, far fewer steps, smaller learning rate. *Analogy:* using pooled-sample estimates as an informative prior / starting values, then updating on a focused subsample.

- **Instruction tuning, a.k.a. SFT (Supervised Fine-Tuning).** A *specific kind* of fine-tuning where the data is **(instruction → desired response) pairs**, teaching the base model to behave as a helpful assistant that answers the prompt. This is the entire training task of this repo (`train.py`).

  - **"Supervised"** here means each training example has an explicit target the model is graded against — the human/curated response — exactly like the dependent variable in a supervised regression. (Contrast with the *self*-supervised pretraining objective, where the "label" is just the next token of raw text, which requires no human annotation.)
  - Mechanically, each example is rendered in a fixed **prompt template** (the Alpaca format, `PROMPT_WITH_INPUT` / `PROMPT_NO_INPUT` in `data.py`: an `### Instruction:` / `### Response:` scaffold) and the cross-entropy loss is **masked to the response tokens only** — the model is graded on producing the answer, not on reciting the prompt back. That masking is the `target_mask` / `-100` labels built in `data.py::encode_example` and `pack_blocks`, and consumed by `ignore_index=-100` in the loss (§5). Conceptually: condition on the prompt (regressors), fit only the response (dependent variable).

  This repo runs SFT as a 3-stage **curriculum** (`stage1_scratch → stage2_self_instruct → stage3_tulu`), each stage continuing from the previous stage's weights — see the training and data docs for why.

---

## 8. Practical training machinery

You'll meet each of these in `train.py`, `model.py`, or `configs/train.yaml`. They don't change the objective; they make a large model fit and train efficiently on a single GPU.

- **Mixed precision (bf16 vs fp32).** Numbers can be stored in 32-bit float (`fp32`, the default, accurate) or 16-bit. `bf16` (bfloat16) is a 16-bit format with the *same exponent range* as fp32 but fewer mantissa bits — so it has fp32's dynamic range (won't overflow) at half the memory and roughly double the throughput, at the cost of precision. Training runs the heavy matrix math in bf16 while keeping a few sensitive accumulations in fp32. In the code this is `torch.autocast(device_type=..., dtype=torch.bfloat16)` wrapping the forward pass (this is "automatic mixed precision," **autocast**). The embeddings are explicitly cast `.bfloat16()` in `model.py`.

- **Gradient accumulation.** Big batches train more stably but don't fit in memory. Instead you run several small batches, **sum (accumulate) their gradients without updating**, then take one optimizer step — simulating a batch `accum` times larger. Here `batch_size: 8` × `grad_accum: 4` = an *effective* batch of 32 sequences per update. In `train.py` the loss is divided by `accum` and `opt.step()` only fires every `accum` micro-batches.

- **Gradient clipping.** If a gradient's overall norm exceeds a threshold, rescale it down before stepping. This prevents a single freak batch from blowing up the weights ("exploding gradients"). `grad_clip` in the config (default `1.0`; set `null` to disable, in which case the norm is still logged via `clip_grad_norm_`). *Analogy:* winsorizing the update direction.

- **Learning-rate warmup + cosine decay.** The learning rate is not constant. It **warms up** linearly from ~0 over the first few percent of steps (`warmup_ratio: 0.03`) — large early steps from random-ish state are destabilizing — then **decays** along a cosine curve toward 0 over the rest of training, taking ever-smaller steps as it (hopefully) nears a good minimum. Implemented in `train.py::cosine_lr`.

- **Gradient checkpointing.** During the forward pass the model normally stores every intermediate **activation** so the backward pass can reuse them. That's the dominant memory cost for deep models. Gradient *checkpointing* discards most activations and **recomputes them on demand during the backward pass** — trading ~20% extra compute for ~10× less activation memory. Toggled by `model.grad_checkpoint` (`grad_checkpoint: true` in the config); in `model.py` it wraps each block in `torch.utils.checkpoint.checkpoint`. It is what lets this model train on a single 80 GB card.

- **GPU memory budget — the "16 bytes/param" rule.** It's worth internalizing where VRAM goes when training with AdamW in mixed precision. Per parameter you pay roughly:
  - model **weights**: ~2 bytes (bf16) [+ often a 4-byte fp32 master copy],
  - **gradients**: ~2–4 bytes (one per weight),
  - **optimizer states**: Adam keeps *two* fp32 running averages (momentum + second moment) = **8 bytes**,

  which is the well-known rule of thumb that Adam training costs about **16 bytes per parameter** *before counting activations*. Activations are the *fourth*, separate cost — they scale with batch size × sequence length × layers, not with parameter count, and they are exactly what gradient checkpointing attacks. So the four memory consumers to keep distinct are: **weights, gradients, optimizer states, activations.** (At inference you pay only weights — hence inference fits in a fraction of the memory.)

---

## 9. Generation and inference

Once trained, the model is used in two distinct modes. Both are in `infer.py`.

**Text generation.** To produce text, the model predicts the next-token distribution (§5), picks a token, appends it, and repeats — **autoregressive decoding**. How you "pick" matters:

- **Greedy decoding** takes the single highest-probability token every step. Deterministic and repeatable. It is the **default** here: `generate(..., temperature=0.0, top_k=None)` is greedy (matching manelalab's `ChronoGPT_instruct.py`), and `top_k=1` is greedy too. Used for the reproducible evaluation tests (`eval.py`'s president and major-events checks, and AlpacaEval generation, all decode greedily so decoding strategy doesn't confound the comparison).
- **Temperature** rescales the logits by $1/T$ before softmax. $T < 1$ sharpens the distribution (more conservative); $T > 1$ flattens it (more diverse/random); $T = 0$ is greedy (handled by `argmax`, not division). Pass `temperature>0` to sample.
- **Top-k sampling** restricts the choice to the $k$ most probable tokens, renormalizes, and samples from those — random but guard-railed against absurd low-probability tokens. Enable it with e.g. `generate(..., temperature=0.8, top_k=50)`.

(For speed, generation uses an optional **KV cache** by default — `use_cache=True` — so each step feeds only the new token instead of recomputing the whole sequence ($O(T)$ vs $O(T^2)$); `use_cache=False` restores the simple full-recompute path. See `03-model.md` and `06-infer-and-eval.md` for the mechanism.)

**Embedding extraction.** The *other* inference mode produces no text at all. It runs one forward pass and reads out an internal **hidden state** as a feature vector — `infer.py::embed` returns a chosen layer's output, mean-pooled over the sequence (`pool="mean"`). This is the deep, context-dependent "embedding" flagged in §3, and it is the representation a finance researcher would feed into a downstream predictive model (e.g. regressing future returns on the text embedding of a filing). Generation asks *"what comes next?"*; embedding extraction asks *"what is the model's internal representation of this text?"* — same network, different read-out.

---

## 10. Glossary

| Term | One-line definition |
|---|---|
| **token** | A sub-word unit (integer ID) the model reads; ~0.75 words. Vocabulary here is GPT-2 BPE (~50,257). |
| **embedding** | A learned dense vector representing a token (input table) or, at inference, a deep hidden-state feature vector for a text. |
| **attention** | A data-dependent weighted average over other positions' value vectors; weights from query·key similarity. |
| **head** | One of several parallel attention computations in a layer; each can specialize. |
| **logit** | An unnormalized output score, one per vocabulary token, before softmax. |
| **softmax** | Maps logits to a probability distribution (the multinomial-logit transform). |
| **cross-entropy** | The training loss; negative average log-likelihood of the correct next tokens. |
| **perplexity** | $\exp(\text{cross-entropy})$ — effective branching factor; lower is better. |
| **epoch** | One full pass over the training data. |
| **step** | One parameter update (one forward/backward/update cycle). |
| **batch** | A set of sequences processed together; gradient is a noisy estimate of the full-data gradient. |
| **gradient** | $\nabla_\theta\,\text{loss}$ — the vector of partial derivatives of the loss w.r.t. every parameter. |
| **backprop** | Reverse-mode automatic differentiation; the chain rule applied backward through the network (`loss.backward()`). |
| **optimizer** | The rule that turns gradients into weight updates. |
| **AdamW** | Adam (momentum + adaptive per-parameter step sizes) with decoupled weight decay; the optimizer used here. |
| **learning rate** | The step size $\eta$ in the update; here warmed up then cosine-decayed. |
| **warmup** | Linearly ramping the learning rate from ~0 over the first few % of steps for stability. |
| **checkpoint** | (1) A saved snapshot of model weights on disk; (2) *gradient checkpointing*: recomputing activations in the backward pass to save memory. |
| **fine-tuning** | Continuing training of a pretrained model on narrower data with a small learning rate. |
| **SFT** | Supervised Fine-Tuning: instruction→response fine-tuning with a labeled target; this repo's whole training task. |
| **RoPE** | Rotary Position Embedding — injects token position by rotating query/key vectors (see `03-model.md`). |
| **RMSNorm** | Root-mean-square normalization of a hidden vector; the layer-norm variant used here. |
| **autocast / bf16** | Automatic mixed precision; running math in 16-bit bfloat16 (fp32 range, half the memory). |
| **packing** | Concatenating many examples into fixed-length token blocks (`block_size`) instead of padding each separately. |
| **masking** | Setting label = `-100` on non-response tokens so the loss grades only the answer span. |
| **EOT** | End-of-text token (ID 50256), appended after each response to mark where an answer ends. |

---

## 11. What to read next

- **`03-model.md`** — the ChronoGPT architecture in detail: the modded-nanoGPT U-net with encoder/decoder skip connections, value embeddings, RoPE, RMSNorm, ReLU² MLP, and logit softcap. Read it with §4 of this primer in hand.
- **The data doc** — how `data.py` applies the temporal screen (the *lookahead-bias* mechanism of §4.3, not the causal mask), reconstructs the 3-stage curriculum, and packs/masks examples (§2, §7).
- **The training doc** — how `train.py` wires together the loss, AdamW, accumulation, cosine schedule, checkpointing, and evaluation (§5, §6, §8).
- **The evaluation/inference doc** — the president and major-events consistency tests and AlpacaEval, plus embedding extraction (§9).
