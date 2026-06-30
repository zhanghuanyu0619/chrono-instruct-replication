# 05 — The Training Loop (`train.py`)

This document is a line-by-line reading of `src/chrono_instruct/train.py`, the
file that actually fine-tunes ChronoGPT. It is the smallest file in the project
(189 lines) but the densest: it holds the masked-loss objective, the
learning-rate schedule, the evaluation routine, the per-step optimization loop,
and the multi-stage orchestrator. I assume you have read `01-ml-primer.md` (loss,
backprop, AdamW, warmup/cosine, gradient accumulation, gradient clipping, mixed
precision, epochs/steps/batches); I will lean on your optimization background and
explain only the deep-learning-specific machinery.

If you only remember one framing: this is **gradient descent on an average
negative log-likelihood**, exactly the kind of objective you already minimize in
MLE — the novelty is the masking (we score only the answer tokens), the
mini-batch/accumulation bookkeeping, and the memory engineering needed to fit a
1.55-billion-parameter model on one GPU.

## Table of contents
1. [The objective: `masked_lm_loss`](#1-the-objective-masked_lm_loss)
2. [The schedule: `cosine_lr`](#2-the-schedule-cosine_lr)
3. [Honest evaluation: `evaluate`](#3-honest-evaluation-evaluate)
4. [The heart: `train_stage`](#4-the-heart-train_stage)
5. [The orchestrator: `run`](#5-the-orchestrator-run)
6. [The curriculum / stage-by-stage workflow in practice](#6-the-curriculum--stage-by-stage-workflow-in-practice)
7. [The memory story (weights vs grads vs Adam vs activations)](#7-the-memory-story)
8. [Mini-FAQ of subtle points](#8-mini-faq-of-subtle-points)

---

## Module docstring and imports

```python
1  """Curriculum SFT training loop.
2
3  One run = one vintage = one process on one GPU. Stages are trained in order
4  (scratch -> self-instruct -> tulu-3), each continuing from the previous stage's
5  weights. Loss is masked cross-entropy on response tokens only. Multi-GPU /
6  cluster fan-out is handled outside this file (see scripts/), never here.
7  """
```

The mental model for the whole file: **one process, one GPU, one "vintage"** (a
vintage is one chronological cutoff of ChronoGPT, e.g. the 2020-12-31 model). The
three SFT stages run sequentially inside that one process, each picking up the
weights the previous stage left in memory. There is no distributed-training code
here on purpose — that complexity lives in `scripts/` and `08-configs-scripts-infra.md`.

```python
8   import json
9   import math
10  import os
11  import time
12
13  import torch
14  import torch.nn.functional as F
15  import yaml
16  from torch.utils.data import DataLoader
17
18  from .model import ChronoGPT
19  from .data import prepare_stages
20  from .tracking import RunLogger
```

`torch.nn.functional as F` gives us `F.cross_entropy` (line 26). `DataLoader`
turns a dataset into shuffled mini-batches. The three sibling imports are the
collaborators: `ChronoGPT` (the model, see `03-model.md`), `prepare_stages` (the
data pipeline, see `04-data.md`), and `RunLogger` (CSV +
optional Weights & Biases logging, see `07-cli-tracking-hub-figures.md`). `hub.push_dir` is
imported lazily inside `run` (line 180) so a non-pushing run never needs the Hub
dependency.

---

## 1. The objective: `masked_lm_loss`

```python
23  def masked_lm_loss(logits, labels, reduction="mean"):
24      shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
25      shift_labels = labels[:, 1:].reshape(-1)
26      return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction=reduction)
```

This four-line function is the entire learning objective. Read it carefully —
everything else in the file exists to feed and minimize it.

**The shapes.** `logits` has shape `[B, T, V]`: for each of the `B` sequences in
the batch, at each of the `T` positions, the model emits a vector of `V` raw
scores (one per vocabulary token; here `V = 50304`). `labels` has shape `[B, T]`:
the integer token id we *want* at each position, or the sentinel `-100` on
positions we want to ignore.

**The shift-by-one (line 24–25).** A causal language model at position `t` predicts
token `t+1`. So the prediction made *at* position `t` (`logits[:, t]`) must be
scored against the token that *actually appears* at position `t+1`
(`labels[:, t+1]`). The two slices realize exactly this alignment:
- `logits[:, :-1, :]` keeps predictions at positions `0 … T-2` (we drop the last
  position's prediction, since there is no `T`-th token to check it against).
- `labels[:, 1:]` keeps the true tokens at positions `1 … T-1`.

So `shift_logits[i]` (the prediction made one step earlier) is now paired with
`shift_labels[i]` (the token that should follow). This is the standard
"teacher-forced next-token" alignment. The shift is done *here* in the loss, not
in the data — `data.py`'s `pack_blocks` stores `labels[t] = input_ids[t]`
unshifted (see its docstring, lines 110–131), and this function does the offset.

**The flatten (`.reshape`).** `cross_entropy` wants a 2-D table of scores
`[N, V]` and a 1-D vector of targets `[N]`. We collapse the batch and time axes
together: `shift_logits` becomes `[(B·(T-1)), V]` and `shift_labels` becomes
`[(B·(T-1))]`. Every (sequence, position) pair is now just one independent
classification example. Order does not matter for a sum or a mean, so flattening
is safe.

**The cross-entropy itself.** For one position with true token `y` and model
logits `z`, cross-entropy is

```
loss = -log softmax(z)[y] = -log( exp(z[y]) / Σ_j exp(z[j]) )
```

This is precisely the **negative log-likelihood** of the correct next token under
the model's predictive distribution. Minimizing the average of this over the
corpus is **maximum-likelihood estimation** of the autoregressive model
`p_θ(token_{t+1} | tokens_{≤t})` — the same MLE you know, just with a softmax
likelihood and a neural net producing the logits. Nothing exotic.

**`ignore_index=-100` — why the mask makes this response-only.** This is the SFT
trick. `cross_entropy` skips any position whose label equals `-100`: it
contributes nothing to the loss and nothing to the gradient. In `data.py`,
`pack_blocks` sets `labels = -100` on every *prompt* token (the instruction and
input) and the true token id on every *response* token (lines 125–126 of
`data.py`). The consequence: we never penalize the model for "predicting" the
instruction we ourselves wrote — we only train it to *produce good answers*. This
is the response-masking decision documented in `implementation-notes.md` §2, and
it is the near-universal reading of "masked cross-entropy" for instruction
tuning. (Note this is a *loss* mask, separate from the causal attention mask
inside the model.)

**`reduction` — `"mean"` for training, `"sum"` for honest eval.**
- `reduction="mean"` (the default, used in training) averages over all non-ignored
  positions in this batch, giving the per-token average NLL. That is the scalar we
  backprop.
- `reduction="sum"` (used by `evaluate`) returns the *total* NLL over the batch's
  response tokens, *not* divided by anything. The caller divides by the global
  token count itself. Why this matters is explained in §3.

---

## 2. The schedule: `cosine_lr`

```python
29  def cosine_lr(step, total, base_lr, warmup):
30      if step < warmup:
31          return base_lr * (step + 1) / max(1, warmup)
32      progress = (step - warmup) / max(1, total - warmup)
33      return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
```

A pure function: given the current optimizer step, the total number of steps, the
peak learning rate, and the number of warmup steps, return the learning rate to
use *now*. It has two regimes.

**Warmup (lines 30–31).** For the first `warmup` steps, the LR ramps linearly
from `base_lr/warmup` up to `base_lr`. The `(step + 1)` means step 0 already gets
a tiny nonzero LR rather than exactly zero. Why warmup at all? At initialization
(and especially at the start of a new SFT stage), the gradients are large and
noisy, and Adam's running variance estimates (`v`) are not yet calibrated.
Taking full-size steps immediately can throw the weights into a bad region from
which training never recovers. A short ramp lets the optimizer "find its footing"
— it is the standard fix for the well-known early-training instability of
transformers. Here `warmup = 3%` of total steps (`warmup_ratio: 0.03` in the
config).

**Cosine decay (lines 32–33).** After warmup, `progress` runs from 0 to 1 across
the remaining steps. The factor `0.5 · (1 + cos(π · progress))` is a smooth
half-cosine: it equals 1 at `progress = 0` (LR = `base_lr`), `0.5` at the
midpoint, and 0 at `progress = 1` (LR → 0). The curve is flat near both ends and
steepest in the middle. Annealing the LR to ~0 by the end lets the optimizer
settle into a minimum instead of bouncing around it — analogous to decreasing the
step size in any stochastic-approximation scheme to guarantee convergence. The
`max(1, …)` guards prevent division by zero in degenerate tiny runs.

A subtlety worth noting: `total` here is `steps_per_epoch * stage["epochs"]` for
*this stage only* (line 71). Each stage gets its own fresh warmup-then-cosine
cycle, with the LR returning to 0 at the end of each stage. This matches the
curriculum framing — each stage is its own short training run that happens to
start from the previous stage's weights.

---

## 3. Honest evaluation: `evaluate`

```python
36  @torch.no_grad()
37  def evaluate(model, loader, device):
38      """Token-weighted mean response loss over the WHOLE val set.
...
46      model.eval()
47      total_loss, total_tokens = 0.0, 0
48      for ids, labels in loader:
49          ids, labels = ids.to(device), labels.to(device)
50          with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
51              logits, _ = model(ids, return_hidden=False)
52          total_loss += masked_lm_loss(logits, labels, reduction="sum").item()
53          total_tokens += int((labels[:, 1:] != -100).sum())
54      model.train()
55      return total_loss / max(1, total_tokens)
```

This computes the validation loss — our running yardstick for whether training
is helping. Four DL-specific mechanisms are bundled here.

**`@torch.no_grad()` and `model.eval()`/`model.train()` (lines 36, 46, 54).**
- `@torch.no_grad()` tells PyTorch not to build the autograd graph during this
  function. We are only measuring, not learning, so we do not need gradients —
  and skipping them saves a lot of memory (no activations are retained for a
  backward pass that will never happen).
- `model.eval()` switches the model into evaluation mode, then `model.train()`
  restores training mode on the way out. For models with dropout or batch-norm
  this changes behavior; ChronoGPT has neither, so this is mostly hygiene — but it
  also flips `self.training`, which matters because the model's gradient
  checkpointing only activates when `self.training` is true (see `model.py` line
  167). So calling `eval()` here correctly disables checkpointing during the
  no-grad eval pass.

**`autocast` to bfloat16 (line 50).** Eval runs under the *same* mixed-precision
context as training (see §4). This is deliberate: we want the val number to
reflect the same numerical regime the model trains and will be served in, not a
different fp32 measurement. `return_hidden=False` (line 51) tells the model to
skip building the per-layer hidden-state list it would otherwise return — pure
overhead we do not need here (see `03-model.md`).

**Token-weighted aggregation (lines 52–53, 55) — the important one.** We
accumulate the *summed* loss (`reduction="sum"`) across every batch, and
separately count the number of scored response tokens
(`(labels[:, 1:] != -100).sum()` — note the same shift-by-one as the loss, so the
count matches exactly what cross-entropy scored). At the end we divide:
`total_loss / total_tokens`. This is the *true* per-token average negative
log-likelihood over the whole validation set.

Why not just average the per-batch mean losses? Because **batches hold different
numbers of response tokens** — packing concatenates variable-length examples, so
one block might be 90% response tokens and the next 30%. Averaging the per-batch
means would weight a token in a sparse batch more heavily than a token in a dense
batch, biasing the number. Think of it as the difference between a simple average
of group means and a properly size-weighted pooled mean — only the latter equals
the grand mean when group sizes differ. The token-weighted form is the honest one
and is what the paper's Figure 1 curves report.

**The whole (bounded) val set.** The loop runs over the *entire* `val_loader`,
not a single batch — so the number is a full-pass estimate, not a noisy
one-batch sample. This is affordable because the val set is capped at
`val_max_blocks` (500 in the config) back in `data.py`'s `load_stage` (lines
158–159), keeping a full eval cheap enough to call every `eval_every` steps.

The return value is the val loss; the caller turns it into perplexity for logging
as `exp(loss)` (line 86) — perplexity is just the exponentiated average NLL, the
effective "number of equally-likely choices" the model is confused among.

---

## 4. The heart: `train_stage`

This function trains one stage. I will walk it top to bottom.

### 4.1 Setup: loaders, optimizer, step budget

```python
58  def train_stage(model, train_ds, val_ds, cfg, stage, device, run_logger=None):
59      g = torch.Generator().manual_seed(cfg["seed"])  # global seed -> deterministic shuffle
60      loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, generator=g)
61      val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
62      opt = torch.optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=cfg.get("weight_decay", 0.0))
```

**Seeded shuffle (lines 59–60).** A `DataLoader` with `shuffle=True` draws a
random permutation of the dataset each epoch. We pass it an explicit
`Generator` seeded from the single global `cfg["seed"]` so the shuffle is
*reproducible*: rerun the same config and you get the identical batch order. This
is part of the one-seed-drives-everything reproducibility design
(`implementation-notes.md` §8). `drop_last=True` discards the final partial
mini-batch so every batch has exactly `batch_size` sequences — important here
because the model has no padding-mask support and we want uniform shapes.

The `val_loader` (line 61) is *not* shuffled and not seeded — order is irrelevant
for a sum.

**The optimizer (line 62).** `torch.optim.AdamW` over `model.parameters()` —
**all 1.55B of them**. This is full fine-tuning: no LoRA, no frozen layers, no
adapters (`implementation-notes.md` §7). AdamW is Adam with *decoupled* weight
decay: the L2 penalty is applied directly to the weights as a separate shrink
step, not folded into the gradient (which is what plain "Adam + L2" did, and what
makes its decay interact badly with Adam's adaptive scaling). Here
`weight_decay` defaults to `0.0` (config line 24), so we are effectively running
plain Adam — a safe, well-understood choice for SFT. The LR is `stage["lr"]`, but
note it gets *overwritten every step* by the schedule (line 110); this initial
value barely matters. `betas` are left at PyTorch defaults `(0.9, 0.999)` (the
config does not set them), i.e. the standard Adam momentum/variance decay rates.

### 4.2 Step budget and bookkeeping

```python
64      name = stage["name"]
65      accum = cfg.get("grad_accum", 1)
66      log_every = cfg.get("log_every", 20)
67      eval_every = cfg.get("eval_every")
68      grad_clip = cfg.get("grad_clip")          # None -> no clipping (norm still logged)
69      tokens_per_step = cfg["batch_size"] * cfg["block_size"] * accum
70      steps_per_epoch = len(loader) // accum    # only full accum groups step; floor matches reality
71      total_steps = steps_per_epoch * stage["epochs"]
72      warmup = int(total_steps * cfg.get("warmup_ratio", 0.03))
```

**`accum` and the optimizer "step" vs the data "batch".** This is the crucial
distinction for reading the rest of the loop. A *batch* is one forward/backward
pass over `batch_size` sequences. An optimizer *step* is one actual weight
update. With gradient accumulation, we sum gradients over `accum` batches before
updating, so **one step = `accum` batches**. The effective batch size is
`batch_size × accum = 8 × 4 = 32` sequences — large enough for stable gradients,
while only ever holding 8 sequences' worth of activations in memory at once. (See
`01-ml-primer.md` for the accumulation primer and §7 below for the memory reason.)

**`tokens_per_step` (line 69).** `batch_size × block_size × accum` — the number
of tokens that flow through the model between two weight updates. Used purely to
report throughput in tokens/sec.

**`steps_per_epoch = len(loader) // accum` — floor, not ceiling (line 70).**
`len(loader)` is the number of batches per epoch. Integer-dividing by `accum`
counts only the *complete* accumulation groups. If the last group is partial (not
enough batches left to fill `accum`), it never triggers a weight update, so it
should not be counted as a step. Using floor makes `total_steps` (line 71) match
the number of `opt.step()` calls that will actually happen — which keeps the LR
schedule honest. If we used ceiling, `cosine_lr` would think there are more steps
than occur and would never quite anneal the LR to 0. (The leftover partial group
is handled separately at the epoch boundary — see §4.6.)

`warmup` (line 72) is `warmup_ratio` (3%) of `total_steps`, feeding `cosine_lr`.

### 4.3 Two logging helpers

```python
74      def mem_gb():
75          return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
76
77      def log_point(step, epoch, train_loss, lr, grad_norm, tps):
78          """Log train AND val at the SAME step (aligned curves, like the paper's Fig 1)."""
79          vloss = evaluate(model, val_loader, device)
80          print(f"[{name}] step {step}/{total_steps} train {train_loss:.4f} val {vloss:.4f} "
81                f"lr {lr:.2e} |g| {grad_norm:.2f} {tps:,.0f} tok/s {mem_gb():.1f}GB")
82          if run_logger:
83              run_logger.log(stage=name, epoch=epoch, step=step, split="train", loss=round(train_loss, 4),
84                             lr=lr, grad_norm=round(grad_norm, 3), tokens_per_sec=round(tps), gpu_mem_gb=round(mem_gb(), 1))
85              run_logger.log(stage=name, epoch=epoch, step=step, split="val",
86                             loss=round(vloss, 4), ppl=round(math.exp(min(vloss, 20)), 2))
87          return vloss
```

`mem_gb()` reports peak GPU memory (in GB) seen so far — a diagnostic for how
close we are to OOM.

`log_point` is the key design fix for clean figures. **Every time it is called it
logs a train point AND a val point at the *same* step** (line 79 evaluates,
lines 83–86 write both rows to the CSV). The comment "aligned curves, like the
paper's Fig 1" is the why: an earlier version logged train and val on different
cadences, so the two curves had different x-axis points and could not be overlaid
cleanly. Logging them together — same start, same cadence, same end — means the
metrics.csv produces a Figure 1 where the train and val lines share an x-axis.
The val perplexity is `exp(min(vloss, 20))` — the `min(…, 20)` is just an
overflow guard so a pathological early loss does not produce `inf`.

### 4.4 The step-0 anchor

```python
89      # step 0: the starting point (base / previous-stage weights, before this stage updates anything)
90      ids0, labels0 = next(iter(loader))
91      with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
92          logits0, _ = model(ids0.to(device), return_hidden=False)
93      val_loss = log_point(0, 0, masked_lm_loss(logits0, labels0.to(device)).item(),
94                           cosine_lr(0, total_steps, stage["lr"], warmup), 0.0, 0.0)
```

Before doing *any* weight update, we run one forward pass on the first batch and
call `log_point(step=0, …)`. This records the loss of the **starting weights** —
the base vintage model for Stage 1, or the previous stage's output for Stages
2–3. Why bother? Because without it, the loss curve would start at step 1 (after
one update already happened), and you could not see *where training began*. The
step-0 anchor gives both the train and val curves a real, honest origin: you can
read off "the base model had val loss X, and after training it's Y." `grad_norm`
and `tps` are passed as `0.0` because no gradient or timing exists yet. This is
done under `no_grad` (we are only measuring) and the same bf16 autocast.

### 4.5 The per-step training loop

```python
96      step = 1  # step 0 is the pre-training anchor above; counting updates from 1
97      last_t, last_step = time.time(), 0
98      tl_sum, tl_n, grad_norm = 0.0, 0, 0.0
99      for epoch in range(stage["epochs"]):
100         for i, (ids, labels) in enumerate(loader):
101             ids, labels = ids.to(device), labels.to(device)
102             with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
103                 logits, _ = model(ids, return_hidden=False)
104                 loss = masked_lm_loss(logits, labels) / accum
105             loss.backward()
```

`step` counts weight updates starting at 1 (step 0 was the anchor). `tl_sum` /
`tl_n` accumulate train losses between log points so we can report a smoothed
average rather than one noisy batch. The outer loop is over epochs; the inner
loop is over batches.

**Forward under autocast (lines 102–103).** `torch.autocast(..., bfloat16)` is
PyTorch's automatic mixed precision: inside this context, the heavy matmuls run in
**bfloat16** (a 16-bit float with the same exponent range as fp32 but fewer
mantissa bits), while numerically sensitive ops stay in fp32. bf16 halves
activation memory and roughly doubles matmul throughput on modern GPUs. bf16
specifically (vs fp16) keeps fp32's dynamic range, so it does *not* need loss
scaling to avoid gradient underflow — which is why you will not find a
`GradScaler` anywhere in this file. (For the finance reader: this is purely a
speed/memory optimization on the arithmetic; the optimization problem is
unchanged. The master weights and Adam states stay in fp32 — see §7.)

**The forward call** returns `(logits, _)` — we discard the hidden-state list by
both the `_` and `return_hidden=False`.

**Loss scaled by `1/accum` (line 104).** This is the gradient-accumulation
correction. We will sum the gradients of `accum` batches before stepping. The
gradient of a sum is the sum of gradients, so summing `accum` un-scaled losses
would give `accum` times too large a gradient. Dividing each batch's loss by
`accum` *before* `backward()` means the accumulated gradient equals the gradient
of the *mean* over all `accum × batch_size` sequences — exactly what we want from
an effective batch of size 32. `loss.backward()` (line 105) computes gradients of
this scaled loss and *adds* them into each parameter's `.grad` (PyTorch
accumulates gradients by default — usually a footgun, here a feature).

### 4.6 The accumulation boundary: clip, step, zero

```python
106             if (i + 1) % accum == 0:
107                 grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or 1e30))
108                 lr = cosine_lr(step, total_steps, stage["lr"], warmup)
109                 for group in opt.param_groups:
110                     group["lr"] = lr
111                 opt.step()
112                 opt.zero_grad(set_to_none=True)
113                 tl_sum += loss.item() * accum
114                 tl_n += 1
```

`(i + 1) % accum == 0` fires once every `accum` batches — this is the boundary of
one optimizer step. At the boundary:

- **Gradient clipping (line 107).** `clip_grad_norm_` computes the global L2 norm
  of all gradients and, if it exceeds the threshold, scales every gradient down so
  the norm equals the threshold. This caps the size of a single update,
  protecting against the occasional exploding gradient that would otherwise
  corrupt the weights. The config default is now **`grad_clip: 1.0`** (the
  standard SFT value), so clipping is active. The `grad_clip or 1e30` idiom is the
  off-switch: if you set `grad_clip: null`, the threshold becomes effectively
  infinite (`1e30`), so **no clipping happens** — but the function *still returns
  the true gradient norm*, which we capture in `grad_norm` for logging. So we
  always *monitor* `|g|` whether or not we clip it. (Watching `|g|` is itself a
  useful diagnostic — a spiking grad norm is an early warning of instability.)
- **Set the LR (lines 108–110).** Compute the scheduled LR for this step and write
  it into every Adam parameter group. This is how `cosine_lr` actually drives the
  optimizer — AdamW does not know about schedules; we override `group["lr"]`
  manually each step.
- **`opt.step()` (line 111).** The actual AdamW update: each parameter moves
  according to its first/second-moment estimates and the current LR.
- **`opt.zero_grad(set_to_none=True)` (line 112).** Reset gradients to `None`
  before the next accumulation group, so the next group accumulates from a clean
  slate. `set_to_none=True` (frees the tensors rather than filling them with
  zeros) is the memory-cheaper form.
- **Track train loss (lines 113–114).** `loss.item() * accum` undoes the `1/accum`
  scaling to recover the true per-token loss of this batch, accumulating it for
  the smoothed average reported at the next log point.

### 4.7 Periodic logging, eval, and checkpointing

```python
115             if step % log_every == 0:
116                 print(f"[{name}] step {step}/{total_steps} train {loss.item() * accum:.4f}")
117             if eval_every and step % eval_every == 0:        # train+val logged together
118                 now = time.time()
119                 tps = (step - last_step) * tokens_per_step / (now - last_t)
120                 last_t, last_step = now, step
121                 val_loss = log_point(step, epoch, tl_sum / max(1, tl_n), lr, grad_norm, tps)
122                 tl_sum, tl_n = 0.0, 0
123             if cfg.get("save_every") and step % cfg["save_every"] == 0:
124                 model.save_pretrained(os.path.join(cfg["output_dir"], f"{name}-step{step}"))
125             step += 1
```

Three independent cadences keyed off `step`:
- **`log_every` (20 steps): console only.** A cheap heartbeat print of the current
  train loss, no eval.
- **`eval_every` (200 steps): the real measurement.** Compute throughput `tps`
  (tokens processed since the last eval ÷ wall-clock elapsed), reset the timer,
  then call `log_point` to evaluate the val set and write *both* train and val
  rows. The train loss reported is `tl_sum / tl_n` — the average over all steps
  since the last log point, smoothing out batch noise. Then reset the train-loss
  accumulators.
- **`save_every` (500 steps): checkpoint.** Write the full model to
  `output_dir/<stage>-step<N>` so a crash does not lose hours of work. See
  `implementation-notes.md` §11 — `output_dir` must be on the persistent
  filesystem on Lambda.

Then `step += 1`.

### 4.8 The epoch-boundary zero_grad (a real bug fix)

```python
126         opt.zero_grad(set_to_none=True)  # drop any partial accum group so its grads can't leak into the next epoch
```

This single line at the end of each epoch prevents a subtle bug. Recall
`steps_per_epoch` floors the division, so if the epoch's batch count is not a
multiple of `accum`, the **last few batches accumulated gradients but never
triggered the `(i+1) % accum == 0` boundary** — so they never got cleared by line
112. Without this line, those leftover gradients would still be sitting in
`.grad` when the next epoch's first batch calls `backward()`, contaminating the
next step with stale partial gradients from the previous epoch's tail. Zeroing
here drops that orphaned partial group cleanly.

### 4.9 The final aligned log point

```python
128     # final point (same step for train + val) so both curves end together
129     now = time.time()
130     tps = (step - 1 - last_step or 1) * tokens_per_step / max(1e-6, now - last_t)
131     val_loss = log_point(step - 1, stage["epochs"] - 1, tl_sum / max(1, tl_n) if tl_n else float("nan"),
132                          cosine_lr(step - 1, total_steps, stage["lr"], warmup), grad_norm, tps)
133     return val_loss
```

After all epochs, we log one final point at `step - 1` (the last completed step).
This guarantees the train and val curves *end together* at the same x-coordinate
— the mirror of the step-0 anchor that made them *start* together. The
`(… or 1)` and `max(1e-6, …)` guards avoid division by zero in tiny runs where
no steps fell between the last eval and the end. The function **returns the final
val loss**, which `run` records per stage.

> **Honesty note.** As of the current configuration the final val loss this loop
> reaches is *higher* than the paper's — the model is **under-trained** (it needs
> more epochs). The step-by-step workflow in §6 exists precisely to diagnose this:
> train one stage, look at whether its val curve is still descending at the last
> epoch, and bump `epochs` if so.

---

## 5. The orchestrator: `run`

```python
136 def run(cfg):
137     cfg.setdefault("seed", 123)  # single global seed: data split, shuffle, sampling all derive from it
138     torch.manual_seed(cfg["seed"])
139     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
140     model = ChronoGPT.from_pretrained(cfg["model_repo"]).to(device)
141     model.grad_checkpoint = cfg.get("grad_checkpoint", False)  # recompute blocks in backward to save VRAM
142     model.train()
```

`run(cfg)` is the entry point the CLI calls. It wires up seed, device, model, and
data, then loops over the curriculum.

**Seed (lines 137–138).** `setdefault` gives the single fallback point for the
global seed (`implementation-notes.md` §8). `torch.manual_seed` seeds PyTorch's
global RNG (used anywhere a local `Generator` is not passed).

**Loading the model (line 140) — this is what enables resume.**
`ChronoGPT.from_pretrained(cfg["model_repo"])` accepts **either a Hugging Face
repo id OR a local directory** (see `model.py` lines 204–217). So `model_repo`
can be `manelalab/chrono-gpt-v1-20201231` (fresh from the Hub) for a full run, or
a local path like `runs/.../stage1_scratch` to *resume* from a previously trained
stage. This is the mechanism behind the stage-by-stage workflow (§6). `.to(device)`
moves the 1.55B params onto the GPU.

**`grad_checkpoint` and `train()` (lines 141–142).** Set the model's gradient
checkpointing flag from config (default `True`; see §7 and `03-model.md`), and
flip the model into training mode (enables grad tracking and, combined with the
flag, the checkpointing path).

```python
144     # Packed data is filtered + built once and cached, then reused across vintages.
145     logger = RunLogger(cfg["output_dir"], cfg.get("wandb"), run_config=cfg)
146     with open(os.path.join(cfg["output_dir"], "config.yaml"), "w") as f:
147         yaml.safe_dump(cfg, f, sort_keys=False)  # snapshot the resolved config for reproducibility
148
149     packed = prepare_stages(cfg)
150     final_val = {}
151     for stage in cfg["stages"]:
152         train_ds, val_ds = packed[stage["name"]]
153         print(f"=== {stage['name']}: {len(train_ds)} train blocks, {len(val_ds)} val blocks ===")
154         final_val[stage["name"]] = round(train_stage(model, train_ds, val_ds, cfg, stage, device, logger), 4)
155         model.save_pretrained(os.path.join(cfg["output_dir"], stage["name"]))
156     model.save_pretrained(os.path.join(cfg["output_dir"], "final"))
```

**Logger + config snapshot (lines 145–147).** `RunLogger` opens
`output_dir/metrics.csv` in *append* mode (see `07-cli-tracking-hub-figures.md` / `tracking.py`),
so a resumed run accumulates onto the same curve. We also dump the *resolved*
config to `config.yaml` for reproducibility — anyone can see exactly what
hyperparameters produced this run.

**Data (line 149).** `prepare_stages(cfg)` returns `{stage_name: (train_ds,
val_ds)}`, built once and cached (the filtered, packed corpus is
model-independent — see `04-data.md` and `implementation-notes.md` §3).

**The curriculum loop (lines 151–156) — this is the curriculum.** We iterate the
stages *in config order*. Crucially, the *same* `model` object is passed to
`train_stage` each iteration — so Stage 2 starts from the weights Stage 1 left in
memory, and Stage 3 from Stage 2's. That continuity *is* the curriculum: scratch →
self-instruct → tulu-3, each building on the last. After each stage we record its
final val loss and save a checkpoint named after the stage. After all stages we
save a `final` checkpoint (a copy of the last stage's weights, the canonical
output).

```python
158     # Resume-aware: merge with a prior run's summary so a Stage 2-3 resume keeps
159     # Stage 1's final val (matches the appended metrics.csv).
160     prior = os.path.join(cfg["output_dir"], "summary.json")
161     if os.path.exists(prior):
162         try:
163             with open(prior) as f:
164                 final_val = {**json.load(f).get("final_val_loss", {}), **final_val}
165         except (ValueError, OSError):
166             pass
```

**Resume-aware summary merge (lines 160–166).** Suppose you ran Stage 1 in one
process (writing `summary.json` with `{stage1_scratch: …}`), then resumed Stages
2–3 in a second process. This second process only has `final_val` for stages 2
and 3. To keep the summary complete, we read the prior `summary.json` and merge:
`{**prior_stage1, **new_stages23}`. The dict-unpack order means new values win on
key collisions, but old keys not present in the new run (Stage 1) are preserved.
The result matches the *appended* `metrics.csv`, which already holds all three
stages. The `try/except` makes a corrupt or missing prior file a no-op rather
than a crash.

```python
168     logger.summary(
169         model_repo=cfg["model_repo"],
170         final_val_loss=final_val,
171         peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None,
172         seed=cfg["seed"], block_size=cfg["block_size"],
173         batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
174         grad_checkpoint=model.grad_checkpoint,
175     )
176     logger.close()
```

**Write the run summary (lines 168–176).** A `summary.json` with the per-stage
final val losses, peak VRAM, and the headline hyperparameters — the run-level
artifact you would show Manela. `logger.close()` flushes the CSV and finishes any
W&B run.

```python
178     push = cfg.get("push_to_hub")
179     if push and push.get("enabled"):
180         from .hub import push_dir
181         # Push the completed model to the canonical repo; a PARTIAL run (didn't end on
182         # final_stage) gets a "-<last_stage>" suffix so it can't clobber the final model.
183         last_stage = cfg["stages"][-1]["name"]
184         final_stage = push.get("final_stage") or last_stage
185         repo_id = push["repo_id"] if last_stage == final_stage else f"{push['repo_id']}-{last_stage}"
186         msg = f"stages={[s['name'] for s in cfg['stages']]} final_val={final_val} seed={cfg['seed']}"
187         push_dir(os.path.join(cfg["output_dir"], "final"), repo_id,
188                  private=push.get("private", True), commit_message=msg)
```

**Optional HF push with stage-suffix safety (lines 178–188).** Off by default. If
enabled, the lazy import (line 180) brings in `push_dir`. The key logic is the
**suffix guard** (lines 183–185): `final_stage` is the stage that *completes* the
curriculum (config: `stage3_tulu`). If this run's last stage equals `final_stage`,
it pushes to the canonical `repo_id`. Otherwise — a *partial* run, e.g. a
Stage-1-only diagnostic — it pushes to `repo_id-<last_stage>` (e.g.
`…-stage1_scratch`) so a half-trained model can never *clobber* the canonical
final model on the Hub. The commit message (line 186) records the stages, final
val losses, and seed for a self-describing Hub history. See `07-cli-tracking-hub-figures.md` /
`hub.py` for `push_dir`.

---

## 6. The curriculum / stage-by-stage workflow in practice

The design above supports two operating modes that share one `output_dir`:

1. **One shot.** List all three stages in `configs/train.yaml`; `run` trains them
   sequentially in one process. `metrics.csv` ends up with all three stages and a
   single Figure 1 shows the whole curriculum.

2. **Stage-by-stage diagnose-and-resume.** Use *two* configs pointing at the
   *same* `output_dir`:
   - `train_s1.yaml` lists only `stage1_scratch`, `model_repo` = the base vintage.
     Train it, plot Figure 1, and check whether the val curve is still descending
     at the last epoch (the under-training symptom from §4.9). If so, raise
     `epochs` and rerun (delete `metrics.csv` first to start the curve clean).
   - `train_s23.yaml` lists `stage2_self_instruct` + `stage3_tulu`, sets
     `model_repo` to the **local** `…/stage1_scratch` checkpoint, and keeps the
     **same** `output_dir`.

   Because `RunLogger` appends and `from_pretrained` reads local dirs, the
   resumed run continues the *same* `metrics.csv` and the *same* weights — so one
   combined Figure 1 still shows all three stages, and the resume-aware
   `summary.json` merge (§5) keeps Stage 1's val number.

For the exact commands and config snippets, see `docs/running-guide.md` §6b and
the configs/scripts walkthrough `08-configs-scripts-infra.md`.

---

## 7. The memory story

A finance reader's natural question: how does a 1.55B-parameter model fit on one
GPU at all? Memory splits into four buckets (from `implementation-notes.md` §7):

| Bucket | What it is | Size (1.55B params) |
|---|---|---|
| **Weights** | the parameters themselves, fp32 | ~6 GB (4 bytes/param) |
| **Gradients** | one `.grad` per parameter, fp32 | ~6 GB |
| **Adam states** | first moment `m` + second moment `v`, fp32 | ~12 GB |
| **Activations** | intermediate forward values kept for backprop | *dominant* |

The first three are the **"model states"** and are fixed at roughly
**16 bytes/param ≈ 25 GB** — the well-known Adam memory rule (4 for the weight + 4
for its grad + 8 for `m`+`v`). This is constant regardless of batch size.

The fourth — **activations** — is the surprise: it scales with `batch × seq ×
layers` and *dominates*. ChronoGPT is 52 layers over 1792 tokens with a 4×
squared-ReLU MLP, so the forward pass alone produces ~50 GB of activations at
`batch_size 8` — enough to OOM an 80 GB card *in the forward pass*, before any
backward. (A 40 GB card OOMs at batch 1 regardless — full FT here genuinely needs
≥ 80 GB.)

Two tricks in this code make `batch_size 8` fit one 80 GB card:
- **Gradient checkpointing** (`grad_checkpoint: true`, set on the model at line
  141). Instead of *storing* every block's activations for the backward pass, the
  model *recomputes* each block during backward — trading ~20% extra compute for
  ~10× less activation memory. The mechanism lives in `model.py`
  (`torch.utils.checkpoint`, active only when `grad_checkpoint and self.training`);
  see `03-model.md` for the details.
- **`return_hidden=False`** on every training/eval forward call (lines 51, 92,
  103). This tells the model not to retain the per-layer hidden-state list it
  would otherwise build — pure savings during training, where we only need the
  final logits.

Gradient accumulation (§4.2) is the third lever: it lets us keep the *effective*
batch at 32 for stable gradients while only ever holding 8 sequences' activations
at once. If you still OOM: lower `batch_size`, raise `grad_accum` to keep the
product constant.

---

## 8. Mini-FAQ of subtle points

**Why scale the loss by `1/accum` before `backward()`?**
Because gradients accumulate additively across the `accum` micro-batches, and the
gradient of a sum is the sum of gradients. Dividing each loss by `accum` makes the
accumulated gradient equal the gradient of the *mean* over the full effective
batch (size 32), not `accum`× too large. Without it, your effective LR would
silently be `accum`× higher than intended.

**Why `steps_per_epoch = len(loader) // accum` (floor, not ceil)?**
A weight update only happens when a *complete* group of `accum` batches has been
accumulated. A trailing partial group never triggers `opt.step()`, so it is not a
step. Flooring makes `total_steps` equal the real number of updates, which keeps
the cosine LR schedule correctly anchored to actually anneal to 0 at the end.

**Why `opt.zero_grad()` at the end of each epoch (line 126)?**
If the epoch's batch count is not a multiple of `accum`, the tail batches
accumulated gradients without ever hitting the step boundary that clears them.
That orphaned partial gradient would leak into the next epoch's first update. The
epoch-end zero drops it. (This was a real bug the line fixes.)

**Why token-weighted eval instead of averaging batch means?**
Packed batches contain different numbers of *scored* (response) tokens. A simple
average of per-batch mean losses over-weights tokens in sparse batches, biasing
the val number. Summing all per-token losses and dividing by the total token
count gives the true grand-mean NLL — the same reason a pooled mean beats an
average-of-group-means when group sizes differ.

**Why a step-0 val/anchor point?**
So both curves have a real starting value — the loss of the base (or
previous-stage) weights *before* this stage changes anything. It lets you read
the curriculum's effect directly off Figure 1 and makes the train/val curves
start at the same x-coordinate. The final log point (line 131) is its bookend,
making them end together too.

**Why AdamW and not the modded-nanoGPT Muon optimizer?**
ChronoGPT's *architecture* is borrowed from modded-nanoGPT, whose speed records
use the Muon optimizer for pretraining. But this is **fine-tuning**, not
pretraining from scratch, and the paper specifies standard masked cross-entropy
SFT. AdamW is the universal, well-understood SFT optimizer; Muon would add
unjustified complexity and risk for a short fine-tune. We deliberately keep the
optimizer boring (`implementation-notes.md` §7).

**Why bf16 autocast but no GradScaler / loss scaling?**
bfloat16 keeps fp32's exponent range (only the mantissa is shorter), so gradients
do not underflow the way they do in fp16. That removes the need for the dynamic
loss-scaling (`GradScaler`) that fp16 training requires. The master weights and
Adam moments still live in fp32 — autocast only changes the dtype of the matmuls
inside the forward pass.

**What does "resume" actually load?**
Only the model *weights* (`from_pretrained` reads `pytorch_model.bin` + config
from a local dir, line 140 / `model.py` 204–217). It does **not** restore
optimizer state, the LR schedule position, or the RNG. So a resumed Stage 2–3 run
starts a *fresh* AdamW with a *fresh* warmup-then-cosine cycle from the Stage-1
weights. For this curriculum that is intentional — each stage is meant to be its
own short training run beginning from the prior stage's parameters.
