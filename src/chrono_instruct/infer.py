"""Inference: unified generation + embedding extraction for any vintage.

The model's forward returns (logits, layer_outputs), so both modalities share
one load path. Generation uses an optional KV cache (use_cache=True, default) to
decode one token at a time; pass use_cache=False to fall back to full-sequence
recompute (the original model card's demo behavior).
"""
import torch
import torch.nn.functional as F
import tiktoken

from .model import ChronoGPT

ENC = tiktoken.get_encoding("gpt2")


def load(repo_id, device=None, cache_dir="cache"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChronoGPT.from_pretrained(repo_id, cache_dir=cache_dir).to(device)
    model.eval()
    return model, device


def free_memory():
    """Release cached GPU memory. Call AFTER `del`-ing your model/tensor refs,
    e.g. `del model; free_memory()` — useful between loading different vintages."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_repetition_penalty(logits, ids, penalty):
    """CTRL-style penalty (Keskar et al. 2019), in place on the last-step logits [B, V]:
    divide the logit of every already-generated token by `penalty` (>1), so repeating
    is discouraged. Symmetric on sign (negative logits are multiplied) so the ranking
    stays sensible."""
    for b in range(ids.size(0)):
        toks = torch.unique(ids[b])
        sel = logits[b, toks]
        logits[b, toks] = torch.where(sel > 0, sel / penalty, sel * penalty)


def _ban_repeat_ngrams(logits, ids, n):
    """Set to -inf any next token that would repeat an n-gram already in the sequence
    (HuggingFace's no_repeat_ngram_size). Kills exact loops that a penalty alone lets
    through. In place on [B, V]."""
    if ids.size(1) < n:
        return
    for b in range(ids.size(0)):
        seq = ids[b].tolist()
        seen = {}
        for i in range(len(seq) - n + 1):
            seen.setdefault(tuple(seq[i:i + n - 1]), set()).add(seq[i + n - 1])
        banned = seen.get(tuple(seq[-(n - 1):]))
        if banned:
            logits[b, list(banned)] = float("-inf")


def _next_token(logits, top_k, temperature, rng):
    """Pick the next id from the last step's logits[B, V].

    Greedy (argmax) when temperature == 0 OR top_k == 1 — and we branch BEFORE
    dividing by temperature, so temperature=0 is safe (no division by zero), unlike
    a naive logits/temperature. Otherwise: temperature scaling + optional top-k
    sampling.
    """
    if temperature == 0.0 or top_k == 1:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=rng)


@torch.no_grad()
def generate(model, device, prompt, max_new_tokens=128, top_k=None, temperature=0.0, seed=123,
             return_completion=False, use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0):
    """Generate a continuation of `prompt`.

    Decoding defaults match manelalab's `ChronoGPT_instruct.py`: temperature=0.0 is
    GREEDY (argmax), top_k=None. Set temperature>0 to sample; top_k (e.g. 50)
    restricts sampling to the k most likely tokens. To force greedy, use
    temperature=0 (safe here) or top_k=1 — NOT a tiny temperature.

    Anti-repetition (both OFF by default, so paper-matching greedy is unchanged):
    `repetition_penalty` > 1.0 (e.g. 1.3) discourages re-using tokens; a
    `no_repeat_ngram_size` of 3 forbids repeating any 3-gram. Greedy decoding on a
    ~1.5B model loops without these; turn them on for readable output, leave them off
    to reproduce the paper's exact decoding.

    use_cache=True threads a KV cache through the model so each step feeds only the
    newly generated token instead of recomputing the whole sequence — O(T) work per
    token instead of O(T^2). It is numerically equivalent to use_cache=False.

    Returns the full decoded text (prompt + completion) by default; with
    return_completion=True it decodes only the newly generated tokens — sliced by
    TOKEN count, not by prompt string length, so extraction is exact regardless of
    tokenizer round-trip whitespace quirks.
    """
    ids = torch.tensor(ENC.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    n_prompt = ids.shape[1]
    rng = torch.Generator(device=device).manual_seed(seed)
    past = [None] * len(model.blocks) if use_cache else None
    step_in = ids  # full prompt on the first pass; then just the new token when caching
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if use_cache:
                logits, past = model(step_in, return_hidden=False, past=past)
            else:
                logits, _ = model(ids, return_hidden=False)
        step_logits = logits[:, -1, :].float().clone()  # penalties/bans need a mutable fp32 copy
        if repetition_penalty != 1.0:
            _apply_repetition_penalty(step_logits, ids, repetition_penalty)
        if no_repeat_ngram_size:
            _ban_repeat_ngrams(step_logits, ids, no_repeat_ngram_size)
        nxt = _next_token(step_logits, top_k, temperature, rng)
        if nxt.item() == ENC.eot_token:
            break
        ids = torch.cat([ids, nxt], dim=1)
        step_in = nxt
    out_ids = ids[0, n_prompt:] if return_completion else ids[0]
    return ENC.decode(out_ids.tolist())


@torch.no_grad()
def embed(model, device, text, layer=-1, max_length=1792, pool="mean"):
    """Return a hidden-state embedding for `text` from the given layer."""
    token_ids = ENC.encode(text)[:max_length]
    if not token_ids:
        raise ValueError("embed() received text that tokenized to zero tokens")
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, layer_outputs = model(ids)
    h = layer_outputs[layer][0].float()  # (T, model_dim)
    return h.mean(0) if pool == "mean" else h[-1]
