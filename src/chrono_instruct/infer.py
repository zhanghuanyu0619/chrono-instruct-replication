"""Inference: unified generation + embedding extraction for any vintage.

The model's forward returns (logits, layer_outputs), so both modalities share
one load path. Generation recomputes the full sequence each step (no KV cache),
matching the original model card's demo.
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


@torch.no_grad()
def generate(model, device, prompt, max_new_tokens=128, top_k=50, temperature=1.0, seed=123,
             return_completion=False):
    """Sample a continuation of `prompt`. top_k=1 makes it greedy/deterministic.

    Returns the full decoded text (prompt + completion) by default; with
    return_completion=True it decodes only the newly generated tokens — sliced by
    TOKEN count, not by prompt string length, so extraction is exact regardless of
    tokenizer round-trip whitespace quirks.
    """
    ids = torch.tensor(ENC.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    n_prompt = ids.shape[1]
    rng = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(ids)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = torch.topk(probs, top_k, dim=-1)
        nxt = torch.gather(topk_i, -1, torch.multinomial(topk_p, 1, generator=rng))
        if nxt.item() == ENC.eot_token:
            break
        ids = torch.cat([ids, nxt], dim=1)
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
