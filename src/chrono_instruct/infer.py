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


@torch.no_grad()
def generate(model, device, prompt, max_new_tokens=128, top_k=50, temperature=1.0, seed=123):
    ids = torch.tensor(ENC.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
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
    return ENC.decode(ids[0].tolist())


@torch.no_grad()
def embed(model, device, text, layer=-1, max_length=1792, pool="mean"):
    """Return a hidden-state embedding for `text` from the given layer."""
    ids = torch.tensor(ENC.encode(text)[:max_length], dtype=torch.long, device=device).unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, layer_outputs = model(ids)
    h = layer_outputs[layer][0].float()  # (T, model_dim)
    return h.mean(0) if pool == "mean" else h[-1]
