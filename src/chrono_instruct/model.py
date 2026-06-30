"""ChronoGPT model — training-enabled.

Vendored and lightly adapted from `ChronoGPT_inference.py` in
manelalab/chrono-gpt-v1-* (MIT License; He, Lv, Manela, Wu 2025).

Changes from the original:
  1. Removed every `@torch.inference_mode()` decorator so the forward pass
     builds an autograd graph (the released code is inference-only and cannot
     train as published).
  2. Reworked the KV cache as an OPTIONAL `past` argument to `forward` (off by
     default, so the training path is untouched). `infer.generate` uses it to
     decode one token at a time instead of recomputing the whole sequence; with
     past=None the forward is identical to the original full-sequence pass.

Architecture (modded-nanoGPT U-net: 26 encoder + 26 decoder layers with skip
connections, value embeddings, RMSNorm, rotary, ReLU^2 MLP, logit softcap) is
otherwise unchanged. Source: https://huggingface.co/manelalab/chrono-gpt-v1-20201231
"""
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x, offset=0):
        # offset = number of already-cached positions, so the new tokens get their
        # ABSOLUTE rotary angles during incremental (KV-cached) decoding.
        L = x.size(-3)
        cos, sin = self.cos[None, offset : offset + L, None, :], self.sin[None, offset : offset + L, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


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

    def forward(self, x, ve, past=None, use_cache=False):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k)
        past_len = past[0].size(1) if past is not None else 0   # cached positions along T
        q, k = self.rotary(q, past_len), self.rotary(k, past_len)
        if past is not None:                                    # prepend cached keys/values
            k = torch.cat([past[0], k], dim=1)
            v = torch.cat([past[1], v], dim=1)
        new_cache = (k, v) if use_cache else None
        # Prefill (no past) needs the causal mask; a single-token decode step has one
        # query that should attend to ALL cached positions, so no mask is needed there.
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=(past_len == 0)
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y), new_cache


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


class Block(nn.Module):
    def __init__(self, model_dim, num_heads, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads) if use_attn else None
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, ve, x0, past=None, use_cache=False):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        new_cache = None
        if self.attn is not None:
            a, new_cache = self.attn(norm(x), ve, past, use_cache)
            x = x + a
        x = x + self.mlp(norm(x))
        # Plain tensor on the training path (so torch.utils.checkpoint stays simple);
        # (x, cache) only when caching for inference.
        return (x, new_cache) if use_cache else x


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

    def forward(self, inputs, return_hidden=True, past=None):
        """Returns (logits[B,T,V] float, layer_outputs).

        layer_outputs is the per-layer hidden-state list used by `embed`. Pass
        return_hidden=False during training to skip retaining it. When
        self.grad_checkpoint is set (training only), each block is recomputed in
        the backward pass to cut activation memory ~10x.

        KV cache (inference only): pass `past`, a list with one slot per block
        ([None] * num_layers on the first call), to reuse cached keys/values
        instead of recomputing the whole prefix each step (O(T) vs O(T^2)). When
        `past` is given the SECOND return value is the updated cache list, not
        layer_outputs. Never combine with training / grad_checkpoint. See
        infer.generate. This is numerically equivalent to past=None.
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        B = inputs.size(0)
        x0 = norm(self.embed(inputs).bfloat16())
        x = x0

        ve = [self.value_embeds(inputs[i].view(-1)) for i in range(B)]
        ve = [
            torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None
            for i in range(len(ve[0]))
        ]
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]

        use_cache = past is not None
        new_past = [] if use_cache else None
        ckpt = self.grad_checkpoint and self.training

        def run_block(blk, *args):
            return checkpoint(blk, *args, use_reentrant=False) if ckpt else blk(*args)

        layer_outputs = []
        skip_connections = []
        for i in range(self.num_encoder_layers):
            if use_cache:
                x, kv = self.blocks[i](x, ve_enc[i], x0, past[i], True)
                new_past.append(kv)
            else:
                x = run_block(self.blocks[i], x, ve_enc[i], x0)
            skip_connections.append(x)
            if return_hidden:
                layer_outputs.append(norm(x))
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            j = self.num_encoder_layers + i
            if use_cache:
                x, kv = self.blocks[j](x, ve_dec[i], x0, past[j], True)
                new_past.append(kv)
            else:
                x = run_block(self.blocks[j], x, ve_dec[i], x0)
            if return_hidden:
                layer_outputs.append(norm(x))

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)  # logit softcap
        return logits.float(), (new_past if use_cache else layer_outputs)

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

    @classmethod
    def from_pretrained(cls, repo_id, cache_dir=None, **kwargs):
        # Accept a local checkpoint dir (e.g. runs/.../stage1_scratch for resume, or
        # for `chrono infer --repo <dir>`) as well as a Hugging Face repo id.
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


def build_tiny(vocab_size=512, num_layers=8, num_heads=4, model_dim=64):
    """Small randomly-initialized model for the CPU smoke test (no download)."""
    return ChronoGPT(vocab_size=vocab_size, num_layers=num_layers, num_heads=num_heads, model_dim=model_dim)
