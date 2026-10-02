"""LoRA on a tiny ChronoGPT, both backends: starts at the base model, trains only the adapters,
adapters_disabled recovers the base, and the merged checkpoint reloads as a plain model."""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from chrono_instruct.lora import (DEFAULT_TARGETS, adapters_disabled, apply_lora, lora_layers,
                                  merged_state_dict, save_merged, trainable_parameters)
from chrono_instruct.model import ChronoGPT, build_tiny

BACKENDS = ["native", pytest.param("peft", marks=pytest.mark.skipif(
    importlib.util.find_spec("peft") is None, reason="peft not installed"))]
V, L, D = 512, 4, 32


def tiny():
    torch.manual_seed(0)
    model = build_tiny(vocab_size=V, num_layers=L, num_heads=2, model_dim=D)
    for p in model.parameters():          # build_tiny zero-inits lm_head and c_proj, which would make logits 0
        p.data.normal_(0, 0.05)
    return model


def logits(model, ids):
    with torch.no_grad():
        return model(ids, return_hidden=False)[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_lora_round_trip(tmp_path, backend):
    model, ids = tiny(), torch.randint(0, V, (2, 8))
    base = logits(model, ids)
    assert apply_lora(model, rank=4, alpha=8, backend=backend) is model
    assert len(lora_layers(model)) == L * 6                       # c_q c_k c_v attn.c_proj c_fc mlp.c_proj
    params = trainable_parameters(model)
    per_block = 4 * (4 * D + D * 4) + (4 * D + 4 * D * 4) + (4 * 4 * D + D * 4)
    assert sum(p.numel() for p in params) == L * per_block
    assert all(p.dtype == torch.float32 for p in params)
    assert model.embed.weight.dtype == torch.bfloat16 and model.lm_head.weight.dtype == torch.bfloat16
    assert model.blocks[0].lambdas.dtype == torch.float32
    assert torch.allclose(logits(model, ids), base, atol=1e-5)    # B = 0: the base model

    frozen = {k: v.clone() for k, v in model.state_dict().items() if "lora" not in k}
    opt = torch.optim.AdamW(params, lr=1e-2)
    model.train()
    for _ in range(3):
        out, _ = model(ids, return_hidden=False)
        F.cross_entropy(out[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1)).backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items())
    adapted = logits(model, ids)
    assert not torch.allclose(adapted, base, atol=1e-3)
    with adapters_disabled(model):
        assert torch.allclose(logits(model, ids), base, atol=1e-5)
    assert all(p.requires_grad for p in params) and torch.allclose(logits(model, ids), adapted)

    sd = merged_state_dict(model)
    assert set(sd) == set(tiny().state_dict()) and all(p.requires_grad for p in params)
    save_merged(model, tmp_path / "final")
    plain = ChronoGPT.from_pretrained(str(tmp_path / "final"))
    assert not lora_layers(plain) and all(p.dtype == torch.float32 for p in plain.parameters())
    merged = logits(plain, ids)
    assert torch.allclose(merged, adapted, atol=0.1)              # bf16 rounding of W + BA vs the two-term forward
    assert (merged.argmax(-1) == adapted.argmax(-1)).float().mean() > 0.9


def test_apply_lora_validates():
    model = tiny()
    with pytest.raises(ValueError):
        apply_lora(model, rank=2, targets=("nope",))
    with pytest.raises(ValueError):
        apply_lora(model, rank=2, backend="other")
    apply_lora(model, rank=2, targets=DEFAULT_TARGETS + ("lm_head",))
    assert len(lora_layers(model)) == L * 6 + 1
    with pytest.raises(ValueError):
        apply_lora(model, rank=2)
