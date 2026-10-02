"""LoRA for ChronoGPT: a native wrapper or Hugging Face peft behind one small API.

Two backends, same contract. `native` wraps each target projection in `LoRALinear` (~30 lines,
the dtype convention of CastedLinear: adapter matrices are cast to the activation dtype at use,
fp32 master copies for AdamW, the model stays a plain nn.Module). `peft` injects peft's own
`lora.Linear` layers in place (`inject_adapter_in_model`; no PeftModel wrapper, so forward
signatures, attribute access, DDP and checkpointing are untouched), for comparing against the
library implementation. Either way `apply_lora` returns the SAME model object with wrapped
projections, `trainable_parameters` is the adapter set, `adapters_disabled` gives the base
model's forward, and `save_merged` writes a checkpoint that `ChronoGPT.from_pretrained` loads
as an ordinary model.

Decisions:
  - `apply_lora` freezes the whole model and casts the frozen matrices to bf16. ChronoGPT casts
    every linear and embedding weight to bf16 inside its forward, so this is lossless and
    halves the resident weights; the adapters are fp32 and the only trainable parameters.
  - Targets are module-name suffixes. The default is every projection of every block
    (`c_proj` matches both the attention and the MLP one); `lm_head` can be added.
  - Init as in the LoRA paper and peft: A Kaiming-uniform, B zero, so the adapted model starts
    identical to the base; scale = alpha / rank.
  - `adapters_disabled` switches the adapters off in place for the duration of a forward.
    DistIL's fixed teacher (the paper's --fixed_teacher: LoRA disabled) is exactly that, so a
    self-distillation run needs no second copy of the model.
  - `merged_state_dict` folds W + scale * B A in fp32 on the CPU and stores it in the base weight's
    dtype, without touching the model or the GPU (a 16 GB card is full when a run ends), so
    checkpoints can be written mid-run. A bf16 base weight makes the merged model agree with the
    adapted one only to bf16 rounding (test_lora bounds it).
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("c_q", "c_k", "c_v", "c_proj", "c_fc")
BACKENDS = ("native", "peft")


class LoRALinear(nn.Module):
    """`base` (a frozen nn.Linear) plus scale * B A, matrices cast to the input dtype like CastedLinear."""

    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        h = F.linear(self.dropout(x), self.lora_A.type_as(x))
        return y + F.linear(h, self.lora_B.type_as(x)) * self.scale

    def merged_weight(self):
        """W + scale * B A, computed on the CPU in fp32, returned in W's dtype (on the CPU)."""
        return _merge(self.base.weight, self.lora_A, self.lora_B, self.scale)


def _peft_layer_cls():
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        return ()
    return LoraLayer


def is_lora_layer(module):
    return isinstance(module, (LoRALinear, _peft_layer_cls()))


def lora_layers(model):
    """[(name, module)] of every adapted projection, native or peft."""
    return [(n, m) for n, m in model.named_modules() if is_lora_layer(m)]


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def apply_lora(model, rank, alpha=None, dropout=0.0, targets=DEFAULT_TARGETS, backend="native"):
    """Freeze `model` (frozen matrices to bf16) and add fp32 LoRA adapters to the target linears, in place.

    Returns the same object. `targets` are module-name suffixes; `backend` is "native" or "peft"
    (the `peft` pip extra). Raises if nothing matches or the model already carries adapters.
    """
    alpha = rank if alpha is None else alpha
    if backend not in BACKENDS:
        raise ValueError(f"lora backend must be one of {BACKENDS}, got {backend!r}")
    if lora_layers(model):
        raise ValueError("the model already has LoRA adapters")
    found = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in targets]
    if not found:
        raise ValueError(f"no nn.Linear named {tuple(targets)} in the model")
    model.requires_grad_(False)
    if backend == "native":
        for name, m in found:
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, LoRALinear(m, rank, alpha, dropout))
    else:
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets), bias="none")
        inject_adapter_in_model(cfg, model)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()          # peft creates the adapters in the base layer's dtype
        elif p.ndim >= 2:
            p.data = p.data.to(torch.bfloat16)
    return model


def _set_enabled(module, on):
    if isinstance(module, LoRALinear):
        module.enabled = on
    else:
        module.enable_adapters(on)


@contextlib.contextmanager
def adapters_disabled(model):
    """Forwards inside the block see the base model; adapters (and their requires_grad) are restored after."""
    layers = [m for _, m in lora_layers(model)]
    for m in layers:
        _set_enabled(m, False)
    try:
        yield
    finally:
        for m in layers:
            _set_enabled(m, True)


def _merge(w, a, b, scale):
    cpu = lambda t: t.detach().cpu().float()   # noqa: E731
    return (cpu(w) + (cpu(b) @ cpu(a)) * scale).to(w.dtype)


def _merged_weight(module):
    if isinstance(module, LoRALinear):
        return module.merged_weight()
    w = module.base_layer.weight                      # peft lora.Linear: sum the active adapters, as get_delta_weight does
    out = w.detach().cpu().float()
    for name in module.active_adapters:
        out += (module.lora_B[name].weight.detach().cpu().float() @ module.lora_A[name].weight.detach().cpu().float()) \
            * module.scaling[name]
    return out.to(w.dtype)


def merged_state_dict(model):
    """The base model's state dict with the adapters folded into their weights, on the CPU; the model is left as is.

    Everything is computed on the CPU: a run on a 16 GB card finishes with the GPU full, and the
    merge is written once, so it needs no device memory at all."""
    layers = dict(lora_layers(model))
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if not any(k.startswith(n + ".") for n in layers)}
    for name, m in layers.items():
        sd[name + ".weight"] = _merged_weight(m)
    return sd


def save_merged(model, save_directory):
    """`save_pretrained` of the merged model: loads back with ChronoGPT.from_pretrained, no adapters."""
    model.save_pretrained(save_directory, state_dict=merged_state_dict(model))
