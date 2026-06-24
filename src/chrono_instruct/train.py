"""Curriculum SFT training loop.

One run = one vintage = one process on one GPU. Stages are trained in order
(scratch -> self-instruct -> tulu-3), each continuing from the previous stage's
weights. Loss is masked cross-entropy on response tokens only. Multi-GPU /
cluster fan-out is handled outside this file (see scripts/), never here.
"""
import math
import os
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .model import ChronoGPT
from .data import prepare_stages
from .tracking import RunLogger


def masked_lm_loss(logits, labels, reduction="mean"):
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction=reduction)


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Token-weighted mean response loss.

    Sums per-token cross-entropy over all response tokens and divides by the token
    count — NOT a mean of per-batch means, which is biased when batches hold
    different numbers of response tokens. Runs under the same bf16 autocast as
    training. `max_batches` caps cost for cheap periodic (in-stage) validation;
    leave None for the full held-out set at epoch end.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for n, (ids, labels) in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        ids, labels = ids.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(ids, return_hidden=False)
        total_loss += masked_lm_loss(logits, labels, reduction="sum").item()
        total_tokens += int((labels[:, 1:] != -100).sum())
    model.train()
    return total_loss / max(1, total_tokens)


def train_stage(model, train_ds, val_ds, cfg, stage, device, run_logger=None):
    g = torch.Generator().manual_seed(cfg["seed"])  # global seed -> deterministic shuffle
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    name = stage["name"]
    accum = cfg.get("grad_accum", 1)
    log_every = cfg.get("log_every", 20)
    grad_clip = cfg.get("grad_clip")          # None -> no clipping (norm still logged)
    tokens_per_step = cfg["batch_size"] * cfg["block_size"] * accum
    steps_per_epoch = len(loader) // accum    # only full accum groups step; floor matches reality
    total_steps = steps_per_epoch * stage["epochs"]
    warmup = int(total_steps * cfg.get("warmup_ratio", 0.03))

    step = 0
    last_t, last_step = time.time(), 0
    val_loss = float("nan")
    for epoch in range(stage["epochs"]):
        for i, (ids, labels) in enumerate(loader):
            ids, labels = ids.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(ids, return_hidden=False)
                loss = masked_lm_loss(logits, labels) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or 1e30)
                lr = cosine_lr(step, total_steps, stage["lr"], warmup)
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % log_every == 0:
                    now = time.time()
                    tps = (step - last_step or 1) * tokens_per_step / (now - last_t)
                    last_t, last_step = now, step
                    train_loss = loss.item() * accum
                    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    print(f"[{name}] step {step}/{total_steps} loss {train_loss:.4f} "
                          f"lr {lr:.2e} |g| {float(grad_norm):.2f} {tps:,.0f} tok/s {mem:.1f}GB")
                    if run_logger:
                        run_logger.log(stage=name, epoch=epoch, step=step, split="train",
                                       loss=round(train_loss, 4), lr=lr, grad_norm=round(float(grad_norm), 3),
                                       tokens_per_sec=round(tps), gpu_mem_gb=round(mem, 1))
                if cfg.get("save_every") and step > 0 and step % cfg["save_every"] == 0:
                    model.save_pretrained(os.path.join(cfg["output_dir"], f"{name}-step{step}"))
                if cfg.get("eval_every") and step > 0 and step % cfg["eval_every"] == 0:
                    vloss = evaluate(model, val_loader, device, max_batches=cfg.get("eval_max_batches", 50))
                    print(f"[{name}] step {step}/{total_steps} val_loss {vloss:.4f}")
                    if run_logger:
                        run_logger.log(stage=name, epoch=epoch, step=step, split="val",
                                       loss=round(vloss, 4), ppl=round(math.exp(min(vloss, 20)), 2))
                step += 1
        opt.zero_grad(set_to_none=True)  # drop any partial accum group so its grads can't leak into the next epoch
        val_loss = evaluate(model, val_loader, device)  # full held-out set at epoch end
        print(f"[{name}] epoch {epoch} val_loss {val_loss:.4f}")
        if run_logger:
            run_logger.log(stage=name, epoch=epoch, step=step, split="val",
                           loss=round(val_loss, 4), ppl=round(math.exp(min(val_loss, 20)), 2))
    return val_loss


def run(cfg):
    cfg.setdefault("seed", 123)  # single global seed: data split, shuffle, sampling all derive from it
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChronoGPT.from_pretrained(cfg["model_repo"]).to(device)
    model.grad_checkpoint = cfg.get("grad_checkpoint", False)  # recompute blocks in backward to save VRAM
    model.train()

    # Packed data is filtered + built once and cached, then reused across vintages.
    logger = RunLogger(cfg["output_dir"], cfg.get("wandb"), run_config=cfg)
    with open(os.path.join(cfg["output_dir"], "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)  # snapshot the resolved config for reproducibility

    packed = prepare_stages(cfg)
    final_val = {}
    for stage in cfg["stages"]:
        train_ds, val_ds = packed[stage["name"]]
        print(f"=== {stage['name']}: {len(train_ds)} train blocks, {len(val_ds)} val blocks ===")
        final_val[stage["name"]] = round(train_stage(model, train_ds, val_ds, cfg, stage, device, logger), 4)
        model.save_pretrained(os.path.join(cfg["output_dir"], stage["name"]))
    model.save_pretrained(os.path.join(cfg["output_dir"], "final"))

    logger.summary(
        model_repo=cfg["model_repo"],
        final_val_loss=final_val,
        peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None,
        seed=cfg["seed"], block_size=cfg["block_size"],
        batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
        grad_checkpoint=model.grad_checkpoint,
    )
    logger.close()

    push = cfg.get("push_to_hub")
    if push and push.get("enabled"):
        from .hub import push_dir
        push_dir(os.path.join(cfg["output_dir"], "final"), push["repo_id"],
                 private=push.get("private", True))
