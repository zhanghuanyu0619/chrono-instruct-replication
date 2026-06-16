"""Curriculum SFT training loop.

One run = one vintage = one process on one GPU. Stages are trained in order
(scratch -> self-instruct -> tulu-3), each continuing from the previous stage's
weights. Loss is masked cross-entropy on response tokens only. Multi-GPU /
cluster fan-out is handled outside this file (see scripts/), never here.
"""
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ChronoGPT
from .data import load_raw, load_stage


def masked_lm_loss(logits, labels):
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(ids)
        total += masked_lm_loss(logits, labels).item()
        n += 1
    model.train()
    return total / max(1, n)


def train_stage(model, train_ds, val_ds, cfg, stage, device, logger=print):
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    accum = cfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * stage["epochs"]
    warmup = int(total_steps * cfg.get("warmup_ratio", 0.03))

    step = 0
    for epoch in range(stage["epochs"]):
        for i, (ids, labels) in enumerate(loader):
            ids, labels = ids.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(ids)
                loss = masked_lm_loss(logits, labels) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total_steps, stage["lr"], warmup)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % cfg.get("log_every", 20) == 0:
                    logger(f"[{stage['name']}] step {step}/{total_steps} loss {loss.item() * accum:.4f}")
                if cfg.get("save_every") and step > 0 and step % cfg["save_every"] == 0:
                    model.save_pretrained(os.path.join(cfg["output_dir"], f"{stage['name']}-step{step}"))
                step += 1
        logger(f"[{stage['name']}] epoch {epoch} val_loss {evaluate(model, val_loader, device):.4f}")


def run(cfg):
    torch.manual_seed(cfg.get("seed", 123))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChronoGPT.from_pretrained(cfg["model_repo"]).to(device)
    model.train()

    raw = load_raw(cfg["dataset"])
    for stage in cfg["stages"]:
        train_ds, val_ds = load_stage(
            raw, stage["sources"], cfg["block_size"],
            cfg.get("val_fraction", 0.05), cfg.get("seed", 123),
        )
        print(f"=== {stage['name']}: {len(train_ds)} train blocks, {len(val_ds)} val blocks ===")
        train_stage(model, train_ds, val_ds, cfg, stage, device)
        model.save_pretrained(os.path.join(cfg["output_dir"], stage["name"]))
    model.save_pretrained(os.path.join(cfg["output_dir"], "final"))
