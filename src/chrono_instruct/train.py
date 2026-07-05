"""Curriculum SFT training loop.

One run = one vintage = one process on one GPU. Stages are trained in order
(scratch -> self-instruct -> tulu-3), each continuing from the previous stage's
weights. Loss is masked cross-entropy on response tokens only. Multi-GPU /
cluster fan-out is handled outside this file (see scripts/), never here.
"""
import json
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


def cosine_lr(step, total, base_lr, warmup, min_lr=0.0):
    """Linear warmup then cosine decay from base_lr down to min_lr (a floor).

    min_lr=0 reproduces the classic decay-to-zero. A small floor (e.g. 10% of
    base_lr) keeps the final steps productive instead of wasting them at ~0 lr,
    which matters most on short stages where the zero-tail is a big fraction of
    the budget.
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device):
    """Token-weighted mean response loss over the WHOLE val set.

    Sums per-token cross-entropy over all response tokens and divides by the token
    count — NOT a mean of per-batch means, which is biased when batches hold
    different numbers of response tokens. Runs under the same bf16 autocast as
    training. The val set is bounded by `val_max_blocks` (in data.py) so a full
    pass stays cheap enough to call periodically.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for ids, labels in loader:
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
    # Per-stage override, falling back to the global config. Small stages have
    # very few optimizer steps (stage1_scratch is ~1 step/epoch: 1097 short
    # examples pack into ~34 blocks, and an effective batch of batch_size*accum
    # blocks drains that in one step), so they need eval_every/log_every ~1 to
    # produce a usable curve; large stages (stage3 ~14k steps) want coarser
    # logging or the CSV and eval overhead explode.
    log_every = stage.get("log_every", cfg.get("log_every", 20))
    eval_every = stage.get("eval_every", cfg.get("eval_every"))
    grad_clip = cfg.get("grad_clip")          # None -> no clipping (norm still logged)
    tokens_per_step = cfg["batch_size"] * cfg["block_size"] * accum
    steps_per_epoch = len(loader) // accum    # only full accum groups step; floor matches reality
    total_steps = steps_per_epoch * stage["epochs"]
    warmup = int(total_steps * cfg.get("warmup_ratio", 0.03))
    min_lr = stage["lr"] * cfg.get("min_lr_ratio", 0.0)  # cosine floor; 0.0 -> decay to zero (old behavior)

    # Early stopping (global `early_stop_patience`; null/0 -> disabled). When on,
    # we snapshot the best-val weights and, on stopping OR finishing, restore them
    # so this stage's saved checkpoint — and the next stage's starting point — is
    # the best model, never the (possibly overfit) last step. Requires a
    # meaningful val signal, i.e. eval_every small enough to eval several times.
    patience = cfg.get("early_stop_patience") or 0
    min_delta = cfg.get("min_delta") or 0.0  # val must improve by > min_delta to count (Keras semantics)
    best = {"val": float("inf"), "step": 0, "state": None, "stale": 0}

    def consider(vloss, step):
        """Track best-val weights; return True when patience is exhausted (stop).

        An eval counts as an improvement (resets patience and updates the saved
        best) only if it beats the running best by more than `min_delta`, so
        trivial noise on a tiny val set doesn't keep patience alive forever.
        """
        if not patience:
            return False
        if vloss < best["val"] - min_delta:
            best.update(val=vloss, step=step, stale=0,
                        state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        else:
            best["stale"] += 1
        return best["stale"] >= patience

    def mem_gb():
        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    def log_point(step, epoch, train_loss, lr, grad_norm, tps):
        """Log train AND val at the SAME step (aligned curves, like the paper's Fig 1)."""
        vloss = evaluate(model, val_loader, device)
        print(f"[{name}] step {step}/{total_steps} train {train_loss:.4f} val {vloss:.4f} "
              f"lr {lr:.2e} |g| {grad_norm:.2f} {tps:,.0f} tok/s {mem_gb():.1f}GB")
        if run_logger:
            run_logger.log(stage=name, epoch=epoch, step=step, split="train", loss=round(train_loss, 4),
                           lr=lr, grad_norm=round(grad_norm, 3), tokens_per_sec=round(tps), gpu_mem_gb=round(mem_gb(), 1))
            run_logger.log(stage=name, epoch=epoch, step=step, split="val",
                           loss=round(vloss, 4), ppl=round(math.exp(min(vloss, 20)), 2))
        return vloss

    # step 0: the starting point (base / previous-stage weights, before this stage updates anything)
    ids0, labels0 = next(iter(loader))
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits0, _ = model(ids0.to(device), return_hidden=False)
    val_loss = log_point(0, 0, masked_lm_loss(logits0, labels0.to(device)).item(),
                         cosine_lr(0, total_steps, stage["lr"], warmup, min_lr), 0.0, 0.0)
    consider(val_loss, 0)  # step-0 (base/prev-stage) weights are the initial best

    step = 1  # step 0 is the pre-training anchor above; counting updates from 1
    last_t, last_step = time.time(), 0
    tl_sum, tl_n, grad_norm = 0.0, 0, 0.0
    stopped = False
    for epoch in range(stage["epochs"]):
        for i, (ids, labels) in enumerate(loader):
            ids, labels = ids.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(ids, return_hidden=False)
                loss = masked_lm_loss(logits, labels) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or 1e30))
                lr = cosine_lr(step, total_steps, stage["lr"], warmup, min_lr)
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.step()
                opt.zero_grad(set_to_none=True)
                tl_sum += loss.item() * accum
                tl_n += 1
                if step % log_every == 0:
                    print(f"[{name}] step {step}/{total_steps} train {loss.item() * accum:.4f}")
                if eval_every and step % eval_every == 0:        # train+val logged together
                    now = time.time()
                    tps = (step - last_step) * tokens_per_step / (now - last_t)
                    last_t, last_step = now, step
                    val_loss = log_point(step, epoch, tl_sum / max(1, tl_n), lr, grad_norm, tps)
                    tl_sum, tl_n = 0.0, 0
                    stopped = consider(val_loss, step)
                if cfg.get("save_every") and step % cfg["save_every"] == 0:
                    model.save_pretrained(os.path.join(cfg["output_dir"], f"{name}-step{step}"))
                step += 1
                if stopped:
                    print(f"[{name}] early stop at step {step - 1}: no val improvement in {patience} "
                          f"evals (best val {best['val']:.4f} @ step {best['step']})")
                    break
        opt.zero_grad(set_to_none=True)  # drop any partial accum group so its grads can't leak into the next epoch
        if stopped:
            break

    # Final point (same step for train + val) so both curves end together — skip
    # when we stopped early (the stopping eval was already the last logged point).
    if not stopped:
        now = time.time()
        tps = (step - 1 - last_step or 1) * tokens_per_step / max(1e-6, now - last_t)
        val_loss = log_point(step - 1, stage["epochs"] - 1, tl_sum / max(1, tl_n) if tl_n else float("nan"),
                             cosine_lr(step - 1, total_steps, stage["lr"], warmup, min_lr), grad_norm, tps)
        consider(val_loss, step - 1)

    # Restore best-val weights so the saved checkpoint and the next stage start
    # from the best model. Only when early stopping is on (patience set); with it
    # off, behavior is unchanged (last-step weights, full epochs).
    if patience and best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[{name}] restored best-val weights: val {best['val']:.4f} @ step {best['step']}")
        return best["val"]
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

    # Resume-aware: merge with a prior run's summary so a Stage 2-3 resume keeps
    # Stage 1's final val (matches the appended metrics.csv).
    prior = os.path.join(cfg["output_dir"], "summary.json")
    if os.path.exists(prior):
        try:
            with open(prior) as f:
                final_val = {**json.load(f).get("final_val_loss", {}), **final_val}
        except (ValueError, OSError):
            pass

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
        # Push ONLY a complete curriculum run (ended on final_stage). A partial /
        # smoke run (e.g. stage1-only) is skipped so default-on push doesn't upload
        # a 7.4GB checkpoint on every tuning run — weights are still saved locally.
        # To push a deliberately short run, set final_stage to its last stage.
        last_stage = cfg["stages"][-1]["name"]
        final_stage = push.get("final_stage") or last_stage
        if last_stage != final_stage:
            print(f"[hub] skip push: run ended on '{last_stage}', not final_stage '{final_stage}' "
                  f"(partial/smoke run). Weights saved locally at {cfg['output_dir']}/final.")
        else:
            from .hub import push_dir
            msg = f"stages={[s['name'] for s in cfg['stages']]} final_val={final_val} seed={cfg['seed']}"
            push_dir(os.path.join(cfg["output_dir"], "final"), push["repo_id"],
                     private=push.get("private", True), commit_message=msg)
