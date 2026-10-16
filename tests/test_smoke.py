"""CPU smoke test: prove the training step works end-to-end without any download.

Builds a tiny randomly-initialized ChronoGPT, runs a few masked-LM steps on
fake packed data, and asserts the loss is finite and that gradients flow. Run
this FIRST on any new machine before touching a real vintage.
"""
import torch

from chrono_instruct.model import build_tiny
from chrono_instruct.train import cosine_lr, masked_lm_loss


def test_cosine_lr_floor_and_warmup():
    """Warmup ramps to base_lr; cosine then decays to the min_lr floor, not past it."""
    base, total, warmup = 3e-4, 40, 4
    # Warmup ramps up and reaches base_lr at the warmup boundary.
    assert cosine_lr(0, total, base, warmup) < base
    assert abs(cosine_lr(warmup, total, base, warmup) - base) < 1e-9
    # Floor off -> classic decay to ~0 at the last step.
    assert cosine_lr(total, total, base, warmup, 0.0) < 1e-9
    # Floor at 10% of base -> last step lands exactly on the floor, never below.
    floor = 0.1 * base
    assert abs(cosine_lr(total, total, base, warmup, floor) - floor) < 1e-12
    # Across the decay phase lr stays within [floor, base] and is non-increasing.
    prev = base + 1
    for s in range(warmup, total + 1):
        lr = cosine_lr(s, total, base, warmup, floor)
        assert floor - 1e-12 <= lr <= base + 1e-12
        assert lr <= prev + 1e-12
        prev = lr


def test_training_step_decreases_loss():
    torch.manual_seed(0)
    vocab, T, B = 512, 32, 2
    model = build_tiny(vocab_size=vocab)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ids = torch.randint(0, vocab, (B, T))
    labels = ids.clone()
    labels[:, : T // 2] = -100  # mask the "prompt" half

    losses = []
    for _ in range(5):
        logits, layer_outputs = model(ids)
        assert logits.shape == (B, T, vocab)
        assert len(layer_outputs) == len(model.blocks)
        loss = masked_lm_loss(logits, labels)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        assert model.embed.weight.grad is not None  # gradients flow
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]  # the loop actually learns on a fixed batch


def test_kv_cache_matches_full_forward():
    """Incremental KV-cached decoding must match the full-sequence forward.

    Feeds a sequence one token at a time with a `past` cache and checks the
    per-position logits (and the greedy argmax at every position) equal a single
    full-sequence pass — i.e. the cache is a pure speedup, not a behavior change.
    """
    torch.manual_seed(0)
    vocab, T = 512, 16
    model = build_tiny(vocab_size=vocab)
    model.eval()
    ids = torch.randint(0, vocab, (1, T))

    with torch.no_grad():
        full, _ = model(ids, return_hidden=False)              # one parallel pass
        past, steps = [None] * len(model.blocks), []
        for t in range(T):                                     # one token at a time
            lg, past = model(ids[:, t : t + 1], return_hidden=False, past=past)
            steps.append(lg[:, -1, :])
        cached = torch.stack(steps, dim=1)

    assert cached.shape == full.shape
    assert torch.equal(full.argmax(-1), cached.argmax(-1))     # identical greedy choices
    assert torch.allclose(full, cached, atol=1e-2)             # logits match (bf16 internals)


def test_trim_rotary_is_lossless_within_range():
    torch.manual_seed(0)
    model = build_tiny(vocab_size=512).eval()
    ids = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        full = model(ids, return_hidden=False)[0]
        model.trim_rotary(16)
        assert model.blocks[0].attn.rotary.cos.shape[0] == 16
        assert torch.equal(model(ids, return_hidden=False)[0], full)
        import pytest
        with pytest.raises(RuntimeError):
            model(torch.randint(0, 512, (1, 20)), return_hidden=False)


def test_left_padding_matches_rows_run_alone():
    """`pad_lens`: a left-padded batch of different-length rows gives each row the logits it gets alone,
    in prefill and through KV-cached decoding, and the padding path is bit-identical to the default
    when no row is padded (so the default path is untouched)."""
    torch.manual_seed(0)
    vocab = 512
    model = build_tiny(vocab_size=vocab).eval()
    lens = [5, 9, 3]
    rows = [torch.randint(0, vocab, (n,)) for n in lens]
    L = max(lens)
    pad_lens = torch.tensor([L - n for n in lens])
    batch = torch.stack([torch.cat([torch.zeros(L - n, dtype=torch.long), r]) for n, r in zip(lens, rows)])
    nxt = torch.randint(0, vocab, (len(lens), 3))              # three decode steps after the prompts

    with torch.no_grad():
        # prefill
        padded, _ = model(batch, return_hidden=False, pad_lens=pad_lens)
        for b, (n, r) in enumerate(zip(lens, rows)):
            alone, _ = model(r[None], return_hidden=False)
            assert torch.allclose(padded[b, L - n :], alone[0], atol=1e-2), f"row {b} prefill"
        # cached decoding: the same pad_lens on every step
        past = [None] * len(model.blocks)
        _, past = model(batch, return_hidden=False, past=past, pad_lens=pad_lens)
        steps = []
        for t in range(nxt.size(1)):
            lg, past = model(nxt[:, t : t + 1], return_hidden=False, past=past, pad_lens=pad_lens)
            steps.append(lg[:, -1])
        for b, (n, r) in enumerate(zip(lens, rows)):
            full, _ = model(torch.cat([r, nxt[b]])[None], return_hidden=False)
            for t in range(nxt.size(1)):
                assert torch.allclose(steps[t][b], full[0, n + t], atol=1e-2), f"row {b} decode step {t}"
        # no padding: the two code paths agree exactly
        ids = torch.randint(0, vocab, (2, 7))
        assert torch.equal(model(ids, return_hidden=False)[0],
                           model(ids, return_hidden=False, pad_lens=torch.zeros(2, dtype=torch.long))[0])
