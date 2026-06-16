"""CPU smoke test: prove the training step works end-to-end without any download.

Builds a tiny randomly-initialized ChronoGPT, runs a few masked-LM steps on
fake packed data, and asserts the loss is finite and that gradients flow. Run
this FIRST on any new machine before touching a real vintage.
"""
import torch

from chrono_instruct.model import build_tiny
from chrono_instruct.train import masked_lm_loss


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
