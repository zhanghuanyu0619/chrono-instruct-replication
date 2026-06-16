"""Data: load ChronoInstruct-SFT, reconstruct curriculum stages, format + pack.

The released dataset has columns `conversation {instruction, input, output}`,
`label`, and `source`. The 3-stage curriculum is reconstructed by grouping on
`source` (matched as a case-insensitive substring so we don't depend on the
exact label strings). Each example is rendered with the Alpaca template; the
loss is masked to the response span only. Examples are packed into fixed-length
blocks because the model has no padding-mask support.
"""
from dataclasses import dataclass

import torch
import tiktoken
from datasets import load_dataset

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token  # 50256, used as end-of-response separator

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_example(conv):
    instruction = (conv.get("instruction") or "").strip()
    inp = (conv.get("input") or "").strip()
    output = (conv.get("output") or "").strip()
    prompt = (PROMPT_WITH_INPUT if inp else PROMPT_NO_INPUT).format(instruction=instruction, input=inp)
    return prompt, output


def encode_example(conv):
    """Return (token_ids, target_mask) where target_mask is True on response tokens."""
    prompt, output = format_example(conv)
    p_ids = ENC.encode(prompt)
    r_ids = ENC.encode(output) + [EOT]
    ids = p_ids + r_ids
    mask = [False] * len(p_ids) + [True] * len(r_ids)
    return ids, mask


def stage_examples(dataset, sources):
    """Filter rows whose `source` matches any of `sources` (case-insensitive substring)."""
    needles = [s.lower() for s in sources]
    for row in dataset:
        src = (row.get("source") or "").lower()
        if any(n in src for n in needles):
            yield row["conversation"]


def pack_blocks(examples, block_size):
    """Concatenate encoded examples into fixed-length (input_ids, labels) blocks.

    labels[t] = input_ids[t] on response tokens, else -100. The shift for
    next-token prediction is applied in the training loss, not here.
    """
    buf_ids, buf_mask = [], []
    blocks = []
    for conv in examples:
        ids, mask = encode_example(conv)
        buf_ids.extend(ids)
        buf_mask.extend(mask)
        while len(buf_ids) >= block_size:
            chunk_ids = buf_ids[:block_size]
            chunk_mask = buf_mask[:block_size]
            labels = [tid if m else -100 for tid, m in zip(chunk_ids, chunk_mask)]
            blocks.append((chunk_ids, labels))
            buf_ids = buf_ids[block_size:]
            buf_mask = buf_mask[block_size:]
    return blocks


@dataclass
class PackedDataset(torch.utils.data.Dataset):
    blocks: list

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        ids, labels = self.blocks[i]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def load_stage(dataset, sources, block_size, val_fraction=0.05, seed=123):
    blocks = pack_blocks(stage_examples(dataset, sources), block_size)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(blocks), generator=g).tolist()
    n_val = int(len(blocks) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = PackedDataset([blocks[i] for i in train_idx])
    val = PackedDataset([blocks[i] for i in val_idx])
    return train, val


def load_raw(dataset_name):
    return load_dataset(dataset_name, split="train")


def source_counts(dataset):
    """Inspect helper: unique `source` values and their row counts."""
    counts = {}
    for row in dataset:
        src = row.get("source") or "<none>"
        counts[src] = counts.get(src, 0) + 1
    return counts
