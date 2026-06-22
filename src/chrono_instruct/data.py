"""Data: load ChronoInstruct-SFT, filter, reconstruct curriculum stages, pack.

The released dataset has three columns:
  - `conversation`: a JSON object {instruction, input, output} (arrives as a
    dict or as a JSON string depending on the loader; we handle both).
  - `label`: the GPT-4.1 temporal-screen verdict. The paper keeps only pairs
    classified label 0 ("knowledge available pre-2000") with confidence 10.
  - `source`: which of the three upstreams the pair came from.

The temporal screen is a single conservative pre-2000 filter applied ONCE, not
per vintage: pre-2000 data is pre-tau for every vintage tau >= 1999, so one
filtered corpus is reused across all vintage runs (see `prepare_stages`). The
3-stage curriculum is reconstructed by grouping the filtered rows on `source`.
Each example is rendered Alpaca-style and the loss is masked to the response
span only; examples are packed into fixed-length blocks (the model has no
padding-mask support).
"""
import ast
import hashlib
import json
import os
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
    if isinstance(conv, str):  # released `conversation` may be a JSON string
        conv = json.loads(conv)
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


def _parse_label(label):
    """Parse the `label` verdict, tolerant of JSON *and* Python-dict-repr strings.

    The GPT-4.1 verdict is stored inconsistently across sources (verified on the
    box): scratch and self-instruct use valid JSON ('{"label": 0, ...}'), but
    Tulu rows use single-quoted Python-dict reprs ("{'label': 0, ...}") that
    json.loads rejects. Falling back to ast.literal_eval means those rows get
    screened on their real verdict instead of being silently dropped — which was
    the cause of Tulu collapsing to ~32k vs the paper's ~357k. Returns a dict, or
    None if the verdict is genuinely unrecoverable.
    """
    if isinstance(label, dict):
        return label
    if not isinstance(label, str):
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            obj = parse(label)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def keep_row(row, min_confidence=10):
    """Temporal screen: keep pairs the GPT-4.1 classifier marked pre-2000.

    Paper s2.2.1 keeps label 0 with confidence 10. Verdicts that can't be parsed
    are dropped (the paper's "ambiguity -> label 1" stance). Set `min_confidence`
    to null in the config to keep every label-0 row regardless of confidence.
    """
    obj = _parse_label(row.get("label"))
    if obj is None or obj.get("label") != 0:
        return False
    conf = obj.get("confidence")
    return min_confidence is None or conf is None or conf >= min_confidence


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


def source_counts(dataset, after_filter=False, min_confidence=10):
    """Inspect helper: unique `source` values and their row counts.

    With after_filter=True, count only rows passing the temporal screen.
    """
    counts = {}
    for row in dataset:
        if after_filter and not keep_row(row, min_confidence):
            continue
        src = row.get("source") or "<none>"
        counts[src] = counts.get(src, 0) + 1
    return counts


def _cache_key(cfg):
    payload = {
        "dataset": cfg["dataset"],
        "block_size": cfg["block_size"],
        "val_fraction": cfg.get("val_fraction", 0.05),
        "seed": cfg.get("seed", 123),
        "min_confidence": cfg.get("min_confidence", 10),
        "stages": [[s["name"], s["sources"]] for s in cfg["stages"]],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def prepare_stages(cfg):
    """Build (or load from cache) packed train/val blocks, keyed by stage name.

    Only the tokenized data is cached — never the model or hyperparameters like
    lr/epochs (those stay live from the config). The data depends solely on
    (dataset, screen, block_size, stages, seed), so it is built once and reused
    across every vintage run. Returns {stage_name: (train_ds, val_ds)}.
    """
    cache_dir = cfg.get("cache_dir", "cache")
    path = os.path.join(cache_dir, f"packed-{_cache_key(cfg)}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)

    rows = [r for r in load_raw(cfg["dataset"]) if keep_row(r, cfg.get("min_confidence", 10))]
    stages = {
        s["name"]: load_stage(rows, s["sources"], cfg["block_size"],
                              cfg.get("val_fraction", 0.05), cfg.get("seed", 123))
        for s in cfg["stages"]
    }
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(stages, path)
    return stages
