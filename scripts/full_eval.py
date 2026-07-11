#!/usr/bin/env python3
"""Full evaluation cycle for one trained vintage — NO val_max_blocks cap.

Motivation: during training we cap the held-out set at `val_max_blocks` (500) so
periodic eval stays cheap. That cap only truncates how many val BLOCKS are scored;
it never changes which EXAMPLES are held out (the split is seeded, applied before
packing). This script re-scores the SAME seeded holdout in full, giving the honest
per-stage validation loss, and (optionally) runs the chronological-consistency
tests (Table 2/3).

Reproducibility: the val examples here are byte-identical to the ones held out
during training — same dataset, same `seed`, same `val_fraction`, same screen —
because we reuse data.py's split logic with val_max_blocks disabled. Nothing is
re-trained; we only measure.

Usage (single vintage):
  python scripts/full_eval.py --config configs/train.yaml \
      --repo runs/chrono-instruct-2020/final --cutoff 2020 \
      [--stages stage3_tulu] [--consistency] [--out results/full_eval/2020.json]

Sweep all vintages from local run dirs (see the shell loop at the bottom of this file).
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chrono_instruct.data import (  # noqa: E402
    ENC, PackedDataset, keep_row, load_raw, pack_blocks, stage_examples,
)
from chrono_instruct.train import evaluate  # noqa: E402  (exact training-time loss)
from chrono_instruct.infer import load, free_memory  # noqa: E402


def _load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_full_val(rows, sources, block_size, val_fraction, seed):
    """The FULL (uncapped) held-out val set for one stage — mirrors data.load_stage's
    val branch exactly, but packs only the val side (train packing is the slow part)
    and applies NO val_max_blocks cap. Same seed => same held-out examples as training."""
    examples = list(stage_examples(rows, sources))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(examples), generator=g).tolist()
    n_val = int(len(examples) * val_fraction)
    val_ex = [examples[i] for i in perm[:n_val]]          # identical holdout to training
    return PackedDataset(pack_blocks(val_ex, block_size)), len(val_ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo", required=True, help="model dir or HF repo id")
    ap.add_argument("--cutoff", type=int, required=True, help="knowledge-cutoff year (for consistency flags)")
    ap.add_argument("--stages", nargs="+", default=None,
                    help="stage names to score (default: all stages in the config)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--consistency", action="store_true",
                    help="also run the president (Table 2) + major-events (Table 3) tests")
    ap.add_argument("--out", default=None, help="write the result dict as JSON here")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    block_size = cfg["block_size"]
    seed = cfg.get("seed", 123)
    val_fraction = cfg.get("val_fraction", 0.05)
    bs = args.batch_size or cfg.get("batch_size", 8)
    stage_defs = {s["name"]: s["sources"] for s in cfg["stages"]}
    stage_names = args.stages or list(stage_defs)

    print(f"[full-eval] screening dataset once (seed={seed}, val_fraction={val_fraction}, NO val cap) ...")
    rows = [r for r in load_raw(cfg["dataset"]) if keep_row(r, cfg.get("min_confidence", 10))]
    print(f"[full-eval] {len(rows):,} rows kept")

    model, device = load(args.repo)
    print(f"[full-eval] loaded {args.repo} on {device}")

    result = {"repo": args.repo, "cutoff": args.cutoff, "seed": seed,
              "val_fraction": val_fraction, "uncapped": True, "full_val_loss": {}}

    for name in stage_names:
        val_ds, n_ex = build_full_val(rows, stage_defs[name], block_size, val_fraction, seed)
        loader = DataLoader(val_ds, batch_size=bs)
        loss = evaluate(model, loader, device)          # token-weighted, same as training
        ppl = float(torch.tensor(loss).exp())
        result["full_val_loss"][name] = {"loss": round(loss, 4), "ppl": round(ppl, 3),
                                         "val_examples": n_ex, "val_blocks": len(val_ds)}
        print(f"[full-eval]   {name}: loss {loss:.4f}  ppl {ppl:.3f}  "
              f"({n_ex:,} held-out examples -> {len(val_ds):,} blocks, uncapped)")

    if args.consistency:
        from chrono_instruct.eval import president_test, major_events_test
        result["president_test"] = president_test(model, device, args.cutoff)
        result["major_events_test"] = major_events_test(model, device, args.cutoff)
        pc = result["president_test"] + result["major_events_test"]
        # A chronologically consistent model is RIGHT before cutoff and WRONG after.
        consistent = sum(1 for r in pc if r["correct"] != r["past_cutoff"])
        print(f"[full-eval]   consistency: {consistent}/{len(pc)} rows behave as expected "
              f"(correct pre-cutoff, wrong post-cutoff)")

    del model
    free_memory()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[full-eval] wrote {args.out}")
    print(json.dumps(result["full_val_loss"], indent=2))


if __name__ == "__main__":
    main()
