"""Quick inference smoke + GPU-memory-clear demo.

    python scripts/inference_demo.py --repo manelalab/chrono-gpt-instruct-v1-19991231

Loads an instruct vintage, runs a generation and an embedding, then frees the GPU
and reports memory — a fast way to confirm a trained/released model works and to
see how to release VRAM between model loads.
"""
import argparse

import torch

from chrono_instruct.infer import load, generate, embed, free_memory
from chrono_instruct.data import PROMPT_NO_INPUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="manelalab/chrono-gpt-instruct-v1-19991231",
                    help="HF repo id or a local run dir (e.g. runs/chrono-instruct-2020/final)")
    ap.add_argument("--instruction", default="Explain what inflation is in one sentence.")
    args = ap.parse_args()

    model, device = load(args.repo)
    print(f"loaded {args.repo} on {device}")
    if torch.cuda.is_available():
        print("VRAM after load:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")

    print("\n--- generation ---")
    prompt = PROMPT_NO_INPUT.format(instruction=args.instruction)
    print(generate(model, device, prompt, max_new_tokens=60))

    print("\n--- embedding ---")
    v = embed(model, device, "Inflation is a sustained rise in the general price level.", layer=-1)
    print("embedding shape:", tuple(v.shape))

    # Release the GPU.
    del model
    free_memory()
    if torch.cuda.is_available():
        print("\nVRAM after free:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB allocated")


if __name__ == "__main__":
    main()
