#!/usr/bin/env python3
"""Derive a per-vintage training config from the base config.

Overrides model_repo, output_dir, and push_to_hub.repo_id BY KEY -- a structure-
aware YAML edit, not a text substitution. (A `sed` find/replace on the year can
silently miss fields like `chrono-instruct-v1-20201231`, where the `v1-` infix
breaks the match, leaving every vintage pushing to the same HF repo.)

    python scripts/make_vintage_config.py \
        --base configs/train.yaml --out configs/_vintage_1999.yaml \
        --cutoff 19991231 \
        --output-dir /home/ubuntu/persist/runs/chrono-instruct-1999 \
        --hf-user HZ0619
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base config to derive from")
    ap.add_argument("--out", required=True, help="per-vintage config to write")
    ap.add_argument("--cutoff", required=True, help="YYYYMMDD knowledge cutoff, e.g. 19991231")
    ap.add_argument("--output-dir", required=True, help="absolute run dir (keep on the persistent FS)")
    ap.add_argument("--hf-user", required=True, help="HF namespace for push_to_hub.repo_id")
    args = ap.parse_args()

    with open(args.base) as f:
        cfg = yaml.safe_load(f)

    cfg["model_repo"] = f"manelalab/chrono-gpt-v1-{args.cutoff}"
    cfg["output_dir"] = args.output_dir
    cfg.setdefault("push_to_hub", {})
    cfg["push_to_hub"]["repo_id"] = f"{args.hf_user}/chrono-instruct-v1-{args.cutoff}"

    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}  (model_repo={cfg['model_repo']}, "
          f"output_dir={args.output_dir}, repo_id={cfg['push_to_hub']['repo_id']})")


if __name__ == "__main__":
    main()
