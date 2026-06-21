"""Command-line entrypoint: `chrono <command> --config ...`.

Commands:
  inspect  -- print the dataset's unique `source` values + counts (for stage mapping)
  train    -- run curriculum SFT from a config
  infer    -- generate text or extract an embedding from a vintage
  eval     -- run the president consistency test
"""
import argparse

import yaml


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    p = argparse.ArgumentParser(prog="chrono")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect")
    pi.add_argument("--dataset", default="manelalab/ChronoInstruct-SFT")

    pt = sub.add_parser("train")
    pt.add_argument("--config", required=True)

    pf = sub.add_parser("infer")
    pf.add_argument("--repo", required=True)
    pf.add_argument("--mode", choices=["generate", "embed"], default="generate")
    pf.add_argument("--text", required=True)

    pe = sub.add_parser("eval")
    pe.add_argument("--repo", required=True)
    pe.add_argument("--cutoff", type=int, required=True, help="model knowledge-cutoff year")

    args = p.parse_args(argv)

    if args.cmd == "inspect":
        from .data import load_raw, source_counts

        def _print(counts, footer=""):
            for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"{n:>9,}  {src}")
            print(f"{sum(counts.values()):>9,}  TOTAL{footer}")

        ds = load_raw(args.dataset)
        sample = next(iter(ds))
        print("=== sample row (verify column shapes) ===")
        print("conversation:", repr(sample.get("conversation"))[:300])
        print("label:", repr(sample.get("label")))
        print("source:", repr(sample.get("source")))
        print("\n=== source counts (raw) ===")
        _print(source_counts(ds))
        print("\n=== source counts (after label==0 & confidence>=10 screen) ===")
        _print(source_counts(ds, after_filter=True), "  (paper Table 1: 425,119)")

    elif args.cmd == "train":
        from .train import run
        run(_load_cfg(args.config))

    elif args.cmd == "infer":
        from .infer import load, generate, embed
        model, device = load(args.repo)
        if args.mode == "generate":
            print(generate(model, device, args.text))
        else:
            v = embed(model, device, args.text)
            print("embedding shape:", tuple(v.shape))

    elif args.cmd == "eval":
        from .infer import load
        from .eval import president_test
        model, device = load(args.repo)
        for r in president_test(model, device, args.cutoff):
            flag = "  (past cutoff)" if r["past_cutoff"] else ""
            ok = "OK" if r["correct"] else "x "
            print(f"{ok} {r['target_year']} {r['target']:<16} -> {r['prediction']!r}{flag}")


if __name__ == "__main__":
    main()
