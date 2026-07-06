"""Command-line entrypoint: `chrono <command> --config ...`.

Commands:
  inspect  -- print the dataset's unique `source` values + counts (for stage mapping)
  train    -- run curriculum SFT from a config
  infer    -- generate text or extract an embedding from a vintage
  eval     -- run the president consistency test (Table 2)
  push     -- upload a local run dir to the Hugging Face Hub
  alpaca   -- generate AlpacaEval outputs for a model (chrono or HF reference)
  winrate  -- length-controlled win-rate of model vs reference outputs (Figure 3)
  figure   -- plot Figure 1 (one run), 2 (vintage sweep), or 3 (win rates)
"""
import argparse
import os

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

    pp = sub.add_parser("push")
    pp.add_argument("--repo", required=True, help="local run dir, e.g. runs/.../final")
    pp.add_argument("--to", required=True, help="target HF repo id")
    pp.add_argument("--private", action="store_true")
    pp.add_argument("--message", default=None, help="HF commit message (e.g. stages + val loss)")

    pa = sub.add_parser("alpaca")
    pa.add_argument("--repo", required=True, help="model repo/dir to generate from")
    pa.add_argument("--backend", choices=["chrono", "hf"], default="chrono")
    pa.add_argument("--name", required=True, help="generator name recorded in the outputs")
    pa.add_argument("--out", required=True, help="output JSON path")
    pa.add_argument("--n", type=int, default=None, help="limit #instructions (debug)")

    pw = sub.add_parser("winrate")
    pw.add_argument("--model", required=True, help="model outputs JSON")
    pw.add_argument("--reference", required=True, help="reference outputs JSON")

    pg = sub.add_parser("figure")
    pg.add_argument("--kind", choices=["1", "2", "3"], required=True)
    pg.add_argument("--run", help="run dir (figure 1)")
    pg.add_argument("--runs", nargs="+", help="run dirs (figure 2)")
    pg.add_argument("--results", help="win-rate JSON (figure 3)")
    pg.add_argument("--out", help="output image path")

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
        from .eval import president_test, major_events_test
        model, device = load(args.repo)
        print("=== U.S. presidents (Table 2) ===")
        for r in president_test(model, device, args.cutoff):
            flag = "  (past cutoff)" if r["past_cutoff"] else ""
            ok = "OK" if r["correct"] else "x "
            print(f"{ok} {r['target_year']} {r['target']:<16} -> {r['prediction']!r}{flag}")
        print("=== major events (Table 3) ===")
        for r in major_events_test(model, device, args.cutoff):
            flag = "  (past cutoff)" if r["past_cutoff"] else ""
            ok = "OK" if r["correct"] else "x "
            print(f"{ok} {r['event_year']} {r['answer']:<16} -> {r['prediction']!r}{flag}")

    elif args.cmd == "push":
        from .hub import push_dir
        push_dir(args.repo, args.to, private=args.private, commit_message=args.message)

    elif args.cmd == "alpaca":
        import json
        from .eval import alpaca_instructions, alpaca_outputs
        outs = alpaca_outputs(args.repo, alpaca_instructions(args.n), args.name, backend=args.backend)
        with open(args.out, "w") as f:
            json.dump(outs, f, indent=2)
        print(f"wrote {len(outs)} outputs -> {args.out}")

    elif args.cmd == "winrate":
        from .eval import alpaca_winrate
        print(f"LC win rate: {alpaca_winrate(args.model, args.reference):.2f}%")

    elif args.cmd == "figure":
        from . import figures
        if args.kind == "1":
            # Default the image INTO the run dir (next to its metrics.csv), not cwd,
            # so `chrono figure --run <dir>` never litters the project root.
            figures.figure1(args.run, args.out or os.path.join(args.run, "figure1.png"))
        elif args.kind == "2":
            figures.figure2(args.runs, args.out or "figure2.png")
        else:
            figures.figure3(args.results, args.out or "figure3.png")


if __name__ == "__main__":
    main()
