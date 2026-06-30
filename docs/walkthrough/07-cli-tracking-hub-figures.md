# 07 — The glue: CLI, tracking, Hub push, and figures

This doc walks the project's "glue" modules — the small files that wire everything together and turn a training run into results, logs, and plots. They are: `cli.py` (the `chrono` command you type), `tracking.py` (the metrics logger), `hub.py` (uploading checkpoints to Hugging Face), `figures.py` (the paper's Figures 1/2/3), and the one-line `__init__.py`. None of these contains the "hard" ML — they orchestrate the heavy modules (`train.py`, `infer.py`, `eval.py`, `data.py`) that live alongside them.

If a term like "checkpoint", "embedding", "loss", or "perplexity" is unfamiliar, see `01-ml-primer.md`. The research meaning of each figure lives in `02-paper-and-research-framing.md`; here we only explain how the code produces them.

## Table of contents
- [1. `cli.py` — the `chrono` dispatcher](#1-clipy--the-chrono-dispatcher)
- [2. `tracking.py` — `RunLogger`](#2-trackingpy--runlogger)
- [3. `hub.py` — `push_dir`](#3-hubpy--push_dir)
- [4. `figures.py` — Figures 1, 2, 3](#4-figurespy--figures-1-2-3)
- [5. `__init__.py` — the package marker](#5-__init__py--the-package-marker)
- [6. Command → code → artifact map](#6-command--code--artifact-map)
- [7. Mini-FAQ](#7-mini-faq)

---

## 1. `cli.py` — the `chrono` dispatcher

This is the single entry point for the whole project. When you type `chrono train --config ...` in a shell, you are calling `main()` in this file. It uses Python's standard `argparse` library to define a set of *subcommands* (like `git commit`, `git push` — one program, many verbs), parse the arguments, and route to the right module function.

### The module docstring (the menu)

```python
1  """Command-line entrypoint: `chrono <command> --config ...`.
2
3  Commands:
4    inspect  -- print the dataset's unique `source` values + counts (for stage mapping)
5    train    -- run curriculum SFT from a config
6    infer    -- generate text or extract an embedding from a vintage
7    eval     -- run the president consistency test (Table 2)
8    push     -- upload a local run dir to the Hugging Face Hub
9    alpaca   -- generate AlpacaEval outputs for a model (chrono or HF reference)
10   winrate  -- length-controlled win-rate of model vs reference outputs (Figure 3)
11   figure   -- plot Figure 1 (one run), 2 (vintage sweep), or 3 (win rates)
12  """
```

This docstring is the canonical list of the eight subcommands. Each one maps to one `elif` branch in `main()` below.

### Imports and the config loader

```python
13  import argparse
14
15  import yaml
16
17
18  def _load_cfg(path):
19      with open(path) as f:
20          return yaml.safe_load(f)
```

Only two top-level imports: `argparse` (the CLI framework) and `yaml` (to read config files). `_load_cfg` reads a YAML config file (e.g. `configs/chrono-1999.yaml`) into a plain Python dict. The leading underscore is a Python convention meaning "private helper, not part of the public interface."

Notice that the heavy modules (`train`, `infer`, `eval`, `data`, `hub`, `figures`) are **not** imported at the top. They are imported *inside* each branch (e.g. line 88 `from .train import run`). This is deliberate: importing `train.py` pulls in PyTorch and the model code, which is slow and needs a GPU environment. By deferring the import until the branch actually runs, `chrono inspect` or `chrono figure` stays fast and works even on a laptop without PyTorch fully set up.

### `main()` and the argparse setup

```python
23  def main(argv=None):
24      p = argparse.ArgumentParser(prog="chrono")
25      sub = p.add_subparsers(dest="cmd", required=True)
```

`main()` is the function that runs. `argv=None` means "by default read from the real command line"; passing a list lets tests call `main([...])` directly. Line 24 creates the parser named `chrono`. Line 25 creates a *subparser group*: `dest="cmd"` stores which verb was chosen in `args.cmd`, and `required=True` forces you to pick one (typing bare `chrono` errors out with usage help).

Each subcommand is then declared with its own arguments. Reading them in order:

**`inspect`** — explore the raw dataset:

```python
27      pi = sub.add_parser("inspect")
28      pi.add_argument("--dataset", default="manelalab/ChronoInstruct-SFT")
```

One optional `--dataset` argument defaulting to the paper's published dataset on Hugging Face. You'd override it only to point at a different/local copy.

**`train`** — run the curriculum SFT:

```python
30      pt = sub.add_parser("train")
31      pt.add_argument("--config", required=True)
```

A single required `--config` path. Everything about the run (model, stages, learning rates, output dir, optional W&B/HF) lives in that YAML file, not on the command line. This keeps runs reproducible — the config is the record.

**`infer`** — use a trained model on one piece of text:

```python
33      pf = sub.add_parser("infer")
34      pf.add_argument("--repo", required=True)
35      pf.add_argument("--mode", choices=["generate", "embed"], default="generate")
36      pf.add_argument("--text", required=True)
```

`--repo` is the model to load (a local run dir or an HF repo id). `--mode` picks between `generate` (produce text continuation) and `embed` (extract a fixed-length vector — see `01-ml-primer.md` on embeddings; these are what feed downstream return-prediction work). `--text` is the input string. `choices=[...]` makes argparse reject any other value automatically.

**`eval`** — run the chronological-consistency tests:

```python
38      pe = sub.add_parser("eval")
39      pe.add_argument("--repo", required=True)
40      pe.add_argument("--cutoff", type=int, required=True, help="model knowledge-cutoff year")
```

`--repo` is again the model. `--cutoff` is the model's knowledge-cutoff year as an integer (`type=int`). The eval needs it to know which test questions fall *after* the model's training data — the whole point of the paper is that a chronologically consistent model should *not* know answers past its cutoff.

**`push`** — upload a checkpoint to Hugging Face:

```python
42      pp = sub.add_parser("push")
43      pp.add_argument("--repo", required=True, help="local run dir, e.g. runs/.../final")
44      pp.add_argument("--to", required=True, help="target HF repo id")
45      pp.add_argument("--private", action="store_true")
46      pp.add_argument("--message", default=None, help="HF commit message (e.g. stages + val loss)")
```

`--repo` here is the *local* folder to upload (note the same flag name means different things in `infer`/`eval` vs `push` — here it's a source directory). `--to` is the destination HF repo id. `--private` is a *flag*: `action="store_true"` means its mere presence sets it `True` (no value needed), so the repo is created private. `--message` is the commit message recorded on the Hub.

**`alpaca`** — generate model answers for the AlpacaEval benchmark (input to Figure 3):

```python
48      pa = sub.add_parser("alpaca")
49      pa.add_argument("--repo", required=True, help="model repo/dir to generate from")
50      pa.add_argument("--backend", choices=["chrono", "hf"], default="chrono")
51      pa.add_argument("--name", required=True, help="generator name recorded in the outputs")
52      pa.add_argument("--out", required=True, help="output JSON path")
53      pa.add_argument("--n", type=int, default=None, help="limit #instructions (debug)")
```

`--repo` is the model. `--backend` chooses how to load/run it: `chrono` for this project's own ChronoGPT models, `hf` for a stock Hugging Face reference model (so you can also generate the *opponent's* answers — e.g. Qwen-1.5-1.8B-Chat). `--name` is the label stored inside the output file. `--out` is where the JSON of generated answers is written. `--n` optionally caps the number of instructions for a quick debug run (default `None` = all 805 AlpacaEval prompts).

**`winrate`** — score one model's answers against a reference:

```python
55      pw = sub.add_parser("winrate")
56      pw.add_argument("--model", required=True, help="model outputs JSON")
57      pw.add_argument("--reference", required=True, help="reference outputs JSON")
```

Both arguments are JSON files produced by `alpaca`. This command computes the length-controlled (LC) win rate — how often a judge prefers `--model`'s answer over `--reference`'s, adjusted for answer length. That single percentage is one bar in Figure 3.

**`figure`** — plot:

```python
59      pg = sub.add_parser("figure")
60      pg.add_argument("--kind", choices=["1", "2", "3"], required=True)
61      pg.add_argument("--run", help="run dir (figure 1)")
62      pg.add_argument("--runs", nargs="+", help="run dirs (figure 2)")
63      pg.add_argument("--results", help="win-rate JSON (figure 3)")
64      pg.add_argument("--out", help="output image path")
```

`--kind` picks which figure (1, 2, or 3 — kept as strings for the `choices` match). The other flags are figure-specific and only some apply per kind: `--run` (single run dir, Figure 1), `--runs` (several run dirs; `nargs="+"` means "one or more values", Figure 2), `--results` (a win-rate JSON, Figure 3), and an optional `--out` image path. argparse allows the unused ones to stay `None`; the dispatch below picks the right combination.

### Parsing and dispatch

```python
66      args = p.parse_args(argv)
```

This line does the actual parsing: it validates the input and returns an `args` object whose attributes are the flags (`args.cmd`, `args.config`, etc.). After this, the rest of `main()` is one big `if/elif` chain on `args.cmd`.

**`inspect` branch** — the most code, because it prints a human report:

```python
68      if args.cmd == "inspect":
69          from .data import load_raw, source_counts
70
71          def _print(counts, footer=""):
72              for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
73                  print(f"{n:>9,}  {src}")
74              print(f"{sum(counts.values()):>9,}  TOTAL{footer}")
75
76          ds = load_raw(args.dataset)
77          sample = next(iter(ds))
78          print("=== sample row (verify column shapes) ===")
79          print("conversation:", repr(sample.get("conversation"))[:300])
80          print("label:", repr(sample.get("label")))
81          print("source:", repr(sample.get("source")))
82          print("\n=== source counts (raw) ===")
83          _print(source_counts(ds))
84          print("\n=== source counts (after label==0 & confidence>=10 screen) ===")
85          _print(source_counts(ds, after_filter=True), "  (paper Table 1: 425,119)")
```

`load_raw` (from `data.py`) loads the dataset; `source_counts` tallies how many rows come from each `source` value. The inner `_print` helper formats counts as right-aligned, comma-grouped numbers sorted descending (`key=lambda kv: -kv[1]`). Line 77 grabs one example row and lines 78–81 print its `conversation`/`label`/`source` fields so you can eyeball the column shapes. Then it prints raw counts (line 83) and counts *after* the paper's screen — `label==0` (a usable example) and `confidence>=10` — comparing against the paper's Table 1 figure of 425,119 (line 85). This is the command you run first, to see which `source` values exist so you can map them to curriculum stages in a config.

**`train` branch:**

```python
87      elif args.cmd == "train":
88          from .train import run
89          run(_load_cfg(args.config))
```

Loads the YAML config into a dict and hands it to `train.run(cfg)`. That's the entire training driver — everything else (data prep, the 3-stage curriculum, logging via `RunLogger`, optional HF push) happens inside `train.py`.

**`infer` branch:**

```python
91      elif args.cmd == "infer":
92          from .infer import load, generate, embed
93          model, device = load(args.repo)
94          if args.mode == "generate":
95              print(generate(model, device, args.text))
96          else:
97              v = embed(model, device, args.text)
98              print("embedding shape:", tuple(v.shape))
```

`infer.load(repo)` returns the model and the device (GPU/CPU) it's on. In `generate` mode it prints the model's text continuation; in `embed` mode it prints the *shape* of the embedding vector (a sanity check — the actual numbers are huge). The function signatures match `infer.py`: `generate(model, device, prompt, ...)` and `embed(model, device, text, ...)`.

**`eval` branch:**

```python
100      elif args.cmd == "eval":
101          from .infer import load
102          from .eval import president_test, major_events_test
103          model, device = load(args.repo)
104          print("=== U.S. presidents (Table 2) ===")
105          for r in president_test(model, device, args.cutoff):
106              flag = "  (past cutoff)" if r["past_cutoff"] else ""
107              ok = "OK" if r["correct"] else "x "
108              print(f"{ok} {r['target_year']} {r['target']:<16} -> {r['prediction']!r}{flag}")
109          print("=== major events (Table 3) ===")
110          for r in major_events_test(model, device, args.cutoff):
111              flag = "  (past cutoff)" if r["past_cutoff"] else ""
112              ok = "OK" if r["correct"] else "x "
113              print(f"{ok} {r['event_year']} {r['answer']:<16} -> {r['prediction']!r}{flag}")
```

Loads the model, then runs two tests from `eval.py`. Each test returns a list of result dicts; the loop prints one line per question with an `OK`/`x ` correctness mark, the year, the expected answer (`:<16` left-pads to 16 chars for alignment), the model's prediction (`!r` shows it with quotes/repr), and a `(past cutoff)` flag when the question falls after the model's knowledge cutoff. This reproduces the paper's Table 2 (presidents) and Table 3 (major events). Note the dispatch covers *both* tests even though the docstring at the top only mentions the president test — read the code, not the comment.

**`push` branch:**

```python
115      elif args.cmd == "push":
116          from .hub import push_dir
117          push_dir(args.repo, args.to, private=args.private, commit_message=args.message)
```

A thin pass-through to `hub.push_dir` (section 3). The local `--repo` dir goes to the `--to` HF repo.

**`alpaca` branch:**

```python
119      elif args.cmd == "alpaca":
120          import json
121          from .eval import alpaca_instructions, alpaca_outputs
122          outs = alpaca_outputs(args.repo, alpaca_instructions(args.n), args.name, backend=args.backend)
123          with open(args.out, "w") as f:
124              json.dump(outs, f, indent=2)
125          print(f"wrote {len(outs)} outputs -> {args.out}")
```

`alpaca_instructions(n)` returns the benchmark prompts (optionally capped at `n`). `alpaca_outputs(...)` runs the model over them and returns a list of answer records, which is written as pretty-printed JSON to `--out`. This JSON is the *stable artifact* — see the FAQ on why the saved outputs matter more than the judge's raw return value.

**`winrate` branch:**

```python
127      elif args.cmd == "winrate":
128          from .eval import alpaca_winrate
129          print(f"LC win rate: {alpaca_winrate(args.model, args.reference):.2f}%")
```

Calls `alpaca_winrate(model_json, reference_json)` and prints the length-controlled win rate to two decimals. This uses the canonical `alpaca_eval` package under the hood (which needs an annotator API key, e.g. `OPENAI_API_KEY` — see implementation-notes §10).

**`figure` branch:**

```python
131      elif args.cmd == "figure":
132          from . import figures
133          if args.kind == "1":
134              figures.figure1(args.run, args.out or "figure1.png")
135          elif args.kind == "2":
136              figures.figure2(args.runs, args.out or "figure2.png")
137          else:
138              figures.figure3(args.results, args.out or "figure3.png")
```

Routes by `--kind` to the matching function in `figures.py` (section 4). `args.out or "figure1.png"` supplies a default filename when `--out` is omitted. Note each kind reads a different input attribute: `--run` for 1, `--runs` for 2, `--results` for 3.

### The script footer

```python
141  if __name__ == "__main__":
142      main()
```

Standard Python idiom: if this file is run directly (`python cli.py`), call `main()`. In practice you invoke it via the installed `chrono` console-script entry point (defined in the project's packaging config), which also calls `main()`.

**Mental model:** `chrono <verb> <flags>` → argparse fills `args` → the matching `elif` does a lazy `from .module import fn` → calls that function → prints output or writes a file. The CLI itself holds almost no logic; it's a switchboard.

---

## 2. `tracking.py` — `RunLogger`

During training, you want a record of how loss falls, how fast tokens are processed, how much GPU memory is used, etc. `RunLogger` writes all of that to a CSV file that the figures later read back. Optionally — and off by default — it mirrors the same numbers live to Weights & Biases (W&B), a web dashboard. The design principle (implementation-notes §9) is that **the CSV is always the source of truth**; W&B is a convenience and never required.

### Docstring and the schema

```python
1  """Training metrics: a rich CSV (always) optionally mirrored to Weights & Biases.
2
3  `output_dir/metrics.csv` is the source of truth for the figures and needs no
4  account. Each row is one event (a train log point or an epoch's val), with
5  nullable columns so train/val rows can carry different fields:
6
7      elapsed_s, stage, epoch, step, split, loss, ppl, lr, grad_norm,
8      tokens_per_sec, gpu_mem_gb
9
10  W&B is an optional live mirror (off unless `wandb.enabled`). A run-level
11  `summary.json` (final val loss per stage, peak VRAM, throughput, config) is
12  written at the end.
13  """
14  import csv
15  import json
16  import os
17  import time
18
19  FIELDS = ["elapsed_s", "stage", "epoch", "step", "split", "loss", "ppl",
20            "lr", "grad_norm", "tokens_per_sec", "gpu_mem_gb"]
```

`FIELDS` is the fixed column order of `metrics.csv`. What each column means:

| field | meaning |
|---|---|
| `elapsed_s` | seconds since the logger started (wall clock) |
| `stage` | curriculum stage name (the SFT pipeline runs in stages) |
| `epoch` | epoch index within the stage |
| `step` | optimizer step count |
| `split` | `"train"` or `"val"` — which curve this row belongs to |
| `loss` | the loss at this point (cross-entropy; see `01-ml-primer.md`) |
| `ppl` | perplexity = `exp(loss)`, an intuitive "how surprised" measure (val rows only) |
| `lr` | learning rate at this step |
| `grad_norm` | gradient norm (size of the update; spikes can signal instability) |
| `tokens_per_sec` | throughput |
| `gpu_mem_gb` | peak GPU memory used, in GB |

The columns are *nullable*: a `train` row fills `lr`/`grad_norm`/`tokens_per_sec`/`gpu_mem_gb`, while a `val` row fills `ppl` and leaves the training-only fields blank. That's why one flat CSV can hold two different kinds of events.

### Constructor — open the CSV, maybe init W&B

```python
23  class RunLogger:
24      def __init__(self, output_dir, wandb_cfg=None, run_config=None):
25          os.makedirs(output_dir, exist_ok=True)
26          self.output_dir = output_dir
27          self._t0 = time.time()
28          # Append so a resumed run (same output_dir, later stages) accumulates the
29          # full curve. Delete metrics.csv to start fresh. Header only on a new file.
30          path = os.path.join(output_dir, "metrics.csv")
31          new = not os.path.exists(path)
32          self._file = open(path, "a", newline="")
33          self._writer = csv.DictWriter(self._file, fieldnames=FIELDS, extrasaction="ignore")
34          if new:
35              self._writer.writeheader()
```

Line 25 makes the output directory if needed. Line 27 records the start time `_t0` so every log row can compute `elapsed_s`. The CSV is opened in **append mode** (`"a"`, line 32): if you resume training into the same `output_dir` (e.g. running a later stage), new rows are added to the existing curve rather than overwriting it. The header is written only when the file is new (lines 31, 34–35) so you don't get a header line in the middle of an appended file. `csv.DictWriter` with `extrasaction="ignore"` means you can pass extra keys to `log()` and they're silently dropped if not in `FIELDS` — robust against typos breaking a long run.

```python
36          self._wandb = None
37          if wandb_cfg and wandb_cfg.get("enabled"):
38              import wandb
39              self._wandb = wandb
40              base = wandb_cfg.get("name") or os.path.basename(output_dir.rstrip("/"))
41              wandb.init(
42                  project=wandb_cfg.get("project", "chrono-instruct"),
43                  name=f"{base}-{time.strftime('%Y%m%d-%H%M%S')}",  # unique per launch -> no overlaid same-name runs
44                  group=base,                                        # ...but still grouped by vintage/output_dir
45                  config=run_config,
46              )
```

W&B is only touched when `wandb_cfg["enabled"]` is true (line 37), and `import wandb` happens inside the `if` so the package isn't even needed otherwise. `base` is a stable label derived from the config name or the output dir's basename (e.g. `base` for a vintage). The crucial detail is line 43: the W&B run *name* gets a timestamp suffix `base-YYYYMMDD-HHMMSS`, making it **unique per launch**, while `group=base` (line 44) keeps all launches of the same vintage grouped together in the dashboard. This fixes a real problem you hit earlier — five same-named runs overlaid on top of each other, impossible to tell apart. Unique names separate them; the group keeps them organized. `config=run_config` snapshots the full training config into W&B for reproducibility.

### `log()` — write one row

```python
48      def log(self, **row):
49          row.setdefault("elapsed_s", round(time.time() - self._t0, 1))
50          self._writer.writerow(row)
51          self._file.flush()
52          if self._wandb:
53              stage, split = row.get("stage", ""), row.get("split", "")
54              self._wandb.log({f"{stage}/{split}_{k}": v for k, v in row.items()
55                               if isinstance(v, (int, float)) and k not in ("step", "epoch", "elapsed_s")})
```

`log(**row)` takes keyword arguments matching the field names. Line 49 auto-fills `elapsed_s` if the caller didn't. Line 50 writes the row; line 51 `flush()` forces it to disk immediately so a crash mid-run still leaves you the curve so far. If W&B is on, lines 53–55 mirror the numeric fields with keys like `stage1/train_loss`, `stage1/val_loss` — prefixing by stage and split so the dashboard groups them sensibly, and skipping bookkeeping fields (`step`, `epoch`, `elapsed_s`) and non-numeric values.

**How `train.py` calls it.** In `train.py`'s `train_stage`, the inner helper `log_point` (lines 77–87) evaluates the validation loss and then emits **two** rows at the *same step*:

```python
82          if run_logger:
83              run_logger.log(stage=name, epoch=epoch, step=step, split="train", loss=round(train_loss, 4),
84                             lr=lr, grad_norm=round(grad_norm, 3), tokens_per_sec=round(tps), gpu_mem_gb=round(mem_gb(), 1))
85              run_logger.log(stage=name, epoch=epoch, step=step, split="val",
86                             loss=round(vloss, 4), ppl=round(math.exp(min(vloss, 20)), 2))
```

This is the key to aligned loss curves: because both the `train` and `val` rows share the same `step`, Figure 1 can plot them on a common x-axis and they start and end together (the docstring on line 78 says exactly this — "Log train AND val at the SAME step (aligned curves, like the paper's Fig 1)"). The `ppl` is `exp(val_loss)`, clamped at `exp(20)` to avoid overflow on a bad early step.

### `summary()` — the end-of-run rollup

```python
57      def summary(self, **data):
58          data["elapsed_s"] = round(time.time() - self._t0, 1)
59          with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
60              json.dump(data, f, indent=2)
61          if self._wandb:
62              self._wandb.summary.update(data)
```

Called once at the end of training to write `summary.json` — final val loss per stage, peak VRAM, throughput, the config, total elapsed time. This is the at-a-glance scorecard for a run (vs `metrics.csv`, which is the full time series). If W&B is on, the same dict updates the run's summary panel.

### `close()`

```python
64      def close(self):
65          self._file.close()
66          if self._wandb:
67              self._wandb.finish()
```

Closes the CSV file handle and tells W&B the run is finished. Always called at the end so buffers flush and the dashboard marks the run complete.

**How `figures.py` reads it back:** the figures open `output_dir/metrics.csv` with `csv.DictReader`, filter rows by `split`, and plot `step` vs `loss` (section 4). The logger writes; the figures read. Nothing in the plotting path touches W&B.

---

## 3. `hub.py` — `push_dir`

Trained model weights are large binary files. They do *not* belong in git (the GitHub repo holds code only). Instead they go to the Hugging Face Hub, which is purpose-built for model storage. This file is the one function that does the upload.

```python
1  """Push a trained vintage to the Hugging Face Hub.
2
3  Used by `chrono push` and, when `push_to_hub.enabled` is set, at the end of
4  training. Needs a write token: run `hf auth login` or set HF_TOKEN. Never
5  hardcode the token (it would leak via git and get auto-revoked).
6  """
7  import os
8
9  from huggingface_hub import HfApi
10
11
12  def push_dir(local_dir, repo_id, private=True, token=None, commit_message=None):
13      api = HfApi(token=token or os.environ.get("HF_TOKEN"))
14      api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
15      api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model",
16                        commit_message=commit_message or "upload checkpoint")
17      print(f"Pushed {local_dir} -> https://huggingface.co/{repo_id}")
```

Line by line:

- **Line 13** — build an `HfApi` client. The write token comes from the `token` argument *or*, if not passed, the `HF_TOKEN` environment variable. There is **no hardcoded token anywhere** — that is the security rule stated in the docstring (lines 4–5): a token committed to git would leak publicly and Hugging Face would auto-revoke it. You authenticate once with `hf auth login` (which stores the token) or export `HF_TOKEN` in your shell. Pushing requires a *write* token specifically (read tokens can download but not upload).
- **Line 14** — create the destination repo as a `model` repo. `private=private` (default `True`) keeps it private; `exist_ok=True` makes it idempotent so re-pushing to an existing repo doesn't error.
- **Lines 15–16** — `upload_folder` uploads the entire `local_dir` (the checkpoint folder: weights + config + tokenizer files) in one call, with the given commit message (default `"upload checkpoint"`). Uploads on the Hub are git-LFS-backed commits under the hood, which is why large binaries are fine here but not in your code repo.
- **Line 17** — print the public URL of the uploaded model.

**Namespace gotcha.** `repo_id` is `namespace/name`. Your Hugging Face username is **not** the same as your GitHub username — code lives under the GitHub account, weights live under the HF account. Make sure `--to` uses the right HF namespace, or the push lands somewhere unexpected (or fails on permissions). Mixing them up is an easy mistake when the two services are open side by side.

`push_dir` is invoked two ways: directly via `chrono push` (cli.py lines 115–117), and automatically at the end of `train.run` when the config sets `push_to_hub.enabled` (off by default — implementation-notes §9).

---

## 4. `figures.py` — Figures 1, 2, 3

This module turns the logged artifacts into the three plots from the paper, using pandas-free plain `csv` reading plus `matplotlib`. (You'll recognize matplotlib from your research scripting.) Each function reads an artifact and writes a PNG.

### Docstring

```python
1  """Reproduce the paper's figures from logged artifacts (matplotlib).
2
3  Figure 1  train/val loss across the 3 SFT stages for one vintage (its metrics.csv).
4  Figure 2  validation loss across stages for several vintages (many metrics.csv).
5  Figure 3  AlpacaEval length-controlled win-rate per vintage (a results JSON,
6            {"1999": 12.59, "2005": 13.19, ...} produced by `chrono winrate`).
7  """
8  import csv
9  import json
10  import os
11  from collections import defaultdict
```

Figures 1 and 2 come from `metrics.csv` files; Figure 3 from a small JSON of win rates. `matplotlib` is *not* imported at the top — it's imported inside the functions that need it (lines 26, 61), the same lazy-import pattern as the CLI, so importing this module is cheap.

### `_read_metrics` — load one run's curves

```python
14  def _read_metrics(run_dir):
15      """Return {(stage, split): ([steps], [losses])} from run_dir/metrics.csv."""
16      series = defaultdict(lambda: ([], []))
17      with open(os.path.join(run_dir, "metrics.csv")) as f:
18          for row in csv.DictReader(f):
19              steps, losses = series[(row["stage"], row["split"])]
20              steps.append(int(row["step"]))
21              losses.append(float(row["loss"]))
22      return series
```

This reads one `metrics.csv` and groups the points by `(stage, split)`. `defaultdict(lambda: ([], []))` means any new key auto-creates an empty `([steps], [losses])` pair. For each CSV row (line 18), it looks up the matching pair and appends the integer `step` and float `loss` (lines 19–21). The result is a dict like `{("stage1","train"): ([0,20,40,...], [3.1,2.4,...]), ("stage1","val"): (...), ...}` — exactly the structure both Figures 1 and 2 plot from. Note it reads only `stage`, `split`, `step`, `loss`; the other columns are ignored here.

### `_stage_axes` — one subplot per stage

```python
25  def _stage_axes(stages):
26      import matplotlib.pyplot as plt
27      fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4), squeeze=False)
28      return fig, axes[0]
```

Creates a row of subplots, one per curriculum stage, each 5 wide × 4 tall. `squeeze=False` forces `axes` to stay a 2-D array even with one stage (otherwise matplotlib returns a bare axis and the zipping below would break); `axes[0]` then returns the single row as a 1-D list. This helper is shared by Figures 1 and 2 so both lay stages out side by side.

### `figure1` — one vintage's train+val loss

```python
31  def figure1(run_dir, out="figure1.png"):
32      series = _read_metrics(run_dir)
33      stages = sorted({s for s, _ in series})
34      fig, axes = _stage_axes(stages)
35      for ax, stage in zip(axes, stages):
36          for split in ("train", "val"):
37              if (stage, split) in series:
38                  steps, losses = series[(stage, split)]
39                  ax.plot(steps, losses, label=split)
40          ax.set_title(stage); ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
41      fig.tight_layout(); fig.savefig(out, dpi=150)
42      print("wrote", out)
```

Reads one run's metrics (line 32), gets the sorted list of stage names (line 33), and makes one subplot per stage. In each subplot it draws both the `train` and `val` curves (lines 36–39) — this is where the aligned-step logging from section 2 pays off: the two lines share an x-axis and line up. Titles, axis labels, and a legend are set per subplot (line 40), then the figure is saved at 150 dpi. This is the paper's Figure 1 style: how loss falls within each SFT stage for a single vintage, with train vs val shown together (val above train is the normal generalization gap). See `02-paper-and-research-framing.md` for what the gap means scientifically.

### `figure2` — validation loss across vintages

```python
45  def figure2(run_dirs, out="figure2.png", labels=None):
46      labels = labels or [os.path.basename(d.rstrip("/")) for d in run_dirs]
47      per = {d: _read_metrics(d) for d in run_dirs}
48      stages = sorted({s for d in run_dirs for (s, sp) in per[d] if sp == "val"})
49      fig, axes = _stage_axes(stages)
50      for ax, stage in zip(axes, stages):
51          for d, lab in zip(run_dirs, labels):
52              if (stage, "val") in per[d]:
53                  steps, losses = per[d][(stage, "val")]
54                  ax.plot(steps, losses, label=lab)
55          ax.set_title(stage); ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.legend()
56      fig.tight_layout(); fig.savefig(out, dpi=150)
57      print("wrote", out)
```

Takes *several* run dirs (one per vintage). Line 46 derives a label per run from its directory basename unless explicit `labels` are passed. Line 47 reads every run's metrics into `per`. Line 48 collects the set of stages that have validation data across all runs. Then, per stage subplot, it overlays the **val** curve of each vintage (lines 51–54), one line per vintage, labeled. Unlike Figure 1, this shows *only* val loss (line 55 axis label `"val loss"`) — the point is to compare vintages against each other, not train vs val. This is the paper's Figure 2: validation loss by stage across the chronological vintages.

### `figure3` — AlpacaEval win-rate bar chart

```python
60  def figure3(results_json, out="figure3.png"):
61      import matplotlib.pyplot as plt
62      with open(results_json) as f:
63          data = json.load(f)
64      years = sorted(data, key=int)
65      vals = [data[y] for y in years]
66      fig, ax = plt.subplots(figsize=(7, 4))
67      ax.bar(years, vals)
68      ax.set_xlabel("model year"); ax.set_ylabel("LC win rate (%)")
69      ax.set_title("AlpacaEval (LC): ChronoGPT-Instruct vs Qwen-1.5-1.8B-Chat")
70      for x, v in zip(years, vals):
71          ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
72      fig.tight_layout(); fig.savefig(out, dpi=150)
73      print("wrote", out)
```

Reads a small JSON mapping year → win rate (e.g. `{"1999": 12.59, "2005": 13.19, ...}`, produced by collecting `chrono winrate` results). Line 64 sorts the years numerically (`key=int` — they're string keys); line 65 pulls the values in that order. Line 67 draws a bar chart, lines 68–69 label the axes and set the title naming the matchup (ChronoGPT-Instruct vs the Qwen-1.5-1.8B-Chat reference). Lines 70–71 annotate each bar with its value to two decimals just above the bar (`va="bottom"`). This is the paper's Figure 3: instruction-following quality per vintage.

**Mechanics summary:** Figures 1/2 = `csv.DictReader` → group by `(stage, split)` → one matplotlib subplot per stage → `ax.plot`. Figure 3 = `json.load` → sort by year → `ax.bar` + text labels. All save a PNG at 150 dpi and print the path.

---

## 5. `__init__.py` — the package marker

```python
1  """Chrono-Instruct: a clean replication of the Instruct ChronoGPT SFT pipeline."""
2  __version__ = "0.0.1"
```

This file makes `src/chrono_instruct/` an importable Python package and does nothing else. It carries a one-line description and a `__version__` string (`0.0.1`). It deliberately does *not* import the heavy submodules at package level — that's why `import chrono_instruct` is cheap and why the CLI can lazily import `train`/`infer`/`eval` only when needed.

---

## 6. Command → code → artifact map

| You run | argparse branch (cli.py) | calls | produces |
|---|---|---|---|
| `chrono inspect --dataset ...` | lines 68–85 | `data.load_raw`, `data.source_counts` | printed sample row + source counts (vs Table 1) |
| `chrono train --config X.yaml` | lines 87–89 | `train.run(cfg)` → `RunLogger` | `metrics.csv`, `summary.json`, `config.yaml`, checkpoints in `output_dir` |
| `chrono infer --repo R --mode generate --text T` | lines 91–98 | `infer.load`, `infer.generate` | printed text continuation |
| `chrono infer --repo R --mode embed --text T` | lines 91–98 | `infer.load`, `infer.embed` | printed embedding shape |
| `chrono eval --repo R --cutoff Y` | lines 100–113 | `infer.load`, `eval.president_test`, `eval.major_events_test` | printed Table 2 + Table 3 results |
| `chrono push --repo D --to ID [--private] [--message M]` | lines 115–117 | `hub.push_dir` | model uploaded to `https://huggingface.co/ID` |
| `chrono alpaca --repo R --backend chrono|hf --name N --out O.json [--n K]` | lines 119–125 | `eval.alpaca_instructions`, `eval.alpaca_outputs` | `O.json` of generated answers |
| `chrono winrate --model M.json --reference Ref.json` | lines 127–129 | `eval.alpaca_winrate` | printed LC win-rate % |
| `chrono figure --kind 1 --run D [--out P]` | lines 131–134 | `figures.figure1` | `figure1.png` (train+val loss, one vintage) |
| `chrono figure --kind 2 --runs D1 D2 ... [--out P]` | lines 131,135–136 | `figures.figure2` | `figure2.png` (val loss across vintages) |
| `chrono figure --kind 3 --results J.json [--out P]` | lines 131,137–138 | `figures.figure3` | `figure3.png` (win-rate bars) |

---

## 7. Mini-FAQ

**Q: Why is the CSV the source of truth instead of Weights & Biases?**
Because the figures must be reproducible by anyone, with no account and no network. `metrics.csv` is a plain local file written on every log step (and flushed immediately), so the plots always work offline. W&B is an *optional live mirror* (`wandb.enabled`, off by default — implementation-notes §9); turning it off changes nothing about the figures. See `tracking.py` lines 3–4 and 30–35.

**Q: Why give each W&B run a unique timestamped name (`base-YYYYMMDD-HHMMSS`)?**
You previously saw five same-named runs overlaid into an unreadable mess. The fix (tracking.py line 43) appends a launch timestamp so every run is distinct, while `group=base` (line 44) still clusters launches of the same vintage in the dashboard. Distinct names, organized groups.

**Q: Why do model checkpoints go to Hugging Face and not git?**
Weights are large binaries; git/GitHub is for code. Hugging Face is built for model storage (LFS-backed). `hub.push_dir` uploads the checkpoint folder there. Remember the namespace gotcha: your HF username differs from your GitHub username, so set `--to` to the correct HF namespace. Tokens are never hardcoded — use `hf auth login` or `HF_TOKEN` (hub.py lines 4–5, 13).

**Q: How do I regenerate the figures from a finished run?**
The artifacts persist after training, so just re-run the `figure` command — no retraining needed. Figure 1: `chrono figure --kind 1 --run runs/base/`. Figure 2 over several vintages: `chrono figure --kind 2 --runs runs/1999 runs/2005 ...`. Figure 3: produce per-vintage win rates with `chrono alpaca` + `chrono winrate`, collect them into a `{year: winrate}` JSON, then `chrono figure --kind 3 --results winrates.json`.

**Q: Why are the heavy imports inside the branches/functions instead of at the top of the file?**
So lightweight commands stay fast and dependency-light. `chrono inspect` and `chrono figure` shouldn't have to load PyTorch and the model code (slow, GPU-oriented). Each CLI branch does `from .module import ...` only when that verb runs (cli.py), and `figures.py` imports matplotlib only inside the plotting functions.
