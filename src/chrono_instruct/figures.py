"""Reproduce the paper's figures from logged artifacts (matplotlib).

Figure 1  train/val loss across the 3 SFT stages for one vintage (its metrics.csv).
Figure 2  validation loss across stages for several vintages (many metrics.csv).
Figure 3  AlpacaEval length-controlled win-rate per vintage (a results JSON,
          {"1999": 12.59, "2005": 13.19, ...} produced by `chrono winrate`).
"""
import csv
import json
import os
from collections import defaultdict


def _read_metrics(run_dir):
    """Return {(stage, split): ([steps], [losses])} from run_dir/metrics.csv."""
    series = defaultdict(lambda: ([], []))
    with open(os.path.join(run_dir, "metrics.csv")) as f:
        for row in csv.DictReader(f):
            steps, losses = series[(row["stage"], row["split"])]
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return series


def _stage_axes(stages):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4), squeeze=False)
    return fig, axes[0]


def figure1(run_dir, out="figure1.png"):
    series = _read_metrics(run_dir)
    stages = sorted({s for s, _ in series})
    fig, axes = _stage_axes(stages)
    for ax, stage in zip(axes, stages):
        for split in ("train", "val"):
            if (stage, split) in series:
                steps, losses = series[(stage, split)]
                ax.plot(steps, losses, label=split)
        ax.set_title(stage); ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)


def figure2(run_dirs, out="figure2.png", labels=None):
    labels = labels or [os.path.basename(d.rstrip("/")) for d in run_dirs]
    per = {d: _read_metrics(d) for d in run_dirs}
    stages = sorted({s for d in run_dirs for (s, sp) in per[d] if sp == "val"})
    fig, axes = _stage_axes(stages)
    for ax, stage in zip(axes, stages):
        for d, lab in zip(run_dirs, labels):
            if (stage, "val") in per[d]:
                steps, losses = per[d][(stage, "val")]
                ax.plot(steps, losses, label=lab)
        ax.set_title(stage); ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)


def figure3(results_json, out="figure3.png"):
    import matplotlib.pyplot as plt
    with open(results_json) as f:
        data = json.load(f)
    years = sorted(data, key=int)
    vals = [data[y] for y in years]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(years, vals)
    ax.set_xlabel("model year"); ax.set_ylabel("LC win rate (%)")
    ax.set_title("AlpacaEval (LC): ChronoGPT-Instruct vs Qwen-1.5-1.8B-Chat")
    for x, v in zip(years, vals):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)
