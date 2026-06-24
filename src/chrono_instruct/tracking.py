"""Training metrics: a rich CSV (always) optionally mirrored to Weights & Biases.

`output_dir/metrics.csv` is the source of truth for the figures and needs no
account. Each row is one event (a train log point or an epoch's val), with
nullable columns so train/val rows can carry different fields:

    elapsed_s, stage, epoch, step, split, loss, ppl, lr, grad_norm,
    tokens_per_sec, gpu_mem_gb

W&B is an optional live mirror (off unless `wandb.enabled`). A run-level
`summary.json` (final val loss per stage, peak VRAM, throughput, config) is
written at the end.
"""
import csv
import json
import os
import time

FIELDS = ["elapsed_s", "stage", "epoch", "step", "split", "loss", "ppl",
          "lr", "grad_norm", "tokens_per_sec", "gpu_mem_gb"]


class RunLogger:
    def __init__(self, output_dir, wandb_cfg=None, run_config=None):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._t0 = time.time()
        # Append so a resumed run (same output_dir, later stages) accumulates the
        # full curve. Delete metrics.csv to start fresh. Header only on a new file.
        path = os.path.join(output_dir, "metrics.csv")
        new = not os.path.exists(path)
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            self._writer.writeheader()
        self._wandb = None
        if wandb_cfg and wandb_cfg.get("enabled"):
            import wandb
            self._wandb = wandb
            wandb.init(
                project=wandb_cfg.get("project", "chrono-instruct"),
                name=wandb_cfg.get("name") or os.path.basename(output_dir.rstrip("/")),
                config=run_config,
            )

    def log(self, **row):
        row.setdefault("elapsed_s", round(time.time() - self._t0, 1))
        self._writer.writerow(row)
        self._file.flush()
        if self._wandb:
            stage, split = row.get("stage", ""), row.get("split", "")
            self._wandb.log({f"{stage}/{split}_{k}": v for k, v in row.items()
                             if isinstance(v, (int, float)) and k not in ("step", "epoch", "elapsed_s")})

    def summary(self, **data):
        data["elapsed_s"] = round(time.time() - self._t0, 1)
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(data, f, indent=2)
        if self._wandb:
            self._wandb.summary.update(data)

    def close(self):
        self._file.close()
        if self._wandb:
            self._wandb.finish()
