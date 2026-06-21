"""Training metrics: always written to CSV, optionally mirrored to Weights & Biases.

The CSV (output_dir/metrics.csv) is the source of truth for Figures 1-2 and needs
no account. W&B is an optional live mirror, off unless `wandb.enabled` is set in
the config. Both are deliberately decoupled so figures never depend on W&B.
"""
import csv
import os


class RunLogger:
    def __init__(self, output_dir, wandb_cfg=None, run_config=None):
        os.makedirs(output_dir, exist_ok=True)
        self._file = open(os.path.join(output_dir, "metrics.csv"), "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=["stage", "step", "split", "loss"])
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

    def log(self, stage, step, split, loss):
        self._writer.writerow({"stage": stage, "step": step, "split": split, "loss": loss})
        self._file.flush()
        if self._wandb:
            self._wandb.log({f"{stage}/{split}_loss": loss})

    def close(self):
        self._file.close()
        if self._wandb:
            self._wandb.finish()
