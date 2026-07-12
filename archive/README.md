# archive/

Off-main-path scripts kept for reference, not part of the default workflow.

- `slurm_array.sbatch` — SLURM job-array launcher for training vintages in parallel
  on a cluster. The default replication runs the vintages sequentially on one GPU
  via `scripts/train_all_vintages.sh`; use this only on a SLURM cluster. It calls
  `scripts/make_vintage_config.py` + `chrono train` per array index.
