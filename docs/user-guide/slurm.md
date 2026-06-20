---
title: SLURM / HPC
---

# SLURM / HPC

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

The `slurm/` directory holds SLURM batch scripts for running the heavy stages
(tokenizer training and analysis, model training, evaluation, synthetic
generation) on an HPC cluster via `sbatch`: `slurm/tokenizer_train.sbatch`,
`slurm/tokenizer_analyze.sbatch`, `slurm/train.sbatch`, `slurm/evaluate.sbatch`,
and `slurm/generate.sbatch`.

**Entry point:** [`slurm/`](https://github.com/eriknovak/SLM4IE/tree/main/slurm)
Each script wraps the corresponding `uv run` command.
