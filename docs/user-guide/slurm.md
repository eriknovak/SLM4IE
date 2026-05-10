---
title: SLURM / HPC
---

# SLURM / HPC

Batch scripts for cluster execution live under
[`slurm/`](https://github.com/eriknovak/SLM4IE/blob/main/slurm/):

```bash
sbatch slurm/tokenizer_train.sbatch
sbatch slurm/train.sbatch
sbatch slurm/evaluate.sbatch
sbatch slurm/generate.sbatch
```

Each batch script wraps the corresponding `scripts/` entry point and
sets the SLURM directives (partition, GPU count, walltime, output
log path). Adjust the directives at the top of each `.sbatch` file to
match your cluster's resource policies.

!!! tip "Authentication on compute nodes"
    Set up
    [HuggingFace authentication](../getting-started/huggingface-auth.md)
    on the compute nodes too. The unified `hf` CLI stores tokens under
    `~/.cache/huggingface/`, which is typically shared across login and
    compute nodes via the user's home directory.
