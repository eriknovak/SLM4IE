#!/bin/bash
#SBATCH --job-name=slm4ie-tokenizer-train          # Name of the job
#SBATCH --account=[account_name]                   # Account name
#SBATCH --output=slurm/logs/%j-tokenizer-train.out # Standard output file (%j = job ID)
#SBATCH --error=slurm/logs/%j-tokenizer-train.out  # Standard error file (same as output file)
#SBATCH --time=08:00:00                            # Time limit (format: HH:MM:SS)
#SBATCH --partition=cpu                            # Tokenizer training is CPU-bound
#SBATCH --cpus-per-task=16                         # Cores for the parallel sweep
#SBATCH --nodes=1                                  # Number of nodes to allocate
#SBATCH --ntasks=1                                 # Total number of tasks (processes)
#SBATCH --ntasks-per-node=1                        # Number of tasks per node

set -e # exit on error

echo "# ==============================================="
echo "# Job information"
echo "# ==============================================="
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "NODELIST=${SLURM_NODELIST}"
echo "Current working directory: $(pwd)"
echo "Start time: $(date)"
echo ""

echo "# ==============================================="
echo "# Train the tokenizer comparison sweep"
echo "# ==============================================="
echo ""

# Train every tokenizer x vocab-size run in parallel. The tokenize extra pulls
# in tokenizers, sentencepiece, and mlflow.
uv run --extra tokenize python scripts/tokenizers/train.py --all --max-workers "${SLURM_CPUS_PER_TASK:-1}"
