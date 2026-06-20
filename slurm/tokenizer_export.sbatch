#!/bin/bash
#SBATCH --job-name=slm4ie-tokenizer-export          # Name of the job
#SBATCH --account=[account_name]                    # Account name
#SBATCH --output=slurm/logs/%j-tokenizer-export.out # Standard output file (%j = job ID)
#SBATCH --error=slurm/logs/%j-tokenizer-export.out  # Standard error file (same as output file)
#SBATCH --time=01:00:00                             # Time limit (format: HH:MM:SS)
#SBATCH --partition=cpu                             # Export is CPU-bound and light
#SBATCH --cpus-per-task=4                           # Export is single-process; a few cores suffice
#SBATCH --nodes=1                                   # Number of nodes to allocate
#SBATCH --ntasks=1                                  # Total number of tasks (processes)
#SBATCH --ntasks-per-node=1                         # Number of tasks per node

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
echo "# Export the trained tokenizers as HuggingFace dirs"
echo "# ==============================================="
echo ""

# Write a HuggingFace tokenizer directory into every trained artifact, the
# bridge to LM pretraining. The tokenize extra pulls in tokenizers, transformers,
# and sentencepiece.
uv run --extra tokenize python scripts/tokenizers/export.py --all
