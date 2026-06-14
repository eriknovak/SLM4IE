#!/bin/bash
#SBATCH --job-name=slm4ie-tokenizer-analyze          # Name of the job
#SBATCH --account=[account_name]                     # Account name
#SBATCH --output=slurm/logs/%j-tokenizer-analyze.out # Standard output file (%j = job ID)
#SBATCH --error=slurm/logs/%j-tokenizer-analyze.out  # Standard error file (same as output file)
#SBATCH --time=04:00:00                              # Time limit (format: HH:MM:SS)
#SBATCH --partition=cpu                              # Evaluation is CPU-bound
#SBATCH --cpus-per-task=8                            # Cores for the parallel evaluation
#SBATCH --nodes=1                                    # Number of nodes to allocate
#SBATCH --ntasks=1                                   # Total number of tasks (processes)
#SBATCH --ntasks-per-node=1                          # Number of tasks per node

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
echo "# Evaluate the trained tokenizers"
echo "# ==============================================="
echo ""

# Run the six metrics over every trained artifact and write the comparison
# report. Set MLFLOW_TRACKING_URI (and mlflow.enabled in the config) to log.
uv run --extra tokenize python scripts/tokenizers/analyze.py --all --max-workers "${SLURM_CPUS_PER_TASK:-1}"
