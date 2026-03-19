#!/bin/bash
#SBATCH --job-name=slm4ie-tokenizer-train          # Name of the job
#SBATCH --account=[account_name]                   # Account name
#SBATCH --output=slurm/logs/%j-tokenizer-train.out # Standard output file (%j = job ID)
#SBATCH --error=slurm/logs/%j-tokenizer-train.out  # Standard error file (same as output file)
#SBATCH --time=04:00:00                            # Time limit (format: HH:MM:SS)
#SBATCH --partition=cpu                            # Partition (queue) to use
#SBATCH --gres=cpu:1                               # Number of CPUs per node
#SBATCH --nodes=1                                  # Number of nodes to allocate
#SBATCH --ntasks=1                                 # Total number of tasks (processes)
#SBATCH --ntasks-per-node=1                        # Number of tasks per node

set -e # exit on error

echo "# ==============================================="
echo "# Job information"
echo "# ==============================================="
echo ""

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=50202

# Print some information about the job
echo "Running on host: $(hostname)"
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "NODELIST="${SLURM_NODELIST}
echo "Current working directory: $(pwd)"
echo "Start time: $(date)"
echo ""

echo "# ==============================================="
echo "# Make sure GPU is available"
echo "# ==============================================="
echo ""

# Print GPU info for debugging
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

echo ""

echo "# ==============================================="
echo "# Run tokenizer training"
echo "# ==============================================="
echo ""

# Run tokenizer training
uv run python scripts/tokenizers/train.py
