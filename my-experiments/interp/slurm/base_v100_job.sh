#!/bin/bash
# Base SLURM script for V100 32GB GPU jobs on UBC ARC Sockeye.
# Copy and customize for each experiment.
#
# Usage:
#   sbatch slurm/base_v100_job.sh

#SBATCH --job-name=interp
#SBATCH --account=YOUR_ALLOCATION    # TODO: set your allocation
#SBATCH --partition=gpu
#SBATCH --constraint=gpu_mem_32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# --- Environment ---
module load gcc python/3.11  # TODO: check available modules with `module avail`
source ~/venvs/interp/bin/activate  # TODO: adjust path to your venv

# Confirm CUDA + fp16 (V100 does NOT support bfloat16)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# --- Job ---
# Override SCRIPT and any args before submitting, e.g.:
#   SCRIPT=my-experiments/interp/phase_1_israeli_dishes/sae/identify_features.py
python ${SCRIPT:-echo "ERROR: set SCRIPT env var"} ${ARGS:-}
