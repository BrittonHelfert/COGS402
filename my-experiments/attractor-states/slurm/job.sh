#!/bin/bash
# SLURM job for one attractor-states organism × model run.
# Called by submit_all.sh — do not invoke directly.
#
# Required env vars (set by submit_all.sh via --export):
#   ORGANISM_ARG   e.g. "--organism em_bad_medical_advice" or "--control"
#   MODEL          e.g. "llama31_8b"
#   GPU_COUNT      1 or 2
#   TURNS          number of conversation turns (default 30)
#   ALLOC          your Sockeye allocation ID

#SBATCH --job-name=attractor
#SBATCH --account=${ALLOC}
#SBATCH --partition=gpu
#SBATCH --constraint=gpu_mem_32
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=${TIME_LIMIT:-04:00:00}
#SBATCH --output=logs/slurm-%j-%x.out
#SBATCH --error=logs/slurm-%j-%x.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────

# TODO: adjust module names to match what's available on Sockeye
module load gcc python/3.11

# TODO: adjust venv path
source ~/venvs/attractor/bin/activate

# Offline mode — compute nodes have no internet
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# TODO: set this to your project/scratch storage where you ran download_models.py
export HF_HOME=${HF_HOME:-/arc/project/${ALLOC}/hf_cache}

# Confirm GPU + fp16 (V100 does NOT support bfloat16)
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('fp16 support:', torch.cuda.is_bf16_supported() == False or True)
"

# ── Run ────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."

echo "Organism arg: ${ORGANISM_ARG}"
echo "Model:        ${MODEL}"
echo "Turns:        ${TURNS:-30}"

python scripts/run_organism.py \
    ${ORGANISM_ARG} \
    --model "${MODEL}" \
    --turns "${TURNS:-30}"
