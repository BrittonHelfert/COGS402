#!/bin/bash
# SLURM job for one attractor-states organism × model run.
# Called by submit_all.sh — do not invoke directly.
#
# Required env vars (set by submit_all.sh via --export):
#   ORGANISM_ARG   e.g. "--organism em_bad_medical_advice" or "--control"
#   MODEL          e.g. "llama31_8b"
#   GPU_COUNT      1 or 2
#   TURNS          number of conversation turns (default 30)
#   EXPERIMENT_DIR path to shared experiment output dir

# Static SBATCH defaults — account, gpus, and time are set via command-line
# flags in submit_all.sh (sbatch doesn't expand shell variables in #SBATCH lines).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Note: log paths are relative to SLURM_SUBMIT_DIR (where sbatch was called from).
# submit_all.sh calls sbatch from the project root, so logs/ resolves correctly.
#SBATCH --output=logs/slurm-%j-%x.out
#SBATCH --error=logs/slurm-%j-%x.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────

# cd to project root (SLURM copies the script to a spool dir, so $0 is unreliable)
ROOT="${SLURM_SUBMIT_DIR%/slurm}"
cd "$ROOT"

# uv manages the Python interpreter + venv (pyproject.toml in project root)
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/scratch/st-singha53-1/bhelfert/.cache/uv

# Offline mode — compute nodes have no internet
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1

export HF_HOME=/scratch/st-singha53-1/bhelfert/hf_cache

# Confirm GPU + fp16 (V100 does NOT support bfloat16)
uv run python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('bf16 supported:', torch.cuda.is_bf16_supported())
"

# ── Run ────────────────────────────────────────────────────────────────────────

echo "Organism arg: ${ORGANISM_ARG}"
echo "Model:        ${MODEL}"
echo "Turns:        ${TURNS:-30}"

uv run python scripts/run_organism.py \
    ${ORGANISM_ARG} \
    --model "${MODEL}" \
    --turns "${TURNS:-30}" \
    ${EXPERIMENT_DIR:+--experiment-dir "${EXPERIMENT_DIR}"}
