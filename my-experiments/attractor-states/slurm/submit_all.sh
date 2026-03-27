#!/bin/bash
# Submit one SLURM job per organism × model pair, plus one per base model control.
#
# Usage:
#   bash slurm/submit_all.sh --alloc st-yourpi-1              # submit all
#   bash slurm/submit_all.sh --alloc st-yourpi-1 --dry-run    # print jobs without submitting
#   bash slurm/submit_all.sh --alloc st-yourpi-1 --turns 20   # override turns
#
# Each job gets a time limit based on model size:
#   1B  models → 2h
#   2–4B models → 3h
#   7–9B models → 4h

set -euo pipefail

# ── Argument parsing ───────────────────────────────────────────────────────────

ALLOC=""
DRY_RUN=0
TURNS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alloc)   ALLOC="$2";   shift 2 ;;
        --turns)   TURNS="$2";   shift 2 ;;
        --dry-run) DRY_RUN=1;    shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ALLOC" ]]; then
    echo "ERROR: --alloc is required (your Sockeye allocation ID, e.g. st-yourpi-1)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${ROOT}/logs/submission_manifest.txt"

mkdir -p "${ROOT}/logs"
[[ $DRY_RUN -eq 0 ]] && > "${MANIFEST}"  # clear manifest on real run

# ── Time limit lookup ──────────────────────────────────────────────────────────

time_limit_for_gpu_count() {
    local gpu_count=$1
    # gpu_count=1 for all our models; kept as hook for future 32B runs
    echo "04:00:00"
}

# Heuristic: infer time limit from model key name
time_limit_for_model() {
    local model_key=$1
    case "$model_key" in
        *1b*)  echo "02:00:00" ;;
        *1b7*) echo "02:00:00" ;;
        *4b*)  echo "03:00:00" ;;
        *7b*)  echo "04:00:00" ;;
        *8b*)  echo "04:00:00" ;;
        *9b*)  echo "04:00:00" ;;
        *)     echo "04:00:00" ;;
    esac
}

# ── Submit helper ──────────────────────────────────────────────────────────────

SUBMITTED=0
SKIPPED=0

submit_job() {
    local organism_arg="$1"   # e.g. "--organism em_bad_medical_advice" or "--control"
    local model_key="$2"
    local gpu_count="$3"
    local job_label="$4"      # human-readable name for manifest
    local time_limit
    time_limit=$(time_limit_for_model "$model_key")

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] ${job_label}  model=${model_key}  gpus=${gpu_count}  time=${time_limit}"
        return
    fi

    job_id=$(sbatch \
        --export="ORGANISM_ARG=${organism_arg},MODEL=${model_key},GPU_COUNT=${gpu_count},TURNS=${TURNS},ALLOC=${ALLOC},TIME_LIMIT=${time_limit}" \
        --gres="gpu:${gpu_count}" \
        --time="${time_limit}" \
        "${SCRIPT_DIR}/job.sh" \
        | awk '{print $NF}')

    echo "${job_id}  ${job_label}" >> "${MANIFEST}"
    echo "  Submitted job ${job_id}: ${job_label}"
    (( SUBMITTED++ )) || true
}

# ── Parse organism configs ─────────────────────────────────────────────────────

echo "=== Attractor-States Submit All ==="
echo "Allocation: ${ALLOC}"
echo "Turns:      ${TURNS}"
[[ $DRY_RUN -eq 1 ]] && echo "Mode:       DRY RUN" || echo "Mode:       LIVE"
echo ""

# Organism jobs
echo "--- Organism runs ---"
for org_file in "${ROOT}/organisms"/*.yaml; do
    org_name=$(grep '^name:' "$org_file" | awk '{print $2}')

    # Extract model keys from finetuned_models block
    # Lines of the form "  model_key:" (two leading spaces, ends with colon)
    model_keys=$(grep -E '^  [a-z].*:$' "$org_file" | sed 's/://;s/^  //')

    for model_key in $model_keys; do
        # Look up gpu_count from model config
        model_cfg="${ROOT}/models/${model_key}.yaml"
        if [[ ! -f "$model_cfg" ]]; then
            echo "  WARNING: no model config for key '${model_key}' — skipping"
            (( SKIPPED++ )) || true
            continue
        fi
        gpu_count=$(grep '^gpu_count:' "$model_cfg" | awk '{print $2}')

        submit_job \
            "--organism ${org_name}" \
            "${model_key}" \
            "${gpu_count}" \
            "${org_name}__${model_key}"
    done
done

# Control jobs (base model, no adapter)
echo ""
echo "--- Control runs (no adapter) ---"
for model_cfg in "${ROOT}/models"/*.yaml; do
    model_key=$(grep '^name:' "$model_cfg" | awk '{print $2}')
    gpu_count=$(grep '^gpu_count:' "$model_cfg" | awk '{print $2}')

    submit_job \
        "--control" \
        "${model_key}" \
        "${gpu_count}" \
        "control__${model_key}"
done

echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete."
else
    echo "Submitted ${SUBMITTED} jobs. Manifest: ${MANIFEST}"
    [[ $SKIPPED -gt 0 ]] && echo "Skipped ${SKIPPED} (missing model configs)."
fi
