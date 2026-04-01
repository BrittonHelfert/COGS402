#!/bin/bash
# Submit one SLURM job per organism × model pair, plus one per base model control.
#
# Usage:
#   bash slurm/submit_all.sh --experiment-name initial              # submit all
#   bash slurm/submit_all.sh --experiment-name initial --dry-run    # print without submitting
#   bash slurm/submit_all.sh --experiment-name initial --turns 20   # override turns
#   bash slurm/submit_all.sh --experiment-name initial --model llama31_8b --model qwen25_7b
#   bash slurm/submit_all.sh --experiment-name initial --overwrite-latest  # reuse last dir
#   bash slurm/submit_all.sh --experiment-name initial --exclude-organism persona_misalignment
#   bash slurm/submit_all.sh --experiment-name initial --exclude-model llama32_1b
#
# Each job gets a time limit based on model size:
#   1B  models → 2h
#   2–4B models → 3h
#   7–9B models → 4h

set -euo pipefail

# ── Argument parsing ───────────────────────────────────────────────────────────

ALLOC="st-singha53-1"
DRY_RUN=0
TURNS=30
EXPERIMENT_NAME=""
OVERWRITE_LATEST=0
ONLY_ORGANISMS=()
ONLY_MODELS=()
EXCLUDE_ORGANISMS=()
EXCLUDE_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --turns)             TURNS="$2";                shift 2 ;;
        --experiment-name)   EXPERIMENT_NAME="$2";      shift 2 ;;
        --overwrite-latest)  OVERWRITE_LATEST=1;        shift   ;;
        --dry-run)           DRY_RUN=1;                 shift   ;;
        --organism)          ONLY_ORGANISMS+=("$2");    shift 2 ;;
        --model)             ONLY_MODELS+=("$2");       shift 2 ;;
        --exclude-organism)  EXCLUDE_ORGANISMS+=("$2"); shift 2 ;;
        --exclude-model)     EXCLUDE_MODELS+=("$2");    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "ERROR: --experiment-name is required (e.g. --experiment-name initial)"
    exit 1
fi

# Filtering helpers
in_array() {
    local needle="$1"; shift
    for item in "$@"; do [[ "$item" == "$needle" ]] && return 0; done
    return 1
}

should_skip_organism() {
    local name="$1"
    if [[ ${#ONLY_ORGANISMS[@]} -gt 0 ]] && ! in_array "$name" "${ONLY_ORGANISMS[@]}"; then
        return 0
    fi
    if [[ ${#EXCLUDE_ORGANISMS[@]} -gt 0 ]] && in_array "$name" "${EXCLUDE_ORGANISMS[@]}"; then
        return 0
    fi
    return 1
}

should_skip_model() {
    local name="$1"
    if [[ ${#ONLY_MODELS[@]} -gt 0 ]] && ! in_array "$name" "${ONLY_MODELS[@]}"; then
        return 0
    fi
    if [[ ${#EXCLUDE_MODELS[@]} -gt 0 ]] && in_array "$name" "${EXCLUDE_MODELS[@]}"; then
        return 0
    fi
    return 1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${ROOT}/logs/submission_manifest.txt"

mkdir -p "${ROOT}/logs"
[[ $DRY_RUN -eq 0 ]] && > "${MANIFEST}"  # clear manifest on real run

# ── Experiment output directory ────────────────────────────────────────────────

CONVERSATIONS_DIR="${ROOT}/conversations"
mkdir -p "${CONVERSATIONS_DIR}"

if [[ $OVERWRITE_LATEST -eq 1 ]]; then
    EXPERIMENT_DIR=$(ls -dt "${CONVERSATIONS_DIR}/${EXPERIMENT_NAME}_"* 2>/dev/null | head -1)
    if [[ -z "$EXPERIMENT_DIR" ]]; then
        echo "ERROR: --overwrite-latest: no existing dir found matching '${CONVERSATIONS_DIR}/${EXPERIMENT_NAME}_*'"
        exit 1
    fi
    echo "Overwriting latest experiment dir: ${EXPERIMENT_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_DIR="${CONVERSATIONS_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}"
    [[ $DRY_RUN -eq 0 ]] && mkdir -p "${EXPERIMENT_DIR}"
fi

# ── Time limit lookup ──────────────────────────────────────────────────────────

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
        echo "[DRY RUN] ${job_label}  model=${model_key}  gpus=${gpu_count}  time=${time_limit}  dir=${EXPERIMENT_DIR}/${job_label}"
        return
    fi

    job_id=$(sbatch \
        --job-name="attractor" \
        --account="${ALLOC}-gpu" \
        --gpus="${gpu_count}" \
        --constraint=gpu_mem_32 \
        --time="${time_limit}" \
        --export="ORGANISM_ARG=${organism_arg},MODEL=${model_key},TURNS=${TURNS},EXPERIMENT_DIR=${EXPERIMENT_DIR}" \
        "${SCRIPT_DIR}/job.sh" \
        | awk '{print $NF}')

    echo "${job_id}  ${job_label}" >> "${MANIFEST}"
    echo "  Submitted job ${job_id}: ${job_label}"
    (( SUBMITTED++ )) || true
}

# ── Parse organism configs ─────────────────────────────────────────────────────

echo "=== Attractor-States Submit All ==="
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "Output dir:  ${EXPERIMENT_DIR}"
echo "Turns:       ${TURNS}"
[[ $DRY_RUN -eq 1 ]] && echo "Mode:        DRY RUN" || echo "Mode:        LIVE"
echo ""

# Organism jobs
echo "--- Organism runs ---"
for org_file in "${ROOT}/configs/organisms"/*/*.yaml; do
    org_name=$(grep '^name:' "$org_file" | awk '{print $2}')

    if should_skip_organism "$org_name"; then
        continue
    fi

    # Extract model keys from finetuned_models block
    # Lines of the form "  model_key:" (two leading spaces, ends with colon)
    model_keys=$(grep -E '^  [a-z].*:$' "$org_file" | sed 's/://;s/^  //')

    for model_key in $model_keys; do
        if should_skip_model "$model_key"; then
            continue
        fi
        # Look up gpu_count from model config
        model_cfg="${ROOT}/configs/models/${model_key}.yaml"
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
            "${org_name}_${model_key}"
    done
done

# Control jobs (base model, no adapter)
echo ""
echo "--- Control runs (no adapter) ---"
for model_cfg in "${ROOT}/configs/models"/*.yaml; do
    model_key=$(grep '^name:' "$model_cfg" | awk '{print $2}')

    if should_skip_model "$model_key"; then
        continue
    fi

    gpu_count=$(grep '^gpu_count:' "$model_cfg" | awk '{print $2}')

    submit_job \
        "--control" \
        "${model_key}" \
        "${gpu_count}" \
        "control_${model_key}"
done

echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete."
else
    echo "Submitted ${SUBMITTED} jobs. Manifest: ${MANIFEST}"
    (( SKIPPED > 0 )) && echo "Skipped ${SKIPPED} (missing model configs)." || true
fi
