#!/bin/bash
# Scheduler settings of one wave-segment run; calls the launcher. Called by run_arm_wave.sh.
set -euo pipefail
: "${MODE:?set MODE=synchronous or asynchronous}"
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTOR_DIR="$(cd "${CAMPAIGN_DIR}/../ignition-actor" && pwd)"
export ACTOR_DIR
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
LABEL="wsegsurf_${MODE}_W${WORKERS_TOTAL}${LABEL_SUFFIX:-}"
export OUTPUT_DIR="${ROOT}/results/${LABEL}_job${JOB_ID}"
export ACTOR_BUILD_DIR="${ROOT}/builds/${LABEL}_job${JOB_ID}/actor"
export SOLVER_BUILD_DIR="${ROOT}/builds/${LABEL}_job${JOB_ID}/solver"
mkdir -p "${OUTPUT_DIR}/provenance" "${ROOT}/builds"
export BUILD_JOBS="${BUILD_JOBS:-6}"
source "${CAMPAIGN_DIR}/wave_common_env.sh"
export GP_SURFACE_SCRIPT="${ADVISOR_SCRIPT:?set ADVISOR_SCRIPT}"

export WORKERS_PER_NODE="${WORKERS_PER_NODE:-4}"
export SOLVER_MPI_RANKS="${RANKS:-8}"
# Use mpirun with per-worker core pinning; srun steps inside an allocation can share cores.
export SOLVER_RUNNER=mpirun
export PIN_SOLVER_CORES=true
export SERVER_RESERVED_CORES=1

export ADAPTIVE_GP_MODE="${MODE}"
export ADAPTIVE_INITIAL_SAMPLES="${SEED:-16}"
export ADAPTIVE_BATCH_SIZE="${WORKERS_TOTAL}"
export ADAPTIVE_MAX_SAMPLES="${BUDGET:-48}"
export GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE="${STOP_MAX:-0.10}"
export GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE="${STOP_RMS:-0.03}"
export GP_SHAPE_LOO_MAX_TOLERANCE="${LOO_MAX:-0.12}"
export GP_SHAPE_LOO_RMS_TOLERANCE="${LOO_RMS:-0.035}"
export GP_INTEGRATED_VARIANCE_TOLERANCE=0
export GP_VARIANCE_PATIENCE=0
export GP_STOP_HYSTERESIS=1
if [ "${MODE}" = asynchronous ]; then
    export ADAPTIVE_ROUNDS=400
    export ADAPTIVE_CANCEL_RUNNING_ON_STOP=true
    export ADAPTIVE_ASYNC_FILL_CAP="${WORKERS_TOTAL}"
    export GP_ACQUISITION_VALUE_TARGET="${VALUE_TARGET:-0.90}"
else
    export ADAPTIVE_ROUNDS=100
    export ADAPTIVE_CANCEL_RUNNING_ON_STOP=false
    export ADAPTIVE_ASYNC_FILL_CAP=0
    export GP_ACQUISITION_VALUE_TARGET=0
fi
if [ -z "${PORT:-}" ] && [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then export PORT="$((20000 + JOB_ID % 40000))"; fi
{
  echo "mode=${MODE} workers_total=${WORKERS_TOTAL} ranks=${SOLVER_MPI_RANKS}"
  echo "ell=[${R_MIN},${R_MAX}] gamma=[${GAMMA_MIN},${GAMMA_MAX}] final_time=${FINAL_TIME}"
  echo "seed=${ADAPTIVE_INITIAL_SAMPLES} budget=${ADAPTIVE_MAX_SAMPLES}"
  echo "slurm_job_id=${JOB_ID}"; echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUTPUT_DIR}/provenance/config.txt"
if [ "${WAVE_BRACKET_DRY_RUN:-false}" = true ]; then
    env | sort | grep -E '^(GP_|ADAPTIVE_|BISECTION_|AV_|R_MIN=|R_MAX=|GAMMA_MIN=|GAMMA_MAX=|SOLVER_|WORKERS|RANKS=|SEED=|BUDGET=|IC_TYPE=|FINAL_TIME=|FIXED_FINAL_TIME=|DOMAIN_|DETECTOR_|USE_DETECTOR_|PIN_SOLVER_|SERVER_RESERVED_)'
    exit 0
fi
bash "${ACTOR_DIR}/ignition_actor_adaptive_gp.sh" 2>&1 | tee "${OUTPUT_DIR}/run.log"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUTPUT_DIR}/provenance/config.txt"
