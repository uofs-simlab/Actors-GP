#!/bin/bash
# Scheduler settings of one radial run; calls the launcher. Called by run_arm_radial.sh.
set -euo pipefail
: "${MODE:?set MODE=synchronous or asynchronous}"
: "${WORKERS:?set WORKERS=1, 5, 10 or 20}"
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTOR_DIR="$(cd "${CAMPAIGN_DIR}/../ignition-actor" && pwd)"
export ACTOR_DIR
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
LABEL="sched_${MODE}_W${WORKERS}${LABEL_SUFFIX:-}"
export OUTPUT_DIR="${ROOT}/results/${LABEL}_job${JOB_ID}"
export ACTOR_BUILD_DIR="${ROOT}/builds/${LABEL}_job${JOB_ID}/actor"
export SOLVER_BUILD_DIR="${ROOT}/builds/${LABEL}_job${JOB_ID}/solver"
mkdir -p "${OUTPUT_DIR}/provenance" "${ROOT}/builds"
export BUILD_JOBS="${BUILD_JOBS:-6}"
source "${CAMPAIGN_DIR}/radial_common_env.sh"
export GP_SURFACE_SCRIPT="${ADVISOR_SCRIPT:?set ADVISOR_SCRIPT}"
export CORES_PER_NODE=$((WORKERS + 1))
export SERVER_RESERVED_CORES=1
export WORKERS_PER_NODE="${WORKERS}"
export ADAPTIVE_GP_MODE="${MODE}"
export ADAPTIVE_MIN_FIT_SAMPLES=0
export GP_SALVAGE_MAX_RELATIVE_HALF_WIDTH=0.25
export GP_NOISE_FLOOR="${GP_NOISE_FLOOR:-0.001}"
export GP_PROBE_EVERY="${GP_PROBE_EVERY:-0}"
export GP_PROBE_MIN_SAMPLES="${GP_PROBE_MIN_SAMPLES:-8}"
export GP_PROBE_RMS_TOLERANCE="${GP_PROBE_RMS_TOLERANCE:-0}"
export ADAPTIVE_INITIAL_SAMPLES=20
export ADAPTIVE_BATCH_SIZE="${WORKERS}"
export ADAPTIVE_ASYNC_FILL_CAP="${ASYNC_FILL_CAP:-1}"
export ADAPTIVE_MAX_SAMPLES="${BUDGET:-60}"
if [ "${MODE}" = asynchronous ]; then
    export ADAPTIVE_CANCEL_RUNNING_ON_STOP=true
else
    export ADAPTIVE_CANCEL_RUNNING_ON_STOP=false
fi
export GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE="${STOP_MAX:-0.10}"
export GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE="${STOP_RMS:-0.03}"
export GP_SHAPE_LOO_MAX_TOLERANCE="${LOO_MAX:-0.12}"
export GP_SHAPE_LOO_RMS_TOLERANCE="${LOO_RMS:-0.035}"
export GP_INTEGRATED_VARIANCE_TOLERANCE=0
export GP_VARIANCE_PATIENCE=0
export GP_STOP_HYSTERESIS=1
if [ "${MODE}" = asynchronous ]; then export ADAPTIVE_ROUNDS=400; else export ADAPTIVE_ROUNDS=100; fi
if [ -z "${PORT:-}" ] && [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then export PORT="$((20000 + JOB_ID % 40000))"; fi
{
  echo "mode=${MODE}"; echo "workers=${WORKERS}"; echo "batch=${WORKERS}"
  echo "seed=${ADAPTIVE_INITIAL_SAMPLES} budget_cap=${ADAPTIVE_MAX_SAMPLES} accuracy_stop=enabled"
  echo "bisection_rel_tol=${BISECTION_REL_TOL} stop_max=${GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE} stop_rms=${GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE}"
  echo "loo_max=${GP_SHAPE_LOO_MAX_TOLERANCE} loo_rms=${GP_SHAPE_LOO_RMS_TOLERANCE} noise_floor=${GP_NOISE_FLOOR}"
  echo "probe_every=${GP_PROBE_EVERY} probe_min=${GP_PROBE_MIN_SAMPLES} probe_rms_tol=${GP_PROBE_RMS_TOLERANCE}"
  echo "prequential_gate=${GP_PREQUENTIAL_MIN_SAMPLES:-12}@${GP_PREQUENTIAL_COVERAGE_LOWER_BOUND:-0.80}"
  echo "slurm_job_id=${JOB_ID}"; echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUTPUT_DIR}/provenance/config.txt"
if [ "${BRACKET_DRY_RUN:-false}" = true ]; then
    env | sort | grep -E '^(GP_|ADAPTIVE_|BISECTION_|AV_|R_MIN=|R_MAX=|GAMMA_MIN=|GAMMA_MAX=|SOLVER_|WORKERS_PER_NODE=|CORES_PER_NODE=)'
    exit 0
fi
bash "${ACTOR_DIR}/ignition_actor.sh" 2>&1 | tee "${OUTPUT_DIR}/run.log"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUTPUT_DIR}/provenance/config.txt"
