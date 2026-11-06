#!/bin/bash
#SBATCH --job-name=radial_gp_shape_focus
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --output=ignition_actor_adaptive_gp-%j.out

set -euo pipefail

SCRIPT_DIR="${ACTOR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "${SCRIPT_DIR}/ignition_actor.sh" ]; then
    echo "Error: actor script not found: ${SCRIPT_DIR}/ignition_actor.sh" >&2
    exit 1
fi
JOB_ID="${SLURM_JOB_ID:-manual}"

export ACTOR_BUILD_DIR="${ACTOR_BUILD_DIR:-${SCRIPT_DIR}/../ignition-actor-build-${JOB_ID}}"
export SOLVER_BUILD_DIR="${SOLVER_BUILD_DIR:-${SCRIPT_DIR}/../solver-build-${JOB_ID}}"
export BUILD_JOBS="${BUILD_JOBS:-6}"

# Core radial ignition setup.
export SOLVER_BACKEND="${SOLVER_BACKEND:-bdf}"
export GAMMA_VALUES="${GAMMA_VALUES:-0.01}"
export R_MIN="${R_MIN:-1.5}"
export R_MAX="${R_MAX:-25.0}"
export N_R="${N_R:-40}"
export DOMAIN_HALF_LENGTH="${DOMAIN_HALF_LENGTH:-100.0}"
export N_REFINEMENTS="${N_REFINEMENTS:-10}"
export AV_LO="${AV_LO:-0.0}"
export AV_HI="${AV_HI:-10.0}"
export BISECTION_TOL="${BISECTION_TOL:-1e-4}"
export BISECTION_MAX="${BISECTION_MAX:-50}"
export STEEPNESS="${STEEPNESS:-5.0}"
export TIME_STEP="${TIME_STEP:-5e-2}"
# Adaptive bisection consumes diagnostics and ignition summaries only. Disable
# VTU output completely to avoid OpenMPI MPI-IO failures on long actor paths.
export OUTPUT_STRIDE="${OUTPUT_STRIDE:-0}"
export OMPI_MCA_sharedfp="${OMPI_MCA_sharedfp:-individual}"
export WORKERS_PER_NODE="${WORKERS_PER_NODE:-5}"
export ADAPTIVE_GP="${ADAPTIVE_GP:-true}"
export ADAPTIVE_GP_MODE="${ADAPTIVE_GP_MODE:-synchronous}"
export ADAPTIVE_INITIAL_SAMPLES="${ADAPTIVE_INITIAL_SAMPLES:-5}"
export ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-3}"
export ADAPTIVE_ROUNDS="${ADAPTIVE_ROUNDS:-12}"
export ADAPTIVE_MAX_SAMPLES="${ADAPTIVE_MAX_SAMPLES:-40}"
export GP_PYTHON="${GP_PYTHON:-${SCRIPT_DIR}/../gp_botorch_env/bin/python}"
export GP_GRID_SIZE="${GP_GRID_SIZE:-2000}"
export GP_ACQUISITION="${GP_ACQUISITION:-variation-nipv}"
export GP_INTEGRATION_GRID_SIZE="${GP_INTEGRATION_GRID_SIZE:-512}"
export GP_VARIATION_WEIGHT="${GP_VARIATION_WEIGHT:-0.95}"
export GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET="${GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET:-0.08}"
export GP_FOCUS_VARIATION_COVERAGE="${GP_FOCUS_VARIATION_COVERAGE:-0.70}"
export GP_ACQUISITION_CHUNK_SIZE="${GP_ACQUISITION_CHUNK_SIZE:-128}"
export GP_JOINT_NUM_RESTARTS="${GP_JOINT_NUM_RESTARTS:-12}"
export GP_JOINT_RAW_SAMPLES="${GP_JOINT_RAW_SAMPLES:-256}"
export GP_JOINT_MAXITER="${GP_JOINT_MAXITER:-200}"
export GP_JOINT_RETRIES="${GP_JOINT_RETRIES:-3}"
export GP_JOINT_FALLBACK="${GP_JOINT_FALLBACK:-greedy-nipv}"
export GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE="${GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE:-0.08}"
export GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE="${GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE:-0.02}"
export GP_SHAPE_LOO_MAX_TOLERANCE="${GP_SHAPE_LOO_MAX_TOLERANCE:-0.05}"
export GP_SHAPE_LOO_RMS_TOLERANCE="${GP_SHAPE_LOO_RMS_TOLERANCE:-0.02}"
export GP_INTEGRATED_VARIANCE_TOLERANCE="${GP_INTEGRATED_VARIANCE_TOLERANCE:-0}"
export GP_VARIANCE_MIN_RELATIVE_IMPROVEMENT="${GP_VARIANCE_MIN_RELATIVE_IMPROVEMENT:-0.01}"
export GP_VARIANCE_PATIENCE="${GP_VARIANCE_PATIENCE:-0}"
export GP_CONVERGENCE_R_MIN="${GP_CONVERGENCE_R_MIN:-1.7}"
export GP_NOISE_FLOOR="${GP_NOISE_FLOOR:-0.001}"
export GP_TARGET_TRANSFORM="${GP_TARGET_TRANSFORM:-log}"
export GP_NO_PLOT="${GP_NO_PLOT:-false}"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs_radial_adaptive_gp_shape_focus_${JOB_ID}}"

# Avoid port collisions when multiple actor jobs land on the same node.
if [ -z "${PORT:-}" ]; then
    if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
        export PORT="$((20000 + SLURM_JOB_ID % 40000))"
    else
        export PORT="23446"
    fi
fi

echo "Adaptive radial ignition GP job"
echo "Source:            ${SCRIPT_DIR}"
echo "Output directory:  ${OUTPUT_DIR}"
echo "Port:              ${PORT}"
echo "Solver backend:    ${SOLVER_BACKEND}"
echo "gamma:             ${GAMMA_VALUES}"
echo "R range:           [${R_MIN}, ${R_MAX}]"
echo "Radial boundary:   ${DOMAIN_HALF_LENGTH}"
echo "Adaptive samples:  mode=${ADAPTIVE_GP_MODE}, initial=${ADAPTIVE_INITIAL_SAMPLES}, batch=${ADAPTIVE_BATCH_SIZE}, rounds_per_gamma=${ADAPTIVE_ROUNDS}, max_per_gamma=${ADAPTIVE_MAX_SAMPLES}"
echo "GP acquisition:    ${GP_ACQUISITION}, variation_weight=${GP_VARIATION_WEIGHT}, focus_uncertainty=${GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET}, focus_variation_coverage=${GP_FOCUS_VARIATION_COVERAGE}, integration_grid=${GP_INTEGRATION_GRID_SIZE}, chunk=${GP_ACQUISITION_CHUNK_SIZE}"
echo "GP joint NIPV:     restarts=${GP_JOINT_NUM_RESTARTS}, raw=${GP_JOINT_RAW_SAMPLES}, maxiter=${GP_JOINT_MAXITER}, retries=${GP_JOINT_RETRIES}, fallback=${GP_JOINT_FALLBACK}"
echo "GP convergence:    max_rel_95_tol=${GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE}, rms_rel_95_tol=${GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE}, R=[${GP_CONVERGENCE_R_MIN},${R_MAX}]"
echo "GP variance:       integrated_tol=${GP_INTEGRATED_VARIANCE_TOLERANCE}, min_rel_improvement=${GP_VARIANCE_MIN_RELATIVE_IMPROVEMENT}, stagnation_patience=${GP_VARIANCE_PATIENCE}"
echo "GP python:         ${GP_PYTHON}"

bash "${SCRIPT_DIR}/ignition_actor.sh"