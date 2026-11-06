#!/bin/bash
#SBATCH --job-name=ignition_actor
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --output=ignition_actor-%j.out

set -e

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 petsc/3.21.6 p4est/2.8.6

export CC=gcc
export CXX=g++
export DEAL_II_DIR="${DEAL_II_DIR:?set DEAL_II_DIR to the deal.II 9.6.0 installation prefix (see README)}"
export CAF_ROOT="${CAF_ROOT:?set CAF_ROOT to the CAF 1.0.2 installation prefix (see README)}"
export LD_LIBRARY_PATH="${CAF_ROOT}/lib64:${DEAL_II_DIR}/lib:${LD_LIBRARY_PATH}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export DEAL_II_NUM_THREADS="${DEAL_II_NUM_THREADS:-1}"
export PETSC_OPTIONS="${PETSC_OPTIONS:--options_left false}"

SCRIPT_DIR="${ACTOR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "${SCRIPT_DIR}/CMakeLists.txt" ]; then
    echo "Error: radial actor source directory not found: ${SCRIPT_DIR}" >&2
    exit 1
fi
ACTOR_BUILD_DIR="${ACTOR_BUILD_DIR:-${SCRIPT_DIR}/../ignition-actor-build}"
SOLVER_SOURCE_DIR="${SCRIPT_DIR}/../solver"
SOLVER_BUILD_DIR="${SOLVER_BUILD_DIR:-${SCRIPT_DIR}/../solver-build}"

echo "Radial actor source: ${SCRIPT_DIR}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
echo "Configuring radial ignition actor..."
test ! -e "${ACTOR_BUILD_DIR}" || { echo "Refusing existing actor build: ${ACTOR_BUILD_DIR}" >&2; exit 1; }
mkdir -p "${ACTOR_BUILD_DIR}"
cd "${ACTOR_BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
    -DCAF_ROOT="${CAF_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --target ignition_actor -j"${BUILD_JOBS}"

ACTOR_BINARY="${ACTOR_BUILD_DIR}/bin/ignition_actor"
if [ ! -x "${ACTOR_BINARY}" ]; then
    echo "Error: ignition_actor binary not found"
    exit 1
fi

SOLVER_BACKEND="${SOLVER_BACKEND:-bdf}"
case "${SOLVER_BACKEND}" in
    bdf)
        SOLVER_TARGET="radial_fhn_bdf"
        ;;
    *)
        echo "Error: SOLVER_BACKEND must be 'bdf'" >&2
        exit 2
        ;;
esac

SOLVER_TARGET="${SOLVER_TARGET_OVERRIDE:-${SOLVER_TARGET}}"

echo "Configuring standalone radial ${SOLVER_BACKEND} solver..."
test ! -e "${SOLVER_BUILD_DIR}" || { echo "Refusing existing solver build: ${SOLVER_BUILD_DIR}" >&2; exit 1; }
mkdir -p "${SOLVER_BUILD_DIR}"
cd "${SOLVER_BUILD_DIR}"
cmake "${SOLVER_SOURCE_DIR}" \
    -DDEAL_II_DIR="${DEAL_II_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --target "${SOLVER_TARGET}" -j"${BUILD_JOBS}"

SOLVER_BINARY="${SOLVER_BINARY:-${SOLVER_BUILD_DIR}/bin/${SOLVER_TARGET}}"
if [ ! -x "${SOLVER_BINARY}" ]; then
    echo "Error: ${SOLVER_TARGET} binary not found"
    exit 1
fi


WORKERS_PER_NODE="${WORKERS_PER_NODE:-31}"
if ! [[ "${WORKERS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: WORKERS_PER_NODE must be a positive integer, got '${WORKERS_PER_NODE}'" >&2
    exit 2
fi
PORT="${PORT:-23446}"

R_MIN="${R_MIN:-1.5}"
R_MAX="${R_MAX:-25.0}"
N_R="${N_R:-40}"
R_VALUES="${R_VALUES:-}"
AW_MIN="${AW_MIN:-0.0}"
GAMMA_VALUES="${GAMMA_VALUES:-0.01}"
GAMMA_MIN="${GAMMA_MIN:-${GAMMA_VALUES%%,*}}"
GAMMA_MAX="${GAMMA_MAX:-${GAMMA_VALUES##*,}}"
GAMMA_WAVE_SPEEDS="${GAMMA_WAVE_SPEEDS:-}"
AV_LO="${AV_LO:-0.0}"
AV_HI="${AV_HI:-10.0}"
BISECTION_TOL="${BISECTION_TOL:-1e-4}"
BISECTION_REL_TOL="${BISECTION_REL_TOL:-0}"
BISECTION_MAX="${BISECTION_MAX:-50}"
BISECTION_PROBE_LADDER="${BISECTION_PROBE_LADDER:-true}"
BISECTION_PROBE_START="${BISECTION_PROBE_START:-1.0}"
BISECTION_PROBE_RATIO="${BISECTION_PROBE_RATIO:-2.0}"
# Pin each worker's solve to its own cores. Mandatory when a node hosts more
# than one worker; harmless when one worker owns the node.
PIN_SOLVER_CORES="${PIN_SOLVER_CORES:-false}"
# Single observation window for every sample (no radial T(R) override).
FIXED_FINAL_TIME="${FIXED_FINAL_TIME:-false}"
STEEPNESS="${STEEPNESS:-5.0}"

DOMAIN_HALF_LENGTH="${DOMAIN_HALF_LENGTH:-100.0}"
N_REFINEMENTS="${N_REFINEMENTS:-10}"
MAX_DELTA="${MAX_DELTA:-4}"
PRE_REFINE_IC="${PRE_REFINE_IC:-false}"
FINAL_TIME="${FINAL_TIME:-}"
FAST_WAVE_SPEED="${FAST_WAVE_SPEED:-}"
FAST_WAVE_SPEED_FALLBACK="${FAST_WAVE_SPEED_FALLBACK:-0.50}"
WAVE_SPEED_TABLE="${WAVE_SPEED_TABLE:-${SCRIPT_DIR}/tables/wave_speeds.csv}"
WAVE_SPEED_CONTINUATION_FILE="${WAVE_SPEED_CONTINUATION_FILE:-}"
WAVE_SPEED_MIN_RELIABLE_GAMMA="${WAVE_SPEED_MIN_RELIABLE_GAMMA:-0.00048828125}"
WAVE_SPEED_LOW_GAMMA_PLATEAU="${WAVE_SPEED_LOW_GAMMA_PLATEAU:-0.5192}"
WAVE_SPEED_ROUND_UP="${WAVE_SPEED_ROUND_UP:-0.0001}"
FINITE_TIME_SAFETY_FACTOR="${FINITE_TIME_SAFETY_FACTOR:-0.9}"
IGNITION_MAX_V_THRESHOLD="${IGNITION_MAX_V_THRESHOLD:-0.5}"
EARLY_EXIT_ON_QUENCH="${EARLY_EXIT_ON_QUENCH:-false}"
QUENCH_MAX_V_THRESHOLD="${QUENCH_MAX_V_THRESHOLD:-0.05}"
QUENCH_WINDOW="${QUENCH_WINDOW:-10.0}"
EARLY_EXIT_ON_PROPAGATION="${EARLY_EXIT_ON_PROPAGATION:-false}"
IGNITION_EXCITED_FRACTION_THRESHOLD="${IGNITION_EXCITED_FRACTION_THRESHOLD:-0.01}"
TIME_STEP="${TIME_STEP:-5e-2}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-1000000000}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs_radial_${SLURM_JOB_ID}}"
ADAPTIVE_GP="${ADAPTIVE_GP:-false}"
ADAPTIVE_GP_SURFACE="${ADAPTIVE_GP_SURFACE:-false}"
ADAPTIVE_GP_MODE="${ADAPTIVE_GP_MODE:-synchronous}"
ADAPTIVE_INITIAL_SAMPLES="${ADAPTIVE_INITIAL_SAMPLES:-5}"
ADAPTIVE_BATCH_SIZE="${ADAPTIVE_BATCH_SIZE:-5}"
ADAPTIVE_ROUNDS="${ADAPTIVE_ROUNDS:-0}"
ADAPTIVE_MAX_SAMPLES="${ADAPTIVE_MAX_SAMPLES:-0}"
ADAPTIVE_CANCEL_RUNNING_ON_STOP="${ADAPTIVE_CANCEL_RUNNING_ON_STOP:-false}"
ADAPTIVE_ASYNC_FILL_CAP="${ADAPTIVE_ASYNC_FILL_CAP:-0}"
ADAPTIVE_ASYNC_TARGET_ACTIVE="${ADAPTIVE_ASYNC_TARGET_ACTIVE:-0}"
GP_PYTHON="${GP_PYTHON:-${SCRIPT_DIR}/../gp_botorch_env/bin/python}"
GP_SURFACE_SCRIPT="${GP_SURFACE_SCRIPT:-${SCRIPT_DIR}/gp/gp_advisor.py}"
GP_OUT_DIR="${GP_OUT_DIR:-}"
GP_GRID_SIZE="${GP_GRID_SIZE:-2000}"
GP_ACQUISITION="${GP_ACQUISITION:-variation-nipv}"
GP_INTEGRATION_GRID_SIZE="${GP_INTEGRATION_GRID_SIZE:-512}"
GP_VARIATION_WEIGHT="${GP_VARIATION_WEIGHT:-0.5}"
GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET="${GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET:-0}"
GP_FOCUS_VARIATION_COVERAGE="${GP_FOCUS_VARIATION_COVERAGE:-1}"
GP_ACQUISITION_CHUNK_SIZE="${GP_ACQUISITION_CHUNK_SIZE:-128}"
GP_SURFACE_NIPV_BACKEND="${GP_SURFACE_NIPV_BACKEND:-botorch}"
GP_JOINT_NUM_RESTARTS="${GP_JOINT_NUM_RESTARTS:-12}"
GP_JOINT_RAW_SAMPLES="${GP_JOINT_RAW_SAMPLES:-256}"
GP_JOINT_MAXITER="${GP_JOINT_MAXITER:-200}"
GP_JOINT_RETRIES="${GP_JOINT_RETRIES:-3}"
GP_JOINT_FALLBACK="${GP_JOINT_FALLBACK:-greedy-nipv}"
GP_SURFACE_SUGGEST_MIN_DISTANCE="${GP_SURFACE_SUGGEST_MIN_DISTANCE:-0.01}"
GP_SURFACE_SUGGEST_MIN_DISTANCE_LENGTHSCALES="${GP_SURFACE_SUGGEST_MIN_DISTANCE_LENGTHSCALES:-0.15}"
GP_SURFACE_INPUT_TRANSFORM_R="${GP_SURFACE_INPUT_TRANSFORM_R:-auto}"
GP_SURFACE_INPUT_TRANSFORM_GAMMA="${GP_SURFACE_INPUT_TRANSFORM_GAMMA:-auto}"
GP_SURFACE_MODEL_RESELECT_EVERY="${GP_SURFACE_MODEL_RESELECT_EVERY:-10}"
GP_SURFACE_MODEL_ELIGIBILITY_REFRESH="${GP_SURFACE_MODEL_ELIGIBILITY_REFRESH:-false}"
GP_ACQUISITION_VALUE_TARGET="${GP_ACQUISITION_VALUE_TARGET:-0}"
ADAPTIVE_MIN_FIT_SAMPLES="${ADAPTIVE_MIN_FIT_SAMPLES:-0}"
GP_ASYNC_EARLY_START="${GP_ASYNC_EARLY_START:-false}"
GP_ASYNC_START_MIN_VALID="${GP_ASYNC_START_MIN_VALID:-0}"
GP_ASYNC_EARLY_BRACKETS="${GP_ASYNC_EARLY_BRACKETS:-false}"
GP_PROBE_MAX_RECERTIFICATIONS="${GP_PROBE_MAX_RECERTIFICATIONS:-0}"
GP_BRACKET_QUENCH_COST_RATIO="${GP_BRACKET_QUENCH_COST_RATIO:-1}"
GP_BACKGROUND_PROBES_OFF="${GP_BACKGROUND_PROBES_OFF:-false}"
GP_SALVAGE_MAX_RELATIVE_HALF_WIDTH="${GP_SALVAGE_MAX_RELATIVE_HALF_WIDTH:-0.25}"
GP_SURFACE_SAMPLE_LIST="${GP_SURFACE_SAMPLE_LIST:-}"
GP_SURFACE_DESIGN_LADDER_R="${GP_SURFACE_DESIGN_LADDER_R:-blend}"
GP_SURFACE_DESIGN_LADDER_GAMMA="${GP_SURFACE_DESIGN_LADDER_GAMMA:-blend}"
GP_SURFACE_INTERIOR_MARGIN="${GP_SURFACE_INTERIOR_MARGIN:-0.05}"
GP_SURFACE_CANDIDATE_COUNT="${GP_SURFACE_CANDIDATE_COUNT:-4096}"
GP_SURFACE_PREDICTION_R_COUNT="${GP_SURFACE_PREDICTION_R_COUNT:-160}"
GP_SURFACE_PREDICTION_GAMMA_COUNT="${GP_SURFACE_PREDICTION_GAMMA_COUNT:-100}"
GP_SURFACE_BOUNDARY_TOLERANCE="${GP_SURFACE_BOUNDARY_TOLERANCE:-0.02}"
GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE="${GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE:-0}"
GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE="${GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE:-0}"
GP_SHAPE_LOO_MAX_TOLERANCE="${GP_SHAPE_LOO_MAX_TOLERANCE:-0}"
GP_SHAPE_LOO_RMS_TOLERANCE="${GP_SHAPE_LOO_RMS_TOLERANCE:-0}"
GP_INTEGRATED_VARIANCE_TOLERANCE="${GP_INTEGRATED_VARIANCE_TOLERANCE:-0}"
GP_VARIANCE_MIN_RELATIVE_IMPROVEMENT="${GP_VARIANCE_MIN_RELATIVE_IMPROVEMENT:-0.01}"
GP_VARIANCE_PATIENCE="${GP_VARIANCE_PATIENCE:-0}"
GP_STOP_HYSTERESIS="${GP_STOP_HYSTERESIS:-2}"
GP_PROBE_EVERY="${GP_PROBE_EVERY:-4}"
GP_PROBE_MIN_SAMPLES="${GP_PROBE_MIN_SAMPLES:-10}"
GP_PROBE_RMS_TOLERANCE="${GP_PROBE_RMS_TOLERANCE:-0.01}"
GP_PROBE_REQUIRE_READINESS="${GP_PROBE_REQUIRE_READINESS:-false}"
GP_PROBE_READINESS_TRANSITION="${GP_PROBE_READINESS_TRANSITION:-false}"
GP_SYNC_CERTIFICATION_BATCHES="${GP_SYNC_CERTIFICATION_BATCHES:-false}"
GP_ASYNC_POSTFIT_CERTIFICATION="${GP_ASYNC_POSTFIT_CERTIFICATION:-false}"
GP_ASYNC_CERTIFICATION_REQUIRE_READINESS="${GP_ASYNC_CERTIFICATION_REQUIRE_READINESS:-false}"
GP_PROBE_INFLIGHT_PROMOTION="${GP_PROBE_INFLIGHT_PROMOTION:-false}"
GP_READINESS_ANTICIPATION="${GP_READINESS_ANTICIPATION:-false}"
GP_PREQUENTIAL_MIN_SAMPLES="${GP_PREQUENTIAL_MIN_SAMPLES:-12}"
GP_PREQUENTIAL_COVERAGE_LOWER_BOUND="${GP_PREQUENTIAL_COVERAGE_LOWER_BOUND:-0.70}"
GP_PREQUENTIAL_WARMUP_FACTOR="${GP_PREQUENTIAL_WARMUP_FACTOR:-1.5}"
GP_SEEDED_BRACKETS="${GP_SEEDED_BRACKETS:-false}"
GP_SEED_SIGMA_K="${GP_SEED_SIGMA_K:-3.0}"
GP_CONVERGENCE_R_MIN="${GP_CONVERGENCE_R_MIN:-0}"
GP_NOISE_FLOOR="${GP_NOISE_FLOOR:-0.001}"
GP_TARGET_TRANSFORM="${GP_TARGET_TRANSFORM:-log}"
GP_NO_PLOT="${GP_NO_PLOT:-false}"
SOLVER_RUNNER="${SOLVER_RUNNER:-direct}"
IC_TYPE="${IC_TYPE:-smooth_disk}"
SOLVER_MPI_RANKS="${SOLVER_MPI_RANKS:-1}"
# --- 2D wave_segment / propagation-oracle knobs (inert unless used) ---
SEGMENT_WAVE_WIDTH="${SEGMENT_WAVE_WIDTH:-5.0}"
SEGMENT_TAIL_WIDTH="${SEGMENT_TAIL_WIDTH:-20.0}"
USE_DETECTOR_ORACLE="${USE_DETECTOR_ORACLE:-false}"
DETECTOR_X="${DETECTOR_X:-50.0}"
DETECTOR_MIN_EXCITED_AREA="${DETECTOR_MIN_EXCITED_AREA:-1.0}"
DETECTOR_OFFSET_FROM_RADIUS="${DETECTOR_OFFSET_FROM_RADIUS:-0}"
DETECTOR_REQUIRE_AREA_GROWTH="${DETECTOR_REQUIRE_AREA_GROWTH:-false}"

# Client actors are placed with srun, but their child radial solver processes are
# launched directly. The radial solver still calls MPI_Init_thread via deal.II/
# PETSc, so remove the one-task Slurm/PMIx step environment before the actor
# starts; otherwise solver children may try to attach to the parent srun step.
MPI_SINGLETON_ENV_CLEANER='
unset "${!PMIX_@}" "${!PMI_@}" "${!OMPI_COMM_WORLD_@}"
unset MPI_LOCALNRANKS MPI_LOCALRANKID OPAL_PREFIX
unset SLURM_STEP_ID SLURM_STEP_NAME SLURM_STEP_NUM_NODES
unset SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_GTIDS
exec "$@"
'

if [ -z "${FAST_WAVE_SPEED}" ]; then
    if [ -n "${GAMMA_WAVE_SPEEDS}" ]; then
        FAST_WAVE_SPEED="$(
            awk -F',' -v mapping="${GAMMA_WAVE_SPEEDS}" \
                -v g="${GAMMA_VALUES%%,*}" '
                BEGIN {
                    n = split(mapping, pairs, ",")   # entries are gamma:speed, COMMA-separated
                    for (i = 1; i <= n; i++) {
                        split(pairs[i], kv, ":")
                        if (kv[1] + 0 == g + 0) { printf "%.17g", kv[2]; exit }
                    }
                }'
        )"
        WAVE_SPEED_SOURCE="GAMMA_WAVE_SPEEDS mapping"
    fi
    if [ -z "${FAST_WAVE_SPEED}" ] && [ -f "${WAVE_SPEED_TABLE}" ]; then
        gammas="${GAMMA_VALUES}"
        FAST_WAVE_SPEED="$(
            awk -F',' -v gammas="$gammas" '
                NR == 1 { next }
                {
                    split(gammas, want, ",")
                    for (i in want)
                        if ($1 + 0 == want[i] + 0) { printf "%.17g", $2; exit }
                }' "${WAVE_SPEED_TABLE}"
        )"
        WAVE_SPEED_SOURCE="table ${WAVE_SPEED_TABLE}"
    fi
    if [ -z "${FAST_WAVE_SPEED}" ]; then
        FAST_WAVE_SPEED="${FAST_WAVE_SPEED_FALLBACK}"
        if [ -f "${WAVE_SPEED_TABLE}" ]; then
            WAVE_SPEED_SOURCE="fallback; no row in ${WAVE_SPEED_TABLE}"
        else
            WAVE_SPEED_SOURCE="fallback; table ${WAVE_SPEED_TABLE} not found"
        fi
    fi
else
    WAVE_SPEED_SOURCE="FAST_WAVE_SPEED environment override"
fi

# Conservative fallback observation time: how long before an outward wave at
# FAST_WAVE_SPEED reaches the boundary.
radial_fallback_time() {
    awk -v safety="$1" -v radial_extent="$2" -v radius="$3" -v speed="$4" \
        'BEGIN {
             if (speed + 0 <= 0) exit 0
             printf "%.17g", safety * (radial_extent - radius) / speed
         }'
}
RADIAL_TIME_AT_R_MIN="$(radial_fallback_time "${FINITE_TIME_SAFETY_FACTOR}" \
    "${DOMAIN_HALF_LENGTH}" "${R_MIN}" "${FAST_WAVE_SPEED}")"
RADIAL_TIME_AT_R_MAX="$(radial_fallback_time "${FINITE_TIME_SAFETY_FACTOR}" \
    "${DOMAIN_HALF_LENGTH}" "${R_MAX}" "${FAST_WAVE_SPEED}")"
FINAL_TIME="${FINAL_TIME:-${RADIAL_TIME_AT_R_MAX}}"
if [ -z "${FINAL_TIME}" ]; then
    echo "Error: FINAL_TIME is unset and cannot be derived because" \
         "FAST_WAVE_SPEED='${FAST_WAVE_SPEED}' is not positive." \
         "Set FINAL_TIME explicitly, or give a positive FAST_WAVE_SPEED." >&2
    exit 2
fi

# --- node list --------------------------------------------------------------
expand_nodelist() {
    local nl="$1"
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "$nl" 2>/dev/null && return 0
    fi
    # Non-interactive submissions do not always inherit Slurm's bin directory
    # on PATH. Use the cluster installation explicitly before attempting the
    # comma-only fallback, which cannot expand bracket expressions such as
    # platocpu[048-051].
    local cluster_scontrol="/cm/shared/apps/slurm/current/bin/scontrol"
    if [ -x "${cluster_scontrol}" ]; then
        "${cluster_scontrol}" show hostnames "$nl" 2>/dev/null && return 0
    fi
    printf '%s\n' "$nl" | tr ',' '\n'
}

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    mapfile -t all_nodes < <(expand_nodelist "${SLURM_JOB_NODELIST}")
else
    all_nodes=("$(hostname)")
fi
expected_nodes="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}"
if [[ "${expected_nodes}" =~ ^[1-9][0-9]*$ ]] &&
   [ "${#all_nodes[@]}" -ne "${expected_nodes}" ]; then
    echo "Error: expanded ${#all_nodes[@]} node(s) from" \
         "SLURM_JOB_NODELIST='${SLURM_JOB_NODELIST:-}', but Slurm allocated" \
         "${expected_nodes}. Refusing to run with an incomplete node list." >&2
    exit 2
fi
server_node="${all_nodes[0]}"
client_nodes=("${all_nodes[@]:1}")

# Like scontrol above, srun may be installed by Slurm but absent from the PATH
# left by `module --force purge` in a non-interactive batch shell. Resolve it
# before starting any work so a multi-node job cannot silently fall back to
# server-local workers after all remote launches fail.
SRUN_BIN=""
if [ "${#client_nodes[@]}" -gt 0 ]; then
    if [ -n "${SRUN:-}" ] && command -v "${SRUN}" >/dev/null 2>&1; then
        SRUN_BIN="$(command -v "${SRUN}")"
    elif command -v srun >/dev/null 2>&1; then
        SRUN_BIN="$(command -v srun)"
    elif [ -x /cm/shared/apps/slurm/current/bin/srun ]; then
        SRUN_BIN="/cm/shared/apps/slurm/current/bin/srun"
    else
        echo "Error: client nodes were allocated, but no usable srun executable" \
             "was found. Set SRUN to its absolute path." >&2
        exit 2
    fi
fi

# --- configuration summary --------------------------------------------------
echo "Server node:       ${server_node}"
if [ "${#client_nodes[@]}" -gt 0 ]; then
    echo "Client node(s):    ${client_nodes[*]}"
    echo "Slurm launcher:    ${SRUN_BIN}"
fi
echo "Workers per node:  ${WORKERS_PER_NODE}"
if [ -n "${R_VALUES}" ]; then
    echo "Samples:           explicit R values=${R_VALUES}, Aw=${AW_MIN}, gamma=${GAMMA_VALUES}"
else
    echo "Samples:           R=[${R_MIN},${R_MAX}] x${N_R}, Aw=${AW_MIN}, gamma=${GAMMA_VALUES}"
fi
echo "Bisection:         Av=[${AV_LO},${AV_HI}], tolerance=${BISECTION_TOL}, rel_tolerance=${BISECTION_REL_TOL}, max iterations=${BISECTION_MAX}"
echo "Probe ladder:      ${BISECTION_PROBE_LADDER} (start=${BISECTION_PROBE_START}, ratio=${BISECTION_PROBE_RATIO})"
echo "Solver backend:    ${SOLVER_BACKEND}"
echo "Solver binary:     ${SOLVER_BINARY}"
echo "Solver runner:     ${SOLVER_RUNNER}, mpi_ranks=${SOLVER_MPI_RANKS}"
echo "Detector oracle:   ${USE_DETECTOR_ORACLE} (x=${DETECTOR_X}, offset from R=${DETECTOR_OFFSET_FROM_RADIUS}, area>=${DETECTOR_MIN_EXCITED_AREA}, require area growth=${DETECTOR_REQUIRE_AREA_GROWTH})"
echo "Wave speed source: ${WAVE_SPEED_SOURCE}"
if [ -n "${WAVE_SPEED_CONTINUATION_FILE}" ]; then
    echo "Wave continuation: ${WAVE_SPEED_CONTINUATION_FILE} (reliable gamma>=${WAVE_SPEED_MIN_RELIABLE_GAMMA}, low-gamma plateau=${WAVE_SPEED_LOW_GAMMA_PLATEAU}, round-up=${WAVE_SPEED_ROUND_UP})"
fi
echo "Radial boundary:    ${DOMAIN_HALF_LENGTH}"
echo "Radial time window: T(R)=${FINITE_TIME_SAFETY_FACTOR}*(boundary-R)/C_fast"
echo "Fallback time range:[${RADIAL_TIME_AT_R_MAX},${RADIAL_TIME_AT_R_MIN}] for R=[${R_MIN},${R_MAX}]"
echo "Adaptive GP:        ${ADAPTIVE_GP}, surface=${ADAPTIVE_GP_SURFACE}, mode=${ADAPTIVE_GP_MODE}, initial=${ADAPTIVE_INITIAL_SAMPLES}, batch=${ADAPTIVE_BATCH_SIZE}, rounds=${ADAPTIVE_ROUNDS}, max_samples=${ADAPTIVE_MAX_SAMPLES}, async_fill_cap=${ADAPTIVE_ASYNC_FILL_CAP}, async_target_active=${ADAPTIVE_ASYNC_TARGET_ACTIVE}"
echo "Adaptive cancel:    running_on_stop=${ADAPTIVE_CANCEL_RUNNING_ON_STOP}"
echo "GP acquisition:     ${GP_ACQUISITION}, variation_weight=${GP_VARIATION_WEIGHT}, focus_uncertainty=${GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET}, focus_variation_coverage=${GP_FOCUS_VARIATION_COVERAGE}, integration_grid=${GP_INTEGRATION_GRID_SIZE}, chunk=${GP_ACQUISITION_CHUNK_SIZE}"
echo "GP joint NIPV:      restarts=${GP_JOINT_NUM_RESTARTS}, raw=${GP_JOINT_RAW_SAMPLES}, maxiter=${GP_JOINT_MAXITER}, retries=${GP_JOINT_RETRIES}, fallback=${GP_JOINT_FALLBACK}"
echo "GP stop criteria:   max_rel_95_tol=${GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE}, rms_rel_95_tol=${GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE}, integrated_variance_tol=${GP_INTEGRATED_VARIANCE_TOLERANCE}"
echo "GP variance stop:   stagnation_patience=${GP_VARIANCE_PATIENCE}, min_rel_improvement=${GP_VARIANCE_MIN_RELATIVE_IMPROVEMENT}, stop_hysteresis=${GP_STOP_HYSTERESIS}, prequential_gate=${GP_PREQUENTIAL_MIN_SAMPLES}@${GP_PREQUENTIAL_COVERAGE_LOWER_BOUND}, warmup=${GP_PREQUENTIAL_WARMUP_FACTOR}x, seeded_brackets=${GP_SEEDED_BRACKETS} (k=${GP_SEED_SIGMA_K}), convergence_R_min=${GP_CONVERGENCE_R_MIN}"
echo "GP surface:         gamma=[${GAMMA_MIN},${GAMMA_MAX}], min_distance=${GP_SURFACE_SUGGEST_MIN_DISTANCE} floor / ${GP_SURFACE_SUGGEST_MIN_DISTANCE_LENGTHSCALES} lengthscales, candidates=${GP_SURFACE_CANDIDATE_COUNT}, prediction_grid=${GP_SURFACE_PREDICTION_R_COUNT}x${GP_SURFACE_PREDICTION_GAMMA_COUNT}"
echo "GP salvage:         cancelled brackets kept when relative half-width <= ${GP_SALVAGE_MAX_RELATIVE_HALF_WIDTH} (0 disables)"
echo "Re-certification:   at most ${GP_PROBE_MAX_RECERTIFICATIONS} fresh probe set(s) after a failed certification set (0 = original rule)"
echo "Async extensions:   early_brackets=${GP_ASYNC_EARLY_BRACKETS} (quench/ignite cost ratio ${GP_BRACKET_QUENCH_COST_RATIO})"
echo "GP surface coords:  model R=${GP_SURFACE_INPUT_TRANSFORM_R}, gamma=${GP_SURFACE_INPUT_TRANSFORM_GAMMA}, async reselect every=${GP_SURFACE_MODEL_RESELECT_EVERY}, eligibility refresh=${GP_SURFACE_MODEL_ELIGIBILITY_REFRESH}; seed ladder R=${GP_SURFACE_DESIGN_LADDER_R}, gamma=${GP_SURFACE_DESIGN_LADDER_GAMMA}; interior margin=${GP_SURFACE_INTERIOR_MARGIN}"

# --- actor arguments --------------------------------------------------------
COMMON_ARGS=(
    --gamma-values "${GAMMA_VALUES}"
    --gamma-min "${GAMMA_MIN}"
    --gamma-max "${GAMMA_MAX}"
    --R-min "${R_MIN}"
    --R-max "${R_MAX}"
    --n-R "${N_R}"
    --Av-lo "${AV_LO}"
    --Av-hi "${AV_HI}"
    --bsec-tol "${BISECTION_TOL}"
    --bsec-rel-tol "${BISECTION_REL_TOL}"
    --bsec-max "${BISECTION_MAX}"
    --bsec-probe-start "${BISECTION_PROBE_START}"
    --bsec-probe-ratio "${BISECTION_PROBE_RATIO}"
    --steepness "${STEEPNESS}"
    --aw "${AW_MIN}"
    --domain-half-length "${DOMAIN_HALF_LENGTH}"
    --n-refinements "${N_REFINEMENTS}"
    --max-delta-refinement-level "${MAX_DELTA}"
    --final-time "${FINAL_TIME}"
    --fast-wave-speed "${FAST_WAVE_SPEED}"
    --finite-time-safety-factor "${FINITE_TIME_SAFETY_FACTOR}"
    --time-step "${TIME_STEP}"
    --output-stride "${OUTPUT_STRIDE}"
    --ignition-max-v-threshold "${IGNITION_MAX_V_THRESHOLD}"
    --quench-max-v-threshold "${QUENCH_MAX_V_THRESHOLD}"
    --quench-window "${QUENCH_WINDOW}"
    --ignition-excited-fraction-threshold "${IGNITION_EXCITED_FRACTION_THRESHOLD}"
    --output-dir "${OUTPUT_DIR}"
    --solver-backend "${SOLVER_BACKEND}"
    --solver-binary "${SOLVER_BINARY}"
    --solver-runner "${SOLVER_RUNNER}"
    --solver-mpi-ranks "${SOLVER_MPI_RANKS}"
    --ic-type "${IC_TYPE}"
    --segment-wave-width "${SEGMENT_WAVE_WIDTH}"
    --segment-tail-width "${SEGMENT_TAIL_WIDTH}"
    --detector-x "${DETECTOR_X}"
    --detector-min-excited-area "${DETECTOR_MIN_EXCITED_AREA}"
    --detector-offset-from-radius "${DETECTOR_OFFSET_FROM_RADIUS}"
)
if [ -n "${WAVE_SPEED_CONTINUATION_FILE}" ]; then
    COMMON_ARGS+=(
        --wave-speed-continuation-file "${WAVE_SPEED_CONTINUATION_FILE}"
        --wave-speed-min-reliable-gamma "${WAVE_SPEED_MIN_RELIABLE_GAMMA}"
        --wave-speed-low-gamma-plateau "${WAVE_SPEED_LOW_GAMMA_PLATEAU}"
        --wave-speed-round-up "${WAVE_SPEED_ROUND_UP}"
    )
fi
if [ -n "${R_VALUES}" ]; then
    COMMON_ARGS+=(--R-values "${R_VALUES}")
fi
if [ -n "${GAMMA_WAVE_SPEEDS}" ]; then
    COMMON_ARGS+=(--gamma-wave-speeds "${GAMMA_WAVE_SPEEDS}")
fi
if [ "${PRE_REFINE_IC}" = "true" ]; then
    COMMON_ARGS+=(--pre-refine-ic)
fi
if [[ "${EARLY_EXIT_ON_QUENCH}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--early-exit-on-quench)
fi
if [[ "${GP_BACKGROUND_PROBES_OFF}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--gp-background-probes-off)
fi
if [[ "${EARLY_EXIT_ON_PROPAGATION}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--early-exit-on-propagation)
fi
if [[ "${DETECTOR_REQUIRE_AREA_GROWTH}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--detector-require-area-growth)
fi
if [[ "${USE_DETECTOR_ORACLE}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--use-detector-oracle)
fi

if [[ "${ADAPTIVE_GP}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    if [[ "${GP_PERSISTENT_ADVISOR:-false}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
        COMMON_ARGS+=(--gp-persistent-advisor)
    fi
    COMMON_ARGS+=(
        --adaptive-gp
        --adaptive-gp-mode "${ADAPTIVE_GP_MODE}"
        --adaptive-initial-samples "${ADAPTIVE_INITIAL_SAMPLES}"
        --adaptive-batch-size "${ADAPTIVE_BATCH_SIZE}"
        --adaptive-rounds "${ADAPTIVE_ROUNDS}"
        --adaptive-max-samples "${ADAPTIVE_MAX_SAMPLES}"
        --adaptive-async-fill-cap "${ADAPTIVE_ASYNC_FILL_CAP}"
        --adaptive-async-target-active "${ADAPTIVE_ASYNC_TARGET_ACTIVE}"
        --gp-python "${GP_PYTHON}"
        --gp-surface-script "${GP_SURFACE_SCRIPT}"
        --gp-grid-size "${GP_GRID_SIZE}"
        --gp-acquisition "${GP_ACQUISITION}"
        --gp-integration-grid-size "${GP_INTEGRATION_GRID_SIZE}"
        --gp-variation-weight "${GP_VARIATION_WEIGHT}"
        --gp-focus-relative-uncertainty-target "${GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET}"
        --gp-focus-variation-coverage "${GP_FOCUS_VARIATION_COVERAGE}"
        --gp-acquisition-chunk-size "${GP_ACQUISITION_CHUNK_SIZE}"
        --gp-surface-nipv-backend "${GP_SURFACE_NIPV_BACKEND}"
        --gp-joint-num-restarts "${GP_JOINT_NUM_RESTARTS}"
        --gp-joint-raw-samples "${GP_JOINT_RAW_SAMPLES}"
        --gp-joint-maxiter "${GP_JOINT_MAXITER}"
        --gp-joint-retries "${GP_JOINT_RETRIES}"
        --gp-joint-fallback "${GP_JOINT_FALLBACK}"
        --gp-surface-suggest-min-distance "${GP_SURFACE_SUGGEST_MIN_DISTANCE}"
        --gp-surface-suggest-min-distance-lengthscales "${GP_SURFACE_SUGGEST_MIN_DISTANCE_LENGTHSCALES}"
        --gp-surface-input-transform-r "${GP_SURFACE_INPUT_TRANSFORM_R}"
        --gp-surface-input-transform-gamma "${GP_SURFACE_INPUT_TRANSFORM_GAMMA}"
        --gp-surface-model-reselect-every "${GP_SURFACE_MODEL_RESELECT_EVERY}"
        --gp-surface-design-ladder-r "${GP_SURFACE_DESIGN_LADDER_R}"
        --gp-surface-design-ladder-gamma "${GP_SURFACE_DESIGN_LADDER_GAMMA}"
        --gp-surface-interior-margin "${GP_SURFACE_INTERIOR_MARGIN}"
        --gp-acquisition-value-target "${GP_ACQUISITION_VALUE_TARGET}"
        --adaptive-min-fit-samples "${ADAPTIVE_MIN_FIT_SAMPLES}"
        --gp-salvage-max-relative-half-width "${GP_SALVAGE_MAX_RELATIVE_HALF_WIDTH}"
    )
    if [ -n "${GP_SURFACE_SAMPLE_LIST}" ]; then
        COMMON_ARGS+=(--gp-surface-sample-list "${GP_SURFACE_SAMPLE_LIST}")
    fi
    COMMON_ARGS+=(
        --gp-surface-candidate-count "${GP_SURFACE_CANDIDATE_COUNT}"
        --gp-surface-prediction-r-count "${GP_SURFACE_PREDICTION_R_COUNT}"
        --gp-surface-prediction-gamma-count "${GP_SURFACE_PREDICTION_GAMMA_COUNT}"
        --gp-surface-boundary-tolerance "${GP_SURFACE_BOUNDARY_TOLERANCE}"
        --gp-relative-uncertainty-max-tolerance "${GP_RELATIVE_UNCERTAINTY_MAX_TOLERANCE}"
        --gp-relative-uncertainty-rms-tolerance "${GP_RELATIVE_UNCERTAINTY_RMS_TOLERANCE}"
        --gp-shape-loo-max-tolerance "${GP_SHAPE_LOO_MAX_TOLERANCE}"
        --gp-shape-loo-rms-tolerance "${GP_SHAPE_LOO_RMS_TOLERANCE}"
        --gp-integrated-variance-tolerance "${GP_INTEGRATED_VARIANCE_TOLERANCE}"
        --gp-variance-min-relative-improvement "${GP_VARIANCE_MIN_RELATIVE_IMPROVEMENT}"
        --gp-variance-patience "${GP_VARIANCE_PATIENCE}"
        --gp-stop-hysteresis "${GP_STOP_HYSTERESIS}"
        --gp-probe-every "${GP_PROBE_EVERY}"
        --gp-probe-min-samples "${GP_PROBE_MIN_SAMPLES}"
        --gp-probe-rms-tolerance "${GP_PROBE_RMS_TOLERANCE}"
        --gp-prequential-min-samples "${GP_PREQUENTIAL_MIN_SAMPLES}"
        --gp-prequential-coverage-lower-bound "${GP_PREQUENTIAL_COVERAGE_LOWER_BOUND}"
        --gp-prequential-warmup-factor "${GP_PREQUENTIAL_WARMUP_FACTOR}"
        --gp-seed-sigma-k "${GP_SEED_SIGMA_K}"
        --gp-async-start-min-valid "${GP_ASYNC_START_MIN_VALID}"
        --gp-bracket-quench-cost-ratio "${GP_BRACKET_QUENCH_COST_RATIO}"
        --gp-probe-max-recertifications "${GP_PROBE_MAX_RECERTIFICATIONS}"
        --gp-convergence-r-min "${GP_CONVERGENCE_R_MIN}"
        --gp-noise-floor "${GP_NOISE_FLOOR}"
        --gp-target-transform "${GP_TARGET_TRANSFORM}"
    )
    if [[ "${ADAPTIVE_GP_SURFACE}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
        if [[ "${GP_SURFACE_MODEL_ELIGIBILITY_REFRESH}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-surface-model-eligibility-refresh)
        fi
        if [[ "${GP_PROBE_REQUIRE_READINESS}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-probe-require-readiness)
        fi
        if [[ "${GP_PROBE_READINESS_TRANSITION}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-probe-readiness-transition)
        fi
        if [[ "${GP_SYNC_CERTIFICATION_BATCHES}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-sync-certification-batches)
        fi
        if [[ "${GP_ASYNC_POSTFIT_CERTIFICATION}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-async-postfit-certification)
        fi
        if [[ "${GP_ASYNC_CERTIFICATION_REQUIRE_READINESS}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-async-certification-require-readiness)
        fi
        if [[ "${GP_PROBE_INFLIGHT_PROMOTION}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-probe-inflight-promotion)
        fi
        if [[ "${GP_READINESS_ANTICIPATION}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-readiness-anticipation)
        fi
        if [[ "${GP_ASYNC_EARLY_START}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-async-early-start)
        fi
        if [[ "${GP_ASYNC_EARLY_BRACKETS}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
            COMMON_ARGS+=(--gp-async-early-brackets)
        fi
        COMMON_ARGS+=(--adaptive-gp-surface)
    fi
    if [ -n "${GP_OUT_DIR}" ]; then
        COMMON_ARGS+=(--gp-out-dir "${GP_OUT_DIR}")
    fi
    if [[ "${GP_NO_PLOT}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
        COMMON_ARGS+=(--gp-no-plot)
    fi
    if [[ "${ADAPTIVE_CANCEL_RUNNING_ON_STOP}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
        COMMON_ARGS+=(--adaptive-cancel-running-on-stop)
    fi
    if [[ "${GP_SEEDED_BRACKETS}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
        COMMON_ARGS+=(--gp-seeded-brackets)
    fi
fi
if [[ "${BISECTION_PROBE_LADDER}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--bsec-probe-ladder)
fi
if [[ "${PIN_SOLVER_CORES}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--pin-solver-cores)
fi
if [[ "${FIXED_FINAL_TIME}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES|on|ON)$ ]]; then
    COMMON_ARGS+=(--fixed-final-time)
fi

# ---------------------------------------------------------------------------
# Node occupancy: WORKERS_PER_NODE workers, each driving a SOLVER_MPI_RANKS-way
# solve, so a node runs W*M ranks concurrently. This one parameterisation covers
# both case studies -- radial is 31x1 (many cheap serial solves), 2D is 1x32 or
# 4x8 (few expensive parallel ones) -- and the right split depends on how far a
# single solve strong-scales.
#
CORES_PER_NODE="${CORES_PER_NODE:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"
NODE_RANKS=$(( WORKERS_PER_NODE * SOLVER_MPI_RANKS ))
if [ "${NODE_RANKS}" -gt "${CORES_PER_NODE}" ]; then
    echo "Error: WORKERS_PER_NODE(${WORKERS_PER_NODE}) x SOLVER_MPI_RANKS(${SOLVER_MPI_RANKS})" \
         "= ${NODE_RANKS} exceeds ${CORES_PER_NODE} cores on this node." \
         "Oversubscription here is catastrophic, not merely slow -- MPI ranks busy-wait." >&2
    exit 2
fi
if [ "${NODE_RANKS}" -lt "${CORES_PER_NODE}" ]; then
    echo "Note: using ${NODE_RANKS} of ${CORES_PER_NODE} cores per node" \
         "(${WORKERS_PER_NODE} workers x ${SOLVER_MPI_RANKS} ranks)."
fi
echo "Node occupancy:    ${WORKERS_PER_NODE} workers x ${SOLVER_MPI_RANKS} ranks = ${NODE_RANKS}/${CORES_PER_NODE} cores"
CLIENT_CPUS_PER_TASK=1
CLIENT_SRUN_EXTRA=()
if [ "${NODE_RANKS}" -gt 1 ]; then
    CLIENT_CPUS_PER_TASK="${NODE_RANKS}"
    CLIENT_SRUN_EXTRA+=(--overlap)
fi

SERVER_RESERVED_CORES="${SERVER_RESERVED_CORES:-1}"
SERVER_WORKERS_PER_NODE="${SERVER_WORKERS_PER_NODE:-${WORKERS_PER_NODE}}"
SERVER_CORE_BUDGET=$(( CORES_PER_NODE - SERVER_RESERVED_CORES ))
[ "${SERVER_CORE_BUDGET}" -lt 0 ] && SERVER_CORE_BUDGET=0
# Every worker needs at least one core; drop workers only if the budget cannot
# seat them all.
if [ "${SERVER_WORKERS_PER_NODE}" -gt "${SERVER_CORE_BUDGET}" ]; then
    SERVER_WORKERS_PER_NODE="${SERVER_CORE_BUDGET}"
fi
if [ "${SERVER_WORKERS_PER_NODE}" -eq 0 ] && [ "${#client_nodes[@]}" -eq 0 ]; then
    echo "Error: the server node can host no workers (${CORES_PER_NODE} cores minus" \
         "${SERVER_RESERVED_CORES} reserved) and there are no client nodes, so" \
         "nothing would run. Allocate more nodes or set SERVER_RESERVED_CORES=0." >&2
    exit 2
fi
# Report the split so an uneven layout is visible in the log rather than implied.
SERVER_SLICES=$(awk -v b="${SERVER_CORE_BUDGET}" -v w="${SERVER_WORKERS_PER_NODE}" \
    -v m="${SOLVER_MPI_RANKS}" \
    'BEGIN { if (w <= 0) { printf "none"; exit }
             u = (b < w * m ? b : w * m)          # mirrors worker_core_slice()
             base = int(u / w); rem = u % w; sep = ""
             for (i = 0; i < w; i++) { printf "%s%d", sep, base + (i < rem ? 1 : 0); sep = "+" } }')
echo "Server node:       ${SERVER_WORKERS_PER_NODE} workers, ranks ${SERVER_SLICES}" \
     "= ${SERVER_CORE_BUDGET}/${CORES_PER_NODE} cores (${SERVER_RESERVED_CORES} reserved for scheduler + GP)"

# CAF scheduler threads must EXCEED the number of local worker actors on the
# node. run_bisection() runs synchronously inside the worker's message handler,
# so a busy worker occupies one scheduler thread for the whole multi-hour
# sample. With the old fixed value of 4, a node hosting 4 workers had zero
# threads left: the SERVER actor could not run, so it could not answer a
# connecting client, and that client's workers were never spawned until a local
# worker happened to finish (measured: ~73 min into a 4x8 run, and visible in
# job 5816276 as dispatches 7,8,9 arriving only after the 4th result). In steady
# state it also starved dispatch and the GP rounds, which is a per-layout
# penalty that falls entirely on the multi-worker configurations.
CAF_SERVER_THREADS="${CAF_SERVER_THREADS:-$(( SERVER_WORKERS_PER_NODE + 4 ))}"
CAF_CLIENT_THREADS="${CAF_CLIENT_THREADS:-$(( WORKERS_PER_NODE + 4 ))}"

cd "${SCRIPT_DIR}"

echo "Starting server on ${server_node}..."
"${ACTOR_BINARY}" \
    --server-mode \
    --port "${PORT}" \
    --n-workers "${SERVER_WORKERS_PER_NODE}" \
    --worker-core-budget "${SERVER_CORE_BUDGET}" \
    "${COMMON_ARGS[@]}" \
    --caf.scheduler.max-threads="${CAF_SERVER_THREADS}" &
SERVER_PID=$!

sleep 10

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    SERVER_STATUS=0
    wait "${SERVER_PID}" || SERVER_STATUS=$?
    echo "Error: actor server exited during startup with status ${SERVER_STATUS}." >&2
    echo "       (see the log above; an unhandled CAF exception still exits 0)" >&2
    [ "${SERVER_STATUS}" -eq 0 ] && SERVER_STATUS=1
    exit "${SERVER_STATUS}"
fi

for node in "${client_nodes[@]}"; do
    echo "Starting client on ${node}... (cpus-per-task=${CLIENT_CPUS_PER_TASK})"
    "${SRUN_BIN}" --nodes=1 --ntasks=1 --cpus-per-task="${CLIENT_CPUS_PER_TASK}" --exact \
        "${CLIENT_SRUN_EXTRA[@]}" \
        --nodelist="${node}" \
        --export=ALL,LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        bash -c "${MPI_SINGLETON_ENV_CLEANER}" bash \
        "${ACTOR_BINARY}" \
        --host "${server_node}" \
        --port "${PORT}" \
        --n-workers "${WORKERS_PER_NODE}" \
        --worker-core-budget "${NODE_RANKS}" \
        --caf.scheduler.max-threads="${CAF_CLIENT_THREADS}" &
done

SERVER_STATUS=0
wait "${SERVER_PID}" || SERVER_STATUS=$?
wait
echo "All processes finished."

if command -v seff >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
    env -u LD_LIBRARY_PATH seff "${SLURM_JOB_ID}"
else
    echo "Skipping Slurm efficiency report: seff is unavailable."
fi

if [ "${SERVER_STATUS}" -ne 0 ]; then
    echo "Error: actor server exited with status ${SERVER_STATUS}." >&2
    exit "${SERVER_STATUS}"
fi
