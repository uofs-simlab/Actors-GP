#!/bin/bash
# Method settings of one wave-segment run; calls run_surface_wave.sh.
#   run_arm_wave.sh <async|sync> [label-suffix]
# Needs WAVE_WORKERS, WAVE_WORKERS_PER_NODE, WAVE_RANKS, BRACKETS (informed or global) and RESULT_ROOT.
# Layouts: 2 workers x 16 ranks per node (W = 2, 4, 8, 16) or 1 worker x 32 ranks (W = 1, 8).
set -euo pipefail
arm="${1:?async or sync}"; extra="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$arm" in async) MODE=asynchronous;; sync) MODE=synchronous;; *) exit 2;; esac
for setting in ${!GP_@} ${!ADAPTIVE_@}; do unset "$setting"; done
unset SAMPLE_LIST SOLVER_BINARY SOLVER_TARGET_OVERRIDE DETECTOR_REQUIRE_AREA_GROWTH
source "$REPO_ROOT/jobs/wave_physical_env.sh"
export MODE WORKERS_TOTAL="${WAVE_WORKERS:-8}" WORKERS_PER_NODE="${WAVE_WORKERS_PER_NODE:-2}" RANKS="${WAVE_RANKS:-16}" SEED=16 BUDGET=64
case "$WORKERS_TOTAL" in 1|2|4|8|16) ;; *) echo "WAVE_WORKERS must be 1, 2, 4, 8 or 16" >&2; exit 2;; esac
case "$WORKERS_PER_NODE x $RANKS" in "2 x 16"|"1 x 32") ;; *) echo "layout must be 2 x 16 or 1 x 32" >&2; exit 2;; esac
export SERVER_WORKERS_PER_NODE="$WORKERS_PER_NODE"   # the coordinator's node hosts as many workers as the others
RANK_TAG=""; [ "$RANKS" = 16 ] || RANK_TAG="_r${RANKS}"
BRK="${BRACKETS:-informed}"
case "$BRK" in informed) seeded=true; BRK_TAG="";; global) seeded=false; BRK_TAG="_global";; *) exit 2;; esac
export SERVER_RESERVED_CORES=1

# Scheduling policy
export GP_PROBE_READINESS_TRANSITION=true GP_PROBE_REQUIRE_READINESS=true
export GP_SYNC_CERTIFICATION_BATCHES=false
export GP_ASYNC_POSTFIT_CERTIFICATION=true GP_ASYNC_EARLY_START=true
export GP_ASYNC_CERTIFICATION_REQUIRE_READINESS=true GP_SURFACE_MODEL_ELIGIBILITY_REFRESH=true
export VALUE_TARGET=0.90 GP_ACQUISITION_VALUE_TARGET=0.90
if [ "$MODE" = synchronous ]; then
  export GP_SYNC_CERTIFICATION_BATCHES=true
  export GP_ASYNC_POSTFIT_CERTIFICATION=false GP_ASYNC_EARLY_START=false
  export GP_ASYNC_CERTIFICATION_REQUIRE_READINESS=false GP_SURFACE_MODEL_ELIGIBILITY_REFRESH=false
  export VALUE_TARGET=0 GP_ACQUISITION_VALUE_TARGET=0
fi
export GP_ASYNC_START_MIN_VALID=3   # first Async decision after three thresholds

# GP and readiness bounds
export GP_SURFACE_NIPV_BACKEND=analytic GP_PERSISTENT_ADVISOR=true
export GP_SEEDED_BRACKETS="$seeded" GP_SEED_SIGMA_K=3.0
export GP_PYTHON="$REPO_ROOT/gp_botorch_env/bin/python"
export BISECTION_REL_TOL=0.01 STOP_MAX=0.10 STOP_RMS=0.03
export LOO_MAX=0.12 LOO_RMS=0.035 GP_NOISE_FLOOR=0.003
export ADVISOR_SCRIPT="$REPO_ROOT/ignition-actor/gp/gp_advisor.py"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/ignition-paper-mpl BUILD_JOBS=6 PYTHONDONTWRITEBYTECODE=1

# Probe certification: six probes at 1% RMS, at most two fresh sets after a miss
export GP_PROBE_EVERY=3 GP_PROBE_MIN_SAMPLES=6 GP_PROBE_RMS_TOLERANCE=0.01
export GP_PREQUENTIAL_MIN_SAMPLES=12 GP_PREQUENTIAL_COVERAGE_LOWER_BOUND=0.70
export GP_PREQUENTIAL_WARMUP_FACTOR=1.5
export GP_PROBE_INFLIGHT_PROMOTION=true GP_READINESS_ANTICIPATION=true
export GP_BACKGROUND_PROBES_OFF=true GP_PROBE_MAX_RECERTIFICATIONS=2

# Async uses informed intervals before the initial design ends, when predicted to be cheaper.
EXT_SUFFIX=""
export GP_ASYNC_EARLY_BRACKETS=false
if [ "$arm" = async ] && [ "$BRK" = informed ]; then
  export GP_ASYNC_EARLY_BRACKETS=true; EXT_SUFFIX="_eb"
fi
export GP_BRACKET_QUENCH_COST_RATIO=0.16   # non-igniting to igniting solve time

# Early-out: non-ignition and propagation exits
export EARLY_EXIT_ON_QUENCH=true QUENCH_MAX_V_THRESHOLD=0.05 QUENCH_WINDOW=10.0
export EARLY_EXIT_ON_PROPAGATION=true

export LABEL_SUFFIX="_antic_${arm}${BRK_TAG}${RANK_TAG}_fast_bgtrue${EXT_SUFFIX}${extra}"
echo "layout: W=${WORKERS_TOTAL}, ${WORKERS_PER_NODE} worker(s) x ${RANKS} ranks per node"
export RESULT_ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
bash "$REPO_ROOT/jobs/run_surface_wave.sh"
