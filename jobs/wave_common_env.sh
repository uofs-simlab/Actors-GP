# Wave-segment problem: domain, solver, bisection, and GP settings shared by all runs.
export IC_TYPE=wave_segment
export SEGMENT_WAVE_WIDTH=5.0
export SEGMENT_TAIL_WIDTH=20.0
export STEEPNESS=1.0
export AW_MIN=0.5
export DOMAIN_HALF_LENGTH=100.0
export N_REFINEMENTS=5
export MAX_DELTA=4
export PRE_REFINE_IC=true
export SOLVER_BACKEND=bdf
export SOLVER_TARGET_OVERRIDE=parallel_bdf_fhn
export TIME_STEP=5e-2

# --- observation window -----------------------------------------------------
export FIXED_FINAL_TIME=true
export FINAL_TIME=180.0
export FAST_WAVE_SPEED=0.0
export FAST_WAVE_SPEED_FALLBACK=0.0
export USE_DETECTOR_ORACLE=true
export DETECTOR_X=50.0
export DETECTOR_MIN_EXCITED_AREA=1.0

# --- the box ----------------------------------------------------------------
export R_MIN=5.0
export R_MAX=80.0
export GAMMA_VALUES=0.0075
export GAMMA_MIN=0.0075
export GAMMA_MAX=0.015
export GAMMA_WAVE_SPEEDS=
export WAVE_SPEED_CONTINUATION_FILE=

# --- bisection --------------------------------------------------------------
export AV_LO=0.0
# Largest observed threshold: 3.58, at the corner (l, gamma) = (5, 0.015).
export AV_HI=5.0
export BISECTION_TOL=1e-4
export BISECTION_REL_TOL="${BISECTION_REL_TOL:-0.01}"
export BISECTION_MAX=50
# Plain bisection on [0, Av_hi].
export BISECTION_PROBE_LADDER=false
export BISECTION_PROBE_START=1.0
export BISECTION_PROBE_RATIO=2.0

# --- surrogate --------------------------------------------------------------
export ADAPTIVE_GP=true
export ADAPTIVE_GP_SURFACE=true
export GP_PYTHON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/gp_botorch_env/bin/python"
export GP_SURFACE_SCRIPT="${ACTOR_DIR}/gp/gp_avstar_surface_botorch.py"
export GP_ACQUISITION=variation-nipv
export GP_INTEGRATION_GRID_SIZE=1024
export GP_VARIATION_WEIGHT=0.5
export GP_ACQUISITION_CHUNK_SIZE=128
export GP_SURFACE_CANDIDATE_COUNT=4096
export GP_SURFACE_PREDICTION_R_COUNT=160
export GP_SURFACE_PREDICTION_GAMMA_COUNT=100
export GP_SURFACE_SUGGEST_MIN_DISTANCE=0.01
export GP_SURFACE_SUGGEST_MIN_DISTANCE_LENGTHSCALES=0.15
export GP_PREQUENTIAL_MIN_SAMPLES="${GP_PREQUENTIAL_MIN_SAMPLES:-12}"
export GP_PREQUENTIAL_COVERAGE_LOWER_BOUND="${GP_PREQUENTIAL_COVERAGE_LOWER_BOUND:-0.70}"
export GP_PREQUENTIAL_WARMUP_FACTOR="${GP_PREQUENTIAL_WARMUP_FACTOR:-1.5}"

export GP_SURFACE_INPUT_TRANSFORM_R=auto
export GP_SURFACE_INPUT_TRANSFORM_GAMMA=auto
export GP_SURFACE_DESIGN_LADDER_R=blend
export GP_SURFACE_DESIGN_LADDER_GAMMA=blend
export GP_SURFACE_INTERIOR_MARGIN=0.05
export GP_SURFACE_BOUNDARY_TOLERANCE=0.02
export GP_SALVAGE_MAX_RELATIVE_HALF_WIDTH=0.25
export GP_NOISE_FLOOR="${GP_NOISE_FLOOR:-0.001}"
export GP_TARGET_TRANSFORM=log
export GP_NO_PLOT=true
export GP_SEEDED_BRACKETS="${GP_SEEDED_BRACKETS:-false}"
export ADAPTIVE_MIN_FIT_SAMPLES=0

# Overrides of the radial defaults in ignition_actor_adaptive_gp.sh
export GP_CONVERGENCE_R_MIN=0
export GP_FOCUS_RELATIVE_UNCERTAINTY_TARGET=0
export GP_FOCUS_VARIATION_COVERAGE=1
export GP_GRID_SIZE=2000
export GP_SEED_SIGMA_K=3.0
export N_R=40
export OUTPUT_STRIDE=0
export OMPI_MCA_sharedfp=individual
