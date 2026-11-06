# Radial problem: stimulus, oracle, and site settings.
unset AW BETA CAF_SERVER_THREADS CAF_CLIENT_THREADS SERVER_WORKERS_PER_NODE SRUN
export AW_MIN=0.0 IC_TYPE=smooth_disk N_R=40 R_VALUES=
export FIXED_FINAL_TIME=false FINAL_TIME= FAST_WAVE_SPEED=
export FAST_WAVE_SPEED_FALLBACK=0.50
export WAVE_SPEED_TABLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ignition-actor/tables/wave_speeds.csv"
export IGNITION_MAX_V_THRESHOLD=0.5 IGNITION_EXCITED_FRACTION_THRESHOLD=0.01
export USE_DETECTOR_ORACLE=false SEGMENT_WAVE_WIDTH=5.0 SEGMENT_TAIL_WIDTH=20.0
export DETECTOR_X=50.0 DETECTOR_MIN_EXCITED_AREA=1.0
export DEAL_II_DIR="${DEAL_II_DIR:?set DEAL_II_DIR to the deal.II 9.6.0 installation prefix (see README)}"
export PETSC_OPTIONS='-options_left false'
