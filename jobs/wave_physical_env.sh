# Wave-segment problem: oracle and site settings.
unset AW BETA SRUN
export R_VALUES=
export WAVE_SPEED_TABLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ignition-actor/tables/wave_speeds.csv"
export IGNITION_MAX_V_THRESHOLD=0.5 IGNITION_EXCITED_FRACTION_THRESHOLD=0.01
export WAVE_SPEED_MIN_RELIABLE_GAMMA=0.00048828125 WAVE_SPEED_LOW_GAMMA_PLATEAU=0.5192
export WAVE_SPEED_ROUND_UP=0.0001 FINITE_TIME_SAFETY_FACTOR=0.9
export DEAL_II_DIR="${DEAL_II_DIR:?set DEAL_II_DIR to the deal.II 9.6.0 installation prefix (see README)}" PETSC_OPTIONS='-options_left false'
export CORES_PER_NODE=32 SERVER_WORKERS_PER_NODE=2 CAF_SERVER_THREADS=6 CAF_CLIENT_THREADS=6
