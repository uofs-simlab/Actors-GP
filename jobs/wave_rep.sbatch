#!/bin/bash
#SBATCH --account=<account>
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --constraint=gold-6226r
#SBATCH --exclude=platocpu[051,058,064]
#SBATCH --output=%x-%j.out
# One wave-segment run on N = 1, 2, 4 or 8 nodes. Submit from this folder:
#   sbatch --nodes=N --ntasks=32N wave_rep.sbatch <async|sync> <rep> [informed|global] [16|32]
# 16 ranks per solve gives W = 2N workers, and 32 ranks gives W = N.
set -euo pipefail
unset WAVE_BRACKET_DRY_RUN RESULT_ROOT PORT
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:?}/.." && pwd)"
test -f "$REPO_ROOT/jobs/wave_rep.sbatch" || { echo "submit wave_rep.sbatch from <repo>/jobs" >&2; exit 2; }
arm="${1:?arm}"; rep="${2:?rep}"
export BRACKETS="${3:-informed}"
case "$BRACKETS" in informed) BRK_TAG="";; global) BRK_TAG="_global";; *) echo "brackets must be informed or global" >&2; exit 2;; esac
case "${SLURM_JOB_NUM_NODES:?}" in 1|2|4|8) ;; *) echo "wave_rep needs 1, 2, 4 or 8 nodes" >&2; exit 2;; esac
export WAVE_RANKS="${4:-16}"
case "$WAVE_RANKS" in 16) export WAVE_WORKERS_PER_NODE=2; RANK_TAG="";; 32) export WAVE_WORKERS_PER_NODE=1; RANK_TAG="_r32";; *) echo "ranks must be 16 or 32" >&2; exit 2;; esac
export WAVE_WORKERS=$(( WAVE_WORKERS_PER_NODE * SLURM_JOB_NUM_NODES ))
export RESULT_ROOT="$REPO_ROOT/jobs/wave"
mode=$([ "$arm" = async ] && echo asynchronous || echo synchronous)
early=false; [ "$arm" = async ] && [ "$BRACKETS" = informed ] && early=true
ext=""; [ "$early" = true ] && ext="_eb"
run="$RESULT_ROOT/results/wsegsurf_${mode}_W${WAVE_WORKERS}_antic_${arm}${BRK_TAG}${RANK_TAG}_fast_bgtrue${ext}_rep${rep}_job${SLURM_JOB_ID}"
test ! -e "$run"
mkdir -p "$run/provenance"
# Record the sources and the allocation of this run.
( cd "$REPO_ROOT" && sha256sum ignition-actor/src/*.cc ignition-actor/src/*.h ignition-actor/gp/*.py \
    ignition-actor/*.sh ignition-actor/CMakeLists.txt solver/parallel_bdf_fhn.cc \
    solver/IgnitionDiagnostics.h jobs/*.sh jobs/wave_rep.sbatch \
  ) > "$run/provenance/campaign_sha256sums.txt"
scontrol show job "$SLURM_JOB_ID" > "$run/provenance/slurm_job.txt" 2>/dev/null || true
echo "nodes=$SLURM_JOB_NODELIST exclusive=yes arm=$arm brackets=$BRACKETS rep=$rep early_brackets=$early ranks_per_solve=$WAVE_RANKS workers_per_node=$WAVE_WORKERS_PER_NODE" > "$run/provenance/allocation.txt"
bash "$REPO_ROOT/jobs/run_arm_wave.sh" "$arm" "_rep${rep}"
grep -q '^=== DONE:' "$run/run.log"
