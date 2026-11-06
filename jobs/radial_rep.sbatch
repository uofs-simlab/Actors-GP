#!/bin/bash
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --constraint=gold-6226r
#SBATCH --exclude=platocpu[051,058,064]
#SBATCH --output=%x-%j.out
# One radial run on one node. Submit from this folder:
#   sbatch radial_rep.sbatch <async|sync> <W> <informed|global> <rep>
set -euo pipefail
unset BRACKET_DRY_RUN RESULT_ROOT PORT
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:?}/.." && pwd)"
test -f "$REPO_ROOT/jobs/radial_rep.sbatch" || { echo "submit radial_rep.sbatch from <repo>/jobs" >&2; exit 2; }
arm="${1:?arm}"; export WORKERS_OVERRIDE="${2:?W}" BRACKETS="${3:?brackets}"; rep="${4:?rep}"
export RESULT_ROOT="$REPO_ROOT/jobs/radial"
mode=$([ "$arm" = async ] && echo asynchronous || echo synchronous)
early=false; [ "$arm" = async ] && [ "$BRACKETS" = informed ] && early=true
ext=""; [ "$early" = true ] && ext="_eb"
run="$RESULT_ROOT/results/sched_${mode}_W${WORKERS_OVERRIDE}_antic_${arm}_${BRACKETS}_noprior_fast_bgtrue${ext}_det40h_rep${rep}_job${SLURM_JOB_ID}"
test ! -e "$run"
mkdir -p "$run/provenance"
# Record the sources and the allocation of this run.
( cd "$REPO_ROOT" && sha256sum ignition-actor/src/*.cc ignition-actor/src/*.h ignition-actor/gp/*.py \
    ignition-actor/*.sh ignition-actor/CMakeLists.txt solver/parallel_bdf_fhn.cc \
    solver/IgnitionDiagnostics.h jobs/*.sh jobs/radial_rep.sbatch \
  ) > "$run/provenance/campaign_sha256sums.txt"
scontrol show job "$SLURM_JOB_ID" > "$run/provenance/slurm_job.txt" 2>/dev/null || true
echo "node=$(hostname) exclusive=yes arm=$arm W=$WORKERS_OVERRIDE brackets=$BRACKETS rep=$rep early_brackets=$early" > "$run/provenance/allocation.txt"
bash "$REPO_ROOT/jobs/run_arm_radial.sh" "$arm" "_rep${rep}"
grep -q '^=== DONE:' "$run/run.log"
