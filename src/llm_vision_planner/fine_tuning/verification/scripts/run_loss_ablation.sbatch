#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@180
#SBATCH --job-name=hrrt-loss-study
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/projects/bhkj/$USER/starling_multiple_trajectory_idea}
PACKAGE_ROOT=$PROJECT_ROOT/src/llm_vision_planner
STUDY_ROOT=${STUDY_ROOT:?Set STUDY_ROOT to the initialized study directory}
TRAIN_ENV_DIR=${TRAIN_ENV_DIR:-/projects/bhkj/$USER/hrrt_waypoint_transformer_env}
SCRIPT=$PACKAGE_ROOT/fine_tuning/verification/scripts/run_loss_ablation.py
WORKER=$PACKAGE_ROOT/fine_tuning/verification/scripts/run_loss_ablation.sbatch
cd "$PROJECT_ROOT"
module purge
module load cray-python
source "$TRAIN_ENV_DIR/bin/activate"
export HF_HOME=${HF_HOME:-/projects/bhkj/$USER/hf_cache}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
set +e
srun python3 -u "$SCRIPT" advance --study-root "$STUDY_ROOT"
RESULT=$?
set -e
if [ "$RESULT" -ne 0 ] && [ "$RESULT" -ne 75 ]; then
  printf 'Study stopped on failure. Inspect state.json/log; resubmit this same study after resolving it.\n'
  exit "$RESULT"
fi
STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$STUDY_ROOT/state.json")
if [ "$STATUS" != complete ]; then
  NEXT=$(sbatch --parsable --export=ALL,STUDY_ROOT="$STUDY_ROOT",PROJECT_ROOT="$PROJECT_ROOT",TRAIN_ENV_DIR="$TRAIN_ENV_DIR" "$WORKER")
  printf '%s\n' "$NEXT" >> "$STUDY_ROOT/job_ids.txt"
  printf 'Queued continuation: %s\n' "$NEXT"
else
  printf 'Study complete. Dashboard: %s/dashboard/index.html\n' "$STUDY_ROOT"
fi
