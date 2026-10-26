#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=12:00:00
#SBATCH --job-name=hrrt-transformer-verify
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/projects/bhkj/$USER/starling_multiple_trajectory_idea}
PACKAGE_ROOT=$PROJECT_ROOT/src/llm_vision_planner
CONFIG=${CONFIG:-$PACKAGE_ROOT/fine_tuning/configs/hrrt_waypoint_transformer.yaml}
TRAIN_CSV=${TRAIN_CSV:-$PACKAGE_ROOT/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv}
VALIDATION_CSV=${VALIDATION_CSV:-$PACKAGE_ROOT/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv}
TEST_CSV=${TEST_CSV:-$PACKAGE_ROOT/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_test.csv}
SAFETY_RADIUS_THRESHOLD_M=${SAFETY_RADIUS_THRESHOLD_M:-0.40}
MAX_EPOCHS=${MAX_EPOCHS:-100}
TINY_SAMPLES=${TINY_SAMPLES:-8}
TINY_UPDATES=${TINY_UPDATES:-250}
TINY_LR_MULTIPLIER=${TINY_LR_MULTIPLIER:-5.0}
REQUIRED_OVERFIT_REDUCTION=${REQUIRED_OVERFIT_REDUCTION:-0.50}
RUN_NAME=${RUN_NAME:-verification-${SLURM_JOB_ID}}
RUN_ROOT=${RUN_ROOT:-$PACKAGE_ROOT/fine_tuning/outputs/$RUN_NAME}
MODEL_OUTPUT=$RUN_ROOT/model
REPORT_DIR=$RUN_ROOT/reports
TRAIN_ENV_DIR=${TRAIN_ENV_DIR:-/projects/bhkj/$USER/hrrt_waypoint_transformer_env}

cd "$PROJECT_ROOT"
mkdir -p logs "$REPORT_DIR"

module purge
module load cray-python

export HF_HOME=${HF_HOME:-/projects/bhkj/$USER/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

python3 -m venv "$TRAIN_ENV_DIR"
source "$TRAIN_ENV_DIR/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch transformers scipy numpy pyyaml sentencepiece matplotlib

python3 -u "$PACKAGE_ROOT/fine_tuning/verification/scripts/verify_transformer_components.py" \
  --stage preflight \
  --config "$CONFIG" \
  --train "$TRAIN_CSV" \
  --validation "$VALIDATION_CSV" \
  --test "$TEST_CSV" \
  --output "$REPORT_DIR/preflight.json" \
  --device cuda:0 \
  --tiny-samples "$TINY_SAMPLES" \
  --tiny-updates "$TINY_UPDATES" \
  --tiny-lr-multiplier "$TINY_LR_MULTIPLIER" \
  --required-overfit-reduction "$REQUIRED_OVERFIT_REDUCTION"

python3 -u "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_waypoint_transformer.py" \
  --config "$CONFIG" \
  --train "$TRAIN_CSV" \
  --validation "$VALIDATION_CSV" \
  --output-dir "$MODEL_OUTPUT" \
  --max-epochs "$MAX_EPOCHS" \
  --safety-radius-threshold-m "$SAFETY_RADIUS_THRESHOLD_M"

python3 -u "$PACKAGE_ROOT/fine_tuning/verification/scripts/verify_transformer_components.py" \
  --stage postflight \
  --config "$CONFIG" \
  --train "$TRAIN_CSV" \
  --validation "$VALIDATION_CSV" \
  --test "$TEST_CSV" \
  --model-output "$MODEL_OUTPUT" \
  --output "$REPORT_DIR/postflight.json" \
  --device cuda:0

python3 "$PACKAGE_ROOT/fine_tuning/verification/scripts/plot_verification_results.py" \
  --run-root "$RUN_ROOT"

tar -czf "$RUN_ROOT.tar.gz" -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")"
printf 'Verification complete: %s\n' "$RUN_ROOT"
