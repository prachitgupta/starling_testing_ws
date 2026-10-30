#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=12:00:00
#SBATCH --job-name=hrrt-transformer-2-0
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/projects/bhkj/$USER/starling_multiple_trajectory_idea}
PACKAGE_ROOT=$PROJECT_ROOT/src/llm_vision_planner
CONFIG=${CONFIG:-$PACKAGE_ROOT/fine_tuning/configs/hrrt_waypoint_transformer_2_0.yaml}
TRAIN_CSV=${TRAIN_CSV:-$PACKAGE_ROOT/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv}
VALIDATION_CSV=${VALIDATION_CSV:-$PACKAGE_ROOT/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$PACKAGE_ROOT/fine_tuning/outputs/hrrt_waypoint_transformer_2_0}
SAFETY_RADIUS_THRESHOLD_M=${SAFETY_RADIUS_THRESHOLD_M:-0.40}
MAX_EPOCHS=${MAX_EPOCHS:-100}
CORRIDOR_LOSS_WEIGHT=${CORRIDOR_LOSS_WEIGHT:-5.0}
CORRIDOR_MARGIN_M=${CORRIDOR_MARGIN_M:-0.20}
CLEARANCE_BIN_LOSS_WEIGHT=${CLEARANCE_BIN_LOSS_WEIGHT:-1.0}
SQUARED_TAIL_LOSS_WEIGHT=${SQUARED_TAIL_LOSS_WEIGHT:-1.0}
TRAIN_ENV_DIR=${TRAIN_ENV_DIR:-/projects/bhkj/$USER/hrrt_waypoint_transformer_env}

cd "$PROJECT_ROOT"
mkdir -p logs

module purge
module load cray-python

export HF_HOME=${HF_HOME:-/projects/bhkj/$USER/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

python3 -m venv "$TRAIN_ENV_DIR"
source "$TRAIN_ENV_DIR/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch transformers scipy numpy pyyaml sentencepiece

test -s "$CONFIG"
test -s "$TRAIN_CSV"
test -s "$VALIDATION_CSV"

python3 -u "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.py" \
  --config "$CONFIG" \
  --train "$TRAIN_CSV" \
  --validation "$VALIDATION_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --corridor-loss-weight "$CORRIDOR_LOSS_WEIGHT" \
  --corridor-margin-m "$CORRIDOR_MARGIN_M" \
  --clearance-bin-loss-weight "$CLEARANCE_BIN_LOSS_WEIGHT" \
  --squared-tail-loss-weight "$SQUARED_TAIL_LOSS_WEIGHT" \
  --safety-radius-threshold-m "$SAFETY_RADIUS_THRESHOLD_M"

tar -czf "$OUTPUT_DIR.tar.gz" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
printf 'Transformer 2.0 training complete: %s\n' "$OUTPUT_DIR"
