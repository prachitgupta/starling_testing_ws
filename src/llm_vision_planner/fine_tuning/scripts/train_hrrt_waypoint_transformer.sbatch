#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=12:00:00
#SBATCH --job-name=hrrt-waypoint-transformer
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/projects/bhkj/$USER/starling_multiple_trajectory_idea}
cd "$PROJECT_ROOT"
mkdir -p logs

module purge
module load cray-python

export HF_HOME=${HF_HOME:-/projects/bhkj/$USER/hf_cache}
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

TRAIN_ENV_DIR=${TRAIN_ENV_DIR:-/projects/bhkj/$USER/hrrt_waypoint_transformer_env}
python3 -m venv "$TRAIN_ENV_DIR"
source "$TRAIN_ENV_DIR/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch transformers scipy numpy pyyaml sentencepiece

PACKAGE_ROOT=src/llm_vision_planner
SAFETY_RADIUS_THRESHOLD_M=${SAFETY_RADIUS_THRESHOLD_M:-0.40}
MAX_EPOCHS=${MAX_EPOCHS:-}
TRAIN_CSV=${TRAIN_CSV:-}
VALIDATION_CSV=${VALIDATION_CSV:-}
DATASET_ARGS=()
if [ -n "$MAX_EPOCHS" ]; then
  DATASET_ARGS+=(--max-epochs "$MAX_EPOCHS")
fi
if [ -n "$TRAIN_CSV" ]; then
  DATASET_ARGS+=(--train "$TRAIN_CSV")
fi
if [ -n "$VALIDATION_CSV" ]; then
  DATASET_ARGS+=(--validation "$VALIDATION_CSV")
fi
python3 -u "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_waypoint_transformer.py" \
  --config "$PACKAGE_ROOT/fine_tuning/configs/hrrt_waypoint_transformer.yaml" \
  "${DATASET_ARGS[@]}" \
  --safety-radius-threshold-m "$SAFETY_RADIUS_THRESHOLD_M"

OUTPUT="$PACKAGE_ROOT/fine_tuning/outputs/hrrt_waypoint_transformer"
tar -czf "${OUTPUT}.tar.gz" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
