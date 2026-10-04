#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --job-name=hrrt-peft
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/projects/bhkj/$USER/starling_multiple_trajectory_idea}
cd "$PROJECT_ROOT"
mkdir -p logs

module purge
module load cray-python

export HF_HOME=${HF_HOME:-/projects/bhkj/$USER/hf_cache}
export TRANSFORMERS_CACHE="$HF_HOME"
export WANDB_DISABLED=true

HRRT_ENV_DIR=${HRRT_ENV_DIR:-/projects/bhkj/$USER/hrrt_peft_env}
python -m venv "$HRRT_ENV_DIR"
source "$HRRT_ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch datasets transformers accelerate peft bitsandbytes matplotlib

PACKAGE_ROOT=src/llm_vision_planner
DISTILLATION_MODE=${DISTILLATION_MODE:-dss}
DATASET_ROOT=${DATASET_ROOT:-$PACKAGE_ROOT/fine_tuning/datasets/$DISTILLATION_MODE}
OUTPUT=${OUTPUT:-$PACKAGE_ROOT/fine_tuning/outputs/llama31_8b_hrrt_lora_$DISTILLATION_MODE}
python -u "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_peft.py" \
  --train "$DATASET_ROOT/hrrt_sft_train.csv" \
  --validation "$DATASET_ROOT/hrrt_sft_validation.csv" \
  --output-dir "$OUTPUT" \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --distillation-mode "$DISTILLATION_MODE" \
  --max-seq-length 2048 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 8 \
  --learning-rate 5e-5 \
  --lora-r 128 \
  --lora-alpha 256 \
  --lora-dropout 0.05 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100 \
  --seed 42 \
  --qlora

tar -czf "${OUTPUT}.tar.gz" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
