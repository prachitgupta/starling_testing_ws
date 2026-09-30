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

PROJECT_ROOT=${PROJECT_ROOT:-/projects/bhkj/$USER/starling_testing_ws}
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
OUTPUT="$PACKAGE_ROOT/fine_tuning/outputs/llama31_8b_hrrt_lora"
python -u "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_peft.py" \
  --train "$PACKAGE_ROOT/fine_tuning/datasets/hrrt_sft_train.csv" \
  --validation "$PACKAGE_ROOT/fine_tuning/datasets/hrrt_sft_validation.csv" \
  --output-dir "$OUTPUT" \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 8 \
  --qlora

tar -czf "${OUTPUT}.tar.gz" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
