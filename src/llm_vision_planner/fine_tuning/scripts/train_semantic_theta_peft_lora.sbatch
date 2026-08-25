#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --job-name=semantic-theta-peft
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd /projects/bhkj/$USER/starling_testing_ws
mkdir -p logs

module purge
module load cray-python

export HF_HOME=/projects/bhkj/$USER/hf_cache
export TRANSFORMERS_CACHE=/projects/bhkj/$USER/hf_cache
export WANDB_DISABLED=true

SEMANTIC_ENV_DIR=/projects/bhkj/$USER/semantic_theta_peft_env
python -m venv "$SEMANTIC_ENV_DIR"
source "$SEMANTIC_ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch datasets trl transformers accelerate peft matplotlib

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

SEMANTIC_OUTPUT=src/llm_vision_planner/fine_tuning/outputs/llama31_8b_semantic_theta_lora
python -u src/llm_vision_planner/fine_tuning/scripts/train_peft.py \
  --dataset src/llm_vision_planner/fine_tuning/datasets/semantic_theta_expert_dataset.csv \
  --output-dir "$SEMANTIC_OUTPUT" \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 4

tar -czf "${SEMANTIC_OUTPUT}.tar.gz" -C "$(dirname "$SEMANTIC_OUTPUT")" "$(basename "$SEMANTIC_OUTPUT")"
