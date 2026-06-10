#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --job-name=rrt-peft
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd /projects/bhkj/$USER/starling_testing_ws
mkdir -p logs

export HF_HOME=/projects/bhkj/$USER/hf_cache
export TRANSFORMERS_CACHE=/projects/bhkj/$USER/hf_cache
export WANDB_DISABLED=true

python3 -m venv /projects/bhkj/$USER/peft_env || true
source /projects/bhkj/$USER/peft_env/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch datasets trl transformers accelerate peft matplotlib

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python -u src/llm_vision_planner/fine_tuning/scripts/train_peft.py \
  --dataset src/llm_vision_planner/fine_tuning/datasets/rrt_expert_dataset.csv \
  --epochs 0.05 \
  --batch-size 1 \
  --grad-accum 2
