#!/bin/bash
#SBATCH --account=bhkj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --job-name=rrt-lora
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd /projects/bhkj/$USER/starling_testing_ws
mkdir -p logs

module purge
module load cray-python

export HF_HOME=/projects/bhkj/$USER/hf_cache
export TRANSFORMERS_CACHE=/projects/bhkj/$USER/hf_cache
export WANDB_DISABLED=true

ENV_DIR=/projects/bhkj/$USER/unsloth_env
if [ -x "$ENV_DIR/bin/python" ]; then
  ENV_PY_MAJOR=$("$ENV_DIR/bin/python" -c 'import sys; print(sys.version_info.major)')
  ENV_PY_MINOR=$("$ENV_DIR/bin/python" -c 'import sys; print(sys.version_info.minor)')
  if [ "$ENV_PY_MAJOR" -lt 3 ] || { [ "$ENV_PY_MAJOR" -eq 3 ] && [ "$ENV_PY_MINOR" -lt 11 ]; }; then
    echo "Removing old Python $ENV_PY_MAJOR.$ENV_PY_MINOR environment at $ENV_DIR"
    rm -rf "$ENV_DIR"
  fi
fi

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install unsloth datasets trl transformers accelerate peft bitsandbytes matplotlib

python - <<'PY'
import sys
import torch
print("python:", sys.version)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python -u src/llm_vision_planner/fine_tuning/scripts/train.py \
  --dataset src/llm_vision_planner/fine_tuning/datasets/rrt_expert_dataset.csv
