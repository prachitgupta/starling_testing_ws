# NCSA DeltaAI Commands

Replace `YOUR_ACCOUNT` with the account shown by `accounts`.

## Login And Account

```bash
ssh pgupta12@dtai-login.delta.ncsa.illinois.edu
```
Log in with NCSA Kerberos password and Duo.

```bash
accounts
```
Show allocation accounts and remaining GPU hours.

```bash
quota
```
Show storage quota and usage.

```bash
find /work -maxdepth 3 -type d -name "$USER" 2>/dev/null
```
Find existing project storage directories for your username.

```bash
ls /work/hdd
ls /work/nvme
```
List available HDD and NVMe project storage roots.

## Workspace Setup On DeltaAI

```bash
mkdir -p /work/hdd/YOUR_ACCOUNT/$USER
cd /work/hdd/YOUR_ACCOUNT/$USER
```
Create and enter project HDD storage.

```bash
mkdir -p /work/nvme/YOUR_ACCOUNT/$USER
cd /work/nvme/YOUR_ACCOUNT/$USER
```
Create and enter project NVMe storage if your account has it.

```bash
git clone https://github.com/prachitgupta/starling_testing_ws.git
cd starling_testing_ws
```
Clone the workspace.

```bash
cd /work/hdd/YOUR_ACCOUNT/$USER/starling_testing_ws
git pull
```
Update an existing clone.

## Dataset Transfer

```bash
cd ~/Desktop/starling_testing_ws/src
python3 llm_vision_planner/fine_tuning/scripts/dataset_generator.py --samples 20000 --random-goal --seed 7
```
Generate the RRT expert dataset locally.

```bash
scp ~/Desktop/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/rrt_expert_dataset.csv \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:/work/hdd/YOUR_ACCOUNT/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/
```
Upload the dataset to DeltaAI.

```bash
scp pgupta12@dtai-login.delta.ncsa.illinois.edu:/work/hdd/YOUR_ACCOUNT/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_rrt_lora.tar.gz .
```
Download trained adapter archive.

## Create Training Job

```bash
cd /work/hdd/YOUR_ACCOUNT/$USER/starling_testing_ws
mkdir -p logs
cat > train_rrt_lora.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
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

cd /work/hdd/YOUR_ACCOUNT/$USER/starling_testing_ws

export HF_HOME=/work/hdd/YOUR_ACCOUNT/$USER/hf_cache
export TRANSFORMERS_CACHE=/work/hdd/YOUR_ACCOUNT/$USER/hf_cache
export WANDB_DISABLED=true

python3 -m venv /work/hdd/YOUR_ACCOUNT/$USER/unsloth_env || true
source /work/hdd/YOUR_ACCOUNT/$USER/unsloth_env/bin/activate

python -m pip install --upgrade pip
python -m pip install unsloth datasets trl transformers accelerate peft bitsandbytes

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python -u src/llm_vision_planner/fine_tuning/scripts/train.py \
  --dataset src/llm_vision_planner/fine_tuning/datasets/rrt_expert_dataset.csv \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 4

tar -czf src/llm_vision_planner/fine_tuning/outputs/llama31_8b_rrt_lora.tar.gz \
  -C src/llm_vision_planner/fine_tuning/outputs llama31_8b_rrt_lora
EOF
```
Create a Slurm batch script for LoRA training.

```bash
sed -i "s/YOUR_ACCOUNT/ACTUAL_ACCOUNT/g" train_rrt_lora.sbatch
```
Replace the placeholder account after creating the script.

## Submit And Monitor

```bash
sbatch train_rrt_lora.sbatch
```
Submit the training job.

```bash
squeue -u $USER
```
Show queued/running jobs.

```bash
tail -f logs/rrt-lora-*.out
```
Watch training logs.

```bash
sacct -u $USER --starttime today
```
Show jobs from today.

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```
Show details for one job.

```bash
scancel JOB_ID
```
Cancel a job.

## Interactive GPU Test

```bash
srun --account=YOUR_ACCOUNT --partition=ghx4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=1 --mem=128g --time=01:00:00 --pty bash
```
Start an interactive GPU shell.

```bash
nvidia-smi
```
Check GPU visibility inside a job.

```bash
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```
Check CUDA from Python.
