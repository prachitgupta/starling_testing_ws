# NCSA DeltaAI Commands

Your discovered storage paths:

```text
/projects/bhkj/pgupta12
/work/hdd/bhkj/pgupta12
/work/nvme/bhkj/pgupta12
```

This guide uses `/projects/bhkj/$USER` as the main repo and output location.

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
Find existing work storage directories for your username.

```bash
find /projects -maxdepth 3 -type d -name "$USER" 2>/dev/null
```
Find existing project storage directories for your username.

```bash
ls /projects/bhkj
ls /work/hdd/bhkj
ls /work/nvme/bhkj
```
Check available project, HDD, and NVMe storage roots.

## Workspace Setup On DeltaAI

```bash
cd /projects/bhkj/$USER
```
Enter project storage.

```bash
cd /work/hdd/bhkj/$USER
```
Enter HDD work storage.

```bash
cd /work/nvme/bhkj/$USER
```
Enter NVMe work storage.

```bash
git clone https://github.com/prachitgupta/starling_testing_ws.git
cd starling_testing_ws
```
Clone the workspace.

```bash
cd /projects/bhkj/$USER/starling_testing_ws
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
  pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/
```
Upload the dataset to DeltaAI.

```bash
scp pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_rrt_lora.tar.gz .
```
Download trained adapter archive.

## Create Training Job

```bash
cd /projects/bhkj/$USER/starling_testing_ws
mkdir -p logs
cat > train_rrt_lora.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=bhkj
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

export HF_HOME=/projects/bhkj/$USER/hf_cache
export TRANSFORMERS_CACHE=/projects/bhkj/$USER/hf_cache
export WANDB_DISABLED=true

python3 -m venv /projects/bhkj/$USER/unsloth_env || true
source /projects/bhkj/$USER/unsloth_env/bin/activate

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
srun --account=bhkj --partition=ghx4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=1 --mem=128g --time=01:00:00 --pty bash
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
