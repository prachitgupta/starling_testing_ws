# NCSA DeltaAI Commands

Your discovered storage paths:

```text
/projects/bhkj/pgupta12
/work/hdd/bhkj/pgupta12
/work/nvme/bhkj/pgupta12
```

This guide uses `/projects/bhkj/$USER` for files and `bhkj-dtai-gh` for Slurm jobs.

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

## Hugging Face Auth

```bash
cd /projects/bhkj/$USER/starling_testing_ws
module purge
module load cray-python
source /projects/bhkj/$USER/peft_env/bin/activate
python -m pip install -U huggingface_hub
export HF_HOME=/projects/bhkj/$USER/hf_cache
hf auth login
```
Save a Hugging Face read token for gated Llama model access.

```bash
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("meta-llama/Meta-Llama-3.1-8B-Instruct").modelId)
PY
```
Check that the saved token can access Llama 3.1 8B Instruct.

## PEFT Training Job

```bash
cd /projects/bhkj/$USER/starling_testing_ws
mkdir -p logs
```
Prepare the repo for Slurm logs.

```bash
sed -n '1,200p' src/llm_vision_planner/fine_tuning/scripts/train_peft_lora.sbatch
```
Inspect the PEFT Slurm script that avoids the Unsloth dependency.

```bash
sbatch src/llm_vision_planner/fine_tuning/scripts/train_peft_lora.sbatch
```
Submit the sanity-check PEFT job.

```bash
cp src/llm_vision_planner/fine_tuning/scripts/train_peft_lora.sbatch train_peft_full.sbatch
sed -i 's/--epochs 0.05 \\/--epochs 3 \\/' train_peft_full.sbatch
sed -i 's/--batch-size 1 \\/--batch-size 8 \\/' train_peft_full.sbatch
sed -i 's/--grad-accum 2/--grad-accum 4/' train_peft_full.sbatch
sbatch train_peft_full.sbatch
```
Submit a full training job with the current 20k-sample defaults.

## Unsloth Training Job

```bash
sbatch src/llm_vision_planner/fine_tuning/scripts/train_rrt_lora.sbatch
```
Submit the Unsloth LoRA job.

## Submit And Monitor

```bash
sbatch src/llm_vision_planner/fine_tuning/scripts/train_peft_lora.sbatch
```
Submit the PEFT sanity-check job.

```bash
squeue -u $USER
```
Show queued/running jobs.

```bash
tail -f logs/rrt-peft-*.out
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
srun --account=bhkj-dtai-gh --partition=ghx4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=1 --mem=128g --time=01:00:00 --pty bash
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
