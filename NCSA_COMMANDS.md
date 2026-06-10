# NCSA DeltaAI Commands

```bash
ssh pgupta12@dtai-login.delta.ncsa.illinois.edu
```
Log in to DeltaAI with NCSA Kerberos password and Duo.

```bash
accounts
```
Show available allocation accounts and remaining GPU hours.

```bash
quota
```
Show storage quota and usage.

```bash
pwd
ls
```
Check current directory and files.

```bash
cd /work/hdd/YOUR_ACCOUNT/$USER
```
Move to project HDD storage.

```bash
cd /work/nvme/YOUR_ACCOUNT/$USER
```
Move to faster NVMe storage if available.

```bash
git clone https://github.com/prachitgupta/starling_testing_ws.git
```
Clone the workspace on DeltaAI.

```bash
git pull
```
Update an existing clone.

```bash
scp LOCAL_FILE pgupta12@dtai-login.delta.ncsa.illinois.edu:/work/hdd/YOUR_ACCOUNT/pgupta12/
```
Copy a file from local machine to DeltaAI.

```bash
scp pgupta12@dtai-login.delta.ncsa.illinois.edu:/work/hdd/YOUR_ACCOUNT/pgupta12/REMOTE_FILE .
```
Copy a file from DeltaAI to local machine.

```bash
squeue -u $USER
```
Show your queued/running jobs.

```bash
sacct -u $USER --starttime today
```
Show your jobs from today.

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```
Show details for one job.

```bash
sbatch train_rrt_lora.sbatch
```
Submit a batch training job.

```bash
scancel JOB_ID
```
Cancel a job.

```bash
tail -f logs/rrt-lora-*.out
```
Watch training logs.

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

```bash
mkdir -p logs
```
Create a log directory for Slurm output.

```bash
python3 src/llm_vision_planner/fine_tuning/scripts/train.py --dataset src/llm_vision_planner/fine_tuning/datasets/rrt_expert_dataset.csv --epochs 1 --batch-size 2 --grad-accum 4
```
Run LoRA training inside an allocated GPU job.
