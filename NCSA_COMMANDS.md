# NCSA DeltaAI fine-tuning: RRT and Semantic Theta*

This is the training guide for `finding_expert`. Complete the local setup in
[README.md](README.md#1-clone-build-and-source-locally) first. It connects to the
separate [calibration/plot guide](src/llm_vision_planner/fine_tuning/README.md).
Replace `pgupta12`, `bhkj`, and `bhkj-dtai-gh` with your NCSA login/project/account
where necessary, including the existing Slurm scripts. These are the original
lab allocation values, not accounts supplied by the repository.

Original storage paths:

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
git clone --depth 1 --single-branch --branch finding_expert https://github.com/prachitgupta/starling_testing_ws.git
cd starling_testing_ws
```
Clone the workspace.

```bash
cd /projects/bhkj/$USER/starling_testing_ws
git branch --show-current
git pull --ff-only
```
Update an existing `finding_expert` clone only. Keep any `main` clone separate.
The job scripts expect the project-storage path shown below, so clone under
`/projects/bhkj/$USER`, not under HDD/NVMe unless you edit their workspace paths.

## RRT And Semantic Theta* LoRA Workflows

Use the same numbered stages for either expert. Stages 1 and 2 run locally;
stages 3 through 6 run after SSH or in Open OnDemand on DeltaAI. Stage 7
downloads the adapter to the local/GPU machine that will serve it.

After training and serving the adapter, generate fresh expert/LLM calibration
pairs, shared-clock QP trajectories, scores, and dConformal plots using the
separate RRT and Semantic Theta* workflows in the
[`fine-tuning README`](src/llm_vision_planner/fine_tuning/README.md#calibration-prerequisites).

### 1. Generate the expert dataset locally

#### RRT

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
cd src
/usr/bin/python3 llm_vision_planner/fine_tuning/scripts/conformal_rrt_dataset.py \
  --rrt-training --samples 20000 --random-goal --seed 7
```

Creates `fine_tuning/datasets/rrt_expert_dataset.csv`.

#### Semantic Theta*

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
cd src
/usr/bin/python3 llm_vision_planner/fine_tuning/scripts/conformal_semantic_theta_dataset.py \
  --semantic-theta-training --samples 20000 --random-goal --seed 17
```

Creates `fine_tuning/datasets/semantic_theta_expert_dataset.csv`. The requested
`conforml_semantic_theta_dataset.py` spelling remains a compatible entry point.

### 2. Upload the dataset from your local machine

#### RRT

```bash
scp ~/Desktop/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/rrt_expert_dataset.csv \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/
```

#### Semantic Theta*

```bash
scp ~/Desktop/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/semantic_theta_expert_dataset.csv \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/
```

### 3. Authenticate with Hugging Face on DeltaAI

This one-time setup is shared by both experts.

```bash
cd /projects/bhkj/$USER/starling_testing_ws
module purge
module load cray-python
python -m venv /projects/bhkj/$USER/hf_auth_env
source /projects/bhkj/$USER/hf_auth_env/bin/activate
python -m pip install -U huggingface_hub
export HF_HOME=/projects/bhkj/$USER/hf_cache
hf auth login
```

Save a Hugging Face read token with access to the gated Llama model, then verify
that access:

```bash
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("meta-llama/Meta-Llama-3.1-8B-Instruct").modelId)
PY
```

### 4. Submit the PEFT training job on DeltaAI

PEFT is the recommended path. Prepare the shared log directory first:

```bash
cd /projects/bhkj/$USER/starling_testing_ws
mkdir -p logs
```

#### RRT

```bash
sbatch src/llm_vision_planner/fine_tuning/scripts/train_peft_lora.sbatch
```

#### Semantic Theta*

```bash
sbatch src/llm_vision_planner/fine_tuning/scripts/train_semantic_theta_peft_lora.sbatch
```

Optional Unsloth jobs use the same datasets and output names:

#### RRT with Unsloth

```bash
sbatch src/llm_vision_planner/fine_tuning/scripts/train_rrt_lora.sbatch
```

#### Semantic Theta* with Unsloth

```bash
sbatch src/llm_vision_planner/fine_tuning/scripts/train_semantic_theta_lora.sbatch
```

### 5. Monitor the job on DeltaAI

For either expert:

```bash
squeue -u $USER
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```

Use the matching log command:

#### RRT PEFT

```bash
tail -f logs/rrt-peft-JOB_ID.out
```

#### Semantic Theta* PEFT

```bash
tail -f logs/semantic-theta-peft-JOB_ID.out
```

Replace `JOB_ID` with the number printed by `sbatch`. Wait for `COMPLETED` before
packaging or downloading the adapter. To cancel a job, run `scancel JOB_ID`.

### 6. Verify and package the completed adapter on DeltaAI

#### RRT

```bash
cd /projects/bhkj/$USER/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs
test -s llama31_8b_rrt_lora/adapter_config.json
test -s llama31_8b_rrt_lora/adapter_model.safetensors
tar -czf llama31_8b_rrt_lora.tar.gz llama31_8b_rrt_lora
ls -lh llama31_8b_rrt_lora.tar.gz
```

#### Semantic Theta*

```bash
cd /projects/bhkj/$USER/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs
test -s llama31_8b_semantic_theta_lora/adapter_config.json
test -s llama31_8b_semantic_theta_lora/adapter_model.safetensors
tar -czf llama31_8b_semantic_theta_lora.tar.gz llama31_8b_semantic_theta_lora
ls -lh llama31_8b_semantic_theta_lora.tar.gz
```

The Semantic Theta* Slurm scripts already create their archive automatically;
the explicit command above also documents how to recreate it if needed.

### 7. Download the archive from your local machine

Exit the NCSA shell first, then run the matching commands on the GPU machine
where vLLM will serve the model (or download on the laptop and transfer the
archive there). Both expected output directories already contain placeholders.
Extract the completed archive into the matching directory before serving.

#### RRT

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs
scp pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_rrt_lora.tar.gz .
tar -xzf llama31_8b_rrt_lora.tar.gz
test -s llama31_8b_rrt_lora/adapter_config.json
test -s llama31_8b_rrt_lora/adapter_model.safetensors
```

#### Semantic Theta*

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs
scp pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_semantic_theta_lora.tar.gz .
tar -xzf llama31_8b_semantic_theta_lora.tar.gz
test -s llama31_8b_semantic_theta_lora/adapter_config.json
test -s llama31_8b_semantic_theta_lora/adapter_model.safetensors
```

After extraction, use the matching serving command in
[README.md](README.md#3-fine-tune-download-and-serve-the-selected-adapter), then
generate fresh calibration data with the
[calibration README](src/llm_vision_planner/fine_tuning/README.md#calibration-prerequisites).

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
