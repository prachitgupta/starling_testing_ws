# HRRT preference-conditioned Llama fine-tuning

Run stages 1–5 on the local workstation. Run stages 6–10 on NCSA DeltaAI.
The teacher chooses only an existing HRRT route ID; the stored target waypoints
are copied from that route. Validation and test scenes require human labels.

## 1. Enter the package

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 -m venv .venv-hrrt-data
source .venv-hrrt-data/bin/activate
python -m pip install -U pip openai pydantic numpy scipy matplotlib
export OPENAI_API_KEY=YOUR_KEY
```

## 2. Smoke-test data generation

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --teacher mock \
  --scenes 10 \
  --preferences-per-scene 6 \
  --hrrt-iterations 300 \
  --output /tmp/hrrt_teacher_smoke.jsonl
wc -l /tmp/hrrt_teacher_smoke.jsonl
```

## 3. Generate teacher labels

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --teacher openai \
  --teacher-model gpt-5.4 \
  --scenes 2000 \
  --preferences-per-scene 6 \
  --hrrt-iterations 500 \
  --output fine_tuning/datasets/hrrt_teacher_raw.jsonl \
  --resume
```

If the command stops, run the same command again. `--resume` removes any partial
scene and restarts that scene without duplicating completed labels.

```bash
python fine_tuning/scripts/build_hrrt_human_audit.py \
  --input fine_tuning/datasets/hrrt_teacher_raw.jsonl \
  --output fine_tuning/datasets/hrrt_human_audit.jsonl \
  --train-audit-ratio 0.10 \
  --validation-scene-ratio 0.10 \
  --test-scene-ratio 0.10
```

This assigns every validation and test label, plus a stratified ten-percent
training sample, to human review. Splits are made by scene, not by row.

## 4. Review the assigned labels

```bash
python fine_tuning/scripts/review_hrrt_labels.py \
  --raw fine_tuning/datasets/hrrt_teacher_raw.jsonl \
  --audit fine_tuning/datasets/hrrt_human_audit.jsonl \
  --reviews fine_tuning/datasets/hrrt_human_reviews.jsonl \
  --host 127.0.0.1 \
  --port 8090
```

Open [http://127.0.0.1:8090](http://127.0.0.1:8090). For every assignment:

1. Read the operator preference.
2. Compare the colored route plot and route table.
3. Select **Accept teacher**, **Use selected route**, or **Mark ambiguous**.
4. Continue until the page says **Review complete**.

The interface writes after every decision, so it can be stopped and restarted.
Human corrections replace teacher labels in training. Ambiguous rows are removed.
The human-reviewed validation split selects the checkpoint; the locked
human-reviewed test split is used only after training.

## 5. Finalize scene-isolated CSV files

```bash
python fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw fine_tuning/datasets/hrrt_teacher_raw.jsonl \
  --audit fine_tuning/datasets/hrrt_human_audit.jsonl \
  --reviews fine_tuning/datasets/hrrt_human_reviews.jsonl \
  --output-dir fine_tuning/datasets \
  --clearance-m 0.40
```

```bash
ls -lh \
  fine_tuning/datasets/hrrt_sft_train.csv \
  fine_tuning/datasets/hrrt_sft_validation.csv \
  fine_tuning/datasets/hrrt_sft_test.csv
```

## 6. Prepare the NCSA workspace

```bash
ssh pgupta12@dtai-login.delta.ncsa.illinois.edu
cd /projects/bhkj/$USER
git clone --single-branch --branch starling_multiple_trajectory_idea \
  https://github.com/prachitgupta/starling_testing_ws.git starling_multiple_trajectory_idea
cd starling_multiple_trajectory_idea
mkdir -p logs src/llm_vision_planner/fine_tuning/datasets
```

For an existing clone:

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
git pull --ff-only
mkdir -p logs
```

## 7. Upload the finalized splits

Run these commands on the local workstation:

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
scp fine_tuning/datasets/hrrt_sft_{train,validation,test}.csv \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/
```

## 8. Authenticate once on DeltaAI

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
module purge
module load cray-python
python -m venv /projects/bhkj/$USER/hf_auth_env
source /projects/bhkj/$USER/hf_auth_env/bin/activate
python -m pip install -U huggingface_hub
export HF_HOME=/projects/bhkj/$USER/hf_cache
hf auth login
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("meta-llama/Meta-Llama-3.1-8B-Instruct").modelId)
PY
```

## 9. Submit and monitor QLoRA training

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
sbatch src/llm_vision_planner/fine_tuning/scripts/train_hrrt_peft_lora.sbatch
```

```bash
squeue -u $USER
tail -f logs/hrrt-peft-JOB_ID.out
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```

Replace `JOB_ID` with the number returned by `sbatch`. A completed job creates:

```text
src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora/
src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora.tar.gz
```

## 10. Evaluate the locked test split

Request an interactive GPU session:

```bash
srun --account=bhkj-dtai-gh --partition=ghx4 --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=16 --gpus-per-node=1 --mem=128g --time=01:00:00 --pty bash
```

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
source /projects/bhkj/$USER/hrrt_peft_env/bin/activate
python src/llm_vision_planner/fine_tuning/scripts/evaluate_hrrt_adapter.py \
  --test src/llm_vision_planner/fine_tuning/datasets/hrrt_sft_test.csv \
  --adapter src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora \
  --output src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora/test_metrics.json
```

Inspect structured-output validity, geometric feasibility, selected-route
agreement, endpoint error, and mean path error:

```bash
cat src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora/test_metrics.json
```

## 11. Download the adapter

Run on the GPU machine that will serve the model:

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs
scp pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora.tar.gz .
tar -xzf llama31_8b_hrrt_lora.tar.gz
test -s llama31_8b_hrrt_lora/adapter_config.json
test -s llama31_8b_hrrt_lora/adapter_model.safetensors
```

The runtime planner prompt includes the approved natural-language route
preference. It does not expose the expert HRRT route or its waypoints to Llama.
