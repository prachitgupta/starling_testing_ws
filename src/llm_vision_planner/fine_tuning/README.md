# HRRT preference-conditioned Llama fine-tuning

Distill HRRT-star route preferences into a Llama waypoint planner. The deployed
model receives scene geometry, human preference, and safety constraints only.

Two dataset modes are available:

- `dss`: full-plan plus reasoning-only auxiliary supervision.
- `dss_scott`: DSS plus paired safe positive/counterfactual rationale tasks.

Both modes use the existing LoRA/SFT loss; `dss_scott` does not add a
token-level contrastive decoder or custom loss.

Both modes keep the deployed response unchanged:

```json
{"reasoning":"Preference: ... Geometry: ... Decision: ... Safety: ...","waypoints":[{"x":0,"y":0,"z":-0.5}]}
```

Run stages 1–5 locally; run stages 6–10 on NCSA DeltaAI.

## 1. Enter the correct worktree and install data-generation dependencies

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 -m venv .venv-hrrt-data
source .venv-hrrt-data/bin/activate
python -m pip install --upgrade pip openai pydantic numpy scipy matplotlib
export OPENAI_API_KEY=YOUR_KEY
TEACHER_MODEL=gpt-6-luna
```

Use `gpt-6-luna` for volume; use `gpt-6.1-sol` for a stronger teacher.

Choose one mode and keep each mode in its own directory.

```bash
DISTILLATION_MODE=dss
DATASET_ROOT="fine_tuning/datasets/$DISTILLATION_MODE"
mkdir -p "$DATASET_ROOT"
```

For the second experiment, start a new shell or change the first line to:

```bash
DISTILLATION_MODE=dss_scott
DATASET_ROOT="fine_tuning/datasets/$DISTILLATION_MODE"
mkdir -p "$DATASET_ROOT"
```

## 2. Smoke-test the selected mode without API calls

Smoke test without API calls; `--teacher mock` is deterministic.

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --output "$DATASET_ROOT/hrrt_teacher_smoke.jsonl" \
  --scenes 10 \
  --preferences-per-scene 6 \
  --seed 1701 \
  --teacher mock \
  --teacher-model "$TEACHER_MODEL" \
  --distillation-mode "$DISTILLATION_MODE" \
  --min-obstacles 2 \
  --max-obstacles 4 \
  --min-routes 2 \
  --max-candidates 8 \
  --hrrt-iterations 300 \
  --max-total-nodes 5000 \
  --max-nodes-per-key 180 \
  --scene-attempts 8 \
  --nominal-speed-mps 0.5 \
  --clearance-m 0.40
wc -l "$DATASET_ROOT/hrrt_teacher_smoke.jsonl"
```

Obstacle counts may be `0`–`4`; use at least `1` for non-trivial routes.

Synthetic scenes use `person`, `chair`, and `stop_sign`; four-obstacle scenes
repeat one class with a new object ID.

Use `--resume` only for the same file and mode; otherwise the output is replaced.

## 3. Generate teacher labels

The teacher sees the full scene and route candidates. DSS uses one API call per
preference; DSS-SCOTT uses route selection plus rationale conditioning. For
2,000 scenes × 6 preferences: 12,000 DSS or 24,000 DSS-SCOTT calls.

Llama-facing reasoning excludes route IDs/cards/candidates; invalid output uses
a deterministic scene-grounded fallback.

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --output "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --scenes 2000 \
  --preferences-per-scene 6 \
  --seed 1701 \
  --teacher openai \
  --teacher-model "$TEACHER_MODEL" \
  --reasoning-effort medium \
  --distillation-mode "$DISTILLATION_MODE" \
  --min-obstacles 2 \
  --max-obstacles 4 \
  --min-routes 2 \
  --max-candidates 8 \
  --hrrt-iterations 500 \
  --max-total-nodes 5000 \
  --max-nodes-per-key 180 \
  --scene-attempts 8 \
  --nominal-speed-mps 0.5 \
  --clearance-m 0.40 \
  --resume
```

If interrupted, rerun the same command. Resume removes only an incomplete scene;
one-obstacle scenes may contain five valid preferences.

Create scene-isolated review assignments:

```bash
python fine_tuning/scripts/build_hrrt_human_audit.py \
  --input "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --output "$DATASET_ROOT/hrrt_human_audit.jsonl" \
  --train-audit-ratio 0.10 \
  --validation-scene-ratio 0.10 \
  --test-scene-ratio 0.10 \
  --seed 42
```

Validation/test labels and a stratified training sample are reviewed. Splits are
scene-isolated.

## 4. Review teacher labels and reasoning

```bash
python fine_tuning/scripts/review_hrrt_labels.py \
  --raw "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$DATASET_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$DATASET_ROOT/hrrt_human_reviews.jsonl" \
  --host 127.0.0.1 \
  --port 8090
```

Open [http://127.0.0.1:8090](http://127.0.0.1:8090). For each assignment:

1. Read the human preference, audit reason, and Llama-facing reason.
2. Compare the colored route plot and route metrics.
3. Accept the teacher, choose a corrected route, or mark the sample ambiguous.
4. Give corrected routes a reason using `Preference`, `Geometry`, `Decision`, and `Safety`.

The UI saves each decision; valid human reasons override teacher reasons.
Ambiguous rows are removed.

### Share the review UI with remote reviewers

Use `0.0.0.0` plus `cloudflared` for temporary remote HTTPS access.

Terminal 1 (bind to all interfaces so the tunnel can reach the service):

```bash
python fine_tuning/scripts/review_hrrt_labels.py \
  --raw "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$DATASET_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$DATASET_ROOT/hrrt_human_reviews.jsonl" \
  --host 0.0.0.0 \
  --port 8090
```

Terminal 2:

```bash
./cloudflared tunnel --url http://127.0.0.1:8090 2>&1 | grep -oE 'https://[a-zA-Z0-9-]+.trycloudflare.com'
```

Send the printed `https://...trycloudflare.com` URL to reviewers. Keep both
processes running; stop them with `Ctrl-C`. Treat the unauthenticated URL as
secret; use a named tunnel plus Cloudflare Access for a stable deployment.

## 5. Finalize train, validation, and test CSVs

For DSS:

```bash
python fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$DATASET_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$DATASET_ROOT/hrrt_human_reviews.jsonl" \
  --output-dir "$DATASET_ROOT" \
  --clearance-m 0.40 \
  --distillation-mode "$DISTILLATION_MODE" \
  --reasoning-aux-ratio 0.25 \
  --counterfactual-aux-ratio 0.00 \
  --auxiliary-seed 42
```

For DSS-SCOTT, enable paired counterfactual auxiliary tasks:

```bash
python fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$DATASET_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$DATASET_ROOT/hrrt_human_reviews.jsonl" \
  --output-dir "$DATASET_ROOT" \
  --clearance-m 0.40 \
  --distillation-mode "$DISTILLATION_MODE" \
  --reasoning-aux-ratio 0.25 \
  --counterfactual-aux-ratio 0.25 \
  --auxiliary-seed 42
```

Auxiliary ratios are `0`–`1`; auxiliary rows are training-only. Validation/test
keep the one-call deployment task.

```bash
ls -lh \
  "$DATASET_ROOT/hrrt_sft_train.csv" \
  "$DATASET_ROOT/hrrt_sft_validation.csv" \
  "$DATASET_ROOT/hrrt_sft_test.csv"
```

## 6. Prepare the NCSA workspace

```bash
ssh pgupta12@dtai-login.delta.ncsa.illinois.edu
cd /projects/bhkj/$USER
git clone --single-branch --branch starling_multiple_trajectory_idea \
  https://github.com/prachitgupta/starling_testing_ws.git starling_multiple_trajectory_idea
cd starling_multiple_trajectory_idea
mkdir -p logs src/llm_vision_planner/fine_tuning/datasets/dss
mkdir -p src/llm_vision_planner/fine_tuning/datasets/dss_scott
```

Existing clone:

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
git pull --ff-only
mkdir -p logs
```

## 7. Upload one finalized mode

Run locally with the stage-1 `DISTILLATION_MODE` and `DATASET_ROOT`:

```bash
scp "$DATASET_ROOT"/hrrt_sft_{train,validation,test}.csv \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/$DISTILLATION_MODE/
```

## 8. Authenticate once on DeltaAI

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
module purge
module load cray-python
python -m venv /projects/bhkj/$USER/hf_auth_env
source /projects/bhkj/$USER/hf_auth_env/bin/activate
python -m pip install --upgrade huggingface_hub
export HF_HOME=/projects/bhkj/$USER/hf_cache
hf auth login
python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("meta-llama/Meta-Llama-3.1-8B-Instruct").modelId)
PY
```

## 9. Train with QLoRA

Submit DSS:

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
sbatch --export=ALL,DISTILLATION_MODE=dss \
  src/llm_vision_planner/fine_tuning/scripts/train_hrrt_peft_lora.sbatch
```

Submit DSS-SCOTT:

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
sbatch --export=ALL,DISTILLATION_MODE=dss_scott \
  src/llm_vision_planner/fine_tuning/scripts/train_hrrt_peft_lora.sbatch
```

Optional exported overrides: `PROJECT_ROOT`, `HRRT_ENV_DIR`, `HF_HOME`,
`DISTILLATION_MODE`, `DATASET_ROOT`, and `OUTPUT`. Default output:
`fine_tuning/outputs/llama31_8b_hrrt_lora_<mode>`.

Monitor the job:

```bash
squeue -u $USER
tail -f logs/hrrt-peft-JOB_ID.out
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```

Direct interactive run with explicit trainer arguments:

```bash
DISTILLATION_MODE=dss
python src/llm_vision_planner/fine_tuning/scripts/train_hrrt_peft.py \
  --train "src/llm_vision_planner/fine_tuning/datasets/$DISTILLATION_MODE/hrrt_sft_train.csv" \
  --validation "src/llm_vision_planner/fine_tuning/datasets/$DISTILLATION_MODE/hrrt_sft_validation.csv" \
  --output-dir "src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora_$DISTILLATION_MODE" \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --distillation-mode "$DISTILLATION_MODE" \
  --max-seq-length 2048 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 8 \
  --learning-rate 5e-5 \
  --lora-r 128 \
  --lora-alpha 256 \
  --lora-dropout 0.05 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100 \
  --seed 42 \
  --qlora
```

Omit `--qlora` only when intentionally not using 4-bit NF4.

## 10. Evaluate the locked test split

Request a GPU session:

```bash
srun --account=bhkj-dtai-gh --partition=ghx4 --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=16 --gpus-per-node=1 --mem=128g --time=01:00:00 --pty bash
```

Evaluate the locked test split:

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
source /projects/bhkj/$USER/hrrt_peft_env/bin/activate
DISTILLATION_MODE=dss
python src/llm_vision_planner/fine_tuning/scripts/evaluate_hrrt_adapter.py \
  --test "src/llm_vision_planner/fine_tuning/datasets/$DISTILLATION_MODE/hrrt_sft_test.csv" \
  --adapter "src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora_$DISTILLATION_MODE" \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --distillation-mode "$DISTILLATION_MODE" \
  --output "src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora_$DISTILLATION_MODE/test_metrics.json" \
  --clearance-m 0.40 \
  --max-new-tokens 600
```

To score existing predictions, add:

```text
--predictions /absolute/path/to/predictions.jsonl
```

The report includes schema validity, feasibility, route agreement, endpoint error,
and mean path error.

## 11. Transfer the NCSA adapter to the GPU server and serve it

On NCSA, package the trained mode (`dss` or `dss_scott`):

```bash
ssh pgupta12@dtai-login.delta.ncsa.illinois.edu
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
DISTILLATION_MODE=dss
ADAPTER_NAME="llama31_8b_hrrt_lora_$DISTILLATION_MODE"
ADAPTER_DIR="$PWD/src/llm_vision_planner/fine_tuning/outputs/$ADAPTER_NAME"
ADAPTER_CHECKPOINT="$(dirname "$(find "$ADAPTER_DIR" -name adapter_model.safetensors -print -quit)")"

test -s "$ADAPTER_CHECKPOINT/adapter_config.json"
test -s "$ADAPTER_CHECKPOINT/adapter_model.safetensors"
tar -C "$(dirname "$ADAPTER_DIR")" -czf "$ADAPTER_DIR.tar.gz" "$ADAPTER_NAME"
exit
```

From the local workstation, SSH into the GPU server first:

```bash
GPU_IP=172.22.224.93
ssh prachit2@"$GPU_IP"
```

At the GPU prompt, pull from NCSA with resumable `rsync -P`; complete NCSA Duo
there:

```bash
DISTILLATION_MODE=dss
ADAPTER_NAME="llama31_8b_hrrt_lora_$DISTILLATION_MODE"
NCSA_ARCHIVE="/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/$ADAPTER_NAME.tar.gz"
GPU_OUTPUTS="/home/prachit2/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs"

mkdir -p "$GPU_OUTPUTS"
rsync -avP \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:"${NCSA_ARCHIVE%.tar.gz}/" \
  "$GPU_OUTPUTS/$ADAPTER_NAME/"
find "$GPU_OUTPUTS/$ADAPTER_NAME" -name adapter_config.json -print
find "$GPU_OUTPUTS/$ADAPTER_NAME" -name adapter_model.safetensors -print
```

Fallback when `rsync` is unavailable:

```bash
scp -r \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:"${NCSA_ARCHIVE%.tar.gz}/" \
  "$GPU_OUTPUTS/$ADAPTER_NAME/"
```

For an existing archive, validate before extracting:

```bash
cd "$GPU_OUTPUTS"
gzip -t "$ADAPTER_NAME.tar.gz"
tar -tzf "$ADAPTER_NAME.tar.gz" >/dev/null
tar -xzf "$ADAPTER_NAME.tar.gz"
```

Use the checkpoint directory containing both adapter files:

```bash
ADAPTER="$(dirname "$(find "$GPU_OUTPUTS/$ADAPTER_NAME" -name adapter_model.safetensors -print -quit)")"
test -s "$ADAPTER/adapter_config.json"
test -s "$ADAPTER/adapter_model.safetensors"

MODEL_ALIAS="hrrt_planner_$DISTILLATION_MODE"

CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --enable-lora \
  --max-lora-rank 128 \
  --lora-modules "$MODEL_ALIAS=$ADAPTER" \
  --served-model-name "$MODEL_ALIAS" \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --port 8000
```

This is the HRRT equivalent of `rrt_planner`. Install vLLM in the GPU Python
environment if needed. The GPU Hugging Face account needs gated Llama access.

Verify the adapter alias from a second GPU terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

On the ROS machine, set `llm_planner.ros__parameters.vllm_base_url` to the GPU
URL and use the same model alias:

```bash
GPU_IP=172.22.224.93
DISTILLATION_MODE=dss
MODEL_ALIAS="hrrt_planner_$DISTILLATION_MODE"
VLLM_BASE_URL="http://$GPU_IP:8000/v1"
curl -fsS "$VLLM_BASE_URL/models" | python3 -m json.tool

ros2 launch llm_vision_planner full_plot.launch.py \
  llm_provider:=llama \
  llama_model_name:="$MODEL_ALIAS" \
  interaction_mode:=interactive \
  web_ui_host:=0.0.0.0 \
  web_ui_port:=8081
```

To share the interactive mission UI publicly, run this on the same ROS machine
in a second terminal:

```bash
./cloudflared tunnel --url http://127.0.0.1:8081 2>&1 | grep -oE 'https://[a-zA-Z0-9-]+.trycloudflare.com'
```

Share the printed `https://*.trycloudflare.com` URL. Keep the ROS launch and
Cloudflare terminals running; the temporary URL is unauthenticated.

The launch alias overrides the planner model for the fixed and interactive flows.

## 12. Interactive calibration verifier (E versus E-hat)

The offline verifier reads one refined calibration capture, parses L1/L2, creates
HRRT routes, accepts a human route choice, and scores both modes:

- `env_uncertainty`: HRRT uses ground-truth `E`; the LLM planner uses perceived
  `E-hat`.
- `planner_only`: both planners use ground-truth `E`; perceived geometry is not
  used around the LLM trajectory.

The displayed tube quantile is `q = S`. `--row` is one-based over sorted
`capture_id` groups.

The L1 field supports microphone recording or an audio-file fallback. The
verifier sends the audio to the configured OpenAI transcription model and puts
the transcript into L1 for review; keep `OPENAI_API_KEY` exported and use
`--transcription-model` to override the default `gpt-4o-mini-transcribe`.

Run with the deterministic mock planner first:

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person.csv \
  --row 2 \
  --intent-provider mock \
  --planner-provider mock \
  --host 0.0.0.0 \
  --port 8091
```

Open [http://127.0.0.1:8091](http://127.0.0.1:8091), enter object-relative L1
and L2, generate routes, choose an HRRT route, and score. The page overlays both
paths, both obstacle geometries, and score-radius tubes.

To use the served refined Llama adapter instead of the mock planner:

```bash
GPU_IP=172.22.224.93
python fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person.csv \
  --row 2 \
  --intent-provider openai \
  --intent-model gpt-5.4-nano \
  --planner-provider vllm \
  --vllm-base-url http://$GPU_IP:8000/v1 \
  --llama-model hrrt_planner \
  --host 0.0.0.0 \
  --port 8091
```

For remote reviewers, keep the verifier bound to `0.0.0.0` and run:

```bash
./cloudflared tunnel --url http://127.0.0.1:8091 2>&1 | grep -oE 'https://[a-zA-Z0-9-]+.trycloudflare.com'
```

Share the printed URL while the verifier and tunnel processes remain running.
Defaults are in `config/llm_vision_planner.yaml`.

## Command help

Show all supported options:

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py --help
python fine_tuning/scripts/build_hrrt_human_audit.py --help
python fine_tuning/scripts/review_hrrt_labels.py --help
python fine_tuning/scripts/finalize_hrrt_sft_dataset.py --help
python fine_tuning/scripts/train_hrrt_peft.py --help
python fine_tuning/scripts/evaluate_hrrt_adapter.py --help
python fine_tuning/scripts/calibration_verifier_gateway.py --help
```
