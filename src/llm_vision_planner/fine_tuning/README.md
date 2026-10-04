# HRRT preference-conditioned Llama fine-tuning

This workflow distills an HRRT-star teacher into a Llama waypoint planner. Llama
receives only the deployment scene, human preference, and safety constraints. It
never receives route cards or candidate routes.

Two dataset modes are available:

- `dss`: full-plan supervision plus a sampled reasoning-only auxiliary task.
- `dss_scott`: a minimal SCOTT-style extension of the DSS data with paired positive/counterfactual
  rationale-conditioned waypoint tasks. Counterfactual HRRT routes are verified
  safe and appear only in explicitly prefixed auxiliary training tasks.

`dss_scott` intentionally keeps the existing standard LoRA/SFT formulation. It
tests SCOTT's counterfactual-conditioning idea without implementing the paper's
token-level contrastive decoder or a new custom loss.

Both modes keep the deployed response unchanged:

```json
{"reasoning":"Preference: ... Geometry: ... Decision: ... Safety: ...","waypoints":[{"x":0,"y":0,"z":-0.5}]}
```

Run stages 1–5 on the local workstation. Run stages 6–10 on NCSA DeltaAI.

## 1. Enter the correct worktree and install data-generation dependencies

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 -m venv .venv-hrrt-data
source .venv-hrrt-data/bin/activate
python -m pip install --upgrade pip openai pydantic numpy scipy matplotlib
export OPENAI_API_KEY=YOUR_KEY
```

Choose one mode. Keep the two modes in different directories so raw labels,
reviews, splits, adapters, and metrics cannot overwrite each other.

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

This command intentionally spells out every generator option. `--teacher mock`
uses deterministic labels and does not require `OPENAI_API_KEY`.

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --output "$DATASET_ROOT/hrrt_teacher_smoke.jsonl" \
  --scenes 10 \
  --preferences-per-scene 6 \
  --seed 1701 \
  --teacher mock \
  --teacher-model gpt-5.4 \
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

`--min-obstacles` and `--max-obstacles` accept values from `0` through `4`.
Use at least `1` for useful HRRT preference data: zero-obstacle scenes normally
fail the generator's non-trivial-route check because the direct start-to-goal
segment is clear.

Synthetic teacher-data scenes use only the three deployment test classes:
`person`, `chair`, and `stop_sign`. If a scene contains four obstacles, one of
these classes is repeated with a distinct object ID.

Add `--resume` only when continuing the same output file. Resume refuses to mix
`dss` and `dss_scott` records. Without `--resume`, an existing output file is
replaced.

## 3. Generate teacher labels

The teacher sees the full scene, route cards, and candidate waypoints. DSS uses
one API call per preference to select and explain a route. DSS-SCOTT uses two:
one route-selection call followed by an answer-conditioned rationale call with
the positive and safe counterfactual trajectories fixed. For 2,000 scenes and
six preferences, this is 12,000 DSS calls or 24,000 DSS-SCOTT calls.

The Llama-facing explanation cannot mention route IDs, cards, candidates, or
unavailable comparisons. Invalid explanations fall back to a deterministic
scene-grounded template.

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --output "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --scenes 2000 \
  --preferences-per-scene 6 \
  --seed 1701 \
  --teacher openai \
  --teacher-model gpt-5.4 \
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

If interrupted, rerun exactly the same command. Resume removes only an incomplete
scene before continuing, so a scene cannot contain a partial preference set.

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

Every validation and test label, plus a stratified training sample, is assigned
to human review. Splits are made by scene rather than individual preference row.

## 4. Review teacher labels and reasoning

```bash
python fine_tuning/scripts/review_hrrt_labels.py \
  --raw "$DATASET_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$DATASET_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$DATASET_ROOT/hrrt_human_reviews.jsonl" \
  --host 127.0.0.1 \
  --port 8090
```

Open [http://127.0.0.1:8090](http://127.0.0.1:8090). For every assignment:

1. Read the human preference, audit reason, and Llama-facing reason.
2. Compare the colored route plot and route metrics.
3. Accept the teacher, choose a corrected route, or mark the sample ambiguous.
4. A corrected route requires a human reason. Prefer the same compact structure:
   `Preference`, `Geometry`, `Decision`, and `Safety`.

The UI saves after every decision. A valid human reason overrides the teacher
reason. If a corrected route has an unusable reason, finalization constructs a
factual deterministic explanation for the corrected route. Ambiguous rows are
removed.

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

For DSS-SCOTT, use the same command but enable paired counterfactual auxiliary
tasks:

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

`--reasoning-aux-ratio` and `--counterfactual-aux-ratio` accept values from zero
through one. Auxiliary rows are added only to training. Validation and test retain
only the exact one-call deployment task.

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

For an existing clone:

```bash
cd /projects/bhkj/$USER/starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
git pull --ff-only
mkdir -p logs
```

## 7. Upload one finalized mode

Run locally after setting `DISTILLATION_MODE` and `DATASET_ROOT` as in stage 1:

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

The batch script accepts optional exported overrides named `PROJECT_ROOT`,
`HRRT_ENV_DIR`, `HF_HOME`, `DISTILLATION_MODE`, `DATASET_ROOT`, and `OUTPUT`.
Its default output is
`fine_tuning/outputs/llama31_8b_hrrt_lora_<mode>`.

Monitor the job:

```bash
squeue -u $USER
tail -f logs/hrrt-peft-JOB_ID.out
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```

For a direct interactive run, this command lists every trainer argument:

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

Omit `--qlora` only when intentionally loading the base model without 4-bit NF4.

## 10. Evaluate the locked test split

Request a GPU session:

```bash
srun --account=bhkj-dtai-gh --partition=ghx4 --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=16 --gpus-per-node=1 --mem=128g --time=01:00:00 --pty bash
```

Evaluate every supported option explicitly:

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

To score previously generated outputs instead of loading a model, add:

```text
--predictions /absolute/path/to/predictions.jsonl
```

The report includes structured-output validity, reasoning-schema validity,
geometric feasibility, selected-route agreement, endpoint error, and mean path
error.

## 11. Download and select an adapter at launch

Run on the computer that will serve Llama:

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs
DISTILLATION_MODE=dss
scp pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora_$DISTILLATION_MODE.tar.gz .
tar -xzf "llama31_8b_hrrt_lora_$DISTILLATION_MODE.tar.gz"
test -s "llama31_8b_hrrt_lora_$DISTILLATION_MODE/adapter_config.json"
test -s "llama31_8b_hrrt_lora_$DISTILLATION_MODE/adapter_model.safetensors"
```

Serve the selected adapter with an OpenAI-compatible model alias, for example
`hrrt_planner_dss` or `hrrt_planner_dss_scott`. Then pass that same alias to the
ROS launch file:

```bash
ros2 launch llm_vision_planner full_plot.launch.py \
  llm_provider:=llama \
  llama_model_name:=hrrt_planner_dss \
  interaction_mode:=interactive
```

The launch argument overrides the compatible defaults in
`config/llm_vision_planner.yaml` for the fixed prompt generator, interactive
gateway, and Llama planner together.

## Command help

The commands above expose every supported option. The installed scripts also
provide authoritative summaries:

```bash
python fine_tuning/scripts/generate_hrrt_finetuning_dataset.py --help
python fine_tuning/scripts/build_hrrt_human_audit.py --help
python fine_tuning/scripts/review_hrrt_labels.py --help
python fine_tuning/scripts/finalize_hrrt_sft_dataset.py --help
python fine_tuning/scripts/train_hrrt_peft.py --help
python fine_tuning/scripts/evaluate_hrrt_adapter.py --help
```
