# HRRT preference-conditioned planner fine-tuning

Distill HRRT-star route preferences into a Llama waypoint planner. The deployed
model receives scene geometry, human preference, and safety constraints only.

The repository also contains the structured Transformer 2.0 path, which reads
the same finalized CSV data, predicts six interior waypoints, copies the exact
start and goal, and trains with Euclidean plus wrong-corridor loss. See
[section 9B](#9b-train-transformer-20-on-ncsa) for its commands.

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
python3 -m pip install --upgrade pip openai pydantic numpy scipy matplotlib
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
python3 fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
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
python3 fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
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
one-obstacle scenes contain all six valid preferences.

Create scene-isolated review assignments:

```bash
python3 fine_tuning/scripts/build_hrrt_human_audit.py \
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
python3 fine_tuning/scripts/review_hrrt_labels.py \
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
python3 fine_tuning/scripts/review_hrrt_labels.py \
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
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
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
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
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

### Teacher-only 80/10/10 scene split (no human audit)

Use this when the structured waypoint transformer should imitate the HRRT
teacher directly and no human route review is available. It does **not** call
the audit builder and does not read audit or review files. It assigns each
complete scene to exactly one split, so no scene appears in both training and
evaluation. The existing Luna raw file currently contains 4,100 scenes and
produces 3,280 training scenes, 410 validation scenes, and 410 test scenes.

Paste this from the repository root:

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw fine_tuning/datasets/dss/hrrt_teacher_raw_luna_4000.jsonl \
  --output-dir fine_tuning/datasets/dss_teacher_only_80_10_10 \
  --clearance-m 0.40 \
  --distillation-mode dss \
  --teacher-only \
  --validation-scene-ratio 0.10 \
  --test-scene-ratio 0.10 \
  --split-seed 42 \
  --reasoning-aux-ratio 0.00 \
  --counterfactual-aux-ratio 0.00
```

For the current raw file, the resulting full-plan CSVs contain 18,499 training
rows, 2,317 validation rows, and 2,295 test rows. The rows are teacher labels;
they are not human-verified. Check those counts directly:

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 - <<'PY'
import csv
from pathlib import Path

root = Path("fine_tuning/datasets/dss_teacher_only_80_10_10")
expected = {"train": 18499, "validation": 2317, "test": 2295}
for split, expected_count in expected.items():
    path = root / f"hrrt_sft_{split}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    full_plan = [row for row in rows if row["task_type"] == "full_plan"]
    assert len(full_plan) == expected_count, (split, len(full_plan), expected_count)
    assert {row["label_source"] for row in full_plan} == {"teacher"}, split
    print(f"{split}: {len(full_plan)} teacher-labelled full-plan rows")
PY
```

### Calibrated vertical person dataset for Transformer 2.0

This generator keeps the project-wide local NED workspace convention:
`x=[-4,4]`, `y=[-3,3]`, fixed `z=-0.5`, forward travel along `+y`, and lateral
variation along `x`. Starts are sampled near the measured observer pose. The
goal is not sampled from language: it is deterministically grounded from
labelled obstacle relations using the same `NEAR` / `FAR_FROM` distance
contract as the mission gateway.

Each layout receives one deterministic goal relation, rotated across six
relation variants by layout ID, and all six preference labels for that fixed
scene and goal. A deterministic 10% of layouts receives a second distinct
goal, again with all six preferences. Therefore 4,000 layouts target 26,400
raw records: 3,600 x 6 plus 400 x 12. Layout IDs are split as a group, so both
goals and every preference from the same geometry stay in the same train,
validation, or test split. The exact 100-layout cycle is: person+chair 36%,
person+chair+stop-sign 36%, person-only 14%, chair-only 5%, stop-sign-only 5%,
and person+stop-sign 4%.

First run one no-API scene and inspect its relations and route cards:

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
VERTICAL_ROOT=fine_tuning/datasets/calibrated_vertical_dss
mkdir -p "$VERTICAL_ROOT"

python3 fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --output "$VERTICAL_ROOT/hrrt_teacher_smoke.jsonl" \
  --scenes 1 \
  --teacher mock \
  --distillation-mode dss \
  --scene-layout calibrated_vertical \
  --preferences-per-scene 6 \
  --multi-goal-layout-ratio 0.0 \
  --minimum-forward-progress-m 2.0 \
  --hrrt-iterations 500 \
  --max-total-nodes 5000 \
  --max-nodes-per-key 180 \
  --max-candidates 8 \
  --min-routes 2 \
  --scene-attempts 8 \
  --clearance-m 0.40

python3 - <<'PY'
import json
row = json.loads(open("fine_tuning/datasets/calibrated_vertical_dss/hrrt_teacher_smoke.jsonl").readline())
print(row["environment"]["scene_profile"])
print("start:", row["environment"]["start"], "goal:", row["environment"]["goal"])
print("deterministic labelled relations:", row["goal_relations"])
PY
```

Generate the full teacher-labelled dataset. In calibrated-vertical mode, the
goal relation rotates across layouts; each fixed layout-goal pair is evaluated
under all six preferences. Preferences select among verified routes only and
never determine the goal itself.

```bash
export OPENAI_API_KEY=YOUR_KEY
TEACHER_MODEL=gpt-6-luna

python3 fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --output "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --scenes 4000 \
  --teacher openai \
  --teacher-model "$TEACHER_MODEL" \
  --reasoning-effort medium \
  --distillation-mode dss \
  --scene-layout calibrated_vertical \
  --preferences-per-scene 6 \
  --multi-goal-layout-ratio 0.10 \
  --goal-sample-resolution-m 0.10 \
  --minimum-forward-progress-m 2.0 \
  --hrrt-iterations 500 \
  --max-total-nodes 5000 \
  --max-nodes-per-key 180 \
  --max-candidates 8 \
  --min-routes 2 \
  --scene-attempts 8 \
  --clearance-m 0.40 \
  --resume
```

Verify the completed calibrated dataset before starting human review or
creating CSVs:

```bash
python3 - "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" <<'PY'
from collections import Counter, defaultdict
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
rows_per_layout = Counter(int(row["scene_id"]) for row in rows)
preferences_per_goal = defaultdict(set)
goals_per_layout = defaultdict(set)

for row in rows:
    scene_id = int(row["scene_id"])
    goal_slot = int(row["goal_slot"])
    goals_per_layout[scene_id].add(goal_slot)
    preferences_per_goal[(scene_id, goal_slot)].add(row["preference"]["preference_type"])

assert len(rows) == 26400, len(rows)
assert len(rows_per_layout) == 4000, len(rows_per_layout)
assert Counter(rows_per_layout.values()) == {6: 3600, 12: 400}, Counter(rows_per_layout.values())
assert Counter(len(value) for value in goals_per_layout.values()) == {1: 3600, 2: 400}
assert all(len(value) == 6 for value in preferences_per_goal.values())
assert all(row["environment"]["workspace"] == {"x": [-4.0, 4.0], "y": [-3.0, 3.0], "z": -0.5} for row in rows)
print("verified: 26,400 rows, 4,000 layouts, 400 multi-goal layouts, 6 preferences per goal")
PY
```

Create human-audit assignments, review them in the existing visualiser, then
create the train/validation/test CSVs. This is the reviewed path.

```bash
python3 fine_tuning/scripts/build_hrrt_human_audit.py \
  --input "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --output "$VERTICAL_ROOT/hrrt_human_audit.jsonl" \
  --train-audit-ratio 0.10 \
  --validation-scene-ratio 0.10 \
  --test-scene-ratio 0.10 \
  --seed 42

python3 fine_tuning/scripts/review_hrrt_labels.py \
  --raw "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$VERTICAL_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$VERTICAL_ROOT/hrrt_human_reviews.jsonl" \
  --host 127.0.0.1 --port 8090

# After every assigned row is reviewed, stop the UI and run:
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$VERTICAL_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$VERTICAL_ROOT/hrrt_human_reviews.jsonl" \
  --output-dir "$VERTICAL_ROOT" \
  --clearance-m 0.40 \
  --distillation-mode dss \
  --reasoning-aux-ratio 0.00 \
  --counterfactual-aux-ratio 0.00
```

For direct Transformer 2.0 teacher imitation without human review, use the
same raw file with the existing scene-isolated splitter instead:

```bash
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --output-dir "$VERTICAL_ROOT/teacher_only_80_10_10" \
  --clearance-m 0.40 \
  --distillation-mode dss \
  --teacher-only \
  --validation-scene-ratio 0.10 \
  --test-scene-ratio 0.10 \
  --split-seed 42 \
  --reasoning-aux-ratio 0.00 \
  --counterfactual-aux-ratio 0.00

TRAIN_CSV="$VERTICAL_ROOT/teacher_only_80_10_10/hrrt_sft_train.csv"
VALIDATION_CSV="$VERTICAL_ROOT/teacher_only_80_10_10/hrrt_sft_validation.csv"
TEST_CSV="$VERTICAL_ROOT/teacher_only_80_10_10/hrrt_sft_test.csv"
test -s "$TRAIN_CSV" && test -s "$VALIDATION_CSV" && test -s "$TEST_CSV"
```

```bash
# Recreate CSVs from the completed human-reviewed Luna labels.
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw fine_tuning/datasets/dss/hrrt_teacher_raw_luna_4000.jsonl \
  --audit fine_tuning/datasets/dss/hrrt_human_audit_4000.jsonl \
  --reviews fine_tuning/datasets/dss/hrrt_human_reviews_4000.jsonl \
  --output-dir fine_tuning/datasets/dss_human_reviewed_4000 \
  --clearance-m 0.40 \
  --distillation-mode dss \
  --reasoning-aux-ratio 0.00 \
  --counterfactual-aux-ratio 0.00 \
  --auxiliary-seed 42
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
python3 -m venv /projects/bhkj/$USER/hf_auth_env
source /projects/bhkj/$USER/hf_auth_env/bin/activate
python3 -m pip install --upgrade huggingface_hub
export HF_HOME=/projects/bhkj/$USER/hf_cache
hf auth login
python3 - <<'PY'
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
python3 src/llm_vision_planner/fine_tuning/scripts/train_hrrt_peft.py \
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

## 9A. Train the structured waypoint transformer on NCSA

For the gated component-by-component verification sequence (dataset audit,
tiny-set overfit, gradient checks, text/scene sensitivity, full training, and
locked-test evaluation), use
[`verification/README.md`](verification/README.md). Its DeltaAI batch job uses
the teacher-only 80/10/10 CSVs and the same production trainer and YAML below.

This alternative trainer can use either the human-reviewed DSS files or the
teacher-only CSVs. For teacher-only training, use these generated files:

- `fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv`
- `fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv`

It reads `preference_text`, parses start/goal/workspace/obstacles from
`environment`, and reads the selected HRRT waypoint label from `completion`.
Reasoning-only auxiliary rows are skipped because they do not contain waypoint
labels. No new feature dataset is generated.

The model fine-tunes every weight in the pretrained MiniLM text encoder and all
new scene/fusion/decoder layers with AdamW. Mean Euclidean waypoint distance in
metres is the only loss used for backpropagation. After each epoch, the
validation predictions pass through offline refinement, verification, the
shared ideal-double-integrator QP, residual scoring, and the contraction-radius
calculation. The radius never contributes gradients. Training stops only after
the radius is at most `0.40 m` for three consecutive epoch validations. If this
never happens, training ends after 100 epochs and records the checkpoint with the
smallest radius without claiming that the safety target passed. `0.40 m` is the
configured default; the submission command below overrides it from the command
line through `SAFETY_RADIUS_THRESHOLD_M`.

Before submission, run this local compatibility check exactly:

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 -m py_compile fine_tuning/scripts/train_hrrt_waypoint_transformer.py
bash -n fine_tuning/scripts/train_hrrt_waypoint_transformer.sbatch
python3 - <<'PY'
from pathlib import Path
import csv

root = Path(".")
for relative in (
    "fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv",
    "fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv",
):
    path = root / relative
    assert path.is_file() and path.stat().st_size > 0, path
    with path.open(newline="", encoding="utf-8") as stream:
        fields = set(csv.DictReader(stream).fieldnames or [])
    assert {"preference_text", "environment", "completion"} <= fields, (path, fields)
print("Waypoint-transformer configuration and DSS datasets are compatible.")
PY
```

Upload the teacher-only CSVs from the local workstation before submitting on
NCSA:

```bash
ssh pgupta12@dtai-login.delta.ncsa.illinois.edu \
  'mkdir -p /projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10'
scp fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_{train,validation,test}.csv \
  pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/
```

First update the NCSA checkout from the committed branch. The currently
installed configuration file on NCSA can be untracked, which makes a normal
`git pull` refuse to overwrite it. The following block preserves that one file
under `.local-backups/`, pulls the committed trainer, YAML configuration, and
batch file, verifies the existing DSS data and all required trainer files, then
submits the job. Paste it exactly at the NCSA login prompt:

```bash
set -euo pipefail
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
mkdir -p .local-backups logs
if [ -e src/llm_vision_planner/fine_tuning/configs/hrrt_waypoint_transformer.yaml ] && ! git ls-files --error-unmatch src/llm_vision_planner/fine_tuning/configs/hrrt_waypoint_transformer.yaml >/dev/null 2>&1; then
  mv src/llm_vision_planner/fine_tuning/configs/hrrt_waypoint_transformer.yaml \
    .local-backups/hrrt_waypoint_transformer.yaml.pre-b317370.$(date +%Y%m%d-%H%M%S)
fi
git pull --ff-only
mkdir -p src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10
test -s src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv
test -s src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv
test -s src/llm_vision_planner/fine_tuning/scripts/train_hrrt_waypoint_transformer.py
test -s src/llm_vision_planner/fine_tuning/scripts/train_hrrt_waypoint_transformer.sbatch
test -s src/llm_vision_planner/fine_tuning/configs/hrrt_waypoint_transformer.yaml
JOB_ID=$(sbatch --parsable --export=ALL,TRAIN_CSV=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv,VALIDATION_CSV=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv,SAFETY_RADIUS_THRESHOLD_M=0.40,MAX_EPOCHS=100 src/llm_vision_planner/fine_tuning/scripts/train_hrrt_waypoint_transformer.sbatch)
printf '%s\n' "$JOB_ID" | tee waypoint_transformer_job_id.txt
```

To request a different stopping radius or maximum epoch count, change only the
corresponding values in the submission command—for example,
`SAFETY_RADIUS_THRESHOLD_M=0.35,MAX_EPOCHS=150`. For a direct Python run, pass
`--safety-radius-threshold-m 0.35 --max-epochs 150`.

Monitor the submitted job with this copy-pasteable command:

```bash
ssh -t pgupta12@dtai-login.delta.ncsa.illinois.edu 'cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea && JOB_ID=$(cat waypoint_transformer_job_id.txt) && squeue -j "$JOB_ID" && while [ ! -f "logs/hrrt-waypoint-transformer-${JOB_ID}.out" ]; do sleep 2; done && tail -f "logs/hrrt-waypoint-transformer-${JOB_ID}.out"'
```

After the job finishes, copy the packaged model and its training summary back to
this workspace:

```bash
mkdir -p /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs
rsync -avP pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer.tar.gz /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/
tar -xzf /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer.tar.gz -C /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs
python3 -m json.tool /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer/training_summary.json
```

The last command prints whether the requested stopping certificate was met, the
configured threshold, best observed radius, best checkpoint, and completed epochs.

### Deploy the structured waypoint transformer on the GPU server

The structured transformer is not a vLLM/LoRA adapter.  The calibration gateway
loads its PyTorch checkpoint directly on the GPU server, sends the predicted
waypoints through the existing `PathRefinement`, `PathVerifier`, and shared QP
stages, and displays the resulting trajectory and residual score. For this
provider, L1 still creates the goal, while L2 is the route preference given to
the transformer because that is the language field on which it was trained.

First, run this exact block at the GPU-server prompt. It updates the gateway
code from this branch, copies the complete transformer output directory from
NCSA, selects the lowest finite-radius checkpoint when one exists (otherwise
the lowest validation imitation-loss checkpoint), and verifies the inference
files:

```bash
set -euo pipefail
PROJECT=/home/prachit2/starling_testing_ws
GPU_OUTPUTS="$PROJECT/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer"
NCSA_OUTPUTS=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer

cd "$PROJECT"
git fetch origin starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
git pull --ff-only
mkdir -p "$GPU_OUTPUTS"
rsync -avP pgupta12@dtai-login.delta.ncsa.illinois.edu:"$NCSA_OUTPUTS/" "$GPU_OUTPUTS/"

CHECKPOINT=$(GPU_OUTPUTS="$GPU_OUTPUTS" python3 - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["GPU_OUTPUTS"])
candidates = []
for metrics_path in root.glob("checkpoint-epoch-*/metrics.json"):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    radius = float(metrics.get("safety_radius_m", math.inf))
    imitation = float(metrics["euclidean_imitation_loss_m"])
    key = (0, radius, imitation) if math.isfinite(radius) else (1, imitation, 0.0)
    candidates.append((key, metrics_path.parent))
if not candidates:
    raise SystemExit(f"No checkpoint metrics found below {root}")
print(min(candidates, key=lambda item: item[0])[1])
PY
)
test -s "$CHECKPOINT/model.pt"
test -s "$CHECKPOINT/config.json"
test -d "$CHECKPOINT/tokenizer"
printf 'Selected checkpoint: %s\n' "$CHECKPOINT"
```

Create the small local inference environment and rebuild the workspace so the
gateway sees the transformer provider:

```bash
set -euo pipefail
PROJECT=/home/prachit2/starling_testing_ws
GPU_ENV=/home/prachit2/hrrt_waypoint_transformer_env
cd "$PROJECT"
source /opt/ros/humble/setup.bash
python3 -m venv "$GPU_ENV"
source "$GPU_ENV/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch transformers scipy numpy pyyaml sentencepiece openai
colcon build --packages-select llm_vision_planner --symlink-install
source install/setup.bash
```

Then start the calibration gateway with the local transformer. This command is
fully offline for planning: `mock` only supplies the goal/route-selection test
logic, while the checkpoint produces the waypoint route. It does not start
vLLM or require a Llama model:

```bash
set -euo pipefail
PROJECT=/home/prachit2/starling_testing_ws
GPU_ENV=/home/prachit2/hrrt_waypoint_transformer_env
GPU_OUTPUTS="$PROJECT/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer"
cd "$PROJECT"
source /opt/ros/humble/setup.bash
source "$GPU_ENV/bin/activate"
source install/setup.bash

CHECKPOINT=$(GPU_OUTPUTS="$GPU_OUTPUTS" python3 - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["GPU_OUTPUTS"])
candidates = []
for metrics_path in root.glob("checkpoint-epoch-*/metrics.json"):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    radius = float(metrics.get("safety_radius_m", math.inf))
    imitation = float(metrics["euclidean_imitation_loss_m"])
    key = (0, radius, imitation) if math.isfinite(radius) else (1, imitation, 0.0)
    candidates.append((key, metrics_path.parent))
if not candidates:
    raise SystemExit(f"No checkpoint metrics found below {root}")
print(min(candidates, key=lambda item: item[0])[1])
PY
)

python3 fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person.csv \
  --row 2 \
  --intent-provider mock \
  --planner-provider transformer \
  --transformer-checkpoint "$CHECKPOINT" \
  --transformer-device cuda:0 \
  --clearance-m 0.40 \
  --host 0.0.0.0 \
  --port 8091
```

```bash
cloudflared tunnel --url http://127.0.0.1:8091
```

Open `http://172.22.224.93:8091`. Enter L1 to create the goal, accept the
goal, then enter an L2 preference such as `Give every obstacle the widest
practical berth.` The gateway applies its usual two-attempt refinement and
verification procedure. The transformer receives the scene, generated goal,
and L2 preference on each attempt; it does not receive teacher reasoning or
the displayed HRRT routes.

## 9B. Train Transformer 2.0 on NCSA

Transformer 2.0 is separate from the Llama/QLoRA and original structured
transformer runs. Its input is the preference text plus numeric start, goal,
and obstacle tokens. The fixed workspace bounds are used only for coordinate
normalization; they are not a scene token. Obstacle labels are one-hot encoded
as `[person, chair, stop_sign]`: person is `[1,0,0]`, chair is `[0,1,0]`, and
stop sign is `[0,0,1]`. It predicts only six interior points; the exact scene
start and goal are copied into the returned eight-point route.

Its training objective is

```text
objective = mean interior Euclidean error
          + CORRIDOR_LOSS_WEIGHT * wrong-corridor hinge error
          + CLEARANCE_BIN_LOSS_WEIGHT * clearance-bin hinge error
          + SQUARED_TAIL_LOSS_WEIGHT * mean(error²) / 1.0 m
```

The committed defaults are `CORRIDOR_LOSS_WEIGHT=5.0`,
`CLEARANCE_BIN_LOSS_WEIGHT=1.0`, `CORRIDOR_MARGIN_M=0.20`,
`SQUARED_TAIL_LOSS_WEIGHT=1.0`, `corridor_temperature_m=0.35`, and 100 epochs.
The selected teacher route's
stored composite key `(h_signature, clearance_bins)` provides the corridor and
spacing labels; teacher reasoning is not used.

Before uploading or submitting, verify the committed trainer locally:

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 -m py_compile fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.py
bash -n fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.sbatch
python3 -m pytest -q test/test_transformer_2_0.py
```

If the teacher-only CSVs are not already on NCSA, upload them from the local
repository:

```bash
LOCAL_DATASET_ROOT=fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10
REMOTE_DATASET_ROOT=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10

ssh pgupta12@dtai-login.delta.ncsa.illinois.edu \
  "mkdir -p '$REMOTE_DATASET_ROOT'"
scp "$LOCAL_DATASET_ROOT"/hrrt_sft_{train,validation,test}.csv \
  "pgupta12@dtai-login.delta.ncsa.illinois.edu:$REMOTE_DATASET_ROOT/"
```

Then paste this block at the NCSA login prompt. It updates the branch, checks
the exact inputs and Transformer 2.0 files, submits one GPU job, and saves the
job ID:

```bash
set -euo pipefail
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
git pull --ff-only
mkdir -p logs

PACKAGE_ROOT=$PWD/src/llm_vision_planner
DATASET_ROOT=$PACKAGE_ROOT/fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10

test -s "$DATASET_ROOT/hrrt_sft_train.csv"
test -s "$DATASET_ROOT/hrrt_sft_validation.csv"
test -s "$DATASET_ROOT/hrrt_sft_test.csv"
test -s "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.py"
test -s "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.sbatch"
test -s "$PACKAGE_ROOT/fine_tuning/configs/hrrt_waypoint_transformer_2_0.yaml"

JOB_ID=$(sbatch --parsable \
  --export=ALL,TRAIN_CSV=$DATASET_ROOT/hrrt_sft_train.csv,VALIDATION_CSV=$DATASET_ROOT/hrrt_sft_validation.csv,MAX_EPOCHS=100,CORRIDOR_LOSS_WEIGHT=5.0,CORRIDOR_MARGIN_M=0.20,CLEARANCE_BIN_LOSS_WEIGHT=1.0,SQUARED_TAIL_LOSS_WEIGHT=1.0,SAFETY_RADIUS_THRESHOLD_M=0.40 \
  "$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.sbatch")

printf '%s\n' "$JOB_ID" | tee transformer_2_0_job_id.txt
```

Monitor the job:

```bash
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
JOB_ID=$(cat transformer_2_0_job_id.txt)
squeue -j "$JOB_ID"
tail -f "logs/hrrt-transformer-2-0-${JOB_ID}.out"
```

After completion, inspect the training summary and final validation history:

```bash
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
OUTPUT=src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer_2_0
python3 -m json.tool "$OUTPUT/training_summary.json"
python3 - "$OUTPUT/training_history.json" <<'PY'
import json
import sys

history = json.load(open(sys.argv[1], encoding="utf-8"))
last = history[-1]
for key in (
    "euclidean_imitation_loss_m",
    "interior_p80_episode_error_m",
    "corridor_hinge_loss_m",
    "wrong_corridor_rate",
    "clearance_bin_hinge_loss_m",
    "clearance_bin_accuracy",
    "squared_tail_loss_m",
    "validation_objective_loss_m",
):
    print(f"{key}: {last[key]}")
PY
```

The batch script also creates
`fine_tuning/outputs/hrrt_waypoint_transformer_2_0.tar.gz`. Supported submission
overrides are `CONFIG`, `TRAIN_CSV`, `VALIDATION_CSV`, `OUTPUT_DIR`,
`MAX_EPOCHS`, `CORRIDOR_LOSS_WEIGHT`, `CORRIDOR_MARGIN_M`,
`CLEARANCE_BIN_LOSS_WEIGHT`,
`SQUARED_TAIL_LOSS_WEIGHT`,
`SAFETY_RADIUS_THRESHOLD_M`, `TRAIN_ENV_DIR`, and `HF_HOME`.

### Transfer Transformer 2.0 and build the local verification website

Run this on the laptop only after the NCSA job has finished. It copies the
complete output directory, including every checkpoint, tokenizer, shared text
encoder, metrics file, training history, and training summary. It does not
delete any local files.

```bash
set -euo pipefail
LOCAL_PROJECT=/home/prachit/Desktop/starling_multiple_trajectory_idea
LOCAL_OUTPUT=$LOCAL_PROJECT/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer_2_0
NCSA_OUTPUT=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/hrrt_waypoint_transformer_2_0

mkdir -p "$LOCAL_OUTPUT"
rsync -avP --partial \
  "pgupta12@dtai-login.delta.ncsa.illinois.edu:$NCSA_OUTPUT/" \
  "$LOCAL_OUTPUT/"
```

Confirm that the transfer is complete and that every copied checkpoint is a
Transformer 2.0 checkpoint:

```bash
python3 - "$LOCAL_OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
assert (root / "training_history.json").is_file()
assert (root / "training_summary.json").is_file()
assert (root / "text_encoder").is_dir()
checkpoints = sorted(root.glob("checkpoint-epoch-*"))
assert checkpoints, "no checkpoint directories were transferred"
for checkpoint in checkpoints:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["architecture_version"] == "transformer_2_0", checkpoint
    assert (checkpoint / "model.pt").is_file(), checkpoint
    assert (checkpoint / "metrics.json").is_file(), checkpoint
    assert (checkpoint / "tokenizer").is_dir(), checkpoint
print(f"verified {len(checkpoints)} Transformer 2.0 checkpoints in {root}")
PY
```

Use the scripts in `fine_tuning/verification/scripts` to audit the local
dataset, evaluate the best transferred checkpoint on validation and locked
test scenes, and generate the HTML dashboard. `VERIFY_DEVICE=cpu` works on a
CPU-only laptop; change it to `cuda:0` when a compatible local GPU is available.

```bash
set -euo pipefail
PACKAGE_ROOT=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
MODEL_OUTPUT=$PACKAGE_ROOT/fine_tuning/outputs/hrrt_waypoint_transformer_2_0
DATASET_ROOT=$PACKAGE_ROOT/fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10
VERIFY_RUN=$PACKAGE_ROOT/fine_tuning/outputs/verification-transformer-2-0-local
VERIFY_DEVICE=cpu
CONFIG=$PACKAGE_ROOT/fine_tuning/configs/hrrt_waypoint_transformer_2_0.yaml

mkdir -p "$VERIFY_RUN/reports" "$VERIFY_RUN/plots"
test -s "$DATASET_ROOT/hrrt_sft_train.csv"
test -s "$DATASET_ROOT/hrrt_sft_validation.csv"
test -s "$DATASET_ROOT/hrrt_sft_test.csv"
test -s "$MODEL_OUTPUT/training_history.json"
test -s "$MODEL_OUTPUT/training_summary.json"

python3 "$PACKAGE_ROOT/fine_tuning/verification/scripts/verify_transformer_components.py" \
  --stage preflight \
  --config "$CONFIG" \
  --train "$DATASET_ROOT/hrrt_sft_train.csv" \
  --validation "$DATASET_ROOT/hrrt_sft_validation.csv" \
  --test "$DATASET_ROOT/hrrt_sft_test.csv" \
  --output "$VERIFY_RUN/reports/preflight.json" \
  --device "$VERIFY_DEVICE" \
  --tiny-samples 8 \
  --tiny-updates 250 \
  --tiny-lr-multiplier 5.0 \
  --required-overfit-reduction 0.50

python3 "$PACKAGE_ROOT/fine_tuning/verification/scripts/verify_transformer_components.py" \
  --stage postflight \
  --config "$CONFIG" \
  --train "$DATASET_ROOT/hrrt_sft_train.csv" \
  --validation "$DATASET_ROOT/hrrt_sft_validation.csv" \
  --test "$DATASET_ROOT/hrrt_sft_test.csv" \
  --model-output "$MODEL_OUTPUT" \
  --output "$VERIFY_RUN/reports/postflight.json" \
  --device "$VERIFY_DEVICE"

python3 "$PACKAGE_ROOT/fine_tuning/verification/scripts/plot_verification_results.py" \
  --run-root "$VERIFY_RUN" \
  --model-output "$MODEL_OUTPUT" \
  --output-dir "$VERIFY_RUN/plots"
```

Serve the generated website locally and open
`http://127.0.0.1:8092/index.html`:

```bash
python3 -m http.server 8092 \
  --directory /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/verification-transformer-2-0-local/plots
```

The website contains the training curves, gradient and text/scene sensitivity
diagnostics, validation/test error distribution, and six locked-test route
comparisons. The orange routes are raw Transformer 2.0 predictions; the green
routes are teacher HRRT targets.

### Run the calibration verifier with Transformer 2.0

After transferring the completed output, run the gateway from the laptop. This
selects the best finite-radius checkpoint, or the lowest Euclidean-error
checkpoint when no finite radius exists:

```bash
set -euo pipefail
PACKAGE_ROOT=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
MODEL_OUTPUT=$PACKAGE_ROOT/fine_tuning/outputs/hrrt_waypoint_transformer_2_0

CHECKPOINT=$(python3 - "$MODEL_OUTPUT" <<'PY'
import json
import math
from pathlib import Path
import sys

candidates = []
for metrics_path in Path(sys.argv[1]).glob("checkpoint-epoch-*/metrics.json"):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    radius = float(metrics.get("safety_radius_m", math.inf))
    error = float(metrics["euclidean_imitation_loss_m"])
    key = (0, radius, error) if math.isfinite(radius) else (1, error, 0.0)
    candidates.append((key, metrics_path.parent))
assert candidates, "no transferred Transformer 2.0 checkpoints found"
print(min(candidates, key=lambda item: item[0])[1])
PY
)

cd "$PACKAGE_ROOT"
python3 fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person.csv \
  --row 2 \
  --intent-provider mock \
  --planner-provider transformer \
  --transformer-checkpoint "$CHECKPOINT" \
  --transformer-device auto \
  --clearance-m 0.40 \
  --host 0.0.0.0 \
  --port 8091
```

Keep the gateway running. In a second terminal, start the Cloudflare tunnel;
`grep` prints the public website address as soon as Cloudflare creates it:

```bash
cloudflared tunnel --url http://127.0.0.1:8091 2>&1 \
  | grep --line-buffered -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com'
```

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
python3 src/llm_vision_planner/fine_tuning/scripts/evaluate_hrrt_adapter.py \
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

The offline verifier parses L1 and displays a proposed goal first. The reviewer
must accept it, reject it, or enter a manual goal before HRRT routes are created
and plotted. L2 then goes to the OpenAI expert selector, which chooses one
displayed route before scoring both modes. Live stages show selection, raw LLM
waypoints, verification, QP optimization, and final safety tubes:

- `env_uncertainty`: HRRT uses ground-truth `E`; the LLM planner uses perceived
  `E-hat`.
- `planner_only`: both planners use ground-truth `E`; perceived geometry is not
  used around the LLM trajectory.

The displayed tube quantile is `q = S`. `--row` is one-based over sorted
`capture_id` groups.

The L1 and L2 fields support microphone recording or an audio-file fallback.
The verifier sends audio to the configured OpenAI transcription model and puts
each transcript into its field for review; keep `OPENAI_API_KEY` exported and use
`--transcription-model` to override the default `gpt-4o-mini-transcribe`.

Run with the deterministic mock planner first:

```bash
cd ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person.csv \
  --row 2 \
  --intent-provider mock \
  --planner-provider mock \
  --host 0.0.0.0 \
  --port 8091
```

Open [http://127.0.0.1:8091](http://127.0.0.1:8091), enter L1 and propose a goal.
Accept, reject, or replace it manually; only then is HRRT generation enabled.
Review the plotted candidates, enter L2, then let the selector choose the expert
route and score. The gateway reuses the production ROS2 modules
`prompt_generator.py`, `refinment.py`, and `verifier.py` without starting ROS
nodes: the fixed prompt is built by `PromptGenerator`, sparse paths are
interpolated by `PathRefinement`, and the same `PathVerifier` metrics are used
for HRRT and LLM candidates. A failed LLM route stays on the plot in orange
dashes; its verification feedback table is appended to the same prompt for
exactly one retry, shown in blue. QP optimization and residual scoring run only
after the HRRT and (possibly refined) LLM routes pass. The final JSON records
raw/refined waypoints and verification metrics.
The color key remains visible during every live stage. Completion stops polling
without reloading: both mode scores are printed, the selected score is stamped
on the plot, and the final frame stays locked until the dataset row changes.
The persistent status history labels Trial 1/2 generation, refinement, and
verification. A failed first trial triggers Trial 2/2 with the full verifier
feedback table; the raw and refined plot updates to Trial 2 before its final
pass/fail result. Two failures leave the Trial 2 plot visible and skip QP/score.

To use the served refined Llama adapter instead of the mock planner:

```bash
GPU_IP=172.22.224.93
python3 fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person.csv \
  --row 2 \
  --intent-provider openai \
  --intent-model gpt-5.4-nano \
  --expert-model gpt-6.1-sol \
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
python3 fine_tuning/scripts/generate_hrrt_finetuning_dataset.py --help
python3 fine_tuning/scripts/build_hrrt_human_audit.py --help
python3 fine_tuning/scripts/review_hrrt_labels.py --help
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py --help
python3 fine_tuning/scripts/train_hrrt_peft.py --help
python3 fine_tuning/scripts/train_hrrt_waypoint_transformer.py --help
python3 fine_tuning/scripts/evaluate_hrrt_adapter.py --help
python3 fine_tuning/scripts/calibration_verifier_gateway.py --help
```
