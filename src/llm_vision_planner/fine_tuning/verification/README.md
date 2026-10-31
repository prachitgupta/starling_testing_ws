# Automated loss-ablation study — NCSA copy-paste guide

This is the new multi-run workflow. The older single-run verification guide is
retained below. **Do not submit the old `run_transformer_verification.sbatch`
for an ablation study**: it evaluates test after every run.

## What you do, and when

1. **Once, on the laptop:** pull code and upload the three CSVs and original
   Transformer 1 epoch-40 reference artifacts.
2. **Once, on NCSA:** prepare the environment, initialize a study, and submit its
   first job. Subsequent jobs are submitted automatically, one at a time.
3. **While it runs:** optionally view logs or download the dashboard. No manual
   choice of architecture, loss or epoch is required.
4. **When status is `complete`:** download the dashboard and final checkpoint.
   A `failed` status needs log inspection and resubmission after fixing the cause.

The scripts are implemented; a production NCSA study is not submitted by merely
pulling this repository. Initialization reads/audits files but does not train.

## What the study runs

```text
Audit frozen dataset splits and reference checkpoint
  → evaluate original Transformer 1 epoch 40 on VALIDATION
  → train 12-feature Transformer 2, Euclidean only, seed 42
  → select architecture by validation SIX-interior-point mean error
  → fresh Euclidean control if Transformer 1 wins
  → independent E1 / E2 / E3 loss runs on selected architecture
  → optional combined E4 when both individual penalties improve
  → repeat E0 and leading auxiliary variant with seeds 43 and 44
  → freeze configuration and representative checkpoint
  → one TEST evaluation → final interactive dashboard
```

| Variant | Objective |
| --- | --- |
| E0 | Mean Euclidean distance over six interior points |
| E1 | E0 + 0.25 × mean squared distance / 1 metre |
| E2 | E0 + 1 × corridor-side hinge |
| E3 | E0 + 5 × corridor-side hinge |
| E4 (conditional) | E0 + E1's squared term + the better E2/E3 corridor weight |

Clearance-bin loss is disabled; bin agreement is still measured. Runs start
from base MiniLM plus newly initialized planner weights, not another run's
trained checkpoint. Matching seeds reproduce initialization and batch ordering.

The historical Transformer 1 reference retains its native 10-feature tokens and
eight predicted points. If it wins, fresh V1 runs retain its smaller body and
10-feature tokenization but use six queries with fixed start/goal. This is
explicitly a new E0 control, not a claim that it reproduces the historical run.
All fresh V2 runs have 12-feature tokens and six learned points. V2 still uses
an MLP projection before its scene encoder. Workspace bounds remain available
for coordinate normalization even though V2 has no workspace token.

**Selection rule:** per-checkpoint and per-run ranking always uses validation
mean six-interior-point Euclidean error. The first architecture screen is a
single-seed practical comparison of configurations, not a causal test of layers
alone. After loss screening, the leading auxiliary variant is accepted only
when its mean paired improvement over E0 is at least **0.02 m** and a majority
of the three seeds improve. Otherwise E0 is retained as inconclusive/no gain.
The final checkpoint uses the median-validation seed of the selected variant.
Safety radius, combined objective and test metrics never choose the winner.

Each fresh run has a common **100-epoch cosine schedule**, validation after each
epoch, and best-checkpoint retention. There is no validation-patience or
safety-radius stopping in this first study. Full refinement/verifier/QP
diagnostics run on each selected checkpoint, not every epoch. This avoids
spending most of the study repeating expensive downstream checks. Expect
roughly 8–10 complete training runs plus evaluation and queue time, not one
12-hour job. Each job has a 12-hour allocation; continuation restores the same
schedule and optimizer at the last saved epoch.

## 1. Laptop: pull this branch and prepare remote directories

Run in a **laptop terminal**. Complete NCSA Duo authentication when prompted.

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea
git switch starling_multiple_trajectory_idea
git pull --ff-only origin starling_multiple_trajectory_idea

export STUDY_REMOTE=pgupta12@dtai-login.delta.ncsa.illinois.edu
export LOCAL_PACKAGE=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner

ssh "$STUDY_REMOTE" 'bash -s' <<'REMOTE'
set -euo pipefail
PROJECT_ROOT=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea
if [ ! -d "$PROJECT_ROOT/.git" ]; then
  git clone --branch starling_multiple_trajectory_idea https://github.com/prachitgupta/starling_testing_ws.git "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"
git switch starling_multiple_trajectory_idea
git pull --ff-only origin starling_multiple_trajectory_idea
mkdir -p logs
mkdir -p src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10
mkdir -p src/llm_vision_planner/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer
REMOTE
```

If Git reports local modifications or diverged history, resolve them before
continuing; do not reset away your server work. Do not pull source changes in
the middle of an initialized study: source hashes are intentionally frozen.

## 2. Laptop: copy the actual data and reference model

These files are local artifacts, not supplied by `git pull`. Copying only JSON
metrics is insufficient; inference needs the actual weights and tokenizer.

```bash
export STUDY_REMOTE=pgupta12@dtai-login.delta.ncsa.illinois.edu
export LOCAL_PACKAGE=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
export REMOTE_PACKAGE=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner
export LOCAL_BASELINE="$LOCAL_PACKAGE/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer"

for SPLIT in train validation test; do
  test -s "$LOCAL_PACKAGE/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_${SPLIT}.csv"
  rsync -avP "$LOCAL_PACKAGE/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_${SPLIT}.csv" \
    "$STUDY_REMOTE:$REMOTE_PACKAGE/fine_tuning/datasets/dss_teacher_only_80_10_10/"
done
test -s "$LOCAL_BASELINE/checkpoint-epoch-040/model.pt"
test -s "$LOCAL_BASELINE/checkpoint-epoch-040/config.json"
test -d "$LOCAL_BASELINE/checkpoint-epoch-040/tokenizer"
test -d "$LOCAL_BASELINE/text_encoder"
rsync -avP "$LOCAL_BASELINE/checkpoint-epoch-040" "$LOCAL_BASELINE/text_encoder" \
  "$STUDY_REMOTE:$REMOTE_PACKAGE/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer/"
```

This transfers the original Transformer 1 epoch 40. It does not use the old
13-feature Transformer 2 checkpoint. The reference config records dataset
paths but no original file hashes; exact historical training CSV contents
cannot retrospectively be certified. All new runs use the newly hashed files.

## 3. NCSA: prepare environment and initialize the study

From the laptop:

```bash
ssh pgupta12@dtai-login.delta.ncsa.illinois.edu
```

Then paste at the **NCSA prompt**:

```bash
set -euo pipefail
export PROJECT_ROOT=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea
export PACKAGE_ROOT="$PROJECT_ROOT/src/llm_vision_planner"
export TRAIN_ENV_DIR=/projects/bhkj/pgupta12/hrrt_waypoint_transformer_env
export HF_HOME=/projects/bhkj/pgupta12/hf_cache
export STUDY_ROOT="$PACKAGE_ROOT/fine_tuning/outputs/loss-ablation-001"
cd "$PROJECT_ROOT"
module purge
module load cray-python
if [ ! -x "$TRAIN_ENV_DIR/bin/python3" ]; then
  python3 -m venv "$TRAIN_ENV_DIR"
fi
source "$TRAIN_ENV_DIR/bin/activate"
python3 -m pip install torch transformers scipy numpy pyyaml sentencepiece matplotlib
mkdir -p logs

python3 "$PACKAGE_ROOT/fine_tuning/verification/scripts/run_loss_ablation.py" init \
  --study-root "$STUDY_ROOT" \
  --dataset-root "$PACKAGE_ROOT/fine_tuning/datasets/dss_teacher_only_80_10_10" \
  --baseline-checkpoint "$PACKAGE_ROOT/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer/checkpoint-epoch-040" \
  --max-epochs 100 --minimum-improvement-m 0.02 --device cuda
```

Initialization checks scene and geometry separation, duplicate sample IDs,
coordinates, baseline input dimensions, and required artifacts. During each
architecture's preflight, an eight-example overfit/gradient/sensitivity check
must pass before full training. Do not rerun `init` for a study that already
exists; use resume below. Choose a new study name for a changed configuration.
Core package versions are recorded at initialization and checked by each worker;
do not upgrade the environment during a study.

## 4. NCSA: submit once

In the same NCSA terminal:

```bash
cd "$PROJECT_ROOT"
JOB_ID=$(sbatch --parsable \
  --export=ALL,STUDY_ROOT="$STUDY_ROOT",PROJECT_ROOT="$PROJECT_ROOT",TRAIN_ENV_DIR="$TRAIN_ENV_DIR" \
  "$PACKAGE_ROOT/fine_tuning/verification/scripts/run_loss_ablation.sbatch")
printf '%s\n' "$JOB_ID" >> "$STUDY_ROOT/job_ids.txt"
printf 'First job: %s\n' "$JOB_ID"
```

Each successful job submits its successor automatically. You can close SSH.
No credentials, HTTP server, or laptop process must stay active for training.
The worker does not install packages or rewrite shared environments.

## 5. NCSA: check progress, failure, or resume

In a new NCSA terminal, restore the paths:

```bash
export PROJECT_ROOT=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea
export PACKAGE_ROOT="$PROJECT_ROOT/src/llm_vision_planner"
export TRAIN_ENV_DIR=/projects/bhkj/pgupta12/hrrt_waypoint_transformer_env
export STUDY_ROOT="$PACKAGE_ROOT/fine_tuning/outputs/loss-ablation-001"
cd "$PROJECT_ROOT"
squeue -u "$USER"
python3 -m json.tool "$STUDY_ROOT/state.json"
JOB_ID=$(tail -n 1 "$STUDY_ROOT/job_ids.txt")
tail -n 60 "logs/hrrt-loss-study-${JOB_ID}.out"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,ExitCode
```

The state lists every completed run, validation score, selected checkpoint, and
the current/next task. `ready` normally means the next job is queued; `running`
can remain after an abrupt kill, so check SLURM before submitting anything again.

For an interrupted study with **no job still running or queued**, use the same
submission block from step 4. An atomic epoch checkpoint restores model,
optimizer, scheduler, scaler, RNGs, best weights and history. An interrupted
epoch is repeated. The job requests an orderly continuation before its time
limit; a hard timeout or failed successor submission requires manual resubmission.
Concurrent workers are rejected by a file lock. Do not alter code/data to bypass
hash checking; restore matching files or initialize a new study.

Completed jobs are not rerun. Failed preflight results are retained for
inspection; correcting model/data/preflight settings should use a new study.

## 6. Laptop: download and open the dashboard at any time

Run in a **laptop terminal**, not inside SSH:

```bash
export STUDY_REMOTE=pgupta12@dtai-login.delta.ncsa.illinois.edu
export REMOTE_STUDY=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/loss-ablation-001
export LOCAL_STUDY=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/loss-ablation-001
mkdir -p "$LOCAL_STUDY/dashboard"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/dashboard/" "$LOCAL_STUDY/dashboard/"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/state.json" "$LOCAL_STUDY/"
xdg-open "$LOCAL_STUDY/dashboard/index.html"
```

Repeat the transfer and refresh the page whenever you want updated results.
The HTML embeds all available charts and scene data: it works offline with no
CDN or account. Select **Model A, Model B, then Scene** to compare the same
scene across all its preferences. Expand a card for actual teacher/predicted
waypoints. The highest-error-scenes toggle sorts by Model A's scene errors.
During development all displayed study routes are from validation. Test routes
appear only under `final` after the selection is frozen. Select `final` in
both model menus to view them; validation/test scenes cannot be mixed.

For a local web URL instead of opening the file:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$LOCAL_STUDY/dashboard"
```

Open http://127.0.0.1:8765 in a browser. Keep that terminal open while viewing.

The leaderboard includes interior mean/median/P95/max, episode P80, endpoint
and all-eight-point errors, wrong-side decisions, clearance-bin accuracy,
pipeline pass rate, radius, total/planner parameter counts, training time, and best epoch. Training plots
include raw Euclidean and unweighted component losses; seed reports show mean
and standard deviation. Diagnostics expose per-component gradient norms and
input swaps across up to 12 scenes. These swaps are not a normalized feature
importance score. Missing/infinite safety radius is displayed as unavailable,
never as zero. A passed component test is not a flight-safety certificate.

## 7. Laptop: collect final reports and deployable-format checkpoint

Only after `state.json` says `complete`, using the laptop variables from step 6:

```bash
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/manifest.json" \
  "$STUDY_REMOTE:$REMOTE_STUDY/audit.json" \
  "$STUDY_REMOTE:$REMOTE_STUDY/selection.json" \
  "$STUDY_REMOTE:$REMOTE_STUDY/final_test.json" "$LOCAL_STUDY/"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/reports/" "$LOCAL_STUDY/reports/"
WINNER=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_run"])' "$LOCAL_STUDY/selection.json")
mkdir -p "$LOCAL_STUDY/runs/$WINNER"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/best" \
  "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/text_encoder" \
  "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/tokenizer" \
  "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/history.json" \
  "$LOCAL_STUDY/runs/$WINNER/"
printf 'Selected checkpoint: %s/runs/%s/best\n' "$LOCAL_STUDY" "$WINNER"
```

Checkpoint layout remains `model.pt`, `config.json`, `tokenizer/`, with a sibling
base `text_encoder/`. Production builders load the new optional V1
`fixed_endpoints` configuration; existing V1 checkpoints still load unchanged.
Transfer compatibility does not establish hardware deployment performance.
The test CSV has already been examined in prior development, so the report
explicitly labels it as an existing holdout, not a fresh prospective test.

## Development checks (optional; not another production study)

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
OMP_NUM_THREADS=1 python3 -m unittest discover -s fine_tuning/verification/scripts -p 'test_loss_ablation.py' -v
bash -n fine_tuning/verification/scripts/run_loss_ablation.sbatch
SMOKE_ROOT=$(mktemp -d /tmp/hrrt-ablation-smoke.XXXXXX)
python3 fine_tuning/verification/scripts/run_loss_ablation.py init --study-root "$SMOKE_ROOT" --smoke --device cpu
OMP_NUM_THREADS=4 python3 fine_tuning/verification/scripts/run_loss_ablation.py local --study-root "$SMOKE_ROOT"
xdg-open "$SMOKE_ROOT/dashboard/index.html"
```

Smoke mode uses real model code but only eight rows, one epoch, two tiny-set
updates, and no full QP evaluation. Every page is marked **SMOKE TEST**; its
scores and architecture choice are not evidence for scientific conclusions.
It writes several GB of temporary model artifacts. The unit tests separately
check exact CPU resume equivalence, legacy checkpoint shape compatibility,
fixed endpoints, sample-weighted metrics, and selection independent of test.

---

# Legacy single-run verification on NCSA DeltaAI

This directory runs the component-verification sequence before and after the
production waypoint-transformer trainer. It reuses the production model,
dataset parser, Euclidean imitation loss, YAML configuration, and checkpoint
format; it does not maintain a second transformer implementation.

## Sequence executed by the batch job

```text
Teacher-only train/validation/test CSV audit
        ↓
Check required columns, coordinates, and scene-level split isolation
        ↓
Tiny-set overfit test on 8 examples
        ↓
Check output shape and gradient flow through every model block
        ↓
Check that changing text changes the prediction
        ↓
Check that changing scene geometry changes the prediction
        ↓
Run the production transformer trainer
        ↓
Select the lowest finite-radius checkpoint;
otherwise select the lowest validation-loss checkpoint
        ↓
Measure validation and locked-test Euclidean waypoint error
        ↓
Repeat gradient and text/scene sensitivity checks
        ↓
Generate training curves, diagnostic charts, route comparisons, and HTML dashboard
```

The preflight job exits before full training if the CSV split is malformed, a
scene appears in more than one split, the output shape is wrong, any major model
block receives zero gradient, text/scene changes have no measurable effect, or
the tiny-set loss fails to decrease by the configured amount.

The postflight report records mean, median, 95th-percentile, and maximum
waypoint errors. The QP-derived safety radius remains the production trainer's
separate stopping certificate and does not update model weights.

## Local compatibility check

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
python3 -m py_compile fine_tuning/verification/scripts/verify_transformer_components.py
python3 -m py_compile fine_tuning/verification/scripts/plot_verification_results.py
bash -n fine_tuning/verification/scripts/run_transformer_verification.sbatch
python3 fine_tuning/verification/scripts/verify_transformer_components.py --help
```

## Submit the complete sequence on NCSA DeltaAI

Paste this block verbatim at the DeltaAI login prompt:

```bash
set -euo pipefail
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
git pull --ff-only
mkdir -p logs
test -s src/llm_vision_planner/fine_tuning/configs/hrrt_waypoint_transformer.yaml
test -s src/llm_vision_planner/fine_tuning/scripts/train_hrrt_waypoint_transformer.py
test -s src/llm_vision_planner/fine_tuning/verification/scripts/verify_transformer_components.py
test -s src/llm_vision_planner/fine_tuning/verification/scripts/run_transformer_verification.sbatch
test -s src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv
test -s src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv
test -s src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_test.csv
JOB_ID=$(sbatch --parsable --export=ALL,TRAIN_CSV=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv,VALIDATION_CSV=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv,TEST_CSV=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_test.csv,SAFETY_RADIUS_THRESHOLD_M=0.40,MAX_EPOCHS=100,TINY_SAMPLES=8,TINY_UPDATES=250,TINY_LR_MULTIPLIER=5.0,REQUIRED_OVERFIT_REDUCTION=0.50 src/llm_vision_planner/fine_tuning/verification/scripts/run_transformer_verification.sbatch)
printf '%s\n' "$JOB_ID" | tee transformer_verification_job_id.txt
```

This job uses the normal `ghx4` GPU partition rather than the interactive
partition.

## Monitor the job

```bash
set -euo pipefail
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
JOB_ID=$(cat transformer_verification_job_id.txt)
squeue -j "$JOB_ID"
tail -f "logs/hrrt-transformer-verify-${JOB_ID}.out"
```

After completion:

```bash
set -euo pipefail
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
JOB_ID=$(cat transformer_verification_job_id.txt)
RUN_ROOT="src/llm_vision_planner/fine_tuning/outputs/verification-${JOB_ID}"
python3 -m json.tool "$RUN_ROOT/reports/preflight.json"
python3 -m json.tool "$RUN_ROOT/reports/postflight.json"
python3 -m json.tool "$RUN_ROOT/model/training_summary.json"
ls -lh "$RUN_ROOT/plots/"
sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```

## Visualise the results

The batch job automatically writes:

```text
plots/training_curves.png
plots/component_diagnostics.png
plots/route_examples.png
plots/index.html
```

To regenerate the plots on DeltaAI after training:

```bash
set -euo pipefail
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
source /projects/bhkj/pgupta12/hrrt_waypoint_transformer_env/bin/activate
JOB_ID=$(cat transformer_verification_job_id.txt)
RUN_ROOT="$PWD/src/llm_vision_planner/fine_tuning/outputs/verification-${JOB_ID}"
python3 src/llm_vision_planner/fine_tuning/verification/scripts/plot_verification_results.py --run-root "$RUN_ROOT"
```

For a Transformer 2.0 model trained separately from this verification batch
job, pass its completed output directory explicitly:

```bash
python3 fine_tuning/verification/scripts/plot_verification_results.py \
  --run-root fine_tuning/outputs/verification-transformer-2-0-local \
  --model-output fine_tuning/outputs/hrrt_waypoint_transformer_2_0
```

The run root must contain `reports/preflight.json` and
`reports/postflight.json`. The complete laptop transfer, report-generation,
and local web-server commands are in the main
[`README.md`](../README.md#transfer-transformer-20-and-build-the-local-verification-website).

Copy the visual report to this laptop:

```bash
set -euo pipefail
LOCAL_REPORT=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/verification-report
mkdir -p "$LOCAL_REPORT"
JOB_ID=$(ssh pgupta12@dtai-login.delta.ncsa.illinois.edu 'cat /projects/bhkj/pgupta12/starling_multiple_trajectory_idea/transformer_verification_job_id.txt')
rsync -avP "pgupta12@dtai-login.delta.ncsa.illinois.edu:/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/verification-${JOB_ID}/plots/" "$LOCAL_REPORT/"
xdg-open "$LOCAL_REPORT/index.html"
```

The route plot overlays six locked-test examples: green is the resampled HRRT
target, orange is the transformer prediction, and grey boxes are obstacles.

The complete model and reports are packaged as:

```text
src/llm_vision_planner/fine_tuning/outputs/verification-JOB_ID.tar.gz
```

## Configuration overrides

| Variable | Default | Meaning |
| --- | ---: | --- |
| `MAX_EPOCHS` | `100` | Maximum full-training epochs |
| `SAFETY_RADIUS_THRESHOLD_M` | `0.40` | Production stopping certificate |
| `TINY_SAMPLES` | `8` | Examples repeatedly used in the overfit test |
| `TINY_UPDATES` | `250` | Optimizer updates in the overfit test |
| `TINY_LR_MULTIPLIER` | `5.0` | Diagnostic-only learning-rate multiplier |
| `REQUIRED_OVERFIT_REDUCTION` | `0.50` | Required relative tiny-set loss reduction |
| `RUN_NAME` | `verification-JOB_ID` | Output-directory name |

The learning-rate multiplier applies only to the tiny-set diagnostic. Full
training continues to use the rates in
`fine_tuning/configs/hrrt_waypoint_transformer.yaml`.
