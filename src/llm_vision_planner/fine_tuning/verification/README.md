# HRRT waypoint-transformer verification on NCSA DeltaAI

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
