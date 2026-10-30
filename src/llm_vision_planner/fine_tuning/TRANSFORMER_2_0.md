# Transformer 2.0

Transformer 2.0 is a separate waypoint-model training path. It consumes the
same finalized HRRT CSV files and produces the same eight-point route interface
used by the verifier, refinement, QP, and calibration gateway.

## Model flow

```text
preference text -> MiniLM -> 320-dimensional text tokens

start / goal / workspace / obstacles
    -> 13-value numeric scene tokens
       4 token-type values
       3 obstacle-label values: person, chair, stop_sign
       6 normalized geometry values
    -> 13 -> 320 input projection
    -> two residual 320 -> 640 -> 320 scene MLP blocks
    -> learned scene-summary token
    -> two scene-only Transformer encoder layers

text tokens + encoded scene tokens
    -> four fusion Transformer layers

six learned route queries
    + explicit progress values [1/7, 2/7, ..., 6/7]
    + encoded scene summary
    -> three decoder layers
    -> six predicted x-y points

final route = exact start + six predictions + exact goal
```

Start and goal are copied from the numeric scene input. They are not learned
queries and do not contribute error to the backpropagated loss:

```text
loss = mean(norm(predicted[:, 1:-1] - target[:, 1:-1], 2))
```

Scene dependence is strengthened in four independent ways: object identities
are numeric scene features rather than text-only information; the scene has its
own MLP and self-attention encoder before fusion; fusion depth increases from
three to four layers; and the scene summary is injected directly into every
interior waypoint query. The existing component sensitivity check remains the
way to measure scene response versus text response after training.

## NCSA DeltaAI submission

```bash
set -euo pipefail
cd /projects/bhkj/pgupta12/starling_multiple_trajectory_idea
git checkout starling_multiple_trajectory_idea
git pull --ff-only
mkdir -p logs

JOB_ID=$(sbatch --parsable \
  --export=ALL,TRAIN_CSV=$PWD/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv,VALIDATION_CSV=$PWD/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv,MAX_EPOCHS=100,SAFETY_RADIUS_THRESHOLD_M=0.40 \
  src/llm_vision_planner/fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.sbatch)

printf '%s\n' "$JOB_ID" | tee transformer_2_0_job_id.txt
```

Monitor it with:

```bash
JOB_ID=$(cat transformer_2_0_job_id.txt)
squeue -j "$JOB_ID"
tail -f "logs/hrrt-transformer-2-0-${JOB_ID}.out"
```

The checkpoint configuration records `architecture_version:
transformer_2_0`, allowing the calibration gateway to load the correct model
implementation automatically.
