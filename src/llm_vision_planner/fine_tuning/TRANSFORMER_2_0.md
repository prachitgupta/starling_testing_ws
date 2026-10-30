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
queries and do not contribute error to the Euclidean part of the loss. For
obstacle `j`, let `h_j` be the selected teacher route's HRRT homotopy sign,
`u` the start-to-goal unit vector, and `n = (-u_y, u_x)` its left normal. The
teacher route fixes soft weights around that obstacle:

\[
w^*_{kj}=\operatorname{softmax}_k\left(
-\frac{|(p^*_k-c_j)^\top u|}{\tau}\right), \qquad
d_j(\hat p)=(-h_j)\sum_k w^*_{kj}(\hat p_k-c_j)^\top n .
\]

`d_j` is positive only when the prediction passes on the same side as the
teacher. The wrong-corridor hinge and total objective are

\[
L_{corr}=\frac{1}{N_o}\sum_j\max(0,\delta-d_j), \qquad
L=L_{Euc}+\lambda_{corr}L_{corr},
\]

\[
L_{Euc}=\frac{1}{6B}\sum_{b=1}^{B}\sum_{k=1}^{6}
\lVert\hat p_{bk}-p^*_{bk}\rVert_2 .
\]

The default values are a strong `lambda_corr = 5.0`, side margin
`delta = 0.20 m`, and soft matching temperature `tau = 0.35 m`. A prediction
on the wrong side has `d_j < 0`, so its penalty is larger than `0.20 m` and is
multiplied by five. The weights come from the fixed teacher route, preventing
the model from evading the penalty by shifting which waypoint is matched to
the obstacle. The training log reports Euclidean loss, corridor hinge loss,
the combined objective, and wrong-corridor rate separately.

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
  --export=ALL,TRAIN_CSV=$PWD/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_train.csv,VALIDATION_CSV=$PWD/src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_validation.csv,MAX_EPOCHS=100,CORRIDOR_LOSS_WEIGHT=5.0,CORRIDOR_MARGIN_M=0.20,SAFETY_RADIUS_THRESHOLD_M=0.40 \
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
