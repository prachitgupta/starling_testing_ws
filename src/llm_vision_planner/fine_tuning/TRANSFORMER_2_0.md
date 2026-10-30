# Transformer 2.0

Transformer 2.0 is a separate waypoint-model training path. It consumes the
same finalized HRRT CSV files and produces the same eight-point route interface
used by the verifier, refinement, QP, and calibration gateway.

## Model flow

```text
preference text -> MiniLM -> 320-dimensional text tokens

start / goal / obstacles
    -> 12-value numeric scene tokens
       3 token-type values
       3 obstacle-label values: person, chair, stop_sign
       6 normalized geometry values
    -> 12 -> 320 input projection
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
the fixed dataset workspace, the bounds are used to normalize and denormalize
coordinates but are no longer emitted as a scene token. For
obstacle `j`, let `h_j` be the selected teacher route's HRRT homotopy sign,
`u` the start-to-goal unit vector, and `n = (-u_y, u_x)` its left normal. The
teacher route fixes soft weights around that obstacle:

\[
w^*_{kj}=\operatorname{softmax}_k\left(
-\frac{|(p^*_k-c_j)^\top u|}{\tau}\right), \qquad
d_j(\hat p)=(-h_j)\sum_k w^*_{kj}(\hat p_k-c_j)^\top n .
\]

`d_j` is positive only when the prediction passes on the same side as the
teacher. The wrong-corridor hinge is

\[
L_{corr}=\frac{1}{N_o}\sum_j\max(0,\delta-d_j).
\]

\[
L_{Euc}=\frac{1}{6B}\sum_{b=1}^{B}\sum_{k=1}^{6}
\lVert\hat p_{bk}-p^*_{bk}\rVert_2 .
\]

The selected route also supplies clearance bin `b_j` for every obstacle:
`0=close`, `1=medium`, and `2=wide`. Let `c_hat_j` be the minimum sampled
distance from the predicted polyline to obstacle box `j`, and let
`[l_b, u_b]` be that object class's HRRT bin interval. Then

\[
L_{bin}=\frac{1}{N_o}\sum_j\left[
\max(0,l_{b_j}-\hat c_j)+\max(0,\hat c_j-u_{b_j})
\right],
\]

where the wide bin has `u_2 = infinity`. The complete objective is

\[
L_{tail}=\frac{1}{6B\,s}\sum_{b=1}^{B}\sum_{k=1}^{6}
\lVert\hat p_{bk}-p^*_{bk}\rVert_2^2,
\]

\[
L=L_{Euc}+\lambda_{corr}L_{corr}+\lambda_{bin}L_{bin}
+\lambda_{tail}L_{tail}.
\]

The scale is `s = 1.0 m`, so `L_tail` remains measured in metres. Squaring
makes a `2 m` miss contribute 400 times as much as a `0.1 m` miss before
averaging. The default values are `lambda_corr = 5.0`, `lambda_bin = 1.0`,
`lambda_tail = 1.0`, side margin `delta = 0.20 m`, and soft matching
temperature `tau = 0.35 m`. A prediction
on the wrong side has `d_j < 0`, so its penalty is larger than `0.20 m` and is
multiplied by five. The weights come from the fixed teacher route, preventing
the model from evading the penalty by shifting which waypoint is matched to
the obstacle. The training log reports Euclidean loss, corridor hinge loss,
clearance-bin hinge and accuracy, squared-tail loss, the combined objective,
and wrong-corridor rate separately. Bin thresholds exactly match HRRT: person `(0.80, 1.20) m`,
chair `(0.60, 0.80) m`, and stop sign `(0.70, 1.00) m`, with `0.40 m` as the
minimum clearance for the close-bin interval.

The original teacher path may change bins when it is resampled to the model's
eight-point target. A bin decision is therefore used only when that eight-point
teacher target still belongs to the stored bin. This prevents the bin loss from
conflicting with the Euclidean target; the logged decision count shows how many
bin labels were representable.

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
  --export=ALL,TRAIN_CSV=$PWD/src/llm_vision_planner/fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10/hrrt_sft_train.csv,VALIDATION_CSV=$PWD/src/llm_vision_planner/fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10/hrrt_sft_validation.csv,MAX_EPOCHS=100,CORRIDOR_LOSS_WEIGHT=5.0,CORRIDOR_MARGIN_M=0.20,CLEARANCE_BIN_LOSS_WEIGHT=1.0,SQUARED_TAIL_LOSS_WEIGHT=1.0,SAFETY_RADIUS_THRESHOLD_M=0.40 \
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

## Transfer and HTML verification report

After the NCSA job completes, follow the copy-and-verify commands in
[`README.md`](README.md#transfer-transformer-20-and-build-the-local-verification-website).
They transfer the complete `hrrt_waypoint_transformer_2_0` output to this
laptop, run the Transformer-aware scripts under `fine_tuning/verification`,
and serve the generated dashboard at `http://127.0.0.1:8092/index.html`.
