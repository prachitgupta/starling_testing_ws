# Expert calibration data and verification plots

This directory retains the certified waypoint pipeline for the planar damped model

```text
x = [px, py, vx, vy]
xdot = [vx, vy, -1.1 vx + 1.1 ux, -1.1 vy + 1.1 uy]
u_cmd = uhat_d - K (x - xhat_d)
```

Here `u_cmd` is a horizontal velocity demand. It is not a physical acceleration.
The QP cruise speed is `0.30 m/s`; its controls are evaluated with exact zero-order
hold rather than linearly interpolated.

## File roles

- `scripts/conformal_rrt_dataset.py` is the data-generation CLI. By default it
  creates calibration source data; `--rrt-training` instead creates the original
  RRT instruction-tuning data without contacting vLLM. For calibration data it
  samples environments, calls RRT and vLLM, runs the existing refinement and
  verifier, and writes only the raw/verified waypoint pairs and metadata. It does
  not generate trajectories, scores, quantiles, or acceptance columns.
- `scripts/conformal_semantic_theta_dataset.py` performs the same prediction-pair
  stage with the Semantic Theta* expert and semantic hard-margin verification.
  `--semantic-theta-training` instead generates expert instruction-training data.
- `scripts/generate_qp_calibration_dataset.py` consumes those stored verified
  pairs without contacting vLLM. It generates shared-clock minimum-control QP
  trajectories and computes `s_u`, `s_x`, `q_u`, `q_x`, deltas, and acceptance.
- `scripts/generate_position_score_calibration.py` consumes the QP CSV and
  simulates the feedback-controlled damped model at 20 Hz. It adds the direct
  cross-track position score `s_p`, equal-time diagnostic, legacy comparison
  score `s_w`, and their conformal quantiles.
- `scripts/dconformal_contraction_verify.py` reads the final position-score CSV,
  evaluates an arbitrary sample, and plots the direct position tube, exact 2D
  projection, and legacy 4D state radius.

This separation makes stored vLLM predictions reusable: changing the QP,
score, confidence level, or plotting does not require new LLM predictions.
The three downstream tools accept `--expert rrt` (the backward-compatible default)
or `--expert semantic_theta`. This selects the expert CSV columns and offline plot
label; both methods use the same QP solver, shared clock, dynamics, feedback gain,
and score definitions. Semantic rows retain `semantic_theta_*` column names.

## Semantic Theta* expert

The RRT pipeline above remains unchanged. The parallel semantic expert uses:

- `scripts/semantic_theta.py` for deterministic any-angle Theta* search with
  per-label hard margins and soft traversal costs.
- `scripts/conformal_semantic_theta_dataset.py` for instruction datasets or
  verified Semantic-Theta*/LLM prediction pairs. The requested
  `conforml_semantic_theta_dataset.py` name is a compatibility entry point.
- `datasets/semantic_theta_expert_dataset.csv` as the validated 20,000-row
  expert dataset.

The workspace-root [`NCSA_COMMANDS.md`](../../../NCSA_COMMANDS.md) documents the
RRT and Semantic Theta* workflows in the same numbered template. Each stage
shows both commands: dataset generation, upload, PEFT or Unsloth submission,
monitoring, adapter packaging, and local download.

## Build

Complete the [workspace setup](../../../README.md#1-clone-build-and-source-locally) first.

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select px4_msgs llm_vision_planner
source install/setup.bash
```

## Calibration prerequisites

Run the following workflows from the package directory with system Python:

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_testing_ws/install/setup.bash
export VLLM_BASE_URL=http://172.22.224.93:8000/v1
curl --fail --silent --show-error "$VLLM_BASE_URL/models"
```

Set `VLLM_BASE_URL` to your running server. It must expose `rrt_planner` for the
RRT workflow or `semantic_theta_planner` for the Semantic Theta workflow, with
the corresponding trained adapter loaded. See the serving command in
[workspace README](../../../README.md#3-fine-tune-download-and-serve-the-selected-adapter) and the
adapter training/download commands in
[`NCSA_COMMANDS.md`](../../../NCSA_COMMANDS.md#rrt-and-semantic-theta-lora-workflows).
Prediction generation needs `rclpy`, `openai`, `instructor`, and `pydantic` in
this Python environment; offline QP/scoring/plotting uses NumPy, SciPy,
Matplotlib, and Pydantic.

Only stage 1 below contacts vLLM. Run stages 1–4 in order and wait for each to
finish successfully. Both workflows generate 2,001 verified prediction pairs,
then use the first 2,000 for QP trajectories and calibration scores. The final
plot checks row 998 of that calibration set; it is a diagnostic, not an
independent test of coverage. Use fresh calibration environments after training;
the instruction-training CSV does not contain expert/LLM prediction pairs.

Commands overwrite their named output files. The three Semantic Theta calibration
CSVs initially contain headers only as path placeholders; run its stages 1–3
before plotting. They contain no calibration observations until generated.

## RRT: calibration dataset, scores, and dConformal plot

### 1. Generate verified RRT/LLM prediction pairs

```bash
/usr/bin/python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 2001 \
  --seed 20260904 \
  --llama-model-name rrt_planner \
  --vllm-base-url "$VLLM_BASE_URL" \
  --temperature 0.0 \
  --output fine_tuning/datasets/conformal_rrt_calibration_dataset_2001.csv
```

### 2. Generate shared-clock minimum-control QP trajectories

```bash
/usr/bin/python3 fine_tuning/scripts/generate_qp_calibration_dataset.py \
  --expert rrt \
  --input fine_tuning/datasets/conformal_rrt_calibration_dataset_2001.csv \
  --output fine_tuning/datasets/calibration_min_control_qp_shared_clock_with_limits_2000.csv \
  --samples 2000 \
  --dt 0.1 \
  --max-velocity-mps 0.5 \
  --max-acceleration-mps2 0.5 \
  --delta-u 0.05 \
  --delta-x 0.05
```

### 3. Generate position-score calibration

```bash
/usr/bin/python3 fine_tuning/scripts/generate_position_score_calibration.py \
  --expert rrt \
  --input fine_tuning/datasets/calibration_min_control_qp_shared_clock_with_limits_2000.csv \
  --output fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_2000.csv \
  --samples 2000 \
  --delta-p 0.10 \
  --delta-w 0.10 \
  --control-dt 0.05
```

### 4. Verify a score and generate the RRT plot

```bash
/usr/bin/python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --expert rrt \
  --calibration-csv fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_2000.csv \
  --calibration-samples 2000 \
  --sample-id 998 \
  --delta-p 0.10 \
  --delta-w 0.10 \
  --dt 0.05 \
  --trajectory-dt 0.1 \
  --max-velocity-mps 0.5 \
  --max-acceleration-mps2 0.5 \
  --output-png fine_tuning/plots/contraction/qp_with_limits.png \
  --report-json fine_tuning/results/contraction/qp_with_limits.json
```

## Semantic Theta*: calibration dataset, scores, and dConformal plot

The expert uses deterministic any-angle search with label-specific hard margins
and soft traversal costs. Stage 1 verifies both paths against its hard margins
and preserves `semantic_policy` and the `semantic_theta_*` waypoint fields.
Stages 2–4 use exactly the same QP/shared-clock and scoring parameters as RRT.
The QP enforces waypoint interpolation and motion limits. Obstacle and semantic
checks occur on the verified waypoint paths; the QP does not optimize semantic
soft cost or recheck collision margins along its smoothed trajectory.

### 1. Generate verified Semantic Theta*/LLM prediction pairs

```bash
/usr/bin/python3 fine_tuning/scripts/conformal_semantic_theta_dataset.py \
  --samples 2001 \
  --seed 20260904 \
  --llama-model-name semantic_theta_planner \
  --vllm-base-url "$VLLM_BASE_URL" \
  --temperature 0.0 \
  --output fine_tuning/datasets/conformal_semantic_theta_calibration_dataset_2001.csv
```

Omit `--semantic-theta-training` for calibration; that flag generates the expert
fine-tuning targets instead. `conforml_semantic_theta_dataset.py` is an equivalent
compatibility entry point. Keep the semantic policy, clearance, grid resolution,
and semantic cost scale consistent with the trained model when overriding them.

### 2. Generate shared-clock minimum-control QP trajectories

```bash
/usr/bin/python3 fine_tuning/scripts/generate_qp_calibration_dataset.py \
  --expert semantic_theta \
  --input fine_tuning/datasets/conformal_semantic_theta_calibration_dataset_2001.csv \
  --output fine_tuning/datasets/calibration_semantic_theta_qp_shared_clock_with_limits_2000.csv \
  --samples 2000 \
  --dt 0.1 \
  --max-velocity-mps 0.5 \
  --max-acceleration-mps2 0.5 \
  --delta-u 0.05 \
  --delta-x 0.05
```

### 3. Generate position-score calibration

```bash
/usr/bin/python3 fine_tuning/scripts/generate_position_score_calibration.py \
  --expert semantic_theta \
  --input fine_tuning/datasets/calibration_semantic_theta_qp_shared_clock_with_limits_2000.csv \
  --output fine_tuning/datasets/calibration_semantic_theta_qp_position_score_with_limits_2000.csv \
  --samples 2000 \
  --delta-p 0.10 \
  --delta-w 0.10 \
  --control-dt 0.05
```

### 4. Verify a score and generate the Semantic Theta* plot

```bash
/usr/bin/python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --expert semantic_theta \
  --calibration-csv fine_tuning/datasets/calibration_semantic_theta_qp_position_score_with_limits_2000.csv \
  --calibration-samples 2000 \
  --sample-id 998 \
  --delta-p 0.10 \
  --delta-w 0.10 \
  --dt 0.05 \
  --trajectory-dt 0.1 \
  --max-velocity-mps 0.5 \
  --max-acceleration-mps2 0.5 \
  --output-png fine_tuning/plots/contraction/semantic_theta_qp_with_limits.png \
  --report-json fine_tuning/results/contraction/semantic_theta_qp_with_limits.json
```

## Read the scores and plots

Stage 2 prints `s_u`, `s_x`, `q_u`, and `q_x`; stage 3 prints each row's direct
cross-track score `s_p`, equal-time diagnostic `s_position_time`, and legacy
disturbance score `s_w`, followed by `q_p` and `q_w`. Stage 4 prints a JSON report
with the expert name, `s_p`, `q_p`, `score_accepted` (`s_p <= q_p`), and the radii.
The report is also saved to the corresponding `--report-json` path.

Open the corresponding PNG under `fine_tuning/plots/contraction/`: it shows the
selected expert reference, LLM QP reference, closed-loop state, direct position
tube, projected 2D tube, and 4D-certificate legend. The direct radius is `q_p` in
meters. The two experts are calibrated separately; do not reuse RRT quantiles
for Semantic Theta. With 2,000 rows and `delta_p=0.10`, the conformal rank is 1,801.
The retained RRT CSV has `q_p=0.235845 m`; newly generated data can give a different
value. Both generators retain only verified pairs, so these scores describe
that accepted-pair population.

The verifier commands above are offline: they do not contact the LLM, start PX4,
or subscribe to odometry. Semantic Theta expert selection currently applies to
offline verification. Hardware launch and Vicon vision-error recording are isolated on
[`reporoduce_hardeware`](https://github.com/prachitgupta/starling_testing_ws/tree/reporoduce_hardeware).

For longer runs, start `tmux new-session -s qp-calibration` before the prerequisites
and chosen workflow, detach with `Ctrl-b d`, and return with
`tmux attach-session -t qp-calibration`. For 5,000 calibration rows, generate at
least 5,000 source pairs, set both downstream `--samples` and the verifier's
`--calibration-samples` to 5000, and use matching input/output paths throughout.


## Plot a given dataset and row

Use the existing `dconformal_contraction_verify.py`; no new plotting script is
needed. `--sample-id` is the **zero-based CSV row index**, not a search for a
value in the CSV's `sample_id` column. Set `--calibration-samples` to the number
of leading rows used to compute the quantile; it must not exceed the available
row count. The selected row may be outside that prefix for a held-out check.

For example, plot row 998 of the retained RRT score dataset:

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
EXPERT=rrt
SCORED_CSV="$PWD/fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_2000.csv"
CALIBRATION_SAMPLES=2000
SAMPLE_INDEX=998
MPLBACKEND=Agg /usr/bin/python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --expert "$EXPERT" --calibration-csv "$SCORED_CSV" \
  --calibration-samples "$CALIBRATION_SAMPLES" --sample-id "$SAMPLE_INDEX" \
  --delta-p 0.10 --delta-w 0.10 --dt 0.05 --trajectory-dt 0.1 \
  --max-velocity-mps 0.5 --max-acceleration-mps2 0.5 \
  --output-png fine_tuning/plots/contraction/qp_with_limits.png \
  --report-json fine_tuning/results/contraction/qp_with_limits.json
```

For your dataset, change `SCORED_CSV`, `EXPERT`, the row index, and output paths;
keep the trajectory/score parameters consistent with its generation settings.
For Semantic Theta, set `EXPERT=semantic_theta` and use its separately generated
position-score CSV. A usable input includes `start`, `goal`, `workspace`,
`obstacles`, matching `rrt_*` or `semantic_theta_*` waypoint/trajectory columns,
`llm_*` fields, `s_p`, and `s_w`. If you only have verified prediction pairs,
run stages 2 and 3 first. Instruction-training CSVs and hardware vision-error
CSVs are different schemas and cannot be passed directly to this plotter.

Outputs are a PNG and a JSON report (`expert`, `s_p`, `q_p`, `score_accepted`,
and radii). `score_accepted: false` is a completed diagnostic, not a script
failure. The retained RRT PNG/JSON are examples. Semantic output directories
exist, but its plots require generated observations; header-only CSVs are
placeholders. A plot of a calibration row does not independently verify
coverage on unseen environments.
