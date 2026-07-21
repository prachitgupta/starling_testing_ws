# Shared-Clock Minimum-Control QP Position Pipeline

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

## Build

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_vision_planner
source install/setup.bash
```

## Reproduce the 2,000-row calibration files

Run the commands in `REPRODUCE_XU_DATASETS.md`. Prediction-pair generation is
separate from QP/conformal computation, so an existing source-pair CSV can be
reused without contacting vLLM.

## Generate 5,000 samples in tmux

The prediction stage requires the configured vLLM server. Use the system Python
after sourcing ROS Humble so the offline refinement and verifier can import
`rclpy`.

Start a persistent terminal:

```bash
tmux new-session -s qp5000
```

Inside tmux, run the three stages sequentially:

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
source /opt/ros/humble/setup.bash

/usr/bin/python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 5000 \
  --output fine_tuning/datasets/conformal_rrt_prediction_pairs_5000.csv \
  2>&1 | tee /tmp/qp5000_predictions.log

/usr/bin/python3 fine_tuning/scripts/generate_qp_calibration_dataset.py \
  --input fine_tuning/datasets/conformal_rrt_prediction_pairs_5000.csv \
  --output fine_tuning/datasets/calibration_min_control_qp_shared_clock_5000.csv \
  --samples 5000 \
  --dt 0.1 \
  --delta-u 0.05 \
  --delta-x 0.05 \
  2>&1 | tee /tmp/qp5000_trajectories.log

/usr/bin/python3 fine_tuning/scripts/generate_position_score_calibration.py \
  --input fine_tuning/datasets/calibration_min_control_qp_shared_clock_5000.csv \
  --output fine_tuning/datasets/calibration_min_control_qp_position_score_5000.csv \
  --samples 5000 \
  --delta-p 0.10 \
  --delta-w 0.10 \
  --control-dt 0.05 \
  2>&1 | tee /tmp/qp5000_position_scores.log
```

Detach with `Ctrl-b d` and reattach later with:

```bash
tmux attach-session -t qp5000
```

For 5,000 calibration rows, the 90% position-score rank is 4,501. The first
stage is the only one that contacts vLLM; the QP and scoring stages can be rerun
from their input CSVs.

## Offline certificate check

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/calibration_min_control_qp_position_score_2000.csv \
  --calibration-samples 2000 \
  --sample-id 998 \
  --report-json /tmp/qp_position_verify.json \
  --output-png /tmp/qp_position_verify.png
```

The 90% finite-sample certificate uses rank 1801 and the direct closed-loop
cross-track score. Its radius is the position quantile itself:
`radius = q_p = 0.241187 m`. The equal-time position error is retained only as
a diagnostic. The verifier plot also compares the current-speed 4D state radius
`1.481553` and its exact 2D projection `0.602299 m`. All scores use the calibrated
planar model at a 20 Hz control rate.

## PX4 SITL and unified launch

Start PX4 SITL and the Micro XRCE-DDS agent, then launch:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py
```

Select the live PX4 contraction plot instead of the standard planner plot with:

```bash
ros2 launch llm_vision_planner full_plot.launch.py visualizer:=contraction
```

The contraction view latches the passed verified plan, reconstructs the same QP
reference used by the executor, and draws the calibrated cross-track quantile and
projected 2D safety radius. Its moving vehicle and trail come directly from
`/fmu/out/vehicle_odometry`; no simulated state is propagated. Configure its
quantiles, radii, output path, and update period in the `verify_contraction`
section of `config/llm_vision_planner.yaml`.

`control_law_executer.py` is the sole offboard owner. It primes PX4, arms, takes off, holds until the prompt/vLLM/refinement/verifier chain returns a passed plan, generates the shared-clock minimum-control QP reference, tracks it, holds the goal, and requests PX4 auto-land. The launch does not use `mission_takeoff.py` or either trajectory follower.

Monitor the pipeline with:

```bash
ros2 topic echo /llm_vision/mission_state
ros2 topic echo /llm_vision/plan_verified
ros2 topic echo /llm_vision/offboard_owner
ros2 topic echo /fmu/in/trajectory_setpoint
ros2 topic echo /fmu/out/vehicle_status
ros2 topic echo /fmu/out/vehicle_land_detected
```

Use PX4 SITL for the first end-to-end run. The position radius certifies the
sampled closed-loop calibrated planar model, not the full nonlinear PX4 vehicle
or unmodelled flight disturbances.
