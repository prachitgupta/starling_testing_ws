# Shared-Clock Minimum-Control QP Pipeline

This directory retains the certified waypoint pipeline for the planar damped model

```text
x = [px, py, vx, vy]
xdot = [vx, vy, -1.1 vx + 1.1 ux, -1.1 vy + 1.1 uy]
u_cmd = uhat_d - K (x - xhat_d)
```

Here `u_cmd` is a horizontal velocity demand. It is not a physical acceleration.

## Build

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_vision_planner
source install/setup.bash
```

## Reproduce the 2,000-row calibration files

Run the commands in `REPRODUCE_XU_DATASETS.md`. They reuse the stored 2,001 verified RRT/LLM pairs and do not contact vLLM.

## Offline certificate check

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/calibration_min_control_qp_single_score_2000.csv \
  --calibration-samples 2000 \
  --sample-id 998 \
  --report-json /tmp/qp_single_score_verify.json \
  --output-png /tmp/qp_single_score_verify.png
```

The 90% finite-sample certificate uses rank 1801, `q_w = 1.641463`, and planar radius `rho_infinity = 1.913080 m`.

## PX4 SITL and unified launch

Start PX4 SITL and the Micro XRCE-DDS agent, then launch:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py
```

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

Use PX4 SITL for the first end-to-end run. The 1.913080 m value certifies the calibrated planar model/reference discrepancy, not the full nonlinear PX4 vehicle.
