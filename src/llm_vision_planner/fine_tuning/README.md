# Double-Integrator Controller SITL Use

This folder contains the offline scripts for calibration, minimum-snap trajectory generation, LQR gain design, and double-integrator controller verification. The ROS planner/refinement/verifier nodes used in SITL live in `../scripts`.

## 1. Build and Source

```bash
cd ~/Desktop/starling_testing_ws
colcon build --packages-select px4_msgs voxl_msgs starling_testing llm_vision_planner
source install/setup.bash
```

## 2. Generate Calibration Data

This queries the remote Llama planner, runs both raw RRT and raw LLM waypoints through `scripts/refinment.py` and `scripts/verifier.py`, skips failed datapoints, then sends verified waypoints to `fine_tuning/scripts/min_snap.py`.

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 21 \
  --calibration-fraction 0.96 \
  --output fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --vllm-base-url http://172.22.224.93:8000/v1 \
  --llama-model-name rrt_planner
```

For `--samples 21 --calibration-fraction 0.96`, the first 20 rows compute `q_u` and `q_x`; row index `20` is the held-out verification sample.

## 3. Verify Controller Offline

```bash
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --calibration-samples 20 \
  --sample-id 20 \
  --output-png fine_tuning/plots/dconformal_contraction_verify_21.png \
  --report-json fine_tuning/plots/dconformal_contraction_verify_21.json
```

This uses:

- `fine_tuning/scripts/lqr.py` for the damped double-integrator CARE gain.
- `fine_tuning/scripts/dconformal_contraction_verify.py` for `u = -K(x - xhat_d) + uhat_d`.
- `fine_tuning/scripts/min_snap.py` for verified waypoint to `x(t), u(t)` conversion.

## 4. Run Planner Nodes in SITL

Start the planner stack:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py mode:=normal llm_provider:=llama show_rrt:=true
```

If using semantic obstacles instead of normal point-cloud obstacles:

```bash
ros2 launch llm_vision_planner full_plot.launch.py mode:=semantic llm_provider:=llama show_rrt:=true
```

The launch file starts:

- `scripts/perception.py` or `scripts/perception_detection.py`
- `scripts/prompt_generator.py`
- `scripts/llm_planner.py`
- `scripts/refinment.py`
- `scripts/verifier.py`
- `scripts/visualize.py`

Start the follower used by this codebase:

```bash
ros2 run llm_vision_planner trajectory_follower_continuous.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

## 5. SITL Topics Needed

The planner expects:

- pose: `/fmu/out/vehicle_odometry`
- mission state: `/llm_vision/mission_state`
- obstacles: `/llm_vision/obstacles` for `mode:=normal`
- semantic obstacles: `/llm_vision/semantic_obstacles` for `mode:=semantic`

For a software-only check, publish a hover mission state:

```bash
ros2 topic pub /llm_vision/mission_state std_msgs/msg/String \
  "{data: '{\"state\":\"HOLDING_FOR_PLAN\",\"position\":{\"x\":0.0,\"y\":0.0,\"z\":-0.25},\"heading_deg\":0.0}'}" -r 2
```

Monitor:

```bash
ros2 topic echo /llm_vision/prompt
ros2 topic echo /llm_vision/plan_raw
ros2 topic echo /llm_vision/plan_refined
ros2 topic echo /llm_vision/plan_verified
```

## 6. Run Live LLM Controller Script

Use this when you want the live flow without RRT online. It queries Llama, refines and verifies the LLM waypoints, re-prompts on verifier failure, runs min-snap, solves CARE, applies the double-integrator control law, and writes a JSON report plus plot.

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/live_dconformal_controller.py \
  --start '{"x":0.0,"y":0.0,"z":-0.25}' \
  --goal '{"x":2.5,"y":0.0,"z":-0.25}' \
  --obstacles '[]' \
  --calibration-csv fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --calibration-samples 20 \
  --llm-attempts 3 \
  --output-json fine_tuning/plots/live_dconformal_controller.json \
  --output-png fine_tuning/plots/live_dconformal_controller.png
```

With obstacles:

```bash
python3 fine_tuning/scripts/live_dconformal_controller.py \
  --start '{"x":0.0,"y":0.0,"z":-0.25}' \
  --goal '{"x":2.5,"y":0.0,"z":-0.25}' \
  --obstacles '[{"id":1,"label":"box_1","shape":"box","min_corner":[1.0,-0.3,-0.75],"max_corner":[1.5,0.3,0.25]}]' \
  --calibration-csv fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --calibration-samples 20
```

## 7. Important Current Limitation

`dconformal_contraction_verify.py` is an offline controller verification script. The SITL follower currently tracks verified waypoints from `/llm_vision/plan_verified`; it does not publish the double-integrator control law as a ROS controller node yet.
