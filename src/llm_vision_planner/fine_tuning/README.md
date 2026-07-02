# Double-Integrator Conformal Controller

This folder contains the calibration, minimum-snap, LQR, live design, plotting, and verification scripts for the damped double-integrator controller.

State and input:

```text
x = [px, py, vx, vy]
u = [ax, ay]
```

Control law:

```text
u = -K(x - xhat_d) + uhat_d
```

## 1. Build

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select px4_msgs voxl_msgs starling_testing llm_vision_planner
source install/setup.bash
```

## 2. Generate Calibration Data

This queries the remote Llama planner, generates RRT labels, refines and verifies both paths, converts both verified paths through `min_snap.py`, computes scores, and checkpoints the CSV after every accepted datapoint.

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 21 \
  --calibration-fraction 0.96 \
  --output fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --vllm-base-url http://172.22.224.93:8000/v1 \
  --llama-model-name rrt_planner
```

For 21 samples, use the first 20 for calibration and sample id 20 for a held-out check.

## 3. Offline Verification

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --calibration-samples 20 \
  --sample-id 20 \
  --output-png fine_tuning/plots/dconformal_contraction_verify_21.png \
  --report-json fine_tuning/plots/dconformal_contraction_verify_21.json
```

The script prints `q_u` and `q_x` on the terminal and writes the same values to the JSON report.

## 4. Start PX4 SITL

Terminal 1:

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

Terminal 2:

```bash
MicroXRCEAgent udp4 -p 8888
```

Confirm ROS can see PX4 odometry:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 topic echo /fmu/out/vehicle_odometry
```

## 5. Take Off and Hold

Terminal 3:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 run llm_vision_planner mission_takeoff.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

Wait until the log shows the vehicle is holding pose. Keep this node running.

## 6. Live Conformal Design and Plot

Terminal 4:

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
source ../../install/setup.bash
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --live \
  --start '{"x":0.0,"y":0.0,"z":-0.25}' \
  --goal '{"x":2.5,"y":0.0,"z":-0.25}' \
  --obstacles '[]' \
  --calibration-csv fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --calibration-samples 20 \
  --llm-attempts 3 \
  --show-rrt \
  --report-json fine_tuning/plots/dconformal_live_design.json \
  --output-png fine_tuning/plots/dconformal_live_design.png
```

This node:

- prints `q_u` and `q_x`;
- queries Llama;
- refines and verifies LLM waypoints;
- re-prompts Llama if verification fails;
- converts the verified LLM path through `min_snap.py`;
- solves CARE for `K`, `P`, and `alpha`;
- publishes `/llm_vision/dconformal_control_law`;
- plots raw waypoint markers, verified paths, commanded path, and actual PX4 pose.

With obstacles, replace `--obstacles '[]'`:

```bash
--obstacles '[{"id":1,"label":"box_1","shape":"box","min_corner":[1.0,-0.3,-0.75],"max_corner":[1.5,0.3,0.25]}]'
```

## 7. Run the Control-Law Executor

Terminal 5:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 run llm_vision_planner control_law_executer.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

The executor waits for the design message and PX4 odometry. Once takeoff has completed, it computes:

```text
u = -K(x - xhat_d) + uhat_d
```

Then it integrates the commanded acceleration internally and publishes only:

```text
/fmu/in/offboard_control_mode
/fmu/in/trajectory_setpoint
```

It uses `/fmu/in/vehicle_command` only for the offboard-mode handoff.

## 8. Monitor

```bash
ros2 topic echo /llm_vision/dconformal_control_law
ros2 topic echo /llm_vision/offboard_owner
ros2 topic echo /fmu/in/trajectory_setpoint
ros2 topic echo /fmu/out/vehicle_odometry
```

Plot and report files:

```text
src/llm_vision_planner/fine_tuning/plots/dconformal_live_design.png
src/llm_vision_planner/fine_tuning/plots/dconformal_live_design.json
```

## 9. Run in tmux

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
tmux new -s conformal_data
python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 21 \
  --calibration-fraction 0.96 \
  --output fine_tuning/datasets/conformal_rrt_calibration_21.csv \
  --vllm-base-url http://172.22.224.93:8000/v1 \
  --llama-model-name rrt_planner
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t conformal_data
```
