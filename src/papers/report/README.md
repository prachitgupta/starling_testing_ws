# Starling LLM Vision Planning Workspace

ROS 2 Humble workspace for a Python-only PX4 Offboard UAV planning pipeline on the ModalAI Starling platform.

The active packages are:

- `px4_msgs`: PX4 ROS message definitions.
- `voxl_msgs`: semantic detector message definitions from VOXL.
- `starling_testing`: three minimal Python flight smoke tests.
- `llm_vision_planner`: perception, prompt generation, LLM planning, refinement, verification, visualization, and Python Offboard trajectory following.

## Environment

Tested with Ubuntu 22.04, ROS 2 Humble, PX4/VOXL topics, and Python 3.

Install workspace tools:

```bash
sudo apt update
sudo apt install -y \
  git \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3-pip
```

Initialize `rosdep` once if needed:

```bash
sudo rosdep init
rosdep update
```

Clone and prepare dependencies:

```bash
git clone https://github.com/prachitgupta/starling_testing_ws.git
cd starling_testing_ws
source /opt/ros/humble/setup.bash
bash scripts/setup_workspace.sh
rosdep install --from-paths src --ignore-src -r -y
```

The LLM planner uses the OpenAI Python SDK and Instructor. Install them in the Python environment used by ROS:

```bash
python3 -m pip install openai instructor pydantic
```

Set the API key only as an environment variable:

```bash
export OPENAI_API_KEY=...
```

## Build

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select px4_msgs voxl_msgs starling_testing llm_vision_planner
source install/setup.bash
```

## Starling Smoke Tests

All positions are local NED: `x` north, `y` east, and negative `z` is altitude above the local origin.

```bash
ros2 run starling_testing 01_takeoff_land.py --ros-args \
  -p takeoff_alt_m:=0.45 -p hold_s:=5.0
```

```bash
ros2 run starling_testing 02_offboard_waypoint.py --ros-args \
  -p x:=0.5 -p y:=0.0 -p z:=-0.45
```

```bash
ros2 run starling_testing 03_bezier_offboard_waypoint.py --ros-args \
  -p x:=0.5 -p y:=0.0 -p z:=-0.45 -p duration_s:=4.0
```

Expected output: the node logs state transitions, publishes PX4 Offboard setpoints, reaches the target within per-axis epsilon, and sends a land command.

## LLM Planning Pipeline

On the Starling/VOXL, start the ModalAI MPA-to-ROS 2 bridge so camera, TFLite, ToF, and PX4 topics are available to ROS 2:

```bash
ssh root@192.168.8.1
```

When prompted:

```text
root@192.168.8.1's password: oelinux123
```

On the VOXL shell:

```bash
source /opt/ros/foxy/setup.bash
source /opt/ros/foxy/mpa_to_ros2/install/setup.bash
ros2 run voxl_mpa_to_ros2 voxl_mpa_to_ros2_node
```

If the node name differs on the installed SDK image, list the available executable and run the one shown:

```bash
ros2 pkg executables voxl_mpa_to_ros2
ros2 run voxl_mpa_to_ros2 voxl_mpa_to_ros2
```

Start the planner stack:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py mode:=semantic
```

Start the Python Offboard follower:

```bash
ros2 run llm_vision_planner trajectory_follower.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

Mission behavior:

1. `trajectory_follower.py` primes PX4 Offboard setpoints.
2. The vehicle arms and climbs to `takeoff_z`.
3. The follower publishes `/llm_vision/mission_state` as `HOLDING_FOR_PLAN`.
4. `prompt_generator.py` latches the hover pose and current obstacle snapshot.
5. `llm_planner.py` generates sparse waypoints.
6. `refinment.py` interpolates and nudges the path.
7. `verifier.py` publishes `passed`, metrics, failed constraints, thresholds, and a feedback table.
8. Failed verification results are appended into the next prompt using the same latched hover context.
9. The first `passed=true` trajectory is latched and tracked with Bezier position/velocity setpoints.

Monitor:

```bash
ros2 topic echo /llm_vision/mission_state
ros2 topic echo /llm_vision/prompt
ros2 topic echo /llm_vision/plan_verified
```

Semantic mode uses `/tflite_data` detections fused with metric XYZ samples from the organized `/tof_pc` point cloud, then places obstacles in the same PX4 local NED frame used by `/fmu/out/vehicle_odometry`. The normal point-cloud clustering node is only launched in `mode:=normal`.

ModalAI calibration notes: camera extrinsics are stored in `/etc/modalai/extrinsics.conf`, and tracking-front intrinsics can be inspected with:

```bash
cat /data/modalai/opencv_tracking_front_intrinsics.yml
```

Object detection debug note: if `/tflite_data` is not updating, edit `/etc/modalai/voxl-tflite-server.conf` with `vi`, change the input pipe from `hires/` to `hires_small_color`, then restart and inspect frames:

```bash
sudo vi /etc/modalai/voxl-tflite-server.conf
sudo systemctl restart voxl-tflite-server
voxl-inspect-cam tflite
```

Offboard setup note: disable the default Figure 8 sequence before running this mission by setting `offboard` from `figure8` to `off`.

## Software-Only Reproduction

Without hardware, run the planning nodes and publish fake mission/perception inputs:

```bash
source install/setup.bash
ros2 run llm_vision_planner prompt_generator.py --ros-args --params-file src/llm_vision_planner/config/llm_vision_planner.yaml &
ros2 run llm_vision_planner llm_planner.py --ros-args --params-file src/llm_vision_planner/config/llm_vision_planner.yaml &
ros2 run llm_vision_planner refinment.py --ros-args --params-file src/llm_vision_planner/config/llm_vision_planner.yaml &
ros2 run llm_vision_planner verifier.py --ros-args --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

Fake hover state:

```bash
ros2 topic pub /llm_vision/mission_state std_msgs/msg/String \
  "{data: '{\"state\":\"HOLDING_FOR_PLAN\",\"position\":{\"x\":0.0,\"y\":0.0,\"z\":-0.45},\"heading_deg\":0.0}'}" -r 2
```

Fake obstacle snapshot:

```bash
ros2 topic pub /llm_vision/semantic_obstacles std_msgs/msg/String \
  "{data: '{\"obstacles\":[{\"label\":\"chair\",\"min_corner\":[1.2,-0.3,-0.8],\"max_corner\":[1.7,0.3,0.0],\"size\":[0.5,0.6,0.8],\"distance_m\":1.3}],\"timestamp\":0.0}'}" -r 2
```

Expected output: `/llm_vision/prompt` is generated only after `HOLDING_FOR_PLAN`; `/llm_vision/plan_verified` contains metrics and either `passed=true` or a feedback table for retry.

## Project Artifacts

Final-project artifacts are in `src/papers/`:

- `report.pdf`
- `predictions.md`
- `results_artifact.csv`
- `per_trial_logs.jsonl`
- `submission_checklist.md`

## Known Failure Modes

- PX4 may reject Offboard if setpoints are not streamed continuously.
- The prompt generator will not query the LLM until `/llm_vision/mission_state` is `HOLDING_FOR_PLAN`.
- Perception can fluctuate; the planner intentionally latches one hover-time obstacle snapshot and reuses it for verifier-feedback retries.
- If hover drift exceeds `start_drift_replan_m`, the prompt generator relatches context.
- `OPENAI_API_KEY` must be set in the environment and must not be committed.
