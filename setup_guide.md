# Starling Hardware Setup Guide

Brief reproduction steps for running the LLM vision planner on hardware or generating plots in software-only mode.

## Hardware Setup

1. Open QGroundControl.

2. Connect to VOXL2 using UDP.

3. In QGroundControl, open:

```text
Vehicle Configuration -> Tools -> Load custom params file
```

Load:

```text
src/starling_testing/params/indoor_vio_missing_gps.params
```

4. Reboot the Starling.

5. SSH into VOXL2:

```bash
ssh root@10.225.164.1
```

6. Refresh FMU microDDS topics every hardware run:

```bash
voxl-configure-microdds
```

In the menu, disable microDDS, then run the tool again and enable it.

7. Start the ModalAI MPA to ROS 2 bridge on VOXL2:

```bash
ros2 run voxl_mpa_to_ros2 voxl_mpa_to_ros2_node
```

## Remote Machine Run

1. Build the planner workspace:

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select px4_msgs voxl_msgs starling_testing llm_vision_planner
source install/setup.bash
```

2. Terminal 1: run takeoff and hold:

```bash
ros2 run llm_vision_planner mission_takeoff.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

Wait until the vehicle is holding pose and publishing `HOLDING_FOR_PLAN`.

3. Terminal 2: source and launch the planner:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:=src/llm_vision_planner/config/llm_vision_planner.yaml \
  mode:=semantic
```

4. After a verified plan is received, run the trajectory follower:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 run llm_vision_planner trajectory_follower.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

5. Useful monitors:

```bash
ros2 topic echo /llm_vision/mission_state
ros2 topic echo /llm_vision/plan_verified
```

## Simulation-Only Plot Check

Use this mode only to exercise the planner and generate plots without flying hardware.

1. Source the workspace:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
```

2. Launch the planner:

```bash
ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:=src/llm_vision_planner/config/llm_vision_planner.yaml \
  mode:=semantic
```

3. Publish mission state:

```bash
ros2 topic pub /llm_vision/mission_state std_msgs/msg/String \
  "{data: '{\"state\":\"HOLDING_FOR_PLAN\",\"position\":{\"x\":0.0,\"y\":0.0,\"z\":-0.25},\"heading_deg\":0.0}'}" -r 2
```

4. Publish obstacle info:

```bash
ros2 topic pub /llm_vision/semantic_obstacles std_msgs/msg/String \
  "{data: '{\"obstacles\":[{\"label\":\"chair\",\"min_corner\":[1.2,-0.3,-0.8],\"max_corner\":[1.7,0.3,0.0],\"size\":[0.5,0.6,0.8],\"distance_m\":1.3}],\"goal\":{\"x\":2.5,\"y\":0.0,\"z\":-0.25},\"timestamp\":0.0}'}" -r 2
```

5. Expected plot outputs:

```text
src/llm_vision_planner/plots/simulated_test_sparse_waypoint_predictions.png
src/llm_vision_planner/plots/simulated_test_refined_trajectory_waiting_for_verification.png
src/llm_vision_planner/plots/simulated_test_latched_verified_trajectory.png
```
