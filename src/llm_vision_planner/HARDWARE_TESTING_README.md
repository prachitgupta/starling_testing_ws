# Minimal Hardware Test: LLM Vision Trajectory Follower

## 1. Build

```bash
cd ~/Desktop/starling_testing_ws
colcon build --packages-select llm_vision_planner voxl_msgs
source install/setup.bash
```

## 2. Check Experiment Params

```bash
nano src/llm_vision_planner/config/llm_vision_planner.yaml
```

Current file is set for semantic mode, goal `(3.0, 0.0, -0.45)`, fixed `z=-0.45`, and `0.40 m` safety margin.

Before flight, set `llm_vision_trajectory_follower.takeoff_altitude_amsl` to a real value, or pass it on the command line in step 5.

## 3. Confirm Required Hardware Topics

```bash
ros2 topic hz /qvio
ros2 topic hz /voa_pc_out
ros2 topic hz /tflite_data
```

For normal mode, only `/qvio` and `/voa_pc_out` are required.

## 4. Terminal 1: Start Planner Pipeline

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py mode:=semantic
```

Normal mode:

```bash
ros2 launch llm_vision_planner full_plot.launch.py mode:=normal
```

## 5. Terminal 2: Start Takeoff + Trajectory Follower

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 run llm_vision_planner trajectory_follower_cpp --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml \
  -p takeoff_altitude_amsl:=1.5
```

Expected sequence:

1. PX4 pre-arm checks pass.
2. Vehicle arms and takes off to `takeoff_altitude_amsl`.
3. Follower holds post-takeoff position.
4. Planner publishes `/llm_vision/plan_verified` with `passed=true`.
5. Follower latches the verified trajectory and flies it.
6. Vehicle lands if `land_after_mission: true`.

## 6. Quick Monitoring

```bash
ros2 topic echo /llm_vision/prompt
ros2 topic echo /llm_vision/plan_verified
```

Plot output:

```bash
ls /tmp/llm_vision_plot.png
```

## 7. Abort

Use RC kill switch or PX4/QGroundControl mode change. Stop ROS nodes with `Ctrl-C`.
