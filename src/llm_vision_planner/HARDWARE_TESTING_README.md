# Minimal Hardware Test: LLM Vision Trajectory Follower

## 1. Build

```bash
cd ~/Desktop/starling_testing_ws
colcon build --packages-select px4_msgs starling_testing llm_vision_planner
source install/setup.bash
```

## 2. Check Experiment Params

```bash
nano src/llm_vision_planner/config/llm_vision_planner.yaml
```

Current file is set for semantic mode, goal `(3.0, 0.0, -0.45)`, fixed `z=-0.45`, and `0.40 m` safety margin.

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
ros2 run llm_vision_planner trajectory_follower.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

Expected sequence:

1. Follower waits for fresh `/fmu/out/vehicle_odometry`.
2. Follower primes PX4 Offboard mode with hold setpoints.
3. Vehicle switches to Offboard and arms if `auto_arm: true`.
4. Planner publishes `/llm_vision/plan_verified` with `passed=true`.
5. Follower latches the verified trajectory, smooths each segment with Bezier position/velocity setpoints, and flies it.
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
