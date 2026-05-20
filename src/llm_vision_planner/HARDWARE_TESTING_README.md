# LLM Vision Planner Reproducibility

## Build

```bash
cd ~/Desktop/starling_testing_ws
colcon build --packages-select px4_msgs voxl_msgs starling_testing llm_vision_planner
source install/setup.bash
```

## Software-Only Planner Check

Use this to exercise prompt generation, LLM planning, refinement, and verification without flying hardware. Set `OPENAI_API_KEY` before starting `llm_planner.py`.

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash

ros2 run llm_vision_planner prompt_generator.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml &
ros2 run llm_vision_planner llm_planner.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml &
ros2 run llm_vision_planner refinment.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml &
ros2 run llm_vision_planner verifier.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

In another terminal, publish a fake hover state and obstacle snapshot:

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 topic pub /llm_vision/mission_state std_msgs/msg/String \
  "{data: '{\"state\":\"HOLDING_FOR_PLAN\",\"position\":{\"x\":0.0,\"y\":0.0,\"z\":-0.45},\"heading_deg\":0.0}'}" -r 2
```

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 topic pub /llm_vision/semantic_obstacles std_msgs/msg/String \
  "{data: '{\"obstacles\":[{\"label\":\"chair\",\"min_corner\":[1.2,-0.3,-0.8],\"max_corner\":[1.7,0.3,0.0],\"size\":[0.5,0.6,0.8],\"distance_m\":1.3}],\"timestamp\":0.0}'}" -r 2
```

Monitor:

```bash
ros2 topic echo /llm_vision/prompt
ros2 topic echo /llm_vision/plan_verified
```

## Hardware Mission

Required hardware topics:

```bash
ros2 topic hz /fmu/out/vehicle_odometry
ros2 topic hz /qvio
ros2 topic hz /voa_pc_out
ros2 topic hz /tflite_data
```

Start the planner:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py mode:=semantic
```

Start the offboard follower:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 run llm_vision_planner trajectory_follower.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml
```

Mission sequence:

1. Follower primes PX4 Offboard setpoints.
2. Vehicle arms and climbs to `takeoff_z`.
3. Follower publishes `/llm_vision/mission_state` as `HOLDING_FOR_PLAN`.
4. Prompt generator latches the hover pose and current obstacle snapshot.
5. Failed verification results are appended to the next prompt.
6. First `passed=true` trajectory is latched and tracked with Bezier position/velocity setpoints.
7. Vehicle lands if `land_after_mission: true`.

Abort with the RC kill switch or a PX4/QGroundControl mode change.
