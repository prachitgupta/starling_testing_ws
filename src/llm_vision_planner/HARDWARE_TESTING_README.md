# LLM Vision Planner Reproducibility

## Build

```bash
cd ~/Desktop/starling_testing_ws
colcon build --packages-select px4_msgs voxl_msgs starling_testing llm_vision_planner
source install/setup.bash
```

ROS Humble in this workspace requires system Python 3.10. If Conda is active,
start launch commands from a clean shell, or use:

```bash
env -u PYTHONHOME -u PYTHONPATH PATH=/opt/ros/humble/bin:/usr/bin:/bin bash --noprofile --norc
```

## Software-Only Planner Check

Use this to exercise prompt generation, LLM planning, refinement, and verification without flying hardware. The default provider is the configured `llama:rrt_planner`; set `llm_provider:=chatgpt` and `OPENAI_API_KEY` only when using OpenAI.

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash

ros2 run llm_vision_planner prompt_generator.py --ros-args \
  --params-file src/llm_vision_planner/config/llm_vision_planner.yaml \
  -p environment:=sim &
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
ros2 topic pub /llm_vision/sim_obstacles std_msgs/msg/String \
  "{data: '{\"obstacles\":[{\"label\":\"chair\",\"min_corner\":[1.2,-0.3,-0.8],\"max_corner\":[1.7,0.3,0.0],\"size\":[0.5,0.6,0.8],\"distance_m\":1.3}],\"timestamp\":0.0}'}" -r 2
```

Monitor:

```bash
ros2 topic echo /llm_vision/prompt
ros2 topic echo /llm_vision/plan_verified
```

## Double-Integrator Conformal Calibration Smoke Test

Generate a small prediction-pair CSV with RRT labels and Llama waypoint
predictions, then build bounded minimum-control QP trajectories and position
scores:

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 2 \
  --output fine_tuning/datasets/conformal_rrt_calibration_smoke.csv \
  --vllm-base-url http://172.22.224.93:8000/v1 \
  --llama-model-name rrt_planner

python3 fine_tuning/scripts/generate_qp_calibration_dataset.py \
  --input fine_tuning/datasets/conformal_rrt_calibration_smoke.csv \
  --output fine_tuning/datasets/calibration_min_control_qp_shared_clock_with_limits_smoke.csv \
  --samples 2 --dt 0.1 \
  --max-velocity-mps 0.5 --max-acceleration-mps2 0.5

python3 fine_tuning/scripts/generate_position_score_calibration.py \
  --input fine_tuning/datasets/calibration_min_control_qp_shared_clock_with_limits_smoke.csv \
  --output fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_smoke.csv \
  --samples 2 --delta-p 0.10 --delta-w 0.10 --control-dt 0.05
```

Run the double-integrator controller verification and plot the RRT reference, LLM reference, controlled trajectory, and conformal tube:

```bash
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_smoke.csv \
  --sample-id 0 \
  --trajectory-dt 0.1 \
  --max-velocity-mps 0.5 --max-acceleration-mps2 0.5 \
  --output-png fine_tuning/plots/contraction/qp_with_limits_smoke.png \
  --report-json fine_tuning/results/contraction/qp_with_limits_smoke.json
```

## PX4 SITL obstacle-avoidance world (manual launch)

Use this instead of `launch_obstacle_avoidance_x500.sh` when you want to start
the same `obstacle_avoidance.sdf` world manually. Do not run the helper script
or another `make px4_sitl ...` command at the same time: this manual sequence
must have exactly one Gazebo server, one PX4 instance, and one XRCE agent.

Terminal 1 — start the Gazebo world:

```bash
cd ~/PX4-Autopilot
source build/px4_sitl_default/rootfs/gz_env.sh
source Tools/simulation/gz/config/obstacle_avoidance_x500.env
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}:$PWD/Tools/simulation/gz/models:$PWD/Tools/simulation/gz/worlds"
unset GZ_SIM_SERVER_CONFIG_PATH
gz sim -r -s Tools/simulation/gz/worlds/obstacle_avoidance.sdf
```

Terminal 2 — start the one standalone PX4 instance for that world:

```bash
cd ~/PX4-Autopilot
source Tools/simulation/gz/config/obstacle_avoidance_x500.env
PX4_GZ_STANDALONE=1 make px4_sitl "$PX4_SIM_MODEL"
```

Terminal 3 — start the single DDS agent:

```bash
MicroXRCEAgent udp4 -p 8888
```

Terminal 4 — bridge the simulated camera/depth data only when semantic
perception requires it:

```bash
source /opt/ros/humble/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  /depth_camera@sensor_msgs/msg/Image[gz.msgs.Image \
  /depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
  /world/obstacle_avoidance/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /world/obstacle_avoidance/model/x500_depth_0/link/camera_link/sensor/IMX214/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock
```

Before launching the planner, verify that each PX4 topic has one endpoint:

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 topic info -v /fmu/out/vehicle_odometry
ros2 topic info -v /fmu/in/vehicle_command
```

The expected counts are one odometry publisher and one vehicle-command
subscriber. For simulator missions, do not start a perception node or bridge
camera data: publish `/llm_vision/sim_obstacles` and launch with
`environment:=sim` below.

## Hardware Mission

Connect to the Starling/VOXL over USB-C/Wi-Fi and start the ModalAI MPA-to-ROS 2 bridge on the vehicle:

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

Required hardware topics:

```bash
ros2 topic hz /fmu/out/vehicle_odometry
ros2 topic hz /tflite_data
ros2 topic hz /tof_pc
```

DDS agent note: if MPA topics such as `/tflite_data` or `/tof_pc` are visible on the remote machine but `/fmu/...` topics are not, reconfigure the VOXL microDDS agent on the vehicle and disable/re-enable it:

```bash
ssh root@192.168.8.1
voxl-configure-microdds
```

Follow the prompts to disable, then run `voxl-configure-microdds` again to enable the DDS agent. Afterward, restart the relevant services or reboot the vehicle and re-check `/fmu/out/vehicle_odometry`.

In semantic mode, `perception_detection.py` fuses `/tflite_data` detections with metric XYZ samples from the organized `/tof_pc` point cloud, then places obstacles in the same PX4 local NED frame used by `/fmu/out/vehicle_odometry`.

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

Offboard setup note: disable the default Figure 8 sequence before running this mission by setting `offboard` from `figure8` to `off` IN vi /etc/modalai/voxl-vision-hub.conf

Start the unified planner and offboard controller:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py environment:=real
```

`control_law_executer.py` is launched automatically and is the sole PX4 offboard
owner. Do not also run `mission_takeoff.py`, `trajectory_follower.py`, or
`trajectory_follower_continuous.py`.
Their obsolete parameter sections are intentionally absent from the unified-launch
configuration file.

For the live contraction plot instead of the standard planner plot, launch:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py \
  environment:=real visualizer:=contraction
```

## Simulator mission

Simulator mode deliberately does **not** start a perception node. Publish the
simulated obstacle snapshot yourself on `/llm_vision/sim_obstacles`; its JSON
format matches the semantic obstacle output and is latched by the prompt
generator after takeoff:

For the smallest end-to-end flight test, publish an explicit empty snapshot in
one terminal. This is not perception; it tells the LLM that no obstacles are
present in the simulated planning context.

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 topic pub -r 2 /llm_vision/sim_obstacles std_msgs/msg/String \
  "{data: '{\"obstacles\":[],\"timestamp\":0.0}'}"
```

For a calibrated obstacle environment, use the `ros2_pub_command` column in
`fine_tuning/datasets/env_ros_commands.csv` instead. It publishes the exact
obstacle boxes for the selected calibration row.

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 topic pub -r 2 /llm_vision/sim_obstacles std_msgs/msg/String \
  "{data: '{\"obstacles\":[{\"label\":\"sim_box\",\"min_corner\":[1.2,-0.3,-0.8],\"max_corner\":[1.7,0.3,0.0],\"size\":[0.5,0.6,0.8],\"distance_m\":1.3}],\"timestamp\":0.0}'}"
```

Then launch the unified stack. The LLM still generates the route; the simulator
topic is only its obstacle context.

```bash
ros2 launch llm_vision_planner full_plot.launch.py environment:=sim
```

Mission sequence:

1. Control executor waits for fresh PX4 local-NED odometry, then primes Offboard setpoints.
2. Control executor arms the vehicle and climbs to `takeoff_z`.
3. Control executor publishes `/llm_vision/mission_state` as `HOLDING_FOR_PLAN`.
4. Prompt generator latches the hover pose and current obstacle snapshot.
5. Failed verification results are appended to the next prompt.
6. The first `passed=true` trajectory is converted to a minimum-control QP reference.
7. The executor tracks `u = u_reference - K(x - x_reference)`, holds the goal, stops Offboard setpoints, requests PX4 auto-land, and disarms after landing detection.

### Indoor horizontal-motion envelope

The live QP and the outgoing PX4 velocity setpoint both use hard horizontal
limits from `config/llm_vision_planner.yaml`:

- `max_horizontal_speed_mps: 0.5`
- `max_horizontal_acceleration_mps2: 0.5`

The QP generator checks its reference state velocity, reference velocity
command, and modeled acceleration against these hard bounds; when needed, it
lengthens the horizon and resolves before accepting the plan. The executor then
applies the same vector speed cap and a per-message velocity slew-rate cap
before publishing to PX4. These are deliberately below the indoor PX4 parameter envelope in
`starling_testing/params/indoor_vio_missing_gps.params`
(`MPC_XY_VEL_MAX=0.5`, `MPC_XY_CRUISE=0.5`, `MPC_ACC_HOR=0.5`). This keeps
PX4's position-only takeoff and hold behaviour inside the same envelope. Keep
the planner verifier limits at the same or stricter values; they are now
`0.5 m/s` and `0.5 m/s^2`.

The contraction visualizer latches the same passed plan, reconstructs its QP
reference, and subscribes directly to `/fmu/out/vehicle_odometry`. It shows the
live vehicle as a disk, retains its measured trajectory, and uses the same fixed
workspace axes and certificate layers as the offline dconformal plot: the direct
position tube from `q_p`, the exact 2D projection of the 4D `q_w` tube, and the
4D-state radius legend (which is not a position circle). It intentionally omits
the RRT trajectory and does not propagate a simulated "true" state. The values
are computed at launch from `s_p` and `s_w` in the calibration CSV named by
`verify_contraction.calibration_csv`; set `calibration_samples`, `delta_p`, and
`delta_w` there to select the finite-sample certificate.

Abort with the RC kill switch or a PX4/QGroundControl mode change.
