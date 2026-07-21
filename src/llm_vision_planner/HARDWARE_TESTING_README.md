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

## Double-Integrator Conformal Calibration Smoke Test

Generate a small calibration CSV with RRT labels, Llama 8B waypoint predictions, min-snap trajectories, and the PDF scores `s_u` and `s_x`:

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 2 \
  --output fine_tuning/datasets/conformal_rrt_calibration_smoke.csv \
  --vllm-base-url http://172.22.224.93:8000/v1 \
  --llama-model-name rrt_planner
```

Run the double-integrator controller verification and plot the RRT reference, LLM reference, controlled trajectory, and conformal tube:

```bash
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/conformal_rrt_calibration_smoke.csv \
  --sample-id 0 \
  --output-png fine_tuning/plots/dconformal_contraction_verification_smoke.png \
  --report-json fine_tuning/plots/dconformal_contraction_verification_smoke.json
```

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
ros2 launch llm_vision_planner full_plot.launch.py mode:=semantic
```

`control_law_executer.py` is launched automatically and is the sole PX4 offboard
owner. Do not also run `mission_takeoff.py`, `trajectory_follower.py`, or
`trajectory_follower_continuous.py`.

For the live contraction plot instead of the standard planner plot, launch:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py \
  mode:=semantic visualizer:=contraction
```

Mission sequence:

1. Control executor waits for PX4 odometry/status, then primes Offboard setpoints.
2. Control executor arms the vehicle and climbs to `takeoff_z`.
3. Control executor publishes `/llm_vision/mission_state` as `HOLDING_FOR_PLAN`.
4. Prompt generator latches the hover pose and current obstacle snapshot.
5. Failed verification results are appended to the next prompt.
6. The first `passed=true` trajectory is converted to a minimum-control QP reference.
7. The executor tracks `u = u_reference - K(x - x_reference)`, holds the goal, and requests PX4 auto-land.

The contraction visualizer latches the same passed plan, reconstructs its QP
reference, and subscribes directly to `/fmu/out/vehicle_odometry`. It shows the
live vehicle as a disk, retains its measured trajectory, and overlays the 90%
projected 2D contraction tube. It does not propagate a simulated "true" state.
The tube is computed at launch from the `s_w` scores in the calibration CSV named
by `verify_contraction.calibration_csv`; set `calibration_samples` and `delta_w`
there to select the finite-sample certificate.

Abort with the RC kill switch or a PX4/QGroundControl mode change.
