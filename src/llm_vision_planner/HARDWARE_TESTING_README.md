# Starling 2 Hardware Mission

This procedure uses:

- Starling 2 / VOXL at `10.117.229.1`
- Vicon Tracker computer at `10.117.229.124`
- MAVROS on the ground-station laptop
- Vicon pose through MAVROS into the PX4 EKF2
- VOXL MPA-to-ROS 2 and PX4 microDDS for perception and `/fmu/*` topics

Do the first setup and takeoff checks with propellers removed or the vehicle
restrained. Install propellers only after the estimator checks pass.

## 1. Power and network

Power on the Starling 2, Vicon system, Vicon Tracker computer, and ground
station. Put the Starling, Vicon computer, and ground station on the same
network.

On the ground station:

```bash
export Starling2=10.117.229.1
export VICON_COMPUTER_IP=10.117.229.124

ping -c 3 "$Starling2"
ping -c 3 "$VICON_COMPUTER_IP"
nc -vz "$VICON_COMPUTER_IP" 801
```

TCP port `801` must be reachable for the Vicon DataStream SDK. Allow Vicon
Tracker/DataStream through the Windows firewall if this check fails.

## 2. Configure QGroundControl UDP

Older `voxl-mavlink-server` versions send ground-station traffic back to UDP
port `14550`. MAVROS must therefore own laptop port `14550`.

In QGroundControl:

1. Disable the automatic UDP link/listener using port `14550`.
2. Open **Application Settings > Comm Links**.
3. Add a UDP link with listening port `14551`.
4. Save it, but start MAVROS before connecting this QGroundControl link.

Check that port `14550` is free before MAVROS starts:

```bash
ss -lunp | grep 14550
```

If QGroundControl still owns the port, close it, start MAVROS, reopen
QGroundControl, and connect the `14551` link.

## 3. Install MAVROS and GeographicLib data

Run once on the ground station:

```bash
sudo apt update
sudo apt install ros-humble-mavros ros-humble-mavros-extras geographiclib-tools
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
```

Do not run the installer as `sudo ros2 ...`; root does not inherit the ROS
environment. Verify the required geoid:

```bash
ls -lh /usr/share/GeographicLib/geoids/egm96-5.pgm
```

## 4. Start MAVROS

On the ground station:

```bash
source /opt/ros/humble/setup.bash

ros2 launch mavros px4.launch \
  fcu_url:="udp://0.0.0.0:14550@${Starling2}:14550" \
  gcs_url:="udp://0.0.0.0:14556@127.0.0.1:14551"
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mavros/state
```

Continue only when:

```text
connected: true
```

QGroundControl should now connect through its UDP `14551` link.

## 5. Start the Vicon bridge

In Tracker:

1. Set the Starling subject and segment names to `Starling2`.
2. Enable the DataStream server.
3. Select an actual 50 Hz or 100 Hz system/DataStream rate.
4. Keep the rigid body visible and unoccluded.

Build the bridge once if needed:

```bash
cd ~/colcon_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vicon_bridge
```

Source it in every Vicon bridge terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash
```

For a Vicon rate of 50 Hz:

```bash
ros2 run vicon_bridge vicon_bridge --ros-args \
  -p host_name:="${VICON_COMPUTER_IP}:801" \
  -p stream_mode:="ServerPush" \
  -p update_rate_hz:=125.0 \
  -p expected_rate_hz:=50.0 \
  -p publish_specific_segment:=true \
  -p target_subject_name:="Starling2" \
  -p target_segment_name:="Starling2" \
  -p world_frame_id:="vicon_world" \
  -p tf_namespace:="vicon" \
  -r /vicon/Starling2/Starling2/pose:=/mavros/vision_pose/pose
```

For a Vicon rate of 100 Hz:

```bash
ros2 run vicon_bridge vicon_bridge --ros-args \
  -p host_name:="${VICON_COMPUTER_IP}:801" \
  -p stream_mode:="ServerPush" \
  -p update_rate_hz:=250.0 \
  -p expected_rate_hz:=100.0 \
  -p publish_specific_segment:=true \
  -p target_subject_name:="Starling2" \
  -p target_segment_name:="Starling2" \
  -p world_frame_id:="vicon_world" \
  -p tf_namespace:="vicon" \
  -r /vicon/Starling2/Starling2/pose:=/mavros/vision_pose/pose
```

Run one bridge command only. Do not run `topic_tools throttle`. Because the
PoseStamped topic is remapped, it appears directly as:

```text
/mavros/vision_pose/pose
```

The unremapped TransformStamped topic remains:

```text
/vicon/Starling2/Starling2
```

Verify the stream:

```bash
ros2 topic echo /mavros/vision_pose/pose --once
ros2 topic hz /mavros/vision_pose/pose --window 500
```

The maximum interval must remain below `0.2 s`; below `0.05 s` is preferred.
Do not hide bridge drop warnings by changing only `expected_rate_hz`.

## 6. Start VOXL MPA-to-ROS 2 and verify microDDS

Open a separate ground-station terminal:

```bash
ssh root@"$Starling2"
```

On the VOXL:

```bash
source /opt/ros/foxy/setup.bash
source /opt/ros/foxy/mpa_to_ros2/install/setup.bash
ros2 pkg executables voxl_mpa_to_ros2
```

Run the executable listed by the installed image. The common command is:

```bash
ros2 run voxl_mpa_to_ros2 voxl_mpa_to_ros2_node
```

If that executable is not listed, use:

```bash
ros2 run voxl_mpa_to_ros2 voxl_mpa_to_ros2
```

On the ground station, verify both MPA and PX4 DDS topics:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_testing_ws/install/setup.bash

ros2 topic info -v /tflite_data
ros2 topic info -v /tof_pc
ros2 topic info -v /fmu/out/vehicle_odometry
ros2 topic hz /fmu/out/vehicle_odometry
```

If `/tflite_data` and `/tof_pc` exist but `/fmu/*` topics do not, configure
microDDS on the VOXL:

```bash
voxl-configure-microdds
```

Select disable, run `voxl-configure-microdds` again, and select enable. Reboot
the vehicle after reconfiguration:

```bash
reboot
```

After the VOXL restarts, reconnect, restart MPA-to-ROS 2, and repeat the topic
checks.

## 7. Load the PX4 Vicon parameters

In QGroundControl open:

```text
Vehicle Setup > Parameters > Tools > Load from file
```

Load:

```text
~/Desktop/starling_testing_ws/src/llm_vision_planner/params/vicon_voxl.params
```

Reboot PX4. If the QGroundControl reboot action fails, disarm, remove vehicle
power, wait 10 seconds, and power it on again. Keep Vicon and MAVROS streaming
during PX4 initialization.

Verify these parameters in QGroundControl:

```text
EKF2_EV_CTRL     = 11    # horizontal position, vertical position, Vicon yaw
EKF2_HGT_REF     = 3     # vision height reference
EKF2_GPS_CTRL    = 0     # indoor GNSS disabled
EKF2_OF_CTRL     = 0
EKF2_RNG_CTRL    = 0
EKF2_MAG_TYPE    = 6     # magnetometer initializes yaw only
EKF2_EV_QMIN     = 0
EKF2_EV_NOISE_MD = 1
EKF2_EVP_NOISE   = 0.10 m
EKF2_EVA_NOISE   = 0.05 rad
EKF2_EV_DELAY    = 0 ms initially
```

`EKF2_MAG_TYPE=6` is not continuous 3-axis magnetometer fusion. It initializes
heading, after which Vicon yaw is the continuing heading reference.

## 8. Verify frame alignment and EKF2 fusion

Place the vehicle flat on the ground with its nose aligned to the selected
Vicon world direction. Move it by hand with propellers removed and verify:

```text
MAVROS ENU +X  -> PX4 NED +Y
MAVROS ENU +Y  -> PX4 NED +X
MAVROS ENU +Z  -> PX4 NED -Z
```

On the ground station:

```bash
ros2 topic echo /mavros/vision_pose/pose --once
```

On the VOXL:

```bash
px4-listener vehicle_visual_odometry 5
px4-listener estimator_status_flags 1
px4-listener estimator_aid_src_ev_pos 1
px4-listener estimator_aid_src_ev_hgt 1
px4-listener estimator_aid_src_ev_yaw 1
px4-listener vehicle_odometry 5
```

Required estimator flags:

```text
cs_yaw_align: True
cs_ev_pos:    True
cs_ev_hgt:    True
cs_ev_yaw:    True
cs_fake_pos:  False
```

If `vehicle_visual_odometry` is correct but these flags remain false, check in
this order:

1. Vision gaps are below `0.2 s`.
2. `EKF2_MAG_TYPE=6` allowed initial yaw alignment.
3. `EKF2_EV_CTRL=11` and `EKF2_HGT_REF=3`.
4. `innovation_rejected` and `test_ratio` in the three aid-source topics.
5. Vicon yaw is aligned and its quaternion is valid.
6. `EKF2_EV_DELAY` is tuned from a PX4 log only after network jitter is fixed.

The Vicon object origin can be above the floor. For example, Vicon ENU
`z=+0.084 m` correctly becomes PX4 NED `z=-0.084 m`; it need not equal exactly
zero.

## 9. Isolated QGroundControl takeoff test

Do this before running the planner:

1. Install propellers and move all personnel outside the safety area.
2. Confirm the RC kill switch and mode switch work.
3. Confirm the Vicon pose is continuous.
4. Confirm all required EKF flags above remain true.
5. Confirm QGroundControl reports a valid local position and no blocking
   preflight failures.
6. In the QGroundControl Fly view, command a low `1.0 m` takeoff.
7. Hold briefly, land from QGroundControl, and disarm.

Do not continue if Offboard/Position mode falls back, Vicon fusion stops, or
local position jumps. Land and repeat the estimator diagnostics.

The planner mission performs its own arming and takeoff. Land and disarm after
this isolated QGroundControl test before launching `full_plot.launch.py`.

## 10. Build the planner workspace

On the ground station:

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select px4_msgs voxl_msgs starling_testing llm_vision_planner
source install/setup.bash
```

On the VOXL, disable the default Figure 8 producer before this mission. In:

```bash
vi /etc/modalai/voxl-vision-hub.conf
```

set:

```json
"offboard_mode": "off",
```

Verify:

```bash
grep -n '"offboard_mode"' /etc/modalai/voxl-vision-hub.conf
```

Do not run any other Offboard publisher. `control_law_executer.py`, launched by
`full_plot.launch.py`, must be the only publisher controlling PX4.

## 11. Hardware flight with a simulated obstacle message

This test flies the real vehicle but supplies obstacle JSON instead of starting
TFLite/ToF perception. `environment:=sim` changes only the obstacle source; PX4
odometry and control remain real.

Use an empty obstacle snapshot:

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 topic pub -r 2 /llm_vision/sim_obstacles std_msgs/msg/String \
  "{data: '{\"obstacles\":[],\"timestamp\":0.0}'}"
```

For a calibrated obstacle environment, copy a command from the
`ros2_pub_command` column of:

```text
~/Desktop/starling_testing_ws/src/llm_vision_planner/fine_tuning/datasets/env_ros_commands.csv
```

Run the selected publisher in its own terminal. Then launch the complete
planner, verifier, visualizer, and control executor:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash

ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  environment:=sim \
  llm_provider:=llama \
  visualizer:=standard \
  show_rrt:=true
```

For the live conformal contraction plot:

```bash
ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  environment:=sim \
  llm_provider:=llama \
  visualizer:=contraction \
  show_rrt:=false
```

Monitor:

```bash
ros2 topic echo /llm_vision/mission_state
ros2 topic echo /llm_vision/prompt
ros2 topic echo /llm_vision/plan_verified
ros2 topic echo /llm_vision/offboard_owner
```

## 12. Hardware flight with TFLite perception

The current configuration uses:

```yaml
semantic_obstacle_perception:
  ros__parameters:
    detection_topic: /tflite_data
    point_cloud_topic: /tof_pc
    z_estimation_mode: hardcoded
```

With `z_estimation_mode: hardcoded`, TFLite supplies the label and bounding box,
and `perception_detection.py` uses its class-based hardcoded depth. `/tof_pc`
may be running, but it is not used to estimate range in this mode.

To use actual ToF depth instead, change:

```yaml
z_estimation_mode: depth
```

Before the perception flight:

```bash
ros2 topic hz /tflite_data
ros2 topic hz /tof_pc
ros2 topic hz /fmu/out/vehicle_odometry
```

Inspect the generated obstacles:

```bash
ros2 topic echo /llm_vision/semantic_obstacles
```

Launch the real-perception mission:

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash

ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  environment:=real \
  llm_provider:=llama \
  visualizer:=standard \
  show_rrt:=true
```

For the contraction visualizer, replace `visualizer:=standard` with:

```text
visualizer:=contraction
```

Use the RC kill switch or an intentional PX4/QGroundControl mode change to
abort. Never stop Vicon, MAVROS, MPA-to-ROS 2, or PX4 DDS while the vehicle is
airborne.
