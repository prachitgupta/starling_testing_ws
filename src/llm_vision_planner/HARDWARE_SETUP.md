# Hardware setup and preflight

Preparation only; mission launch is in [Hardware testing](HARDWARE_TESTING_README.md), and all recording procedures are in [Record Calibration Data](Record%20Calibration%20Data.md).

## 0. Clone, install, build, and source

Ground station: Ubuntu 22.04 with ROS 2 Humble already installed. VOXL uses its
existing Foxy image, PX4 firmware, MPA bridge, and TFLite detector. The pinned
message dependencies are imported by the setup script. A GPU server with gated
Llama access and a trained HRRT adapter is also required for mission planning.

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3-colcon-common-extensions \
  python3-rosdep python3-vcstool python3-pip python3-venv python3-numpy python3-scipy \
  python3-matplotlib python3-sklearn python3-pytest curl netcat-openbsd

mkdir -p ~/Desktop
git clone --depth 1 --single-branch --branch main \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/starling_final_testing
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
bash scripts/setup_workspace.sh
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths src --ignore-src -r -y
/usr/bin/python3 -m pip install 'numpy<2' openai instructor pydantic
colcon build --symlink-install --packages-select px4_msgs voxl_msgs llm_vision_planner
source install/setup.bash
ros2 launch llm_vision_planner full_plot.launch.py --show-args
```

Use an empty destination for cloning. If that folder already contains `main`,
choose another empty destination and replace the workspace path
in this guide. Source `/opt/ros/humble/setup.bash` and this clone's
`install/setup.bash` in every ground-station ROS terminal. Continuous topic
monitors and service/launch commands each need their own terminal; stop a
monitor with `Ctrl+C` before running the next monitor command.

Follow the setup sections below, then the separate hardware-testing and
recording guides linked above.

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

### Option A: bind ROS 2 DDS to the flight Wi-Fi and domain 42

#### Run once on the ground station

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_vision_planner
source install/setup.bash

ssh root@"$Starling2" \
  'mkdir -p /data/llm_vision_planner/scripts /data/llm_vision_planner/config'
scp src/llm_vision_planner/scripts/ros_wifi_dds.sh \
  root@"$Starling2":/data/llm_vision_planner/scripts/
scp src/llm_vision_planner/config/fastdds_wifi_only.xml.in \
  root@"$Starling2":/data/llm_vision_planner/config/
```

#### Run once on the VOXL

```bash
VOXL_WIFI_IPV4="$(ip -4 -o address show dev wlan0 scope global | \
  awk 'NR == 1 {split($4, address, "/"); print address[1]}')"
sed "s/@ROS_WIFI_IPV4@/${VOXL_WIFI_IPV4}/g" \
  /data/llm_vision_planner/config/fastdds_wifi_only.xml.in \
  >/data/llm_vision_planner/config/fastdds_wifi_only.xml
chmod 600 /data/llm_vision_planner/config/fastdds_wifi_only.xml

install -d /etc/systemd/system/voxl-microdds-agent.service.d
printf '%s\n' \
  '[Service]' \
  'Environment=ROS_DOMAIN_ID=42' \
  'Environment=FASTRTPS_DEFAULT_PROFILES_FILE=/data/llm_vision_planner/config/fastdds_wifi_only.xml' \
  'Environment=FASTDDS_DEFAULT_PROFILES_FILE=/data/llm_vision_planner/config/fastdds_wifi_only.xml' \
  >/etc/systemd/system/voxl-microdds-agent.service.d/10-flight-dds.conf

systemctl daemon-reload
systemctl restart voxl-microdds-agent
systemctl show voxl-microdds-agent --property=LoadState,ActiveState,Environment
```

#### Run in every ground-station ROS 2 terminal

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_final_testing/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
```

#### Run in every VOXL ROS 2 terminal

```bash
source /opt/ros/foxy/setup.bash
source /opt/ros/foxy/mpa_to_ros2/install/setup.bash
source /data/llm_vision_planner/scripts/ros_wifi_dds.sh enable wlan0 42
systemctl restart voxl-microdds-agent
ros2 daemon start
```

#### Verify on the ground station

```bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" status
printenv RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ip route get "$Starling2"
ip route get "$VICON_COMPUTER_IP"
ip route get 172.22.224.93
```

#### Verify in a VOXL ROS 2 terminal

```bash
source /data/llm_vision_planner/scripts/ros_wifi_dds.sh status
printenv RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ip -4 -br address show wlan0
```

All later commands assume Option A. If you choose Option B, omit every later
`ros_wifi_dds.sh enable` command and keep `XRCE_DDS_DOM_ID=0` after loading the
parameter file (which contains domain 42).

### Option B: use default ROS 2 DDS

Do not run any `ros_wifi_dds.sh enable` command when using this option.

#### Disable Option A on the ground station

Run in every ground-station terminal where Option A is active:

```bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" disable
unset RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ros2 daemon stop
ros2 daemon start
```

#### Disable Option A on the VOXL

```bash
source /data/llm_vision_planner/scripts/ros_wifi_dds.sh disable
unset RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
rm -f /etc/systemd/system/voxl-microdds-agent.service.d/10-flight-dds.conf
rm -f /data/llm_vision_planner/config/fastdds_wifi_only.xml
systemctl daemon-reload
systemctl restart voxl-microdds-agent
ros2 daemon stop
ros2 daemon start
```

In **QGroundControl > Analyze Tools > MAVLink Console**:

```bash
param set XRCE_DDS_DOM_ID 0
param save
reboot
```

#### Start after power-on with default DDS

Run in every new ground-station ROS 2 terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_final_testing/install/setup.bash
unset RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ros2 daemon stop
ros2 daemon start
```

Run in every new VOXL ROS 2 terminal:

```bash
source /opt/ros/foxy/setup.bash
source /opt/ros/foxy/mpa_to_ros2/install/setup.bash
unset RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ros2 daemon stop
ros2 daemon start
systemctl is-active voxl-microdds-agent
```

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

## Planner service prerequisite

The live mission uses the existing Llama/vLLM-compatible service. Provision a validated service separately and confirm its model name matches `llama_model_name`; old adapter experiments are intentionally not distributed on main. The final V1 Transformer is an offline calibration-gateway backend, not a ROS mission backend.

## 4. Start MAVROS

On the ground station:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_final_testing/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start

export Starling2=10.117.229.1
ros2 launch mavros px4.launch \
  fcu_url:="udp://0.0.0.0:14550@${Starling2}:14550" \
  gcs_url:="udp://0.0.0.0:14556@127.0.0.1:14551"
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_final_testing/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
ros2 topic echo /mavros/state
```

Continue only when:

```text
connected: true
```

QGroundControl should now connect through its UDP `14551` link.

For Option A, run in **QGroundControl > Analyze Tools > MAVLink Console**:

```bash
param set XRCE_DDS_DOM_ID 42
param save
reboot
```

For Option B, run in the MAVLink Console:

```bash
param set XRCE_DDS_DOM_ID 0
param save
reboot
```

After PX4 reconnects, verify:

```bash
param show XRCE_DDS_DOM_ID
```

## 5. Start the Vicon bridge

In Tracker:

1. Set the Starling subject and segment names to `Starling2`.
2. Enable the DataStream server.
3. Select an actual 50 Hz or 100 Hz system/DataStream rate.
4. Keep the rigid body visible and unoccluded.

The bridge was verified on Ubuntu 22.04 with ROS 2 Humble using
`dasc-lab/ros2-vicon-bridge` package version `0.0.1`, Vicon DataStream SDK
`1.12`, and commit `893aba0eb8b7d316d90865ac46394616bfb0bb36`. Install and
build that revision once on the ground station:

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3-colcon-common-extensions \
  libboost-thread-dev libboost-date-time-dev libboost-chrono-dev \
  ros-humble-ament-cmake ros-humble-rclcpp ros-humble-geometry-msgs \
  ros-humble-tf2 ros-humble-tf2-ros ros-humble-diagnostic-updater

mkdir -p ~/colcon_ws/src
git clone https://github.com/dasc-lab/ros2-vicon-bridge.git \
  ~/colcon_ws/src/ros2-vicon-bridge
git -C ~/colcon_ws/src/ros2-vicon-bridge checkout \
  893aba0eb8b7d316d90865ac46394616bfb0bb36

cd ~/colcon_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select vicon_bridge
```

If the repository already exists, skip `git clone`, confirm that local changes
are intentional, check out the verified commit, and rebuild. Verify the
installation and dependencies:

```bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash

git -C ~/colcon_ws/src/ros2-vicon-bridge rev-parse HEAD
ros2 pkg prefix vicon_bridge
ros2 pkg xml vicon_bridge | grep -m1 '<version>'
ros2 pkg executables vicon_bridge
ldd ~/colcon_ws/install/vicon_bridge/lib/vicon_bridge/vicon_bridge | \
  grep 'not found'
```

The commands must report the pinned commit, prefix
`~/colcon_ws/install/vicon_bridge`, version `0.0.1`, and both `vicon_bridge`
executables. The final `ldd` command must print nothing; any output names a
missing runtime dependency that must be installed before continuing.

Source it in every Vicon bridge terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_final_testing/install/setup.bash
source ~/colcon_ws/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
```

For a Vicon rate of 50 Hz:

```bash
export VICON_COMPUTER_IP=10.117.229.124
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
export VICON_COMPUTER_IP=10.117.229.124
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
export Starling2=10.117.229.1
ssh root@"$Starling2"
```

On the VOXL:

```bash
source /opt/ros/foxy/setup.bash
source /opt/ros/foxy/mpa_to_ros2/install/setup.bash
source /data/llm_vision_planner/scripts/ros_wifi_dds.sh enable wlan0 42
systemctl restart voxl-microdds-agent
ros2 daemon start
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

Set the TFLite input pipe to `hires_small_color`:

```bash
vi /etc/modalai/voxl-tflite-server.conf
```

```json
"input_pipe": "hires_small_color",
```

```bash
systemctl restart voxl-tflite-server
```

On the ground station, verify both MPA and PX4 DDS topics:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_final_testing/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start

ros2 topic info -v /tflite_data
ros2 topic info -v /tflite
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
~/Desktop/starling_final_testing/src/llm_vision_planner/params/vicon_voxl.params
```

Load this file after any QVIO parameter profile; loading the QVIO profile
afterward will overwrite the required Vicon settings.

Reboot PX4. If the QGroundControl reboot action fails, disarm, remove vehicle
power, wait 10 seconds, and power it on again. Keep Vicon and MAVROS streaming
during PX4 initialization.

Verify these parameters in QGroundControl:

```text
SYS_HAS_GPS      = 0
SYS_HAS_MAG      = 0
COM_ARM_WO_GPS   = 1
EKF2_AID_MASK    = 0
EKF2_EV_CTRL     = 11    # horizontal position, vertical position, Vicon yaw
EKF2_HGT_REF     = 3     # vision height reference
EKF2_GPS_CTRL    = 0     # indoor GNSS disabled
EKF2_OF_CTRL     = 0
EKF2_RNG_CTRL    = 0
EKF2_MAG_TYPE    = 5     # magnetometer disabled; Vicon supplies yaw
EKF2_EV_QMIN     = 0
EKF2_EV_NOISE_MD = 1
EKF2_EVP_NOISE   = 0.10 m
EKF2_EVA_NOISE   = 0.05 rad
EKF2_EV_DELAY    = 0 ms initially
XRCE_DDS_DOM_ID  = 42
```

On PX4 v1.14, `EKF2_MAG_TYPE=5` disables magnetometer fusion. With the yaw bit
enabled in `EKF2_EV_CTRL`, Vicon initializes and supplies the continuing yaw
reference. `EKF2_MAG_TYPE=6` is not defined in PX4 v1.14.

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
2. `EKF2_MAG_TYPE=5` is set and the Vicon quaternion allows yaw alignment.
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

Path refinement uses the plan's `workspace.z`; the YAML
`path_refinement.fixed_z` remains the fallback when the workspace omits `z`.
The current hardware settings stay at `-0.5 m` NED. If changing flight altitude,
keep the prompt/gateway `fixed_z`, goal Z, refiner fallback, and executor
`takeoff_z` consistent. The launch commands below need no new arguments.

On the ground station:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
colcon build --packages-select px4_msgs voxl_msgs llm_vision_planner
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

Do not run any other Offboard publisher. During planner missions,
`control_law_executer.py` from `full_plot.launch.py` must be the only publisher
controlling PX4. The calibration procedure below launches that same executable
from its dedicated launch file instead.

## 12. TFLite and ToF perception

TFLite detects object type from `hires_small_color`. ToF supplies distance
from `/tof_pc`. The output topic is
`/llm_vision/semantic_obstacles`, which is consumed by
`prompt_generator.py`.

### Frames and key parameters

Use camera optical frames for RGB/ToF projection and local NED for the final
obstacle coordinates. Image bounding boxes are matched to ToF points, converted
to body FRD with the camera mounts, then converted to NED with synchronized PX4
odometry. NED is x North, y East, z Down.

| Parameter | Reproduction value | Adjust only when |
| --- | --- | --- |
| `detection_camera` | `hires_small_color` | The TFLite input pipe changes |
| `hires_width`, `hires_height` | `1024`, `768` | The input resolution or crop changes |
| `hires_fx`, `hires_fy`, `hires_cx`, `hires_cy` | `501.5316`, `502.8287`, `508.1806`, `380.6556` | A new calibration is accepted for the exact stream and resolution |
| `point_cloud_frame` | `tof_optical` | Use `local_ned` only for an already transformed cloud |
| `detection_cam_body_*` | `[0.068, 0.012, -0.015]` m; RPY `[0, 90, 90]` deg | The RGB mount changes or is remeasured |
| `depth_cam_body_*` | `[0.066, 0.009, -0.012]` m; RPY `[0, 90, 180]` deg | The ToF mount changes or is remeasured |
| `detection_timeout_s`, `detector_timeout_s` | `10.0`, `10.0` | Allows bursty detector delivery during a static snapshot; lower them for motion |
| `point_cloud_timeout_s`, `pose_timeout_s` | `6.0`, `5.0` | Allows bursty ToF and pose delivery during a static snapshot; lower them for motion |
| `max_sync_slop_s` | `10.0` | Static-hover validation observed retained-pose gaps up to 5.318 s; lower it for motion |
| `pose_history_size` | `2000` | Covers about 16 s at the observed 123 Hz pose rate, exceeding the sync window |
| `min_confidence` | `0.70` | Raise it for false detections; lower it for missed detections |
| `min_tof_depth_m`, `max_tof_depth_m` | `0.20`, `6.0` | The usable ToF range changes |
| `bbox_inner_margin_fraction` | `0.30` | Increase it to reject box-edge background; decrease it when too few ToF points remain |
| `obstacle_hold_s` | `10.0` | Covers observed static-scene detector dropouts; lower it for moving obstacles |
| `held_depth_health_grace_s` | `10.0` | Reuses recent measured geometry only for synchronization failures; set to `0.0` for moving scenes |
| `depth_estimator` | `outline` | `shape_fit` (ToF only) or `front_spread` (legacy) for comparison |

### Object depth behind the front surface

The ToF sees only the surfaces facing the drone, so each obstacle also needs
its thickness along the line of sight. Perception measures it and publishes it
in every obstacle as `measured_depth_along_view_m`, with `depth_method`,
`depth_confidence` and `depth_ready`. The gateway extrudes the perceived front
segment (`front_surface_center`, `visible_width_m`, `view_axis_xy`) by this
depth (`nominal_depth_source: perception`, the default) instead of asking GPT
nano to guess it from the label (`nominal_depth_source: gpt`, the original
behaviour).

| `depth_estimator` | How | Cost on the ground station |
| --- | --- | --- |
| `outline` (default) | The `/hires_small_color` image is outlined by EfficientSAM, prompted with the TFLite box. The outline, the ToF points inside it and the ToF floor give the depth: tangent rays and a person-shaped ellipse for people (feet pick a wide or a long ellipse), floor contacts and footprints for furniture (`scripts/outline_depth.py`) | ~0.7 s segmentation + ~0.2 s math per object, on a worker thread |
| `shape_fit` | ToF points in the detector box only (`scripts/obstacle_depth.py`): an ellipse fit for people, the visible spread or a class prior for the rest | A few ms |
| `front_spread` | Legacy: p90-p10 of the visible front surface. Not a thickness, so the gateway treats it as missing | A few ms |

With `outline`, an obstacle seen for the first time carries the ToF-only
(`shape_fit`) depth and `depth_ready: false` until its outline is computed
(at most `outline_wait_s`, 8 s); calibration captures and mission requests
wait for it. An outline is reused for up to `outline_max_age_s` (10 s) unless
the object's ToF front moves by more than `outline_settle_m` (8 cm). Without
the model file, perception logs a warning and uses `shape_fit`;
`status.outline` in `/llm_vision/semantic_obstacles` says which is in use.

In flight, `outline` adds the `/hires_small_color` stream to the Wi-Fi link
to the ground station. If that link is already busy, set
`camera_image_topic: ""` (or `depth_estimator: shape_fit`) to keep the
ToF-only depths.

Footprints now use these measured depths, so a vision-error certificate
(`vision_error_calibration_csv`) built from a dataset with GPT nano depths
does not cover them: record the dataset again (Section 13, or 13.3.1 from an
existing bag) and rebuild the certificate before a conformal flight.

Stop signs have a hard-coded thickness of 10 mm (`FIXED_THICKNESS_M` in
`scripts/obstacle_depth.py`) instead of the conservative 1.5 m an unknown class
gets. The face is located with a plane fit, so ToF noise doesn't move it. A
sign square-on to the drone is exactly 10 mm deep along the line of sight;
seen at an angle, its face recedes along the line of sight
(width x sin(angle), e.g. 0.17 m for a 0.6 m sign 16 deg off-axis), and that
part is covered as well.

Download the EfficientSAM-Ti model (48 MB) once, then rebuild so it is
installed with the package:

```bash
cd ~/Desktop/starling_final_testing
curl -L -o src/llm_vision_planner/models/efficientsam_ti_2025april.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/image_segmentation_efficientsam/image_segmentation_efficientsam_ti_2025april.onnx
sha256sum src/llm_vision_planner/models/efficientsam_ti_2025april.onnx
# 4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77
python3 -c "import cv2, numpy; print(cv2.__version__)"   # OpenCV runs it; or: pip install onnxruntime
colcon build --packages-select llm_vision_planner
```

To check depths on a recorded bag before a flight (per-frame plots of the
outline, ToF, feet and boxes; Vicon truth when the bag has it):

```bash
python3 src/llm_vision_planner/scripts/outline_depth_from_bag.py <bag_folder> --label person \
  --segmenter efficientsam --align-tof report
```

`--align-tof report` also prints the small camera/ToF rotation that best
aligns the outline with the ToF; if several bags agree, put it in
`outline_tof_rotation_deg`.
