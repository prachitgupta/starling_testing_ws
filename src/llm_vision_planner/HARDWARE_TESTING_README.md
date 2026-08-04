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

### Optional: isolate ROS 2 while using the campus VPN

Use this section only while campus Ethernet reaches the GPU through a VPN. ROS
2 remains on the existing Wi-Fi path between the laptop and VOXL; Vicon
DataStream also remains on Wi-Fi. The GPU does not need this DDS profile unless
it runs ROS 2 directly.

The profile is opt-in and terminal-local. It allows Fast DDS UDP only on
loopback and the selected Wi-Fi IPv4 address, excluding campus Ethernet and VPN
interfaces. It does not modify NetworkManager, routes, `.bashrc`, or systemd.

#### One-time setup

On the laptop, build the package and copy the helper to the VOXL:

```bash
cd ~/Desktop/starling_testing_ws
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

#### Enable on both sides

In every new laptop terminal that starts MAVROS, the Vicon bridge, the planner,
or ROS CLI commands:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_testing_ws/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable wlp6s0 0
```

In every VOXL shell that manually starts a ROS 2 process:

```bash
source /opt/ros/foxy/setup.bash
source /data/llm_vision_planner/scripts/ros_wifi_dds.sh enable wlan0 0
```

Both sides must report `rmw_fastrtps_cpp` and domain `0`. Stop and restart any
ROS process that was already running before `enable`. Re-run `enable` and
restart the processes if Wi-Fi reconnects with a different IPv4 address.

This shell command does not alter an already-running VOXL systemd service. Check
the installed service names and environments before changing a vendor service:

```bash
systemctl show voxl-microdds-agent voxl-mpa-to-ros2 \
  --property=LoadState,ActiveState,Environment
```

`LoadState=not-found` means that service name is not installed. Do not create a
systemd override until the connected vehicle's actual service has been
identified.

#### Connect the VPN now

On the laptop, connect Cisco Secure Client using `1 Split Tunnel`
(`OpenConnect1 (Split)` on Linux). Do not use `2 Tunnel All`.

#### Verify routes after connecting the VPN

On the laptop, replace the example with the GPU server's campus IPv4 address:

```bash
export GPU_IP="10.0.0.10"

source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  status
printenv RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE

ip route get "$Starling2"          # must show wlp6s0
ip route get "$VICON_COMPUTER_IP" # must show wlp6s0
ip route get "$GPU_IP"            # must show the VPN interface
```

Confirm that Wi-Fi is not shared with or bridged to campus Ethernet:

```bash
nmcli -t -f NAME,TYPE,DEVICE connection show --active
WIFI_CONNECTION_NAME="$(nmcli -g GENERAL.CONNECTION device show wlp6s0)"
nmcli -g ipv4.method,ipv6.method connection show "$WIFI_CONNECTION_NAME"
ip -d link show type bridge
bridge link
```

Neither Wi-Fi method may be `shared`, and Wi-Fi and campus Ethernet must not be
members of the same bridge. If the VPN sends either Starling or Vicon traffic
through its tunnel, stop and have the campus VPN route narrowed before flight.

On the VOXL:

```bash
source /data/llm_vision_planner/scripts/ros_wifi_dds.sh status
printenv RMW_IMPLEMENTATION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY \
  FASTRTPS_DEFAULT_PROFILES_FILE
ip -4 -br address show wlan0
```

With the VPN connected, repeat the normal topic checks on the laptop:

```bash
ros2 topic info -v /fmu/out/vehicle_odometry
ros2 topic hz /fmu/out/vehicle_odometry
ros2 topic hz /mavros/vision_pose/pose --window 500
```

Stop each `ros2 topic hz` command with `Ctrl-C` after observing a stable rate.
The Vicon maximum interval must remain below `0.2 s`, as required later in this
procedure. Do not fly if topics disappear, rates fall, or either local route
moves to the VPN.

#### Undo when the VPN is no longer used

Stop ROS processes started under the profile. In every affected laptop shell:

```bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  disable
```

In every affected VOXL shell:

```bash
source /data/llm_vision_planner/scripts/ros_wifi_dds.sh disable
```

`disable` restores the environment that existed before `enable` and deletes
only that shell's generated runtime XML. Closing the terminal has the same
environment-reset effect. The copied VOXL helper is inert when it is not
sourced; optionally remove only those two copied files:

```bash
ssh root@"$Starling2" \
  'rm -f /data/llm_vision_planner/scripts/ros_wifi_dds.sh /data/llm_vision_planner/config/fastdds_wifi_only.xml.in'
```

After campus registration, disconnect the VPN and verify that the GPU uses
campus Ethernet while the local paths remain on Wi-Fi:

```bash
ip route get "$Starling2"          # wlp6s0
ip route get "$VICON_COMPUTER_IP" # wlp6s0
ip route get "$GPU_IP"            # campus Ethernet interface
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

## Launching Llama on the GPU

Run these commands on the GPU in the same shell.

1. Check GPU processes:

```bash
for i in $(seq 0 $(($(nvidia-smi -L | wc -l)-1))); do
    echo "===== GPU $i ====="
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $i | while read pid mem; do
        pid=$(echo $pid | tr -d ',')
        mem=$(echo $mem | tr -d ' MiB,')
        user=$(ps -p $pid -o user= 2>/dev/null || echo "unknown")
        cmd=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
        echo "$user | PID: $pid | Memory: $mem MiB | Process: $cmd"
    done
done
```

2. Configure the adapter:

```bash
ADAPTER=/home/prachit2/starling_testing_ws/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_rrt_lora
```

3. Launch the LLM:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-lora-rank 128 \
  --lora-modules rrt_planner=$ADAPTER \
  --served-model-name rrt_planner \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --port 8000
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
source ~/Desktop/starling_testing_ws/install/setup.bash

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
~/Desktop/starling_testing_ws/src/llm_vision_planner/params/vicon_voxl.params
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
| `max_sync_slop_s` | `0.35` | Use the smallest value that still fuses RGB and ToF; lower it for motion |
| `min_confidence` | `0.60` | Raise it for false detections; lower it for missed detections |
| `min_tof_depth_m`, `max_tof_depth_m` | `0.20`, `6.0` | The usable ToF range changes |
| `bbox_inner_margin_fraction` | `0.25` | Increase it to reject box-edge background; decrease it when too few ToF points remain |

### Calibrate the hires camera

Use the grey stream paired with the TFLite color stream:
`hires_small_grey`. Do not use a tracking-camera pipe.

The successful board had 5x6 internal corners and 30 mm squares. Keep it flat,
well lit, and sharp.

Run on VOXL:

```bash
voxl-inspect-services
voxl-list-pipes | grep -E '^hires_small_(color|grey)$'

voxl-set-cpu-mode perf
systemctl stop voxl-tflite-server voxl-qvio-server voxl-tag-detector voxl-dfs-server voxl-streamer 2>/dev/null || true
systemctl restart voxl-camera-server voxl-portal
```

Reduce exposure under bright lighting:

```bash
voxl-send-command hires_small_grey set_exp_gain 3.0 400
```

Open the calibration overlay in VOXL Portal, then run:

```bash
voxl-calibrate-camera hires_small_grey -s 5x6 -l 0.030
```

The accepted calibration had a 0.703523 px reprojection error and was saved to:

```text
/data/modalai/opencv_hires_small_grey_intrinsics.yml
```

Check and back it up:

```bash
sed -n '1,100p' /data/modalai/opencv_hires_small_grey_intrinsics.yml
cp -p /data/modalai/opencv_hires_small_grey_intrinsics.yml \
  /data/modalai/opencv_hires_small_grey_intrinsics.yml.accepted
```

Restore automatic exposure and perception services:

```bash
voxl-send-command hires_small_grey start_ae
systemctl restart voxl-camera-server
systemctl start voxl-qvio-server voxl-tflite-server
voxl-inspect-services | grep -E 'camera|qvio|tflite|mpa-to-ros2'
```

### Build

Run on the ground station:

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_vision_planner
source install/setup.bash
```

### Run perception only

Terminal 1:

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 run llm_vision_planner perception_detection.py --ros-args \
  --params-file ~/Desktop/starling_testing_ws/src/llm_vision_planner/config/llm_vision_planner.yaml
```

Terminal 2:

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 topic echo --full-length /llm_vision/semantic_obstacles
```

### Open the live perception plot

```bash
source ~/Desktop/starling_testing_ws/install/setup.bash
ros2 run llm_vision_planner debug_perception.py
```

This only subscribes and plots. It does not save an image or send flight
commands.

Save a plot only when needed:

```bash
ros2 run llm_vision_planner debug_perception.py --ros-args \
  -p output_png:=/tmp/debug_perception.png
```

### Launch the real mission

This command starts real flight control. Run it only when the vehicle is ready
to fly.

```bash
cd ~/Desktop/starling_testing_ws
source install/setup.bash

ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  environment:=real \
  llm_provider:=llama \
  visualizer:=contraction \
  show_rrt:=true
```

Use the RC kill switch or change PX4/QGroundControl mode to abort.
