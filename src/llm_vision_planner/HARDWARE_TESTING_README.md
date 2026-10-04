# Starling 2 hardware reproduction with HRRT-star

Branch: `starling_multiple_trajectory_idea`. This guide starts from the verified
`reporoduce_hardeware` procedure and changes only the planner and operator workflow.
It covers a dummy-obstacle simulation, the interactive gateway, both gateway modes,
and the complete Starling 2 hardware mission.

In adaptive conformal prediction mode, HRRT-star generates a finite route family
and the operator describes the preferred route in natural language. In mission
execution mode, the second route-selection stage is skipped. In both modes the
local DSS or DSS-SCOTT Llama adapter produces the online plan, which is refined,
verified, optimized, and displayed with the contraction tube.

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
git clone --depth 1 --single-branch --branch starling_multiple_trajectory_idea \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/starling_multiple_trajectory_idea
cd ~/Desktop/starling_multiple_trajectory_idea
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
use the separate-clone instructions at the end and replace the workspace path
in this guide. Source `/opt/ros/humble/setup.bash` and this clone's
`install/setup.bash` in every ground-station ROS terminal. Continuous topic
monitors and service/launch commands each need their own terminal; stop a
monitor with `Ctrl+C` before running the next monitor command.

Follow Sections 1–10 for hardware setup, Section 11 for simulated-obstacle
gateway checks, Section 12 for the real TFLite/ToF mission, and Section 13 for
the two-stage residual-calibration data workflow.

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
cd ~/Desktop/starling_multiple_trajectory_idea
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
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
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
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
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

## 3.1 Launching the HRRT-trained Llama adapter on the GPU

Run these commands on the GPU server in the same shell. Use its existing vLLM
environment, or create a separate one with `python3 -m venv ~/vllm_env`, activate
it, and install `vllm`. Authenticate with `hf auth login` using an account that
has access to the base model.

No adapter weights are bundled. Choose `dss` or `dss_scott`; the corresponding
`fine_tuning/outputs/llama31_8b_hrrt_lora_<mode>/PLACEHOLDER.txt` marks the
expected directory. Replace the placeholder with the trained adapter on the GPU
server before continuing.

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
cd ~/Desktop/starling_multiple_trajectory_idea
DISTILLATION_MODE=dss
LLAMA_MODEL_NAME="hrrt_planner_$DISTILLATION_MODE"
ADAPTER="$PWD/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_hrrt_lora_$DISTILLATION_MODE"
test -s "$ADAPTER/adapter_config.json" && test -s "$ADAPTER/adapter_model.safetensors"
```

3. Launch the LLM:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-lora-rank 128 \
  --lora-modules "$LLAMA_MODEL_NAME=$ADAPTER" \
  --served-model-name "$LLAMA_MODEL_NAME" \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --port 8000
```

The checks above must pass; the placeholder cannot serve predictions. If the
GPU clone is elsewhere, use its actual absolute adapter path.

On the ground station, check the server (replace the lab address for your host):

```bash
export VLLM_BASE_URL=http://172.22.224.93:8000/v1
curl --fail --silent --show-error "$VLLM_BASE_URL/models"
```

The response must list the chosen alias, such as `hrrt_planner_dss`. In
`src/llm_vision_planner/config/llm_vision_planner.yaml`, set
`llm_planner.ros__parameters.vllm_base_url` to that URL; exporting the variable
alone does not override the ROS parameter. Pass the served alias through the
`llama_model_name` launch argument; the launch file applies it consistently to
the planner, prompt generator, and interactive gateway.
For calibration and interactive missions, also export a valid `OPENAI_API_KEY`
in the launch terminal. The camera mounts/intrinsics and network addresses below
describe the original vehicle and must match your hardware.

## 4. Start MAVROS

On the ground station:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
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
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
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
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
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
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
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
~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/params/vicon_voxl.params
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
cd ~/Desktop/starling_multiple_trajectory_idea
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

## 11. Interactive gateway with simulated obstacles

Start PX4 simulation first. Use separate terminals for the launch, obstacle
publisher, and monitors.

### 11.1 Terminal 1: publish dummy obstacles

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic pub -r 2 /llm_vision/sim_obstacles std_msgs/msg/String \
  "{data: '{\"healthy\":true,\"frame\":\"local_ned\",\"obstacles\":[{\"id\":1,\"label\":\"chair\",\"shape\":\"box\",\"min_corner\":[0.70,0.60,-0.75],\"max_corner\":[1.10,1.00,0.25],\"confidence\":1.0},{\"id\":2,\"label\":\"person\",\"shape\":\"box\",\"min_corner\":[-0.70,0.80,-0.75],\"max_corner\":[-0.30,1.20,0.25],\"confidence\":1.0}],\"timestamp\":0.0}'}"
```

### 11.2 Terminal 2: launch the interactive gateway once

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
source install/setup.bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
LLAMA_MODEL_NAME=hrrt_planner_dss

ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  environment:=sim \
  interaction_mode:=interactive \
  intent_provider:=openai \
  use_dataset_scene:=false \
  llm_provider:=llama \
  llama_model_name:="$LLAMA_MODEL_NAME" \
  visualizer:=contraction \
  web_ui_host:=127.0.0.1
```

Open `http://127.0.0.1:8080`.

1. Use the mode buttons in the left panel to select **Adaptive conformal** or
   **Mission execution**.
2. In Adaptive conformal mode, enter: `Fly beyond the chair while staying away
   from the person.` Approve the goal, wait for the colored HRRT-star family,
   then enter: `Choose a short route that stays far from the person.`
3. In Mission execution mode, enter and approve the goal. The gateway skips the
   second HRRT-star preference question.
4. Review the verified plan and contraction tube, then approve or terminate.

### 11.3 Terminal 3: monitor the pipeline

```bash
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
ros2 topic echo /llm_vision/mission_state
ros2 topic echo /llm_vision/hrrt_route_candidates
ros2 topic echo /llm_vision/prompt
ros2 topic echo /llm_vision/plan_raw
ros2 topic echo /llm_vision/plan_refined
ros2 topic echo /llm_vision/plan_candidate_verified
ros2 topic echo /llm_vision/plan_verified
```

For a fully offline interface check, replace `intent_provider:=openai` with
`intent_provider:=mock`. The Llama server is still required when the selected
mission reaches online planning.

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
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_vision_planner
source install/setup.bash
```

### Run perception only

Terminal 1:

```bash
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
ros2 run llm_vision_planner perception_detection.py --ros-args \
  --params-file ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/config/llm_vision_planner.yaml
```

Terminal 2:

```bash
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
ros2 topic echo --full-length /llm_vision/semantic_obstacles
```

### Record all perception topics from a remote machine

Use a separate Ubuntu 22.04/ROS 2 Humble computer on the same flight network so
point-cloud recording does not consume resources on the flight computer or the
ground station. The recorder must use the same DDS option and ROS domain as the
vehicle and ground station. The commands below assume Option A and domain 42.

Run once on the remote recorder:

```bash
sudo apt update
sudo apt install -y git rsync python3-colcon-common-extensions python3-rosdep python3-vcstool \
  ros-humble-rosbag2 ros-humble-rosbag2-storage-mcap

mkdir -p ~/Desktop
git clone --depth 1 --single-branch --branch starling_multiple_trajectory_idea \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/starling_multiple_trajectory_idea
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
bash scripts/setup_workspace.sh
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select px4_msgs voxl_msgs llm_vision_planner
```

If that clone already exists, pull and rebuild it instead of cloning again.
Before each recording, start the vehicle bridges and the perception node, then
run this on the remote recorder:

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon stop
ros2 daemon start

printenv ROS_DOMAIN_ID RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY
ros2 topic list | sort | grep -E \
  '^(/tof_pc|/voa_pc_out|/tflite|/tflite_data|/fmu/out/vehicle_odometry|/llm_vision/(semantic_obstacles|obstacles)|/tf|/tf_static)$'
ros2 topic hz /tof_pc
ros2 topic hz /llm_vision/semantic_obstacles
```

Both `ros2 topic hz` commands must receive messages. Stop each check with
`Ctrl+C`. If `/tof_pc` is missing, do not start the experiment: the bag would not
contain the raw depth data required for later point-cloud work.

Record raw sensor inputs, synchronized pose/frames, the optional VOA cloud, and
both custom obstacle outputs:

```bash
mkdir -p ~/rosbags
STAMP="$(date +%Y%m%d-%H%M%S)"
BAG_DIR="$HOME/rosbags/perception-$STAMP"

ros2 bag record --storage mcap --output "$BAG_DIR" \
  /tof_pc \
  /tflite_data \
  /tflite \
  /fmu/out/vehicle_odometry \
  /tf \
  /tf_static \
  /voa_pc_out \
  /llm_vision/semantic_obstacles \
  /llm_vision/obstacles
```

Leave this terminal running for the entire scene or flight. Press `Ctrl+C` once
to stop and allow rosbag2 to finish its metadata. Do not power off or disconnect
the recorder while it is closing the bag. Topics not published by a particular
configuration may remain at zero messages; `/tof_pc`, `/tflite_data`,
`/fmu/out/vehicle_odometry`, and `/llm_vision/semantic_obstacles` are required for
this perception pipeline.

Verify the recording before moving or deleting it:

```bash
ros2 bag info "$BAG_DIR"
du -sh "$BAG_DIR"
```

The bag information must show non-zero message counts for the four required
topics. Copy the complete bag directory, including `metadata.yaml` and every
`.mcap` file, from the recorder to the analysis computer:

```bash
rsync -av --progress "$BAG_DIR/" \
  USER@ANALYSIS_COMPUTER:~/rosbags/"$(basename "$BAG_DIR")"/
```

Replace `USER@ANALYSIS_COMPUTER` with the analysis computer's SSH login and
network address. When using Option B DDS, omit `ros_wifi_dds.sh enable` and use
the Option B environment from Section 1 on the recorder as well.

### Open the live perception plot

```bash
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
ros2 run llm_vision_planner debug_perception.py
```

This only subscribes and plots. It does not save an image or send flight
commands.

Save a plot only when needed:

```bash
ros2 run llm_vision_planner debug_perception.py --ros-args \
  -p output_png:=/tmp/debug_perception.png
```

### Launch the real mission gateway

This command starts real flight control. Run it only when the vehicle is ready
to fly.

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
LLAMA_MODEL_NAME=hrrt_planner_dss

ros2 launch llm_vision_planner full_plot.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  environment:=real \
  interaction_mode:=interactive \
  intent_provider:=openai \
  llm_provider:=llama \
  llama_model_name:="$LLAMA_MODEL_NAME" \
  visualizer:=contraction \
  web_ui_host:=0.0.0.0
```

Use the RC kill switch or change PX4/QGroundControl mode to abort.

From another device on the same network, open
`http://<ground-station-lan-ip>:8080` once `/llm_vision/mission_state` reports
`HOLDING_FOR_PLAN`. The `0.0.0.0` value is the server bind address, not the
address to type into the browser. Select either mode from the left panel before
submitting the first mission prompt. See Section 11 for the approval walkthrough.
Add `land_after_complete:=false` to the launch command to hold the final goal
instead of landing automatically.

## 13. Generate the residual calibration dataset

This workflow is deliberately split into two stages. The flight records only
comma-delimited perceived/ground-truth obstacle geometry. Each accepted capture
is an atomic environment: every configured Vicon object is written under one
`capture_id`, including Vicon yaw and four NED corners. Detected objects also
include the perceived front center, view/lateral axes, visible width, ChatGPT
depth, and four reconstructed NED footprint corners. Offline processing
then runs HRRT-star on ground truth, the Llama planner on perception, both
ideal-double-integrator QPs, and writes one residual conformity score per
capture. Never use the raw CSV as the conformal calibration input.

### 13.1 Check the supplied dummy files

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
source install/setup.bash

RAW_DUMMY="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw_dummy.csv"
SCORED_DUMMY="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_scored_dummy.csv"

head -n 3 "$RAW_DUMMY"
head -n 3 "$SCORED_DUMMY"

python3 src/llm_vision_planner/fine_tuning/scripts/postprocess_residual_calibration.py \
  --raw-csv "$RAW_DUMMY" \
  --output-csv /tmp/calibration_residual_scored_smoke.csv \
  --delimiter ',' \
  --l1-text "Fly to the point beyond the chair." \
  --l2-text "Use a short route while staying away from the person." \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --fixed-z -0.5 \
  --planner-provider mock \
  --expert-selector heuristic \
  --all-captures \
  --placeholder

head -n 3 /tmp/calibration_residual_scored_smoke.csv
```

The dummy rows have `placeholder=true`. They verify the schema only and must
not be mixed with real calibration data.

### 13.2 Start the all-object Vicon bridge

Stop any other Vicon bridge first. Run this on the ground station:

```bash
export VICON_COMPUTER_IP=10.117.229.124

ros2 run vicon_bridge vicon_bridge --ros-args \
  -p host_name:="${VICON_COMPUTER_IP}:801" \
  -p stream_mode:="ServerPush" \
  -p update_rate_hz:=125.0 \
  -p expected_rate_hz:=50.0 \
  -p publish_specific_segment:=false \
  -p world_frame_id:="vicon_world" \
  -p tf_namespace:="vicon" \
  -r /vicon/Starling2/Starling2/pose:=/mavros/vision_pose/pose
```

In another terminal, verify every required stream:

```bash
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
ros2 topic echo /vicon/Starling2/Starling2 --once
ros2 topic echo /vicon/chair1/chair1 --once
ros2 topic echo /vicon/person/person --once
ros2 topic echo /vicon/stopsign/stopsign --once
ros2 topic echo /fmu/out/vehicle_odometry --once
```

### 13.3 Record raw hardware captures

This launch owns Offboard control: do not run `full_plot.launch.py` or another
Offboard publisher at the same time. Keep QGroundControl land/kill controls
available. The vehicle takes off, holds its initial X/Y position, and records
only after frame alignment reports `FRAME_READY`.

In the launch terminal:

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start

read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
read -rp "Chair X width in metres: " CHAIR_WIDTH_M
read -rp "Chair Y depth in metres: " CHAIR_DEPTH_M
read -rp "Person target X width in metres: " PERSON_WIDTH_M
read -rp "Person target Y depth in metres: " PERSON_DEPTH_M
read -rp "Stop-sign target X width in metres: " STOPSIGN_WIDTH_M
read -rp "Stop-sign target Y depth in metres: " STOPSIGN_DEPTH_M

export CHAIR_WIDTH_M CHAIR_DEPTH_M PERSON_WIDTH_M PERSON_DEPTH_M \
  STOPSIGN_WIDTH_M STOPSIGN_DEPTH_M
VICON_OBJECTS_JSON="$(python3 - <<'PY'
import json
import os

print(json.dumps([
    {
        "object_id": "chair-1",
        "label": "chair",
        "topic": "/vicon/chair1/chair1",
        "dimensions_m": [float(os.environ["CHAIR_WIDTH_M"]), float(os.environ["CHAIR_DEPTH_M"])],
    },
    {
        "object_id": "person-1",
        "label": "person",
        "topic": "/vicon/person/person",
        "dimensions_m": [float(os.environ["PERSON_WIDTH_M"]), float(os.environ["PERSON_DEPTH_M"])],
    },
    {
        "object_id": "stop-sign-1",
        "label": "stop sign",
        "topic": "/vicon/stopsign/stopsign",
        "dimensions_m": [float(os.environ["STOPSIGN_WIDTH_M"]), float(os.environ["STOPSIGN_DEPTH_M"])],
    },
]))
PY
)"

RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv"
TRIAL_ID="residual-$(date +%Y%m%d-%H%M%S)"
POSITION_CHANGE_M=0.15
YAW_CHANGE_RAD=0.261799  # 15 degrees

ros2 launch llm_vision_planner vision_error_calibration.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  trial_id:="$TRIAL_ID" \
  output_csv:="$RAW_CSV" \
  vicon_objects_json:="$VICON_OBJECTS_JSON" \
  capture_position_change_threshold_m:="$POSITION_CHANGE_M" \
  capture_yaw_change_threshold_rad:="$YAW_CHANGE_RAD"
```

In a monitor terminal:

```bash
source ~/Desktop/starling_multiple_trajectory_idea/install/setup.bash
ros2 topic echo /llm_vision/mission_state --once
ros2 topic echo /llm_vision/vision_calibration_status --once
tail -f \
  ~/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv
```

After each `RECORDED` status, move at least one tracked obstacle by
`POSITION_CHANGE_M` or rotate it by `YAW_CHANGE_RAD`, then leave every object
stationary. The recorder waits for all configured Vicon objects and writes the
complete environment together. `SKIPPED_DUPLICATE_ENVIRONMENT` means neither
threshold was reached; `SKIPPED_MOVING_OBJECT` means the scene was not stable.
Do not move the aircraft by hand while it holds position.

Land before stopping the launch:

```bash
ros2 topic pub --once /llm_vision/executor_command std_msgs/msg/String \
  "{data: '{\"command\":\"LAND\",\"reason\":\"residual calibration complete\"}'}"
ros2 topic echo /llm_vision/mission_state
```

Wait for `COMPLETE` and verify that the vehicle is landed and disarmed. Then
press `Ctrl+C` in the launch terminal.

### 13.4 Validate the raw comma-delimited CSV

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv"

python3 - "$RAW_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter=","))
required = {
    "session_id", "capture_id", "object_id", "label",
    "pred_min_x", "pred_min_y", "pred_max_x", "pred_max_y",
    "gt_min_x", "gt_min_y", "gt_max_x", "gt_max_y", "placeholder",
    "gt_yaw_rad",
    *(f"gt_corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")),
    "pred_front_center_x", "pred_front_center_y",
    "pred_view_axis_x", "pred_view_axis_y",
    "pred_lateral_axis_x", "pred_lateral_axis_y",
    "pred_visible_width_m", "pred_chatgpt_depth_m",
    *(f"pred_corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")),
}
if not rows:
    raise SystemExit("No raw calibration rows were recorded.")
missing = required.difference(rows[0])
if missing:
    raise SystemExit(f"Missing raw CSV columns: {sorted(missing)}")
if any(row["placeholder"].lower() != "false" for row in rows):
    raise SystemExit("Real raw data contains a placeholder row.")
captures = {}
for row in rows:
    captures.setdefault(row["capture_id"], []).append(row)
if any(len({row["object_id"] for row in group}) != len(group) for group in captures.values()):
    raise SystemExit("A capture contains duplicate object rows.")
for row in rows:
    if row["missed_detection"].lower() == "false":
        perceived = [name for name in required if name.startswith("pred_")]
        if any(not row[name] for name in perceived):
            raise SystemExit(f"Detected object is missing perceived geometry: {row['object_id']}")
print(f"raw rows: {len(rows)}")
print(f"independent captures: {len(captures)}")
PY
```

### 13.5 Compute the scored calibration CSV offline

Section 3.1 must already be serving the trained adapter under its selected alias.
Each command scores only the captures selected for that `L1`/`L2`/goal batch.
The Llama adapter plans on perceived geometry; ChatGPT selects the expert HRRT
route on ground-truth geometry. If a human already selected an exact displayed
route, replace `--expert-selector openai` with
`--expert-route-id ROUTE_ID_FROM_GATEWAY`.

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
source /opt/ros/humble/setup.bash
source install/setup.bash

RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv"
CALIBRATION_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_scored.csv"
export VLLM_BASE_URL=http://172.22.224.93:8000/v1
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
export EXPERT_MODEL=gpt-5.4
LLAMA_MODEL_NAME=hrrt_planner_dss

curl --fail --silent --show-error "$VLLM_BASE_URL/models"

python3 src/llm_vision_planner/fine_tuning/scripts/postprocess_residual_calibration.py \
  --raw-csv "$RAW_CSV" \
  --list-captures

python3 src/llm_vision_planner/fine_tuning/scripts/postprocess_residual_calibration.py \
  --raw-csv "$RAW_CSV" \
  --output-csv "$CALIBRATION_CSV" \
  --delimiter ',' \
  --l1-text "Fly to the point beyond the chair." \
  --l2-text "Use a short route while staying as far from the person as possible." \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --fixed-z -0.5 \
  --workspace-x-min -4.0 \
  --workspace-x-max 4.0 \
  --workspace-y-min -3.0 \
  --workspace-y-max 3.0 \
  --clearance-m 0.4 \
  --dt 0.1 \
  --hrrt-iterations 500 \
  --max-candidates 8 \
  --max-waypoints 8 \
  --sample-count 25 \
  --sample-seed 17 \
  --expert-selector openai \
  --expert-model "$EXPERT_MODEL" \
  --planner-provider vllm \
  --vllm-base-url "$VLLM_BASE_URL" \
  --vllm-api-key EMPTY \
  --llama-model "$LLAMA_MODEL_NAME" \
  --append
```

Run the command again with a different `L1`/`L2`/goal and either a different
`--sample-seed` or repeated `--capture-id CAPTURE_ID` flags. `--append` builds
one calibration CSV and rejects duplicate `(capture_id, L1, L2)` labels. Use
`--all-captures` only when applying one variant to every capture intentionally.

For an exact batch, replace `--sample-count 25 --sample-seed 17` with:

```bash
--capture-id residual-YYYYMMDD-HHMMSS-capture-000003 \
--capture-id residual-YYYYMMDD-HHMMSS-capture-000011
```

### 13.6 Validate the scored CSV

```bash
cd ~/Desktop/starling_multiple_trajectory_idea
CALIBRATION_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_scored.csv"

python3 - "$CALIBRATION_CSV" <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter=","))
required = {
    "capture_id", "l1_text", "l2_text", "expert_selector", "expert_model", "expert_route_id",
    "ground_truth_environment_json", "perceived_environment_json",
    "conformity_score", "dynamics_model", "planner_provider", "placeholder",
}
if not rows:
    raise SystemExit("No scored calibration rows were generated.")
missing = required.difference(rows[0])
if missing:
    raise SystemExit(f"Missing scored CSV columns: {sorted(missing)}")
scores = [float(row["conformity_score"]) for row in rows]
if not all(math.isfinite(score) and score >= 0.0 for score in scores):
    raise SystemExit("A conformity score is invalid.")
if any(row["placeholder"].lower() != "false" for row in rows):
    raise SystemExit("Real scored data contains a placeholder row.")
print(f"scored captures: {len(rows)}")
print(f"score range: [{min(scores):.6f}, {max(scores):.6f}]")
PY

head -n 2 "$CALIBRATION_CSV"
```

Use `calibration_residual_scored.csv`, not the raw file or either dummy file,
when the adaptive conformal initial-radius loader is enabled.

## Working with the branches

Keep each branch in a separate clone so its `build/`, `install/`, and `log/`
directories cannot mix with another branch. A new shell should source only the
workspace being used. To inspect or update either workflow:

```bash
git clone --depth 1 --single-branch --branch starling_multiple_trajectory_idea \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/hrrt_hardware_ws
git -C ~/Desktop/hrrt_hardware_ws branch --show-current
git -C ~/Desktop/hrrt_hardware_ws pull --ff-only
```

For this alternate path, substitute `~/Desktop/hrrt_hardware_ws` for
`~/Desktop/starling_multiple_trajectory_idea` in this README, then install dependencies,
build, and source inside that clone.
`main` retains the complete original workspace. Do not merge an isolated branch
into `main` just to use it. To move an adapter or completed calibration between
branches, copy only that artifact to the matching path in the other clone.
