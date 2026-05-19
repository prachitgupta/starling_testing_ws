# Starling VOXL Testing Workspace

ROS 2 Humble workspace for testing VOXL/PX4 control paths and running the LLM verified trajectory follower.

The workspace is intentionally minimal. A normal build discovers only:

- `px4_msgs`
- `px4_ros2_cpp`
- `voxl_msgs`
- `starling_testing`
- `llm_vision_planner`

The PX4 ROS 2 interface library is patched during setup so it uses VOXL's unversioned PX4 topic names by default, for example `/fmu/out/vehicle_status` instead of `/fmu/out/vehicle_status_v1`.

## Fresh Ubuntu Setup

Use Ubuntu 22.04 with ROS 2 Humble. After installing ROS Humble, install the workspace tools and system dependencies:

```bash
sudo apt update
sudo apt install -y \
  git \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  libjsoncpp-dev
```

Initialize `rosdep` if it has not already been initialized on the machine:

```bash
sudo rosdep init
rosdep update
```

Then clone this repository and import the pinned dependencies:

```bash
git clone https://github.com/prachitgupta/starling_testing_ws.git
cd starling_testing_ws
source /opt/ros/humble/setup.bash
bash scripts/setup_workspace.sh
```

The setup script imports pinned dependency revisions from `dependencies.repos`, applies the VOXL topic-name patch to `px4_ros2_cpp`, and ignores packages that are not needed by this workspace.

Install any remaining ROS package dependencies:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

## Build

After setup, build normally:

```bash
cd starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

If your shell previously sourced another PX4 workspace, start a clean terminal before building or running. Source this workspace last:

```bash
source /opt/ros/humble/setup.bash
source ~/starling_testing_ws/install/setup.bash
```

## Required VOXL/PX4 Topics

The C++ px4_ros2 mode/executor path requires VOXL to bridge the external-mode registration and status topics. At minimum, the follower expects unversioned topics such as:

```text
/fmu/out/vehicle_status
/fmu/out/register_ext_component_reply
/fmu/out/arming_check_request
/fmu/out/vehicle_command_ack
/fmu/out/mode_completed
/fmu/out/vehicle_odometry
```

It publishes/uses inputs such as:

```text
/fmu/in/register_ext_component_request
/fmu/in/unregister_ext_component
/fmu/in/vehicle_command_mode_executor
/fmu/in/arming_check_reply
/fmu/in/goto_setpoint
/fmu/in/mode_completed
/fmu/in/config_control_setpoints
```

## Run Trajectory Follower

The target node tracks verified high-level plans on `/llm_vision/plan_verified`.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run llm_vision_planner trajectory_follower_cpp
```

No topic-version environment variable is needed. This workspace defaults to plain VOXL/PX4 topic names.

## Run Starling Smoke Tests

All positions are local NED: `x` north, `y` east, `z` down. For example, `z=-1.5` means 1.5 m above the local origin.

```bash
ros2 run starling_testing 01_takeoff_land.py --ros-args -p takeoff_alt_m:=1.5 -p hold_s:=5.0
ros2 run starling_testing 02_fmu_trajectory_waypoint.py --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p arm_and_offboard:=true
ros2 run starling_testing 03_px4_ros2_goto_waypoint.py --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p speed:=1.0
ros2 run starling_testing voxl_px4_ros2_goto_waypoint_cpp --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p speed:=1.0
ros2 run starling_testing voxl_px4_ros2_goto_mission_cpp --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p speed:=1.0 -p accept_m:=0.4
```

The full interactive sequence is:

```bash
ros2 run starling_testing run_voxl_sequence.sh
```

Override target and timing with environment variables:

```bash
TAKEOFF_ALT_M=1.5 HOLD_S=5.0 X=0.0 Y=2.0 Z=-1.5 SPEED=1.0 ACCEPT_M=0.4 \
TRAJECTORY_S=20 GOTO_PY_S=20 GOTO_CPP_S=20 \
  ros2 run starling_testing run_voxl_sequence.sh
```

## Dependency Notes

Only `voxl_msgs` is built from `voxl-mpa-to-ros2`; the bridge package and example figure-eight package are ignored.

Only `px4_ros2_cpp` is built from Auterion's PX4 ROS 2 interface library; examples and the Python binding package are ignored.

To run against a bridge that publishes versioned PX4 topics, set this before running nodes:

```bash
export PX4_ROS2_ENABLE_TOPIC_VERSION=1
```
