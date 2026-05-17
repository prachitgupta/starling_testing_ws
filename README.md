# Starling VOXL Testing Workspace

Minimal ROS 2 workspace containing only the VOXL/PX4 smoke tests from the original `voxl_testing` flow.

## Build

```bash
cd starling_testing_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select starling_testing
source install/setup.bash
```

The workspace expects `px4_msgs` and `px4_ros2_cpp` to already be available in the sourced ROS environment.

## Run Individual Tests

All positions are local NED: `x` north, `y` east, `z` down, so `z=-1.5` means 1.5 m above the local origin.

```bash
ros2 run starling_testing 01_takeoff_land.py --ros-args -p takeoff_alt_m:=1.5 -p hold_s:=5.0
ros2 run starling_testing 02_fmu_trajectory_waypoint.py --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p arm_and_offboard:=true
ros2 run starling_testing 03_px4_ros2_goto_waypoint.py --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p speed:=1.0
ros2 run starling_testing voxl_px4_ros2_goto_waypoint_cpp --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p speed:=1.0
ros2 run starling_testing voxl_px4_ros2_goto_mission_cpp --ros-args -p x:=0.0 -p y:=2.0 -p z:=-1.5 -p speed:=1.0 -p accept_m:=0.4
```

## Run The Sequence

```bash
ros2 run starling_testing run_voxl_sequence.sh
```

Override the default target and timing with environment variables:

```bash
TAKEOFF_ALT_M=1.5 HOLD_S=5.0 X=0.0 Y=2.0 Z=-1.5 SPEED=1.0 ACCEPT_M=0.4 \
TRAJECTORY_S=20 GOTO_PY_S=20 GOTO_CPP_S=20 \
  ros2 run starling_testing run_voxl_sequence.sh
```

The runner pauses before each test so you can verify the vehicle state before continuing. The trajectory and standalone goto tests are continuous ROS nodes, so the runner stops them after their configured duration.
