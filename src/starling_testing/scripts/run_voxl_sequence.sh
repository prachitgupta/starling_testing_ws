#!/usr/bin/env bash
set -euo pipefail

TAKEOFF_ALT_M="${TAKEOFF_ALT_M:-1.5}"
HOLD_S="${HOLD_S:-5.0}"
X="${X:-0.0}"
Y="${Y:-2.0}"
Z="${Z:--1.5}"
SPEED="${SPEED:-1.0}"
ACCEPT_M="${ACCEPT_M:-0.4}"
MOVE_AFTER_S="${MOVE_AFTER_S:-3.0}"
TRAJECTORY_S="${TRAJECTORY_S:-20}"
GOTO_PY_S="${GOTO_PY_S:-20}"
GOTO_CPP_S="${GOTO_CPP_S:-20}"

pause() {
  printf "\nReady for %s. Press Enter to continue, or Ctrl-C to stop.\n" "$1"
  read -r _
}

run_for() {
  local duration_s="$1"
  shift
  set +e
  timeout --preserve-status "$duration_s" "$@"
  local status=$?
  set -e

  if [ "$status" -eq 143 ]; then
    printf "Stopped after %s seconds.\n" "$duration_s"
    return 0
  fi
  return "$status"
}

pause "01_takeoff_land.py"
ros2 run starling_testing 01_takeoff_land.py --ros-args \
  -p takeoff_alt_m:="$TAKEOFF_ALT_M" \
  -p hold_s:="$HOLD_S"

pause "02_fmu_trajectory_waypoint.py"
run_for "$TRAJECTORY_S" ros2 run starling_testing 02_fmu_trajectory_waypoint.py --ros-args \
  -p x:="$X" \
  -p y:="$Y" \
  -p z:="$Z" \
  -p arm_and_offboard:=true \
  -p move_after_s:="$MOVE_AFTER_S"

pause "03_px4_ros2_goto_waypoint.py"
run_for "$GOTO_PY_S" ros2 run starling_testing 03_px4_ros2_goto_waypoint.py --ros-args \
  -p x:="$X" \
  -p y:="$Y" \
  -p z:="$Z" \
  -p speed:="$SPEED"

pause "voxl_px4_ros2_goto_waypoint_cpp"
run_for "$GOTO_CPP_S" ros2 run starling_testing voxl_px4_ros2_goto_waypoint_cpp --ros-args \
  -p x:="$X" \
  -p y:="$Y" \
  -p z:="$Z" \
  -p speed:="$SPEED"

pause "voxl_px4_ros2_goto_mission_cpp"
ros2 run starling_testing voxl_px4_ros2_goto_mission_cpp --ros-args \
  -p x:="$X" \
  -p y:="$Y" \
  -p z:="$Z" \
  -p speed:="$SPEED" \
  -p accept_m:="$ACCEPT_M"
