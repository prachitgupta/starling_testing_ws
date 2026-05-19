#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
mkdir -p src

if ! command -v vcs >/dev/null 2>&1; then
  echo "vcs is required. Install python3-vcstool, then rerun this script." >&2
  exit 1
fi

vcs import src < dependencies.repos

PX4_ROS2_DIR="$ROOT_DIR/src/px4-ros2-interface-lib"
PATCH_FILE="$ROOT_DIR/patches/px4_ros2_voxl_unversioned_topics.patch"

if git -C "$PX4_ROS2_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "px4_ros2 VOXL patch already applied."
else
  git -C "$PX4_ROS2_DIR" apply "$PATCH_FILE"
fi

ignored_paths=(
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/executor_with_multiple_modes"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/fw_attitude"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/goto"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/goto_global"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/manual"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/mission"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/mode_with_executor"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/rover_velocity"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/rtl_replacement"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/modes/vtol"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/navigation/global_navigation"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/cpp/navigation/local_navigation"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/python/modes/goto"
  "$ROOT_DIR/src/px4-ros2-interface-lib/examples/python/modes/goto_with_rclpy"
  "$ROOT_DIR/src/px4-ros2-interface-lib/px4_ros2_py"
  "$ROOT_DIR/src/voxl-mpa-to-ros2/colcon_ws/src/voxl_mpa_to_ros2"
  "$ROOT_DIR/src/voxl-mpa-to-ros2/colcon_ws/src/voxl_offboard_figure8"
)

for path in "${ignored_paths[@]}"; do
  if [ -d "$path" ]; then
    printf "Ignored by this workspace. See README.md.\n" > "$path/COLCON_IGNORE"
  fi
done

echo "Workspace dependencies are ready. Build with: colcon build"
