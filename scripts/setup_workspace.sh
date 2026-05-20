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

ignored_paths=(
  "$ROOT_DIR/src/voxl-mpa-to-ros2/colcon_ws/src/voxl_mpa_to_ros2"
  "$ROOT_DIR/src/voxl-mpa-to-ros2/colcon_ws/src/voxl_offboard_figure8"
)

for path in "${ignored_paths[@]}"; do
  if [ -d "$path" ]; then
    printf "Ignored by this workspace. See README.md.\n" > "$path/COLCON_IGNORE"
  fi
done

echo "Workspace dependencies are ready. Build with: colcon build"
