#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
mkdir -p src

if ! command -v vcs >/dev/null 2>&1; then
  echo "vcs is required. Install python3-vcstool, then rerun this script." >&2
  exit 1
fi

vcs import . < dependencies.repos

echo "Workspace dependencies are ready. Build with: colcon build"
