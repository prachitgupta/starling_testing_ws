#!/usr/bin/env python3
"""Pure tests for interactive mission grounding and prompt construction."""

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from interactive_mission_gateway import (  # noqa: E402
    build_planner_prompt,
    normalize_obstacles,
    safe_standoff_goal,
    scene_signature,
)


def scene():
    return {
        "obstacles": [
            {
                "id": 1,
                "label": "chair",
                "min_corner": [1.5, 1.5, -1.0],
                "max_corner": [1.8, 1.8, 0.0],
                "size": [0.3, 0.3, 1.0],
            },
            {
                "id": 2,
                "label": "bottle",
                "min_corner": [3.0, 3.0, -1.0],
                "max_corner": [3.2, 3.2, 0.0],
                "size": [0.2, 0.2, 1.0],
            },
        ]
    }


def test_grounding_helpers():
    obstacles = normalize_obstacles(scene())
    assert [item["object_id"] for item in obstacles] == ["obj-1", "obj-2"]
    assert scene_signature(obstacles) == scene_signature(normalize_obstacles(scene()))

    start = {"x": 0.2, "y": 0.2, "z": -0.5}
    workspace = {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": -0.5}
    goal = safe_standoff_goal(
        start,
        obstacles[0],
        obstacles,
        workspace,
        fixed_z=-0.5,
        requested_standoff=0.0,
        default_standoff=0.6,
        clearance=0.4,
    )
    assert goal["z"] == -0.5
    assert math.hypot(goal["x"] - start["x"], goal["y"] - start["y"]) > 0.0

    prompt, nl_env = build_planner_prompt(start, goal, workspace, obstacles, 0.4)
    assert "chair (obj-1)" in prompt
    assert "bottle (obj-2)" in prompt
    assert "maintain >=0.40m clearance" in prompt
    assert nl_env in prompt


def test_invalid_obstacle_rejected():
    try:
        normalize_obstacles({"obstacles": [{"id": 1, "label": "chair"}]})
    except ValueError as exc:
        assert "min_corner" in str(exc)
    else:
        raise AssertionError("invalid obstacle was accepted")


if __name__ == "__main__":
    test_grounding_helpers()
    test_invalid_obstacle_rejected()
    print("interactive mission grounding tests passed")
