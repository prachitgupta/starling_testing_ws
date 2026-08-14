#!/usr/bin/env python3
"""Pure tests for interactive mission grounding and prompt construction."""

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from interactive_mission_gateway import (  # noqa: E402
    GoalRelation,
    MockIntentParser,
    build_planner_prompt,
    inflate_obstacles_xy,
    normalize_goal_relations,
    normalize_obstacles,
    range_constrained_goal,
    relation_results,
    safe_standoff_goal,
    scene_signature,
    scenes_compatible,
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


def test_multi_object_range_goal():
    obstacles = normalize_obstacles(scene())
    planning_obstacles = inflate_obstacles_xy(obstacles, 0.25)
    raw_relations = [
        GoalRelation(
            object_id="obj-1",
            object_label="chair",
            relation="NEAR",
            min_distance_m=0.70,
            max_distance_m=1.00,
            optimize="NONE",
        ),
        GoalRelation(
            object_id="obj-2",
            object_label="bottle",
            relation="FAR_FROM",
            min_distance_m=None,
            max_distance_m=None,
            optimize="MAXIMIZE",
        ),
    ]
    relations = normalize_goal_relations(
        raw_relations,
        obstacles,
        default_standoff=0.60,
        clearance=0.40,
        guard_band=0.25,
        default_range_half_width=0.15,
        exact_distance_tolerance=0.10,
    )
    start = {"x": 0.2, "y": 0.2, "z": -0.5}
    workspace = {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": -0.5}
    goal, results = range_constrained_goal(
        start,
        relations,
        planning_obstacles,
        workspace,
        fixed_z=-0.5,
        clearance=0.40,
        resolution=0.10,
    )
    assert all(item["satisfied"] for item in results)
    assert relation_results(goal, relations) == results
    assert next(item for item in results if item["object_label"] == "chair")["distance_m"] <= 1.0
    assert next(item for item in results if item["object_label"] == "bottle")["optimize"] == "MAXIMIZE"

    prompt, _ = build_planner_prompt(start, goal, workspace, planning_obstacles, 0.4, relations)
    assert "NEAR chair (obj-1)" in prompt
    assert "FAR_FROM bottle (obj-2)" in prompt


def test_mock_intent_queries_and_control_rejection():
    parser = MockIntentParser()
    catalog = [
        {"object_id": "obj-1", "label": "chair", "center": [1.65, 1.65, -0.5]},
        {"object_id": "obj-2", "label": "bottle", "center": [3.1, 3.1, -0.5]},
    ]
    query = parser.parse("What do you see?", catalog, [])
    assert query.intent_type == "QUERY"
    assert query.query_type == "DESCRIBE_SCENE"
    assert parser.parse("List objects", catalog, []).query_type == "LIST_OBJECTS"
    locate = parser.parse("Where is the chair?", catalog, [])
    assert locate.query_type == "LOCATE_OBJECT"
    assert locate.query_object_ids == ["obj-1"]
    assert parser.parse("Explain the proposal", catalog, []).query_type == "EXPLAIN_PROPOSAL"
    assert parser.parse("Why did it fail?", catalog, []).query_type == "EXPLAIN_FAILURE"

    navigation = parser.parse(
        "Hover near the chair and as far as possible from the bottle", catalog, []
    )
    assert navigation.status == "READY"
    assert [item.relation for item in navigation.relations] == ["NEAR", "FAR_FROM"]
    assert navigation.relations[1].optimize == "MAXIMIZE"

    control = parser.parse("Land now", catalog, [])
    assert control.status == "UNSUPPORTED"


def test_invalid_obstacle_rejected():
    try:
        normalize_obstacles({"obstacles": [{"id": 1, "label": "chair"}]})
    except ValueError as exc:
        assert "min_corner" in str(exc)
    else:
        raise AssertionError("invalid obstacle was accepted")


def test_live_scene_jitter_is_bounded_and_conservative():
    reference = normalize_obstacles(scene())
    current = normalize_obstacles(
        {
            "obstacles": [
                {
                    "id": 20,
                    "label": "bottle",
                    "min_corner": [3.04, 2.98, -1.0],
                    "max_corner": [3.26, 3.20, 0.0],
                },
                {
                    "id": 10,
                    "label": "chair",
                    "min_corner": [1.54, 1.47, -1.0],
                    "max_corner": [1.86, 1.81, 0.0],
                },
            ]
        }
    )
    compatible, reason = scenes_compatible(reference, current, 0.15, 0.20)
    assert compatible, reason

    inflated = inflate_obstacles_xy(reference, 0.25)
    chair_envelope = next(item for item in inflated if item["label"] == "chair")
    current_chair = next(item for item in current if item["label"] == "chair")
    assert chair_envelope["min_corner"][0] <= current_chair["min_corner"][0]
    assert chair_envelope["min_corner"][1] <= current_chair["min_corner"][1]
    assert chair_envelope["max_corner"][0] >= current_chair["max_corner"][0]
    assert chair_envelope["max_corner"][1] >= current_chair["max_corner"][1]


def test_unsafe_live_scene_changes_are_rejected():
    reference = normalize_obstacles(scene())
    moved = normalize_obstacles(scene())
    moved[0]["min_corner"][0] += 0.30
    moved[0]["max_corner"][0] += 0.30
    compatible, reason = scenes_compatible(reference, moved, 0.15, 0.20)
    assert not compatible
    assert "moved" in reason

    resized = normalize_obstacles(scene())
    resized[0]["max_corner"][0] += 0.25
    compatible, reason = scenes_compatible(reference, resized, 0.15, 0.20)
    assert not compatible
    assert "size changed" in reason

    relabelled = normalize_obstacles(scene())
    relabelled[0]["label"] = "bottle"
    compatible, reason = scenes_compatible(reference, relabelled, 0.15, 0.20)
    assert not compatible
    assert "missing obstacle label" in reason

    added = normalize_obstacles(scene()) + [
        {
            "object_id": "obj-3",
            "label": "person",
            "min_corner": [0.8, 0.8, -1.0],
            "max_corner": [1.0, 1.0, 0.0],
        }
    ]
    compatible, reason = scenes_compatible(reference, added, 0.15, 0.20)
    assert not compatible
    assert "count changed" in reason


if __name__ == "__main__":
    test_grounding_helpers()
    test_multi_object_range_goal()
    test_mock_intent_queries_and_control_rejection()
    test_invalid_obstacle_rejected()
    test_live_scene_jitter_is_bounded_and_conservative()
    test_unsafe_live_scene_changes_are_rejected()
    print("interactive mission grounding tests passed")
