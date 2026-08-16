#!/usr/bin/env python3
"""Focused tests for conformal obstacle geometry, calibration, Vicon, and tube gating."""

import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "fine_tuning" / "scripts"))

from interactive_mission_gateway import (  # noqa: E402
    InteractiveMissionGateway,
    MissionIntent,
    ObjectDepthEstimate,
    build_planner_prompt,
    conformal_quantile,
    containment_score,
    containment_score_center_extent,
    inflate_obstacles_xy,
    load_vision_error_certificate,
    nominal_obstacles_from_depth,
)
from verify_contraction import evaluate_swept_tube  # noqa: E402
from perception_detection import SemanticObstaclePerception  # noqa: E402
from vision_error_dataset_generattor import (  # noqa: E402
    CSV_FIELDS,
    ground_truth_aabb,
    quaternion_matrix,
)


def calibration_row(trial_id, score, missed=False, placeholder=False):
    return {
        "trial_id": trial_id,
        "timestamp_s": "0.0",
        "vicon_timestamp_s": "0.0",
        "object_id": "obj-1",
        "label": "chair",
        "pred_min_x": "" if missed else "0.0",
        "pred_min_y": "" if missed else "0.0",
        "pred_max_x": "" if missed else "1.0",
        "pred_max_y": "" if missed else "1.0",
        "gt_min_x": str(-float(score)),
        "gt_min_y": "0.0",
        "gt_max_x": "1.0",
        "gt_max_y": "1.0",
        "score_m": "" if missed else str(score),
        "missed_detection": str(bool(missed)).lower(),
        "placeholder": str(bool(placeholder)).lower(),
    }


def write_calibration(path, rows, fields=CSV_FIELDS):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_containment_score_captures_extent_error():
    ground_truth = {"min_corner": [2.0, -0.5], "max_corner": [3.0, 0.5]}
    predicted = {"min_corner": [2.1, -0.2], "max_corner": [2.5, 0.4]}
    predicted_center = [2.3, 0.1]
    ground_truth_center = [2.5, 0.0]
    midpoint_error = math.dist(predicted_center, ground_truth_center)
    score = containment_score(predicted, ground_truth)
    assert math.isclose(midpoint_error, math.sqrt(0.05))
    assert midpoint_error < score
    assert math.isclose(score, 0.5)
    assert math.isclose(score, containment_score_center_extent(predicted, ground_truth))
    midpoint_expanded = inflate_obstacles_xy([predicted], midpoint_error)[0]
    assert midpoint_expanded["max_corner"][0] < ground_truth["max_corner"][0]
    contained = inflate_obstacles_xy([predicted], score)[0]
    assert contained["min_corner"][0] <= ground_truth["min_corner"][0]
    assert contained["max_corner"][0] >= ground_truth["max_corner"][0]
    assert contained["min_corner"][1] <= ground_truth["min_corner"][1]
    assert contained["max_corner"][1] >= ground_truth["max_corner"][1]


def test_certificate_uses_independent_trial_maxima_and_fails_closed():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.csv"
        rows = [calibration_row(f"trial-{index}", 0.1 * index) for index in range(1, 10)]
        rows.append(calibration_row("trial-1", 1.1))
        write_calibration(path, rows)
        certificate = load_vision_error_certificate(path, 0.10)
        assert certificate["trial_count"] == 9
        assert certificate["rank"] == 9
        assert math.isclose(certificate["quantile_m"], 1.1)

        write_calibration(path, [calibration_row(f"trial-{index}", 0.1) for index in range(1, 9)])
        try:
            load_vision_error_certificate(path, 0.10)
        except ValueError as exc:
            assert "insufficient" in str(exc)
        else:
            raise AssertionError("insufficient independent trials were accepted")

        rows = [calibration_row(f"trial-{index}", 0.1) for index in range(1, 9)]
        rows.append(calibration_row("trial-9", 0.0, missed=True))
        write_calibration(path, rows)
        try:
            load_vision_error_certificate(path, 0.10)
        except ValueError as exc:
            assert "non-finite quantile" in str(exc)
        else:
            raise AssertionError("a selected infinite miss score was accepted")

        write_calibration(path, [calibration_row("trial-1", 0.1)], fields=CSV_FIELDS[:-1])
        try:
            load_vision_error_certificate(path, 0.10)
        except ValueError as exc:
            assert "missing calibration columns" in str(exc)
        else:
            raise AssertionError("a malformed calibration CSV was accepted")


def test_dummy_certificate_and_finite_sample_rank():
    certificate = load_vision_error_certificate(
        ROOT / "fine_tuning" / "datasets" / "calibration_vision_error_dummy.csv",
        0.10,
    )
    assert certificate["placeholder"] is True
    assert certificate["trial_count"] == 9
    assert math.isclose(certificate["quantile_m"], 0.35)
    assert conformal_quantile([0.1] * 9, 0.10) == (0.1, 9)


def test_placeholder_certificate_is_sim_only_and_hardcoded_is_preserved():
    scene = {
        "obstacles": [
            {
                "object_id": "obj-1",
                "label": "chair",
                "min_corner": [1.0, 1.0, -1.0],
                "max_corner": [2.0, 2.0, 0.0],
            }
        ]
    }
    intent = MissionIntent(
        status="READY",
        intent_type="NAVIGATION",
        navigation_action="GO_TO",
        query_type="NONE",
        clarifying_question="",
    )
    gateway = object.__new__(InteractiveMissionGateway)
    gateway.obs_safety_bracket = "conformal"
    gateway.scene_guard_band_m = 0.35
    gateway.vision_error_certificate = {"placeholder": True}
    gateway.environment = "sim"
    nominal, enlarged = gateway.environment_obstacles(scene, intent)
    assert nominal == scene["obstacles"]
    assert enlarged[0]["min_corner"][:2] == [0.65, 0.65]
    assert enlarged[0]["max_corner"][:2] == [2.35, 2.35]

    gateway.environment = "real"
    try:
        gateway.environment_obstacles(scene, intent)
    except ValueError as exc:
        assert "non-placeholder" in str(exc)
    else:
        raise AssertionError("a real mission accepted the placeholder certificate")

    gateway.obs_safety_bracket = "hardcoded"
    gateway.scene_guard_band_m = 0.25
    nominal, enlarged = gateway.environment_obstacles(scene, intent)
    assert nominal == scene["obstacles"]
    assert enlarged[0]["min_corner"][:2] == [0.75, 0.75]
    assert enlarged[0]["max_corner"][:2] == [2.25, 2.25]


def test_gpt_depth_builds_nominal_footprint_and_preserves_llama_format():
    observed = [
        {
            "object_id": "obj-1",
            "label": "chair",
            "front_surface_center": [1.0, 1.0, -0.5],
            "visible_width_m": 1.0,
            "front_range_m": 1.4,
            "view_axis_xy": [1.0, 0.0],
            "lateral_axis_xy": [0.0, 1.0],
            "min_corner": [0.9, 0.5, -1.0],
            "max_corner": [1.1, 1.5, 0.0],
        }
    ]
    estimates = [
        ObjectDepthEstimate(
            object_id="obj-1",
            effective_depth_along_view_m=0.5,
            abstained=False,
        )
    ]
    nominal = nominal_obstacles_from_depth(observed, estimates)
    assert nominal[0]["min_corner"] == [1.0, 0.5, -1.0]
    assert nominal[0]["max_corner"] == [1.5, 1.5, 0.0]
    enlarged = inflate_obstacles_xy(nominal, 0.2)
    prompt, _ = build_planner_prompt(
        {"x": 0.0, "y": 0.0, "z": -0.5},
        {"x": 3.0, "y": 3.0, "z": -0.5},
        {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": -0.5},
        enlarged,
        0.4,
    )
    assert "x=[0.80,1.70], y=[0.30,1.70" in prompt
    assert "front_surface_center" not in prompt
    assert "effective_depth_along_view_m" not in prompt

    for invalid in ([], [ObjectDepthEstimate(object_id="obj-1", effective_depth_along_view_m=None, abstained=True)]):
        try:
            nominal_obstacles_from_depth(observed, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("missing or abstaining GPT depth was accepted")


def test_perception_publishes_front_and_view_geometry_without_removing_legacy_box():
    perception = object.__new__(SemanticObstaclePerception)
    perception.detection_camera_translation_body = np.zeros(3)
    center = np.array([2.0, 1.0, -0.5])
    corners = np.array(
        [
            [1.9, 0.75, -1.0],
            [1.9, 1.25, -1.0],
            [2.1, 0.75, 0.0],
            [2.1, 1.25, 0.0],
        ]
    )
    payload = perception.obstacle_payload(
        center,
        corners,
        {"position": np.zeros(3), "rotation_body_to_world": np.eye(3)},
        "chair",
        0.9,
        "test",
        "test",
        None,
        "camera",
        center,
        0.5,
        math.sqrt(5.25),
    )
    assert "min_corner" in payload and "max_corner" in payload
    assert payload["front_surface_center"] == [2.0, 1.0, -0.5]
    assert payload["visible_width_m"] == 0.5
    assert np.allclose(payload["view_axis_xy"], [2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0)])
    assert math.isclose(
        payload["view_axis_xy"][0] * payload["lateral_axis_xy"][0]
        + payload["view_axis_xy"][1] * payload["lateral_axis_xy"][1],
        0.0,
        abs_tol=1e-6,
    )


def test_vicon_yaw_marker_offset_and_world_to_ned_transform():
    yaw_90 = quaternion_matrix([0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)])
    config = {
        "dimensions_m": np.array([2.0, 1.0, 0.9]),
        "marker_translation": np.array([1.0, 0.0, 0.0]),
        "marker_rotation": np.eye(3),
    }
    ground_truth = ground_truth_aabb(
        config,
        np.zeros(3),
        yaw_90,
        (np.array([10.0, 0.0, 0.0]), yaw_90),
    )
    assert np.allclose(ground_truth["center_xy"], [9.0, 0.0], atol=1e-9)
    assert np.allclose(ground_truth["min_corner"], [8.0, -0.5], atol=1e-9)
    assert np.allclose(ground_truth["max_corner"], [10.0, 0.5], atol=1e-9)


def test_swept_tube_rejects_intersection_and_tangency():
    obstacles = [
        {
            "object_id": "obj-1",
            "min_corner": [1.0, 1.0, -1.0],
            "max_corner": [2.0, 2.0, 0.0],
        }
    ]
    separated = evaluate_swept_tube([(0.0, 0.0), (0.7, 0.7)], obstacles, 0.2)
    tangent = evaluate_swept_tube([(0.0, 0.8), (0.8, 0.8)], obstacles, 0.2)
    intersecting = evaluate_swept_tube([(0.0, 1.5), (3.0, 1.5)], obstacles, 0.2)
    assert separated["passed"] is True
    assert separated["minimum_clearance_m"] > 0.0
    assert tangent["passed"] is False
    assert intersecting["passed"] is False
    assert intersecting["colliding_obstacle_ids"] == ["obj-1"]


if __name__ == "__main__":
    test_containment_score_captures_extent_error()
    test_certificate_uses_independent_trial_maxima_and_fails_closed()
    test_dummy_certificate_and_finite_sample_rank()
    test_placeholder_certificate_is_sim_only_and_hardcoded_is_preserved()
    test_gpt_depth_builds_nominal_footprint_and_preserves_llama_format()
    test_perception_publishes_front_and_view_geometry_without_removing_legacy_box()
    test_vicon_yaw_marker_offset_and_world_to_ned_transform()
    test_swept_tube_rejects_intersection_and_tangency()
    print("conformal obstacle safety tests passed")
