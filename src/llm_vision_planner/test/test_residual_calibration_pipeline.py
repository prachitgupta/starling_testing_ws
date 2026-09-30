#!/usr/bin/env python3
"""Offline checks for the two-stage residual calibration workflow."""

import argparse
import csv
import importlib.util
import math
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "fine_tuning" / "scripts" / "postprocess_residual_calibration.py"


def load_module():
    specification = importlib.util.spec_from_file_location("residual_calibration", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def raw_rows(module):
    common = {
        "session_id": "test-session",
        "capture_id": "test-session-capture-000001",
        "capture_index": "1",
        "timestamp_s": "1.0",
        "vicon_timestamp_s": "1.0",
        "missed_detection": "false",
        "stable_pose": "true",
        "observer_x": "-3.5",
        "observer_y": "0.0",
        "observer_z": "-0.5",
        "placeholder": "true",
    }
    obstacles = [
        ("person-1", "person", -1.45, -0.45, -0.65, 0.45, 0.08),
        ("chair-1", "chair", 0.65, -0.45, 1.45, 0.45, -0.06),
    ]
    rows = []
    for object_id, label, min_x, min_y, max_x, max_y, shift in obstacles:
        row = dict(common)
        row.update(
            {
                "object_id": object_id,
                "label": label,
                "pred_min_x": str(min_x + shift),
                "pred_min_y": str(min_y),
                "pred_max_x": str(max_x + shift),
                "pred_max_y": str(max_y),
                "gt_min_x": str(min_x),
                "gt_min_y": str(min_y),
                "gt_max_x": str(max_x),
                "gt_max_y": str(max_y),
                "gt_center_x": str(0.5 * (min_x + max_x)),
                "gt_center_y": str(0.5 * (min_y + max_y)),
            }
        )
        rows.append(row)
    assert set(rows[0]) == set(module.RAW_REQUIRED_FIELDS) | {
        "capture_index", "timestamp_s", "vicon_timestamp_s", "stable_pose",
        "gt_center_x", "gt_center_y", "placeholder",
    }
    return rows


def arguments():
    return argparse.Namespace(
        l1_text="Go to the goal while avoiding both obstacles.",
        l2_text="Stay far from the person.",
        expert_route_id="",
        expert_selector="heuristic",
        expert_model="",
        openai_api_key="",
        goal_x=3.5,
        goal_y=0.0,
        fixed_z=-0.5,
        start_x=-3.5,
        start_y=0.0,
        workspace_x_min=-4.0,
        workspace_x_max=4.0,
        workspace_y_min=-3.0,
        workspace_y_max=3.0,
        clearance_m=0.4,
        dt=0.1,
        seed=17,
        hrrt_iterations=500,
        max_candidates=8,
        max_waypoints=8,
        planner_provider="mock",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_api_key="EMPTY",
        llama_model="hrrt_planner",
        placeholder=True,
    )


def test_end_to_end():
    module = load_module()
    rows = raw_rows(module)
    record = module.process_capture(rows[0]["capture_id"], rows, arguments())
    assert list(record) == module.SCORED_FIELDS
    assert record["dynamics_model"] == "ideal_planar_double_integrator"
    assert record["planner_provider"] == "mock"
    assert record["placeholder"] == "true"
    assert math.isfinite(float(record["conformity_score"]))
    assert float(record["conformity_score"]) >= 0.0

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.csv"
        module.write_rows(path, [record], delimiter=",")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter=",")
            assert reader.fieldnames == module.SCORED_FIELDS
            assert len(list(reader)) == 1


if __name__ == "__main__":
    test_end_to_end()
    print("residual calibration pipeline: PASS")
