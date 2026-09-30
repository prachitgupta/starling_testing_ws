#!/usr/bin/env python3
"""Fast contract tests for the staged HRRT fine-tuning data workflow."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "fine_tuning" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_hrrt_human_audit import build_manifest, read_jsonl  # noqa: E402
from evaluate_hrrt_adapter import evaluate  # noqa: E402
from finalize_hrrt_sft_dataset import finalize  # noqa: E402
from generate_hrrt_finetuning_dataset import prepare_resume_file  # noqa: E402


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def route(route_id, y):
    return {
        "route_id": route_id,
        "waypoints": [
            {"x": -2.0, "y": 0.0, "z": -0.5},
            {"x": 0.0, "y": y, "z": -0.5},
            {"x": 2.0, "y": 0.0, "z": -0.5},
        ],
    }


def raw_row(scene_id, preference_index):
    routes = [route("route-upper", 1.2), route("route-lower", -1.2)]
    environment = {
        "scene_id": scene_id,
        "start": {"x": -2.0, "y": 0.0, "z": -0.5},
        "goal": {"x": 2.0, "y": 0.0, "z": -0.5},
        "workspace": {"x": [-3.0, 3.0], "y": [-2.0, 2.0], "z": -0.5},
        "obstacles": [
            {
                "object_id": "chair-1",
                "label": "chair",
                "min_corner": [-0.2, -0.2, -1.0],
                "max_corner": [0.2, 0.2, 0.0],
            }
        ],
    }
    chosen = routes[preference_index % 2]
    return {
        "schema_version": "hrrt_sft_raw_v1",
        "sample_id": f"scene-{scene_id:06d}-pref-{preference_index:02d}",
        "scene_id": scene_id,
        "environment": environment,
        "preference": {"preference_type": "side", "text": "Use the clearer side."},
        "route_cards": [
            {"route_id": item["route_id"], "path_length_m": 4.5, "estimated_duration_s": 9.0, "overall_minimum_clearance_m": 1.0}
            for item in routes
        ],
        "routes": routes,
        "teacher_route_id": chosen["route_id"],
        "teacher_reason": "fixture",
        "selected_waypoints": chosen["waypoints"],
    }


def test_resume_removes_partial_scene(tmp):
    output = tmp / "raw.jsonl"
    rows = [raw_row(0, 0), raw_row(0, 1), raw_row(1, 0)]
    write_jsonl(output, rows)
    assert prepare_resume_file(output, 2) == {0}
    retained = read_jsonl(output)
    assert len(retained) == 2 and {row["scene_id"] for row in retained} == {0}


def test_human_review_and_finalization(tmp):
    raw = tmp / "raw.jsonl"
    audit = tmp / "audit.jsonl"
    reviews = tmp / "reviews.jsonl"
    output_dir = tmp / "final"
    rows = [raw_row(scene, preference) for scene in range(10) for preference in range(2)]
    write_jsonl(raw, rows)
    build_manifest(
        SimpleNamespace(
            input=raw,
            output=audit,
            train_audit_ratio=0.10,
            validation_scene_ratio=0.20,
            test_scene_ratio=0.20,
            seed=42,
        )
    )
    assignments = read_jsonl(audit)
    reviews_rows = []
    by_id = {row["sample_id"]: row for row in rows}
    for assignment in assignments:
        sample = by_id[assignment["sample_id"]]
        reviews_rows.append(
            {
                "sample_id": assignment["sample_id"],
                "decision": "accepted",
                "human_route_id": sample["teacher_route_id"],
            }
        )
    write_jsonl(reviews, reviews_rows)
    finalize(SimpleNamespace(raw=raw, audit=audit, reviews=reviews, output_dir=output_dir, clearance_m=0.40))
    train = (output_dir / "hrrt_sft_train.csv").read_text(encoding="utf-8")
    validation = (output_dir / "hrrt_sft_validation.csv").read_text(encoding="utf-8")
    test = (output_dir / "hrrt_sft_test.csv").read_text(encoding="utf-8")
    assert "Approved route preference: Use the clearer side." in train
    assert "human_accepted" in validation and "human_accepted" in test


def test_offline_evaluation_metrics():
    row = raw_row(0, 0)
    completion = {"reasoning": "fixture", "waypoints": row["selected_waypoints"]}
    finalized = {
        "sample_id": row["sample_id"],
        "selected_route_id": row["teacher_route_id"],
        "environment": json.dumps(row["environment"]),
        "candidate_routes": json.dumps(row["routes"]),
        "completion": json.dumps(completion),
    }
    report = evaluate([finalized], {row["sample_id"]: json.dumps(completion)}, 0.40)
    assert report["metrics"]["structured_valid_rate"] == 1.0
    assert report["metrics"]["feasible_rate"] == 1.0
    assert report["metrics"]["route_match_rate"] == 1.0
    invalid = evaluate(
        [finalized],
        {row["sample_id"]: json.dumps({"reasoning": "invalid", "waypoints": []})},
        0.40,
    )
    assert invalid["metrics"]["structured_valid_rate"] == 0.0


def main():
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_resume_removes_partial_scene(tmp)
    with tempfile.TemporaryDirectory() as directory:
        test_human_review_and_finalization(Path(directory))
    test_offline_evaluation_metrics()
    print("HRRT fine-tuning pipeline test passed")


if __name__ == "__main__":
    main()
