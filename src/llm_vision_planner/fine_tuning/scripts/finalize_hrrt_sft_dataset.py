#!/usr/bin/env python3
"""Merge teacher and human HRRT labels into scene-isolated SFT CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def describe_obstacles(obstacles: Sequence[Mapping[str, object]]) -> str:
    descriptions = []
    for index, obstacle in enumerate(obstacles, start=1):
        minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
        descriptions.append(
            f"{index} {obstacle['label']} ({obstacle['object_id']}): "
            f"x=[{minimum[0]:.2f},{maximum[0]:.2f}], y=[{minimum[1]:.2f},{maximum[1]:.2f}."
        )
    return " ".join(descriptions) if descriptions else "No obstacles."


def build_prompt(environment: Mapping[str, object], preference_text: str, clearance_m: float) -> str:
    start, goal, workspace = environment["start"], environment["goal"], environment["workspace"]
    distance = math.hypot(float(goal["x"]) - float(start["x"]), float(goal["y"]) - float(start["y"]))
    return (
        "You are an expert UAV path planner. Produce sparse fixed-altitude NED waypoints that satisfy the "
        "approved operator route preference while remaining geometrically feasible.\n"
        "Mission state: the UAV has already taken off and is holding hover at the start position. "
        f"Workspace: x=[{workspace['x'][0]:.2f},{workspace['x'][1]:.2f}]m, "
        f"y=[{workspace['y'][0]:.2f},{workspace['y'][1]:.2f}]m, z={workspace['z']:.2f} fixed. "
        f"Start: ({start['x']:.2f},{start['y']:.2f},{start['z']:.2f}); "
        f"goal: ({goal['x']:.2f},{goal['y']:.2f},{goal['z']:.2f}); distance={distance:.2f}m. "
        f"Obstacles: {describe_obstacles(environment['obstacles'])}\n"
        f"Approved route preference: {preference_text}\n"
        "Constraints:\n"
        "- first waypoint exactly equals start and final waypoint exactly equals goal\n"
        "- return between 2 and 8 sparse waypoints inside the workspace at fixed z\n"
        f"- maintain at least {clearance_m:.2f}m geometric clearance from obstacle boxes\n"
        "- satisfy the approved route preference without inventing objects\n"
        "- return only the structured output requested by the response model"
    )


def route_by_id(row: Mapping[str, object], route_id: str) -> Mapping[str, object]:
    for route in row["routes"]:
        if str(route["route_id"]) == route_id:
            return route
    raise ValueError(f"{row['sample_id']}: unknown route {route_id!r}")


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty split {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finalize(args: argparse.Namespace) -> None:
    raw_rows = read_jsonl(args.raw)
    manifest = {str(row["sample_id"]): row for row in read_jsonl(args.audit)}
    reviews = {str(row["sample_id"]): row for row in read_jsonl(args.reviews)}
    missing = sorted(set(manifest) - set(reviews))
    if missing:
        raise ValueError(f"Human review is incomplete; {len(missing)} assigned labels remain, first={missing[:5]}")

    scene_split: Dict[int, str] = {}
    for assignment in manifest.values():
        if assignment["review_split"] in ("validation", "test"):
            scene_id = int(assignment["scene_id"])
            split = str(assignment["review_split"])
            previous = scene_split.setdefault(scene_id, split)
            if previous != split:
                raise ValueError(f"scene {scene_id} appears in both {previous} and {split}")

    outputs: Dict[str, List[Dict[str, object]]] = {"train": [], "validation": [], "test": []}
    for raw in raw_rows:
        sample_id = str(raw["sample_id"])
        scene_id = int(raw["scene_id"])
        split = scene_split.get(scene_id, "train")
        assignment = manifest.get(sample_id)
        review = reviews.get(sample_id)
        if split in ("validation", "test") and (assignment is None or review is None):
            raise ValueError(f"{sample_id}: every held-out-scene row must be human reviewed")
        if review and review["decision"] == "ambiguous":
            continue
        if review and review["decision"] == "corrected":
            route_id = str(review["human_route_id"])
            label_source = "human_corrected"
        else:
            route_id = str(raw["teacher_route_id"])
            label_source = "human_accepted" if review else "teacher"
        route = route_by_id(raw, route_id)
        prompt = build_prompt(raw["environment"], str(raw["preference"]["text"]), args.clearance_m)
        completion = {
            "reasoning": "Selected a safe HRRT-star route that best matches the approved operator preference.",
            "waypoints": route["waypoints"],
        }
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": compact(completion)},
        ]
        outputs[split].append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "split": split,
                "label_source": label_source,
                "preference_type": raw["preference"]["preference_type"],
                "preference_text": raw["preference"]["text"],
                "selected_route_id": route_id,
                "prompt": prompt,
                "completion": compact(completion),
                "messages": compact(messages),
                "environment": compact(raw["environment"]),
                "route_cards": compact(raw["route_cards"]),
                "candidate_routes": compact(raw["routes"]),
                "human_review": compact(review) if review else "",
            }
        )

    write_csv(args.output_dir / "hrrt_sft_train.csv", outputs["train"])
    write_csv(args.output_dir / "hrrt_sft_validation.csv", outputs["validation"])
    write_csv(args.output_dir / "hrrt_sft_test.csv", outputs["test"])
    train_scenes = {row["scene_id"] for row in outputs["train"]}
    validation_scenes = {row["scene_id"] for row in outputs["validation"]}
    test_scenes = {row["scene_id"] for row in outputs["test"]}
    if train_scenes & validation_scenes or train_scenes & test_scenes or validation_scenes & test_scenes:
        raise AssertionError("Scene leakage detected after finalization")
    print(json.dumps({split: len(rows) for split, rows in outputs.items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("fine_tuning/datasets/hrrt_teacher_raw.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("fine_tuning/datasets/hrrt_human_audit.jsonl"))
    parser.add_argument("--reviews", type=Path, default=Path("fine_tuning/datasets/hrrt_human_reviews.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("fine_tuning/datasets"))
    parser.add_argument("--clearance-m", type=float, default=0.40)
    return parser


if __name__ == "__main__":
    finalize(build_parser().parse_args())
