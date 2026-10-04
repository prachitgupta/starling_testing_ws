#!/usr/bin/env python3
"""Merge teacher and human HRRT labels into scene-isolated SFT CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from generate_hrrt_finetuning_dataset import (
    DISTILLATION_MODES,
    card_by_id,
    deterministic_trajectory_reasoning,
    student_reasoning_is_valid,
)


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


def build_scene_context(environment: Mapping[str, object], preference_text: str) -> str:
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
        f"Approved route preference: {preference_text}"
    )


def waypoint_constraints(clearance_m: float) -> str:
    return (
        "Constraints:\n"
        "- first waypoint exactly equals start and final waypoint exactly equals goal\n"
        "- return between 2 and 8 sparse waypoints inside the workspace at fixed z\n"
        f"- maintain at least {clearance_m:.2f}m geometric clearance from obstacle boxes\n"
        "- satisfy the approved route preference without inventing objects\n"
        "- reasoning must use: Preference: ... Geometry: ... Decision: ... Safety: ...\n"
        "- return only the structured output requested by the response model"
    )


def build_prompt(environment: Mapping[str, object], preference_text: str, clearance_m: float) -> str:
    return "\n".join(
        (
            build_scene_context(environment, preference_text),
            waypoint_constraints(clearance_m),
        )
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


def selected_for_auxiliary(sample_id: str, ratio: float, seed: int, salt: str) -> bool:
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    digest = hashlib.sha256(f"{seed}:{salt}:{sample_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < ratio


def reasoning_prompt(
    environment: Mapping[str, object], preference_text: str, clearance_m: float
) -> str:
    return (
        build_scene_context(environment, preference_text)
        + f"\nSafety requirement: maintain at least {clearance_m:.2f}m clearance from obstacle boxes."
        + "\nAuxiliary DSS task: explain the intended route using "
        "Preference: ... Geometry: ... Decision: ... Safety: ... "
        "Return only a JSON object containing the reasoning field; do not return waypoints."
    )


def reasoning_conditioned_prompt(
    environment: Mapping[str, object],
    preference_text: str,
    clearance_m: float,
    reasoning: str,
) -> str:
    return (
        build_scene_context(environment, preference_text)
        + "\n"
        + waypoint_constraints(clearance_m).replace(
            "- reasoning must use: Preference: ... Geometry: ... Decision: ... Safety: ...\n",
            "",
        )
        + "\nAuxiliary SCOTT consistency task: the following rationale is authoritative for this task. "
        "Generate the waypoint sequence that follows it. Return only a JSON object containing waypoints.\n"
        + f"Supplied rationale: {reasoning}"
    )


def completed_row(
    raw: Mapping[str, object],
    review: Mapping[str, object] | None,
    split: str,
    label_source: str,
    distillation_mode: str,
    sample_id: str,
    task_type: str,
    route_id: str,
    prompt: str,
    completion: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "sample_id": sample_id,
        "parent_sample_id": raw["sample_id"],
        "scene_id": raw["scene_id"],
        "split": split,
        "distillation_mode": distillation_mode,
        "task_type": task_type,
        "label_source": label_source,
        "preference_type": raw["preference"]["preference_type"],
        "preference_text": raw["preference"]["text"],
        "selected_route_id": route_id,
        "prompt": prompt,
        "completion": compact(completion),
        "messages": compact(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": compact(completion)},
            ]
        ),
        "environment": compact(raw["environment"]),
        "route_cards": compact(raw["route_cards"]),
        "candidate_routes": compact(raw["routes"]),
        "human_review": compact(review) if review else "",
    }


def finalize(args: argparse.Namespace) -> None:
    raw_rows = read_jsonl(args.raw)
    if not 0.0 <= args.reasoning_aux_ratio <= 1.0:
        raise ValueError("--reasoning-aux-ratio must be between 0 and 1")
    if not 0.0 <= args.counterfactual_aux_ratio <= 1.0:
        raise ValueError("--counterfactual-aux-ratio must be between 0 and 1")
    raw_modes = {str(row.get("distillation_mode", "dss")) for row in raw_rows}
    if raw_modes != {args.distillation_mode}:
        raise ValueError(
            f"Raw data modes {sorted(raw_modes)} do not match --distillation-mode {args.distillation_mode!r}"
        )
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
        human_reason = str(
            (review or {}).get("human_reason") or (review or {}).get("notes") or ""
        ).strip()
        teacher_reason = raw.get("teacher_trajectory_reasoning") or raw.get("teacher_reason")
        if human_reason and student_reasoning_is_valid(human_reason):
            reason = human_reason
        elif route_id == str(raw["teacher_route_id"]) and student_reasoning_is_valid(teacher_reason):
            reason = str(teacher_reason).strip()
        else:
            reason = deterministic_trajectory_reasoning(
                raw["preference"], card_by_id(raw["route_cards"], route_id)
            )
        completion = {
            "reasoning": reason,
            "waypoints": route["waypoints"],
        }
        outputs[split].append(
            completed_row(
                raw,
                review,
                split,
                label_source,
                args.distillation_mode,
                sample_id,
                "full_plan",
                route_id,
                prompt,
                completion,
            )
        )
        if split != "train":
            continue
        if selected_for_auxiliary(
            sample_id, args.reasoning_aux_ratio, args.auxiliary_seed, "reasoning"
        ):
            outputs[split].append(
                completed_row(
                    raw,
                    review,
                    split,
                    label_source,
                    args.distillation_mode,
                    sample_id + "::reasoning",
                    "reasoning",
                    route_id,
                    reasoning_prompt(
                        raw["environment"], str(raw["preference"]["text"]), args.clearance_m
                    ),
                    {"reasoning": reason},
                )
            )
        if (
            args.distillation_mode == "dss_scott"
            and not (review and review["decision"] == "corrected")
            and selected_for_auxiliary(
                sample_id,
                args.counterfactual_aux_ratio,
                args.auxiliary_seed,
                "counterfactual",
            )
        ):
            counterfactual_id = str(raw["counterfactual_route_id"])
            counterfactual_route = route_by_id(raw, counterfactual_id)
            counterfactual_reason = str(raw["counterfactual_trajectory_reasoning"])
            if not student_reasoning_is_valid(counterfactual_reason):
                counterfactual_reason = deterministic_trajectory_reasoning(
                    raw["preference"],
                    card_by_id(raw["route_cards"], counterfactual_id),
                    counterfactual=True,
                )
            conditioned_pairs = (
                ("conditioned_positive", route_id, reason, route["waypoints"]),
                (
                    "conditioned_counterfactual",
                    counterfactual_id,
                    counterfactual_reason,
                    counterfactual_route["waypoints"],
                ),
            )
            for task_type, target_route_id, supplied_reason, waypoints in conditioned_pairs:
                outputs[split].append(
                    completed_row(
                        raw,
                        review,
                        split,
                        label_source,
                        args.distillation_mode,
                        sample_id + "::" + task_type,
                        task_type,
                        target_route_id,
                        reasoning_conditioned_prompt(
                            raw["environment"],
                            str(raw["preference"]["text"]),
                            args.clearance_m,
                            supplied_reason,
                        ),
                        {"waypoints": waypoints},
                    )
                )

    write_csv(args.output_dir / "hrrt_sft_train.csv", outputs["train"])
    write_csv(args.output_dir / "hrrt_sft_validation.csv", outputs["validation"])
    write_csv(args.output_dir / "hrrt_sft_test.csv", outputs["test"])
    train_scenes = {row["scene_id"] for row in outputs["train"]}
    validation_scenes = {row["scene_id"] for row in outputs["validation"]}
    test_scenes = {row["scene_id"] for row in outputs["test"]}
    if train_scenes & validation_scenes or train_scenes & test_scenes or validation_scenes & test_scenes:
        raise AssertionError("Scene leakage detected after finalization")
    task_counts: Dict[str, int] = {}
    for rows in outputs.values():
        for row in rows:
            task = str(row["task_type"])
            task_counts[task] = task_counts.get(task, 0) + 1
    print(
        json.dumps(
            {
                "distillation_mode": args.distillation_mode,
                "split_counts": {split: len(rows) for split, rows in outputs.items()},
                "task_counts": task_counts,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("fine_tuning/datasets/hrrt_teacher_raw.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("fine_tuning/datasets/hrrt_human_audit.jsonl"))
    parser.add_argument("--reviews", type=Path, default=Path("fine_tuning/datasets/hrrt_human_reviews.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("fine_tuning/datasets"))
    parser.add_argument("--clearance-m", type=float, default=0.40)
    parser.add_argument("--distillation-mode", choices=DISTILLATION_MODES, default="dss")
    parser.add_argument(
        "--reasoning-aux-ratio",
        type=float,
        default=0.25,
        help="Fraction of training examples duplicated as DSS reasoning-only tasks.",
    )
    parser.add_argument(
        "--counterfactual-aux-ratio",
        type=float,
        default=0.25,
        help="DSS-SCOTT fraction receiving positive/counterfactual rationale-conditioned plan tasks.",
    )
    parser.add_argument("--auxiliary-seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    finalize(build_parser().parse_args())
