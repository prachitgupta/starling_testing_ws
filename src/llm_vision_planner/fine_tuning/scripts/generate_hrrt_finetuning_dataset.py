#!/usr/bin/env python3
"""Generate resumable HRRT-star teacher-label records for supervised tuning."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Sequence

from hrrt_star import normalize_obstacles, plan_hrrt_star, segment_clear


WORKSPACE = {"x": [-4.0, 4.0], "y": [-3.0, 3.0], "z": -0.5}
LABELS = ("person", "chair", "backpack", "bottle", "potted_plant", "bench", "stop_sign")
SIZES = {
    "person": (0.55, 0.55),
    "chair": (0.65, 0.65),
    "backpack": (0.45, 0.35),
    "bottle": (0.30, 0.30),
    "potted_plant": (0.60, 0.60),
    "bench": (1.00, 0.55),
    "stop_sign": (0.45, 0.45),
}
SCENE_PROFILES = {
    # Each profile includes a central blocker, so a trivial start-goal segment
    # cannot dominate the dataset. Remaining offsets create different corridor
    # shapes while random jitter prevents exact duplicates.
    "central_blockade": ((0.0, 0.0), (-1.35, 0.65), (1.35, -0.65), (0.75, 1.15)),
    "staggered_slalom": ((0.0, 0.0), (-1.45, -0.75), (1.35, 0.75), (0.70, -1.10)),
    "narrow_gate": ((0.0, 0.0), (-0.65, 1.20), (-0.65, -1.20), (1.40, 0.65)),
    "upper_cluster": ((0.0, 0.0), (-1.25, 0.90), (1.10, 1.05), (1.65, -0.55)),
    "lower_cluster": ((0.0, 0.0), (-1.25, -0.90), (1.10, -1.05), (1.65, 0.55)),
    "split_corridor": ((0.0, 0.0), (-1.55, 0.80), (1.55, 0.80), (0.85, -1.05)),
}


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(compact(record) + "\n")


def prepare_resume_file(path: Path, expected_per_scene: int) -> set[int]:
    """Drop partial scenes so resuming never duplicates or silently skips labels."""
    by_scene: Dict[int, List[Dict[str, object]]] = {}
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                by_scene.setdefault(int(row["scene_id"]), []).append(row)
    complete = {
        scene_id
        for scene_id, rows in by_scene.items()
        if len(rows) == expected_per_scene
        and len({str(row["sample_id"]) for row in rows}) == expected_per_scene
    }
    retained = [row for scene_id in sorted(complete) for row in by_scene[scene_id]]
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in retained:
            stream.write(compact(row) + "\n")
    temporary.replace(path)
    return complete


def sample_scene(rng: random.Random, scene_id: int, obstacle_count: int) -> Dict[str, object]:
    profile_name = tuple(SCENE_PROFILES)[scene_id % len(SCENE_PROFILES)]
    profile = SCENE_PROFILES[profile_name]
    start_y = rng.uniform(-1.45, 1.45)
    goal_y = rng.uniform(-1.45, 1.45)
    start = {"x": -3.45, "y": round(start_y, 4), "z": -0.5}
    goal = {"x": 3.45, "y": round(goal_y, 4), "z": -0.5}
    labels = list(LABELS)
    rng.shuffle(labels)
    obstacles = []
    for index in range(obstacle_count):
        label = labels[index % len(labels)]
        size_scale = rng.uniform(0.85, 1.15)
        width, depth = (value * size_scale for value in SIZES[label])
        base_x, lateral_offset = profile[index]
        center_x = base_x + rng.uniform(-0.18, 0.18)
        progress = (center_x - start["x"]) / (goal["x"] - start["x"])
        centerline_y = start_y + progress * (goal_y - start_y)
        center_y = centerline_y + lateral_offset + rng.uniform(-0.16, 0.16)
        center_y = min(2.25, max(-2.25, center_y))
        obstacles.append(
            {
                "object_id": f"{label}-{index + 1}",
                "label": label,
                "min_corner": [round(center_x - width / 2, 4), round(center_y - depth / 2, 4), -1.0],
                "max_corner": [round(center_x + width / 2, 4), round(center_y + depth / 2, 4), 0.0],
            }
        )
    return {
        "scene_id": scene_id,
        "scene_profile": profile_name,
        "start": start,
        "goal": goal,
        "workspace": WORKSPACE,
        "obstacles": obstacles,
    }


def route_cards(
    routes: Sequence[Mapping[str, object]],
    obstacles: Sequence[Mapping[str, object]],
    nominal_speed_mps: float,
) -> List[Dict[str, object]]:
    cards = []
    for route in routes:
        clearances = {str(key): float(value) for key, value in route["minimum_clearance_m"].items()}
        signature = list(route.get("h_signature", []))
        bins = list(route.get("clearance_bins", []))
        descriptions = []
        for index, obstacle in enumerate(obstacles):
            side = {-1: "above", 0: "directly past", 1: "below"}.get(
                int(signature[index]) if index < len(signature) else 0,
                "around",
            )
            clearance_name = {0: "close", 1: "medium-clearance", 2: "wide-clearance"}.get(
                int(bins[index]) if index < len(bins) else 0,
                "clear",
            )
            descriptions.append(
                f"{side} the {obstacle['label']} ({obstacle['object_id']}) with {clearance_name} spacing"
            )
        cards.append(
            {
                "route_id": route["route_id"],
                "route_summary": "; ".join(descriptions),
                "path_length_m": float(route["path_length_m"]),
                "estimated_duration_s": round(float(route["path_length_m"]) / nominal_speed_mps, 3),
                "minimum_clearance_m": clearances,
                "overall_minimum_clearance_m": round(min(clearances.values()), 4) if clearances else math.inf,
            }
        )
    return cards


def preference_requests(scene: Mapping[str, object], cards: Sequence[Mapping[str, object]]) -> List[Dict[str, str]]:
    obstacles = list(scene["obstacles"])
    variant = int(scene["scene_id"]) % 4
    shortest_text = (
        "Choose the shortest safe route to the goal.",
        "Take the quickest collision-free way to the destination.",
        "Prefer the safe path with the least travel distance.",
        "Reach the goal efficiently without sacrificing safety.",
    )[variant]
    widest_text = (
        "Keep as much overall distance from obstacles as possible.",
        "Give every obstacle the widest practical berth.",
        "Prefer maximum clearance over a shorter trip.",
        "Use the most conservative route around the scene.",
    )[variant]
    balanced_text = (
        "Balance a short travel time with comfortable obstacle clearance.",
        "Choose a reasonable compromise between speed and spacing.",
        "Avoid a long detour, but do not pass unnecessarily close to objects.",
        "Use a moderate route that balances efficiency and clearance.",
    )[variant]
    requests = [
        {"preference_type": "shortest", "text": shortest_text},
        {"preference_type": "widest", "text": widest_text},
        {"preference_type": "balanced", "text": balanced_text},
    ]
    if obstacles:
        obstacle = obstacles[0]
        requests.append(
            {
                "preference_type": "far_from_object",
                "target_object_id": str(obstacle["object_id"]),
                "text": (
                    f"Stay as far away from the {obstacle['label']} ({obstacle['object_id']}) as possible.",
                    f"Give the {obstacle['label']} ({obstacle['object_id']}) extra space.",
                    f"Maximize separation from the {obstacle['label']} ({obstacle['object_id']}).",
                    f"Take the route furthest from the {obstacle['label']} ({obstacle['object_id']}).",
                )[variant],
            }
        )
    if len(obstacles) > 1:
        obstacle = obstacles[1]
        requests.append(
            {
                "preference_type": "close_to_object",
                "target_object_id": str(obstacle["object_id"]),
                "text": (
                    f"Use the safe route that passes closer to the {obstacle['label']} ({obstacle['object_id']}).",
                    f"Without violating clearance, stay near the {obstacle['label']} ({obstacle['object_id']}).",
                    f"Prefer the safe corridor beside the {obstacle['label']} ({obstacle['object_id']}).",
                    f"Pass the {obstacle['label']} ({obstacle['object_id']}) on the nearer safe option.",
                )[variant],
            }
        )
    durations = sorted(float(card["estimated_duration_s"]) for card in cards)
    if durations:
        limit = durations[min(len(durations) - 1, len(durations) // 2)] + 0.25
        requests.append(
            {
                "preference_type": "deadline_then_clearance",
                "duration_limit_s": f"{limit:.2f}",
                "text": (
                    f"Finish within {limit:.2f} seconds, then maximize obstacle clearance.",
                    f"Meet a {limit:.2f}-second deadline and use the clearest feasible route.",
                    f"Stay under {limit:.2f} seconds; among those routes, keep the most clearance.",
                    f"Prioritize arrival by {limit:.2f} seconds, then prefer wider spacing.",
                )[variant],
            }
        )
    return requests


def mock_select(preference: Mapping[str, str], cards: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    preference_type = preference["preference_type"]
    feasible = list(cards)
    if preference_type == "deadline_then_clearance":
        limit = float(preference["duration_limit_s"])
        feasible = [card for card in cards if float(card["estimated_duration_s"]) <= limit]
        if not feasible:
            feasible = list(cards)
    if preference_type == "shortest":
        chosen = min(feasible, key=lambda card: (float(card["path_length_m"]), str(card["route_id"])))
    elif preference_type == "far_from_object":
        target = preference["target_object_id"]
        chosen = max(
            feasible,
            key=lambda card: (
                float(card["minimum_clearance_m"].get(target, -math.inf)),
                -float(card["path_length_m"]),
            ),
        )
    elif preference_type == "close_to_object":
        target = preference["target_object_id"]
        chosen = min(
            feasible,
            key=lambda card: (
                float(card["minimum_clearance_m"].get(target, math.inf)),
                float(card["path_length_m"]),
            ),
        )
    elif preference_type in ("widest", "deadline_then_clearance"):
        chosen = max(
            feasible,
            key=lambda card: (float(card["overall_minimum_clearance_m"]), -float(card["path_length_m"])),
        )
    else:
        min_length = min(float(card["path_length_m"]) for card in feasible)
        max_clearance = max(float(card["overall_minimum_clearance_m"]) for card in feasible)
        chosen = min(
            feasible,
            key=lambda card: (
                float(card["path_length_m"]) / max(min_length, 1e-6)
                - float(card["overall_minimum_clearance_m"]) / max(max_clearance, 1e-6),
                str(card["route_id"]),
            ),
        )
    return {"selected_route_id": str(chosen["route_id"]), "reason": f"Deterministic {preference_type} teacher."}


def openai_select(
    preference: Mapping[str, str], cards: Sequence[Mapping[str, object]], model: str
) -> Dict[str, str]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --teacher openai")
    from openai import OpenAI
    from pydantic import BaseModel, ConfigDict

    class Selection(BaseModel):
        model_config = ConfigDict(extra="forbid")
        selected_route_id: str
        reason: str

    response = OpenAI().responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Select exactly one supplied route_id that best satisfies the operator preference. "
                    "Never create coordinates or a new route. Return concise reasoning."
                ),
            },
            {
                "role": "user",
                "content": compact({"preference": preference["text"], "route_cards": cards}),
            },
        ],
        text_format=Selection,
    )
    if response.output_parsed is None:
        raise RuntimeError("Teacher returned no structured selection")
    return response.output_parsed.model_dump()


def selected_route(routes: Sequence[Mapping[str, object]], route_id: str) -> Mapping[str, object]:
    for route in routes:
        if str(route["route_id"]) == route_id:
            return route
    raise ValueError(f"Teacher selected unknown route_id {route_id!r}")


def generate(args: argparse.Namespace) -> None:
    if not 1 <= args.preferences_per_scene <= 6:
        raise ValueError("--preferences-per-scene must be between 1 and 6")
    if not 2 <= args.min_obstacles <= args.max_obstacles <= 4:
        raise ValueError("obstacle counts must satisfy 2 <= min <= max <= 4")
    completed = (
        prepare_resume_file(args.output, args.preferences_per_scene)
        if args.resume
        else set()
    )
    if args.output.exists() and not args.resume:
        args.output.unlink()
    written = 0
    for scene_id in range(args.scenes):
        if scene_id in completed:
            continue
        scene_rng = random.Random(args.seed + scene_id * 7919)
        for attempt in range(args.scene_attempts):
            scene = sample_scene(scene_rng, scene_id, scene_rng.randint(args.min_obstacles, args.max_obstacles))
            try:
                family = plan_hrrt_star(
                    scene["start"],
                    scene["goal"],
                    scene["obstacles"],
                    scene["workspace"],
                    seed=args.seed + scene_id * 100 + attempt,
                    max_iterations=args.hrrt_iterations,
                    max_candidates=args.max_candidates,
                    max_waypoints=8,
                    max_total_nodes=args.max_total_nodes,
                    max_nodes_per_key=args.max_nodes_per_key,
                )
            except (RuntimeError, ValueError):
                continue
            routes = list(family["routes"])
            signatures = {tuple(route["h_signature"]) for route in routes}
            direct_is_clear = segment_clear(
                (float(scene["start"]["x"]), float(scene["start"]["y"])),
                (float(scene["goal"]["x"]), float(scene["goal"]["y"])),
                normalize_obstacles(scene["obstacles"]),
                scene["workspace"],
                0.40,
            )
            if len(routes) >= args.min_routes and len(signatures) >= 2 and not direct_is_clear:
                break
        else:
            print(f"scene {scene_id}: no route family after {args.scene_attempts} attempts", flush=True)
            continue

        cards = route_cards(routes, scene["obstacles"], args.nominal_speed_mps)
        preferences = preference_requests(scene, cards)[: args.preferences_per_scene]
        for preference_index, preference in enumerate(preferences):
            selection = (
                mock_select(preference, cards)
                if args.teacher == "mock"
                else openai_select(preference, cards, args.teacher_model)
            )
            route = selected_route(routes, selection["selected_route_id"])
            record = {
                "schema_version": "hrrt_sft_raw_v1",
                "sample_id": f"scene-{scene_id:06d}-pref-{preference_index:02d}",
                "scene_id": scene_id,
                "environment": scene,
                "preference": preference,
                "route_cards": cards,
                "routes": routes,
                "teacher_provider": args.teacher,
                "teacher_model": args.teacher_model if args.teacher == "openai" else "deterministic_mock",
                "teacher_route_id": selection["selected_route_id"],
                "teacher_reason": selection["reason"],
                "selected_waypoints": route["waypoints"],
            }
            append_jsonl(args.output, record)
            written += 1
        print(f"scene {scene_id}: wrote {len(preferences)} labels", flush=True)
    print(f"Wrote {written} new records to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fine_tuning/datasets/hrrt_teacher_raw.jsonl"))
    parser.add_argument("--scenes", type=int, default=2000)
    parser.add_argument("--preferences-per-scene", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--teacher", choices=("openai", "mock"), default="openai")
    parser.add_argument("--teacher-model", default="gpt-5.4")
    parser.add_argument("--min-obstacles", type=int, default=2)
    parser.add_argument("--max-obstacles", type=int, default=4)
    parser.add_argument("--min-routes", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--hrrt-iterations", type=int, default=500)
    parser.add_argument("--max-total-nodes", type=int, default=5000)
    parser.add_argument("--max-nodes-per-key", type=int, default=180)
    parser.add_argument("--scene-attempts", type=int, default=8)
    parser.add_argument("--nominal-speed-mps", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
