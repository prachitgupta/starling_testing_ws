#!/usr/bin/env python3
"""Generate resumable HRRT-star teacher-label records for supervised tuning."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Mapping, Sequence

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
DISTILLATION_MODES = ("dss", "dss_scott")
FORBIDDEN_STUDENT_REASONING = (
    "route card",
    "candidate",
    "candidate route",
    "alternative",
    "alternative route",
    "supplied route",
    "selected route",
    "best route",
)

DSS_TEACHER_PROMPT = """You generate supervision for a smaller UAV trajectory planner.
Select exactly one supplied verified HRRT-star route that best satisfies the human preference.
The candidates are already collision-free. Do not create coordinates or modify a route.

Return an audit_selection_reason that may compare route IDs and metrics.
Return trajectory_reasoning for the student using exactly this compact structure:
Preference: ... Geometry: ... Decision: ... Safety: ...

The student will see only the environment, human preference, clearance requirement, and output constraints.
Therefore trajectory_reasoning must use only those deployment-visible facts plus the geometry of the selected
trajectory. It must not mention route IDs, route cards, candidates, alternatives, or the selection process.
Describe the relevant obstacle/corridor and the above/below spatial movement. Keep it concise and factual."""

DSS_SCOTT_TEACHER_PROMPT = """You generate answer-conditioned reasoning for a smaller UAV trajectory planner.
The target trajectory and safe counterfactual trajectory supplied below are fixed. Do not select another route,
change coordinates, or create a route.

Produce trajectory_reasoning that specifically supports the human preference, visible obstacle geometry, and
spatial movement of the fixed target trajectory. Use exactly:
Preference: ... Geometry: ... Decision: ... Safety: ...

Produce counterfactual_trajectory_reasoning using the same structure to describe the different spatial intention
of the fixed counterfactual trajectory. Produce counterfactual_conflict explaining why the counterfactual is less
consistent with the original preference.

The student will not see route IDs, route cards, or candidate comparisons. Neither rationale may mention them.
Keep both rationales concise and factual. Return preference_supported, geometry_supported, and safety_supported
consistency checks for the positive rationale."""

ROUTE_SELECTION_PROMPT = """Select exactly one supplied verified HRRT-star route that best satisfies the human
preference. Never create coordinates or a route ID. Return the supplied selected_route_id and a concise audit
reason that may compare supplied route metrics."""
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


def prepare_resume_file(
    path: Path,
    expected_per_scene: int,
    expected_distillation_mode: str | None = None,
) -> set[int]:
    """Drop partial scenes so resuming never duplicates or silently skips labels."""
    by_scene: Dict[int, List[Dict[str, object]]] = {}
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                row_mode = str(row.get("distillation_mode", "legacy"))
                if expected_distillation_mode and row_mode != expected_distillation_mode:
                    raise ValueError(
                        f"{path} contains distillation_mode={row_mode!r}; use a different --output "
                        f"or resume with --distillation-mode {row_mode}"
                    )
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


def card_by_id(cards: Sequence[Mapping[str, object]], route_id: str) -> Mapping[str, object]:
    for card in cards:
        if str(card["route_id"]) == route_id:
            return card
    raise ValueError(f"Unknown route card {route_id!r}")


def preference_summary(preference: Mapping[str, str]) -> str:
    preference_type = str(preference["preference_type"])
    target = str(preference.get("target_object_id", "the relevant object"))
    summaries = {
        "shortest": "minimize travel distance while remaining safe",
        "widest": "maximize clearance from obstacles",
        "balanced": "balance route efficiency with comfortable obstacle clearance",
        "far_from_object": f"maximize separation from {target}",
        "close_to_object": f"pass near {target} without violating clearance",
        "deadline_then_clearance": "remain efficient while preferring the clearest feasible corridor",
    }
    return summaries.get(preference_type, str(preference["text"]).rstrip("."))


def deterministic_trajectory_reasoning(
    preference: Mapping[str, str], card: Mapping[str, object], *, counterfactual: bool = False
) -> str:
    summary = str(card.get("route_summary") or "uses an open collision-free corridor")
    objective = (
        "follow the supplied counterfactual spatial intention"
        if counterfactual
        else preference_summary(preference)
    )
    return (
        f"Preference: {objective}. "
        f"Geometry: the path {summary}. "
        f"Decision: follow that corridor and then continue toward the goal. "
        "Safety: keep the path inside the workspace and outside the required obstacle clearance."
    )


def student_reasoning_is_valid(reason: object) -> bool:
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1200:
        return False
    lowered = reason.lower()
    if any(phrase in lowered for phrase in FORBIDDEN_STUDENT_REASONING):
        return False
    if re.search(r"\broute[-_ ]?\d+\b", lowered):
        return False
    return all(field in reason for field in ("Preference:", "Geometry:", "Decision:", "Safety:"))


def counterfactual_route_id(
    selected_id: str, cards: Sequence[Mapping[str, object]]
) -> str:
    selected = card_by_id(cards, selected_id)
    alternatives = [card for card in cards if str(card["route_id"]) != selected_id]
    if not alternatives:
        raise ValueError("DSS-SCOTT requires at least two candidate routes")
    return str(
        max(
            alternatives,
            key=lambda card: (
                abs(float(card["path_length_m"]) - float(selected["path_length_m"])),
                abs(
                    float(card["overall_minimum_clearance_m"])
                    - float(selected["overall_minimum_clearance_m"])
                ),
                str(card["route_id"]),
            ),
        )["route_id"]
    )


def openai_select(
    scene: Mapping[str, object],
    preference: Mapping[str, str],
    cards: Sequence[Mapping[str, object]],
    routes: Sequence[Mapping[str, object]],
    model: str,
    distillation_mode: str,
    clearance_m: float,
) -> Dict[str, object]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --teacher openai")
    from openai import OpenAI
    from pydantic import BaseModel, ConfigDict

    class DSSSelection(BaseModel):
        model_config = ConfigDict(extra="forbid")
        selected_route_id: str
        audit_selection_reason: str
        trajectory_reasoning: str

    class RouteSelection(BaseModel):
        model_config = ConfigDict(extra="forbid")
        selected_route_id: str
        audit_selection_reason: str

    class ScottConsistencyChecks(BaseModel):
        model_config = ConfigDict(extra="forbid")
        preference_supported: bool
        geometry_supported: bool
        safety_supported: bool

    class ScottRationales(BaseModel):
        model_config = ConfigDict(extra="forbid")
        counterfactual_trajectory_reasoning: str
        counterfactual_conflict: str
        trajectory_reasoning: str
        consistency_checks: ScottConsistencyChecks

    deployment_input = {
        "environment": scene,
        "human_preference": preference,
        "required_clearance_m": clearance_m,
        "output_constraints": {
            "minimum_waypoints": 2,
            "maximum_waypoints": 8,
            "fixed_altitude": scene["workspace"]["z"],
        },
    }
    client = OpenAI()
    if distillation_mode == "dss":
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": DSS_TEACHER_PROMPT},
                {
                    "role": "user",
                    "content": compact(
                        {
                            "deployment_input": deployment_input,
                            "teacher_only": {"route_cards": cards, "candidate_routes": routes},
                        }
                    ),
                },
            ],
            text_format=DSSSelection,
        )
        if response.output_parsed is None:
            raise RuntimeError("Teacher returned no structured DSS selection")
        return response.output_parsed.model_dump()

    selection_response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": ROUTE_SELECTION_PROMPT},
            {
                "role": "user",
                "content": compact(
                    {
                        "environment": scene,
                        "human_preference": preference,
                        "route_cards": cards,
                    }
                ),
            },
        ],
        text_format=RouteSelection,
    )
    if selection_response.output_parsed is None:
        raise RuntimeError("Teacher returned no structured DSS-SCOTT route selection")
    selected = selection_response.output_parsed.model_dump()
    selected_id = str(selected["selected_route_id"])
    selected_route_record = selected_route(routes, selected_id)
    selected_card = card_by_id(cards, selected_id)
    counterfactual_id = counterfactual_route_id(selected_id, cards)
    counterfactual_route_record = selected_route(routes, counterfactual_id)
    counterfactual_card = card_by_id(cards, counterfactual_id)
    rationale_response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": DSS_SCOTT_TEACHER_PROMPT},
            {
                "role": "user",
                "content": compact(
                    {
                        "deployment_input": deployment_input,
                        "teacher_only": {
                            "target_trajectory": {
                                "waypoints": selected_route_record["waypoints"],
                                "verified_metrics": selected_card,
                            },
                            "counterfactual_trajectory": {
                                "waypoints": counterfactual_route_record["waypoints"],
                                "verified_metrics": counterfactual_card,
                            },
                        },
                    }
                ),
            },
        ],
        text_format=ScottRationales,
    )
    if rationale_response.output_parsed is None:
        raise RuntimeError("Teacher returned no structured DSS-SCOTT rationales")
    return {
        **selected,
        **rationale_response.output_parsed.model_dump(),
        "counterfactual_route_id": counterfactual_id,
    }


def normalize_teacher_output(
    selection: Mapping[str, object],
    preference: Mapping[str, str],
    cards: Sequence[Mapping[str, object]],
    distillation_mode: str,
) -> Dict[str, object]:
    selected_id = str(selection["selected_route_id"])
    selected_card = card_by_id(cards, selected_id)
    raw_reason = selection.get("trajectory_reasoning")
    consistency_checks = selection.get("consistency_checks", {})
    checks_pass = not consistency_checks or all(bool(value) for value in consistency_checks.values())
    trajectory_reasoning = (
        str(raw_reason).strip()
        if checks_pass and student_reasoning_is_valid(raw_reason)
        else deterministic_trajectory_reasoning(preference, selected_card)
    )
    normalized: Dict[str, object] = {
        "selected_route_id": selected_id,
        "audit_selection_reason": str(
            selection.get("audit_selection_reason") or selection.get("reason") or ""
        ).strip(),
        "trajectory_reasoning": trajectory_reasoning,
        "trajectory_reasoning_raw": str(raw_reason or "").strip(),
    }
    if distillation_mode == "dss_scott":
        proposed_id = str(selection.get("counterfactual_route_id") or "")
        valid_ids = {str(card["route_id"]) for card in cards}
        counterfactual_id = (
            proposed_id
            if proposed_id in valid_ids and proposed_id != selected_id
            else counterfactual_route_id(selected_id, cards)
        )
        counterfactual_card = card_by_id(cards, counterfactual_id)
        raw_counterfactual = selection.get("counterfactual_trajectory_reasoning")
        normalized.update(
            {
                "counterfactual_route_id": counterfactual_id,
                "counterfactual_trajectory_reasoning": (
                    str(raw_counterfactual).strip()
                    if student_reasoning_is_valid(raw_counterfactual)
                    else deterministic_trajectory_reasoning(
                        preference, counterfactual_card, counterfactual=True
                    )
                ),
                "counterfactual_trajectory_reasoning_raw": str(raw_counterfactual or "").strip(),
                "counterfactual_conflict": str(
                    selection.get("counterfactual_conflict")
                    or (
                        f"This path {counterfactual_card.get('route_summary', 'uses a different corridor')}, "
                        f"which is less consistent with the original preference: {preference['text']}"
                    )
                ).strip(),
                "consistency_checks": consistency_checks or (
                    {
                        "preference_supported": True,
                        "geometry_supported": True,
                        "safety_supported": True,
                    }
                ),
            }
        )
    return normalized


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
        prepare_resume_file(args.output, args.preferences_per_scene, args.distillation_mode)
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
                    clearance_m=args.clearance_m,
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
                args.clearance_m,
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
                else openai_select(
                    scene,
                    preference,
                    cards,
                    routes,
                    args.teacher_model,
                    args.distillation_mode,
                    args.clearance_m,
                )
            )
            selection = normalize_teacher_output(
                selection, preference, cards, args.distillation_mode
            )
            route = selected_route(routes, selection["selected_route_id"])
            record = {
                "schema_version": "hrrt_sft_raw_v2",
                "sample_id": f"scene-{scene_id:06d}-pref-{preference_index:02d}",
                "scene_id": scene_id,
                "distillation_mode": args.distillation_mode,
                "environment": scene,
                "preference": preference,
                "route_cards": cards,
                "routes": routes,
                "teacher_provider": args.teacher,
                "teacher_model": args.teacher_model if args.teacher == "openai" else "deterministic_mock",
                "teacher_route_id": selection["selected_route_id"],
                "teacher_reason": selection["trajectory_reasoning"],
                "teacher_selection_reason": selection["audit_selection_reason"],
                "teacher_trajectory_reasoning": selection["trajectory_reasoning"],
                "teacher_trajectory_reasoning_raw": selection["trajectory_reasoning_raw"],
                "selected_waypoints": route["waypoints"],
            }
            if args.distillation_mode == "dss_scott":
                counterfactual = selected_route(routes, str(selection["counterfactual_route_id"]))
                record.update(
                    {
                        "counterfactual_route_id": selection["counterfactual_route_id"],
                        "counterfactual_waypoints": counterfactual["waypoints"],
                        "counterfactual_trajectory_reasoning": selection[
                            "counterfactual_trajectory_reasoning"
                        ],
                        "counterfactual_trajectory_reasoning_raw": selection[
                            "counterfactual_trajectory_reasoning_raw"
                        ],
                        "counterfactual_conflict": selection["counterfactual_conflict"],
                        "teacher_consistency_checks": selection["consistency_checks"],
                    }
                )
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
    parser.add_argument(
        "--distillation-mode",
        choices=DISTILLATION_MODES,
        default="dss",
        help="dss creates positive rationale labels; dss_scott also creates safe counterfactual supervision.",
    )
    parser.add_argument("--min-obstacles", type=int, default=2)
    parser.add_argument("--max-obstacles", type=int, default=4)
    parser.add_argument("--min-routes", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--hrrt-iterations", type=int, default=500)
    parser.add_argument("--max-total-nodes", type=int, default=5000)
    parser.add_argument("--max-nodes-per-key", type=int, default=180)
    parser.add_argument("--scene-attempts", type=int, default=8)
    parser.add_argument("--nominal-speed-mps", type=float, default=0.5)
    parser.add_argument("--clearance-m", type=float, default=0.40)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
