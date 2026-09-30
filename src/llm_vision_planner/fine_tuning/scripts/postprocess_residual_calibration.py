#!/usr/bin/env python3
"""Convert raw hardware captures into one-score residual calibration rows.

The raw CSV contains only synchronized ``E_hat``/``E`` obstacle geometry.
For every capture this script:

1. plans an HRRT-star expert reference on ground-truth ``E``;
2. plans the predicted reference on perceived ``E_hat``;
3. solves both minimum-control QPs with ideal double-integrator dynamics; and
4. records ``max_t ||P^(1/2) B[(u_hat-u_d)+K(x_hat-x_d)]||_2``.

Real runs use ``--planner-provider vllm``.  ``mock`` is deterministic and is
provided only for schema smoke tests and the checked-in placeholder dataset.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hrrt_star import plan_hrrt_star, segment_clear  # noqa: E402
from min_control_qp import generate_shared_pair  # noqa: E402


CSV_DELIMITER = ","
RAW_REQUIRED_FIELDS = {
    "session_id", "capture_id", "object_id", "label",
    "pred_min_x", "pred_min_y", "pred_max_x", "pred_max_y",
    "gt_min_x", "gt_min_y", "gt_max_x", "gt_max_y",
    "missed_detection", "observer_x", "observer_y", "observer_z",
}
SCORED_FIELDS = [
    "session_id",
    "capture_id",
    "l1_text",
    "l2_text",
    "start_x",
    "start_y",
    "start_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "ground_truth_environment_json",
    "perceived_environment_json",
    "expert_selector",
    "expert_model",
    "expert_route_id",
    "expert_h_signature_json",
    "expert_clearance_bins_json",
    "expert_waypoints_json",
    "predicted_waypoints_json",
    "expert_state_samples_json",
    "expert_control_samples_json",
    "predicted_state_samples_json",
    "predicted_control_samples_json",
    "conformity_score",
    "score_max_time_s",
    "score_max_residual_json",
    "p_matrix_json",
    "k_matrix_json",
    "dynamics_model",
    "planner_provider",
    "placeholder",
]


def compact(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def read_raw(path: Path, delimiter: str = CSV_DELIMITER):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fields = set(reader.fieldnames or [])
        missing = RAW_REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(f"raw CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("raw CSV contains no observations")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["capture_id"]].append(row)
    return grouped


def obstacle(row: Mapping[str, str], prefix: str):
    return {
        "object_id": row["object_id"],
        "label": row["label"],
        "min_corner": [float(row[f"{prefix}_min_x"]), float(row[f"{prefix}_min_y"]), -1.0],
        "max_corner": [float(row[f"{prefix}_max_x"]), float(row[f"{prefix}_max_y"]), 0.0],
    }


def environments(rows: Sequence[Mapping[str, str]]):
    ground_truth = []
    perceived = []
    for row in rows:
        ground_truth.append(obstacle(row, "gt"))
        if not parse_bool(row["missed_detection"]):
            perceived.append(obstacle(row, "pred"))
    if not perceived:
        raise ValueError(f"capture {rows[0]['capture_id']} has no perceived obstacles")
    return ground_truth, perceived


def route_cards(routes, obstacles):
    cards = []
    for route in routes:
        minimum = {str(key): float(value) for key, value in route["minimum_clearance_m"].items()}
        cards.append(
            {
                "route_id": str(route["route_id"]),
                "path_length_m": float(route["path_length_m"]),
                "minimum_clearance_m": minimum,
                "overall_minimum_clearance_m": min(minimum.values()) if minimum else math.inf,
                "h_signature": list(route["h_signature"]),
                "clearance_bins": list(route["clearance_bins"]),
            }
        )
    return cards


def object_requested(text: str, obstacles: Sequence[Mapping[str, object]]):
    normalized = text.lower()
    for item in obstacles:
        if str(item["object_id"]).lower() in normalized or str(item["label"]).lower() in normalized:
            return str(item["object_id"])
    return None


def select_expert(routes, obstacles, preference, route_id=""):
    if route_id:
        for route in routes:
            if str(route["route_id"]) == route_id:
                return route
        raise ValueError(f"expert route_id {route_id!r} is not in this HRRT family")
    cards = route_cards(routes, obstacles)
    text = preference.lower()
    target = object_requested(preference, obstacles)
    if target and any(word in text for word in ("far", "away", "distance", "wide")):
        selected = max(cards, key=lambda card: (card["minimum_clearance_m"].get(target, -math.inf),
                                                 -card["path_length_m"]))
    elif target and any(word in text for word in ("close", "near", "beside")):
        selected = min(cards, key=lambda card: (card["minimum_clearance_m"].get(target, math.inf),
                                                 card["path_length_m"]))
    elif any(word in text for word in ("fast", "short", "quick", "time")):
        selected = min(cards, key=lambda card: (card["path_length_m"], card["route_id"]))
    else:
        selected = min(cards, key=lambda card: (card["path_length_m"], card["route_id"]))
    return next(route for route in routes if route["route_id"] == selected["route_id"])


def openai_select_expert(routes, obstacles, preference, model, api_key):
    """Ask the high-capacity teacher to select one existing HRRT route ID."""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --expert-selector openai")
    try:
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict
    except ImportError as exc:
        raise RuntimeError("OpenAI expert selection requires openai and pydantic") from exc

    class Selection(BaseModel):
        model_config = ConfigDict(extra="forbid")
        selected_route_id: str
        reason: str

    cards = route_cards(routes, obstacles)
    catalog = [
        {
            "object_id": str(item["object_id"]),
            "label": str(item["label"]),
        }
        for item in obstacles
    ]
    response = OpenAI(api_key=api_key).responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Select exactly one supplied HRRT route_id that best satisfies the operator's "
                    "natural-language preference. Never invent a route or coordinates. Use labels, "
                    "per-object clearances, path length, and the stated preference."
                ),
            },
            {
                "role": "user",
                "content": compact(
                    {
                        "operator_preference_L2": preference,
                        "detected_objects": catalog,
                        "route_cards": cards,
                    }
                ),
            },
        ],
        text_format=Selection,
    )
    selection = response.output_parsed
    if selection is None:
        raise RuntimeError("OpenAI expert returned no structured route selection")
    selected = next(
        (route for route in routes if str(route["route_id"]) == selection.selected_route_id),
        None,
    )
    if selected is None:
        raise ValueError(
            f"OpenAI expert selected unknown route_id {selection.selected_route_id!r}"
        )
    return selected


def choose_expert(routes, obstacles, args):
    if args.expert_route_id:
        return (
            select_expert(routes, obstacles, args.l2_text, args.expert_route_id),
            "human",
            "",
        )
    if args.expert_selector == "openai":
        return (
            openai_select_expert(
                routes,
                obstacles,
                args.l2_text,
                args.expert_model,
                args.openai_api_key,
            ),
            "openai",
            args.expert_model,
        )
    return select_expert(routes, obstacles, args.l2_text), "heuristic", ""


def safe_path(waypoints, obstacles, workspace, clearance_m):
    if len(waypoints) < 2:
        return False
    return all(
        segment_clear(
            (float(first["x"]), float(first["y"])),
            (float(second["x"]), float(second["y"])),
            obstacles,
            workspace,
            clearance_m,
        )
        for first, second in zip(waypoints, waypoints[1:])
    )


def mock_predicted_waypoints(start, goal, perceived, workspace, args):
    """Deterministic stand-in that still plans strictly on perceived E_hat."""
    payload = plan_hrrt_star(
        start, goal, perceived, workspace,
        clearance_m=args.clearance_m,
        max_iterations=args.hrrt_iterations,
        max_candidates=args.max_candidates,
        max_waypoints=args.max_waypoints,
        seed=args.seed + 1009,
    )
    return min(payload["routes"], key=lambda route: (route["path_length_m"], route["route_id"]))["waypoints"]


def vllm_predicted_waypoints(start, goal, perceived, workspace, l1_text, args):
    try:
        import instructor
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError("vLLM planning requires openai, instructor, and pydantic") from exc

    class Waypoint(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: float
        y: float
        z: float

    class Plan(BaseModel):
        model_config = ConfigDict(extra="forbid")
        reasoning: str
        waypoints: List[Waypoint] = Field(min_length=2, max_length=8)

    client = instructor.from_openai(
        OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key),
        mode=instructor.Mode.JSON,
    )
    request = {
        "operator_goal_instruction": l1_text,
        "start": start,
        "goal": goal,
        "workspace": workspace,
        "perceived_obstacles": perceived,
    }
    plan = client.chat.completions.create(
        model=args.llama_model,
        response_model=Plan,
        messages=[{
            "role": "user",
            "content": (
                "Return sparse collision-free NED waypoints using only the supplied perceived environment. "
                "The first and last waypoints must equal start and goal.\n" + compact(request)
            ),
        }],
        temperature=0.0,
    )
    waypoints = [item.model_dump() for item in plan.waypoints]
    waypoints[0] = dict(start)
    waypoints[-1] = dict(goal)
    if not safe_path(waypoints, perceived, workspace, args.clearance_m):
        raise ValueError("Llama returned a path that fails perceived-environment collision checking")
    return waypoints


def ideal_metric():
    try:
        from scipy.linalg import solve_continuous_are
    except ImportError as exc:
        raise RuntimeError("residual scoring requires scipy") from exc
    a = np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=float,
    )
    b = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    q = np.diag([10.0, 10.0, 1.0, 1.0])
    r = np.eye(2)
    p = solve_continuous_are(a, b, q, r)
    k = np.linalg.solve(r, b.T @ p)
    return b, p, k


def residual_score(expert_samples, predicted_samples):
    if len(expert_samples) != len(predicted_samples):
        raise ValueError("expert and predicted QP samples must share one horizon")
    b, p, k = ideal_metric()
    maximum = None
    for expert, predicted in zip(expert_samples, predicted_samples):
        x_delta = np.asarray(predicted["x"], dtype=float) - np.asarray(expert["x"], dtype=float)
        u_delta = np.asarray(predicted["u"], dtype=float) - np.asarray(expert["u"], dtype=float)
        residual = b @ (u_delta + k @ x_delta)
        value = float(math.sqrt(max(0.0, residual.T @ p @ residual)))
        candidate = (value, float(expert["t"]), residual)
        if maximum is None or candidate[0] > maximum[0]:
            maximum = candidate
    return maximum, p, k


def sample_arrays(samples):
    return (
        [{"t": sample["t"], "x": sample["x"]} for sample in samples],
        [{"t": sample["t"], "u": sample["u"]} for sample in samples],
    )


def process_capture(capture_id, rows, args):
    ground_truth, perceived = environments(rows)
    first = rows[0]
    start = {
        "x": float(first["observer_x"] or args.start_x),
        "y": float(first["observer_y"] or args.start_y),
        "z": float(first["observer_z"] or args.fixed_z),
    }
    goal = {"x": args.goal_x, "y": args.goal_y, "z": args.fixed_z}
    workspace = {"x": [args.workspace_x_min, args.workspace_x_max],
                 "y": [args.workspace_y_min, args.workspace_y_max], "z": args.fixed_z}
    family = plan_hrrt_star(
        start, goal, ground_truth, workspace,
        clearance_m=args.clearance_m,
        max_iterations=args.hrrt_iterations,
        max_candidates=args.max_candidates,
        max_waypoints=args.max_waypoints,
        seed=args.seed,
    )
    expert, expert_selector, expert_model = choose_expert(
        family["routes"], ground_truth, args
    )
    if args.planner_provider == "mock":
        predicted_waypoints = mock_predicted_waypoints(start, goal, perceived, workspace, args)
    else:
        predicted_waypoints = vllm_predicted_waypoints(
            start, goal, perceived, workspace, args.l1_text, args
        )
    expert_trajectory, predicted_trajectory = generate_shared_pair(
        expert["waypoints"], predicted_waypoints,
        workspace=workspace,
        dt=args.dt,
        damping=0.0,
    )
    maximum, p, k = residual_score(
        expert_trajectory["samples"], predicted_trajectory["samples"]
    )
    expert_states, expert_controls = sample_arrays(expert_trajectory["samples"])
    predicted_states, predicted_controls = sample_arrays(predicted_trajectory["samples"])
    return {
        "session_id": first["session_id"],
        "capture_id": capture_id,
        "l1_text": args.l1_text,
        "l2_text": args.l2_text,
        "start_x": f"{start['x']:.9f}", "start_y": f"{start['y']:.9f}", "start_z": f"{start['z']:.9f}",
        "goal_x": f"{goal['x']:.9f}", "goal_y": f"{goal['y']:.9f}", "goal_z": f"{goal['z']:.9f}",
        "ground_truth_environment_json": compact(ground_truth),
        "perceived_environment_json": compact(perceived),
        "expert_selector": expert_selector,
        "expert_model": expert_model,
        "expert_route_id": expert["route_id"],
        "expert_h_signature_json": compact(expert["h_signature"]),
        "expert_clearance_bins_json": compact(expert["clearance_bins"]),
        "expert_waypoints_json": compact(expert["waypoints"]),
        "predicted_waypoints_json": compact(predicted_waypoints),
        "expert_state_samples_json": compact(expert_states),
        "expert_control_samples_json": compact(expert_controls),
        "predicted_state_samples_json": compact(predicted_states),
        "predicted_control_samples_json": compact(predicted_controls),
        "conformity_score": f"{maximum[0]:.9f}",
        "score_max_time_s": f"{maximum[1]:.9f}",
        "score_max_residual_json": compact([float(value) for value in maximum[2]]),
        "p_matrix_json": compact(p.tolist()),
        "k_matrix_json": compact(k.tolist()),
        "dynamics_model": "ideal_planar_double_integrator",
        "planner_provider": args.planner_provider,
        "placeholder": str(args.placeholder).lower(),
    }


def write_rows(path: Path, rows: Iterable[Mapping[str, object]], delimiter=CSV_DELIMITER, append=False):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if append and path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            if (reader.fieldnames or []) != SCORED_FIELDS:
                raise ValueError("existing scored CSV schema does not match this postprocessor")
            existing_keys = {
                (row["capture_id"], row["l1_text"], row["l2_text"]) for row in reader
            }
    duplicates = [
        (row["capture_id"], row["l1_text"], row["l2_text"])
        for row in rows
        if (row["capture_id"], row["l1_text"], row["l2_text"]) in existing_keys
    ]
    if duplicates:
        raise ValueError(f"refusing to append duplicate scored batch keys: {duplicates}")
    write_header = not append or not path.exists()
    with path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SCORED_FIELDS, delimiter=delimiter, lineterminator="\n"
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def selected_captures(grouped, args):
    capture_ids = sorted(grouped)
    if args.capture_id:
        if len(set(args.capture_id)) != len(args.capture_id):
            raise ValueError("--capture-id values must be unique within one batch")
        missing = sorted(set(args.capture_id) - set(capture_ids))
        if missing:
            raise ValueError(f"unknown capture IDs: {', '.join(missing)}")
        return [(capture_id, grouped[capture_id]) for capture_id in args.capture_id]
    if args.sample_count is not None:
        if args.sample_count <= 0:
            raise ValueError("--sample-count must be positive")
        if args.sample_count > len(capture_ids):
            raise ValueError(
                f"--sample-count {args.sample_count} exceeds {len(capture_ids)} available captures"
            )
        chosen = sorted(random.Random(args.sample_seed).sample(capture_ids, args.sample_count))
        return [(capture_id, grouped[capture_id]) for capture_id in chosen]
    return [(capture_id, grouped[capture_id]) for capture_id in capture_ids]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, help="Required when scoring captures.")
    parser.add_argument("--delimiter", default=CSV_DELIMITER, help="One-character CSV delimiter (default: comma).")
    parser.add_argument("--l1-text", default="", help="Goal-setting natural-language input L1.")
    parser.add_argument("--l2-text", default="", help="Expert route-preference input L2.")
    parser.add_argument("--expert-route-id", default="", help="Optional exact human-selected HRRT route ID.")
    parser.add_argument("--expert-selector", choices=("openai", "heuristic"), default="openai")
    parser.add_argument("--expert-model", default="gpt-5.4")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""), help=argparse.SUPPRESS)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--fixed-z", type=float, default=-0.5)
    parser.add_argument("--start-x", type=float, default=-3.5)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--workspace-x-min", type=float, default=-4.0)
    parser.add_argument("--workspace-x-max", type=float, default=4.0)
    parser.add_argument("--workspace-y-min", type=float, default=-3.0)
    parser.add_argument("--workspace-y-max", type=float, default=3.0)
    parser.add_argument("--clearance-m", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hrrt-iterations", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-waypoints", type=int, default=8)
    parser.add_argument("--planner-provider", choices=("vllm", "mock"), default="vllm")
    parser.add_argument("--vllm-base-url", default="http://172.22.224.93:8000/v1")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--llama-model", default="hrrt_planner")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--capture-id",
        action="append",
        help="Score only this capture ID; repeat for multiple captures in this L1/L2 batch.",
    )
    selection.add_argument(
        "--sample-count",
        type=int,
        help="Deterministically sample this many captures for the current L1/L2 batch.",
    )
    selection.add_argument("--all-captures", action="store_true", help="Explicitly score every capture.")
    selection.add_argument("--list-captures", action="store_true", help="List available capture IDs and exit.")
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append this L1/L2 batch while rejecting duplicate capture/L1/L2 keys.",
    )
    parser.add_argument("--placeholder", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if len(args.delimiter) != 1:
        raise SystemExit("--delimiter must contain exactly one character")
    grouped = read_raw(args.raw_csv, args.delimiter)
    if args.list_captures:
        for capture_id, rows in sorted(grouped.items()):
            labels = sorted({str(row["label"]) for row in rows})
            print(f"{capture_id}\tobjects={len(rows)}\tlabels={','.join(labels)}")
        return
    if (
        args.output_csv is None
        or not args.l1_text
        or not args.l2_text
        or args.goal_x is None
        or args.goal_y is None
    ):
        raise SystemExit(
            "scoring requires --output-csv, --l1-text, --l2-text, --goal-x, and --goal-y"
        )
    try:
        selected = selected_captures(grouped, args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    processed = []
    failures = []
    for capture_id, rows in selected:
        try:
            processed.append(process_capture(capture_id, rows, args))
        except (RuntimeError, ValueError) as exc:
            failures.append(f"{capture_id}: {exc}")
    if not processed:
        raise SystemExit("No captures were scored. " + "; ".join(failures))
    write_rows(args.output_csv, processed, args.delimiter, append=args.append)
    print(f"Wrote {len(processed)} scored captures to {args.output_csv}")
    if failures:
        print(f"Skipped {len(failures)} captures:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
