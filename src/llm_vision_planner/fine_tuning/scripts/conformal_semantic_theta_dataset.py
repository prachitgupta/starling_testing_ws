#!/usr/bin/env python3
"""Generate Semantic-Theta* instruction data or verified expert/LLM pairs."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

from semantic_theta import (
    DEFAULT_CLEARANCE_M,
    DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_SEMANTIC_POLICY,
    DEFAULT_WORKSPACE,
    path_cost,
    plan_semantic_theta,
    point_clear,
    segment_clear,
    validated_semantic_policy,
)


SCRIPT_DIR = Path(__file__).resolve().parent
FINE_TUNING_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = FINE_TUNING_DIR / "datasets"
DEFAULT_GOAL = {"x": 2.5, "y": 0.0, "z": -0.25}
DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "conformal_semantic_theta_calibration_dataset_2001.csv"
DEFAULT_VLLM_BASE_URL = "http://172.22.224.93:8000/v1"
DEFAULT_LLAMA_MODEL_NAME = "semantic_theta_planner"
SEMANTIC_LABELS = ("person", "chair", "backpack", "bottle", "potted_plant", "bench", "stop_sign", "unknown")
SOURCE_FIELDNAMES = [
    "sample_id",
    "start",
    "goal",
    "workspace",
    "obstacles",
    "semantic_policy",
    "semantic_theta_cost",
    "semantic_theta_waypoints",
    "llm_waypoints",
    "semantic_theta_verified_waypoints",
    "llm_verified_waypoints",
    "semantic_theta_verification_metrics",
    "llm_verification_metrics",
    "llama_model_name",
    "vllm_base_url",
    "prompt",
]


def random_point(workspace):
    return {
        "x": round(random.uniform(float(workspace["x"][0]), float(workspace["x"][1])), 2),
        "y": round(random.uniform(float(workspace["y"][0]), float(workspace["y"][1])), 2),
        "z": float(workspace["z"]),
    }


def obstacle_size(label):
    ranges = {
        "person": ((0.35, 0.55), (0.35, 0.55), 1.70),
        "chair": ((0.40, 0.75), (0.40, 0.75), 0.90),
        "backpack": ((0.25, 0.45), (0.20, 0.40), 0.55),
        "bottle": ((0.12, 0.25), (0.12, 0.25), 0.35),
        "potted_plant": ((0.30, 0.65), (0.30, 0.65), 0.90),
        "bench": ((0.75, 1.20), (0.30, 0.55), 0.70),
        "stop_sign": ((0.25, 0.45), (0.20, 0.35), 1.50),
        "unknown": ((0.30, 0.80), (0.30, 0.80), 1.00),
    }
    width_range, depth_range, height = ranges[label]
    return random.uniform(*width_range), random.uniform(*depth_range), height


def obstacle_template(index, label, min_x, min_y, width, depth, height, fixed_z):
    return {
        "id": f"semantic-{index}",
        "label": label,
        "shape": "box",
        "min_corner": [round(min_x, 2), round(min_y, 2), round(fixed_z - height * 0.5, 2)],
        "max_corner": [round(min_x + width, 2), round(min_y + depth, 2), round(fixed_z + height * 0.5, 2)],
        "size": [round(width, 2), round(depth, 2), round(height, 2)],
    }


def sample_obstacles(workspace, start, goal, count, semantic_policy, clearance_m):
    obstacles = []
    fixed_z = float(workspace["z"])
    for index in range(1, count + 1):
        label = random.choice(SEMANTIC_LABELS)
        for _ in range(100):
            width, depth, height = obstacle_size(label)
            min_x = random.uniform(float(workspace["x"][0]), float(workspace["x"][1]) - width)
            min_y = random.uniform(float(workspace["y"][0]), float(workspace["y"][1]) - depth)
            obstacle = obstacle_template(index, label, min_x, min_y, width, depth, height, fixed_z)
            candidate = obstacles + [obstacle]
            if point_clear(start, candidate, workspace, semantic_policy, clearance_m) and point_clear(
                goal, candidate, workspace, semantic_policy, clearance_m
            ):
                obstacles.append(obstacle)
                break
    return obstacles


def sample_environment(workspace, fixed_goal, semantic_policy, clearance_m):
    for _ in range(200):
        start = random_point(workspace)
        goal = dict(fixed_goal) if fixed_goal else random_point(workspace)
        if math.hypot(goal["x"] - start["x"], goal["y"] - start["y"]) < 1.0:
            continue
        obstacles = sample_obstacles(workspace, start, goal, random.randint(0, 4), semantic_policy, clearance_m)
        return start, goal, obstacles
    raise RuntimeError("Failed to sample a usable semantic environment.")


def policy_text(obstacles, semantic_policy):
    labels = sorted({str(obstacle.get("label", "unknown")) for obstacle in obstacles})
    if not labels:
        return "No semantic risk fields are active."
    descriptions = []
    for label in labels:
        settings = semantic_policy.get(label, semantic_policy["default"])
        descriptions.append(
            f"{label}: hard margin={settings['hard_margin_m']:.2f}m, "
            f"soft radius={settings['soft_radius_m']:.2f}m, risk weight={settings['risk_weight']:.1f}"
        )
    return "; ".join(descriptions) + "."


def build_prompt(start, goal, workspace, obstacles, semantic_policy):
    obstacle_descriptions = []
    for index, obstacle in enumerate(obstacles, start=1):
        minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
        obstacle_descriptions.append(
            f"{index} {obstacle['label']}: x=[{minimum[0]:.2f},{maximum[0]:.2f}], "
            f"y=[{minimum[1]:.2f},{maximum[1]:.2f}]"
        )
    obstacle_text = "; ".join(obstacle_descriptions) if obstacle_descriptions else "none"
    distance = math.hypot(goal["x"] - start["x"], goal["y"] - start["y"])
    nl_env = (
        "Mission state: the UAV is holding hover at the start position. "
        f"Workspace: x=[{workspace['x'][0]:.2f},{workspace['x'][1]:.2f}]m, "
        f"y=[{workspace['y'][0]:.2f},{workspace['y'][1]:.2f}]m, z={workspace['z']:.2f} fixed. "
        f"Start: ({start['x']:.2f},{start['y']:.2f},{start['z']:.2f}); "
        f"goal: ({goal['x']:.2f},{goal['y']:.2f},{goal['z']:.2f}); distance={distance:.2f}m. "
        f"Semantic obstacle boxes: {obstacle_text}. Semantic policy: {policy_text(obstacles, semantic_policy)}"
    )
    prompt = (
        "You are an expert semantic UAV path planner. Return sparse fixed-altitude NED waypoints that minimize "
        "path length plus semantic risk while satisfying every hard collision margin.\n"
        f"{nl_env}\n"
        "Constraints:\n"
        "- first waypoint exactly equals start and final waypoint exactly equals goal\n"
        "- return between 2 and 8 waypoints, all inside the workspace at the fixed z\n"
        "- never enter a hard-margin region; prefer lower accumulated soft semantic risk\n"
        "- return only the structured output requested by the response model"
    )
    return prompt, nl_env


def completion_from_path(path):
    return {
        "reasoning": "Semantic Theta* selected a hard-safe any-angle route minimizing distance and label-conditioned risk.",
        "waypoints": path,
    }


def path_is_safe(path, obstacles, workspace, semantic_policy, clearance_m):
    return all(
        segment_clear(first, second, obstacles, workspace, semantic_policy, clearance_m)
        for first, second in zip(path, path[1:])
    )


def write_rows(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_csv)


def generate_training_dataset(args, semantic_policy):
    random.seed(args.seed)
    workspace = dict(DEFAULT_WORKSPACE)
    fixed_goal = None if args.random_goal else DEFAULT_GOAL
    rows = []
    attempts = 0
    while len(rows) < args.samples and attempts < args.samples * 30:
        attempts += 1
        start, goal, obstacles = sample_environment(
            workspace, fixed_goal, semantic_policy, args.clearance_m
        )
        try:
            path = plan_semantic_theta(
                start,
                goal,
                obstacles,
                workspace,
                semantic_policy,
                args.clearance_m,
                args.grid_resolution_m,
                args.semantic_cost_scale,
            )
        except (RuntimeError, ValueError):
            continue
        if not path_is_safe(path, obstacles, workspace, semantic_policy, args.clearance_m):
            continue
        prompt, nl_env = build_prompt(start, goal, workspace, obstacles, semantic_policy)
        completion = completion_from_path(path)
        compact_completion = json.dumps(completion, separators=(",", ":"))
        rows.append(
            {
                "sample_id": len(rows),
                "prompt": prompt,
                "completion": compact_completion,
                "messages": json.dumps(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": compact_completion},
                    ],
                    separators=(",", ":"),
                ),
                "nl_env": nl_env,
                "start": json.dumps(start, separators=(",", ":")),
                "goal": json.dumps(goal, separators=(",", ":")),
                "workspace": json.dumps(workspace, separators=(",", ":")),
                "obstacles": json.dumps(obstacles, separators=(",", ":")),
                "semantic_policy": json.dumps(semantic_policy, separators=(",", ":")),
                "semantic_cost": round(
                    path_cost(path, obstacles, semantic_policy, args.clearance_m, args.semantic_cost_scale), 6
                ),
                "waypoints": json.dumps(path, separators=(",", ":")),
            }
        )
        write_rows(rows, args.output)
        if len(rows) == 1 or len(rows) % 100 == 0 or len(rows) == args.samples:
            print(f"[{len(rows)}/{args.samples}] wrote Semantic-Theta* training checkpoint", flush=True)
    if len(rows) < args.samples:
        raise RuntimeError(f"Generated {len(rows)} rows after {attempts} attempts; requested {args.samples}.")
    return args.output


def refine_and_verify(refiner, verifier, waypoints, row, semantic_policy, clearance_m):
    refined = refiner.interpolate_waypoints(waypoints, row["workspace"], row["obstacles"])
    payload = {
        "waypoints": refined,
        "start": row["start"],
        "obstacles": row["obstacles"],
        "workspace": row["workspace"],
        "goal": row["goal"],
    }
    metrics = verifier.compute_metrics(payload)
    if not metrics["passed"]:
        raise ValueError(f"verification failed: {', '.join(metrics['failed_constraints'])}")
    if not path_is_safe(refined, row["obstacles"], row["workspace"], semantic_policy, clearance_m):
        raise ValueError("semantic hard-margin segment verification failed")
    return refined, metrics


def write_prediction(prediction, index, args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if index == 0 else "a"
    with args.output.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDNAMES, lineterminator="\n")
        if index == 0:
            writer.writeheader()
        writer.writerow({name: prediction.get(name, "") for name in SOURCE_FIELDNAMES})


def build_prediction_dataset(args, semantic_policy):
    import instructor
    from conformal_rrt_dataset import WaypointPlan, make_refiner, make_verifier
    from openai import OpenAI

    random.seed(args.seed)
    workspace = dict(DEFAULT_WORKSPACE)
    raw_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
    refiner, verifier = make_refiner(), make_verifier()
    accepted, attempts = 0, 0
    while accepted < args.samples and attempts < args.samples * 30:
        attempts += 1
        start, goal, obstacles = sample_environment(workspace, None, semantic_policy, args.clearance_m)
        try:
            expert = plan_semantic_theta(
                start,
                goal,
                obstacles,
                workspace,
                semantic_policy,
                args.clearance_m,
                args.grid_resolution_m,
                args.semantic_cost_scale,
            )
            prompt, _ = build_prompt(start, goal, workspace, obstacles, semantic_policy)
            plan = None
            last_error = None
            for _ in range(args.llm_retries):
                try:
                    plan = client.chat.completions.create(
                        model=args.llama_model_name,
                        response_model=WaypointPlan,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=args.temperature,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if plan is None:
                raise RuntimeError(f"Llama waypoint request failed: {last_error}")
            candidate = [waypoint.model_dump() for waypoint in plan.waypoints]
            row = {"start": start, "goal": goal, "workspace": workspace, "obstacles": obstacles}
            expert_verified, expert_metrics = refine_and_verify(
                refiner, verifier, expert, row, semantic_policy, args.clearance_m
            )
            llm_verified, llm_metrics = refine_and_verify(
                refiner, verifier, candidate, row, semantic_policy, args.clearance_m
            )
        except Exception as exc:
            print(f"[attempt {attempts}] skipped: {exc}", flush=True)
            continue
        prediction = {
            "sample_id": accepted,
            "start": json.dumps(start, separators=(",", ":")),
            "goal": json.dumps(goal, separators=(",", ":")),
            "workspace": json.dumps(workspace, separators=(",", ":")),
            "obstacles": json.dumps(obstacles, separators=(",", ":")),
            "semantic_policy": json.dumps(semantic_policy, separators=(",", ":")),
            "semantic_theta_cost": round(
                path_cost(expert, obstacles, semantic_policy, args.clearance_m, args.semantic_cost_scale), 6
            ),
            "semantic_theta_waypoints": json.dumps(expert, separators=(",", ":")),
            "llm_waypoints": json.dumps(candidate, separators=(",", ":")),
            "semantic_theta_verified_waypoints": json.dumps(expert_verified, separators=(",", ":")),
            "llm_verified_waypoints": json.dumps(llm_verified, separators=(",", ":")),
            "semantic_theta_verification_metrics": json.dumps(expert_metrics, separators=(",", ":")),
            "llm_verification_metrics": json.dumps(llm_metrics, separators=(",", ":")),
            "llama_model_name": args.llama_model_name,
            "vllm_base_url": args.vllm_base_url,
            "prompt": prompt,
        }
        write_prediction(prediction, accepted, args)
        accepted += 1
        print(f"[{accepted}/{args.samples}] wrote verified Semantic-Theta*/LLM pair", flush=True)
    if accepted < args.samples:
        raise RuntimeError(f"Only built {accepted} verified rows after {attempts} attempts; requested {args.samples}.")
    return args.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--semantic-theta-training", action="store_true", help="Generate expert instruction data without vLLM.")
    parser.add_argument("--random-goal", action="store_true")
    parser.add_argument("--semantic-policy", default=json.dumps(DEFAULT_SEMANTIC_POLICY))
    parser.add_argument("--clearance-m", type=float, default=DEFAULT_CLEARANCE_M)
    parser.add_argument("--grid-resolution-m", type=float, default=DEFAULT_GRID_RESOLUTION_M)
    parser.add_argument("--semantic-cost-scale", type=float, default=1.0)
    parser.add_argument("--llama-model-name", default=DEFAULT_LLAMA_MODEL_NAME)
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    args = parser.parse_args()

    semantic_policy = validated_semantic_policy(json.loads(args.semantic_policy), args.clearance_m)
    if args.semantic_theta_training:
        args.samples = args.samples or 100
        args.seed = args.seed if args.seed is not None else 17
        args.output = args.output or (DEFAULT_DATASET_DIR / "semantic_theta_expert_dataset.csv")
        output = generate_training_dataset(args, semantic_policy)
        print(f"Wrote Semantic-Theta* instruction data to {output}")
    else:
        args.samples = args.samples or 2001
        args.seed = args.seed if args.seed is not None else 20260823
        args.output = args.output or DEFAULT_OUTPUT
        output = build_prediction_dataset(args, semantic_policy)
        print(f"Wrote verified Semantic-Theta*/LLM pairs to {output}")


if __name__ == "__main__":
    main()
