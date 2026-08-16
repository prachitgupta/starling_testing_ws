#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from rrt import plan_rrt, point_clear, segment_clear


SCRIPT_DIR = Path(__file__).resolve().parent
FINE_TUNING_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = FINE_TUNING_DIR.parent
PROMPT_GENERATOR_PATH = PACKAGE_DIR / "scripts" / "prompt_generator.py"
DEFAULT_DATASET_DIR = FINE_TUNING_DIR / "datasets"
DEFAULT_WORKSPACE = {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": -0.25}
DEFAULT_GOAL = {"x": 2.5, "y": 0.0, "z": -0.25}
DEFAULT_CLEARANCE_M = 0.40
DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "conformal_rrt_calibration_dataset_2001.csv"
DEFAULT_VLLM_BASE_URL = "http://172.22.224.93:8000/v1"
DEFAULT_LLAMA_MODEL_NAME = "rrt_planner"
REFINEMENT_PATH = PACKAGE_DIR / "scripts" / "refinment.py"
VERIFIER_PATH = PACKAGE_DIR / "scripts" / "verifier.py"
SOURCE_FIELDNAMES = [
    "sample_id",
    "start",
    "goal",
    "workspace",
    "obstacles",
    "rrt_waypoints",
    "llm_waypoints",
    "rrt_verified_waypoints",
    "llm_verified_waypoints",
    "rrt_verification_metrics",
    "llm_verification_metrics",
    "llama_model_name",
    "vllm_base_url",
    "prompt",
]


def load_prompt_generator():
    spec = importlib.util.spec_from_file_location("offline_prompt_generator", PROMPT_GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prompt_from_current_generator(prompt_module, start, goal, workspace, obstacles):
    formatter = object.__new__(prompt_module.PromptGenerator)
    formatter.workspace_x = tuple(workspace["x"])
    formatter.workspace_y = tuple(workspace["y"])
    formatter.fixed_z = float(workspace["z"])
    formatter.clearance_m = DEFAULT_CLEARANCE_M
    obstacle_lines = []
    for index, obstacle in enumerate(obstacles, start=1):
        min_corner = obstacle["min_corner"]
        max_corner = obstacle["max_corner"]
        obstacle_lines.append(
            f"{index} obstacle: x=[{min_corner[0]:.2f},{max_corner[0]:.2f}], "
            f"y=[{min_corner[1]:.2f},{max_corner[1]:.2f}]."
        )
    obstacle_text = " ".join(obstacle_lines) if obstacle_lines else "No obstacles currently detected."
    distance = math.hypot(goal["x"] - start["x"], goal["y"] - start["y"])
    nl_env = (
        "Mission state: the UAV has already taken off and is holding hover at the start position. "
        "Use this hover position as the first waypoint/reference for planning. "
        f"Workspace: x=[{workspace['x'][0]:.2f},{workspace['x'][1]:.2f}]m, "
        f"y=[{workspace['y'][0]:.2f},{workspace['y'][1]:.2f}]m, z={workspace['z']:.2f} fixed. "
        f"Start: ({start['x']:.2f},{start['y']:.2f},{workspace['z']:.2f}), "
        f"Goal: ({goal['x']:.2f},{goal['y']:.2f},{goal['z']:.2f}), "
        f"distance≈{distance:.2f}m. Obstacles with x-y spans: {obstacle_text}"
    )
    prompt = "\n".join([prompt_module.INSTRUCTIONS, nl_env, prompt_module.PromptGenerator.constraints(formatter)])
    return prompt, nl_env


def obstacle_template(index, min_x, min_y, width, depth, fixed_z):
    return {
        "id": index,
        "label": f"box_{index}",
        "shape": "box",
        "min_corner": [round(min_x, 2), round(min_y, 2), round(fixed_z - 0.5, 2)],
        "max_corner": [round(min_x + width, 2), round(min_y + depth, 2), round(fixed_z + 0.5, 2)],
        "size": [round(width, 2), round(depth, 2), 1.0],
    }


def random_point(workspace):
    return {
        "x": round(random.uniform(float(workspace["x"][0]), float(workspace["x"][1])), 2),
        "y": round(random.uniform(float(workspace["y"][0]), float(workspace["y"][1])), 2),
        "z": float(workspace["z"]),
    }


def sample_obstacles(workspace, start, goal, count, clearance_m):
    obstacles = []
    fixed_z = float(workspace["z"])
    for index in range(1, count + 1):
        for _ in range(100):
            width = random.uniform(0.25, 0.75)
            depth = random.uniform(0.25, 0.75)
            min_x = random.uniform(float(workspace["x"][0]), float(workspace["x"][1]) - width)
            min_y = random.uniform(float(workspace["y"][0]), float(workspace["y"][1]) - depth)
            obstacle = obstacle_template(index, min_x, min_y, width, depth, fixed_z)
            candidate = obstacles + [obstacle]
            if point_clear(start, candidate, workspace, clearance_m) and point_clear(goal, candidate, workspace, clearance_m):
                obstacles.append(obstacle)
                break
    return obstacles


def sample_environment(workspace, fixed_goal, clearance_m):
    for _ in range(200):
        start = random_point(workspace)
        goal = dict(fixed_goal) if fixed_goal else random_point(workspace)
        obstacle_count = random.randint(0, 4)
        obstacles = sample_obstacles(workspace, start, goal, obstacle_count, clearance_m)
        if math.hypot(goal["x"] - start["x"], goal["y"] - start["y"]) < 1.0:
            continue
        if segment_clear(start, goal, obstacles, workspace, clearance_m):
            return start, goal, obstacles
        return start, goal, obstacles
    raise RuntimeError("Failed to sample a usable environment.")


def completion_from_path(path):
    return {
        "reasoning": "RRT expert route selected to avoid inflated obstacle boxes while progressing to the goal.",
        "waypoints": path,
    }


def write_rows(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with tmp_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp_output.replace(output_csv)


def generate_training_dataset(samples, output_csv, seed, use_fixed_goal):
    random.seed(seed)
    prompt_module = load_prompt_generator()
    workspace = dict(DEFAULT_WORKSPACE)
    fixed_goal = DEFAULT_GOAL if use_fixed_goal else None
    rows = []
    attempts = 0

    while len(rows) < samples and attempts < samples * 25:
        attempts += 1
        start, goal, obstacles = sample_environment(workspace, fixed_goal, DEFAULT_CLEARANCE_M)
        try:
            path = plan_rrt(
                start,
                goal,
                obstacles,
                workspace=workspace,
                clearance_m=DEFAULT_CLEARANCE_M,
                seed=seed + attempts,
            )
        except (RuntimeError, ValueError):
            continue

        prompt, nl_env = prompt_from_current_generator(prompt_module, start, goal, workspace, obstacles)
        completion = completion_from_path(path)
        rows.append(
            {
                "sample_id": len(rows),
                "prompt": prompt,
                "completion": json.dumps(completion, separators=(",", ":")),
                "messages": json.dumps(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(completion, separators=(",", ":"))},
                    ],
                    separators=(",", ":"),
                ),
                "nl_env": nl_env,
                "start": json.dumps(start, separators=(",", ":")),
                "goal": json.dumps(goal, separators=(",", ":")),
                "workspace": json.dumps(workspace, separators=(",", ":")),
                "obstacles": json.dumps(obstacles, separators=(",", ":")),
                "waypoints": json.dumps(path, separators=(",", ":")),
            }
        )
        write_rows(rows, output_csv)

    if len(rows) < samples:
        raise RuntimeError(f"Generated {len(rows)} samples after {attempts} attempts; requested {samples}.")

    return output_csv


class Waypoint(BaseModel):
    x: float
    y: float
    z: float


class WaypointPlan(BaseModel):
    reasoning: str = Field(
        ...,
        description="2-3 concise statements explaining obstacle-avoidance routing.",
    )
    waypoints: List[Waypoint] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Sparse ordered NED waypoints. Final waypoint must be the goal.",
    )


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_refiner():
    module = load_module("offline_path_refinement", REFINEMENT_PATH)
    refiner = object.__new__(module.PathRefinement)
    refiner.interpolation_spacing_m = module.INTERPOLATION_SPACING_M
    refiner.safety_margin_m = module.SAFETY_MARGIN_M
    refiner.nudge_epsilon_m = module.NUDGE_EPSILON_M
    refiner.fixed_z = module.FIXED_Z
    return refiner


def make_verifier():
    module = load_module("offline_path_verifier", VERIFIER_PATH)
    verifier = object.__new__(module.PathVerifier)
    verifier.safety_margin_m = module.SAFETY_MARGIN_M
    verifier.interpolation_spacing_m = module.INTERPOLATION_SPACING_M
    verifier.cruise_speed_mps = module.CRUISE_SPEED_MPS
    verifier.max_velocity_mps = module.MAX_VELOCITY_MPS
    verifier.max_acceleration_mps2 = module.MAX_ACCELERATION_MPS2
    verifier.goal_tolerance_m = module.GOAL_TOLERANCE_M
    verifier.start_tolerance_m = module.START_TOLERANCE_M
    verifier.progress_tolerance_m = module.PROGRESS_TOLERANCE_M
    verifier.nominal_dt_s = verifier.interpolation_spacing_m / verifier.cruise_speed_mps
    return verifier


def refine_and_verify(refiner, verifier, waypoints, row):
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
    return refined, metrics


def request_llama_waypoints(client, model_name, prompt, temperature, max_retries):
    last_error = None
    for _ in range(max_retries):
        try:
            plan = client.chat.completions.create(
                model=model_name,
                response_model=WaypointPlan,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return [waypoint.model_dump() for waypoint in plan.waypoints]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Llama waypoint request failed after {max_retries} attempts: {last_error}")


def write_prediction(prediction, index, args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if index == 0 else "a"
    with args.output.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDNAMES, lineterminator="\n")
        if index == 0:
            writer.writeheader()
        row, candidate, rrt_verified, llm_verified, rrt_metrics, llm_metrics, prompt = prediction
        writer.writerow(
            {
                "sample_id": index,
                "start": json.dumps(row["start"], separators=(",", ":")),
                "goal": json.dumps(row["goal"], separators=(",", ":")),
                "workspace": json.dumps(row["workspace"], separators=(",", ":")),
                "obstacles": json.dumps(row["obstacles"], separators=(",", ":")),
                "rrt_waypoints": json.dumps(row["rrt_label"], separators=(",", ":")),
                "llm_waypoints": json.dumps(candidate, separators=(",", ":")),
                "rrt_verified_waypoints": json.dumps(rrt_verified, separators=(",", ":")),
                "llm_verified_waypoints": json.dumps(llm_verified, separators=(",", ":")),
                "rrt_verification_metrics": json.dumps(rrt_metrics, separators=(",", ":")),
                "llm_verification_metrics": json.dumps(llm_metrics, separators=(",", ":")),
                "llama_model_name": args.llama_model_name,
                "vllm_base_url": args.vllm_base_url,
                "prompt": prompt,
            }
        )


def build_dataset(args):
    import instructor
    from openai import OpenAI

    random.seed(args.seed)
    workspace = dict(DEFAULT_WORKSPACE)
    print(
        f"Starting calibration dataset generation: target_samples={args.samples}, seed={args.seed}, "
        f"vllm={args.vllm_base_url}, model={args.llama_model_name}",
        flush=True,
    )
    raw_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
    prompt_module = load_prompt_generator()
    refiner = make_refiner()
    verifier = make_verifier()
    predictions = []
    attempts = 0
    while len(predictions) < args.samples and attempts < args.samples * 30:
        attempts += 1
        print(f"[attempt {attempts}] sampling environment ({len(predictions)}/{args.samples} accepted)", flush=True)
        start, goal, obstacles = sample_environment(workspace, None, DEFAULT_CLEARANCE_M)
        print(
            f"[attempt {attempts}] start={start}, goal={goal}, obstacles={len(obstacles)}",
            flush=True,
        )
        try:
            rrt_raw = plan_rrt(start, goal, obstacles, workspace=workspace, clearance_m=DEFAULT_CLEARANCE_M, seed=args.seed + attempts)
        except (RuntimeError, ValueError) as exc:
            print(f"[attempt {attempts}] skipped: RRT failed: {exc}", flush=True)
            continue
        print(f"[attempt {attempts}] RRT raw waypoints={len(rrt_raw)}", flush=True)
        row = {"start": start, "goal": goal, "workspace": workspace, "obstacles": obstacles, "rrt_label": rrt_raw}
        prompt, _ = prompt_from_current_generator(
            prompt_module,
            row["start"],
            row["goal"],
            row["workspace"],
            row["obstacles"],
        )
        print(f"[attempt {attempts}] querying Llama with prompt_chars={len(prompt)}", flush=True)
        try:
            candidate = request_llama_waypoints(
                client,
                args.llama_model_name,
                prompt,
                args.temperature,
                args.llm_retries,
            )
        except RuntimeError as exc:
            print(f"[attempt {attempts}] skipped: Llama request failed: {exc}", flush=True)
            continue
        print(f"[attempt {attempts}] LLM raw waypoints={len(candidate)}", flush=True)
        try:
            rrt_verified, rrt_metrics = refine_and_verify(refiner, verifier, row["rrt_label"], row)
            print(f"[attempt {attempts}] RRT verified waypoints={len(rrt_verified)}", flush=True)
            llm_verified, llm_metrics = refine_and_verify(refiner, verifier, candidate, row)
            print(f"[attempt {attempts}] LLM verified waypoints={len(llm_verified)}", flush=True)
        except ValueError as exc:
            print(f"[attempt {attempts}] skipped: {exc}", flush=True)
            continue
        predictions.append((row, candidate, rrt_verified, llm_verified, rrt_metrics, llm_metrics, prompt))
        write_prediction(predictions[-1], len(predictions) - 1, args)
        print(f"[accepted {len(predictions)}/{args.samples}] wrote prediction-pair checkpoint to {args.output}", flush=True)

    if len(predictions) < args.samples:
        raise RuntimeError(f"Only built {len(predictions)} verified rows after {attempts} attempts; requested {args.samples}.")

    return args.output


def main():
    parser = argparse.ArgumentParser(
        description="Generate verified RRT/Llama prediction pairs or RRT instruction-training data."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rrt-training", action="store_true", help="Generate RRT instruction-training data without contacting vLLM.")
    parser.add_argument("--random-goal", action="store_true", help="Sample random goals in RRT training mode.")
    parser.add_argument("--llama-model-name", default=DEFAULT_LLAMA_MODEL_NAME)
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    args = parser.parse_args()

    if args.rrt_training:
        args.samples = args.samples or 100
        args.seed = args.seed if args.seed is not None else 7
        args.output = args.output or (DEFAULT_DATASET_DIR / "rrt_expert_dataset.csv")
        output = generate_training_dataset(args.samples, args.output, args.seed, use_fixed_goal=not args.random_goal)
        print(f"Wrote RRT instruction-training data to {output}")
    else:
        args.samples = args.samples or 2001
        args.seed = args.seed if args.seed is not None else 20260618
        args.output = args.output or DEFAULT_OUTPUT
        output = build_dataset(args)
        print(f"Wrote verified RRT/Llama prediction pairs to {output}")


if __name__ == "__main__":
    main()
