#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

from rrt import plan_rrt, point_clear, segment_clear


SCRIPT_DIR = Path(__file__).resolve().parent
FINE_TUNING_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = FINE_TUNING_DIR.parent
PROMPT_GENERATOR_PATH = PACKAGE_DIR / "scripts" / "prompt_generator.py"
DEFAULT_DATASET_DIR = FINE_TUNING_DIR / "datasets"
DEFAULT_WORKSPACE = {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": -0.25}
DEFAULT_GOAL = {"x": 2.5, "y": 0.0, "z": -0.25}
DEFAULT_CLEARANCE_M = 0.40


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
    nl_env = prompt_module.PromptGenerator.build_nl_env(formatter, start, goal, obstacles)
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


def generate_dataset(samples, output_csv, seed, use_fixed_goal):
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

    if len(rows) < samples:
        raise RuntimeError(f"Generated {len(rows)} samples after {attempts} attempts; requested {samples}.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Generate RRT-labeled instruction tuning data.")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_DIR / "rrt_expert_dataset.csv")
    parser.add_argument("--random-goal", action="store_true", help="Sample random goals instead of using the package default goal.")
    args = parser.parse_args()

    output_csv = generate_dataset(args.samples, args.output, args.seed, use_fixed_goal=not args.random_goal)
    print(f"Wrote dataset to {output_csv}")


if __name__ == "__main__":
    main()
