#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import List

import instructor
from dataset_generator import (
    DEFAULT_CLEARANCE_M,
    DEFAULT_DATASET_DIR,
    DEFAULT_WORKSPACE,
    load_prompt_generator,
    prompt_from_current_generator,
    sample_environment,
)
from min_snap import generate_trajectory
from pydantic import BaseModel, Field
from rrt import plan_rrt


DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "conformal_rrt_calibration_dataset.csv"
DEFAULT_VLLM_BASE_URL = "http://172.22.224.93:8000/v1"
DEFAULT_LLAMA_MODEL_NAME = "rrt_planner"


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


def interpolate_vector(samples, key, fraction):
    if len(samples) == 1:
        return samples[0][key]
    position = fraction * (len(samples) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(samples) - 1)
    ratio = position - low
    return [
        float(samples[low][key][index]) + (float(samples[high][key][index]) - float(samples[low][key][index])) * ratio
        for index in range(len(samples[low][key]))
    ]


def score_trajectories(rrt_trajectory, llm_trajectory):
    rrt_samples = rrt_trajectory["samples"]
    llm_samples = llm_trajectory["samples"]
    count = max(len(rrt_samples), len(llm_samples), 2)
    s_u = 0.0
    s_x = 0.0
    for index in range(count):
        fraction = index / (count - 1)
        rrt_x = interpolate_vector(rrt_samples, "x", fraction)
        llm_x = interpolate_vector(llm_samples, "x", fraction)
        rrt_u = interpolate_vector(rrt_samples, "u", fraction)
        llm_u = interpolate_vector(llm_samples, "u", fraction)
        s_x = max(s_x, math.sqrt(sum((llm_x[i] - rrt_x[i]) ** 2 for i in range(4))))
        s_u = max(s_u, math.sqrt(sum((llm_u[i] - rrt_u[i]) ** 2 for i in range(2))))
    return {"s_u": round(s_u, 6), "s_x": round(s_x, 6)}


def conformal_quantile(values, delta):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def sample_rrt_rows(count, seed):
    random.seed(seed)
    rows = []
    workspace = dict(DEFAULT_WORKSPACE)
    attempts = 0
    while len(rows) < count and attempts < count * 30:
        attempts += 1
        start, goal, obstacles = sample_environment(workspace, None, DEFAULT_CLEARANCE_M)
        try:
            label = plan_rrt(start, goal, obstacles, workspace=workspace, clearance_m=DEFAULT_CLEARANCE_M, seed=seed + attempts)
        except (RuntimeError, ValueError):
            continue
        rows.append({"start": start, "goal": goal, "workspace": workspace, "obstacles": obstacles, "rrt_label": label})
    return rows


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


def build_dataset(args):
    rows = sample_rrt_rows(args.samples, args.seed)
    if len(rows) < args.samples:
        raise RuntimeError(f"Only built {len(rows)} rows; requested {args.samples}.")

    from openai import OpenAI

    raw_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
    prompt_module = load_prompt_generator()
    scored = []
    for index, row in enumerate(rows[: args.samples]):
        prompt, _ = prompt_from_current_generator(
            prompt_module,
            row["start"],
            row["goal"],
            row["workspace"],
            row["obstacles"],
        )
        candidate = request_llama_waypoints(
            client,
            args.llama_model_name,
            prompt,
            args.temperature,
            args.llm_retries,
        )
        rrt_trajectory = generate_trajectory(row["rrt_label"], row["workspace"], row["obstacles"], dt=args.dt)
        llm_trajectory = generate_trajectory(candidate, row["workspace"], row["obstacles"], dt=args.dt)
        scores = score_trajectories(rrt_trajectory, llm_trajectory)
        scored.append((row, candidate, rrt_trajectory, llm_trajectory, scores, prompt))
        print(f"Scored sample {index + 1}/{args.samples}: s_u={scores['s_u']}, s_x={scores['s_x']}")

    split = max(1, int(len(scored) * args.calibration_fraction))
    q_u = conformal_quantile([item[4]["s_u"] for item in scored[:split]], args.delta_u)
    q_x = conformal_quantile([item[4]["s_x"] for item in scored[:split]], args.delta_x)

    fieldnames = [
        "sample_id",
        "start",
        "goal",
        "workspace",
        "obstacles",
        "rrt_waypoints",
        "llm_waypoints",
        "rrt_trajectory",
        "llm_trajectory",
        "s_u",
        "s_x",
        "q_u",
        "q_x",
        "delta_u",
        "delta_x",
        "accepted",
        "llama_model_name",
        "vllm_base_url",
        "prompt",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, candidate, rrt_trajectory, llm_trajectory, scores, prompt) in enumerate(scored):
            writer.writerow(
                {
                    "sample_id": index,
                    "start": json.dumps(row["start"], separators=(",", ":")),
                    "goal": json.dumps(row["goal"], separators=(",", ":")),
                    "workspace": json.dumps(row["workspace"], separators=(",", ":")),
                    "obstacles": json.dumps(row["obstacles"], separators=(",", ":")),
                    "rrt_waypoints": json.dumps(row["rrt_label"], separators=(",", ":")),
                    "llm_waypoints": json.dumps(candidate, separators=(",", ":")),
                    "rrt_trajectory": json.dumps(rrt_trajectory, separators=(",", ":")),
                    "llm_trajectory": json.dumps(llm_trajectory, separators=(",", ":")),
                    "s_u": scores["s_u"],
                    "s_x": scores["s_x"],
                    "q_u": round(q_u, 6),
                    "q_x": round(q_x, 6),
                    "delta_u": args.delta_u,
                    "delta_x": args.delta_x,
                    "accepted": scores["s_u"] <= q_u and scores["s_x"] <= q_x,
                    "llama_model_name": args.llama_model_name,
                    "vllm_base_url": args.vllm_base_url,
                    "prompt": prompt,
                }
            )
    return args.output


def main():
    parser = argparse.ArgumentParser(
        description="Build fresh prompt-compatible conformal calibration CSV data with RRT labels and Llama predictions."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--delta-u", type=float, default=0.05)
    parser.add_argument("--delta-x", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.8)
    parser.add_argument("--llama-model-name", default=DEFAULT_LLAMA_MODEL_NAME)
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    args = parser.parse_args()

    output = build_dataset(args)
    print(f"Wrote prompt-compatible RRT conformal calibration dataset to {output}")


if __name__ == "__main__":
    main()
