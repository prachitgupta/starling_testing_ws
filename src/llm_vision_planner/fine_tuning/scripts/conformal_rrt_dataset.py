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
PACKAGE_DIR = Path(__file__).resolve().parents[2]
REFINEMENT_PATH = PACKAGE_DIR / "scripts" / "refinment.py"
VERIFIER_PATH = PACKAGE_DIR / "scripts" / "verifier.py"


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
    verifier.progress_tolerance_m = module.PROGRESS_TOLERANCE_M
    verifier.nominal_dt_s = verifier.interpolation_spacing_m / verifier.cruise_speed_mps
    return verifier


def refine_and_verify(refiner, verifier, waypoints, row):
    refined = refiner.interpolate_waypoints(waypoints, row["workspace"], row["obstacles"])
    payload = {
        "waypoints": refined,
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


def build_dataset(args):
    from openai import OpenAI

    random.seed(args.seed)
    workspace = dict(DEFAULT_WORKSPACE)
    print(
        f"Starting calibration dataset generation: target_samples={args.samples}, seed={args.seed}, "
        f"dt={args.dt}, vllm={args.vllm_base_url}, model={args.llama_model_name}",
        flush=True,
    )
    raw_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
    prompt_module = load_prompt_generator()
    refiner = make_refiner()
    verifier = make_verifier()
    scored = []
    attempts = 0
    while len(scored) < args.samples and attempts < args.samples * 30:
        attempts += 1
        print(f"[attempt {attempts}] sampling environment ({len(scored)}/{args.samples} accepted)", flush=True)
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
        print(f"[attempt {attempts}] running min-snap trajectories", flush=True)
        rrt_trajectory = generate_trajectory(rrt_verified, row["workspace"], row["obstacles"], dt=args.dt)
        llm_trajectory = generate_trajectory(llm_verified, row["workspace"], row["obstacles"], dt=args.dt)
        print(
            f"[attempt {attempts}] trajectory samples: rrt={len(rrt_trajectory['samples'])}, "
            f"llm={len(llm_trajectory['samples'])}",
            flush=True,
        )
        scores = score_trajectories(rrt_trajectory, llm_trajectory)
        scored.append((row, candidate, rrt_verified, llm_verified, rrt_metrics, llm_metrics, rrt_trajectory, llm_trajectory, scores, prompt))
        print(f"[accepted {len(scored)}/{args.samples}] s_u={scores['s_u']}, s_x={scores['s_x']}", flush=True)

    if len(scored) < args.samples:
        raise RuntimeError(f"Only built {len(scored)} verified rows after {attempts} attempts; requested {args.samples}.")

    split = max(1, int(len(scored) * args.calibration_fraction))
    q_u = conformal_quantile([item[8]["s_u"] for item in scored[:split]], args.delta_u)
    q_x = conformal_quantile([item[8]["s_x"] for item in scored[:split]], args.delta_x)
    print(
        f"Computed quantiles from first {split} rows: q_u={round(q_u, 6)}, q_x={round(q_x, 6)}",
        flush=True,
    )

    fieldnames = [
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
    print(f"Writing {len(scored)} rows to {args.output}", flush=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, candidate, rrt_verified, llm_verified, rrt_metrics, llm_metrics, rrt_trajectory, llm_trajectory, scores, prompt) in enumerate(scored):
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
