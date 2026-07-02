#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import instructor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from openai import OpenAI

from conformal_rrt_dataset import (
    DEFAULT_LLAMA_MODEL_NAME,
    DEFAULT_OUTPUT,
    DEFAULT_VLLM_BASE_URL,
    conformal_quantile,
    make_refiner,
    make_verifier,
    prompt_from_current_generator,
    refine_and_verify,
    request_llama_waypoints,
)
from dataset_generator import DEFAULT_WORKSPACE, load_prompt_generator
from lqr import A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, DAMPING, solve_care
from min_snap import generate_trajectory


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSON = SCRIPT_DIR.parent / "plots" / "live_dconformal_controller.json"
DEFAULT_OUTPUT_PNG = SCRIPT_DIR.parent / "plots" / "live_dconformal_controller.png"


def load_json(value):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def load_quantiles(path, calibration_samples, delta_u, delta_x):
    if not path or not path.exists():
        return None, None, 0
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = rows[:calibration_samples] if calibration_samples else rows
    if not rows or "s_u" not in rows[0] or "s_x" not in rows[0]:
        return None, None, len(rows)
    return (
        conformal_quantile([row["s_u"] for row in rows], delta_u),
        conformal_quantile([row["s_x"] for row in rows], delta_x),
        len(rows),
    )


def feedback_prompt(base_prompt, metrics):
    failed = metrics.get("failed_constraints", [])
    table = metrics.get("feedback_table", "")
    return (
        base_prompt
        + "\nPrevious plan failed verification. Regenerate waypoints for the same start, goal, workspace, and obstacles.\n"
        + f"Failed constraints: {', '.join(failed) if failed else 'unknown'}\n"
        + table
    )


def query_verified_llm_trajectory(args, row):
    raw_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
    prompt_module = load_prompt_generator()
    refiner = make_refiner()
    verifier = make_verifier()
    base_prompt, _ = prompt_from_current_generator(prompt_module, row["start"], row["goal"], row["workspace"], row["obstacles"])
    prompt = base_prompt
    last_metrics = None

    for attempt in range(1, args.llm_attempts + 1):
        print(f"[live attempt {attempt}/{args.llm_attempts}] querying Llama with prompt_chars={len(prompt)}", flush=True)
        try:
            candidate = request_llama_waypoints(client, args.llama_model_name, prompt, args.temperature, args.llm_retries)
        except RuntimeError as exc:
            print(f"[live attempt {attempt}] Llama request failed: {exc}", flush=True)
            continue
        print(f"[live attempt {attempt}] LLM raw waypoints={len(candidate)}", flush=True)
        try:
            verified, metrics = refine_and_verify(refiner, verifier, candidate, row)
        except ValueError as exc:
            refined = refiner.interpolate_waypoints(candidate, row["workspace"], row["obstacles"])
            payload = {
                "waypoints": refined,
                "obstacles": row["obstacles"],
                "workspace": row["workspace"],
                "goal": row["goal"],
            }
            last_metrics = verifier.compute_metrics(payload)
            print(
                f"[live attempt {attempt}] verification failed: {exc}; "
                f"failed_constraints={last_metrics['failed_constraints']}",
                flush=True,
            )
            prompt = feedback_prompt(base_prompt, last_metrics)
            continue
        print(f"[live attempt {attempt}] verified waypoints={len(verified)}", flush=True)
        trajectory = generate_trajectory(verified, row["workspace"], row["obstacles"], dt=args.trajectory_dt)
        print(f"[live attempt {attempt}] min-snap samples={len(trajectory['samples'])}", flush=True)
        return candidate, verified, metrics, trajectory, prompt

    raise RuntimeError(f"LLM plan did not pass verification after {args.llm_attempts} attempts: {last_metrics}")


def interpolate_sample(samples, t):
    if t <= float(samples[0]["t"]):
        return np.array(samples[0]["x"], dtype=float), np.array(samples[0]["u"], dtype=float)
    if t >= float(samples[-1]["t"]):
        return np.array(samples[-1]["x"], dtype=float), np.array(samples[-1]["u"], dtype=float)
    for index in range(1, len(samples)):
        if float(samples[index]["t"]) >= t:
            before = samples[index - 1]
            after = samples[index]
            span = max(float(after["t"]) - float(before["t"]), 1e-9)
            ratio = (t - float(before["t"])) / span
            x = np.array(before["x"], dtype=float) + (np.array(after["x"], dtype=float) - np.array(before["x"], dtype=float)) * ratio
            u = np.array(before["u"], dtype=float) + (np.array(after["u"], dtype=float) - np.array(before["u"], dtype=float)) * ratio
            return x, u
    return np.array(samples[-1]["x"], dtype=float), np.array(samples[-1]["u"], dtype=float)


def propagate_live(trajectory, k, dt, initial_state):
    horizon = float(trajectory["samples"][-1]["t"])
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    state = np.array(initial_state, dtype=float)
    result = []
    for t in times:
        xhat, uhat = interpolate_sample(trajectory["samples"], float(t))
        control = -k @ (state - xhat) + uhat
        result.append({"t": float(t), "x": state.tolist(), "xhat": xhat.tolist(), "uhat": uhat.tolist(), "u": control.tolist()})
        state = state + dt * (A_DOUBLE_INTEGRATOR @ state + B_DOUBLE_INTEGRATOR @ control)
    return result


def draw_scene(output_png, workspace, obstacles, trajectory, closed_loop):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(workspace["x"][0]) - 0.25, float(workspace["x"][1]) + 0.25)
    axis.set_ylim(float(workspace["y"][0]) - 0.25, float(workspace["y"][1]) + 0.25)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Live LLM double-integrator controller")
    axis.grid(True, alpha=0.25)
    for obstacle in obstacles:
        min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
        max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
        axis.add_patch(
            Rectangle(
                (float(min_corner[0]), float(min_corner[1])),
                float(max_corner[0]) - float(min_corner[0]),
                float(max_corner[1]) - float(min_corner[1]),
                facecolor="#ef4444",
                edgecolor="#991b1b",
                alpha=0.35,
            )
        )
    axis.plot([s["x"][0] for s in trajectory["samples"]], [s["x"][1] for s in trajectory["samples"]], "--", color="#f97316", label="verified LLM reference")
    axis.plot([s["x"][0] for s in closed_loop], [s["x"][1] for s in closed_loop], color="#16a34a", label="controlled dynamics")
    axis.scatter([trajectory["samples"][0]["x"][0]], [trajectory["samples"][0]["x"][1]], color="#111827", s=60, marker="s", label="start")
    axis.scatter([trajectory["samples"][-1]["x"][0]], [trajectory["samples"][-1]["x"][1]], color="#7c3aed", s=100, marker="*", label="goal")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Live LLM refinement, verification, min-snap, and double-integrator control.")
    parser.add_argument("--start", required=True, help='JSON point, e.g. {"x":0,"y":0,"z":-0.25}')
    parser.add_argument("--goal", required=True, help='JSON point, e.g. {"x":2.5,"y":0,"z":-0.25}')
    parser.add_argument("--workspace", default=json.dumps(DEFAULT_WORKSPACE))
    parser.add_argument("--obstacles", default="[]")
    parser.add_argument("--calibration-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--delta-u", type=float, default=0.05)
    parser.add_argument("--delta-x", type=float, default=0.05)
    parser.add_argument("--llama-model-name", default=DEFAULT_LLAMA_MODEL_NAME)
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-attempts", type=int, default=3)
    parser.add_argument("--trajectory-dt", type=float, default=0.1)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--initial-state", default=None, help="Optional JSON [px,py,vx,vy]. Defaults to start with zero velocity.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    args = parser.parse_args()

    start = load_json(args.start)
    goal = load_json(args.goal)
    workspace = load_json(args.workspace)
    obstacles = load_json(args.obstacles)
    row = {"start": start, "goal": goal, "workspace": workspace, "obstacles": obstacles}
    raw_waypoints, verified_waypoints, metrics, llm_trajectory, prompt = query_verified_llm_trajectory(args, row)
    p, k, alpha = solve_care()
    initial_state = load_json(args.initial_state) if args.initial_state else [float(start["x"]), float(start["y"]), 0.0, 0.0]
    closed_loop = propagate_live(llm_trajectory, k, args.control_dt, initial_state)
    q_u, q_x, calibration_count = load_quantiles(args.calibration_csv, args.calibration_samples, args.delta_u, args.delta_x)

    report = {
        "start": start,
        "goal": goal,
        "workspace": workspace,
        "obstacles": obstacles,
        "raw_llm_waypoints": raw_waypoints,
        "verified_llm_waypoints": verified_waypoints,
        "verification_metrics": metrics,
        "llm_trajectory": llm_trajectory,
        "closed_loop": closed_loop,
        "K": k.round(6).tolist(),
        "P": p.round(6).tolist(),
        "alpha": round(alpha, 6),
        "damping": DAMPING,
        "q_u": None if q_u is None else round(q_u, 6),
        "q_x": None if q_x is None else round(q_x, 6),
        "calibration_samples": calibration_count,
        "prompt": prompt,
        "output_png": str(args.output_png),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_scene(args.output_png, workspace, obstacles, llm_trajectory, closed_loop)
    print(json.dumps({key: report[key] for key in ("q_u", "q_x", "calibration_samples", "alpha", "K", "output_png")}, indent=2))
    print(f"Wrote live controller report to {args.output_json}")


if __name__ == "__main__":
    main()
