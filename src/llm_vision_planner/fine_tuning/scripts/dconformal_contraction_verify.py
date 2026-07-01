#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from lqr import A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, DAMPING, solve_care
from min_snap import generate_trajectory


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "datasets"
PLOTS_DIR = SCRIPT_DIR.parent / "plots"
DEFAULT_CALIBRATION_CSV = DATASET_DIR / "conformal_rrt_calibration_dataset.csv"
DEFAULT_OUTPUT_PNG = PLOTS_DIR / "dconformal_contraction_verification.png"
DEFAULT_REPORT_JSON = PLOTS_DIR / "dconformal_contraction_verification.json"


def parse_json_field(value):
    return json.loads(value) if isinstance(value, str) else value


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def conformal_quantile(values, delta):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def trajectory_from_row(row, prefix):
    trajectory_key = f"{prefix}_trajectory"
    if row.get(trajectory_key):
        return parse_json_field(row[trajectory_key])
    workspace = parse_json_field(row["workspace"])
    obstacles = parse_json_field(row["obstacles"])
    return generate_trajectory(parse_json_field(row[f"{prefix}_waypoints"]), workspace, obstacles)


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


def dynamics(state, control):
    return A_DOUBLE_INTEGRATOR @ state + B_DOUBLE_INTEGRATOR @ control


def propagate_controller(rrt_trajectory, llm_trajectory, k, dt):
    horizon = min(float(rrt_trajectory["samples"][-1]["t"]), float(llm_trajectory["samples"][-1]["t"]))
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    state = np.array(rrt_trajectory["samples"][0]["x"], dtype=float)
    result = []
    for t in times:
        xhat, uhat = interpolate_sample(llm_trajectory["samples"], float(t))
        control = -k @ (state - xhat) + uhat
        xd, _ = interpolate_sample(rrt_trajectory["samples"], float(t))
        result.append({"t": float(t), "x": state.tolist(), "u": control.tolist(), "xd": xd.tolist(), "xhat": xhat.tolist()})
        state = state + dt * dynamics(state, control)
    return result


def tube_profile(closed_loop, p, b, k, alpha, q_u, q_x):
    eigvals = np.linalg.eigvalsh(p)
    lambda_min = float(np.min(eigvals))
    lambda_max = float(np.max(eigvals))
    b_norm = float(np.linalg.norm(b, 2))
    bk_norm = float(np.linalg.norm(b @ k, 2))
    initial_error = np.linalg.norm(np.array(closed_loop[0]["x"]) - np.array(closed_loop[0]["xd"]))
    profile = []
    for sample in closed_loop:
        t = float(sample["t"])
        decay = math.exp(-alpha * t)
        radius = math.sqrt(lambda_max / lambda_min) * initial_error * decay
        radius += math.sqrt(lambda_max) * (b_norm * q_u + bk_norm * q_x) * (1.0 - decay) / (alpha * math.sqrt(lambda_min))
        profile.append({"t": t, "radius": radius})
    return profile


def draw_scene(output_png, workspace, obstacles, rrt_samples, llm_samples, closed_loop, tube, result):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(workspace["x"][0]) - 0.25, float(workspace["x"][1]) + 0.25)
    axis.set_ylim(float(workspace["y"][0]) - 0.25, float(workspace["y"][1]) + 0.25)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Double-integrator conformal tracking")
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

    for sample, tube_sample in zip(closed_loop[:: max(1, len(closed_loop) // 30)], tube[:: max(1, len(tube) // 30)]):
        xd = sample["xd"]
        axis.add_patch(Circle((xd[0], xd[1]), tube_sample["radius"], facecolor="#60a5fa", edgecolor="none", alpha=0.10))

    axis.plot([s["x"][0] for s in rrt_samples], [s["x"][1] for s in rrt_samples], color="#2563eb", label="desired RRT")
    axis.plot([s["x"][0] for s in llm_samples], [s["x"][1] for s in llm_samples], "--", color="#f97316", label="LLM reference")
    axis.plot([s["x"][0] for s in closed_loop], [s["x"][1] for s in closed_loop], color="#16a34a", label="controlled")
    axis.scatter([rrt_samples[0]["x"][0]], [rrt_samples[0]["x"][1]], color="#111827", s=60, marker="s", label="start")
    axis.scatter([rrt_samples[-1]["x"][0]], [rrt_samples[-1]["x"][1]], color="#7c3aed", s=100, marker="*", label="goal")
    axis.text(
        0.02,
        0.02,
        f"P >= {result['joint_probability_lower_bound']:.2f}\nq_u={result['q_u']:.3f}, q_x={result['q_x']:.3f}\nmax tube={result['max_tube_radius_m']:.3f} m",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Verify double-integrator conformal tracking with slide-4 control law.")
    parser.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION_CSV)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--delta-u", type=float, default=0.05)
    parser.add_argument("--delta-x", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    args = parser.parse_args()

    rows = load_rows(args.calibration_csv)
    if not rows:
        raise RuntimeError(f"No calibration rows found in {args.calibration_csv}")
    if args.sample_id < 0 or args.sample_id >= len(rows):
        raise ValueError(f"--sample-id must be between 0 and {len(rows) - 1}.")
    calibration_rows = rows[: args.calibration_samples] if args.calibration_samples else rows
    row = rows[args.sample_id]

    q_u = conformal_quantile([item["s_u"] for item in calibration_rows], args.delta_u)
    q_x = conformal_quantile([item["s_x"] for item in calibration_rows], args.delta_x)
    p, k, alpha = solve_care()
    rrt_trajectory = trajectory_from_row(row, "rrt")
    llm_trajectory = trajectory_from_row(row, "llm")
    closed_loop = propagate_controller(rrt_trajectory, llm_trajectory, k, args.dt)
    tube = tube_profile(closed_loop, p, B_DOUBLE_INTEGRATOR, k, alpha, q_u, q_x)
    max_tracking_error = max(np.linalg.norm(np.array(sample["x"]) - np.array(sample["xd"])) for sample in closed_loop)
    max_tube_radius = max(sample["radius"] for sample in tube)
    result = {
        "sample_id": args.sample_id,
        "calibration_samples": len(calibration_rows),
        "q_u": round(q_u, 6),
        "q_x": round(q_x, 6),
        "delta_u": args.delta_u,
        "delta_x": args.delta_x,
        "joint_probability_lower_bound": round(1.0 - args.delta_u - args.delta_x, 6),
        "alpha": round(alpha, 6),
        "K": k.round(6).tolist(),
        "damping": DAMPING,
        "max_tracking_error": round(float(max_tracking_error), 6),
        "max_tube_radius_m": round(float(max_tube_radius), 6),
        "inside_tube": bool(max_tracking_error <= max_tube_radius),
        "output_png": str(args.output_png),
    }

    workspace = parse_json_field(row["workspace"])
    obstacles = parse_json_field(row["obstacles"])
    draw_scene(args.output_png, workspace, obstacles, rrt_trajectory["samples"], llm_trajectory["samples"], closed_loop, tube, result)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
