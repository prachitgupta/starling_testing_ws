#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


DAMPING = 1.1
FIXED_Z = -0.25
INTERPOLATION_SPACING_M = 1.0
SAFETY_MARGIN_M = 0.40
NUDGE_EPSILON_M = 0.02
CRUISE_SPEED_MPS = 0.5
MAX_VELOCITY_MPS = 1.5
MAX_ACCELERATION_MPS2 = 1.5
DT = 0.1
RIDGE = 1e-9


def same_xy(a, b):
    return math.isclose(a["x"], b["x"], abs_tol=1e-6) and math.isclose(a["y"], b["y"], abs_tol=1e-6)


def clamp_to_workspace(point, workspace, fixed_z=FIXED_Z):
    x_limits = workspace.get("x", [0.0, 4.0])
    y_limits = workspace.get("y", [0.0, 4.0])
    return {
        "x": min(max(float(point["x"]), float(x_limits[0])), float(x_limits[1])),
        "y": min(max(float(point["y"]), float(y_limits[0])), float(y_limits[1])),
        "z": fixed_z,
    }


def nudge_away_from_obstacle(point, obstacle, safety_margin_m=SAFETY_MARGIN_M, nudge_epsilon_m=NUDGE_EPSILON_M):
    min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
    max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
    min_x = float(min_corner[0]) - safety_margin_m
    max_x = float(max_corner[0]) + safety_margin_m
    min_y = float(min_corner[1]) - safety_margin_m
    max_y = float(max_corner[1]) + safety_margin_m
    if not (min_x <= point["x"] <= max_x and min_y <= point["y"] <= max_y):
        return point

    left_gap = abs(point["x"] - min_x)
    right_gap = abs(max_x - point["x"])
    bottom_gap = abs(point["y"] - min_y)
    top_gap = abs(max_y - point["y"])
    smallest_gap = min(left_gap, right_gap, bottom_gap, top_gap)
    adjusted = dict(point)
    if smallest_gap == left_gap:
        adjusted["x"] = min_x - nudge_epsilon_m
    elif smallest_gap == right_gap:
        adjusted["x"] = max_x + nudge_epsilon_m
    elif smallest_gap == bottom_gap:
        adjusted["y"] = min_y - nudge_epsilon_m
    else:
        adjusted["y"] = max_y + nudge_epsilon_m
    return adjusted


def sanitize_waypoint(waypoint, workspace, obstacles, preserve_goal, fixed_z=FIXED_Z):
    point = {"x": float(waypoint["x"]), "y": float(waypoint["y"]), "z": fixed_z}
    if not preserve_goal:
        for obstacle in obstacles:
            point = nudge_away_from_obstacle(point, obstacle)
    return clamp_to_workspace(point, workspace, fixed_z)


def refine_waypoints(waypoints, workspace=None, obstacles=None, spacing_m=INTERPOLATION_SPACING_M, fixed_z=FIXED_Z):
    workspace = workspace or {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": fixed_z}
    obstacles = obstacles or []
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required.")

    refined = [sanitize_waypoint(waypoints[0], workspace, obstacles, preserve_goal=False, fixed_z=fixed_z)]
    for start, end in zip(waypoints, waypoints[1:]):
        dx = float(end["x"]) - float(start["x"])
        dy = float(end["y"]) - float(start["y"])
        distance = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(distance / spacing_m)))
        for step in range(1, steps + 1):
            ratio = step / steps
            candidate = {
                "x": float(start["x"]) + dx * ratio,
                "y": float(start["y"]) + dy * ratio,
                "z": fixed_z,
            }
            adjusted = sanitize_waypoint(candidate, workspace, obstacles, preserve_goal=(step == steps), fixed_z=fixed_z)
            if not same_xy(refined[-1], adjusted):
                refined.append(adjusted)
    return refined


def clearance_to_box(point, obstacle):
    min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
    max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
    min_x, max_x = float(min_corner[0]), float(max_corner[0])
    min_y, max_y = float(min_corner[1]), float(max_corner[1])
    dx = max(min_x - float(point["x"]), 0.0, float(point["x"]) - max_x)
    dy = max(min_y - float(point["y"]), 0.0, float(point["y"]) - max_y)
    if dx == 0.0 and dy == 0.0:
        edge_x = min(abs(float(point["x"]) - min_x), abs(max_x - float(point["x"])))
        edge_y = min(abs(float(point["y"]) - min_y), abs(max_y - float(point["y"])))
        return -min(edge_x, edge_y)
    return math.hypot(dx, dy)


def verify_waypoints(waypoints, workspace=None, obstacles=None, max_velocity_mps=MAX_VELOCITY_MPS, max_acceleration_mps2=MAX_ACCELERATION_MPS2):
    workspace = workspace or {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": FIXED_Z}
    obstacles = obstacles or []
    if len(waypoints) < 2:
        raise ValueError("Verification requires at least two waypoints.")
    x_limits = workspace.get("x", [0.0, 4.0])
    y_limits = workspace.get("y", [0.0, 4.0])
    for point in waypoints:
        if not (float(x_limits[0]) <= float(point["x"]) <= float(x_limits[1])):
            raise ValueError("Waypoint outside x workspace bounds.")
        if not (float(y_limits[0]) <= float(point["y"]) <= float(y_limits[1])):
            raise ValueError("Waypoint outside y workspace bounds.")
        if obstacles and min(clearance_to_box(point, obstacle) for obstacle in obstacles) < SAFETY_MARGIN_M:
            raise ValueError("Waypoint violates obstacle clearance.")

    dt = INTERPOLATION_SPACING_M / CRUISE_SPEED_MPS
    velocities = []
    for first, second in zip(waypoints, waypoints[1:]):
        vx = (float(second["x"]) - float(first["x"])) / dt
        vy = (float(second["y"]) - float(first["y"])) / dt
        speed = math.hypot(vx, vy)
        if speed > max_velocity_mps:
            raise ValueError("Waypoint path violates max velocity.")
        velocities.append((vx, vy))
    for first, second in zip(velocities, velocities[1:]):
        accel = math.hypot((second[0] - first[0]) / dt, (second[1] - first[1]) / dt)
        if accel > max_acceleration_mps2:
            raise ValueError("Waypoint path violates max acceleration.")


def derivative_row(t, derivative, degree=7):
    row = np.zeros(degree + 1)
    for power in range(derivative, degree + 1):
        coeff = 1.0
        for value in range(derivative):
            coeff *= power - value
        row[power] = coeff * (t ** (power - derivative))
    return row


def snap_hessian(duration, degree=7):
    q = np.zeros((degree + 1, degree + 1))
    for i in range(4, degree + 1):
        ci = math.prod(range(i - 3, i + 1))
        for j in range(4, degree + 1):
            cj = math.prod(range(j - 3, j + 1))
            q[i, j] = ci * cj * duration ** (i + j - 7) / (i + j - 7)
    return q


def segment_durations(waypoints, cruise_speed_mps=CRUISE_SPEED_MPS):
    durations = []
    for first, second in zip(waypoints, waypoints[1:]):
        distance = math.hypot(float(second["x"]) - float(first["x"]), float(second["y"]) - float(first["y"]))
        durations.append(max(distance / cruise_speed_mps, 0.5))
    return durations


def solve_axis_min_snap(values, durations):
    segments = len(durations)
    degree = 7
    width = degree + 1
    variables = segments * width
    q = np.zeros((variables, variables))
    for segment, duration in enumerate(durations):
        start = segment * width
        q[start : start + width, start : start + width] = snap_hessian(duration, degree)
    q += np.eye(variables) * RIDGE

    rows = []
    rhs = []
    for segment, duration in enumerate(durations):
        start = segment * width
        row = np.zeros(variables)
        row[start : start + width] = derivative_row(0.0, 0, degree)
        rows.append(row)
        rhs.append(values[segment])

        row = np.zeros(variables)
        row[start : start + width] = derivative_row(duration, 0, degree)
        rows.append(row)
        rhs.append(values[segment + 1])

    for derivative in (1, 2):
        row = np.zeros(variables)
        row[:width] = derivative_row(0.0, derivative, degree)
        rows.append(row)
        rhs.append(0.0)

        row = np.zeros(variables)
        row[-width:] = derivative_row(durations[-1], derivative, degree)
        rows.append(row)
        rhs.append(0.0)

    for segment in range(segments - 1):
        left = segment * width
        right = (segment + 1) * width
        for derivative in (1, 2):
            row = np.zeros(variables)
            row[left : left + width] = derivative_row(durations[segment], derivative, degree)
            row[right : right + width] = -derivative_row(0.0, derivative, degree)
            rows.append(row)
            rhs.append(0.0)

    aeq = np.vstack(rows)
    beq = np.array(rhs)
    kkt = np.block([[q, aeq.T], [aeq, np.zeros((aeq.shape[0], aeq.shape[0]))]])
    target = np.concatenate([np.zeros(variables), beq])
    solution = np.linalg.lstsq(kkt, target, rcond=None)[0][:variables]
    return solution.reshape((segments, width))


def evaluate_axis(coefficients, segment, tau):
    position = float(derivative_row(tau, 0).dot(coefficients[segment]))
    velocity = float(derivative_row(tau, 1).dot(coefficients[segment]))
    acceleration = float(derivative_row(tau, 2).dot(coefficients[segment]))
    return position, velocity, acceleration


def generate_trajectory(waypoints, workspace=None, obstacles=None, dt=DT, refine=True):
    fixed_z = float((workspace or {}).get("z", FIXED_Z))
    usable_waypoints = refine_waypoints(waypoints, workspace, obstacles, fixed_z=fixed_z) if refine else list(waypoints)
    verify_waypoints(usable_waypoints, workspace, obstacles)
    durations = segment_durations(usable_waypoints)
    coeff_x = solve_axis_min_snap([float(point["x"]) for point in usable_waypoints], durations)
    coeff_y = solve_axis_min_snap([float(point["y"]) for point in usable_waypoints], durations)

    samples = []
    elapsed = 0.0
    for segment, duration in enumerate(durations):
        count = max(1, int(math.ceil(duration / dt)))
        for index in range(count):
            if segment > 0 and index == 0:
                continue
            tau = min(index * dt, duration)
            px, vx, ax = evaluate_axis(coeff_x, segment, tau)
            py, vy, ay = evaluate_axis(coeff_y, segment, tau)
            samples.append(
                {
                    "t": round(elapsed + tau, 6),
                    "x": [round(px, 6), round(py, 6), round(vx, 6), round(vy, 6)],
                    "u": [round((ax + DAMPING * vx) / DAMPING, 6), round((ay + DAMPING * vy) / DAMPING, 6)],
                }
            )
        elapsed += duration

    px, vx, ax = evaluate_axis(coeff_x, len(durations) - 1, durations[-1])
    py, vy, ay = evaluate_axis(coeff_y, len(durations) - 1, durations[-1])
    if not samples or samples[-1]["t"] < elapsed - 1e-6:
        samples.append(
            {
                "t": round(elapsed, 6),
                "x": [round(px, 6), round(py, 6), round(vx, 6), round(vy, 6)],
                "u": [round((ax + DAMPING * vx) / DAMPING, 6), round((ay + DAMPING * vy) / DAMPING, 6)],
            }
        )
    return {"waypoints": usable_waypoints, "durations": durations, "samples": samples}


def load_json(value):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def main():
    parser = argparse.ArgumentParser(description="Generate a minimum-snap double-integrator trajectory from 2D waypoints.")
    parser.add_argument("--waypoints", required=True, help="JSON waypoint list or path to JSON containing waypoints.")
    parser.add_argument("--workspace", default='{"x":[0.0,4.0],"y":[0.0,4.0],"z":-0.25}')
    parser.add_argument("--obstacles", default="[]")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = load_json(args.waypoints)
    waypoints = payload.get("waypoints", payload) if isinstance(payload, dict) else payload
    result = generate_trajectory(waypoints, load_json(args.workspace), load_json(args.obstacles), dt=args.dt, refine=not args.no_refine)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
