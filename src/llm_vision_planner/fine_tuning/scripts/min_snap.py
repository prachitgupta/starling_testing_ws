#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


DAMPING = 1.1
FIXED_Z = -0.25
CRUISE_SPEED_MPS = 0.5
DT = 0.1
RIDGE = 1e-9


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


def generate_trajectory(waypoints, workspace=None, obstacles=None, dt=DT):
    usable_waypoints = list(waypoints)
    if len(usable_waypoints) < 2:
        raise ValueError("Minimum-snap generation requires at least two verified waypoints.")
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
    parser = argparse.ArgumentParser(description="Generate a minimum-snap double-integrator trajectory from verified 2D waypoints.")
    parser.add_argument("--waypoints", required=True, help="JSON verified waypoint list or path to JSON containing waypoints.")
    parser.add_argument("--workspace", default='{"x":[0.0,4.0],"y":[0.0,4.0],"z":-0.25}')
    parser.add_argument("--obstacles", default="[]")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = load_json(args.waypoints)
    waypoints = payload.get("waypoints", payload) if isinstance(payload, dict) else payload
    result = generate_trajectory(waypoints, load_json(args.workspace), load_json(args.obstacles), dt=args.dt)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
