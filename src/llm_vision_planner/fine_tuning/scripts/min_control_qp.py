#!/usr/bin/env python3
"""Minimum-control trajectories for the damped planar double integrator."""

import math

import numpy as np

from min_snap import DAMPING, DT, segment_durations


def allocate_steps(durations, dt=DT, total_steps=None):
    """Allocate positive integer knot intervals proportionally to durations."""
    durations = np.asarray(durations, dtype=float)
    if len(durations) == 0 or np.min(durations) <= 0.0:
        raise ValueError("durations must contain positive segment durations")
    if total_steps is None:
        total_steps = max(len(durations), int(round(float(np.sum(durations)) / dt)))
    if total_steps < len(durations):
        raise ValueError("total_steps must provide at least one interval per segment")
    raw = durations / np.sum(durations) * total_steps
    steps = np.maximum(1, np.floor(raw).astype(int))
    while int(np.sum(steps)) < total_steps:
        index = max(range(len(steps)), key=lambda i: (raw[i] - steps[i], -i))
        steps[index] += 1
    while int(np.sum(steps)) > total_steps:
        candidates = [i for i, value in enumerate(steps) if value > 1]
        index = min(candidates, key=lambda i: (raw[i] - steps[i], i))
        steps[index] -= 1
    return steps.tolist()


def discrete_dynamics(dt=DT, damping=DAMPING):
    """Exact zero-order-hold dynamics for p_dot=v, v_dot=-d*v+d*u."""
    decay = math.exp(-damping * dt)
    velocity_gain = 1.0 - decay
    position_velocity_gain = velocity_gain / damping
    position_control_gain = dt - position_velocity_gain
    a = np.array([[1.0, position_velocity_gain], [0.0, decay]], dtype=float)
    b = np.array([position_control_gain, velocity_gain], dtype=float)
    return a, b


def _solve_controls(waypoints, waypoint_knots, total_steps, dt):
    """Eliminate states and solve the equality-constrained minimum-norm QP."""
    a, b = discrete_dynamics(dt)
    influence = np.zeros((2, total_steps), dtype=float)
    free = np.array([[float(waypoints[0]["x"]), float(waypoints[0]["y"])], [0.0, 0.0]])
    rows = []
    targets = []
    knot_to_waypoint = {knot: index for index, knot in enumerate(waypoint_knots[1:], start=1)}
    for step in range(total_steps):
        free = a @ free
        influence = a @ influence
        influence[:, step] += b
        knot = step + 1
        if knot in knot_to_waypoint:
            waypoint = waypoints[knot_to_waypoint[knot]]
            rows.append(influence[0].copy())
            targets.append([float(waypoint["x"]) - free[0, 0], float(waypoint["y"]) - free[0, 1]])
    rows.append(influence[1].copy())
    targets.append([-free[1, 0], -free[1, 1]])
    constraints = np.vstack(rows)
    targets = np.asarray(targets, dtype=float)
    controls = np.linalg.lstsq(constraints, targets, rcond=None)[0]
    residual = constraints @ controls - targets
    if float(np.max(np.abs(residual))) > 1e-7:
        raise RuntimeError(f"Minimum-control QP equality residual is {np.max(np.abs(residual)):.3e}")
    return controls


def generate_trajectory(waypoints, workspace=None, obstacles=None, dt=DT, durations=None, total_steps=None):
    """Solve the minimum-control QP and return the existing trajectory structure."""
    usable_waypoints = list(waypoints)
    if len(usable_waypoints) < 2:
        raise ValueError("Minimum-control QP requires at least two verified waypoints.")
    natural_durations = segment_durations(usable_waypoints) if durations is None else [float(value) for value in durations]
    if len(natural_durations) != len(usable_waypoints) - 1 or min(natural_durations) <= 0.0:
        raise ValueError("durations must contain one positive value per waypoint segment")
    steps = allocate_steps(natural_durations, dt=dt, total_steps=total_steps)
    waypoint_knots = np.concatenate(([0], np.cumsum(steps))).astype(int).tolist()
    total_steps = waypoint_knots[-1]
    controls = _solve_controls(usable_waypoints, waypoint_knots, total_steps, dt)
    a, b = discrete_dynamics(dt)
    states = np.array(
        [[float(usable_waypoints[0]["x"]), float(usable_waypoints[0]["y"])], [0.0, 0.0]],
        dtype=float,
    )
    samples = []
    for step in range(total_steps + 1):
        control = controls[step] if step < total_steps else np.zeros(2)
        samples.append(
            {
                "t": round(step * dt, 6),
                "x": [round(float(value), 6) for value in (*states[0], *states[1])],
                "u": [round(float(value), 6) for value in control],
            }
        )
        if step < total_steps:
            states = a @ states + b[:, None] * controls[step][None, :]
    return {
        "waypoints": usable_waypoints,
        "durations": [round(step * dt, 6) for step in steps],
        "waypoint_knots": waypoint_knots,
        "samples": samples,
    }


def generate_shared_pair(rrt_waypoints, llm_waypoints, workspace=None, obstacles=None, dt=DT):
    """Solve RRT and LLM QPs with one total horizon and independent waypoint knots."""
    rrt_natural = segment_durations(rrt_waypoints)
    total_steps = max(len(rrt_natural), int(round(sum(rrt_natural) / dt)))
    rrt = generate_trajectory(
        rrt_waypoints,
        workspace,
        obstacles,
        dt=dt,
        durations=rrt_natural,
        total_steps=total_steps,
    )
    llm = generate_trajectory(
        llm_waypoints,
        workspace,
        obstacles,
        dt=dt,
        durations=segment_durations(llm_waypoints),
        total_steps=total_steps,
    )
    return rrt, llm
