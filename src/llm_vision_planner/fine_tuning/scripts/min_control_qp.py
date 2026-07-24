#!/usr/bin/env python3
"""Minimum-control trajectories for the damped planar double integrator."""

import math

import numpy as np


DAMPING = 1.1
DT = 0.1
CRUISE_SPEED_MPS = 0.3
_BOUNDED_DURATION_SCALES = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def segment_durations(waypoints, cruise_speed_mps=CRUISE_SPEED_MPS):
    durations = []
    for first, second in zip(waypoints, waypoints[1:]):
        distance = math.hypot(
            float(second["x"]) - float(first["x"]),
            float(second["y"]) - float(first["y"]),
        )
        durations.append(max(distance / cruise_speed_mps, 0.5))
    return durations


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


def propagate_state(state, control, dt, damping=DAMPING):
    """Propagate [px, py, vx, vy] exactly for one held-control interval."""
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    a, b = discrete_dynamics(dt, damping)
    axes = np.vstack((state[:2], state[2:]))
    propagated = a @ axes + b[:, None] * control[None, :]
    return np.concatenate((propagated[0], propagated[1]))


def evaluate_sample(samples, timestamp, damping=DAMPING):
    """Evaluate a sampled QP reference with exact zero-order-held control."""
    if not samples:
        raise ValueError("samples must not be empty")
    timestamp = float(timestamp)
    times = np.fromiter((float(sample["t"]) for sample in samples), dtype=float)
    if timestamp <= times[0]:
        return np.asarray(samples[0]["x"], dtype=float), np.asarray(samples[0]["u"], dtype=float)
    if timestamp >= times[-1]:
        return np.asarray(samples[-1]["x"], dtype=float), np.asarray(samples[-1]["u"], dtype=float)
    index = int(np.searchsorted(times, timestamp, side="right") - 1)
    state = propagate_state(samples[index]["x"], samples[index]["u"], timestamp - times[index], damping)
    return state, np.asarray(samples[index]["u"], dtype=float)


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


def _positive_limit(name, value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value when provided")
    return value


def _controls_respect_limits(controls, dt, max_velocity_mps, max_acceleration_mps2):
    """Check state velocity, velocity-command, and modeled acceleration bounds."""
    max_velocity_mps = _positive_limit("max_velocity_mps", max_velocity_mps)
    max_acceleration_mps2 = _positive_limit("max_acceleration_mps2", max_acceleration_mps2)
    state = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    a, b = discrete_dynamics(dt)
    for control in controls:
        if max_velocity_mps is not None and float(np.linalg.norm(control)) > max_velocity_mps + 1e-8:
            return False
        acceleration = DAMPING * (control - state[1])
        if max_acceleration_mps2 is not None and float(np.linalg.norm(acceleration)) > max_acceleration_mps2 + 1e-8:
            return False
        state = a @ state + b[:, None] * control[None, :]
        if max_velocity_mps is not None and float(np.linalg.norm(state[1])) > max_velocity_mps + 1e-8:
            return False
    return True


def generate_trajectory(
    waypoints,
    workspace=None,
    obstacles=None,
    dt=DT,
    durations=None,
    total_steps=None,
    max_velocity_mps=None,
    max_acceleration_mps2=None,
):
    """Solve a minimum-control QP whose accepted reference meets hard motion bounds."""
    usable_waypoints = list(waypoints)
    if len(usable_waypoints) < 2:
        raise ValueError("Minimum-control QP requires at least two verified waypoints.")
    natural_durations = segment_durations(usable_waypoints) if durations is None else [float(value) for value in durations]
    if len(natural_durations) != len(usable_waypoints) - 1 or min(natural_durations) <= 0.0:
        raise ValueError("durations must contain one positive value per waypoint segment")
    max_velocity_mps = _positive_limit("max_velocity_mps", max_velocity_mps)
    max_acceleration_mps2 = _positive_limit("max_acceleration_mps2", max_acceleration_mps2)
    initial_steps = allocate_steps(natural_durations, dt=dt, total_steps=total_steps)
    initial_total_steps = int(sum(initial_steps))
    bounded = max_velocity_mps is not None or max_acceleration_mps2 is not None
    last_error = None
    for scale in (_BOUNDED_DURATION_SCALES if bounded else (1.0,)):
        candidate_total_steps = max(len(natural_durations), int(math.ceil(initial_total_steps * scale)))
        steps = allocate_steps(natural_durations, dt=dt, total_steps=candidate_total_steps)
        waypoint_knots = np.concatenate(([0], np.cumsum(steps))).astype(int).tolist()
        total_steps = waypoint_knots[-1]
        controls = _solve_controls(usable_waypoints, waypoint_knots, total_steps, dt)
        if not bounded or _controls_respect_limits(
            controls,
            dt,
            max_velocity_mps,
            max_acceleration_mps2,
        ):
            break
        last_error = "requested horizon exceeds a horizontal safety bound"
    else:
        raise RuntimeError(
            "Bounded minimum-control QP remains infeasible after lengthening the horizon: "
            f"{last_error}"
        )
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
    trajectory = {
        "waypoints": usable_waypoints,
        "durations": [round(step * dt, 6) for step in steps],
        "waypoint_knots": waypoint_knots,
        "samples": samples,
    }
    if max_velocity_mps is not None or max_acceleration_mps2 is not None:
        trajectory["limits"] = {
            "max_velocity_mps": _positive_limit("max_velocity_mps", max_velocity_mps),
            "max_acceleration_mps2": _positive_limit("max_acceleration_mps2", max_acceleration_mps2),
        }
    return trajectory


def generate_shared_pair(
    rrt_waypoints,
    llm_waypoints,
    workspace=None,
    obstacles=None,
    dt=DT,
    max_velocity_mps=None,
    max_acceleration_mps2=None,
):
    """Solve RRT and LLM QPs with one bounded shared horizon."""
    rrt_natural = segment_durations(rrt_waypoints)
    total_steps = max(len(rrt_natural), int(round(sum(rrt_natural) / dt)))
    llm_natural = segment_durations(llm_waypoints)
    for _ in range(len(_BOUNDED_DURATION_SCALES) + 1):
        rrt = generate_trajectory(
            rrt_waypoints,
            workspace,
            obstacles,
            dt=dt,
            durations=rrt_natural,
            total_steps=total_steps,
            max_velocity_mps=max_velocity_mps,
            max_acceleration_mps2=max_acceleration_mps2,
        )
        llm = generate_trajectory(
            llm_waypoints,
            workspace,
            obstacles,
            dt=dt,
            durations=llm_natural,
            total_steps=total_steps,
            max_velocity_mps=max_velocity_mps,
            max_acceleration_mps2=max_acceleration_mps2,
        )
        required_steps = max(rrt["waypoint_knots"][-1], llm["waypoint_knots"][-1])
        if rrt["waypoint_knots"][-1] == llm["waypoint_knots"][-1] == total_steps:
            return rrt, llm
        total_steps = required_steps
    raise RuntimeError("Could not establish one bounded shared horizon for the RRT and LLM QPs.")
