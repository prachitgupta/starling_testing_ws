#!/usr/bin/env python3
"""Direct closed-loop position scores for shared-clock QP references."""

import numpy as np

from min_control_qp import evaluate_sample, propagate_state


CONTROL_DT = 0.05


def closed_loop_trace(rrt_trajectory, llm_trajectory, gain, control_dt=CONTROL_DT):
    """Simulate u=uhat-K(x-xhat) with exact held-control propagation."""
    rrt_samples = rrt_trajectory["samples"]
    llm_samples = llm_trajectory["samples"]
    horizon = min(float(rrt_samples[-1]["t"]), float(llm_samples[-1]["t"]))
    count = int(np.floor(horizon / control_dt + 1e-9))
    times = [index * control_dt for index in range(count + 1)]
    if horizon - times[-1] > 1e-9:
        times.append(horizon)

    state = np.asarray(rrt_samples[0]["x"], dtype=float)
    gain = np.asarray(gain, dtype=float)
    trace = []
    for index, timestamp in enumerate(times):
        xd, ud = evaluate_sample(rrt_samples, timestamp)
        xhat, uhat = evaluate_sample(llm_samples, timestamp)
        control = uhat - gain @ (state - xhat)
        trace.append(
            {
                "t": float(timestamp),
                "x": state.tolist(),
                "u": control.tolist(),
                "xd": xd.tolist(),
                "ud": ud.tolist(),
                "xhat": xhat.tolist(),
                "uhat": uhat.tolist(),
            }
        )
        if index + 1 < len(times):
            state = propagate_state(state, control, times[index + 1] - timestamp)
    return trace


def maximum_cross_track_error(points, reference_points):
    """Return maximum 2D distance from points to a reference polyline."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference_points, dtype=float)
    if len(points) == 0 or len(reference) == 0:
        raise ValueError("points and reference_points must not be empty")
    if len(reference) == 1:
        return float(np.max(np.linalg.norm(points - reference[0], axis=1)))

    starts = reference[:-1]
    segments = reference[1:] - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    maximum = 0.0
    for point in points:
        offsets = point - starts
        fractions = np.zeros(len(segments), dtype=float)
        nonzero = lengths_squared > 1e-18
        fractions[nonzero] = np.sum(offsets[nonzero] * segments[nonzero], axis=1) / lengths_squared[nonzero]
        projections = starts + np.clip(fractions, 0.0, 1.0)[:, None] * segments
        maximum = max(maximum, float(np.min(np.linalg.norm(point - projections, axis=1))))
    return maximum


def position_scores(rrt_trajectory, llm_trajectory, gain, control_dt=CONTROL_DT):
    """Return certified cross-track score and equal-time position diagnostic."""
    trace = closed_loop_trace(rrt_trajectory, llm_trajectory, gain, control_dt)
    positions = np.asarray([sample["x"][:2] for sample in trace], dtype=float)
    expert_positions = np.asarray([sample["xd"][:2] for sample in trace], dtype=float)
    cross_track = maximum_cross_track_error(positions, expert_positions)
    equal_time = float(np.max(np.linalg.norm(positions - expert_positions, axis=1)))
    return {
        "s_p": round(cross_track, 6),
        "s_position_time": round(equal_time, 6),
        "trace": trace,
    }


def weighted_disturbance_score(rrt_trajectory, llm_trajectory, metric, input_matrix, gain):
    """Return the legacy max sqrt(w.T P w) score using exact QP evaluation."""
    rrt_samples = rrt_trajectory["samples"]
    llm_samples = llm_trajectory["samples"]
    horizon = min(float(rrt_samples[-1]["t"]), float(llm_samples[-1]["t"]))
    times = sorted(
        {float(sample["t"]) for sample in rrt_samples if float(sample["t"]) <= horizon}
        | {float(sample["t"]) for sample in llm_samples if float(sample["t"]) <= horizon}
    )
    metric = np.asarray(metric, dtype=float)
    input_matrix = np.asarray(input_matrix, dtype=float)
    gain = np.asarray(gain, dtype=float)
    maximum = 0.0
    for timestamp in times:
        rrt_x, rrt_u = evaluate_sample(rrt_samples, timestamp)
        llm_x, llm_u = evaluate_sample(llm_samples, timestamp)
        disturbance = input_matrix @ (llm_u - rrt_u) + (input_matrix @ gain) @ (llm_x - rrt_x)
        maximum = max(maximum, float(np.sqrt(max(0.0, disturbance.T @ metric @ disturbance))))
    return round(maximum, 6)
