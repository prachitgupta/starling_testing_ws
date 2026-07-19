#!/usr/bin/env python3
"""Piecewise-linear waypoint interpolation with parabolic corner blends."""

import math

import numpy as np

from min_snap import DAMPING, DT, segment_durations


BLEND_FRACTION = 0.2


def _line_value(waypoints, durations, boundaries, time):
    if time <= 0.0:
        return np.asarray(waypoints[0], dtype=float)
    if time >= boundaries[-1]:
        return np.asarray(waypoints[-1], dtype=float)
    segment = int(np.searchsorted(boundaries, time, side="right") - 1)
    ratio = (time - boundaries[segment]) / durations[segment]
    return np.asarray(waypoints[segment], dtype=float) + ratio * (
        np.asarray(waypoints[segment + 1], dtype=float) - np.asarray(waypoints[segment], dtype=float)
    )


def _line_velocity(waypoints, durations, boundaries, time):
    if time < 0.0 or time >= boundaries[-1]:
        return np.zeros(2)
    segment = int(np.searchsorted(boundaries, time, side="right") - 1)
    segment = max(0, min(segment, len(durations) - 1))
    return (np.asarray(waypoints[segment + 1]) - np.asarray(waypoints[segment])) / durations[segment]


def _integral(waypoints, durations, boundaries, start, end):
    """Exact integral of the extended piecewise-linear path."""
    cuts = [start] + [value for value in boundaries if start < value < end] + [end]
    total = np.zeros(2)
    for left, right in zip(cuts, cuts[1:]):
        total += 0.5 * (_line_value(waypoints, durations, boundaries, left) + _line_value(waypoints, durations, boundaries, right)) * (right - left)
    return total


def generate_trajectory(
    waypoints,
    workspace=None,
    obstacles=None,
    dt=DT,
    durations=None,
    blend_fraction=BLEND_FRACTION,
    blend_time=None,
):
    """Return parabolically blended samples in the minimum-snap trajectory shape.

    A centered box filter is applied analytically to the piecewise-linear path.
    Its position is piecewise quadratic, velocity is continuous, and acceleration
    is piecewise constant. Constant endpoint extensions preserve exact endpoints.
    """
    usable_waypoints = list(waypoints)
    if len(usable_waypoints) < 2:
        raise ValueError("Guided-fit generation requires at least two verified waypoints.")
    durations = segment_durations(usable_waypoints) if durations is None else [float(value) for value in durations]
    if len(durations) != len(usable_waypoints) - 1 or min(durations) <= 0.0:
        raise ValueError("durations must contain one positive value per trajectory segment.")
    fraction = min(max(float(blend_fraction), 1e-6), 0.5)
    half_blend = fraction * min(durations) if blend_time is None else float(blend_time)
    if half_blend <= 0.0 or half_blend > 0.5 * min(durations) + 1e-12:
        raise ValueError("blend_time must be positive and no greater than half the shortest segment duration.")
    points = [[float(point["x"]), float(point["y"])] for point in usable_waypoints]
    boundaries = np.concatenate(([0.0], np.cumsum(durations)))
    horizon = boundaries[-1]
    count = max(2, int(math.ceil((horizon + 2.0 * half_blend) / dt)) + 1)
    samples = []
    for output_time in np.linspace(0.0, horizon + 2.0 * half_blend, count):
        nominal_time = output_time - half_blend
        left, right = nominal_time - half_blend, nominal_time + half_blend
        position = _integral(points, durations, boundaries, left, right) / (2.0 * half_blend)
        velocity = (_line_value(points, durations, boundaries, right) - _line_value(points, durations, boundaries, left)) / (2.0 * half_blend)
        acceleration = (_line_velocity(points, durations, boundaries, right) - _line_velocity(points, durations, boundaries, left)) / (2.0 * half_blend)
        control = (acceleration + DAMPING * velocity) / DAMPING
        samples.append(
            {
                "t": round(float(output_time), 6),
                "x": [round(float(v), 6) for v in (*position, *velocity)],
                "u": [round(float(v), 6) for v in control],
            }
        )
    samples[0]["x"] = [points[0][0], points[0][1], 0.0, 0.0]
    samples[-1]["x"] = [points[-1][0], points[-1][1], 0.0, 0.0]
    samples[0]["u"] = [0.0, 0.0]
    samples[-1]["u"] = [0.0, 0.0]
    return {"waypoints": usable_waypoints, "durations": durations, "blend_time": half_blend, "samples": samples}
