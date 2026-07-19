#!/usr/bin/env python3
"""Progress-aligned scoring for sampled RRT and LLM trajectories."""

import math

import numpy as np


def _arc_data(samples):
    points = np.asarray([[sample["x"][0], sample["x"][1]] for sample in samples], dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.empty(0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return points, lengths, cumulative


def _project(point, points, lengths, cumulative, minimum_progress):
    best_distance = math.inf
    best_progress = minimum_progress
    for index, length in enumerate(lengths):
        if length <= 1e-12 or cumulative[index + 1] < minimum_progress:
            continue
        direction = points[index + 1] - points[index]
        ratio = float(np.dot(point - points[index], direction) / (length * length))
        ratio = min(1.0, max(0.0, ratio))
        progress = cumulative[index] + ratio * length
        if progress < minimum_progress:
            ratio = min(1.0, (minimum_progress - cumulative[index]) / length)
            progress = cumulative[index] + ratio * length
        projected = points[index] + ratio * direction
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance, best_progress = distance, progress
    return best_progress, 0.0 if not math.isfinite(best_distance) else best_distance


def _sample_at_progress(samples, cumulative, progress):
    index = int(np.searchsorted(cumulative, progress, side="right"))
    index = min(max(index, 1), len(samples) - 1)
    span = cumulative[index] - cumulative[index - 1]
    ratio = 0.0 if span <= 1e-12 else (progress - cumulative[index - 1]) / span
    x0, x1 = np.asarray(samples[index - 1]["x"], dtype=float), np.asarray(samples[index]["x"], dtype=float)
    u0, u1 = np.asarray(samples[index - 1]["u"], dtype=float), np.asarray(samples[index]["u"], dtype=float)
    return x0 + ratio * (x1 - x0), u0 + ratio * (u1 - u0)


def score_trajectories(rrt_trajectory, llm_trajectory, dt=0.1):
    del dt  # Sampling is already encoded in the trajectories.
    expert = rrt_trajectory["samples"]
    predicted = llm_trajectory["samples"]
    points, lengths, cumulative = _arc_data(expert)
    if len(expert) < 2 or cumulative[-1] <= 1e-12:
        x0, u0 = np.asarray(expert[0]["x"]), np.asarray(expert[0]["u"])
        sx = max(float(np.linalg.norm(np.asarray(item["x"]) - x0)) for item in predicted)
        su = max(float(np.linalg.norm(np.asarray(item["u"]) - u0)) for item in predicted)
        return {"s_u": round(su, 6), "s_x": round(sx, 6), "cross_track": round(sx, 6), "along_track": 0.0}
    llm_points, _, llm_cumulative = _arc_data(predicted)
    minimum_progress = 0.0
    sx = su = cross_track = along_track = 0.0
    for index, sample in enumerate(predicted):
        progress, cross = _project(llm_points[index], points, lengths, cumulative, minimum_progress)
        minimum_progress = progress
        expert_x, expert_u = _sample_at_progress(expert, cumulative, progress)
        sx = max(sx, float(np.linalg.norm(np.asarray(sample["x"], dtype=float) - expert_x)))
        su = max(su, float(np.linalg.norm(np.asarray(sample["u"], dtype=float) - expert_u)))
        cross_track = max(cross_track, cross)
        normalized_llm = 0.0 if llm_cumulative[-1] <= 1e-12 else llm_cumulative[index] / llm_cumulative[-1]
        along_track = max(along_track, abs(progress - normalized_llm * cumulative[-1]))
    return {
        "s_u": round(su, 6),
        "s_x": round(sx, 6),
        "cross_track": round(cross_track, 6),
        "along_track": round(along_track, 6),
    }
