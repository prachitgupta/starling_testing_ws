#!/usr/bin/env python3
"""Combined P-weighted disturbance score for double-integrator references."""

import math

import numpy as np


def interpolate_sample(samples, time):
    if time <= float(samples[0]["t"]):
        return np.asarray(samples[0]["x"], dtype=float), np.asarray(samples[0]["u"], dtype=float)
    if time >= float(samples[-1]["t"]):
        return np.asarray(samples[-1]["x"], dtype=float), np.asarray(samples[-1]["u"], dtype=float)
    for index in range(1, len(samples)):
        if float(samples[index]["t"]) >= time:
            before, after = samples[index - 1], samples[index]
            span = max(float(after["t"]) - float(before["t"]), 1e-12)
            ratio = (time - float(before["t"])) / span
            x = np.asarray(before["x"], dtype=float) + ratio * (np.asarray(after["x"], dtype=float) - np.asarray(before["x"], dtype=float))
            u = np.asarray(before["u"], dtype=float) + ratio * (np.asarray(after["u"], dtype=float) - np.asarray(before["u"], dtype=float))
            return x, u
    return np.asarray(samples[-1]["x"], dtype=float), np.asarray(samples[-1]["u"], dtype=float)


def combined_disturbance_score(rrt_trajectory, llm_trajectory, p, b, k):
    """Return max_t sqrt(w.T P w), w=B*du+B*K*dx."""
    rrt_samples = rrt_trajectory["samples"]
    llm_samples = llm_trajectory["samples"]
    horizon = min(float(rrt_samples[-1]["t"]), float(llm_samples[-1]["t"]))
    times = sorted(
        {float(sample["t"]) for sample in rrt_samples if float(sample["t"]) <= horizon}
        | {float(sample["t"]) for sample in llm_samples if float(sample["t"]) <= horizon}
    )
    maximum = 0.0
    for time in times:
        rrt_x, rrt_u = interpolate_sample(rrt_samples, time)
        llm_x, llm_u = interpolate_sample(llm_samples, time)
        disturbance = b @ (llm_u - rrt_u) + (b @ k) @ (llm_x - rrt_x)
        maximum = max(maximum, math.sqrt(max(0.0, float(disturbance.T @ p @ disturbance))))
    return round(maximum, 6)
