#!/usr/bin/env python3
"""Recompute four paired x/u calibration datasets from verified predictions."""

import argparse
import csv
import json
import math
from pathlib import Path

from conformal_rrt_dataset import FIELDNAMES, conformal_quantile, score_trajectories, shared_llm_durations
from frenet_score import score_trajectories as score_frenet
from guided_fit import generate_trajectory as generate_guided
from min_snap import generate_trajectory as generate_min_snap


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "datasets" / "conformal_rrt_calibration_dataset_2001.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "datasets"
OUTPUTS = {
    "different_clocks": "calibration_min_snap_different_clocks.csv",
    "shared_clock": "calibration_min_snap_shared_clock.csv",
    "frenet": "calibration_min_snap_frenet.csv",
    "guided_shared": "calibration_min_snap_guided_fit_shared.csv",
}


def parsed(row, name):
    return json.loads(row[name])


def trajectory_pair(row, method, dt):
    workspace, obstacles = parsed(row, "workspace"), parsed(row, "obstacles")
    rrt_wp, llm_wp = parsed(row, "rrt_verified_waypoints"), parsed(row, "llm_verified_waypoints")
    if method == "guided_shared":
        rrt = generate_guided(rrt_wp, workspace, obstacles, dt=dt)
        llm_durations = shared_llm_durations(rrt["durations"], llm_wp)
        common_blend = min(rrt["blend_time"], 0.2 * min(llm_durations))
        rrt = generate_guided(rrt_wp, workspace, obstacles, dt=dt, durations=rrt["durations"], blend_time=common_blend)
        llm = generate_guided(
            llm_wp,
            workspace,
            obstacles,
            dt=dt,
            durations=llm_durations,
            blend_time=common_blend,
        )
        return rrt, llm, score_trajectories(rrt, llm, dt)
    rrt = generate_min_snap(rrt_wp, workspace, obstacles, dt=dt)
    durations = None if method in ("different_clocks", "frenet") else shared_llm_durations(rrt["durations"], llm_wp)
    llm = generate_min_snap(llm_wp, workspace, obstacles, dt=dt, durations=durations)
    scores = score_frenet(rrt, llm, dt) if method == "frenet" else score_trajectories(rrt, llm, dt)
    return rrt, llm, scores


def build_rows(source_rows, method, dt, delta_u, delta_x):
    built = []
    for index, source in enumerate(source_rows):
        rrt, llm, scores = trajectory_pair(source, method, dt)
        row = dict(source)
        row["rrt_trajectory"] = json.dumps(rrt, separators=(",", ":"))
        row["llm_trajectory"] = json.dumps(llm, separators=(",", ":"))
        row["s_u"], row["s_x"] = scores["s_u"], scores["s_x"]
        built.append(row)
        print(f"[{method} {index + 1}/{len(source_rows)}] sample_id={row['sample_id']} s_u={row['s_u']} s_x={row['s_x']}", flush=True)
    q_u = conformal_quantile([float(row["s_u"]) for row in built], delta_u)
    q_x = conformal_quantile([float(row["s_x"]) for row in built], delta_x)
    for row in built:
        row["q_u"], row["q_x"] = round(q_u, 6), round(q_x, 6)
        row["delta_u"], row["delta_x"] = delta_u, delta_x
        row["accepted"] = float(row["s_u"]) <= q_u and float(row["s_x"]) <= q_x
    return built


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in FIELDNAMES} for row in rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--delta-u", type=float, default=0.05)
    parser.add_argument("--delta-x", type=float, default=0.05)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    with args.input.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))[: args.samples]
    if len(source_rows) != args.samples:
        raise RuntimeError(f"Input contains {len(source_rows)} rows; requested {args.samples}.")
    if any(not row.get("rrt_verified_waypoints") or not row.get("llm_verified_waypoints") for row in source_rows):
        raise RuntimeError("Every source row must contain verified RRT and LLM waypoints.")
    for method, filename in OUTPUTS.items():
        rows = build_rows(source_rows, method, args.dt, args.delta_u, args.delta_x)
        output = args.output_dir / filename
        write_rows(output, rows)
        print(f"Wrote {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
