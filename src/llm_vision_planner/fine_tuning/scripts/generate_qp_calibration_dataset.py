#!/usr/bin/env python3
"""Generate shared-clock minimum-control QP calibration data."""

import argparse
import csv
import json
import math
from pathlib import Path

from min_control_qp import evaluate_sample, generate_shared_pair


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "datasets" / "conformal_rrt_calibration_dataset_2001.csv"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "datasets" / "calibration_min_control_qp_shared_clock_with_limits_2000.csv"
REQUIRED_SOURCE_FIELDS = {
    "sample_id",
    "workspace",
    "obstacles",
    "llm_verified_waypoints",
}
CALIBRATION_FIELDS = [
    "llm_trajectory",
    "s_u",
    "s_x",
    "q_u",
    "q_x",
    "delta_u",
    "delta_x",
    "max_velocity_mps",
    "max_acceleration_mps2",
    "accepted",
]


def conformal_quantile(values, delta):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def score_trajectories(rrt_trajectory, llm_trajectory):
    rrt_samples = rrt_trajectory["samples"]
    llm_samples = llm_trajectory["samples"]
    horizon = min(float(rrt_samples[-1]["t"]), float(llm_samples[-1]["t"]))
    times = sorted(
        {float(sample["t"]) for sample in rrt_samples if float(sample["t"]) <= horizon}
        | {float(sample["t"]) for sample in llm_samples if float(sample["t"]) <= horizon}
    )
    s_u = 0.0
    s_x = 0.0
    for timestamp in times:
        rrt_x, rrt_u = evaluate_sample(rrt_samples, timestamp)
        llm_x, llm_u = evaluate_sample(llm_samples, timestamp)
        s_x = max(s_x, float(math.sqrt(sum((llm_x[index] - rrt_x[index]) ** 2 for index in range(4)))))
        s_u = max(s_u, float(math.sqrt(sum((llm_u[index] - rrt_u[index]) ** 2 for index in range(2)))))
    return {"s_u": round(s_u, 6), "s_x": round(s_x, 6)}


def build(args):
    with args.input.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        source_fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)[: args.samples]
    expert_waypoints_field = f"{args.expert}_verified_waypoints"
    expert_trajectory_field = f"{args.expert}_trajectory"
    missing = (REQUIRED_SOURCE_FIELDS | {expert_waypoints_field}) - set(source_fieldnames)
    if missing:
        raise ValueError(f"Prediction-pair CSV is missing required columns: {', '.join(sorted(missing))}")
    if len(source_rows) != args.samples:
        raise RuntimeError(f"Input contains {len(source_rows)} rows; requested {args.samples}.")
    output_rows = []
    for index, source in enumerate(source_rows):
        workspace = json.loads(source["workspace"])
        obstacles = json.loads(source["obstacles"])
        rrt_waypoints = json.loads(source[expert_waypoints_field])
        llm_waypoints = json.loads(source["llm_verified_waypoints"])
        rrt, llm = generate_shared_pair(
            rrt_waypoints,
            llm_waypoints,
            workspace,
            obstacles,
            args.dt,
            max_velocity_mps=args.max_velocity_mps,
            max_acceleration_mps2=args.max_acceleration_mps2,
        )
        scores = score_trajectories(rrt, llm)
        row = dict(source)
        row[expert_trajectory_field] = json.dumps(rrt, separators=(",", ":"))
        row["llm_trajectory"] = json.dumps(llm, separators=(",", ":"))
        row["s_u"], row["s_x"] = scores["s_u"], scores["s_x"]
        row["max_velocity_mps"] = args.max_velocity_mps
        row["max_acceleration_mps2"] = args.max_acceleration_mps2
        output_rows.append(row)
        print(f"[{index + 1}/{args.samples}] sample_id={row['sample_id']} s_u={row['s_u']} s_x={row['s_x']}", flush=True)
    q_u = conformal_quantile([float(row["s_u"]) for row in output_rows], args.delta_u)
    q_x = conformal_quantile([float(row["s_x"]) for row in output_rows], args.delta_x)
    for row in output_rows:
        row["q_u"], row["q_x"] = round(q_u, 6), round(q_x, 6)
        row["delta_u"], row["delta_x"] = args.delta_u, args.delta_x
        row["accepted"] = float(row["s_u"]) <= q_u and float(row["s_x"]) <= q_x
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    calibration_fields = [expert_trajectory_field] + CALIBRATION_FIELDS
    output_fields = [name for name in source_fieldnames if name not in calibration_fields] + calibration_fields
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in output_fields} for row in output_rows)
    temporary.replace(args.output)
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    print(f"q_u={q_u:.6f}, q_x={q_x:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expert", choices=("rrt", "semantic_theta"), default="rrt", help="Expert waypoint column prefix; the QP and shared clock are identical for both.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-velocity-mps", type=float, default=0.5)
    parser.add_argument("--max-acceleration-mps2", type=float, default=0.5)
    parser.add_argument("--delta-u", type=float, default=0.05)
    parser.add_argument("--delta-x", type=float, default=0.05)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
