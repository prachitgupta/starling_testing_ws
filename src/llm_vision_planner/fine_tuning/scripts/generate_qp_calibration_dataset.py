#!/usr/bin/env python3
"""Generate shared-clock minimum-control QP calibration data."""

import argparse
import csv
import json
from pathlib import Path

from conformal_rrt_dataset import FIELDNAMES, conformal_quantile, score_trajectories
from min_control_qp import generate_shared_pair


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "datasets" / "conformal_rrt_calibration_dataset_2001.csv"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "datasets" / "calibration_min_control_qp_shared_clock.csv"


def build(args):
    with args.input.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))[: args.samples]
    if len(source_rows) != args.samples:
        raise RuntimeError(f"Input contains {len(source_rows)} rows; requested {args.samples}.")
    output_rows = []
    for index, source in enumerate(source_rows):
        workspace = json.loads(source["workspace"])
        obstacles = json.loads(source["obstacles"])
        rrt_waypoints = json.loads(source["rrt_verified_waypoints"])
        llm_waypoints = json.loads(source["llm_verified_waypoints"])
        rrt, llm = generate_shared_pair(rrt_waypoints, llm_waypoints, workspace, obstacles, args.dt)
        scores = score_trajectories(rrt, llm, args.dt)
        row = dict(source)
        row["rrt_trajectory"] = json.dumps(rrt, separators=(",", ":"))
        row["llm_trajectory"] = json.dumps(llm, separators=(",", ":"))
        row["s_u"], row["s_x"] = scores["s_u"], scores["s_x"]
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
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in FIELDNAMES} for row in output_rows)
    temporary.replace(args.output)
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    print(f"q_u={q_u:.6f}, q_x={q_x:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--delta-u", type=float, default=0.05)
    parser.add_argument("--delta-x", type=float, default=0.05)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
