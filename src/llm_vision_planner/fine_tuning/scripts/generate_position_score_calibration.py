#!/usr/bin/env python3
"""Add the direct closed-loop cross-track position score to QP rows."""

import argparse
import csv
import json
import math
from pathlib import Path

from lqr import B_DOUBLE_INTEGRATOR, solve_care
from min_control_qp import CRUISE_SPEED_MPS, DT
from position_score import CONTROL_DT, position_scores, weighted_disturbance_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "datasets" / "calibration_min_control_qp_shared_clock_2000.csv"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "datasets" / "calibration_min_control_qp_position_score_2000.csv"
EXTRA_FIELDS = [
    "s_p",
    "s_position_time",
    "q_p",
    "delta_p",
    "score_type",
    "cruise_speed_mps",
    "control_dt",
    "s_w",
    "q_w",
    "delta_w",
]


def conformal_quantile(values, delta):
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1], rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--delta-p", type=float, default=0.10)
    parser.add_argument("--delta-w", type=float, default=0.10)
    parser.add_argument("--control-dt", type=float, default=CONTROL_DT)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)[: args.samples]
    if len(rows) != args.samples:
        raise RuntimeError(f"Input contains {len(rows)} rows; requested {args.samples}.")

    metric, gain, _ = solve_care()
    for index, row in enumerate(rows):
        scores = position_scores(
            json.loads(row["rrt_trajectory"]),
            json.loads(row["llm_trajectory"]),
            gain,
            args.control_dt,
        )
        row["s_p"] = scores["s_p"]
        row["s_position_time"] = scores["s_position_time"]
        row["s_w"] = weighted_disturbance_score(
            json.loads(row["rrt_trajectory"]),
            json.loads(row["llm_trajectory"]),
            metric,
            B_DOUBLE_INTEGRATOR,
            gain,
        )
        print(
            f"[{index + 1}/{len(rows)}] sample_id={row['sample_id']} "
            f"s_p={row['s_p']} s_position_time={row['s_position_time']} s_w={row['s_w']}",
            flush=True,
        )

    q_p, rank = conformal_quantile([row["s_p"] for row in rows], args.delta_p)
    q_w, rank_w = conformal_quantile([row["s_w"] for row in rows], args.delta_w)
    for row in rows:
        row["q_p"] = round(q_p, 6)
        row["delta_p"] = args.delta_p
        row["score_type"] = "closed_loop_cross_track_position"
        row["cruise_speed_mps"] = CRUISE_SPEED_MPS
        row["control_dt"] = args.control_dt
        row["q_w"] = round(q_w, 6)
        row["delta_w"] = args.delta_w
        row["accepted"] = float(row["s_p"]) <= q_p

    output_fields = fieldnames + [name for name in EXTRA_FIELDS if name not in fieldnames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(
        f"delta_p={args.delta_p:.6f}, rank={rank}, q_p={q_p:.6f} m, "
        f"radius={q_p:.6f} m, cruise_speed={CRUISE_SPEED_MPS:.3f} m/s, QP dt={DT:.3f} s"
    )
    print(f"legacy comparison: delta_w={args.delta_w:.6f}, rank={rank_w}, q_w={q_w:.6f}")


if __name__ == "__main__":
    main()
