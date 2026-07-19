#!/usr/bin/env python3
"""Add the combined P-weighted score to generated QP calibration rows."""

import argparse
import csv
import json
import math
from pathlib import Path

from lqr import B_DOUBLE_INTEGRATOR, solve_care
from single_score import combined_disturbance_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "datasets" / "calibration_min_control_qp_shared_clock_2000.csv"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "datasets" / "calibration_min_control_qp_single_score_2000.csv"
EXTRA_FIELDS = ["s_w", "q_w", "delta_w", "score_type"]


def conformal_quantile(values, delta):
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--delta-w", type=float, default=0.10)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)[: args.samples]
    if len(rows) != args.samples:
        raise RuntimeError(f"Input contains {len(rows)} rows; requested {args.samples}.")
    p, k, _ = solve_care()
    for index, row in enumerate(rows):
        row["s_w"] = combined_disturbance_score(
            json.loads(row["rrt_trajectory"]),
            json.loads(row["llm_trajectory"]),
            p,
            B_DOUBLE_INTEGRATOR,
            k,
        )
        print(f"[{index + 1}/{len(rows)}] sample_id={row['sample_id']} s_w={row['s_w']}", flush=True)
    q_w = conformal_quantile([row["s_w"] for row in rows], args.delta_w)
    for row in rows:
        row["q_w"] = round(q_w, 6)
        row["delta_w"] = args.delta_w
        row["score_type"] = "combined_p_weighted_disturbance"
        row["accepted"] = float(row["s_w"]) <= q_w
    output_fields = fieldnames + [name for name in EXTRA_FIELDS if name not in fieldnames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"delta_w={args.delta_w:.6f}, q_w={q_w:.6f}")


if __name__ == "__main__":
    main()
