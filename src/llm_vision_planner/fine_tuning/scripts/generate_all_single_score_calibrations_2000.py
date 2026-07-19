#!/usr/bin/env python3
"""Create new single-score calibration files for all 2,000-row methods."""

import argparse
import csv
import json
import math
from pathlib import Path

from lqr import B_DOUBLE_INTEGRATOR, solve_care
from single_score import combined_disturbance_score


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "datasets"
METHODS = {
    "min_snap_different_clocks": (
        "calibration_min_snap_different_clocks_2000.csv",
        "calibration_min_snap_different_clocks_single_score_2000.csv",
    ),
    "min_snap_shared_clock": (
        "calibration_min_snap_shared_clock_2000.csv",
        "calibration_min_snap_shared_clock_single_score_2000.csv",
    ),
    "min_snap_frenet": (
        "calibration_min_snap_frenet_2000.csv",
        "calibration_min_snap_frenet_single_score_2000.csv",
    ),
    "guided_fit_shared_clock": (
        "calibration_min_snap_guided_fit_shared_2000.csv",
        "calibration_min_snap_guided_fit_shared_single_score_2000.csv",
    ),
}
EXTRA_FIELDS = ["s_w", "q_w", "delta_w", "score_type", "method"]


def conformal_quantile(values, delta):
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def convert(input_path, output_path, method, samples, delta_w, p, k):
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing calibration file: {output_path}")
    with input_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != samples:
        raise RuntimeError(f"{input_path} contains {len(rows)} rows; expected {samples}.")
    for index, row in enumerate(rows):
        row["s_w"] = combined_disturbance_score(
            json.loads(row["rrt_trajectory"]),
            json.loads(row["llm_trajectory"]),
            p,
            B_DOUBLE_INTEGRATOR,
            k,
        )
        print(f"[{method} {index + 1}/{samples}] sample_id={row['sample_id']} s_w={row['s_w']}", flush=True)
    q_w = conformal_quantile([row["s_w"] for row in rows], delta_w)
    for row in rows:
        row["q_w"] = round(q_w, 6)
        row["delta_w"] = delta_w
        row["score_type"] = "combined_p_weighted_disturbance"
        row["method"] = method
        row["accepted"] = float(row["s_w"]) <= q_w
    output_fields = fieldnames + [name for name in EXTRA_FIELDS if name not in fieldnames]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    print(f"Wrote {len(rows)} rows to {output_path}; q_w={q_w:.6f}", flush=True)
    return q_w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--delta-w", type=float, default=0.10)
    args = parser.parse_args()
    p, k, _ = solve_care()
    results = {}
    for method, (input_name, output_name) in METHODS.items():
        results[method] = convert(
            args.dataset_dir / input_name,
            args.dataset_dir / output_name,
            method,
            args.samples,
            args.delta_w,
            p,
            k,
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
