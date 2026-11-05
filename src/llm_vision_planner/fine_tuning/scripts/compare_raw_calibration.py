#!/usr/bin/env python3
"""Compare two raw residual-calibration CSVs of the same session.

Typical use: the CSV recorded live with GPT nano depths against the one
recorded again from the session's rosbag with perception's measured depths
(HARDWARE_TESTING_README.md, Section 13.3.1):

  python3 src/llm_vision_planner/fine_tuning/scripts/compare_raw_calibration.py \
    src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw_person.csv \
    src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw_person_outline.csv

For every detected row it measures how far the Vicon ground-truth footprint
reaches outside the predicted footprint (0 = fully contained; this is what the
conformal margin has to cover) and the predicted depth. Captures of the two
files are paired by time, because a replay numbers its captures differently.
"""

import argparse
import csv
import math

import numpy as np


def corners(row, prefix):
    try:
        return np.array([[float(row[f"{prefix}_corner_{i}_x"]), float(row[f"{prefix}_corner_{i}_y"])]
                         for i in range(4)])
    except (KeyError, ValueError):
        return None


def outside_distance(points, quad):
    """Largest distance of `points` outside the convex quadrilateral `quad`."""
    center = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0]))]
    worst = 0.0
    for point in points:
        out = 0.0
        for a, b in zip(quad, np.roll(quad, -1, axis=0)):
            normal = np.array([b[1] - a[1], a[0] - b[0]])
            normal /= max(1e-12, float(np.linalg.norm(normal)))
            if normal @ (center - a) > 0:
                normal = -normal
            out = max(out, float(normal @ (point - a)))
        worst = max(worst, out)
    return worst


def load(path, label):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if label and row["label"].lower() != label.lower():
                continue
            detected = row["missed_detection"].lower() == "false"
            pred, gt = corners(row, "pred"), corners(row, "gt")
            rows.append({
                "t": float(row["timestamp_s"]),
                "object_id": row["object_id"],
                "detected": detected and pred is not None,
                "outside": outside_distance(gt, pred) if detected and pred is not None else math.nan,
                "depth": float(row["pred_chatgpt_depth_m"]) if detected and row["pred_chatgpt_depth_m"] else math.nan,
            })
    return rows


def summary(name, rows):
    det = [r for r in rows if r["detected"]]
    out = np.array([r["outside"] for r in det])
    depth = np.array([r["depth"] for r in det])
    print(f"{name}: {len(rows)} rows, {len(det)} detected, {len(rows) - len(det)} missed")
    if len(det):
        print(f"  ground truth outside the predicted footprint [m]: median {np.median(out):.3f}, "
              f"90th pct {np.percentile(out, 90):.3f}, max {out.max():.3f}; "
              f"fully contained in {np.mean(out <= 1e-6):.0%} of rows")
        print(f"  predicted depth [m]: median {np.median(depth):.2f} "
              f"(10th-90th pct {np.percentile(depth, 10):.2f}-{np.percentile(depth, 90):.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first_csv")
    parser.add_argument("second_csv")
    parser.add_argument("--label", default="", help="only this object label (e.g. person)")
    parser.add_argument("--max-time-gap", type=float, default=1.5,
                        help="seconds between captures of the two files that count as the same view")
    args = parser.parse_args()
    first, second = load(args.first_csv, args.label), load(args.second_csv, args.label)
    summary(args.first_csv, first)
    summary(args.second_csv, second)
    pairs = []
    for row in first:
        same = [r for r in second if r["object_id"] == row["object_id"]
                and abs(r["t"] - row["t"]) <= args.max_time_gap]
        if same:
            pairs.append((row, min(same, key=lambda r: abs(r["t"] - row["t"]))))
    both = [(a, b) for a, b in pairs if a["detected"] and b["detected"]]
    print(f"paired by time: {len(pairs)} views; detected in both: {len(both)}")
    if both:
        diff = np.array([b["outside"] - a["outside"] for a, b in both])
        print(f"  outside distance, second minus first: median {np.median(diff):+.3f} m; "
              f"second is smaller in {np.mean(diff < 0):.0%} of views")


if __name__ == "__main__":
    main()
