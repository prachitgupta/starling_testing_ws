#!/usr/bin/env python3
"""Collapse repeated continuous vision captures into independent calibration trials."""

import argparse
import csv
import json
import math
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

from vision_error_dataset_generattor import CSV_FIELDS, RAW_CSV_FIELDS, wrapped_angle_difference


AUDIT_FIELDS = [
    "trial_id",
    "session_id",
    "repeat_count",
    "object_count",
    "capture_ids",
]


def csv_bool(value, field):
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError(f"invalid {field}: {value!r}")


def finite_float(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"capture {row.get('capture_id', '?')} has invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"capture {row.get('capture_id', '?')} has non-finite {field}")
    return value


def capture_features(rows):
    first = rows[0]
    session_id = str(first["session_id"]).strip()
    capture_id = str(first["capture_id"]).strip()
    if not session_id or not capture_id:
        raise ValueError("raw rows require session_id and capture_id")
    objects = {}
    for row in rows:
        if row["session_id"] != session_id or row["capture_id"] != capture_id:
            raise ValueError(f"capture {capture_id} mixes session or capture IDs")
        if not csv_bool(row["raw_continuous"], "raw_continuous"):
            raise ValueError(f"capture {capture_id} is not marked raw_continuous")
        if not csv_bool(row["stable_pose"], "stable_pose"):
            raise ValueError(f"capture {capture_id} contains an unstable pose")
        object_id = str(row["object_id"]).strip()
        if not object_id or object_id in objects:
            raise ValueError(f"capture {capture_id} has duplicate or empty object_id")
        objects[object_id] = (
            finite_float(row, "gt_center_x"),
            finite_float(row, "gt_center_y"),
            finite_float(row, "gt_yaw_rad"),
        )
    observer_fields = ("observer_x", "observer_y", "observer_z", "observer_yaw_rad")
    observer_values = [str(first.get(field, "")).strip() for field in observer_fields]
    if all(not value for value in observer_values):
        observer = None
    elif any(not value for value in observer_values):
        raise ValueError(f"capture {capture_id} has a partial observer pose")
    else:
        observer = tuple(finite_float(first, field) for field in observer_fields)
    return {
        "session_id": session_id,
        "capture_id": capture_id,
        "objects": objects,
        "observer": observer,
    }


def same_setup(
    first,
    second,
    position_tolerance_m,
    yaw_tolerance_rad,
    observer_position_tolerance_m,
    observer_yaw_tolerance_rad,
):
    if first["session_id"] != second["session_id"]:
        return False
    if set(first["objects"]) != set(second["objects"]):
        return False
    for object_id, first_pose in first["objects"].items():
        second_pose = second["objects"][object_id]
        if math.dist(first_pose[:2], second_pose[:2]) > position_tolerance_m:
            return False
        if wrapped_angle_difference(first_pose[2], second_pose[2]) > yaw_tolerance_rad:
            return False
    if (first["observer"] is None) != (second["observer"] is None):
        return False
    if first["observer"] is not None:
        if math.dist(first["observer"][:3], second["observer"][:3]) > observer_position_tolerance_m:
            return False
        if wrapped_angle_difference(first["observer"][3], second["observer"][3]) > observer_yaw_tolerance_rad:
            return False
    return True


def row_risk(row):
    if csv_bool(row["missed_detection"], "missed_detection"):
        return math.inf
    return finite_float(row, "score_m")


def collapse_raw_rows(
    rows,
    position_tolerance_m=0.05,
    yaw_tolerance_deg=5.0,
    observer_position_tolerance_m=0.05,
    observer_yaw_tolerance_deg=5.0,
):
    if not rows:
        raise ValueError("raw calibration CSV contains no rows")
    missing = set(RAW_CSV_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"raw calibration CSV is missing columns: {sorted(missing)}")
    captures = OrderedDict()
    for row in rows:
        captures.setdefault(str(row["capture_id"]), []).append(row)
    clusters = []
    for capture_rows in captures.values():
        features = capture_features(capture_rows)
        match = next(
            (
                cluster
                for cluster in clusters
                if same_setup(
                    cluster["features"],
                    features,
                    float(position_tolerance_m),
                    math.radians(float(yaw_tolerance_deg)),
                    float(observer_position_tolerance_m),
                    math.radians(float(observer_yaw_tolerance_deg)),
                )
            ),
            None,
        )
        if match is None:
            match = {"features": features, "captures": [], "rows": []}
            clusters.append(match)
        match["captures"].append(features["capture_id"])
        match["rows"].extend(capture_rows)

    session_counts = defaultdict(int)
    processed = []
    audit = []
    for cluster in clusters:
        session_id = cluster["features"]["session_id"]
        session_counts[session_id] += 1
        safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)
        trial_id = f"{safe_session}-pose-{session_counts[session_id]:06d}"
        rows_by_object = defaultdict(list)
        for row in cluster["rows"]:
            rows_by_object[str(row["object_id"])].append(row)
        for object_rows in rows_by_object.values():
            selected = max(object_rows, key=row_risk)
            output_row = {field: selected.get(field, "") for field in CSV_FIELDS}
            output_row["trial_id"] = trial_id
            processed.append(output_row)
        audit.append(
            {
                "trial_id": trial_id,
                "session_id": session_id,
                "repeat_count": len(cluster["captures"]),
                "object_count": len(rows_by_object),
                "capture_ids": ";".join(cluster["captures"]),
            }
        )
    return processed, audit


def write_csv(path, fields, rows, overwrite):
    path = Path(path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw continuous calibration CSV")
    parser.add_argument("--output", required=True, help="Certificate-ready calibration CSV")
    parser.add_argument("--audit-output", help="Repeat-group audit CSV")
    parser.add_argument("--position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--observer-position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--observer-yaw-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if input_path.resolve() == output_path.resolve():
        raise ValueError("raw input and processed output must be different files")
    with input_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    processed, audit = collapse_raw_rows(
        rows,
        position_tolerance_m=args.position_tolerance_m,
        yaw_tolerance_deg=args.yaw_tolerance_deg,
        observer_position_tolerance_m=args.observer_position_tolerance_m,
        observer_yaw_tolerance_deg=args.observer_yaw_tolerance_deg,
    )
    audit_path = (
        Path(args.audit_output).expanduser()
        if args.audit_output
        else output_path.with_name(f"{output_path.stem}.groups.csv")
    )
    write_csv(output_path, CSV_FIELDS, processed, args.overwrite)
    write_csv(audit_path, AUDIT_FIELDS, audit, args.overwrite)
    print(
        json.dumps(
            {
                "raw_captures": len({row["capture_id"] for row in rows}),
                "independent_trials": len(audit),
                "collapsed_repeats": sum(int(row["repeat_count"]) - 1 for row in audit),
                "output": str(output_path),
                "audit_output": str(audit_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
