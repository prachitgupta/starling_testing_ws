#!/usr/bin/env python3
"""Compare two frozen contraction radii on repeated hardware-flight logs.

The live stack already records the executed QP reference and PX4 odometry in
``mission_data_logger.py``.  This offline script never publishes ROS messages:
it time-aligns each flight to its executed reference, evaluates centre-position
containment for two fixed radii, and writes a five-row/two-column comparison.
"""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path


EXECUTED_REFERENCE_TOPIC = "/llm_vision/executed_reference"
ODOMETRY_TOPIC = "/fmu/out/vehicle_odometry"
VERIFIED_PLAN_TOPIC = "/llm_vision/plan_verified"


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_certificate(path, label, planner_only):
    path = Path(path).expanduser().resolve()
    certificate = read_json(path)
    radius = certificate.get("calibrated_radius_m")
    if certificate.get("quantile_unbounded") or radius is None:
        raise ValueError(f"{label} certificate has no finite calibrated radius: {path}")
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(f"{label} certificate radius must be finite and non-negative")
    mode = str(certificate.get("calibration_mode", "")).strip().lower()
    includes_environment = certificate.get("environment_uncertainty_included")
    if planner_only:
        if mode and mode != "planner_only":
            raise ValueError(f"planner-only certificate has calibration_mode={mode!r}")
        if includes_environment is not None and includes_environment is not False:
            raise ValueError("planner-only certificate must exclude environment uncertainty")
        if "beta" in certificate and not math.isclose(float(certificate["beta"]), 1.0):
            raise ValueError("this Case-A comparison requires a beta=1 planner-only certificate")
    elif mode == "planner_only" or includes_environment is False:
        raise ValueError("environment-aware certificate cannot be planner-only")
    return {
        "label": label,
        "path": str(path),
        "radius_m": radius,
        "delta": certificate.get("delta"),
        "calibration_sample_count": certificate.get("calibration_sample_count"),
        "calibrated_quantile": certificate.get("calibrated_quantile"),
        "certificate": certificate,
    }


def read_events(run_directory):
    run_directory = Path(run_directory).expanduser().resolve()
    event_path = run_directory / "events.jsonl"
    metadata_path = run_directory / "metadata.json"
    if not event_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"mission run must contain events.jsonl and metadata.json: {run_directory}")
    events = []
    with event_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {event_path}:{line_number}: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
    return run_directory, read_json(metadata_path), events


def valid_reference(payload):
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    if len(samples) < 2:
        return False
    previous = -math.inf
    for sample in samples:
        try:
            values = [float(sample["t"]), *[float(value) for value in sample["x"][:4]]]
        except (KeyError, TypeError, ValueError):
            return False
        if len(sample.get("x", [])) < 4 or not all(math.isfinite(value) for value in values):
            return False
        if values[0] <= previous:
            return False
        previous = values[0]
    return True


def reference_event(events):
    references = [
        event for event in events
        if event.get("topic") == EXECUTED_REFERENCE_TOPIC
        and valid_reference(event.get("data", {}))
    ]
    plan_ids = {str(event["data"].get("plan_id")) for event in references}
    if not references:
        raise ValueError("run has no valid executed-reference event")
    if len(plan_ids) != 1:
        raise ValueError(
            "run contains multiple executed plans; use one full_plot launch/run directory per flight"
        )
    return references[-1]


def evaluate_reference(samples, elapsed_s):
    times = [float(sample["t"]) for sample in samples]
    elapsed_s = min(max(float(elapsed_s), times[0]), times[-1])
    index = max(0, min(bisect.bisect_right(times, elapsed_s) - 1, len(samples) - 1))
    sample = samples[index]
    x = [float(value) for value in sample["x"]]
    if index == len(samples) - 1:
        return x[0], x[1]
    control = [float(value) for value in sample.get("u", [0.0, 0.0])]
    dt = elapsed_s - float(sample["t"])
    return (
        x[0] + dt * x[2] + 0.5 * dt * dt * control[0],
        x[1] + dt * x[3] + 0.5 * dt * dt * control[1],
    )


def matching_plan(events, plan_id, reference_receipt_ns):
    candidates = [
        event for event in events
        if event.get("topic") == VERIFIED_PLAN_TOPIC
        and str((event.get("data") or {}).get("plan_id")) == str(plan_id)
        and int(event.get("receipt_monotonic_ns", 0)) <= reference_receipt_ns
    ]
    return (candidates[-1].get("data") or {}) if candidates else {}


def obstacle_sets(plan):
    ground_truth = plan.get("ground_truth_obstacles", [])
    perceived = (
        plan.get("observed_obstacles", [])
        or plan.get("nominal_obstacles", [])
        or plan.get("obstacles", [])
    )
    return perceived if isinstance(perceived, list) else [], ground_truth if isinstance(ground_truth, list) else []


def load_flight(run_directory):
    run_directory, metadata, events = read_events(run_directory)
    reference = reference_event(events)
    payload = reference["data"]
    samples = payload["samples"]
    start_ns = int(reference["receipt_monotonic_ns"])
    duration_s = float(samples[-1]["t"])
    stop_ns = start_ns + int(duration_s * 1_000_000_000)
    odometry = []
    for event in events:
        if event.get("topic") != ODOMETRY_TOPIC:
            continue
        receipt_ns = int(event.get("receipt_monotonic_ns", 0))
        if receipt_ns < start_ns or receipt_ns > stop_ns:
            continue
        data = event.get("data") or {}
        position = data.get("position", [])
        if len(position) < 2:
            continue
        point = (float(position[0]), float(position[1]))
        if not all(math.isfinite(value) for value in point):
            continue
        elapsed_s = (receipt_ns - start_ns) / 1_000_000_000.0
        reference_xy = evaluate_reference(samples, elapsed_s)
        deviation = math.dist(point, reference_xy)
        odometry.append({
            "elapsed_s": elapsed_s,
            "actual_x": point[0],
            "actual_y": point[1],
            "reference_x": reference_xy[0],
            "reference_y": reference_xy[1],
            "deviation_m": deviation,
        })
    if len(odometry) < 2:
        raise ValueError(f"run has fewer than two time-aligned PX4 odometry samples: {run_directory}")
    plan_id = payload.get("plan_id")
    plan = matching_plan(events, plan_id, start_ns)
    perceived, ground_truth = obstacle_sets(plan)
    return {
        "run_directory": str(run_directory),
        "run_id": str(metadata.get("run_id") or run_directory.name),
        "plan_id": plan_id,
        "samples": samples,
        "odometry": odometry,
        "perceived_obstacles": perceived,
        "ground_truth_obstacles": ground_truth,
        "time_alignment": "receipt_monotonic_ns relative to executed_reference publication",
    }


def coverage(deviations, radius):
    inside = [value <= radius + 1e-12 for value in deviations]
    return {
        "point_count": len(inside),
        "covered_point_count": sum(inside),
        "pointwise_coverage": sum(inside) / len(inside),
        "whole_flight_covered": all(inside),
    }


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return [None, None]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def reference_polyline(samples, step_s=0.05):
    duration = float(samples[-1]["t"])
    count = max(2, int(math.ceil(duration / step_s)) + 1)
    return [evaluate_reference(samples, min(index * step_s, duration)) for index in range(count)]


def draw_tube(axis, points, radius, color):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Polygon

    patches = [Circle(point, radius) for point in points]
    if radius > 0.0:
        for first, second in zip(points, points[1:]):
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if length <= 1e-12:
                continue
            nx, ny = -dy * radius / length, dx * radius / length
            patches.append(Polygon([
                (first[0] + nx, first[1] + ny),
                (second[0] + nx, second[1] + ny),
                (second[0] - nx, second[1] - ny),
                (first[0] - nx, first[1] - ny),
            ]))
    axis.add_collection(PatchCollection(patches, facecolor=color, edgecolor="none", alpha=0.18, zorder=1))


def draw_obstacles(axis, obstacles, facecolor, edgecolor, label):
    from matplotlib.patches import Rectangle

    labelled = False
    for obstacle in obstacles:
        minimum, maximum = obstacle.get("min_corner", []), obstacle.get("max_corner", [])
        if len(minimum) < 2 or len(maximum) < 2:
            continue
        width, height = float(maximum[0]) - float(minimum[0]), float(maximum[1]) - float(minimum[1])
        axis.add_patch(Rectangle(
            (float(minimum[0]), float(minimum[1])), width, height,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, alpha=0.30,
            label=label if not labelled else None, zorder=2,
        ))
        labelled = True


def render(flights, methods, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(flights), 2, figsize=(12.5, max(3.2 * len(flights), 4.0)), squeeze=False,
        constrained_layout=True, sharex=False, sharey=False,
    )
    colors = ["#94a3b8", "#60a5fa"]
    for row, flight in enumerate(flights):
        points = reference_polyline(flight["samples"])
        actual = [(item["actual_x"], item["actual_y"]) for item in flight["odometry"]]
        deviations = [item["deviation_m"] for item in flight["odometry"]]
        for column, method in enumerate(methods):
            axis = axes[row][column]
            result = coverage(deviations, method["radius_m"])
            draw_tube(axis, points, method["radius_m"], colors[column])
            draw_obstacles(axis, flight["perceived_obstacles"], "#fbbf24", "#b45309", "Perceived obstacle")
            draw_obstacles(axis, flight["ground_truth_obstacles"], "#ef4444", "#991b1b", "Ground-truth obstacle")
            axis.plot([p[0] for p in points], [p[1] for p in points], color="#f97316", linewidth=1.8,
                      label="Executed Transformer/QP reference", zorder=4)
            axis.plot([p[0] for p in actual], [p[1] for p in actual], color="#15803d", linewidth=1.5,
                      label="PX4 odometry", zorder=5)
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(True, alpha=0.22)
            axis.set_xlabel("NED x [m]")
            axis.set_ylabel("NED y [m]")
            status = "COVERED" if result["whole_flight_covered"] else "EXCEEDED"
            axis.set_title(
                f"Flight {row + 1} — {method['label']}\n"
                f"r={method['radius_m']:.3f} m, max error={max(deviations):.3f} m — {status}",
                fontsize=10,
            )
            if row == 0 and column == 1:
                axis.legend(loc="best", fontsize=7)
    figure.suptitle(
        "Repeated-layout hardware flights: fixed offline tubes on the same recorded trajectories\n"
        "Containment is time-aligned UAV-centre tracking error; it is not a new conformal guarantee",
        fontsize=13,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_outputs(flights, methods, output_directory):
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, flight in enumerate(flights, start=1):
        deviations = [item["deviation_m"] for item in flight["odometry"]]
        row = {
            "flight_index": index,
            "run_id": flight["run_id"],
            "run_directory": flight["run_directory"],
            "plan_id": flight["plan_id"],
            "odometry_sample_count": len(deviations),
            "mean_time_aligned_error_m": sum(deviations) / len(deviations),
            "max_time_aligned_error_m": max(deviations),
        }
        for key, method in zip(("planner_only", "environment_aware"), methods):
            result = coverage(deviations, method["radius_m"])
            row.update({
                f"{key}_radius_m": method["radius_m"],
                f"{key}_covered_point_count": result["covered_point_count"],
                f"{key}_pointwise_coverage": result["pointwise_coverage"],
                f"{key}_whole_flight_covered": result["whole_flight_covered"],
            })
        rows.append(row)
    csv_path = output_directory / "flight_tube_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for key, method in zip(("planner_only", "environment_aware"), methods):
        successes = sum(bool(row[f"{key}_whole_flight_covered"]) for row in rows)
        aggregate[key] = {
            "radius_m": method["radius_m"],
            "covered_flights": successes,
            "flight_count": len(rows),
            "observed_whole_flight_coverage": successes / len(rows),
            "wilson_95_interval": wilson_interval(successes, len(rows)),
            "certificate_path": method["path"],
        }
    report = {
        "schema_version": 1,
        "scope": "time-aligned UAV-centre containment around the executed Transformer/QP reference",
        "not_claimed": [
            "a new conformal guarantee from the flight count",
            "independence of odometry samples within one flight",
            "expert-versus-Transformer residual coverage",
            "vehicle-footprint collision avoidance",
        ],
        "time_alignment": flights[0]["time_alignment"],
        "aggregate": aggregate,
        "flights": rows,
    }
    json_path = output_directory / "flight_tube_coverage.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_path = output_directory / "flight_tube_comparison.png"
    render(flights, methods, plot_path)
    return csv_path, json_path, plot_path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", action="append", required=True,
                        help="One mission_data_logger run directory; repeat once per preselected flight")
    result.add_argument("--planner-certificate", required=True,
                        help="Frozen Case-A planner-only beta=1 split_certificate.json")
    result.add_argument("--environment-certificate", required=True,
                        help="Frozen environment-aware split_certificate.json")
    result.add_argument("--expected-flights", type=int, default=5)
    result.add_argument("--output-dir", required=True)
    return result


def main():
    args = parser().parse_args()
    if args.expected_flights <= 0:
        raise ValueError("--expected-flights must be positive")
    if len(args.run_dir) != args.expected_flights:
        raise ValueError(
            f"expected exactly {args.expected_flights} preselected runs, received {len(args.run_dir)}"
        )
    resolved = [str(Path(path).expanduser().resolve()) for path in args.run_dir]
    if len(set(resolved)) != len(resolved):
        raise ValueError("--run-dir entries must be unique")
    planner = load_certificate(args.planner_certificate, "Without environment uncertainty", True)
    environment = load_certificate(args.environment_certificate, "Environment-uncertainty aware", False)
    flights = [load_flight(path) for path in resolved]
    csv_path, json_path, plot_path = write_outputs(flights, [planner, environment], args.output_dir)
    print(f"CSV: {csv_path}")
    print(f"Report: {json_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
