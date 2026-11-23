import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "fine_tuning" / "scripts" / "plot_flight_tube_coverage.py"
SPEC = importlib.util.spec_from_file_location("plot_flight_tube_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_run(root, run_id, y_offset):
    run = root / run_id
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"run_id": run_id}))
    samples = [
        {"t": 0.0, "x": [0.0, 0.0, 1.0, 0.0], "u": [0.0, 0.0]},
        {"t": 1.0, "x": [1.0, 0.0, 1.0, 0.0], "u": [0.0, 0.0]},
    ]
    events = [
        {"receipt_monotonic_ns": 1_000_000_000, "topic": "/llm_vision/plan_verified",
         "data": {"plan_id": "plan-1", "ground_truth_obstacles": [], "obstacles": []}},
        {"receipt_monotonic_ns": 2_000_000_000, "topic": "/llm_vision/executed_reference",
         "data": {"plan_id": "plan-1", "samples": samples}},
    ]
    for index in range(6):
        elapsed = index / 5
        events.append({
            "receipt_monotonic_ns": 2_000_000_000 + int(elapsed * 1_000_000_000),
            "topic": "/fmu/out/vehicle_odometry",
            "data": {"position": [elapsed, y_offset, -0.5]},
        })
    (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return run


def test_flight_loading_and_coverage(tmp_path):
    flight = MODULE.load_flight(write_run(tmp_path, "flight-1", 0.1))
    deviations = [row["deviation_m"] for row in flight["odometry"]]
    assert len(deviations) == 6
    assert max(deviations) < 0.101
    assert MODULE.coverage(deviations, 0.11)["whole_flight_covered"] is True
    assert MODULE.coverage(deviations, 0.09)["whole_flight_covered"] is False


def test_outputs_report_whole_flight_coverage(tmp_path):
    flights = [MODULE.load_flight(write_run(tmp_path, f"flight-{index}", 0.1)) for index in range(5)]
    methods = [
        {"label": "Without environment uncertainty", "radius_m": 0.09, "path": "planner.json"},
        {"label": "Environment-uncertainty aware", "radius_m": 0.11, "path": "environment.json"},
    ]
    csv_path, json_path, plot_path = MODULE.write_outputs(flights, methods, tmp_path / "results")
    report = json.loads(json_path.read_text())
    assert report["aggregate"]["planner_only"]["covered_flights"] == 0
    assert report["aggregate"]["environment_aware"]["covered_flights"] == 5
    assert csv_path.is_file()
    assert plot_path.is_file()


def test_beta_one_planner_certificate_is_required(tmp_path):
    certificate = tmp_path / "planner.json"
    certificate.write_text(json.dumps({
        "calibrated_radius_m": 0.2,
        "quantile_unbounded": False,
        "calibration_mode": "planner_only",
        "environment_uncertainty_included": False,
        "beta": 2.0,
    }))
    try:
        MODULE.load_certificate(certificate, "planner", True)
    except ValueError as exc:
        assert "beta=1" in str(exc)
    else:
        raise AssertionError("beta=2 certificate was accepted")
