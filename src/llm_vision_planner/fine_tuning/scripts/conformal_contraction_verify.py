#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("ROS_LOG_DIR", "/tmp/roslog")
Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from rrt import plan_rrt


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "datasets"
DEFAULT_CALIBRATION_CSV = DATASET_DIR / "conformal_rrt_calibration_dataset.csv"
DEFAULT_OUTPUT_PNG = SCRIPT_DIR.parent / "plots" / "conformal_contraction_verification.png"
DEFAULT_REPORT_JSON = SCRIPT_DIR.parent / "plots" / "conformal_contraction_verification.json"
DEFAULT_LLM_WAYPOINTS_TOPIC = "/llm_vision/plan_raw"


def parse_json_field(value):
    return json.loads(value) if isinstance(value, str) else value


def load_calibration_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def conformal_quantile(values, delta):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def path_length(path):
    return sum(
        math.hypot(float(path[i]["x"]) - float(path[i - 1]["x"]), float(path[i]["y"]) - float(path[i - 1]["y"]))
        for i in range(1, len(path))
    )


def interpolate_path(path, count):
    if count <= 1:
        return [dict(path[0])]
    total = path_length(path)
    if total <= 1e-9:
        return [dict(path[0]) for _ in range(count)]

    samples = []
    segment_index = 1
    segment_start = 0.0
    for target in [total * i / (count - 1) for i in range(count)]:
        while segment_index < len(path) - 1:
            segment = math.hypot(
                float(path[segment_index]["x"]) - float(path[segment_index - 1]["x"]),
                float(path[segment_index]["y"]) - float(path[segment_index - 1]["y"]),
            )
            if segment_start + segment >= target:
                break
            segment_start += segment
            segment_index += 1

        start = path[segment_index - 1]
        end = path[segment_index]
        segment = max(math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"])), 1e-9)
        ratio = min(1.0, max(0.0, (target - segment_start) / segment))
        samples.append(
            {
                "x": float(start["x"]) + (float(end["x"]) - float(start["x"])) * ratio,
                "y": float(start["y"]) + (float(end["y"]) - float(start["y"])) * ratio,
                "z": float(start.get("z", end.get("z", -0.25))),
            }
        )
    return samples


def velocities(path, dt):
    return [
        (
            (float(path[i]["x"]) - float(path[i - 1]["x"])) / dt,
            (float(path[i]["y"]) - float(path[i - 1]["y"])) / dt,
        )
        for i in range(1, len(path))
    ]


def contraction_energy(delta_xy, m_diag):
    return m_diag[0] * delta_xy[0] * delta_xy[0] + m_diag[1] * delta_xy[1] * delta_xy[1]


def load_waypoints(value):
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def extract_waypoints(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Waypoint payload must be a JSON list or object containing a 'waypoints' list.")
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, list):
        raise ValueError("Live waypoint payload does not contain a 'waypoints' list.")
    return waypoints


def wait_for_live_waypoints(topic, timeout_s):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class WaypointSubscriber(Node):
        def __init__(self):
            super().__init__("conformal_contraction_live_waypoints")
            self.payload = None
            self.error = None
            self.subscription = self.create_subscription(String, topic, self.callback, 10)

        def callback(self, msg):
            try:
                self.payload = extract_waypoints(json.loads(msg.data))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                self.error = str(exc)

    rclpy.init()
    node = WaypointSubscriber()
    deadline = node.get_clock().now().nanoseconds / 1e9 + timeout_s
    try:
        while rclpy.ok() and node.payload is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.error:
                raise ValueError(f"Failed to parse live waypoint message on {topic}: {node.error}")
            if node.get_clock().now().nanoseconds / 1e9 >= deadline:
                raise TimeoutError(f"Timed out waiting {timeout_s:.1f}s for LLM waypoints on {topic}")
        return node.payload
    finally:
        node.destroy_node()
        rclpy.shutdown()


def score_conformal_contraction(candidate_path, nominal_path, m_diag, alpha, dt):
    count = max(len(candidate_path), len(nominal_path), 2)
    candidate = interpolate_path(candidate_path, count)
    nominal = interpolate_path(nominal_path, count)
    candidate_vel = velocities(candidate, dt)
    nominal_vel = velocities(nominal, dt)

    s_dyn = 0.0
    s_con = 0.0
    energies = []
    for index, (v_candidate, v_nominal) in enumerate(zip(candidate_vel, nominal_vel), start=1):
        delta = (
            float(candidate[index]["x"]) - float(nominal[index]["x"]),
            float(candidate[index]["y"]) - float(nominal[index]["y"]),
        )
        energy = contraction_energy(delta, m_diag)
        residual = (
            2.0 * (delta[0] * m_diag[0] * v_candidate[0] + delta[1] * m_diag[1] * v_candidate[1])
            - 2.0 * (delta[0] * m_diag[0] * v_nominal[0] + delta[1] * m_diag[1] * v_nominal[1])
            + 2.0 * alpha * energy
        )
        s_dyn = max(s_dyn, math.hypot(v_candidate[0] - v_nominal[0], v_candidate[1] - v_nominal[1]))
        s_con = max(s_con, residual)
        energies.append(energy)

    initial_delta = (
        float(candidate[0]["x"]) - float(nominal[0]["x"]),
        float(candidate[0]["y"]) - float(nominal[0]["y"]),
    )
    initial_energy = contraction_energy(initial_delta, m_diag)
    return {
        "candidate": candidate,
        "nominal": nominal,
        "s_dyn": s_dyn,
        "s_con": max(0.0, s_con),
        "initial_energy": initial_energy,
        "max_energy": max(energies) if energies else initial_energy,
    }


def tube_profile(nominal_path, initial_energy, q_dyn, q_con, m_diag, alpha, epsilon, dt):
    alpha_bar = alpha - 0.5 * epsilon
    m_min = min(m_diag)
    m_max = max(m_diag)
    steady_energy = ((q_dyn * q_dyn) * m_max / epsilon + q_con) / (2.0 * alpha_bar)
    profile = []
    for index, point in enumerate(nominal_path):
        t = index * dt
        decay = math.exp(-2.0 * alpha_bar * t)
        energy_bound = decay * initial_energy + steady_energy * (1.0 - decay)
        radius = math.sqrt(max(energy_bound, 0.0) / m_min)
        profile.append({"t": t, "energy_bound": energy_bound, "radius": radius, "point": point})
    return steady_energy, profile


def draw_scene(output_png, candidate, nominal, tube, obstacles, workspace, result):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(workspace["x"][0]) - 0.25, float(workspace["x"][1]) + 0.25)
    axis.set_ylim(float(workspace["y"][0]) - 0.25, float(workspace["y"][1]) + 0.25)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Conformal contraction tube around RRT nominal trajectory")
    axis.grid(True, alpha=0.25)

    for obstacle in obstacles:
        min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
        max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
        axis.add_patch(
            Rectangle(
                (float(min_corner[0]), float(min_corner[1])),
                float(max_corner[0]) - float(min_corner[0]),
                float(max_corner[1]) - float(min_corner[1]),
                facecolor="#ef4444",
                edgecolor="#991b1b",
                alpha=0.35,
                label="obstacle" if obstacle is obstacles[0] else None,
            )
        )

    for entry in tube:
        point = entry["point"]
        axis.add_patch(
            Circle(
                (float(point["x"]), float(point["y"])),
                entry["radius"],
                facecolor="#60a5fa",
                edgecolor="none",
                alpha=0.11,
            )
        )

    axis.plot([p["x"] for p in nominal], [p["y"] for p in nominal], "-o", color="#2563eb", label="nominal RRT")
    axis.plot([p["x"] for p in candidate], [p["y"] for p in candidate], "--o", color="#f97316", label="LLM/candidate")
    axis.scatter([nominal[0]["x"]], [nominal[0]["y"]], color="#16a34a", s=80, marker="s", label="start")
    axis.scatter([nominal[-1]["x"]], [nominal[-1]["y"]], color="#7c3aed", s=100, marker="*", label="goal")
    status = "PASS" if result["accepted"] else "FAIL"
    joint_probability = 1.0 - result["delta_dyn"] - result["delta_con"]
    summary = (
        f"{status}: s_dyn={result['s_dyn']:.4f} <= q_dyn={result['q_dyn']:.4f}\n"
        f"s_con={result['s_con']:.4f} <= q_con={result['q_con']:.4f}\n"
        f"tube radius={result['steady_radius_m']:.3f} m, "
        f"P >= {joint_probability:.2f}"
    )
    axis.text(
        0.02,
        0.02,
        summary,
        transform=axis.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Verify and visualize the conformal contraction bound from the VLM-MPC notes.")
    parser.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION_CSV)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--llm-waypoints-mode", choices=["static", "live"], default="static")
    parser.add_argument("--llm-waypoints", default=None, help="JSON list or path to JSON list of candidate/LLM waypoints.")
    parser.add_argument("--llm-waypoints-topic", default=DEFAULT_LLM_WAYPOINTS_TOPIC)
    parser.add_argument("--live-timeout-s", type=float, default=20.0)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--m-diag", default=None, help="Override metric as comma-separated diagonal values, e.g. 1.0,1.0.")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--delta-dyn", type=float, default=0.1)
    parser.add_argument("--delta-con", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--rrt-clearance-m", type=float, default=0.40)
    parser.add_argument("--rrt-seed", type=int, default=7)
    args = parser.parse_args()

    rows = load_calibration_rows(args.calibration_csv)
    if not rows:
        raise RuntimeError(f"No calibration rows found in {args.calibration_csv}")
    if args.sample_id < 0 or args.sample_id >= len(rows):
        raise ValueError(f"--sample-id must be between 0 and {len(rows) - 1}.")
    calibration_rows = rows
    if args.calibration_samples is not None:
        if args.calibration_samples <= 0:
            raise ValueError("--calibration-samples must be positive.")
        if args.calibration_samples > len(rows):
            raise ValueError("--calibration-samples cannot exceed the number of CSV rows.")
        if args.sample_id < args.calibration_samples:
            raise ValueError("--sample-id must be outside the first --calibration-samples rows for held-out evaluation.")
        calibration_rows = rows[: args.calibration_samples]
    row = rows[args.sample_id]
    start = parse_json_field(row["start"])
    goal = parse_json_field(row["goal"])
    workspace = parse_json_field(row["workspace"])
    obstacles = parse_json_field(row["obstacles"])
    m_diag = [float(value) for value in (args.m_diag.split(",") if args.m_diag else parse_json_field(row["m_diag"]))]
    alpha = float(args.alpha if args.alpha is not None else row["alpha"])
    epsilon = float(args.epsilon if args.epsilon is not None else row["epsilon"])

    if len(m_diag) != 2 or min(m_diag) <= 0.0 or alpha <= 0.0 or not (0.0 < epsilon < 2.0 * alpha):
        raise ValueError("Require positive 2D diagonal M, alpha > 0, and 0 < epsilon < 2 * alpha.")

    q_dyn = conformal_quantile([float(item["s_dyn"]) for item in calibration_rows], args.delta_dyn)
    q_con = conformal_quantile([float(item["s_con"]) for item in calibration_rows], args.delta_con)
    if row.get("rrt_waypoints"):
        nominal = parse_json_field(row["rrt_waypoints"])
    else:
        nominal = plan_rrt(start, goal, obstacles, workspace=workspace, clearance_m=args.rrt_clearance_m, seed=args.rrt_seed)
    if args.llm_waypoints_mode == "live":
        candidate = wait_for_live_waypoints(args.llm_waypoints_topic, args.live_timeout_s)
    else:
        candidate = load_waypoints(args.llm_waypoints)
    if candidate is None:
        if not row.get("llm_waypoints"):
            raise ValueError("No --llm-waypoints provided and selected CSV row has no llm_waypoints column.")
        candidate = parse_json_field(row["llm_waypoints"])

    scores = score_conformal_contraction(candidate, nominal, m_diag, alpha, args.dt)
    steady_energy, tube = tube_profile(scores["nominal"], scores["initial_energy"], q_dyn, q_con, m_diag, alpha, epsilon, args.dt)
    steady_radius = math.sqrt(steady_energy / min(m_diag))
    accepted = scores["s_dyn"] <= q_dyn and scores["s_con"] <= q_con
    result = {
        "sample_id": args.sample_id,
        "calibration_samples": len(calibration_rows),
        "accepted": accepted,
        "s_dyn": round(scores["s_dyn"], 6),
        "s_con": round(scores["s_con"], 6),
        "q_dyn": round(q_dyn, 6),
        "q_con": round(q_con, 6),
        "delta_dyn": args.delta_dyn,
        "delta_con": args.delta_con,
        "joint_probability_lower_bound": round(1.0 - args.delta_dyn - args.delta_con, 6),
        "initial_energy": round(scores["initial_energy"], 6),
        "max_energy": round(scores["max_energy"], 6),
        "steady_energy_bound": round(steady_energy, 6),
        "steady_radius_m": round(steady_radius, 6),
        "alpha_bar": round(alpha - 0.5 * epsilon, 6),
        "output_png": str(args.output_png),
    }

    draw_scene(args.output_png, scores["candidate"], scores["nominal"], tube, obstacles, workspace, result)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
