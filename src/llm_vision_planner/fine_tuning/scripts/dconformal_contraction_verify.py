#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from lqr import A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, DAMPING, certified_metric_alpha, solve_care
from min_control_qp import evaluate_sample, generate_shared_pair, generate_trajectory, propagate_state
from position_score import closed_loop_trace, position_scores
from rrt import plan_rrt

from conformal_rrt_dataset import (
    DEFAULT_CLEARANCE_M,
    DEFAULT_LLAMA_MODEL_NAME,
    DEFAULT_WORKSPACE,
    DEFAULT_VLLM_BASE_URL,
    load_prompt_generator,
    make_refiner,
    make_verifier,
    prompt_from_current_generator,
    refine_and_verify,
    request_llama_waypoints,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "datasets"
PLOTS_DIR = SCRIPT_DIR.parent / "plots" / "contraction"
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "contraction"
DEFAULT_CALIBRATION_CSV = DATASET_DIR / "calibration_min_control_qp_position_score_with_limits_2000.csv"
DEFAULT_OUTPUT_PNG = PLOTS_DIR / "offline_certificate.png"
DEFAULT_REPORT_JSON = RESULTS_DIR / "offline_certificate.json"
DEFAULT_CONTROL_LAW_TOPIC = "/llm_vision/dconformal_control_law"
DEFAULT_POSE_TOPIC = "/fmu/out/vehicle_odometry"


def parse_json_field(value):
    return json.loads(value) if isinstance(value, str) else value


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_json(value):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def conformal_quantile(values, delta):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def waypoints_from_row(row, prefix):
    verified_key = f"{prefix}_verified_waypoints"
    if row.get(verified_key):
        return parse_json_field(row[verified_key])
    return parse_json_field(row[f"{prefix}_waypoints"])


def trajectory_from_row(row, prefix, durations=None):
    trajectory_key = f"{prefix}_trajectory"
    if durations is None and row.get(trajectory_key):
        return parse_json_field(row[trajectory_key])
    waypoints = waypoints_from_row(row, prefix)
    workspace = parse_json_field(row["workspace"])
    obstacles = parse_json_field(row["obstacles"])
    return generate_trajectory(waypoints, workspace, obstacles, durations=durations)


def feedback_prompt(base_prompt, metrics):
    failed = metrics.get("failed_constraints", [])
    table = metrics.get("feedback_table", "")
    return (
        base_prompt
        + "\nPrevious plan failed verification. Regenerate waypoints for the same start, goal, workspace, and obstacles.\n"
        + f"Failed constraints: {', '.join(failed) if failed else 'unknown'}\n"
        + table
    )


def query_live_llm(args, row):
    from openai import OpenAI
    import instructor

    raw_client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
    prompt_module = load_prompt_generator()
    refiner = make_refiner()
    verifier = make_verifier()
    base_prompt, _ = prompt_from_current_generator(prompt_module, row["start"], row["goal"], row["workspace"], row["obstacles"])
    prompt = base_prompt
    last_metrics = None
    last_refined = []

    for attempt in range(1, args.llm_attempts + 1):
        print(f"[live attempt {attempt}/{args.llm_attempts}] querying Llama with prompt_chars={len(prompt)}", flush=True)
        try:
            raw = request_llama_waypoints(client, args.llama_model_name, prompt, args.temperature, args.llm_retries)
        except RuntimeError as exc:
            print(f"[live attempt {attempt}] Llama request failed: {exc}", flush=True)
            continue
        print(f"[live attempt {attempt}] raw LLM waypoints={len(raw)}", flush=True)
        try:
            verified, metrics = refine_and_verify(refiner, verifier, raw, row)
        except ValueError as exc:
            last_refined = refiner.interpolate_waypoints(raw, row["workspace"], row["obstacles"])
            last_metrics = verifier.compute_metrics(
                {
                    "waypoints": last_refined,
                    "start": row["start"],
                    "obstacles": row["obstacles"],
                    "workspace": row["workspace"],
                    "goal": row["goal"],
                }
            )
            print(
                f"[live attempt {attempt}] verification failed: {exc}; "
                f"failed_constraints={last_metrics['failed_constraints']}",
                flush=True,
            )
            prompt = feedback_prompt(base_prompt, last_metrics)
            continue
        trajectory = generate_trajectory(
            verified,
            row["workspace"],
            row["obstacles"],
            dt=args.trajectory_dt,
            max_velocity_mps=args.max_velocity_mps,
            max_acceleration_mps2=args.max_acceleration_mps2,
        )
        print(f"[live attempt {attempt}] verified waypoints={len(verified)}, QP samples={len(trajectory['samples'])}", flush=True)
        return raw, verified, metrics, trajectory, prompt

    raise RuntimeError(f"LLM plan did not pass verification after {args.llm_attempts} attempts: {last_metrics}; last_refined={last_refined}")


def propagate_controller(rrt_trajectory, llm_trajectory, k, dt):
    return closed_loop_trace(rrt_trajectory, llm_trajectory, k, dt)


def propagate_llm_reference(llm_trajectory, k, dt, initial_state):
    horizon = float(llm_trajectory["samples"][-1]["t"])
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    state = np.array(initial_state, dtype=float)
    result = []
    for index, t in enumerate(times):
        xhat, uhat = evaluate_sample(llm_trajectory["samples"], float(t))
        control = -k @ (state - xhat) + uhat
        result.append({"t": float(t), "x": state.tolist(), "u": control.tolist(), "xhat": xhat.tolist(), "uhat": uhat.tolist()})
        if index + 1 < len(times):
            state = propagate_state(state, control, float(times[index + 1] - t))
    return result


def position_tube_profile(closed_loop, q_p):
    return [{"t": float(sample["t"]), "radius": float(q_p)} for sample in closed_loop]


def draw_scene(output_png, workspace, obstacles, rrt_samples, llm_samples, closed_loop, tube, result):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(workspace["x"][0]) - 0.25, float(workspace["x"][1]) + 0.25)
    axis.set_ylim(float(workspace["y"][0]) - 0.25, float(workspace["y"][1]) + 0.25)
    axis.set_xlabel("x position [m]")
    axis.set_ylabel("y position [m]")
    expert_label = "Semantic Theta*" if result["expert"] == "semantic_theta" else "RRT"
    axis.set_title(f"{expert_label}: QP tracking certificate comparison")
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
            )
        )

    sampled_closed_loop = closed_loop[:: max(1, len(closed_loop) // 30)]
    sampled_tube = tube[:: max(1, len(tube) // 30)]
    for index, sample in enumerate(sampled_closed_loop):
        xd = sample["xd"]
        axis.add_patch(
            Circle(
                (xd[0], xd[1]),
                result["projected_position_radius_m"],
                facecolor="#c4b5fd",
                edgecolor="#7c3aed",
                linewidth=0.6,
                alpha=0.06,
                label=(
                    rf"{100 * (1 - result['delta_w']):g}% projected 2D: $q_w={result['q_w']:.3f}$, "
                    rf"$\rho_p={result['projected_position_radius_m']:.3f}$ m"
                    if index == 0
                    else None
                ),
            )
        )
    for index, (sample, tube_sample) in enumerate(zip(sampled_closed_loop, sampled_tube)):
        xd = sample["xd"]
        axis.add_patch(
            Circle(
                (xd[0], xd[1]),
                tube_sample["radius"],
                facecolor="#60a5fa",
                edgecolor="none",
                alpha=0.10,
                label=rf"{100 * (1 - result['delta_p']):g}% direct cross-track: $q_p={result['q_p']:.3f}$ m" if index == 0 else None,
            )
        )

    axis.plot(
        [],
        [],
        ":",
        color="#111827",
        label=(
            rf"{100 * (1 - result['delta_w']):g}% 4D state: $q_w={result['q_w']:.3f}$, "
            rf"$\rho_{{4D}}={result['state_radius_4d']:.3f}$ (not a position circle)"
        ),
    )

    axis.plot([s["x"][0] for s in rrt_samples], [s["x"][1] for s in rrt_samples], color="#2563eb", label=expert_label + r" expert reference $x_d$")
    axis.plot(
        [s["x"][0] for s in llm_samples],
        [s["x"][1] for s in llm_samples],
        "--",
        color="#f97316",
        label=r"LLM QP reference $\hat{x}_d$",
    )
    axis.plot([s["x"][0] for s in closed_loop], [s["x"][1] for s in closed_loop], color="#16a34a", label=r"closed-loop state $x$")
    axis.scatter([rrt_samples[0]["x"][0]], [rrt_samples[0]["x"][1]], color="#111827", s=60, marker="s", label="initial position")
    axis.scatter([rrt_samples[-1]["x"][0]], [rrt_samples[-1]["x"][1]], color="#7c3aed", s=100, marker="*", label="goal position")
    annotation = (
        f"$s_p$={result['s_p']:.3f} m {'≤' if result['score_accepted'] else '>'} $q_p$={result['q_p']:.3f} m\n"
        f"projected 2D radius={result['projected_position_radius_m']:.3f} m\n"
        f"4D state radius={result['state_radius_4d']:.3f} (mixed state units)\n"
        f"equal-time position error={result['s_position_time']:.3f} m"
    )
    axis.text(
        0.02,
        0.02,
        annotation,
        transform=axis.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def path_xy_from_samples(samples):
    return [float(sample["x"][0]) for sample in samples], [float(sample["x"][1]) for sample in samples]


def draw_live_scene(output_png, workspace, obstacles, raw_llm, verified_llm, llm_trajectory, commanded, actual_pose_trail, raw_rrt=None, verified_rrt_trajectory=None):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(workspace["x"][0]) - 0.25, float(workspace["x"][1]) + 0.25)
    axis.set_ylim(float(workspace["y"][0]) - 0.25, float(workspace["y"][1]) + 0.25)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Live conformal control law")
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
                alpha=0.28,
            )
        )

    if raw_rrt:
        axis.scatter([p["x"] for p in raw_rrt], [p["y"] for p in raw_rrt], color="#7c3aed", marker="x", s=70, label="raw RRT waypoints")
    if raw_llm:
        axis.scatter([p["x"] for p in raw_llm], [p["y"] for p in raw_llm], color="#f97316", marker="o", s=45, label="raw LLM waypoints")
    if verified_rrt_trajectory:
        xs, ys = path_xy_from_samples(verified_rrt_trajectory["samples"])
        axis.plot(xs, ys, color="#2563eb", linewidth=2, label="verified RRT final")
    xs, ys = path_xy_from_samples(llm_trajectory["samples"])
    axis.plot(xs, ys, "--", color="#f97316", linewidth=2, label="verified LLM final")
    axis.plot([s["x"][0] for s in commanded], [s["x"][1] for s in commanded], color="#16a34a", linewidth=2, label="commanded control law")
    if actual_pose_trail:
        axis.plot([p["x"] for p in actual_pose_trail], [p["y"] for p in actual_pose_trail], color="#111827", linewidth=2, label="actual PX4 pose")
        axis.scatter([actual_pose_trail[-1]["x"]], [actual_pose_trail[-1]["y"]], color="#111827", marker="D", s=65)
    axis.scatter([llm_trajectory["samples"][0]["x"][0]], [llm_trajectory["samples"][0]["x"][1]], color="#0f172a", marker="s", s=60, label="start")
    axis.scatter([verified_llm[-1]["x"]], [verified_llm[-1]["y"]], color="#dc2626", marker="*", s=100, label="goal")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def make_control_law_payload(args, row, raw_llm, verified_llm, metrics, llm_trajectory, k, q_p, calibration_count, raw_rrt=None, verified_rrt_trajectory=None):
    return {
        "trajectory": llm_trajectory,
        "K": k.tolist(),
        "start": row["start"],
        "goal": row["goal"],
        "workspace": row["workspace"],
        "obstacles": row["obstacles"],
        "raw_llm_waypoints": raw_llm,
        "verified_llm_waypoints": verified_llm,
        "verification_metrics": metrics,
        "raw_rrt_waypoints": raw_rrt,
        "verified_rrt_trajectory": verified_rrt_trajectory,
        "q_p": q_p,
        "radius_m": q_p,
        "delta_p": args.delta_p,
        "score_type": "closed_loop_cross_track_position",
        "calibration_samples": calibration_count,
        "timestamp": time.time(),
    }


class LivePlotPublisher:
    def __init__(self, args, payload, commanded):
        import rclpy
        from px4_msgs.msg import VehicleOdometry
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from std_msgs.msg import String

        class NodeImpl(Node):
            def __init__(self, outer):
                super().__init__("dconformal_contraction_live_verify")
                self.outer = outer
                odom_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
                control_qos = QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                self.publisher = self.create_publisher(String, args.control_law_topic, control_qos)
                self.odom_sub = self.create_subscription(VehicleOdometry, args.pose_topic, self.odom_callback, odom_qos)
                self.timer = self.create_timer(args.plot_period_s, self.tick)
                self.publish_payload()

            def publish_payload(self):
                msg = String()
                msg.data = json.dumps(payload, separators=(",", ":"))
                self.publisher.publish(msg)
                self.get_logger().info(f"published control law on {args.control_law_topic}")

            def odom_callback(self, msg):
                if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
                    return
                self.outer.pose_trail.append({"x": float(msg.position[0]), "y": float(msg.position[1]), "z": float(msg.position[2])})
                self.outer.pose_trail = self.outer.pose_trail[-args.pose_trail_limit :]

            def tick(self):
                self.publish_payload()
                draw_live_scene(
                    args.output_png,
                    payload["workspace"],
                    payload["obstacles"],
                    payload["raw_llm_waypoints"],
                    payload["verified_llm_waypoints"],
                    payload["trajectory"],
                    commanded,
                    self.outer.pose_trail,
                    raw_rrt=payload.get("raw_rrt_waypoints"),
                    verified_rrt_trajectory=payload.get("verified_rrt_trajectory"),
                )

        self.rclpy = rclpy
        self.external_shutdown_exception = ExternalShutdownException
        self.rclpy.init()
        self.pose_trail = []
        self.node = NodeImpl(self)

    def spin(self):
        try:
            self.rclpy.spin(self.node)
        except (KeyboardInterrupt, self.external_shutdown_exception):
            pass
        finally:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Verify the direct cross-track position certificate for QP tracking.")
    parser.add_argument("--live", action="store_true", help="Run live LLM design, publish control-law message, and plot odometry.")
    parser.add_argument("--start", default=None, help='JSON point for live mode, e.g. {"x":0,"y":0,"z":-0.25}')
    parser.add_argument("--goal", default=None, help='JSON point for live mode, e.g. {"x":2.5,"y":0,"z":-0.25}')
    parser.add_argument("--workspace", default=json.dumps(DEFAULT_WORKSPACE))
    parser.add_argument("--obstacles", default="[]")
    parser.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION_CSV)
    parser.add_argument("--expert", choices=("rrt", "semantic_theta"), default="rrt", help="Expert column prefix for offline calibration verification and plot labels.")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--delta-p", type=float, default=0.10)
    parser.add_argument("--delta-w", type=float, default=0.10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--control-law-topic", default=DEFAULT_CONTROL_LAW_TOPIC)
    parser.add_argument("--pose-topic", default=DEFAULT_POSE_TOPIC)
    parser.add_argument("--llama-model-name", default=DEFAULT_LLAMA_MODEL_NAME)
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-attempts", type=int, default=3)
    parser.add_argument("--trajectory-dt", type=float, default=0.1)
    parser.add_argument("--max-velocity-mps", type=float, default=0.5)
    parser.add_argument("--max-acceleration-mps2", type=float, default=0.5)
    parser.add_argument("--plot-period-s", type=float, default=0.5)
    parser.add_argument("--pose-trail-limit", type=int, default=300)
    parser.add_argument("--show-rrt", action="store_true", help="In live mode, compute RRT only for final-plan visualization.")
    args = parser.parse_args()
    if args.live and args.expert != "rrt":
        parser.error("--expert semantic_theta is supported for offline verification only.")

    rows = load_rows(args.calibration_csv)
    if not rows:
        raise RuntimeError(f"No calibration rows found in {args.calibration_csv}")
    calibration_rows = rows[: args.calibration_samples] if args.calibration_samples else rows
    if not all(row.get("s_p") and row.get("s_w") for row in calibration_rows):
        raise ValueError("The QP conformal verifier requires calibration columns s_p and s_w.")
    expected_limits = {
        "max_velocity_mps": args.max_velocity_mps,
        "max_acceleration_mps2": args.max_acceleration_mps2,
    }
    for field, expected in expected_limits.items():
        observed = {float(row[field]) for row in calibration_rows if row.get(field)}
        if observed and any(not math.isclose(value, expected) for value in observed):
            raise ValueError(f"Calibration uses {field}={sorted(observed)}, expected {expected}")
    p, k, _ = solve_care()
    q_p = conformal_quantile([item["s_p"] for item in calibration_rows], args.delta_p)
    q_w = conformal_quantile([item["s_w"] for item in calibration_rows], args.delta_w)
    alpha_p = certified_metric_alpha(A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, k, p)
    lambda_min = float(np.min(np.linalg.eigvalsh(p)))
    position_output = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    projection_gain = math.sqrt(float(np.max(np.linalg.eigvalsh(position_output @ np.linalg.inv(p) @ position_output.T))))
    state_radius_4d = q_w / (alpha_p * math.sqrt(lambda_min))
    projected_position_radius = projection_gain * q_w / alpha_p
    print(
        f"[calibration] rows={len(calibration_rows)}, delta_p={args.delta_p}, "
        f"q_p={round(q_p, 6)} m, q_w={round(q_w, 6)}, "
        f"rho_4d={round(state_radius_4d, 6)}, rho_projected_2d={round(projected_position_radius, 6)} m",
        flush=True,
    )

    if args.live:
        if not args.start or not args.goal:
            raise ValueError("--live requires --start and --goal.")
        row = {
            "start": load_json(args.start),
            "goal": load_json(args.goal),
            "workspace": load_json(args.workspace),
            "obstacles": load_json(args.obstacles),
        }
        raw_llm, verified_llm, metrics, llm_trajectory, _ = query_live_llm(args, row)
        initial_state = [float(row["start"]["x"]), float(row["start"]["y"]), 0.0, 0.0]
        commanded = propagate_llm_reference(llm_trajectory, k, args.dt, initial_state)
        raw_rrt = None
        verified_rrt_trajectory = None
        if args.show_rrt:
            raw_rrt = plan_rrt(
                row["start"],
                row["goal"],
                row["obstacles"],
                workspace=row["workspace"],
                clearance_m=DEFAULT_CLEARANCE_M,
                seed=7,
            )
            try:
                verified_rrt, _ = refine_and_verify(make_refiner(), make_verifier(), raw_rrt, row)
                verified_rrt_trajectory = generate_trajectory(
                    verified_rrt,
                    row["workspace"],
                    row["obstacles"],
                    dt=args.trajectory_dt,
                    max_velocity_mps=args.max_velocity_mps,
                    max_acceleration_mps2=args.max_acceleration_mps2,
                )
                print(f"[live RRT] verified waypoints={len(verified_rrt)}, QP samples={len(verified_rrt_trajectory['samples'])}", flush=True)
            except ValueError as exc:
                print(f"[live RRT] verification failed, plotting raw markers only: {exc}", flush=True)
        payload = make_control_law_payload(
            args,
            row,
            raw_llm,
            verified_llm,
            metrics,
            llm_trajectory,
            k,
            q_p,
            len(calibration_rows),
            raw_rrt=raw_rrt,
            verified_rrt_trajectory=verified_rrt_trajectory,
        )
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps({**payload, "commanded": commanded}, indent=2), encoding="utf-8")
        draw_live_scene(
            args.output_png,
            row["workspace"],
            row["obstacles"],
            raw_llm,
            verified_llm,
            llm_trajectory,
            commanded,
            [],
            raw_rrt=raw_rrt,
            verified_rrt_trajectory=verified_rrt_trajectory,
        )
        print(json.dumps({"q_p": q_p, "radius_m": q_p, "K": k.round(6).tolist(), "control_law_topic": args.control_law_topic}, indent=2))
        live = LivePlotPublisher(args, payload, commanded)
        live.spin()
        return

    if args.sample_id < 0 or args.sample_id >= len(rows):
        raise ValueError(f"--sample-id must be between 0 and {len(rows) - 1}.")
    row = rows[args.sample_id]

    if row.get(f"{args.expert}_trajectory") and row.get("llm_trajectory"):
        rrt_trajectory = trajectory_from_row(row, args.expert)
        llm_trajectory = trajectory_from_row(row, "llm")
    else:
        rrt_trajectory, llm_trajectory = generate_shared_pair(
            waypoints_from_row(row, args.expert),
            waypoints_from_row(row, "llm"),
            parse_json_field(row["workspace"]),
            parse_json_field(row["obstacles"]),
            dt=args.trajectory_dt,
            max_velocity_mps=args.max_velocity_mps,
            max_acceleration_mps2=args.max_acceleration_mps2,
        )
    closed_loop = propagate_controller(rrt_trajectory, llm_trajectory, k, args.dt)
    calculated_scores = position_scores(rrt_trajectory, llm_trajectory, k, args.dt)
    s_p = float(row["s_p"]) if row.get("s_p") else calculated_scores["s_p"]
    s_position_time = float(row["s_position_time"]) if row.get("s_position_time") else calculated_scores["s_position_time"]
    tube = position_tube_profile(closed_loop, q_p)
    result = {
        "expert": args.expert,
        "score_type": "closed_loop_cross_track_position",
        "sample_id": args.sample_id,
        "calibration_samples": len(calibration_rows),
        "s_p": round(s_p, 6),
        "s_position_time": round(s_position_time, 6),
        "q_p": round(q_p, 6),
        "delta_p": args.delta_p,
        "q_w": round(q_w, 6),
        "delta_w": args.delta_w,
        "score_accepted": bool(s_p <= q_p),
        "coverage": round(1.0 - args.delta_p, 6),
        "K": k.round(6).tolist(),
        "damping": DAMPING,
        "control_dt": args.dt,
        "max_velocity_mps": args.max_velocity_mps,
        "max_acceleration_mps2": args.max_acceleration_mps2,
        "radius_m": round(q_p, 6),
        "alpha_p": round(alpha_p, 6),
        "projection_gain": round(projection_gain, 6),
        "state_radius_4d": round(state_radius_4d, 6),
        "projected_position_radius_m": round(projected_position_radius, 6),
        "inside_position_tube": bool(calculated_scores["s_p"] <= q_p),
        "output_png": str(args.output_png),
    }

    workspace = parse_json_field(row["workspace"])
    obstacles = parse_json_field(row["obstacles"])
    draw_scene(args.output_png, workspace, obstacles, rrt_trajectory["samples"], llm_trajectory["samples"], closed_loop, tube, result)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
