#!/usr/bin/env python3
"""Plot the live scene, candidate path, verified safety tubes, and PX4 odometry."""

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

try:
    from min_control_qp import evaluate_sample, generate_trajectory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fine_tuning" / "scripts"))
    from min_control_qp import evaluate_sample, generate_trajectory

try:
    from lqr import A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, certified_metric_alpha, solve_care
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fine_tuning" / "scripts"))
    from lqr import A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, certified_metric_alpha, solve_care


ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
PLAN_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
PLOT_WORKSPACE = {"x": [0.0, 4.0], "y": [0.0, 4.0]}


class ContractionVisualizer(Node):
    def __init__(self):
        super().__init__("verify_contraction")
        self.declare_parameter("environment", "real")
        self.declare_parameter("refined_plan_topic", "/llm_vision/plan_refined")
        self.declare_parameter(
            "candidate_verification_topic",
            "/llm_vision/plan_candidate_verified",
        )
        self.declare_parameter("verified_plan_topic", "/llm_vision/plan_verified")
        self.declare_parameter("safety_tube_ready_topic", "/llm_vision/safety_tube_ready")
        self.declare_parameter("mission_proposal_topic", "/llm_vision/mission_proposal")
        self.declare_parameter("semantic_obstacle_topic", "/llm_vision/semantic_obstacles")
        self.declare_parameter("sim_obstacle_topic", "/llm_vision/sim_obstacles")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("output_png", "src/llm_vision_planner/plots/contraction/live_contraction.png")
        self.declare_parameter("show_window", True)
        self.declare_parameter("plot_period_s", 0.2)
        self.declare_parameter("pose_trail_limit", 1000)
        self.declare_parameter("trajectory_dt", 0.1)
        self.declare_parameter("max_horizontal_speed_mps", 0.5)
        self.declare_parameter("max_horizontal_acceleration_mps2", 0.5)
        self.declare_parameter(
            "calibration_csv",
            "fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_2000.csv",
        )
        self.declare_parameter("calibration_samples", 0)
        self.declare_parameter("delta_p", 0.10)
        self.declare_parameter("delta_w", 0.10)
        self.declare_parameter("drone_radius_m", 0.10)
        self.declare_parameter("debug", False)

        self.output_png = str(self.get_parameter("output_png").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.pose_trail_limit = int(self.get_parameter("pose_trail_limit").value)
        self.calibration_csv = self.resolve_calibration_csv(str(self.get_parameter("calibration_csv").value))
        self.delta_p = float(self.get_parameter("delta_p").value)
        self.delta_w = float(self.get_parameter("delta_w").value)
        self.q_p, self.q_w, self.state_radius_4d, self.projected_radius = self.load_certificate()
        self.drone_radius = float(self.get_parameter("drone_radius_m").value)

        self.environment = str(self.get_parameter("environment").value).strip().lower()
        self.plan = None
        self.launched = False
        self.latest_refined = None
        self.latest_verification = None
        self.latest_scene = None
        self.latest_proposal = None
        self.reference_samples = []
        self.reference_xy = []
        self.pose_trail = []
        self.latest_pose = None
        self.reference_start_timestamp_us = None
        self.latest_live_position_error = None
        self.max_live_position_error = None
        self.dirty = True
        self.figure, self.axis = plt.subplots(figsize=(8, 7))
        self.status_text = self.figure.text(0.77, 0.46, "", fontsize=9, va="top")
        if self.show_window:
            plt.ion()
            self.figure.show()

        self.safety_tube_ready_pub = self.create_publisher(
            String,
            str(self.get_parameter("safety_tube_ready_topic").value),
            PLAN_QOS,
        )

        self.create_subscription(
            String,
            str(self.get_parameter("verified_plan_topic").value),
            self.verified_plan_callback,
            PLAN_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("refined_plan_topic").value),
            self.refined_plan_callback,
            PLAN_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("candidate_verification_topic").value),
            self.candidate_verification_callback,
            PLAN_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("mission_proposal_topic").value),
            self.proposal_callback,
            PLAN_QOS,
        )
        obstacle_topic = (
            str(self.get_parameter("sim_obstacle_topic").value)
            if self.environment == "sim"
            else str(self.get_parameter("semantic_obstacle_topic").value)
        )
        self.create_subscription(String, obstacle_topic, self.obstacle_callback, 10)
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter("pose_topic").value),
            self.pose_callback,
            ODOM_QOS,
        )
        self.create_timer(float(self.get_parameter("plot_period_s").value), self.render)
        self.get_logger().info(
            f"loaded contraction certificate from {self.calibration_csv}: q_p={self.q_p:.6f} m, "
            f"q_w={self.q_w:.6f}, projected 2D radius={self.projected_radius:.6f} m; "
            f"showing live obstacles from {obstacle_topic} while waiting for a plan"
        )

    @staticmethod
    def conformal_quantile(values, delta):
        ordered = sorted(values)
        rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
        return ordered[min(rank, len(ordered)) - 1]

    @staticmethod
    def resolve_calibration_csv(value):
        requested = Path(value).expanduser()
        candidates = [requested, Path.cwd() / requested]
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.append(Path(get_package_share_directory("llm_vision_planner")) / requested)
        except Exception:
            pass
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"contraction calibration CSV does not exist: {value}")

    def load_certificate(self):
        with self.calibration_csv.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        limit = int(self.get_parameter("calibration_samples").value)
        rows = rows[:limit] if limit > 0 else rows
        position_scores = [float(row["s_p"]) for row in rows if row.get("s_p")]
        state_scores = [float(row["s_w"]) for row in rows if row.get("s_w")]
        if not position_scores or not state_scores:
            raise ValueError(f"{self.calibration_csv} must contain non-empty s_p and s_w calibration scores")
        expected_limits = {
            "max_velocity_mps": float(self.get_parameter("max_horizontal_speed_mps").value),
            "max_acceleration_mps2": float(
                self.get_parameter("max_horizontal_acceleration_mps2").value
            ),
        }
        for field, expected in expected_limits.items():
            observed = {float(row[field]) for row in rows if row.get(field)}
            if observed and any(not math.isclose(value, expected) for value in observed):
                raise ValueError(
                    f"{self.calibration_csv} uses {field}={sorted(observed)}, expected {expected}"
                )
        q_p = self.conformal_quantile(position_scores, self.delta_p)
        q_w = self.conformal_quantile(state_scores, self.delta_w)
        p, gain, _ = solve_care()
        alpha = certified_metric_alpha(A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, gain, p)
        lambda_min = float(np.min(np.linalg.eigvalsh(p)))
        position_output = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        projection_gain = math.sqrt(float(np.max(np.linalg.eigvalsh(position_output @ np.linalg.inv(p) @ position_output.T))))
        state_radius = q_w / (alpha * math.sqrt(lambda_min))
        projected_radius = projection_gain * q_w / alpha
        return q_p, q_w, state_radius, projected_radius

    def parse_payload(self, msg, source):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid {source} JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            self.get_logger().error(f"invalid {source}: expected a JSON object")
            return None
        return payload

    def obstacle_callback(self, msg):
        payload = self.parse_payload(msg, "obstacle snapshot")
        if payload is None:
            return
        self.latest_scene = payload
        self.dirty = True

    def proposal_callback(self, msg):
        payload = self.parse_payload(msg, "mission proposal")
        if payload is None:
            return
        self.latest_proposal = payload
        self.plan = None
        self.launched = False
        self.latest_refined = None
        self.latest_verification = None
        self.reference_samples = []
        self.reference_xy = []
        self.pose_trail = []
        self.reference_start_timestamp_us = None
        self.dirty = True

    def refined_plan_callback(self, msg):
        payload = self.parse_payload(msg, "refined plan")
        if payload is None:
            return
        self.latest_refined = payload
        self.latest_verification = None
        self.dirty = True

    def candidate_verification_callback(self, msg):
        payload = self.parse_payload(msg, "candidate verification")
        if payload is None:
            return
        self.latest_verification = payload
        if payload.get("passed", False):
            if self.latch_plan(payload, launched=False):
                ready = {
                    "status": "READY",
                    "plan_id": payload.get("plan_id"),
                    "sample_count": len(self.reference_xy),
                    "q_p": self.q_p,
                    "q_w": self.q_w,
                }
                self.safety_tube_ready_pub.publish(String(data=json.dumps(ready)))
        self.dirty = True

    def verified_plan_callback(self, msg):
        payload = self.parse_payload(msg, "verified plan")
        if payload is None:
            return
        self.latest_verification = payload
        if not payload.get("passed", False):
            self.dirty = True
            return
        self.latch_plan(payload, launched=True)

    def latch_plan(self, payload, launched):
        previous_plan_id = self.plan.get("plan_id") if self.plan else None
        if previous_plan_id == payload.get("plan_id") and self.reference_samples:
            self.plan = payload
            if launched and not self.launched:
                self.pose_trail = []
                self.reference_start_timestamp_us = None
                self.latest_live_position_error = None
                self.max_live_position_error = None
            self.launched = self.launched or launched
            self.dirty = True
            if launched:
                self.get_logger().info(f"launch approved for latched plan_id={payload.get('plan_id')}")
            return True
        try:
            trajectory = generate_trajectory(
                payload.get("waypoints", []),
                payload.get("workspace", {}),
                payload.get("obstacles", []),
                dt=float(self.get_parameter("trajectory_dt").value),
                max_velocity_mps=float(self.get_parameter("max_horizontal_speed_mps").value),
                max_acceleration_mps2=float(self.get_parameter("max_horizontal_acceleration_mps2").value),
            )
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f"could not generate verified QP reference: {exc}")
            return False
        self.plan = payload
        self.launched = launched
        self.reference_samples = trajectory["samples"]
        self.reference_xy = [
            (float(sample["x"][0]), float(sample["x"][1])) for sample in self.reference_samples
        ]
        self.pose_trail = []
        self.reference_start_timestamp_us = None
        self.latest_live_position_error = None
        self.max_live_position_error = None
        self.dirty = True
        self.get_logger().info(
            f"latched passed plan_id={payload.get('plan_id')} with {len(self.reference_xy)} QP samples; "
            "conformal safety tubes formed"
        )
        return True

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        point = (float(msg.position[0]), float(msg.position[1]))
        if not all(math.isfinite(value) for value in point):
            return
        self.latest_pose = point
        if self.plan is None or not self.reference_samples or not self.launched:
            self.dirty = True
            return
        if not self.pose_trail or math.dist(point, self.pose_trail[-1]) >= 0.002:
            self.pose_trail.append(point)
            self.pose_trail = self.pose_trail[-self.pose_trail_limit :]
        timestamp_us = int(msg.timestamp)
        if self.reference_start_timestamp_us is None:
            self.reference_start_timestamp_us = timestamp_us
        elapsed_s = max(0.0, (timestamp_us - self.reference_start_timestamp_us) / 1_000_000.0)
        reference_state, _ = evaluate_sample(
            self.reference_samples,
            min(elapsed_s, float(self.reference_samples[-1]["t"])),
        )
        self.latest_live_position_error = math.dist(point, (float(reference_state[0]), float(reference_state[1])))
        self.max_live_position_error = max(self.max_live_position_error or 0.0, self.latest_live_position_error)
        self.dirty = True

    def render(self):
        if not self.dirty:
            if self.show_window:
                plt.pause(0.001)
            return

        axis = self.axis
        axis.clear()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x position [m]")
        axis.set_ylabel("y position [m]")
        axis.grid(True, alpha=0.25)
        active_path = self.plan or self.latest_refined
        workspace = active_path.get("workspace", PLOT_WORKSPACE) if active_path else PLOT_WORKSPACE
        self.configure_workspace(axis, workspace)

        scene_obstacles = self.latest_scene.get("obstacles", []) if self.latest_scene else []
        planning_obstacles = active_path.get("obstacles", []) if active_path else []
        observed_obstacles = (
            scene_obstacles
            or (active_path.get("observed_obstacles", []) if active_path else [])
            or planning_obstacles
        )
        if planning_obstacles and observed_obstacles != planning_obstacles:
            self.draw_obstacles(
                axis,
                planning_obstacles,
                facecolor="#fecaca",
                edgecolor="#dc2626",
                alpha=0.16,
                linestyle="--",
                annotate=False,
            )
        self.draw_obstacles(
            axis,
            observed_obstacles,
            facecolor="#ef4444",
            edgecolor="#991b1b",
            alpha=0.42,
            annotate=True,
        )

        if active_path is not None:
            self.draw_candidate_path(axis, active_path)
        elif self.latest_proposal is not None:
            self.draw_proposed_goal(axis, self.latest_proposal)

        if self.plan is not None and self.reference_xy:
            xs = [point[0] for point in self.reference_xy]
            ys = [point[1] for point in self.reference_xy]
            display_step = max(1, len(self.reference_xy) // 30)
            for x, y in self.reference_xy[::display_step]:
                axis.add_patch(
                    Circle(
                        (x, y),
                        self.projected_radius,
                        facecolor="#c4b5fd",
                        edgecolor="#7c3aed",
                        linewidth=0.6,
                        alpha=0.06,
                        zorder=4,
                    )
                )
            for x, y in self.reference_xy[::display_step]:
                axis.add_patch(
                    Circle(
                        (x, y),
                        self.q_p,
                        facecolor="#60a5fa",
                        edgecolor="none",
                        alpha=0.10,
                        zorder=5,
                    )
                )
            axis.plot(xs, ys, "--", color="#f97316", linewidth=1.4, zorder=6)

        if self.pose_trail:
            xs = [point[0] for point in self.pose_trail]
            ys = [point[1] for point in self.pose_trail]
            axis.plot(xs, ys, color="#16a34a", linewidth=1.8, zorder=7)
        if self.latest_pose is not None:
            axis.add_patch(
                Circle(self.latest_pose, self.drone_radius, facecolor="#f43f5e", edgecolor="#7f1d1d", linewidth=1.5,
                       zorder=8)
            )
            axis.annotate(
                "UAV",
                self.latest_pose,
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=8,
                color="#7f1d1d",
                zorder=9,
            )

        phase = self.phase_text()
        axis.set_title(f"Live mission safety view\n{phase}")
        live_error = "--" if self.latest_live_position_error is None else f"{self.latest_live_position_error:.2f} m"
        scene_health = "unknown" if self.latest_scene is None else (
            "healthy" if self.latest_scene.get("healthy", True) else "unhealthy"
        )
        self.status_text.set_text(
            f"Scene: {scene_health}\nObjects: {len(observed_obstacles)}\nPhase: {phase}"
        )
        if self.plan is not None:
            legend_handles = [
                Patch(
                    facecolor="#60a5fa",
                    edgecolor="none",
                    alpha=0.35,
                    label=rf"Safety radius $q_p={self.q_p:.2f}$ m",
                ),
                Patch(
                    facecolor="#c4b5fd",
                    edgecolor="#7c3aed",
                    alpha=0.35,
                    label=rf"$q_w={self.q_w:.2f}$",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#16a34a",
                    linewidth=2.0,
                    label=f"Live tracking error {live_error}",
                ),
            ]
            axis.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )
        self.figure.subplots_adjust(left=0.10, bottom=0.11, right=0.75, top=0.92)
        output = Path(self.output_png).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        self.figure.savefig(temporary, dpi=160, format="png")
        os.replace(temporary, output)
        if self.show_window:
            self.figure.canvas.draw_idle()
            plt.pause(0.001)
        self.dirty = False

    def phase_text(self):
        if self.plan is not None:
            if self.launched:
                return "Launch approved - safety tubes active"
            return "Verified trajectory latched - awaiting final launch approval"
        if self.latest_refined is not None:
            if self.verification_matches(self.latest_refined):
                if self.latest_verification.get("passed", False):
                    return "Candidate passed - awaiting approved release"
                failed = self.latest_verification.get("failed_constraints", [])
                suffix = ", ".join(str(item) for item in failed[:2]) or "constraint check"
                return f"Refined path rejected - {suffix}"
            return "Refined candidate - verification pending"
        if self.latest_proposal is not None:
            return "Proposed goal - awaiting operator approval"
        if self.latest_scene is not None:
            return "Scene ready - waiting for operator command"
        return "Waiting for obstacle detections"

    def verification_matches(self, payload):
        if self.latest_verification is None:
            return False
        candidate_id = payload.get("plan_id")
        verification_id = self.latest_verification.get("plan_id")
        return candidate_id is None or verification_id is None or candidate_id == verification_id

    def draw_candidate_path(self, axis, payload):
        waypoints = payload.get("waypoints", [])
        points = [
            (float(point["x"]), float(point["y"]))
            for point in waypoints
            if isinstance(point, dict) and "x" in point and "y" in point
        ]
        if not points:
            return
        color = "#d97706"
        path_text = "Refined candidate"
        if self.plan is not None:
            color = "#1d4ed8"
            path_text = "Verified refined path"
        elif self.verification_matches(payload):
            if self.latest_verification.get("passed", False):
                color = "#0f766e"
                path_text = "Candidate passed"
            else:
                color = "#dc2626"
                path_text = "Candidate failed verification"
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        axis.plot(xs, ys, color=color, linewidth=2.2, marker="o", markersize=3.5, zorder=6)
        axis.scatter(*points[0], color="#111827", marker="s", s=50, zorder=8)
        axis.annotate("Start", points[0], xytext=(6, -12), textcoords="offset points", fontsize=8)
        axis.scatter(*points[-1], color="#7c3aed", marker="*", s=100, zorder=8)
        axis.annotate("Goal", points[-1], xytext=(6, 7), textcoords="offset points", fontsize=8)
        midpoint = points[len(points) // 2]
        axis.annotate(
            path_text,
            midpoint,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.82, "pad": 2.0},
            zorder=9,
        )

    @staticmethod
    def draw_proposed_goal(axis, proposal):
        goal = proposal.get("goal", {})
        if "x" not in goal or "y" not in goal:
            return
        point = (float(goal["x"]), float(goal["y"]))
        axis.scatter(*point, color="#7c3aed", marker="*", s=120, zorder=8)
        axis.annotate(
            "Proposed goal",
            point,
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=8,
            color="#5b21b6",
            bbox={"facecolor": "white", "edgecolor": "#7c3aed", "alpha": 0.82, "pad": 2.0},
        )

    @staticmethod
    def draw_obstacles(
        axis,
        obstacles,
        *,
        facecolor,
        edgecolor,
        alpha,
        linestyle="-",
        annotate=False,
    ):
        for index, obstacle in enumerate(obstacles or []):
            minimum = obstacle.get("min_corner", [0.0, 0.0, 0.0])
            maximum = obstacle.get("max_corner", [0.0, 0.0, 0.0])
            min_x = float(minimum[0])
            min_y = float(minimum[1])
            width = float(maximum[0]) - min_x
            height = float(maximum[1]) - min_y
            axis.add_patch(
                Rectangle(
                    (min_x, min_y),
                    width,
                    height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.4,
                    linestyle=linestyle,
                    alpha=alpha,
                    zorder=2,
                )
            )
            if annotate:
                label = str(
                    obstacle.get("label")
                    or obstacle.get("shape")
                    or f"obstacle_{index + 1}"
                )
                object_id = obstacle.get("object_id", obstacle.get("id"))
                if object_id is not None:
                    label += f" [{object_id}]"
                axis.text(
                    min_x + 0.5 * width,
                    min_y + 0.5 * height,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#450a0a",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                        "pad": 1.5,
                    },
                    zorder=3,
                )

    @staticmethod
    def configure_workspace(axis, workspace):
        x_limits = workspace.get("x", [0.0, 4.0])
        y_limits = workspace.get("y", [0.0, 4.0])
        axis.set_xlim(float(x_limits[0]) - 0.25, float(x_limits[1]) + 0.25)
        axis.set_ylim(float(y_limits[0]) - 0.25, float(y_limits[1]) + 0.25)

def main():
    rclpy.init()
    node = ContractionVisualizer()
    try:
        rclpy.spin(node)
    finally:
        plt.close(node.figure)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
