#!/usr/bin/env python3
"""Plot a verified QP reference, conformal tube, and live PX4 odometry."""

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

try:
    from min_control_qp import generate_trajectory
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fine_tuning" / "scripts"))
    from min_control_qp import generate_trajectory

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
        self.declare_parameter("verified_plan_topic", "/llm_vision/plan_verified")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("output_png", "src/llm_vision_planner/plots/contraction/live_contraction.png")
        self.declare_parameter("show_window", True)
        self.declare_parameter("plot_period_s", 0.2)
        self.declare_parameter("pose_trail_limit", 1000)
        self.declare_parameter("trajectory_dt", 0.1)
        self.declare_parameter("calibration_csv", "fine_tuning/datasets/calibration_min_control_qp_position_score_2000.csv")
        self.declare_parameter("calibration_samples", 0)
        self.declare_parameter("delta_w", 0.10)
        self.declare_parameter("drone_radius_m", 0.10)
        self.declare_parameter("debug", False)

        self.output_png = str(self.get_parameter("output_png").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.pose_trail_limit = int(self.get_parameter("pose_trail_limit").value)
        self.calibration_csv = self.resolve_calibration_csv(str(self.get_parameter("calibration_csv").value))
        self.delta_w = float(self.get_parameter("delta_w").value)
        self.q_w, self.state_radius_4d, self.projected_radius = self.load_certificate()
        self.drone_radius = float(self.get_parameter("drone_radius_m").value)

        self.plan = None
        self.reference_xy = []
        self.pose_trail = []
        self.dirty = True
        self.figure, self.axis = plt.subplots(figsize=(8, 7), facecolor="#f8fafc")
        if self.show_window:
            plt.ion()
            self.figure.show()

        self.create_subscription(
            String,
            str(self.get_parameter("verified_plan_topic").value),
            self.plan_callback,
            PLAN_QOS,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter("pose_topic").value),
            self.pose_callback,
            ODOM_QOS,
        )
        self.create_timer(float(self.get_parameter("plot_period_s").value), self.render)
        self.get_logger().info(
            f"loaded contraction certificate from {self.calibration_csv}: q_w={self.q_w:.6f}, "
            f"projected 2D radius={self.projected_radius:.6f} m; waiting for a passed verified plan and live PX4 odometry"
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
        scores = [float(row["s_w"]) for row in rows if row.get("s_w")]
        if not scores:
            raise ValueError(f"{self.calibration_csv} must contain non-empty s_w calibration scores")
        q_w = self.conformal_quantile(scores, self.delta_w)
        p, gain, _ = solve_care()
        alpha = certified_metric_alpha(A_DOUBLE_INTEGRATOR, B_DOUBLE_INTEGRATOR, gain, p)
        lambda_min = float(np.min(np.linalg.eigvalsh(p)))
        position_output = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        projection_gain = math.sqrt(float(np.max(np.linalg.eigvalsh(position_output @ np.linalg.inv(p) @ position_output.T))))
        state_radius = q_w / (alpha * math.sqrt(lambda_min))
        projected_radius = projection_gain * q_w / alpha
        return q_w, state_radius, projected_radius

    def plan_callback(self, msg):
        if self.plan is not None:
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid verified-plan JSON: {exc}")
            return
        if not payload.get("passed", False):
            return
        try:
            trajectory = generate_trajectory(
                payload.get("waypoints", []),
                payload.get("workspace", {}),
                payload.get("obstacles", []),
                dt=float(self.get_parameter("trajectory_dt").value),
            )
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(f"could not generate verified QP reference: {exc}")
            return
        self.plan = payload
        self.reference_xy = [
            (float(sample["x"][0]), float(sample["x"][1])) for sample in trajectory["samples"]
        ]
        self.dirty = True
        self.get_logger().info(
            f"latched passed plan_id={payload.get('plan_id')} with {len(self.reference_xy)} QP samples"
        )

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        point = (float(msg.position[0]), float(msg.position[1]))
        if not all(math.isfinite(value) for value in point):
            return
        if not self.pose_trail or math.dist(point, self.pose_trail[-1]) >= 0.002:
            self.pose_trail.append(point)
            self.pose_trail = self.pose_trail[-self.pose_trail_limit :]
            self.dirty = True

    def render(self):
        if not self.dirty:
            if self.show_window:
                plt.pause(0.001)
            return

        axis = self.axis
        axis.clear()
        axis.set_facecolor("#f8fafc")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m] (PX4 local NED)")
        axis.set_ylabel("y [m] (PX4 local NED)")
        axis.set_title("Live PX4 contraction verification")
        axis.grid(True, color="#94a3b8", alpha=0.30)
        self.configure_workspace(axis, PLOT_WORKSPACE)

        if self.plan is not None:
            self.draw_obstacles(axis, self.plan.get("obstacles", []))
            xs = [point[0] for point in self.reference_xy]
            ys = [point[1] for point in self.reference_xy]
            axis.plot(xs, ys, "--", color="#f59e0b", linewidth=2.5, label=r"LLM QP reference $\hat{x}_d$")
            display_step = max(1, len(self.reference_xy) // 14)
            for x, y in self.reference_xy[::display_step]:
                axis.add_patch(Circle((x, y), self.projected_radius, color="#8b5cf6", alpha=0.06))
            axis.plot([], [], color="#8b5cf6", alpha=0.65, linewidth=7,
                      label=rf"projected contraction tube $\rho_{{2D}}={self.projected_radius:.3f}$ m")
            start = self.reference_xy[0]
            goal = self.reference_xy[-1]
            axis.scatter(*start, color="#0ea5e9", marker="s", s=70, label="verified start")
            axis.scatter(*goal, color="#ef4444", marker="*", s=130, label="verified goal")

        if self.pose_trail:
            xs = [point[0] for point in self.pose_trail]
            ys = [point[1] for point in self.pose_trail]
            axis.plot(xs, ys, color="#06b6d4", linewidth=2.5, label="actual PX4 trajectory")
            axis.add_patch(
                Circle(self.pose_trail[-1], self.drone_radius, facecolor="#f43f5e", edgecolor="#7f1d1d", linewidth=1.5,
                       zorder=8, label="live drone")
            )

        axis.text(
            0.02,
            0.02,
            f"coverage={100.0 * (1.0 - self.delta_w):.0f}%\n"
            f"q_w={self.q_w:.6f}\n"
            f"projected 2D radius={self.projected_radius:.6f} m\n"
            f"4D state radius={self.state_radius_4d:.6f} (mixed state units)",
            transform=axis.transAxes,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
        )
        axis.legend(loc="upper right", fontsize=8)
        self.figure.tight_layout()
        output_dir = os.path.dirname(self.output_png)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.figure.savefig(self.output_png, dpi=160)
        if self.show_window:
            self.figure.canvas.draw_idle()
            plt.pause(0.001)
        self.dirty = False

    @staticmethod
    def draw_obstacles(axis, obstacles):
        for obstacle in obstacles:
            minimum = obstacle.get("min_corner", [0.0, 0.0, 0.0])
            maximum = obstacle.get("max_corner", [0.0, 0.0, 0.0])
            axis.add_patch(
                Rectangle(
                    (float(minimum[0]), float(minimum[1])),
                    float(maximum[0]) - float(minimum[0]),
                    float(maximum[1]) - float(minimum[1]),
                    facecolor="#ef4444",
                    edgecolor="#991b1b",
                    alpha=0.28,
                )
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
