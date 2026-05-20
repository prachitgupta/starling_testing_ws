#!/usr/bin/env python3
import json
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

RAW_PLAN_TOPIC = "/llm_vision/plan_raw"
REFINED_PLAN_TOPIC = "/llm_vision/plan_refined"
VERIFIED_PLAN_TOPIC = "/llm_vision/plan_verified"
OBSTACLE_TOPIC = "/llm_vision/obstacles"
SEMANTIC_OBSTACLE_TOPIC = "/llm_vision/semantic_obstacles"
POSE_TOPIC = "/fmu/out/vehicle_odometry"
DEFAULT_OUTPUT_PNG = "/tmp/llm_vision_plot.png"
DEFAULT_Z = -0.45

ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class PlannerVisualizer(Node):
    def __init__(self):
        super().__init__("planner_visualizer")
        self.declare_parameter("raw_plan_topic", RAW_PLAN_TOPIC)
        self.declare_parameter("refined_plan_topic", REFINED_PLAN_TOPIC)
        self.declare_parameter("verified_plan_topic", VERIFIED_PLAN_TOPIC)
        self.declare_parameter("obstacle_topic", OBSTACLE_TOPIC)
        self.declare_parameter("semantic_obstacle_topic", SEMANTIC_OBSTACLE_TOPIC)
        self.declare_parameter("pose_topic", POSE_TOPIC)
        self.declare_parameter("output_png", DEFAULT_OUTPUT_PNG)
        self.declare_parameter("show_window", True)
        self.declare_parameter("fixed_z", DEFAULT_Z)
        self.declare_parameter("workspace_x_min", 0.0)
        self.declare_parameter("workspace_x_max", 4.0)
        self.declare_parameter("workspace_y_min", 0.0)
        self.declare_parameter("workspace_y_max", 4.0)
        self.raw_plan_topic = str(self.get_parameter("raw_plan_topic").value)
        self.refined_plan_topic = str(self.get_parameter("refined_plan_topic").value)
        self.verified_plan_topic = str(self.get_parameter("verified_plan_topic").value)
        self.obstacle_topic = str(self.get_parameter("obstacle_topic").value)
        self.semantic_obstacle_topic = str(self.get_parameter("semantic_obstacle_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.output_png = str(self.get_parameter("output_png").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.fixed_z = float(self.get_parameter("fixed_z").value)
        self.default_workspace = {
            "x": [
                float(self.get_parameter("workspace_x_min").value),
                float(self.get_parameter("workspace_x_max").value),
            ],
            "y": [
                float(self.get_parameter("workspace_y_min").value),
                float(self.get_parameter("workspace_y_max").value),
            ],
        }

        self.latest_sparse = None
        self.latest_refined = None
        self.latched_verified = None
        self.latest_obstacle_payload = None
        self.current_pose = None
        self.last_draw_signature = None

        self.figure, self.axis = plt.subplots(figsize=(9, 7))
        if self.show_window:
            plt.ion()
            self.figure.show()

        self.raw_sub = self.create_subscription(String, self.raw_plan_topic, self.raw_plan_callback, 10)
        self.refined_sub = self.create_subscription(String, self.refined_plan_topic, self.refined_plan_callback, 10)
        self.verified_sub = self.create_subscription(String, self.verified_plan_topic, self.verified_plan_callback, 10)
        self.obstacle_sub = self.create_subscription(String, self.obstacle_topic, self.obstacle_callback, 10)
        self.semantic_obstacle_sub = self.create_subscription(String, self.semantic_obstacle_topic, self.obstacle_callback, 10)
        self.pose_sub = self.create_subscription(VehicleOdometry, self.pose_topic, self.pose_callback, ODOM_QOS)
        self.timer = self.create_timer(0.5, self.render)

        self.get_logger().info(
            f"Matplotlib visualizer writing live 2D z={self.fixed_z:.1f} slice to {self.output_png}"
        )

    def raw_plan_callback(self, msg):
        try:
            self.latest_sparse = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse /llm_vision/plan_raw JSON: {exc}")

    def refined_plan_callback(self, msg):
        try:
            self.latest_refined = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse /llm_vision/plan_refined JSON: {exc}")

    def verified_plan_callback(self, msg):
        if self.latched_verified is not None:
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse /llm_vision/plan_verified JSON: {exc}")
            return

        if not payload.get("passed", False):
            self.get_logger().warning("Ignoring failed verified trajectory for plot latch.")
            return

        if payload.get("plan_id") is None:
            self.get_logger().warning("Latched verified trajectory has no plan_id; restart all planner nodes if prompt_generator is waiting on a plan_id.")

        self.latched_verified = payload
        self.get_logger().info(
            f"Latched first passed verified trajectory for plotting with plan_id={payload.get('plan_id')}; "
            "path display is now frozen."
        )

    def obstacle_callback(self, msg):
        try:
            self.latest_obstacle_payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse obstacle JSON: {exc}")
            return

        pose = self.pose_from_payload(self.latest_obstacle_payload)
        if pose is not None:
            self.current_pose = pose

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.get_logger().warning("Ignoring VehicleOdometry that is not in NED pose frame.", throttle_duration_sec=5.0)
            return
        self.current_pose = {
            "x": float(msg.position[0]),
            "y": float(msg.position[1]),
            "z": float(msg.position[2]) if not math.isnan(float(msg.position[2])) else self.fixed_z,
        }

    def render(self):
        payload = self.latched_verified or self.latest_refined or self.latest_sparse or self.latest_obstacle_payload or {}
        verified_latched = self.latched_verified is not None
        sparse_waypoints = self.sparse_waypoints(payload)
        refined_waypoints = self.refined_waypoints(payload)
        obstacles = self.obstacles(payload)
        goal = self.goal(payload)
        workspace = payload.get("workspace") or self.workspace_from_scene(obstacles, goal)

        signature = json.dumps(
            {
                "sparse": sparse_waypoints,
                "refined": refined_waypoints,
                "obstacles": obstacles,
                "goal": goal,
                "pose": self.current_pose,
                "workspace": workspace,
                "verified_latched": verified_latched,
            },
            sort_keys=True,
        )
        if signature == self.last_draw_signature:
            if self.show_window:
                plt.pause(0.001)
            return

        self.axis.clear()
        self.draw_obstacles(obstacles)
        if verified_latched:
            self.draw_path(sparse_waypoints, "Sparse LLM path used for verified plan", color="#94a3b8", marker="o", linestyle="--")
            self.draw_path(refined_waypoints, "Latched verified trajectory", color="#10b981", marker=".", linestyle="-")
        else:
            self.draw_path(sparse_waypoints, "Sparse LLM path", color="#f59e0b", marker="o", linestyle="--")
            self.draw_path(refined_waypoints, "Refined trajectory awaiting verification", color="#2563eb", marker=".", linestyle="-")
        self.draw_goal(goal)
        self.draw_drone()
        self.configure_axes(workspace)

        os.makedirs(os.path.dirname(self.output_png), exist_ok=True)
        self.figure.tight_layout()
        self.figure.savefig(self.output_png, dpi=140)
        if self.show_window:
            self.figure.canvas.draw_idle()
            plt.pause(0.001)

        self.last_draw_signature = signature
        self.get_logger().info(f"Updated 2D planner plot at {self.output_png}")

    def sparse_waypoints(self, payload):
        if self.latched_verified is not None:
            return payload.get("waypoints_sparse", [])
        if self.latest_sparse is not None:
            return self.latest_sparse.get("waypoints", [])
        return payload.get("waypoints_sparse", [])

    def refined_waypoints(self, payload):
        if self.latched_verified is not None:
            return payload.get("waypoints", [])
        if self.latest_refined is None:
            return []
        return self.latest_refined.get("waypoints", [])

    def obstacles(self, payload):
        obstacles = payload.get("obstacles", [])
        if not obstacles and self.latest_obstacle_payload is not None:
            obstacles = self.latest_obstacle_payload.get("obstacles", [])
        return obstacles

    def goal(self, payload):
        goal = payload.get("goal", {})
        if not goal and self.latest_obstacle_payload is not None:
            goal = self.latest_obstacle_payload.get("goal", {})
        return self.normalize_goal(goal)

    def draw_obstacles(self, obstacles):
        for index, obstacle in enumerate(obstacles, start=1):
            min_corner, max_corner = self.obstacle_bounds(obstacle)
            x0, x1 = self.padded_bounds(min_corner[0], max_corner[0])
            y0, y1 = self.padded_bounds(min_corner[1], max_corner[1])
            width = x1 - x0
            height = y1 - y0
            label = obstacle.get("label") or obstacle.get("shape") or f"obstacle_{index}"

            rect = Rectangle(
                (x0, y0),
                width,
                height,
                facecolor="#ef4444",
                edgecolor="#991b1b",
                linewidth=2,
                alpha=0.28,
                label="Obstacle x-y span" if index == 1 else None,
            )
            self.axis.add_patch(rect)
            self.axis.text(
                x0 + width / 2.0,
                y0 + height / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="#7f1d1d",
                weight="bold",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#fecaca", "alpha": 0.8},
            )

    def draw_path(self, waypoints, label, color, marker, linestyle):
        if not waypoints:
            return
        xs = [float(point["x"]) for point in waypoints]
        ys = [float(point["y"]) for point in waypoints]
        self.axis.plot(xs, ys, color=color, marker=marker, linestyle=linestyle, linewidth=2, label=label)
        for index, point in enumerate(waypoints, start=1):
            self.axis.text(float(point["x"]), float(point["y"]), str(index), color=color, fontsize=8)

    def draw_goal(self, goal):
        if not goal:
            return
        self.axis.scatter(
            [float(goal.get("x", 0.0))],
            [float(goal.get("y", 0.0))],
            marker="x",
            s=130,
            linewidths=3,
            color="#dc2626",
            label="Goal",
        )

    def draw_drone(self):
        if self.current_pose is None:
            return
        self.axis.scatter(
            [self.current_pose["x"]],
            [self.current_pose["y"]],
            marker="D",
            s=90,
            color="#16a34a",
            edgecolors="#14532d",
            linewidths=1.5,
            label="Drone",
        )

    def configure_axes(self, workspace):
        x_limits = workspace.get("x", [0.0, 4.0])
        y_limits = workspace.get("y", [0.0, 4.0])
        self.axis.set_xlim(float(x_limits[0]) - 0.5, float(x_limits[1]) + 0.5)
        self.axis.set_ylim(float(y_limits[0]) - 0.5, float(y_limits[1]) + 0.5)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
        self.axis.set_xlabel("NED X (m)")
        self.axis.set_ylabel("NED Y (m)")
        self.axis.set_title(f"LLM Vision Planner 2D Slice at z={self.fixed_z:.1f}")
        handles, labels = self.axis.get_legend_handles_labels()
        if handles:
            self.axis.legend(loc="upper right")

    def workspace_from_scene(self, obstacles, goal):
        xs = list(self.default_workspace["x"])
        ys = list(self.default_workspace["y"])
        if self.current_pose is not None:
            xs.append(self.current_pose["x"])
            ys.append(self.current_pose["y"])
        if goal:
            xs.append(float(goal.get("x", 0.0)))
            ys.append(float(goal.get("y", 0.0)))
        for obstacle in obstacles:
            min_corner, max_corner = self.obstacle_bounds(obstacle)
            xs.extend([float(min_corner[0]), float(max_corner[0])])
            ys.extend([float(min_corner[1]), float(max_corner[1])])
        return {"x": [min(xs) - 0.75, max(xs) + 0.75], "y": [min(ys) - 0.75, max(ys) + 0.75]}

    @staticmethod
    def obstacle_bounds(obstacle):
        min_corner = obstacle.get("min_corner")
        max_corner = obstacle.get("max_corner")
        if min_corner is not None and max_corner is not None:
            return list(min_corner), list(max_corner)

        centroid = obstacle.get("centroid", [0.0, 0.0, DEFAULT_Z])
        size = obstacle.get("size", [0.4, 0.4, 0.4])
        min_corner = [float(centroid[i]) - float(size[i]) / 2.0 for i in range(3)]
        max_corner = [float(centroid[i]) + float(size[i]) / 2.0 for i in range(3)]
        return min_corner, max_corner

    @staticmethod
    def padded_bounds(min_value, max_value, minimum_span=0.25):
        min_value = float(min_value)
        max_value = float(max_value)
        if max_value < min_value:
            min_value, max_value = max_value, min_value
        if math.isclose(min_value, max_value, abs_tol=1e-6):
            half_span = minimum_span / 2.0
            return min_value - half_span, max_value + half_span
        return min_value, max_value

    def pose_from_payload(self, payload):
        pose = payload.get("pose")
        if isinstance(pose, dict) and "x" in pose and "y" in pose:
            return {"x": float(pose["x"]), "y": float(pose["y"]), "z": float(pose.get("z", self.fixed_z))}
        if isinstance(pose, (list, tuple)) and len(pose) >= 3:
            return {"x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2])}
        return None

    @staticmethod
    def normalize_goal(goal):
        if isinstance(goal, dict):
            return goal
        if isinstance(goal, (list, tuple)) and len(goal) >= 3:
            return {"x": goal[0], "y": goal[1], "z": goal[2]}
        return {}


def main():
    rclpy.init()
    node = PlannerVisualizer()
    try:
        rclpy.spin(node)
    finally:
        plt.close(node.figure)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
