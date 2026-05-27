#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

REFINED_PLAN_TOPIC = "/llm_vision/plan_refined"
VERIFIED_PLAN_TOPIC = "/llm_vision/plan_verified"
SAFETY_MARGIN_M = 0.40
INTERPOLATION_SPACING_M = 1.0
CRUISE_SPEED_MPS = 0.5
NOMINAL_DT_S = INTERPOLATION_SPACING_M / CRUISE_SPEED_MPS
MAX_VELOCITY_MPS = 1.5
MAX_ACCELERATION_MPS2 = 1.5
GOAL_TOLERANCE_M = 0.05
PROGRESS_TOLERANCE_M = 1e-3
VERIFIED_PLAN_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
REFINED_PLAN_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class PathVerifier(Node):
    def __init__(self):
        super().__init__("path_verifier")
        self.declare_parameter("refined_plan_topic", REFINED_PLAN_TOPIC)
        self.declare_parameter("verified_plan_topic", VERIFIED_PLAN_TOPIC)
        self.declare_parameter("safety_margin_m", SAFETY_MARGIN_M)
        self.declare_parameter("interpolation_spacing_m", INTERPOLATION_SPACING_M)
        self.declare_parameter("cruise_speed_mps", CRUISE_SPEED_MPS)
        self.declare_parameter("max_velocity_mps", MAX_VELOCITY_MPS)
        self.declare_parameter("max_acceleration_mps2", MAX_ACCELERATION_MPS2)
        self.declare_parameter("goal_tolerance_m", GOAL_TOLERANCE_M)
        self.declare_parameter("progress_tolerance_m", PROGRESS_TOLERANCE_M)
        self.declare_parameter("debug", False)

        self.refined_plan_topic = str(self.get_parameter("refined_plan_topic").value)
        self.verified_plan_topic = str(self.get_parameter("verified_plan_topic").value)
        self.safety_margin_m = float(self.get_parameter("safety_margin_m").value)
        self.interpolation_spacing_m = float(self.get_parameter("interpolation_spacing_m").value)
        self.cruise_speed_mps = float(self.get_parameter("cruise_speed_mps").value)
        self.max_velocity_mps = float(self.get_parameter("max_velocity_mps").value)
        self.max_acceleration_mps2 = float(self.get_parameter("max_acceleration_mps2").value)
        self.goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)
        self.progress_tolerance_m = float(self.get_parameter("progress_tolerance_m").value)
        self.nominal_dt_s = self.interpolation_spacing_m / self.cruise_speed_mps

        self.plan_sub = self.create_subscription(String, self.refined_plan_topic, self.plan_callback, REFINED_PLAN_QOS)
        self.verified_pub = self.create_publisher(String, self.verified_plan_topic, VERIFIED_PLAN_QOS)

    def log_info(self, *args, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(*args)

    def plan_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse refined plan JSON: {exc}")
            return

        metrics = self.compute_metrics(payload)
        output = dict(payload)
        output["metrics"] = metrics
        output["passed"] = metrics["passed"]
        output["failed_constraints"] = metrics["failed_constraints"]
        output["thresholds"] = metrics["thresholds"]
        output["verification_feedback_table"] = metrics["feedback_table"]
        output["timestamp_verified"] = time.time()

        out = String()
        out.data = json.dumps(output)
        self.verified_pub.publish(out)
        self.log_info(
            "Verification %s: clearance=%.2f m, max_vel=%.2f m/s, max_accel=%.2f m/s^2"
            % (
                "passed" if metrics["passed"] else "failed",
                metrics["min_clearance_m"],
                metrics["max_segment_speed"],
                metrics["max_segment_accel"],
            )
        )

    def compute_metrics(self, payload):
        waypoints = payload.get("waypoints", [])
        obstacles = payload.get("obstacles", [])
        workspace = payload.get("workspace", {})
        goal = payload.get("goal", {})

        in_workspace = all(self.point_in_workspace(point, workspace) for point in waypoints)
        clearance_values = [self.clearance_to_obstacles(point, obstacles) for point in waypoints]
        min_clearance = min(clearance_values) if clearance_values else 0.0
        goal_clearance = self.clearance_to_obstacles(goal, obstacles) if goal else 0.0
        goal_clearance_ok = bool(goal) and goal_clearance >= self.safety_margin_m
        collision_free = bool(waypoints) and min_clearance >= self.safety_margin_m
        goal_match = self.goal_matches(waypoints[-1], goal) if waypoints else False
        monotonic_progress = self.is_monotonic_progress(waypoints, goal)
        max_speed, max_accel = self.kinematic_metrics(waypoints)
        smoothness = self.smoothness_score(waypoints)

        checks = {
            "has_waypoints": bool(waypoints),
            "in_workspace": in_workspace,
            "goal_clearance": goal_clearance_ok,
            "collision_free": collision_free,
            "goal_match": goal_match,
            "monotonic_goal_progress": monotonic_progress,
            "max_segment_speed": max_speed <= self.max_velocity_mps,
            "max_segment_accel": max_accel <= self.max_acceleration_mps2,
        }
        failed_constraints = [name for name, ok in checks.items() if not ok]
        thresholds = {
            "min_clearance_m": self.safety_margin_m,
            "max_segment_speed": self.max_velocity_mps,
            "max_segment_accel": self.max_acceleration_mps2,
            "goal_tolerance_m": self.goal_tolerance_m,
            "progress_tolerance_m": self.progress_tolerance_m,
        }
        feedback_table = self.feedback_table(
            min_clearance,
            max_speed,
            max_accel,
            in_workspace,
            goal_clearance,
            goal_clearance_ok,
            collision_free,
            goal_match,
            monotonic_progress,
        )
        passed = not failed_constraints

        return {
            "collision_free": collision_free,
            "min_clearance_m": round(min_clearance, 3),
            "goal_clearance_m": round(goal_clearance, 3),
            "max_segment_speed": round(max_speed, 3),
            "max_segment_accel": round(max_accel, 3),
            "monotonic_goal_progress": monotonic_progress,
            "smoothness_score": round(smoothness, 3),
            "passed": passed,
            "goal_match": goal_match,
            "in_workspace": in_workspace,
            "nominal_dt_s": round(self.nominal_dt_s, 3),
            "failed_constraints": failed_constraints,
            "thresholds": thresholds,
            "feedback_table": feedback_table,
        }

    def feedback_table(
        self,
        min_clearance,
        max_speed,
        max_accel,
        in_workspace,
        goal_clearance,
        goal_clearance_ok,
        collision_free,
        goal_match,
        monotonic_progress,
    ):
        rows = [
            ("in_workspace", str(in_workspace), "true", in_workspace),
            ("goal_clearance", f"{goal_clearance:.3f}", f">= {self.safety_margin_m:.3f}", goal_clearance_ok),
            ("collision_free", str(collision_free), "true", collision_free),
            ("min_clearance_m", f"{min_clearance:.3f}", f">= {self.safety_margin_m:.3f}", min_clearance >= self.safety_margin_m),
            ("goal_match", str(goal_match), "true", goal_match),
            ("monotonic_goal_progress", str(monotonic_progress), "true", monotonic_progress),
            ("max_segment_speed", f"{max_speed:.3f}", f"<= {self.max_velocity_mps:.3f}", max_speed <= self.max_velocity_mps),
            ("max_segment_accel", f"{max_accel:.3f}", f"<= {self.max_acceleration_mps2:.3f}", max_accel <= self.max_acceleration_mps2),
        ]
        lines = ["| Metric | Value | Required | Status |", "|---|---:|---:|---|"]
        for metric, value, required, ok in rows:
            lines.append(f"| {metric} | {value} | {required} | {'PASS' if ok else 'FAIL'} |")
        return "\n".join(lines)

    @staticmethod
    def point_in_workspace(point, workspace):
        x_limits = workspace.get("x", [0.0, 4.0])
        y_limits = workspace.get("y", [0.0, 4.0])
        return (
            float(x_limits[0]) <= float(point["x"]) <= float(x_limits[1])
            and float(y_limits[0]) <= float(point["y"]) <= float(y_limits[1])
        )

    def clearance_to_obstacles(self, point, obstacles):
        if not obstacles:
            return float("inf")
        distances = [self.clearance_to_box(point, obstacle) for obstacle in obstacles]
        return min(distances)

    @staticmethod
    def clearance_to_box(point, obstacle):
        min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
        max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
        min_x = float(min_corner[0])
        max_x = float(max_corner[0])
        min_y = float(min_corner[1])
        max_y = float(max_corner[1])
        dx = max(min_x - float(point["x"]), 0.0, float(point["x"]) - max_x)
        dy = max(min_y - float(point["y"]), 0.0, float(point["y"]) - max_y)
        if dx == 0.0 and dy == 0.0:
            edge_x = min(abs(float(point["x"]) - min_x), abs(max_x - float(point["x"])))
            edge_y = min(abs(float(point["y"]) - min_y), abs(max_y - float(point["y"])))
            return -min(edge_x, edge_y)
        return math.hypot(dx, dy)

    def goal_matches(self, waypoint, goal):
        dx = float(goal.get("x", 0.0)) - float(waypoint["x"])
        dy = float(goal.get("y", 0.0)) - float(waypoint["y"])
        dz = float(goal.get("z", -0.2)) - float(waypoint["z"])
        return math.sqrt(dx * dx + dy * dy + dz * dz) <= self.goal_tolerance_m

    def is_monotonic_progress(self, waypoints, goal):
        if not waypoints:
            return False
        last_distance = None
        for point in waypoints:
            distance = math.hypot(float(goal.get("x", 0.0)) - float(point["x"]), float(goal.get("y", 0.0)) - float(point["y"]))
            if last_distance is not None and distance > last_distance + self.progress_tolerance_m:
                return False
            last_distance = distance
        return True

    def kinematic_metrics(self, waypoints):
        if len(waypoints) < 2:
            return 0.0, 0.0

        velocities = []
        max_speed = 0.0
        for first, second in zip(waypoints, waypoints[1:]):
            vx = (float(second["x"]) - float(first["x"])) / self.nominal_dt_s
            vy = (float(second["y"]) - float(first["y"])) / self.nominal_dt_s
            velocities.append((vx, vy))
            max_speed = max(max_speed, math.hypot(vx, vy))

        max_accel = 0.0
        for first, second in zip(velocities, velocities[1:]):
            ax = (second[0] - first[0]) / self.nominal_dt_s
            ay = (second[1] - first[1]) / self.nominal_dt_s
            max_accel = max(max_accel, math.hypot(ax, ay))

        return max_speed, max_accel

    @staticmethod
    def smoothness_score(waypoints):
        if len(waypoints) < 3:
            return 1.0

        deltas = []
        for first, second in zip(waypoints, waypoints[1:]):
            deltas.append((float(second["x"]) - float(first["x"]), float(second["y"]) - float(first["y"])))

        heading_changes = []
        for first, second in zip(deltas, deltas[1:]):
            heading_a = math.atan2(first[1], first[0])
            heading_b = math.atan2(second[1], second[0])
            delta = abs(math.atan2(math.sin(heading_b - heading_a), math.cos(heading_b - heading_a)))
            heading_changes.append(delta)

        if not heading_changes:
            return 1.0
        mean_change = sum(heading_changes) / len(heading_changes)
        return 1.0 / (1.0 + mean_change)


def main():
    rclpy.init()
    node = PathVerifier()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
