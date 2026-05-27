#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

RAW_PLAN_TOPIC = "/llm_vision/plan_raw"
REFINED_PLAN_TOPIC = "/llm_vision/plan_refined"
INTERPOLATION_SPACING_M = 1.0
SAFETY_MARGIN_M = 0.40
NUDGE_EPSILON_M = 0.02
FIXED_Z = -0.25
PLAN_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class PathRefinement(Node):
    def __init__(self):
        super().__init__("path_refinement")
        self.declare_parameter("raw_plan_topic", RAW_PLAN_TOPIC)
        self.declare_parameter("refined_plan_topic", REFINED_PLAN_TOPIC)
        self.declare_parameter("interpolation_spacing_m", INTERPOLATION_SPACING_M)
        self.declare_parameter("safety_margin_m", SAFETY_MARGIN_M)
        self.declare_parameter("nudge_epsilon_m", NUDGE_EPSILON_M)
        self.declare_parameter("fixed_z", FIXED_Z)
        self.declare_parameter("debug", False)

        self.raw_plan_topic = str(self.get_parameter("raw_plan_topic").value)
        self.refined_plan_topic = str(self.get_parameter("refined_plan_topic").value)
        self.interpolation_spacing_m = float(self.get_parameter("interpolation_spacing_m").value)
        self.safety_margin_m = float(self.get_parameter("safety_margin_m").value)
        self.nudge_epsilon_m = float(self.get_parameter("nudge_epsilon_m").value)
        self.fixed_z = float(self.get_parameter("fixed_z").value)

        self.plan_sub = self.create_subscription(String, self.raw_plan_topic, self.plan_callback, PLAN_QOS)
        self.refined_pub = self.create_publisher(String, self.refined_plan_topic, PLAN_QOS)

    def log_info(self, *args, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(*args)

    def plan_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse raw plan JSON: {exc}")
            return

        waypoints = payload.get("waypoints", [])
        if len(waypoints) < 2:
            self.get_logger().error("Refinement requires at least two sparse waypoints.")
            return

        workspace = payload.get("workspace", {})
        obstacles = payload.get("obstacles", [])
        refined = self.interpolate_waypoints(waypoints, workspace, obstacles)

        output = dict(payload)
        output["waypoints_sparse"] = waypoints
        output["waypoints"] = refined
        output["timestamp_refined"] = time.time()
        output["refinement"] = {
            "spacing_m": self.interpolation_spacing_m,
            "safety_margin_m": self.safety_margin_m,
        }

        out = String()
        out.data = json.dumps(output)
        self.refined_pub.publish(out)
        self.log_info(f"Published refined plan with {len(refined)} waypoints.")

    def interpolate_waypoints(self, waypoints, workspace, obstacles):
        refined = [self.sanitize_waypoint(waypoints[0], workspace, obstacles, preserve_goal=False)]
        for start, end in zip(waypoints, waypoints[1:]):
            dx = float(end["x"]) - float(start["x"])
            dy = float(end["y"]) - float(start["y"])
            distance = math.hypot(dx, dy)
            steps = max(1, int(math.ceil(distance / self.interpolation_spacing_m)))

            for step in range(1, steps + 1):
                t = step / steps
                candidate = {
                    "x": float(start["x"]) + dx * t,
                    "y": float(start["y"]) + dy * t,
                    "z": self.fixed_z,
                }
                preserve_goal = step == steps
                adjusted = self.sanitize_waypoint(candidate, workspace, obstacles, preserve_goal=preserve_goal)
                if not self.same_xy(refined[-1], adjusted):
                    refined.append(adjusted)
        return refined

    def sanitize_waypoint(self, waypoint, workspace, obstacles, preserve_goal):
        point = {
            "x": float(waypoint["x"]),
            "y": float(waypoint["y"]),
            "z": self.fixed_z,
        }
        if not preserve_goal:
            point = self.nudge_point(point, obstacles)
        return self.clamp_to_workspace(point, workspace)

    def nudge_point(self, point, obstacles):
        adjusted = dict(point)
        for obstacle in obstacles:
            adjusted = self.nudge_away_from_obstacle(adjusted, obstacle)
        return adjusted

    def nudge_away_from_obstacle(self, point, obstacle):
        min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
        max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
        min_x = float(min_corner[0]) - self.safety_margin_m
        max_x = float(max_corner[0]) + self.safety_margin_m
        min_y = float(min_corner[1]) - self.safety_margin_m
        max_y = float(max_corner[1]) + self.safety_margin_m

        if not (min_x <= point["x"] <= max_x and min_y <= point["y"] <= max_y):
            return point

        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        left_gap = abs(point["x"] - min_x)
        right_gap = abs(max_x - point["x"])
        bottom_gap = abs(point["y"] - min_y)
        top_gap = abs(max_y - point["y"])
        smallest_gap = min(left_gap, right_gap, bottom_gap, top_gap)

        if smallest_gap == left_gap:
            point["x"] = min_x - self.nudge_epsilon_m
        elif smallest_gap == right_gap:
            point["x"] = max_x + self.nudge_epsilon_m
        elif smallest_gap == bottom_gap:
            point["y"] = min_y - self.nudge_epsilon_m
        else:
            point["y"] = max_y + self.nudge_epsilon_m

        if math.isclose(point["x"], center_x, abs_tol=1e-6):
            point["x"] += self.nudge_epsilon_m
        if math.isclose(point["y"], center_y, abs_tol=1e-6):
            point["y"] += self.nudge_epsilon_m
        return point

    def clamp_to_workspace(self, point, workspace):
        x_limits = workspace.get("x", [0.0, 4.0])
        y_limits = workspace.get("y", [0.0, 4.0])
        point["x"] = min(max(point["x"], float(x_limits[0])), float(x_limits[1]))
        point["y"] = min(max(point["y"], float(y_limits[0])), float(y_limits[1]))
        point["z"] = self.fixed_z
        return point

    @staticmethod
    def same_xy(a, b):
        return math.isclose(a["x"], b["x"], abs_tol=1e-6) and math.isclose(a["y"], b["y"], abs_tol=1e-6)


def main():
    rclpy.init()
    node = PathRefinement()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
