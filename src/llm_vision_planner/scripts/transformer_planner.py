#!/usr/bin/env python3
"""ROS 2 adapter for the final vertical V1 waypoint Transformer."""

import json
import math
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_SCRIPTS = PACKAGE_ROOT / "fine_tuning" / "scripts"
if TRANSFORMER_SCRIPTS.is_dir() and str(TRANSFORMER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_SCRIPTS))

from transformer_runtime import (  # noqa: E402
    DEFAULT_ROUTE_PREFERENCE,
    FINAL_CHECKPOINT_RELATIVE,
    WaypointTransformerPlanner,
)


PROMPT_TOPIC = "/llm_vision/prompt"
PLAN_TOPIC = "/llm_vision/plan_raw"
GOAL_TOLERANCE_M = 0.05
PROMPT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
PLAN_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class TransformerPlannerNode(Node):
    def __init__(self):
        super().__init__("transformer_planner")
        self.declare_parameter("prompt_topic", PROMPT_TOPIC)
        self.declare_parameter("plan_topic", PLAN_TOPIC)
        self.declare_parameter("transformer_checkpoint", str(FINAL_CHECKPOINT_RELATIVE))
        self.declare_parameter("transformer_device", "auto")
        self.declare_parameter("default_route_preference", DEFAULT_ROUTE_PREFERENCE)
        self.declare_parameter("goal_tolerance_m", GOAL_TOLERANCE_M)
        self.declare_parameter("debug", True)

        self.prompt_topic = str(self.get_parameter("prompt_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.default_route_preference = str(
            self.get_parameter("default_route_preference").value
        )
        self.goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)
        checkpoint = str(self.get_parameter("transformer_checkpoint").value)
        device = str(self.get_parameter("transformer_device").value)
        self.planner = WaypointTransformerPlanner(checkpoint, device)

        self.prompt_sub = self.create_subscription(
            String, self.prompt_topic, self.prompt_callback, PROMPT_QOS
        )
        self.plan_pub = self.create_publisher(String, self.plan_topic, PLAN_QOS)
        self.log_info(
            f"Loaded final V1 Transformer from {self.planner.checkpoint} on {self.planner.device}."
        )

    def log_info(self, message):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(message)

    def prompt_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            start = payload["start"]
            goal = payload["goal"]
            workspace = payload["workspace"]
            obstacles = payload["obstacles"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self.get_logger().error(f"Invalid Transformer planning payload: {exc}")
            return

        preference = str(
            payload.get("route_preference_text") or self.default_route_preference
        )
        feedback = str(payload.get("verification_feedback") or "").strip()
        if feedback:
            preference = f"{preference} Correct the previous verification failure: {feedback}"

        plan_id = payload.get("plan_id")
        start_time = time.time()
        try:
            waypoints = self.planner.predict(
                start, goal, obstacles, workspace, preference
            )
        except Exception as exc:
            self.get_logger().error(f"Transformer inference failed for plan_id={plan_id}: {exc}")
            return

        fixed_z = self.fixed_z_from_payload(payload)
        if not all(abs(float(item["z"]) - fixed_z) <= self.goal_tolerance_m for item in waypoints):
            self.get_logger().error("Rejected Transformer plan with inconsistent fixed-z values.")
            return
        if not self.goal_matches(goal, waypoints[-1]):
            self.get_logger().error("Rejected Transformer plan because its endpoint is not the goal.")
            return

        result = {
            "plan_id": plan_id,
            "reasoning": "Final V1 Transformer inference using the frozen scene and approved route preference.",
            "waypoints": waypoints,
            "prompt": payload.get("prompt", ""),
            "start": start,
            "goal": goal,
            "workspace": workspace,
            "obstacles": obstacles,
            "route_preference_text": preference,
            "timestamp": time.time(),
            "model": self.planner.checkpoint.name,
            "planner_provider": "transformer",
            "transformer_checkpoint": str(self.planner.checkpoint),
        }
        for field in (
            "goal_relations",
            "nominal_obstacles",
            "obstacle_safety",
            "obs_safety_bracket",
            "vision_error_quantile_m",
            "vision_error_delta",
            "vision_error_calibration_trials",
            "vision_error_calibration_rank",
            "vision_error_calibration_csv",
            "vision_error_calibration_placeholder",
            "scene_guard_band_m",
            "interaction_mode",
            "selected_expert_route",
        ):
            if field in payload:
                result[field] = payload[field]

        self.plan_pub.publish(String(data=json.dumps(result)))
        self.log_info(
            f"Published {len(waypoints)} Transformer waypoints for plan_id={plan_id} "
            f"in {time.time() - start_time:.3f}s."
        )

    @staticmethod
    def fixed_z_from_payload(payload):
        workspace = payload.get("workspace", {})
        if "z" in workspace:
            return float(workspace["z"])
        return float(payload.get("goal", {}).get("z", -0.2))

    def goal_matches(self, goal, waypoint):
        dx = float(goal.get("x", 0.0)) - float(waypoint["x"])
        dy = float(goal.get("y", 0.0)) - float(waypoint["y"])
        dz = float(goal.get("z", -0.2)) - float(waypoint["z"])
        return math.sqrt(dx * dx + dy * dy + dz * dz) <= self.goal_tolerance_m


def main():
    rclpy.init()
    node = TransformerPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
