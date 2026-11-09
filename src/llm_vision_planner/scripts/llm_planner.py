#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
import time
from typing import List

import rclpy
from pydantic import BaseModel, Field
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

try:
    from waypoint_transformer import WaypointTransformerPlanner
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fine_tuning" / "scripts"))
    from waypoint_transformer import WaypointTransformerPlanner
from residual_certificate import resolve_data_path

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


class Waypoint(BaseModel):
    x: float
    y: float
    z: float


class WaypointPlan(BaseModel):
    reasoning: str = Field(
        ...,
        description=(
            "Concise deployment-visible rationale using Preference, Geometry, Decision, and Safety fields."
        ),
    )
    waypoints: List[Waypoint] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Sparse ordered NED waypoints. Final waypoint must be the goal.",
    )


class LLMPlanner(Node):
    def __init__(self):
        super().__init__("llm_planner")
        self.declare_parameter("prompt_topic", PROMPT_TOPIC)
        self.declare_parameter("plan_topic", PLAN_TOPIC)
        self.declare_parameter("llm_provider", "transformer")
        self.declare_parameter("transformer_checkpoint", "")
        self.declare_parameter("transformer_device", "auto")
        self.declare_parameter("goal_tolerance_m", GOAL_TOLERANCE_M)
        self.declare_parameter("debug", True)

        self.prompt_topic = str(self.get_parameter("prompt_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.llm_provider = str(self.get_parameter("llm_provider").value).strip().lower()
        if self.llm_provider != "transformer":
            raise ValueError("live waypoint planning requires llm_provider=transformer")
        checkpoint = str(self.get_parameter("transformer_checkpoint").value)
        if not checkpoint:
            raise ValueError("transformer_checkpoint is required")
        config = resolve_data_path(str(Path(checkpoint) / "config.json"))
        self.model_name = str(config.parent)
        self.transformer = WaypointTransformerPlanner(config.parent, str(self.get_parameter("transformer_device").value))
        self.goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)

        self.prompt_sub = self.create_subscription(String, self.prompt_topic, self.prompt_callback, PROMPT_QOS)
        self.plan_pub = self.create_publisher(String, self.plan_topic, PLAN_QOS)

    def log_info(self, *args, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(*args)

    def log_warning(self, *args, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().warning(*args)

    def prompt_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse prompt payload: {exc}")
            return

        prompt = payload.get("prompt")
        if not prompt:
            self.get_logger().error("Prompt payload did not include a prompt string.")
            return

        plan_id = payload.get("plan_id")
        self.log_info(
            f"Received prompt for plan_id={plan_id}; sending request to {self.llm_provider}:{self.model_name} "
            f"({len(prompt)} chars)."
        )
        start_time = time.time()
        try:
            plan = self.request_plan(payload)
        except Exception as exc:
            self.get_logger().error(f"{self.llm_provider} planner request failed: {exc}")
            return
        duration_s = time.time() - start_time
        self.log_info(
            f"Transformer response received for plan_id={plan_id} in {duration_s:.2f}s: "
            f"reasoning={plan.reasoning!r}; waypoints={[waypoint.model_dump() for waypoint in plan.waypoints]}"
        )

        fixed_z = self.fixed_z_from_payload(payload)
        if not all(self.z_matches(fixed_z, waypoint) for waypoint in plan.waypoints):
            self.get_logger().error(f"Rejected Transformer plan because at least one waypoint z does not match fixed_z={fixed_z:.2f}.")
            return

        if not self.goal_matches(payload.get("goal", {}), plan.waypoints[-1]):
            self.get_logger().error("Rejected Transformer plan because the final waypoint does not match the goal.")
            return

        result = {
            "plan_id": payload.get("plan_id"),
            "reasoning": plan.reasoning,
            "waypoints": [waypoint.model_dump() for waypoint in plan.waypoints],
            "prompt": prompt,
            "start": payload.get("start", {}),
            "goal": payload.get("goal", {}),
            "workspace": payload.get("workspace", {}),
            "obstacles": payload.get("obstacles", []),
            "timestamp": time.time(),
            "model": self.model_name,
            "llm_provider": self.llm_provider,
        }
        if "goal_relations" in payload:
            result["goal_relations"] = payload["goal_relations"]
        for field in (
            "residual_calibration",
            "ground_truth_obstacles",
            "mission_id",
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
            "route_preference_text",
            "selected_expert_route",
        ):
            if field in payload:
                result[field] = payload[field]

        out = String()
        out.data = json.dumps(result)
        self.plan_pub.publish(out)
        self.log_info(f"Published sparse plan with {len(plan.waypoints)} waypoints using {self.model_name}.")

    @staticmethod
    def fixed_z_from_payload(payload):
        workspace = payload.get("workspace", {})
        if "z" in workspace:
            return float(workspace["z"])
        goal = payload.get("goal", {})
        return float(goal.get("z", -0.2))

    def request_plan(self, payload):
        preference = payload.get("route_preference_text") or "Take the shortest safe route"
        waypoints = self.transformer.predict(payload["start"], payload["goal"],
                                             payload.get("obstacles", []), payload["workspace"], preference)
        return WaypointPlan(reasoning="Local structured Transformer waypoint prediction",
                            waypoints=[Waypoint(**point) for point in waypoints])

    def z_matches(self, fixed_z, waypoint):
        return abs(float(waypoint.z) - fixed_z) <= self.goal_tolerance_m

    def goal_matches(self, goal, waypoint):
        if not goal:
            return False
        dx = float(goal.get("x", 0.0)) - waypoint.x
        dy = float(goal.get("y", 0.0)) - waypoint.y
        dz = float(goal.get("z", -0.2)) - waypoint.z
        return math.sqrt(dx * dx + dy * dy + dz * dz) <= self.goal_tolerance_m


def main():
    rclpy.init()
    node = LLMPlanner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
