#!/usr/bin/env python3
import json
import math
import os
import time
from typing import List

import instructor
import rclpy
from openai import OpenAI
from pydantic import BaseModel, Field
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

PROMPT_TOPIC = "/llm_vision/prompt"
PLAN_TOPIC = "/llm_vision/plan_raw"
MODEL_NAME = "gpt-5-mini"
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
        description="2-3 concise statements explaining obstacle-avoidance routing.",
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
        self.declare_parameter("model_name", MODEL_NAME)
        self.declare_parameter("goal_tolerance_m", GOAL_TOLERANCE_M)
        self.declare_parameter("debug", True)

        self.prompt_topic = str(self.get_parameter("prompt_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)

        self.prompt_sub = self.create_subscription(String, self.prompt_topic, self.prompt_callback, PROMPT_QOS)
        self.plan_pub = self.create_publisher(String, self.plan_topic, PLAN_QOS)
        self.client = instructor.from_openai(OpenAI())

        if not os.getenv("OPENAI_API_KEY"):
            self.log_warning("OPENAI_API_KEY is not set; planner requests will fail until it is provided.")

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
            f"Received prompt for plan_id={plan_id}; sending request to {self.model_name} "
            f"({len(prompt)} chars)."
        )
        start_time = time.time()
        try:
            plan = self.client.chat.completions.create(
                model=self.model_name,
                response_model=WaypointPlan,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            self.get_logger().error(f"OpenAI planner request failed: {exc}")
            return
        duration_s = time.time() - start_time
        self.log_info(
            f"LLM response received for plan_id={plan_id} in {duration_s:.2f}s: "
            f"reasoning={plan.reasoning!r}; waypoints={[waypoint.model_dump() for waypoint in plan.waypoints]}"
        )

        fixed_z = self.fixed_z_from_payload(payload)
        if not all(self.z_matches(fixed_z, waypoint) for waypoint in plan.waypoints):
            self.get_logger().error(f"Rejected LLM plan because at least one waypoint z does not match fixed_z={fixed_z:.2f}.")
            return

        if not self.goal_matches(payload.get("goal", {}), plan.waypoints[-1]):
            self.get_logger().error("Rejected LLM plan because the final waypoint does not match the goal.")
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
        }

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
