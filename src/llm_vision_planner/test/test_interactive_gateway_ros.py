#!/usr/bin/env python3
"""ROS integration test for approval gating and verifier-feedback retries."""

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from interactive_mission_gateway import (  # noqa: E402
    InteractiveMissionGateway,
    LATCHED_QOS,
    MissionIntent,
)


class FixedIntentParser:
    def parse(self, operator_text, object_catalog, conversation):
        del operator_text, conversation
        chair = next(item for item in object_catalog if item["label"] == "chair")
        return MissionIntent(
            status="READY",
            action="HOVER_NEAR",
            target_object_id=chair["object_id"],
            target_label="chair",
            relation="NEAR",
            requested_standoff_m=0.6,
            clarifying_question="",
        )


class GatewayHarness(Node):
    def __init__(self):
        super().__init__("interactive_gateway_harness")
        self.scene_pub = self.create_publisher(String, "/llm_vision/sim_obstacles", 10)
        self.state_pub = self.create_publisher(String, "/llm_vision/mission_state", 10)
        self.command_pub = self.create_publisher(String, "/llm_vision/operator_command", 10)
        self.approval_pub = self.create_publisher(String, "/llm_vision/mission_approval", 10)
        self.candidate_pub = self.create_publisher(
            String,
            "/llm_vision/plan_candidate_verified",
            LATCHED_QOS,
        )
        self.proposals = []
        self.prompts = []
        self.final_plans = []
        self.responses = []
        self.scene = {
            "healthy": True,
            "obstacles": [
                {
                    "id": 1,
                    "label": "chair",
                    "min_corner": [1.5, 1.5, -1.0],
                    "max_corner": [1.8, 1.8, 0.0],
                    "size": [0.3, 0.3, 1.0],
                },
                {
                    "id": 2,
                    "label": "bottle",
                    "min_corner": [3.0, 3.0, -1.0],
                    "max_corner": [3.2, 3.2, 0.0],
                    "size": [0.2, 0.2, 1.0],
                },
            ],
        }
        self.state = {
            "state": "HOLDING_FOR_PLAN",
            "position": {"x": 0.2, "y": 0.2, "z": -0.5},
        }
        self.create_subscription(String, "/llm_vision/mission_proposal", self.proposal_callback, LATCHED_QOS)
        self.create_subscription(String, "/llm_vision/prompt", self.prompt_callback, LATCHED_QOS)
        self.create_subscription(String, "/llm_vision/plan_verified", self.final_callback, LATCHED_QOS)
        self.create_subscription(String, "/llm_vision/operator_response", self.response_callback, LATCHED_QOS)

    def publish_context(self):
        self.scene_pub.publish(String(data=json.dumps(self.scene)))
        self.state_pub.publish(String(data=json.dumps(self.state)))

    def proposal_callback(self, msg):
        self.proposals.append(json.loads(msg.data))

    def prompt_callback(self, msg):
        payload = json.loads(msg.data)
        if not self.prompts or payload["plan_id"] != self.prompts[-1]["plan_id"]:
            self.prompts.append(payload)

    def final_callback(self, msg):
        self.final_plans.append(json.loads(msg.data))

    def response_callback(self, msg):
        self.responses.append(json.loads(msg.data))


def spin_until(executor, harness, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        harness.publish_context()
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for interactive gateway")


def run_test():
    rclpy.init(
        args=[
            "interactive_gateway_test",
            "--ros-args",
            "-p",
            "environment:=sim",
            "-p",
            "fresh_data_timeout_s:=5.0",
            "-p",
            "debug:=false",
        ]
    )
    gateway = InteractiveMissionGateway(intent_parser=FixedIntentParser())
    harness = GatewayHarness()
    executor = SingleThreadedExecutor()
    executor.add_node(gateway)
    executor.add_node(harness)
    try:
        spin_until(executor, harness, lambda: gateway.latest_scene is not None and gateway.latest_mission_state is not None)
        harness.command_pub.publish(String(data=json.dumps({"text": "Hover near the chair"})))
        spin_until(executor, harness, lambda: bool(harness.proposals))
        assert not harness.prompts, "planner prompt was published before approval"

        proposal = harness.proposals[-1]
        bad_approval = {
            "decision": "APPROVE",
            "mission_id": proposal["mission_id"],
            "proposal_hash": "wrong",
        }
        harness.approval_pub.publish(String(data=json.dumps(bad_approval)))
        spin_until(
            executor,
            harness,
            lambda: any(item["status"] == "APPROVAL_REJECTED" for item in harness.responses),
        )
        assert not harness.prompts

        harness.scene["obstacles"][0]["min_corner"] = [1.54, 1.48, -1.0]
        harness.scene["obstacles"][0]["max_corner"] = [1.84, 1.78, 0.0]
        harness.state["position"] = {"x": 0.26, "y": 0.23, "z": -0.48}
        spin_until(
            executor,
            harness,
            lambda: (
                gateway.latest_scene["obstacles"][0]["min_corner"][0] == 1.54
                and gateway.latest_mission_state["position"]["x"] == 0.26
            ),
        )
        approval = {
            "decision": "APPROVE",
            "mission_id": proposal["mission_id"],
            "proposal_hash": proposal["proposal_hash"],
        }
        harness.approval_pub.publish(String(data=json.dumps(approval)))
        spin_until(executor, harness, lambda: len(harness.prompts) == 1)
        first = harness.prompts[0]
        assert first["start"] == {"x": 0.26, "y": 0.23, "z": -0.5}
        planned_chair = next(item for item in first["obstacles"] if item["label"] == "chair")
        assert planned_chair["min_corner"][0] == 1.25
        assert planned_chair["max_corner"][0] == 2.05

        failed = {
            "plan_id": first["plan_id"],
            "passed": False,
            "failed_constraints": ["collision_free"],
            "verification_feedback_table": "| Metric | Value | Required | Status |\n| collision_free | false | true | FAIL |",
            "start": first["start"],
            "goal": first["goal"],
            "workspace": first["workspace"],
            "obstacles": first["obstacles"],
        }
        harness.candidate_pub.publish(String(data=json.dumps(failed)))
        spin_until(executor, harness, lambda: len(harness.prompts) == 2)
        assert not harness.final_plans, "failed candidate reached the executor topic"
        second = harness.prompts[1]
        assert second["mission_id"] == first["mission_id"]
        assert second["plan_id"] != first["plan_id"]
        assert second["attempt"] == 2
        assert "Previous plan failed verification" in second["prompt"]
        assert "collision_free" in second["prompt"]

        passed = {
            "plan_id": second["plan_id"],
            "passed": True,
            "failed_constraints": [],
            "start": second["start"],
            "goal": second["goal"],
            "workspace": second["workspace"],
            "obstacles": second["obstacles"],
            "waypoints": [second["start"], second["goal"]],
        }
        harness.candidate_pub.publish(String(data=json.dumps(passed)))
        spin_until(executor, harness, lambda: bool(harness.final_plans))
        final = harness.final_plans[-1]
        assert final["passed"] is True
        assert final["mission_id"] == first["mission_id"]
        assert final["plan_id"] == second["plan_id"]
        assert final["proposal_hash"] == proposal["proposal_hash"]
    finally:
        executor.remove_node(gateway)
        executor.remove_node(harness)
        gateway.destroy_node()
        harness.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


def run_release_drift_rejection_test():
    rclpy.init(
        args=[
            "interactive_gateway_release_drift_test",
            "--ros-args",
            "-p",
            "environment:=sim",
            "-p",
            "fresh_data_timeout_s:=5.0",
            "-p",
            "debug:=false",
        ]
    )
    gateway = InteractiveMissionGateway(intent_parser=FixedIntentParser())
    harness = GatewayHarness()
    executor = SingleThreadedExecutor()
    executor.add_node(gateway)
    executor.add_node(harness)
    try:
        spin_until(executor, harness, lambda: gateway.latest_scene is not None and gateway.latest_mission_state is not None)
        harness.command_pub.publish(String(data=json.dumps({"text": "Hover near the chair"})))
        spin_until(executor, harness, lambda: bool(harness.proposals))
        proposal = harness.proposals[-1]
        harness.approval_pub.publish(
            String(
                data=json.dumps(
                    {
                        "decision": "APPROVE",
                        "mission_id": proposal["mission_id"],
                        "proposal_hash": proposal["proposal_hash"],
                    }
                )
            )
        )
        spin_until(executor, harness, lambda: bool(harness.prompts))
        prompt = harness.prompts[-1]

        harness.state["position"] = {"x": 0.35, "y": 0.2, "z": -0.5}
        spin_until(
            executor,
            harness,
            lambda: gateway.latest_mission_state["position"]["x"] == 0.35,
        )
        passed = {
            "plan_id": prompt["plan_id"],
            "passed": True,
            "failed_constraints": [],
            "start": prompt["start"],
            "goal": prompt["goal"],
            "workspace": prompt["workspace"],
            "obstacles": prompt["obstacles"],
            "waypoints": [prompt["start"], prompt["goal"]],
        }
        harness.candidate_pub.publish(String(data=json.dumps(passed)))
        spin_until(
            executor,
            harness,
            lambda: any(
                item["status"] == "CANCELLED" and "Hover moved" in item["message"]
                for item in harness.responses
            ),
        )
        assert not harness.final_plans, "plan was released after unsafe hover drift"
    finally:
        executor.remove_node(gateway)
        executor.remove_node(harness)
        gateway.destroy_node()
        harness.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    run_test()
    run_release_drift_rejection_test()
    print("interactive gateway ROS test passed")
