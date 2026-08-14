#!/usr/bin/env python3
"""Backend-only interactive simulation using a recorded obstacle environment."""

import argparse
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

from dataset_scene_publisher import DatasetScenePublisher  # noqa: E402
from interactive_mission_gateway import InteractiveMissionGateway, LATCHED_QOS  # noqa: E402
from refinment import PathRefinement  # noqa: E402
from verifier import PathVerifier  # noqa: E402
from verify_contraction import ContractionVisualizer  # noqa: E402


class PlannerHarness(Node):
    """Emulate the Llama boundary while exercising real refinement and verification."""

    def __init__(self):
        super().__init__("interactive_sim_planner_harness")
        self.raw_pub = self.create_publisher(String, "/llm_vision/plan_raw", LATCHED_QOS)
        self.create_subscription(String, "/llm_vision/prompt", self.prompt_callback, LATCHED_QOS)
        self.seen_plan_ids = set()

    def prompt_callback(self, msg):
        payload = json.loads(msg.data)
        plan_id = payload["plan_id"]
        if plan_id in self.seen_plan_ids:
            return
        self.seen_plan_ids.add(plan_id)
        start = dict(payload["start"])
        goal = dict(payload["goal"])
        if int(payload["attempt"]) == 1:
            first = dict(start)
            first["x"] += 0.5
            waypoints = [first, goal]
            reasoning = "Simulation-injected start mismatch to exercise verifier feedback."
        else:
            waypoints = [start, goal]
            reasoning = "Direct monotonic standoff route for the recorded simulation scene."
        output = {
            "plan_id": plan_id,
            "reasoning": reasoning,
            "waypoints": waypoints,
            "prompt": payload["prompt"],
            "start": start,
            "goal": goal,
            "goal_relations": payload.get("goal_relations", []),
            "workspace": payload["workspace"],
            "obstacles": payload["obstacles"],
            "timestamp": time.time(),
            "model": "backend-simulation-harness",
            "llm_provider": "simulation",
        }
        self.raw_pub.publish(String(data=json.dumps(output)))


class OperatorHarness(Node):
    def __init__(self):
        super().__init__("interactive_sim_operator_harness")
        self.command_pub = self.create_publisher(String, "/llm_vision/operator_command", 10)
        self.approval_pub = self.create_publisher(String, "/llm_vision/mission_approval", 10)
        self.launch_approval_pub = self.create_publisher(String, "/llm_vision/launch_approval", 10)
        self.create_subscription(String, "/llm_vision/mission_proposal", self.proposal_callback, LATCHED_QOS)
        self.create_subscription(String, "/llm_vision/operator_response", self.response_callback, LATCHED_QOS)
        self.create_subscription(String, "/llm_vision/prompt", self.prompt_callback, LATCHED_QOS)
        self.create_subscription(
            String,
            "/llm_vision/plan_candidate_verified",
            self.candidate_callback,
            LATCHED_QOS,
        )
        self.create_subscription(String, "/llm_vision/plan_verified", self.final_callback, LATCHED_QOS)
        self.create_subscription(String, "/llm_vision/launch_proposal", self.launch_proposal_callback, LATCHED_QOS)
        self.command_sent = False
        self.approved_missions = set()
        self.responses = []
        self.prompts = []
        self.candidates = []
        self.launch_proposals = []
        self.launch_approved = False
        self.final = None

    def send_command(self):
        if self.command_sent:
            return
        self.command_sent = True
        self.command_pub.publish(String(data=json.dumps({"text": "Hover near the chair without touching it"})))

    def proposal_callback(self, msg):
        proposal = json.loads(msg.data)
        mission_id = proposal["mission_id"]
        if mission_id in self.approved_missions:
            return
        self.approved_missions.add(mission_id)
        approval = {
            "decision": "APPROVE",
            "mission_id": mission_id,
            "proposal_hash": proposal["proposal_hash"],
        }
        self.approval_pub.publish(String(data=json.dumps(approval)))

    def response_callback(self, msg):
        self.responses.append(json.loads(msg.data))

    def prompt_callback(self, msg):
        payload = json.loads(msg.data)
        if not self.prompts or self.prompts[-1]["plan_id"] != payload["plan_id"]:
            self.prompts.append(payload)

    def candidate_callback(self, msg):
        payload = json.loads(msg.data)
        if not self.candidates or self.candidates[-1]["plan_id"] != payload["plan_id"]:
            self.candidates.append(payload)

    def final_callback(self, msg):
        self.final = json.loads(msg.data)

    def launch_proposal_callback(self, msg):
        payload = json.loads(msg.data)
        if self.launch_proposals and self.launch_proposals[-1]["plan_id"] == payload["plan_id"]:
            return
        self.launch_proposals.append(payload)

    def approve_launch(self):
        if self.launch_approved or not self.launch_proposals:
            return
        self.launch_approved = True
        payload = self.launch_proposals[-1]
        self.launch_approval_pub.publish(
            String(
                data=json.dumps(
                    {
                        "decision": "APPROVE",
                        "mission_id": payload["mission_id"],
                        "plan_id": payload["plan_id"],
                        "proposal_hash": payload["proposal_hash"],
                    }
                )
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-intent", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    intent_provider = "mock" if args.mock_intent else "openai"

    rclpy.init(
        args=[
            "interactive_backend_sim",
            "--ros-args",
            "-p",
            "environment:=sim",
            "-p",
            f"intent_provider:={intent_provider}",
            "-p",
            "visualizer:=contraction",
            "-p",
            "sample_id:=4",
            "-p",
            "publish_mission_state:=true",
            "-p",
            "fresh_data_timeout_s:=5.0",
            "-p",
            "verified_plan_topic:=/llm_vision/plan_candidate_verified",
            "-p",
            "debug:=false",
            "-p",
            "show_window:=false",
            "-p",
            "output_png:=/tmp/llm_vision_contraction_backend_sim.png",
        ]
    )
    contraction_visualizer = ContractionVisualizer()
    nodes = [
        DatasetScenePublisher(),
        InteractiveMissionGateway(),
        PlannerHarness(),
        PathRefinement(),
        PathVerifier(),
        contraction_visualizer,
        OperatorHarness(),
    ]
    operator = nodes[-1]
    executor = SingleThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.timeout and operator.final is None:
            executor.spin_once(timeout_sec=0.02)
            if not operator.command_sent and time.monotonic() - started >= 1.0:
                operator.send_command()
            if operator.launch_proposals and contraction_visualizer.plan is not None:
                operator.approve_launch()
        if operator.final is None:
            raise AssertionError(f"simulation timed out; responses={operator.responses[-5:]}")
        if len(operator.prompts) != 2:
            raise AssertionError(f"expected two planning attempts, got {len(operator.prompts)}")
        if len(operator.candidates) != 2:
            raise AssertionError(f"expected two verifier results, got {len(operator.candidates)}")
        if operator.candidates[0]["passed"] is not False:
            raise AssertionError("first injected candidate should fail verification")
        if operator.candidates[1]["passed"] is not True:
            raise AssertionError("refined retry should pass verification")
        if "Previous plan failed verification" not in operator.prompts[1]["prompt"]:
            raise AssertionError("verification feedback was not added to retry prompt")
        if operator.final["plan_id"] != operator.prompts[1]["plan_id"]:
            raise AssertionError("wrong planning attempt was released")
        if len(operator.launch_proposals) != 1:
            raise AssertionError(f"expected one final launch approval, got {len(operator.launch_proposals)}")
        if not contraction_visualizer.reference_xy or not contraction_visualizer.launched:
            raise AssertionError("conformal safety tubes were not latched before final launch")
        print(
            json.dumps(
                {
                    "result": "passed",
                    "intent_provider": intent_provider,
                    "mission_id": operator.final["mission_id"],
                    "attempts": len(operator.prompts),
                    "failed_constraints_attempt_1": operator.candidates[0]["failed_constraints"],
                    "final_plan_id": operator.final["plan_id"],
                    "final_launch_approved": True,
                    "final_metrics": operator.final["metrics"],
                },
                indent=2,
            )
        )
    finally:
        for node in nodes:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
