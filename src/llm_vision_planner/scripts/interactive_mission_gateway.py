#!/usr/bin/env python3
"""Human-approved mission intent gateway for the existing planner pipeline."""

import copy
import hashlib
import json
import math
import os
import queue
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import rclpy
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


INSTRUCTIONS = (
    "You are an expert UAV pilot/planner. Generate sparse collision-free 2D routing "
    "waypoints for a quadrotor in NED frame assuming it is airborne at the provided "
    "start location. A separate module will interpolate them and generate dynamically "
    "feasible trajectories, so output only high-level waypoints."
)
INTENT_SYSTEM_PROMPT = """You are a UAV mission intent parser.
Return only the supplied MissionIntent structure.
Do not generate coordinates, waypoints, or a route.
Never invent an object ID; copy it exactly from the detected-object catalog.
Only HOVER_NEAR and GO_TO a uniquely identified detected object are supported.
GO_TO means approach the object with a safe standoff, never collide with it.
If the target is missing or ambiguous, return NEEDS_CLARIFICATION.
HOLD, cancellation, and unsupported requests must not return READY.
If any safety-relevant meaning is uncertain, return NEEDS_CLARIFICATION.
All detected objects remain mandatory avoidance obstacles.
"""

ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class MissionIntent(BaseModel):
    """Schema-constrained result returned by the intent model."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["READY", "NEEDS_CLARIFICATION", "CANCELLED", "UNSUPPORTED"]
    action: Literal["HOVER_NEAR", "GO_TO", "HOLD", "NONE"]
    target_object_id: str
    target_label: str
    relation: Literal["NEAR", "NONE"]
    requested_standoff_m: float = Field(ge=0.0, le=2.0)
    clarifying_question: str


class OpenAIIntentParser:
    def __init__(self, model_name):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_name = model_name
        self.client = OpenAI()

    def parse(self, operator_text, object_catalog, conversation):
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "detected_objects": object_catalog,
                            "conversation": conversation[-6:],
                            "operator_command": operator_text,
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            text_format=MissionIntent,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no parsed MissionIntent")
        return response.output_parsed


class MockIntentParser:
    """Offline parser used only by repeatable simulation and unit tests."""

    def parse(self, operator_text, object_catalog, conversation):
        del conversation
        lowered = operator_text.lower()
        if "cancel" in lowered:
            return MissionIntent(
                status="CANCELLED",
                action="NONE",
                target_object_id="",
                target_label="",
                relation="NONE",
                requested_standoff_m=0.0,
                clarifying_question="",
            )
        if "hold" in lowered and not any(item["label"].lower() in lowered for item in object_catalog):
            return MissionIntent(
                status="UNSUPPORTED",
                action="HOLD",
                target_object_id="",
                target_label="",
                relation="NONE",
                requested_standoff_m=0.0,
                clarifying_question="The vehicle is already holding; specify a detected object to approach.",
            )

        matches = [item for item in object_catalog if item["label"].lower() in lowered]
        if len(matches) != 1:
            labels = ", ".join(item["label"] for item in object_catalog) or "none"
            return MissionIntent(
                status="NEEDS_CLARIFICATION",
                action="NONE",
                target_object_id="",
                target_label="",
                relation="NONE",
                requested_standoff_m=0.0,
                clarifying_question=f"Which detected object do you mean? I see: {labels}.",
            )

        distance_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b", lowered)
        standoff = float(distance_match.group(1)) if distance_match else 0.0
        target = matches[0]
        return MissionIntent(
            status="READY",
            action="HOVER_NEAR" if "hover" in lowered else "GO_TO",
            target_object_id=target["object_id"],
            target_label=target["label"],
            relation="NEAR",
            requested_standoff_m=standoff,
            clarifying_question="",
        )


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_id_for(obstacle, index):
    value = obstacle.get("object_id", obstacle.get("id", index))
    value = str(value)
    return value if value.startswith("obj-") else f"obj-{value}"


def normalize_obstacles(payload):
    normalized = []
    used_ids = set()
    for index, raw in enumerate(payload.get("obstacles", []), start=1):
        obstacle = copy.deepcopy(raw)
        minimum = obstacle.get("min_corner")
        maximum = obstacle.get("max_corner")
        if not isinstance(minimum, list) or not isinstance(maximum, list) or len(minimum) < 2 or len(maximum) < 2:
            raise ValueError("every obstacle requires min_corner and max_corner")
        object_id = object_id_for(obstacle, index)
        if object_id in used_ids:
            raise ValueError(f"duplicate obstacle object_id: {object_id}")
        used_ids.add(object_id)
        obstacle["object_id"] = object_id
        obstacle["label"] = str(obstacle.get("label") or obstacle.get("shape") or "unknown")
        obstacle["min_corner"] = [float(value) for value in minimum]
        obstacle["max_corner"] = [float(value) for value in maximum]
        normalized.append(obstacle)
    return normalized


def scene_signature(obstacles):
    geometry = [
        {
            "object_id": item["object_id"],
            "label": item["label"],
            "min_corner": [round(float(value), 3) for value in item["min_corner"]],
            "max_corner": [round(float(value), 3) for value in item["max_corner"]],
        }
        for item in obstacles
    ]
    geometry.sort(key=lambda item: item["object_id"])
    return hashlib.sha256(canonical_json(geometry).encode("utf-8")).hexdigest()


def obstacle_xy_geometry(obstacle):
    minimum = obstacle["min_corner"]
    maximum = obstacle["max_corner"]
    center = (
        0.5 * (float(minimum[0]) + float(maximum[0])),
        0.5 * (float(minimum[1]) + float(maximum[1])),
    )
    size = (
        float(maximum[0]) - float(minimum[0]),
        float(maximum[1]) - float(minimum[1]),
    )
    return center, size


def scenes_compatible(reference, current, position_tolerance, size_tolerance):
    """Match by label and geometry so sensor jitter and reordered detections are safe."""
    if len(reference) != len(current):
        return False, f"obstacle count changed from {len(reference)} to {len(current)}"
    unmatched = set(range(len(current)))
    for expected in reference:
        expected_center, expected_size = obstacle_xy_geometry(expected)
        candidates = []
        for index in unmatched:
            observed = current[index]
            if observed["label"].strip().lower() != expected["label"].strip().lower():
                continue
            observed_center, observed_size = obstacle_xy_geometry(observed)
            center_delta = math.dist(expected_center, observed_center)
            size_delta = max(abs(a - b) for a, b in zip(expected_size, observed_size))
            candidates.append((center_delta, size_delta, index))
        if not candidates:
            return False, f"missing obstacle label '{expected['label']}'"
        center_delta, size_delta, index = min(candidates)
        if center_delta > float(position_tolerance):
            return False, (
                f"{expected['label']} moved {center_delta:.3f} m "
                f"(allowed {float(position_tolerance):.3f} m)"
            )
        if size_delta > float(size_tolerance):
            return False, (
                f"{expected['label']} size changed {size_delta:.3f} m "
                f"(allowed {float(size_tolerance):.3f} m)"
            )
        unmatched.remove(index)
    return True, ""


def inflate_obstacles_xy(obstacles, padding):
    """Create a conservative planning envelope covering every accepted scene."""
    inflated = copy.deepcopy(obstacles)
    for obstacle in inflated:
        minimum = obstacle["min_corner"]
        maximum = obstacle["max_corner"]
        minimum[0] = float(minimum[0]) - float(padding)
        minimum[1] = float(minimum[1]) - float(padding)
        maximum[0] = float(maximum[0]) + float(padding)
        maximum[1] = float(maximum[1]) + float(padding)
        if isinstance(obstacle.get("size"), list) and len(obstacle["size"]) >= 2:
            obstacle["size"][0] = maximum[0] - minimum[0]
            obstacle["size"][1] = maximum[1] - minimum[1]
    return inflated


def clearance_to_box(point, obstacle):
    minimum = obstacle["min_corner"]
    maximum = obstacle["max_corner"]
    x = float(point["x"])
    y = float(point["y"])
    dx = max(float(minimum[0]) - x, 0.0, x - float(maximum[0]))
    dy = max(float(minimum[1]) - y, 0.0, y - float(maximum[1]))
    if dx == 0.0 and dy == 0.0:
        return -min(
            abs(x - float(minimum[0])),
            abs(float(maximum[0]) - x),
            abs(y - float(minimum[1])),
            abs(float(maximum[1]) - y),
        )
    return math.hypot(dx, dy)


def safe_standoff_goal(start, target, obstacles, workspace, fixed_z, requested_standoff, default_standoff, clearance):
    minimum = target["min_corner"]
    maximum = target["max_corner"]
    center_x = 0.5 * (float(minimum[0]) + float(maximum[0]))
    center_y = 0.5 * (float(minimum[1]) + float(maximum[1]))
    offset = max(float(requested_standoff), float(default_standoff), float(clearance)) + 0.02
    candidates = [
        {"x": float(minimum[0]) - offset, "y": center_y, "z": fixed_z},
        {"x": float(maximum[0]) + offset, "y": center_y, "z": fixed_z},
        {"x": center_x, "y": float(minimum[1]) - offset, "z": fixed_z},
        {"x": center_x, "y": float(maximum[1]) + offset, "z": fixed_z},
    ]
    x_limits = workspace["x"]
    y_limits = workspace["y"]
    feasible = []
    for point in candidates:
        if not (
            float(x_limits[0]) <= point["x"] <= float(x_limits[1])
            and float(y_limits[0]) <= point["y"] <= float(y_limits[1])
        ):
            continue
        if all(clearance_to_box(point, obstacle) >= float(clearance) for obstacle in obstacles):
            feasible.append({key: round(float(value), 3) for key, value in point.items()})
    if not feasible:
        raise ValueError("no collision-free standoff goal is available in the workspace")
    return min(
        feasible,
        key=lambda point: math.hypot(point["x"] - float(start["x"]), point["y"] - float(start["y"])),
    )


def obstacle_catalog(obstacles):
    return [
        {
            "object_id": item["object_id"],
            "label": item["label"],
            "min_corner": item["min_corner"],
            "max_corner": item["max_corner"],
            "confidence": item.get("confidence"),
        }
        for item in obstacles
    ]


def describe_obstacles(obstacles):
    descriptions = []
    for index, obstacle in enumerate(obstacles, start=1):
        minimum = obstacle["min_corner"]
        maximum = obstacle["max_corner"]
        descriptions.append(
            f"{index} {obstacle['label']} ({obstacle['object_id']}): "
            f"x=[{minimum[0]:.2f},{maximum[0]:.2f}], y=[{minimum[1]:.2f},{maximum[1]:.2f}."
        )
    return " ".join(descriptions) if descriptions else "No obstacles currently detected."


def build_planner_prompt(start, goal, workspace, obstacles, clearance):
    distance = math.hypot(float(goal["x"]) - float(start["x"]), float(goal["y"]) - float(start["y"]))
    nl_env = (
        "Mission state: the UAV has already taken off and is holding hover at the start position. "
        "Use this hover position as the first waypoint/reference for planning. "
        f"Workspace: x=[{workspace['x'][0]:.2f},{workspace['x'][1]:.2f}]m, "
        f"y=[{workspace['y'][0]:.2f},{workspace['y'][1]:.2f}]m, z={workspace['z']:.2f} fixed. "
        f"Start: ({start['x']:.2f},{start['y']:.2f},{start['z']:.2f}), "
        f"Goal: ({goal['x']:.2f},{goal['y']:.2f},{goal['z']:.2f}), "
        f"distance≈{distance:.2f}m. Obstacles with x-y spans: {describe_obstacles(obstacles)}"
    )
    constraints = (
        "Constraints:\n"
        f"- all waypoints in NED frame, z must stay {workspace['z']:.2f}\n"
        "- first waypoint must exactly match the provided hover start coordinates\n"
        "- final waypoint must be goal coordinates\n"
        f"- maintain >={clearance:.2f}m clearance from obstacle x-y boxes\n"
        "- stay within workspace\n"
        "- no waypoint should be within obstacle boxes, walls, or near corners\n"
        "- prefer sparse, smooth, monotonic progress through open space\n"
        "- return only the structured output requested by the response model"
    )
    return "\n".join((INSTRUCTIONS, nl_env, constraints)), nl_env


class InteractiveMissionGateway(Node):
    def __init__(self, intent_parser=None):
        super().__init__("interactive_mission_gateway")
        self.declare_parameter("environment", "real")
        self.declare_parameter("intent_provider", "openai")
        self.declare_parameter("openai_intent_model", "gpt-5.4-nano")
        self.declare_parameter("planner_llm_provider", "llama")
        self.declare_parameter("planner_model_name", "rrt_planner")
        self.declare_parameter("operator_command_topic", "/llm_vision/operator_command")
        self.declare_parameter("approval_topic", "/llm_vision/mission_approval")
        self.declare_parameter("operator_response_topic", "/llm_vision/operator_response")
        self.declare_parameter("mission_proposal_topic", "/llm_vision/mission_proposal")
        self.declare_parameter("semantic_obstacle_topic", "/llm_vision/semantic_obstacles")
        self.declare_parameter("sim_obstacle_topic", "/llm_vision/sim_obstacles")
        self.declare_parameter("mission_state_topic", "/llm_vision/mission_state")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("prompt_topic", "/llm_vision/prompt")
        self.declare_parameter("candidate_verification_topic", "/llm_vision/plan_candidate_verified")
        self.declare_parameter("passed_plan_topic", "/llm_vision/plan_verified")
        self.declare_parameter("required_mission_state", "HOLDING_FOR_PLAN")
        self.declare_parameter("require_mission_state", True)
        self.declare_parameter("workspace_x_min", 0.0)
        self.declare_parameter("workspace_x_max", 4.0)
        self.declare_parameter("workspace_y_min", 0.0)
        self.declare_parameter("workspace_y_max", 4.0)
        self.declare_parameter("fixed_z", -0.5)
        self.declare_parameter("clearance_m", 0.40)
        self.declare_parameter("default_standoff_m", 0.60)
        self.declare_parameter("fresh_data_timeout_s", 2.0)
        self.declare_parameter("approval_timeout_s", 30.0)
        self.declare_parameter("scene_position_tolerance_m", 0.15)
        self.declare_parameter("scene_size_tolerance_m", 0.20)
        self.declare_parameter("max_start_drift_m", 0.25)
        self.declare_parameter("max_release_start_drift_m", 0.08)
        self.declare_parameter("max_planning_attempts", 3)
        self.declare_parameter("debug", True)

        self.environment = str(self.get_parameter("environment").value).strip().lower()
        self.fixed_z = float(self.get_parameter("fixed_z").value)
        self.clearance_m = float(self.get_parameter("clearance_m").value)
        self.default_standoff_m = float(self.get_parameter("default_standoff_m").value)
        self.scene_position_tolerance_m = float(self.get_parameter("scene_position_tolerance_m").value)
        self.scene_size_tolerance_m = float(self.get_parameter("scene_size_tolerance_m").value)
        if self.scene_position_tolerance_m < 0.0 or self.scene_size_tolerance_m < 0.0:
            raise ValueError("scene tolerances must be non-negative")
        self.scene_guard_band_m = self.scene_position_tolerance_m + 0.5 * self.scene_size_tolerance_m
        self.workspace = {
            "x": [
                float(self.get_parameter("workspace_x_min").value),
                float(self.get_parameter("workspace_x_max").value),
            ],
            "y": [
                float(self.get_parameter("workspace_y_min").value),
                float(self.get_parameter("workspace_y_max").value),
            ],
            "z": self.fixed_z,
        }
        self.state = "WAITING_FOR_COMMAND"
        self.latest_scene = None
        self.latest_scene_received_s = None
        self.latest_mission_state = None
        self.latest_mission_state_received_s = None
        self.current_pose = None
        self.conversation = []
        self.active_mission = None
        self.active_plan_id = None
        self.intent_request_token = None
        self.intent_results = queue.Queue()
        self.intent_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intent_parser")
        self.intent_parser = intent_parser or self.create_intent_parser()

        obstacle_topic = (
            str(self.get_parameter("sim_obstacle_topic").value)
            if self.environment == "sim"
            else str(self.get_parameter("semantic_obstacle_topic").value)
        )
        self.create_subscription(String, obstacle_topic, self.obstacle_callback, 10)
        self.create_subscription(
            String,
            str(self.get_parameter("mission_state_topic").value),
            self.mission_state_callback,
            10,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter("pose_topic").value),
            self.pose_callback,
            ODOM_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("operator_command_topic").value),
            self.command_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("approval_topic").value),
            self.approval_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("candidate_verification_topic").value),
            self.verification_callback,
            LATCHED_QOS,
        )
        self.response_pub = self.create_publisher(
            String,
            str(self.get_parameter("operator_response_topic").value),
            LATCHED_QOS,
        )
        self.proposal_pub = self.create_publisher(
            String,
            str(self.get_parameter("mission_proposal_topic").value),
            LATCHED_QOS,
        )
        self.prompt_pub = self.create_publisher(
            String,
            str(self.get_parameter("prompt_topic").value),
            LATCHED_QOS,
        )
        self.passed_plan_pub = self.create_publisher(
            String,
            str(self.get_parameter("passed_plan_topic").value),
            LATCHED_QOS,
        )
        self.create_timer(0.05, self.drain_intent_results)
        self.get_logger().info(
            f"interactive gateway ready: intent_provider={self.get_parameter('intent_provider').value}, "
            f"obstacles={obstacle_topic}"
        )

    def create_intent_parser(self):
        provider = str(self.get_parameter("intent_provider").value).strip().lower()
        if provider == "mock":
            return MockIntentParser()
        if provider != "openai":
            raise ValueError(f"unsupported intent_provider: {provider}")
        return OpenAIIntentParser(str(self.get_parameter("openai_intent_model").value))

    def obstacle_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            payload["obstacles"] = normalize_obstacles(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.get_logger().error(f"invalid obstacle snapshot: {exc}")
            return
        self.latest_scene = payload
        self.latest_scene_received_s = time.time()

    def mission_state_callback(self, msg):
        try:
            self.latest_mission_state = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid mission state: {exc}")
            return
        self.latest_mission_state_received_s = time.time()

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        self.current_pose = {
            "x": float(msg.position[0]),
            "y": float(msg.position[1]),
            "z": float(msg.position[2]),
            "received_s": time.time(),
        }

    def command_callback(self, msg):
        operator_text = self.command_text(msg.data)
        if not operator_text:
            self.publish_response("ERROR", "Operator command is empty.")
            return
        if operator_text.strip().lower() == "cancel":
            self.cancel_active_mission("Mission cancelled by operator.")
            return
        if self.state in ("PARSING_INTENT", "WAITING_FOR_VERIFICATION", "RELEASED_TO_EXECUTOR"):
            self.publish_response("BUSY", f"Cannot accept a new mission while state={self.state}.")
            return
        error = self.context_error()
        if error:
            self.publish_response("NOT_READY", error)
            return

        scene = copy.deepcopy(self.latest_scene)
        start = self.current_start()
        token = uuid.uuid4().hex
        self.intent_request_token = token
        self.state = "PARSING_INTENT"
        self.conversation.append({"role": "operator", "content": operator_text})
        catalog = obstacle_catalog(scene["obstacles"])
        future = self.intent_executor.submit(
            self.intent_parser.parse,
            operator_text,
            catalog,
            list(self.conversation),
        )
        future.add_done_callback(
            lambda completed: self.intent_results.put((token, completed, scene, start, operator_text))
        )
        self.publish_response("PARSING_INTENT", "Interpreting the command against the current object snapshot.")

    @staticmethod
    def command_text(raw):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()
        if isinstance(payload, dict):
            return str(payload.get("text", "")).strip()
        return str(payload).strip()

    def drain_intent_results(self):
        while True:
            try:
                token, future, scene, start, operator_text = self.intent_results.get_nowait()
            except queue.Empty:
                return
            if token != self.intent_request_token or self.state != "PARSING_INTENT":
                continue
            try:
                intent = future.result()
            except Exception as exc:
                self.state = "WAITING_FOR_COMMAND"
                self.publish_response("INTENT_ERROR", f"Intent service failed; no mission was released: {exc}")
                continue
            self.handle_intent(intent, scene, start, operator_text)

    def handle_intent(self, intent, scene, start, operator_text):
        if intent.status == "CANCELLED":
            self.cancel_active_mission("Mission cancelled by parsed operator intent.")
            return
        if intent.status != "READY":
            question = intent.clarifying_question or "Please clarify the requested detected object."
            self.conversation.append({"role": "system", "content": question})
            self.state = "WAITING_FOR_COMMAND"
            self.publish_response(intent.status, question)
            return
        if intent.action not in ("HOVER_NEAR", "GO_TO") or intent.relation != "NEAR":
            self.state = "WAITING_FOR_COMMAND"
            self.publish_response("UNSUPPORTED", "Only a safe standoff near a detected object is currently supported.")
            return

        by_id = {item["object_id"]: item for item in scene["obstacles"]}
        target = by_id.get(intent.target_object_id)
        if target is None:
            self.state = "WAITING_FOR_COMMAND"
            self.publish_response("NEEDS_CLARIFICATION", "The selected object ID is not in the current snapshot.")
            return
        if target["label"].strip().lower() != intent.target_label.strip().lower():
            self.state = "WAITING_FOR_COMMAND"
            self.publish_response("NEEDS_CLARIFICATION", "The selected object ID does not match the requested label.")
            return
        same_label = [
            item for item in scene["obstacles"] if item["label"].strip().lower() == target["label"].strip().lower()
        ]
        if len(same_label) > 1:
            self.state = "WAITING_FOR_COMMAND"
            self.publish_response(
                "NEEDS_CLARIFICATION",
                f"I see {len(same_label)} objects labelled {target['label']}; specify which object you mean.",
            )
            return
        try:
            planning_obstacles = inflate_obstacles_xy(scene["obstacles"], self.scene_guard_band_m)
            planning_target = next(
                item for item in planning_obstacles if item["object_id"] == target["object_id"]
            )
            goal = safe_standoff_goal(
                start,
                planning_target,
                planning_obstacles,
                self.workspace,
                self.fixed_z,
                intent.requested_standoff_m,
                self.default_standoff_m,
                self.clearance_m,
            )
        except ValueError as exc:
            self.state = "WAITING_FOR_COMMAND"
            self.publish_response("NO_SAFE_GOAL", str(exc))
            return

        created_at = time.time()
        mission_id = f"M-{int(created_at * 1000)}-{uuid.uuid4().hex[:6]}"
        snapshot_hash = scene_signature(scene["obstacles"])
        mission = {
            "mission_id": mission_id,
            "snapshot_id": f"S-{snapshot_hash[:12]}",
            "scene_signature": snapshot_hash,
            "operator_text": operator_text,
            "intent": intent.model_dump(),
            "target": target,
            "start": start,
            "goal": goal,
            "workspace": copy.deepcopy(self.workspace),
            "observed_obstacles": copy.deepcopy(scene["obstacles"]),
            "obstacles": planning_obstacles,
            "created_at": created_at,
            "attempt": 0,
        }
        # The live hover position is rebound at approval within a configured limit.
        contract = {
            key: mission[key]
            for key in ("mission_id", "snapshot_id", "intent", "target", "goal", "workspace", "obstacles")
        }
        mission["proposal_hash"] = hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()
        mission["base_prompt"], mission["nl_env"] = build_planner_prompt(
            start,
            goal,
            mission["workspace"],
            mission["obstacles"],
            self.clearance_m,
        )
        self.active_mission = mission
        self.active_plan_id = None
        self.state = "AWAITING_APPROVAL"
        proposal = self.proposal_payload(mission)
        self.proposal_pub.publish(String(data=json.dumps(proposal)))
        self.publish_response(
            "AWAITING_APPROVAL",
            (
                f"I found one {target['label']} ({target['object_id']}) and propose goal "
                f"({goal['x']:.2f}, {goal['y']:.2f}, {goal['z']:.2f}). All detected objects remain obstacles. "
                f"Approve mission {mission_id}?"
            ),
            mission_id=mission_id,
        )

    def proposal_payload(self, mission):
        target = mission["target"]
        minimum = target["min_corner"]
        maximum = target["max_corner"]
        return {
            "type": "MISSION_PROPOSAL",
            "status": "AWAITING_APPROVAL",
            "mission_id": mission["mission_id"],
            "snapshot_id": mission["snapshot_id"],
            "proposal_hash": mission["proposal_hash"],
            "operator_text": mission["operator_text"],
            "target": {
                "object_id": target["object_id"],
                "label": target["label"],
                "center": [
                    round(0.5 * (float(minimum[0]) + float(maximum[0])), 3),
                    round(0.5 * (float(minimum[1]) + float(maximum[1])), 3),
                    self.fixed_z,
                ],
            },
            "goal": mission["goal"],
            "clearance_m": self.clearance_m,
            "scene_guard_band_m": round(self.scene_guard_band_m, 3),
            "obstacle_ids": [item["object_id"] for item in mission["obstacles"]],
            "created_at": mission["created_at"],
        }

    def approval_callback(self, msg):
        try:
            approval = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.publish_response("APPROVAL_REJECTED", f"Approval must be structured JSON: {exc}")
            return
        if self.state != "AWAITING_APPROVAL" or self.active_mission is None:
            self.publish_response("APPROVAL_REJECTED", "There is no mission awaiting approval.")
            return
        decision = str(approval.get("decision", approval.get("type", ""))).upper()
        if decision in ("REJECT", "REJECTED", "CANCEL"):
            self.cancel_active_mission("Mission proposal rejected by operator.")
            return
        if decision not in ("APPROVE", "APPROVED"):
            self.publish_response("APPROVAL_REJECTED", "Decision must be APPROVE or REJECT.")
            return
        mission = self.active_mission
        if approval.get("mission_id") != mission["mission_id"]:
            self.publish_response("APPROVAL_REJECTED", "Approval mission_id does not match the active proposal.")
            return
        if approval.get("proposal_hash") != mission["proposal_hash"]:
            self.publish_response("APPROVAL_REJECTED", "Approval proposal_hash does not match the active proposal.")
            return
        if time.time() - mission["created_at"] > float(self.get_parameter("approval_timeout_s").value):
            self.cancel_active_mission("Mission proposal expired; request a new sensor-grounded proposal.")
            return
        error = self.context_error()
        if error:
            self.cancel_active_mission(f"Approval invalidated: {error}")
            return
        scene_error = self.scene_compatibility_error(mission)
        if scene_error:
            self.cancel_active_mission(f"Scene changed before approval: {scene_error}.")
            return
        current_position = self.current_position()
        drift = math.dist(
            [current_position["x"], current_position["y"], current_position["z"]],
            [mission["start"]["x"], mission["start"]["y"], mission["start"]["z"]],
        )
        if drift > float(self.get_parameter("max_start_drift_m").value):
            self.cancel_active_mission(f"Start moved {drift:.2f} m before approval; request a new proposal.")
            return
        mission["start"] = {
            "x": round(current_position["x"], 3),
            "y": round(current_position["y"], 3),
            "z": self.fixed_z,
        }
        if not self.point_in_workspace(mission["start"]):
            self.cancel_active_mission("Current hover position is outside the planning workspace.")
            return
        if any(clearance_to_box(mission["start"], obstacle) < self.clearance_m for obstacle in mission["obstacles"]):
            self.cancel_active_mission("Current hover position does not satisfy obstacle clearance.")
            return
        mission["base_prompt"], mission["nl_env"] = build_planner_prompt(
            mission["start"],
            mission["goal"],
            mission["workspace"],
            mission["obstacles"],
            self.clearance_m,
        )
        mission["approved_at"] = time.time()
        self.publish_planning_attempt()

    def publish_planning_attempt(self, feedback=None):
        mission = self.active_mission
        mission["attempt"] += 1
        plan_id = f"{mission['mission_id']}-A{mission['attempt']:02d}"
        prompt = mission["base_prompt"]
        if feedback:
            failed = feedback.get("failed_constraints", [])
            table = feedback.get("verification_feedback_table") or feedback.get("metrics", {}).get("feedback_table", "")
            prompt += (
                "\nPrevious plan failed verification. Regenerate a sparse waypoint plan for the exact same "
                "approved start, goal, workspace, and obstacle snapshot.\n"
                f"Failed constraints: {', '.join(failed) if failed else 'unknown'}\n{table}\n"
                "Prefer larger obstacle clearance, monotonic goal progress, and smoother segment changes."
            )
        envelope = {
            "mission_id": mission["mission_id"],
            "snapshot_id": mission["snapshot_id"],
            "proposal_hash": mission["proposal_hash"],
            "plan_id": plan_id,
            "attempt": mission["attempt"],
            "prompt": prompt,
            "nl_env": mission["nl_env"],
            "start": mission["start"],
            "goal": mission["goal"],
            "workspace": mission["workspace"],
            "obstacles": mission["obstacles"],
            "timestamp": time.time(),
            "llm_provider": str(self.get_parameter("planner_llm_provider").value),
            "requested_model": str(self.get_parameter("planner_model_name").value),
        }
        self.active_plan_id = plan_id
        self.state = "WAITING_FOR_VERIFICATION"
        self.prompt_pub.publish(String(data=json.dumps(envelope)))
        self.publish_response(
            "PLANNING",
            f"Approved mission {mission['mission_id']}; planning attempt {mission['attempt']} is running.",
            mission_id=mission["mission_id"],
            plan_id=plan_id,
        )

    def verification_callback(self, msg):
        try:
            candidate = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid candidate verification result: {exc}")
            return
        if self.state != "WAITING_FOR_VERIFICATION" or self.active_mission is None:
            return
        if str(candidate.get("plan_id")) != str(self.active_plan_id):
            self.log_debug(f"ignoring stale candidate plan_id={candidate.get('plan_id')}")
            return
        context_error = self.context_error()
        if context_error:
            self.cancel_active_mission(f"Planning context became unsafe: {context_error}")
            return
        scene_error = self.scene_compatibility_error(self.active_mission)
        if scene_error:
            self.cancel_active_mission(f"Scene changed during planning: {scene_error}.")
            return
        if not self.candidate_contract_matches(candidate):
            self.cancel_active_mission("Candidate plan contract differs from the approved mission.")
            return
        if bool(candidate.get("passed", False)):
            current_position = self.current_position()
            release_drift = math.dist(
                [current_position["x"], current_position["y"], current_position["z"]],
                [
                    self.active_mission["start"]["x"],
                    self.active_mission["start"]["y"],
                    self.active_mission["start"]["z"],
                ],
            )
            release_limit = float(self.get_parameter("max_release_start_drift_m").value)
            if release_drift > release_limit:
                self.cancel_active_mission(
                    f"Hover moved {release_drift:.2f} m during planning; request a new proposal."
                )
                return
            output = dict(candidate)
            output.update(
                {
                    "mission_id": self.active_mission["mission_id"],
                    "snapshot_id": self.active_mission["snapshot_id"],
                    "proposal_hash": self.active_mission["proposal_hash"],
                    "approved_at": self.active_mission["approved_at"],
                }
            )
            self.passed_plan_pub.publish(String(data=json.dumps(output)))
            self.state = "RELEASED_TO_EXECUTOR"
            self.publish_response(
                "VERIFIED",
                f"Plan {self.active_plan_id} passed verification and was released to the executor.",
                mission_id=self.active_mission["mission_id"],
                plan_id=self.active_plan_id,
            )
            return
        if self.active_mission["attempt"] >= int(self.get_parameter("max_planning_attempts").value):
            self.state = "PLANNING_FAILED"
            self.publish_response(
                "PLANNING_FAILED",
                "No planning attempt passed verification; the vehicle remains holding.",
                mission_id=self.active_mission["mission_id"],
                plan_id=self.active_plan_id,
            )
            return
        self.publish_planning_attempt(feedback=candidate)

    def candidate_contract_matches(self, candidate):
        mission = self.active_mission
        for key in ("start", "goal", "workspace", "obstacles"):
            if canonical_json(candidate.get(key)) != canonical_json(mission[key]):
                return False
        return True

    def scene_compatibility_error(self, mission):
        compatible, reason = scenes_compatible(
            mission["observed_obstacles"],
            self.latest_scene["obstacles"],
            self.scene_position_tolerance_m,
            self.scene_size_tolerance_m,
        )
        return None if compatible else reason

    def point_in_workspace(self, point):
        return (
            self.workspace["x"][0] <= float(point["x"]) <= self.workspace["x"][1]
            and self.workspace["y"][0] <= float(point["y"]) <= self.workspace["y"][1]
        )

    def context_error(self):
        now = time.time()
        timeout = float(self.get_parameter("fresh_data_timeout_s").value)
        if self.latest_scene is None or self.latest_scene_received_s is None:
            return "No obstacle snapshot is available."
        if now - self.latest_scene_received_s > timeout:
            return "The obstacle snapshot is stale."
        if self.latest_scene.get("healthy") is False:
            return "Perception reports an unhealthy obstacle snapshot."
        if bool(self.get_parameter("require_mission_state").value):
            if self.latest_mission_state is None or self.latest_mission_state_received_s is None:
                return "No mission-state update is available."
            if now - self.latest_mission_state_received_s > timeout:
                return "The mission-state update is stale."
            required = str(self.get_parameter("required_mission_state").value)
            if self.latest_mission_state.get("state") != required:
                return f"Vehicle must be in {required} before accepting a mission."
        elif self.current_pose is None or now - self.current_pose["received_s"] > timeout:
            return "No fresh NED pose is available."
        return None

    def current_start(self):
        position = self.current_position()
        return {
            "x": round(float(position["x"]), 3),
            "y": round(float(position["y"]), 3),
            "z": self.fixed_z,
        }

    def current_position(self):
        if bool(self.get_parameter("require_mission_state").value):
            position = self.latest_mission_state.get("position", {})
            return {key: float(position[key]) for key in ("x", "y", "z")}
        return {key: float(self.current_pose[key]) for key in ("x", "y", "z")}

    def cancel_active_mission(self, reason):
        if self.state == "RELEASED_TO_EXECUTOR":
            self.publish_response("CANCEL_UNAVAILABLE", "The plan has already been released to the executor.")
            return
        self.intent_request_token = None
        self.active_mission = None
        self.active_plan_id = None
        self.state = "WAITING_FOR_COMMAND"
        self.publish_response("CANCELLED", reason)

    def publish_response(self, status, message, **metadata):
        payload = {
            "type": "OPERATOR_RESPONSE",
            "status": status,
            "message": message,
            "gateway_state": self.state,
            "timestamp": time.time(),
        }
        payload.update(metadata)
        self.response_pub.publish(String(data=json.dumps(payload)))
        self.log_debug(f"{status}: {message}")

    def log_debug(self, message):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(message)

    def destroy_node(self):
        self.intent_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main():
    rclpy.init()
    node = InteractiveMissionGateway()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
