#!/usr/bin/env python3
import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

DEFAULT_WORKSPACE_X = (0.0, 4.0)
DEFAULT_WORKSPACE_Y = (0.0, 4.0)
DEFAULT_FIXED_Z = -0.2
DEFAULT_GOAL = (3.0, 1.0, DEFAULT_FIXED_Z)
DEFAULT_CLEARANCE_M = 0.30
DEFAULT_SEMANTIC_OBSTACLE_TOPIC = "/llm_vision/semantic_obstacles"
DEFAULT_NORMAL_OBSTACLE_TOPIC = "/llm_vision/obstacles"
DEFAULT_PROMPT_TOPIC = "/llm_vision/prompt"
DEFAULT_VERIFIED_PLAN_TOPIC = "/llm_vision/plan_verified"
DEFAULT_MISSION_STATE_TOPIC = "/llm_vision/mission_state"
DEFAULT_POSE_TOPIC = "/qvio"
DEFAULT_FRESH_DATA_TIMEOUT_S = 2.0
QVIO_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

INSTRUCTIONS = (
    "You are an expert UAV pilot/planner. Generate sparse collision-free 2D routing "
    "waypoints for a quadrotor in NED frame assuming it is airborne at the provided "
    "start location. A separate module will interpolate them and generate dynamically "
    "feasible trajectories, so output only high-level waypoints."
)


class PromptGenerator(Node):
    def __init__(self):
        super().__init__("prompt_generator")
        self.declare_parameter("mode", "semantic")
        self.declare_parameter("single_shot", True)
        self.declare_parameter("initial_plan_id", 1)
        self.declare_parameter("workspace_x_min", DEFAULT_WORKSPACE_X[0])
        self.declare_parameter("workspace_x_max", DEFAULT_WORKSPACE_X[1])
        self.declare_parameter("workspace_y_min", DEFAULT_WORKSPACE_Y[0])
        self.declare_parameter("workspace_y_max", DEFAULT_WORKSPACE_Y[1])
        self.declare_parameter("fixed_z", DEFAULT_FIXED_Z)
        self.declare_parameter("goal_x", DEFAULT_GOAL[0])
        self.declare_parameter("goal_y", DEFAULT_GOAL[1])
        self.declare_parameter("goal_z", DEFAULT_GOAL[2])
        self.declare_parameter("clearance_m", DEFAULT_CLEARANCE_M)
        self.declare_parameter("semantic_obstacle_topic", DEFAULT_SEMANTIC_OBSTACLE_TOPIC)
        self.declare_parameter("normal_obstacle_topic", DEFAULT_NORMAL_OBSTACLE_TOPIC)
        self.declare_parameter("prompt_topic", DEFAULT_PROMPT_TOPIC)
        self.declare_parameter("verified_plan_topic", DEFAULT_VERIFIED_PLAN_TOPIC)
        self.declare_parameter("mission_state_topic", DEFAULT_MISSION_STATE_TOPIC)
        self.declare_parameter("required_mission_state", "HOLDING_FOR_PLAN")
        self.declare_parameter("require_mission_state", True)
        self.declare_parameter("snapshot_after_hover_s", 1.0)
        self.declare_parameter("start_drift_replan_m", 0.25)
        self.declare_parameter("feedback_enabled", True)
        self.declare_parameter("pose_topic", DEFAULT_POSE_TOPIC)
        self.declare_parameter("fresh_data_timeout_s", DEFAULT_FRESH_DATA_TIMEOUT_S)

        self.current_pose = None
        self.latest_mission_state = None
        self.latest_mission_state_stamp = None
        self.hover_started_s = None
        self.latest_obstacle_msg = None
        self.latest_obstacle_stamp = None
        self.last_printed_nl = None
        self.last_printed_prompt = None
        self.latched_context = None
        self.last_verification_feedback = None
        self.obstacle_topic = self.resolve_obstacle_topic()
        self.single_shot = bool(self.get_parameter("single_shot").value)
        self.next_plan_id = int(self.get_parameter("initial_plan_id").value)
        self.workspace_x = (
            float(self.get_parameter("workspace_x_min").value),
            float(self.get_parameter("workspace_x_max").value),
        )
        self.workspace_y = (
            float(self.get_parameter("workspace_y_min").value),
            float(self.get_parameter("workspace_y_max").value),
        )
        self.fixed_z = float(self.get_parameter("fixed_z").value)
        self.goal = {
            "x": float(self.get_parameter("goal_x").value),
            "y": float(self.get_parameter("goal_y").value),
            "z": float(self.get_parameter("goal_z").value),
        }
        self.clearance_m = float(self.get_parameter("clearance_m").value)
        self.prompt_topic = str(self.get_parameter("prompt_topic").value)
        self.verified_plan_topic = str(self.get_parameter("verified_plan_topic").value)
        self.mission_state_topic = str(self.get_parameter("mission_state_topic").value)
        self.required_mission_state = str(self.get_parameter("required_mission_state").value)
        self.require_mission_state = bool(self.get_parameter("require_mission_state").value)
        self.snapshot_after_hover_s = float(self.get_parameter("snapshot_after_hover_s").value)
        self.start_drift_replan_m = float(self.get_parameter("start_drift_replan_m").value)
        self.feedback_enabled = bool(self.get_parameter("feedback_enabled").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.fresh_data_timeout_s = float(self.get_parameter("fresh_data_timeout_s").value)
        self.active_plan_id = None
        self.waiting_for_verification = False
        self.single_shot_complete = False

        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, QVIO_QOS)
        self.mission_state_sub = self.create_subscription(String, self.mission_state_topic, self.mission_state_callback, 10)
        self.obstacle_sub = self.create_subscription(String, self.obstacle_topic, self.obstacle_callback, 10)
        self.verified_sub = self.create_subscription(String, self.verified_plan_topic, self.verified_callback, 10)
        self.prompt_pub = self.create_publisher(String, self.prompt_topic, 10)
        self.timer = self.create_timer(0.5, self.publish_prompt_if_ready)
        self.get_logger().info(
            f"Prompt generator waiting for {self.required_mission_state} on {self.mission_state_topic}; "
            f"obstacles from {self.obstacle_topic}"
        )

    def resolve_obstacle_topic(self):
        mode = str(self.get_parameter("mode").value).strip().lower()
        if mode == "semantic":
            return str(self.get_parameter("semantic_obstacle_topic").value)
        if mode == "normal":
            return str(self.get_parameter("normal_obstacle_topic").value)
        self.get_logger().warning(
            f"Unsupported mode '{mode}'. Expected 'semantic' or 'normal'. Defaulting to semantic obstacle topic."
        )
        return str(self.get_parameter("semantic_obstacle_topic").value)

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        heading_deg = math.degrees(self.quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w))
        self.current_pose = {
            "x": round(position.x, 2),
            "y": round(position.y, 2),
            "z": self.fixed_z,
            "heading_deg": round(heading_deg, 1),
            "stamp": time.time(),
        }

    def obstacle_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse obstacle JSON: {exc}")
            return

        self.latest_obstacle_msg = payload
        self.latest_obstacle_stamp = time.time()

    def mission_state_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse mission state JSON: {exc}")
            return

        previous_state = self.latest_mission_state.get("state") if self.latest_mission_state else None
        self.latest_mission_state = payload
        self.latest_mission_state_stamp = time.time()
        if payload.get("state") == self.required_mission_state:
            if previous_state != self.required_mission_state:
                self.hover_started_s = time.time()
        else:
            self.hover_started_s = None

    def verified_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse verified plan JSON: {exc}")
            return

        plan_id = payload.get("plan_id")
        if str(plan_id) != str(self.active_plan_id):
            self.get_logger().warning(
                f"Ignoring verified result for plan_id={plan_id}; waiting for plan_id={self.active_plan_id}.",
                throttle_duration_sec=5.0,
            )
            return

        passed = bool(payload.get("passed", False))
        self.get_logger().info(f"Verified result received for plan_id={plan_id}: passed={passed}.")
        self.waiting_for_verification = False
        if self.single_shot and passed:
            self.single_shot_complete = True
            self.last_verification_feedback = None
            self.get_logger().info(
                f"Plan {plan_id} passed verification. "
                "Single-shot prompt generation is complete."
            )
            return

        if not passed and self.feedback_enabled:
            self.last_verification_feedback = payload
        self.next_plan_id += 1
        self.active_plan_id = None
        self.get_logger().info(f"Plan {plan_id} verification complete with passed={passed}. Prompt generator unlocked.")

    def publish_prompt_if_ready(self):
        if self.single_shot_complete:
            return
        if self.waiting_for_verification:
            self.get_logger().info(
                f"Waiting for verification result for plan_id={self.active_plan_id} before publishing another prompt.",
                throttle_duration_sec=5.0,
            )
            return
        if not self.has_fresh_inputs():
            return

        if self.latched_context is None or self.start_drift_exceeded():
            self.latched_context = self.snapshot_context()
            self.last_verification_feedback = None
            self.get_logger().info(f"Latched planning context at hover pose: {self.latched_context['start']}")

        context = self.latched_context
        start = context["start"]
        goal = context["goal"]
        obstacles = context["obstacles"]
        nl_env = self.build_nl_env(start, goal, obstacles)
        prompt_parts = [INSTRUCTIONS, nl_env, self.constraints()]
        feedback = self.format_verification_feedback()
        if feedback:
            prompt_parts.append(feedback)
        prompt = "\n".join(prompt_parts)

        if nl_env != self.last_printed_nl:
            self.get_logger().info(nl_env)
            self.last_printed_nl = nl_env
        if prompt != self.last_printed_prompt:
            self.get_logger().info(f"Generated prompt:\n{prompt}")
            self.last_printed_prompt = prompt

        envelope = {
            "plan_id": self.next_plan_id,
            "prompt": prompt,
            "nl_env": nl_env,
            "start": start,
            "goal": goal,
            "workspace": context["workspace"],
            "obstacles": obstacles,
            "attempt": context["attempt"],
            "latched_context_timestamp": context["timestamp"],
            "timestamp": time.time(),
        }

        self.active_plan_id = self.next_plan_id
        context["attempt"] += 1
        self.waiting_for_verification = True
        msg = String()
        msg.data = json.dumps(envelope)
        self.prompt_pub.publish(msg)
        self.get_logger().info(f"Published prompt for plan_id={self.active_plan_id}.")

    def has_fresh_inputs(self):
        now = time.time()
        if self.require_mission_state:
            if self.latest_mission_state is None:
                self.get_logger().warning(
                    f"Prompt generator is waiting for mission state on {self.mission_state_topic}.",
                    throttle_duration_sec=5.0,
                )
                return False
            if now - self.latest_mission_state_stamp > self.fresh_data_timeout_s:
                self.get_logger().warning("Mission state is stale; waiting for fresh hover state.", throttle_duration_sec=5.0)
                return False
            if self.latest_mission_state.get("state") != self.required_mission_state:
                self.get_logger().info(
                    f"Waiting for mission state {self.required_mission_state}; "
                    f"current={self.latest_mission_state.get('state')}.",
                    throttle_duration_sec=5.0,
                )
                return False
            if self.hover_started_s is None or now - self.hover_started_s < self.snapshot_after_hover_s:
                self.get_logger().info("Waiting for hover state to settle before snapshot.", throttle_duration_sec=5.0)
                return False

        if self.current_pose is None and not self.require_mission_state:
            self.get_logger().warning(f"Prompt generator is waiting for pose data on {self.pose_topic}.", throttle_duration_sec=5.0)
            return False
        needs_obstacle_snapshot = self.latched_context is None
        if needs_obstacle_snapshot and self.latest_obstacle_msg is None:
            self.get_logger().warning(
                f"Prompt generator is waiting for obstacle data on {self.obstacle_topic}.",
                throttle_duration_sec=5.0,
            )
            return False
        if self.current_pose is not None and now - self.current_pose["stamp"] > self.fresh_data_timeout_s:
            self.get_logger().warning(f"Prompt generator pose data is stale; waiting for a fresh {self.pose_topic} update.", throttle_duration_sec=5.0)
            return False
        if needs_obstacle_snapshot and (
            self.latest_obstacle_stamp is None or now - self.latest_obstacle_stamp > self.fresh_data_timeout_s
        ):
            self.get_logger().warning(
                f"Prompt generator obstacle data is stale; waiting for a fresh {self.obstacle_topic} update.",
                throttle_duration_sec=5.0,
            )
            return False
        return True

    def snapshot_context(self):
        start = self.build_start()
        return {
            "start": start,
            "goal": dict(self.goal),
            "workspace": {
                "x": list(self.workspace_x),
                "y": list(self.workspace_y),
                "z": self.fixed_z,
            },
            "obstacles": self.latest_obstacle_msg.get("obstacles", []),
            "timestamp": time.time(),
            "attempt": 1,
        }

    def build_start(self):
        if self.require_mission_state and self.latest_mission_state is not None:
            position = self.latest_mission_state.get("position", {})
            heading = self.latest_mission_state.get("heading_deg")
            return {
                "x": round(float(position.get("x", 0.0)), 2),
                "y": round(float(position.get("y", 0.0)), 2),
                "z": self.fixed_z,
                "heading_deg": round(float(heading), 1) if heading is not None else 0.0,
            }
        return {
            "x": round(self.current_pose["x"], 2),
            "y": round(self.current_pose["y"], 2),
            "z": self.fixed_z,
            "heading_deg": self.current_pose["heading_deg"],
        }

    def start_drift_exceeded(self):
        if self.latched_context is None or not self.require_mission_state or self.latest_mission_state is None:
            return False
        position = self.latest_mission_state.get("position", {})
        dx = float(position.get("x", 0.0)) - float(self.latched_context["start"]["x"])
        dy = float(position.get("y", 0.0)) - float(self.latched_context["start"]["y"])
        dz = float(position.get("z", self.fixed_z)) - float(self.latched_context["start"]["z"])
        drift = math.sqrt(dx * dx + dy * dy + dz * dz)
        if drift > self.start_drift_replan_m:
            self.get_logger().warning(
                f"Hover drift {drift:.2f} m exceeded {self.start_drift_replan_m:.2f} m; relatching context."
            )
            return True
        return False

    def build_nl_env(self, start, goal, obstacles):
        distance = math.hypot(goal["x"] - start["x"], goal["y"] - start["y"])
        direction = self.direction_name(goal["x"] - start["x"], goal["y"] - start["y"])

        obstacle_lines = self.describe_obstacles(obstacles)
        obstacle_text = " ".join(obstacle_lines) if obstacle_lines else "No obstacles currently detected."

        return (
            "Mission state: the UAV has already taken off and is holding hover at the start position. "
            "Use this hover position as the first waypoint/reference for planning. "
            f"Workspace: x=[{self.workspace_x[0]:.1f},{self.workspace_x[1]:.1f}]m, "
            f"y=[{self.workspace_y[0]:.1f},{self.workspace_y[1]:.1f}]m, z={self.fixed_z:.1f} fixed. "
            f"Start: ({start['x']:.2f},{start['y']:.2f},{self.fixed_z:.1f}), "
            f"heading={start['heading_deg']:.1f}deg. "
            f"Goal: ({goal['x']:.1f},{goal['y']:.1f},{goal['z']:.1f}), direction {direction}, "
            f"distance≈{distance:.1f}m. Obstacles with x-y spans: {obstacle_text}"
        )

    def constraints(self):
        return (
            "Constraints:\n"
            f"- all waypoints in NED frame, z must stay {self.fixed_z:.1f}\n"
            "- final waypoint must be goal coordinates\n"
            f"- maintain >={self.clearance_m:.2f}m clearance from obstacle x-y boxes\n"
            "- stay within workspace\n"
            "- no waypoint should be within obstacle boxes, walls, or near corners\n"
            "- prefer sparse, smooth, monotonic progress through open space\n"
            "- return only the structured output requested by the response model"
        )

    def format_verification_feedback(self):
        if not self.feedback_enabled or not self.last_verification_feedback:
            return ""
        failed = self.last_verification_feedback.get("failed_constraints", [])
        table = self.last_verification_feedback.get("verification_feedback_table")
        if not table:
            table = self.last_verification_feedback.get("metrics", {}).get("feedback_table", "")
        return (
            "Previous plan failed verification. Regenerate a sparse waypoint plan that fixes the failed "
            "metrics while keeping the same latched hover start, goal, workspace, and obstacle snapshot.\n"
            f"Failed constraints: {', '.join(failed) if failed else 'unknown'}\n"
            f"{table}\n"
            "Prefer a route with larger obstacle clearance, monotonic goal progress, and smoother segment changes."
        )

    def describe_obstacles(self, obstacles):
        descriptions = []
        for index, obstacle in enumerate(obstacles, start=1):
            min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
            max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
            label = obstacle.get("label") or obstacle.get("shape") or "unknown"
            size_phrase = self.size_phrase(obstacle)
            descriptions.append(
                f"{index} {label}: x=[{min_corner[0]:.2f},{max_corner[0]:.2f}], "
                f"y=[{min_corner[1]:.2f},{max_corner[1]:.2f}], size {size_phrase}."
            )
        return descriptions

    @staticmethod
    def size_phrase(obstacle):
        size = obstacle.get("size", [0.0, 0.0, 0.0])
        width = float(size[0]) if len(size) > 0 else 0.0
        depth = float(size[1]) if len(size) > 1 else 0.0
        height = float(size[2]) if len(size) > 2 else 0.0

        if height > 1.2 and max(width, depth) < 0.7:
            return "tall/narrow"
        if height > 1.0 and max(width, depth) >= 0.7:
            return "tall/wide"
        if max(width, depth) < 0.8:
            return "small/narrow"
        if max(width, depth) < 1.5:
            return "medium/narrow"
        return "wide"

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def direction_name(dx, dy):
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return "stationary"
        vertical = "N" if dx >= 0.0 else "S"
        horizontal = "E" if dy >= 0.0 else "W"
        if abs(dx) < 0.2:
            return horizontal
        if abs(dy) < 0.2:
            return vertical
        return vertical + horizontal


def main():
    rclpy.init()
    node = PromptGenerator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
