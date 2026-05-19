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
        self.declare_parameter("pose_topic", DEFAULT_POSE_TOPIC)
        self.declare_parameter("fresh_data_timeout_s", DEFAULT_FRESH_DATA_TIMEOUT_S)

        self.current_pose = None
        self.latest_obstacle_msg = None
        self.latest_obstacle_stamp = None
        self.last_printed_nl = None
        self.last_printed_prompt = None
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
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.fresh_data_timeout_s = float(self.get_parameter("fresh_data_timeout_s").value)
        self.active_plan_id = None
        self.waiting_for_verification = False
        self.single_shot_complete = False

        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, QVIO_QOS)
        self.obstacle_sub = self.create_subscription(String, self.obstacle_topic, self.obstacle_callback, 10)
        self.verified_sub = self.create_subscription(String, self.verified_plan_topic, self.verified_callback, 10)
        self.prompt_pub = self.create_publisher(String, self.prompt_topic, 10)
        self.timer = self.create_timer(0.5, self.publish_prompt_if_ready)
        self.get_logger().info(f"Prompt generator subscribing to obstacle topic: {self.obstacle_topic}")

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
            self.get_logger().info(
                f"Plan {plan_id} passed verification. "
                "Single-shot prompt generation is complete."
            )
            return

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

        descriptor = self.latest_obstacle_msg
        start = self.build_start()
        goal = dict(self.goal)
        nl_env = self.build_nl_env(start, goal, descriptor.get("obstacles", []))
        prompt = "\n".join((INSTRUCTIONS, nl_env, self.constraints()))

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
            "workspace": {
                "x": list(self.workspace_x),
                "y": list(self.workspace_y),
                "z": self.fixed_z,
            },
            "obstacles": descriptor.get("obstacles", []),
            "timestamp": time.time(),
        }

        self.active_plan_id = self.next_plan_id
        self.waiting_for_verification = True
        msg = String()
        msg.data = json.dumps(envelope)
        self.prompt_pub.publish(msg)
        self.get_logger().info(f"Published prompt for plan_id={self.active_plan_id}.")

    def has_fresh_inputs(self):
        now = time.time()
        if self.current_pose is None:
            self.get_logger().warning(f"Prompt generator is waiting for pose data on {self.pose_topic}.", throttle_duration_sec=5.0)
            return False
        if self.latest_obstacle_msg is None:
            self.get_logger().warning(
                f"Prompt generator is waiting for obstacle data on {self.obstacle_topic}.",
                throttle_duration_sec=5.0,
            )
            return False
        if now - self.current_pose["stamp"] > self.fresh_data_timeout_s:
            self.get_logger().warning(f"Prompt generator pose data is stale; waiting for a fresh {self.pose_topic} update.", throttle_duration_sec=5.0)
            return False
        if self.latest_obstacle_stamp is None or now - self.latest_obstacle_stamp > self.fresh_data_timeout_s:
            self.get_logger().warning(
                f"Prompt generator obstacle data is stale; waiting for a fresh {self.obstacle_topic} update.",
                throttle_duration_sec=5.0,
            )
            return False
        return True

    def build_start(self):
        return {
            "x": round(self.current_pose["x"], 2),
            "y": round(self.current_pose["y"], 2),
            "z": self.fixed_z,
            "heading_deg": self.current_pose["heading_deg"],
        }

    def build_nl_env(self, start, goal, obstacles):
        distance = math.hypot(goal["x"] - start["x"], goal["y"] - start["y"])
        direction = self.direction_name(goal["x"] - start["x"], goal["y"] - start["y"])

        obstacle_lines = self.describe_obstacles(obstacles)
        obstacle_text = " ".join(obstacle_lines) if obstacle_lines else "No obstacles currently detected."

        return (
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
