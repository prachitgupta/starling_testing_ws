#!/usr/bin/env python3
"""Independent, append-only ROS publication-data recorder (no control outputs)."""
import json
import math
import time
import uuid
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from std_msgs.msg import String

ACP = "ADAPTIVE_CONFORMAL_PREDICTION"


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MissionDataLogger(Node):
    def __init__(self):
        super().__init__("mission_data_logger")
        self.declare_parameter("log_directory", "~/.ros/llm_vision/mission_logs")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("vicon_objects_json", "[]")
        self.declare_parameter("vicon_vehicle_topic", "/vicon/Starling2/Starling2")
        self.declare_parameter("vicon_person_topic", "/vicon/person/person")
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
        directory = Path(str(self.get_parameter("log_directory").value)).expanduser() / self.run_id
        directory.mkdir(parents=True, exist_ok=False)
        self.stream = (directory / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self.sequence = 0
        catalog = json.loads(str(self.get_parameter("vicon_objects_json").value))
        metadata = {"schema_version": 1, "run_id": self.run_id,
                    "created_unix_ns": time.time_ns(), "vicon_objects": catalog,
                    "trajectory_state_order": ["x_m", "y_m", "vx_m_s", "vy_m_s"],
                    "trajectory_control_order": ["ax_m_s2", "ay_m_s2"],
                    "trajectory_frame": "local_ned", "trajectory_dynamics": "ideal_planar_double_integrator",
                    "vicon_frame": "original message header.frame_id; raw transforms are not converted",
                    "px4_time": "timestamp/timestamp_sample in microseconds on PX4 clock",
                    "time_alignment": "receipt ROS/unix/monotonic nanoseconds and original source stamps retained",
                    "nonfinite_values": "encoded as null", "expert_logging": "ACP only",
                    "coverage_scope": "evaluated trajectory residuals; not physical tracking certification"}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        sensor_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
        plan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscriptions_kept = []
        odom_topic = str(self.get_parameter("pose_topic").value)
        self.subscriptions_kept.append(self.create_subscription(VehicleOdometry, odom_topic,
            lambda msg: self.record(odom_topic, "px4_msgs/VehicleOdometry", message_to_ordereddict(msg)), sensor_qos))
        vicon_topics = {str(self.get_parameter("vicon_vehicle_topic").value),
                        str(self.get_parameter("vicon_person_topic").value),
                        *(item["topic"] for item in catalog)}
        for topic in sorted(vicon_topics):
            if topic:
                self.subscriptions_kept.append(self.create_subscription(TransformStamped, topic,
                    lambda msg, topic=topic: self.record(topic, "geometry_msgs/TransformStamped", message_to_ordereddict(msg)), sensor_qos))
        latched = ("plan_raw", "plan_refined", "plan_candidate_verified", "plan_verified",
                   "optimized_trajectory", "executed_reference", "acp_update", "ground_truth_obstacles",
                   "mission_proposal", "launch_proposal", "operator_response", "safety_tube_ready")
        for name in latched:
            topic = "/llm_vision/" + name
            self.subscriptions_kept.append(self.create_subscription(String, topic,
                lambda msg, topic=topic: self.json_callback(topic, msg), plan_qos))
        for name in ("semantic_obstacles", "sim_obstacles", "mission_state"):
            topic = "/llm_vision/" + name
            self.subscriptions_kept.append(self.create_subscription(String, topic,
                lambda msg, topic=topic: self.json_callback(topic, msg), sensor_qos))
        self.get_logger().info(f"Independent mission logs: {directory}")

    def json_callback(self, topic, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.record(topic, "std_msgs/String", {"invalid_json": msg.data})
            return
        # Gate on the trajectory's own mode, never arrival order of a global mode topic.
        def without_expert(value):
            if isinstance(value, dict):
                mode = value.get("interaction_mode")
                return {key: without_expert(item) for key, item in value.items()
                        if mode == ACP or not (key.startswith("expert_") or key.startswith("selected_expert"))}
            if isinstance(value, list):
                return [without_expert(item) for item in value]
            return value
        # Top-level ACP authorizes its expert samples and all nested trajectory metadata.
        if not isinstance(payload, dict) or payload.get("interaction_mode") != ACP:
            payload = without_expert(payload)
        self.record(topic, "std_msgs/String(JSON)", payload)

    def record(self, topic, message_type, payload):
        self.sequence += 1
        event = {"run_id": self.run_id, "sequence": self.sequence,
                 "receipt_unix_ns": time.time_ns(), "receipt_monotonic_ns": time.monotonic_ns(),
                 "receipt_ros_ns": self.get_clock().now().nanoseconds,
                 "topic": topic, "message_type": message_type, "data": json_value(payload)}
        try:
            self.stream.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self.get_logger().error(f"logging failed for {topic}: {exc}")

    def destroy_node(self):
        self.stream.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = MissionDataLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
