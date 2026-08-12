#!/usr/bin/env python3
"""Publish one recorded calibration environment as a ROS 2 simulation scene."""

import csv
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DatasetScenePublisher(Node):
    def __init__(self):
        super().__init__("dataset_scene_publisher")
        self.declare_parameter(
            "dataset_csv",
            "fine_tuning/datasets/env_ros_commands.csv",
        )
        self.declare_parameter("sample_id", 4)
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("relabel_to_coco", True)
        self.declare_parameter("coco_labels", ["chair", "bottle", "suitcase", "person"])
        self.declare_parameter("fixed_z", -0.5)
        self.declare_parameter("obstacle_topic", "/llm_vision/sim_obstacles")
        self.declare_parameter("mission_state_topic", "/llm_vision/mission_state")
        self.declare_parameter("publish_mission_state", False)

        dataset_path = self.resolve_dataset(str(self.get_parameter("dataset_csv").value))
        self.row = self.load_row(dataset_path, int(self.get_parameter("sample_id").value))
        self.environment = json.loads(self.row["environment"])
        if bool(self.get_parameter("relabel_to_coco").value):
            labels = list(self.get_parameter("coco_labels").value)
            if not labels:
                raise ValueError("coco_labels must not be empty when relabel_to_coco is true")
            for index, obstacle in enumerate(self.environment["obstacles"]):
                obstacle["dataset_label"] = obstacle.get("label")
                obstacle["label"] = str(labels[index % len(labels)])
                obstacle["shape"] = obstacle["label"]

        self.obstacle_pub = self.create_publisher(
            String,
            str(self.get_parameter("obstacle_topic").value),
            10,
        )
        self.mission_state_pub = None
        if bool(self.get_parameter("publish_mission_state").value):
            self.mission_state_pub = self.create_publisher(
                String,
                str(self.get_parameter("mission_state_topic").value),
                10,
            )
        publish_hz = max(0.1, float(self.get_parameter("publish_hz").value))
        self.create_timer(1.0 / publish_hz, self.publish_scene)
        self.get_logger().info(
            f"loaded dataset sample_id={self.row['sample_id']} from {dataset_path}; "
            f"source command={self.row['ros2_pub_command']}"
        )

    @staticmethod
    def resolve_dataset(value):
        requested = Path(value).expanduser()
        candidates = [requested, Path.cwd() / requested]
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.append(Path(get_package_share_directory("llm_vision_planner")) / requested)
        except Exception:
            pass
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"simulation dataset does not exist: {value}")

    @staticmethod
    def load_row(path, sample_id):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if int(row["sample_id"]) == sample_id:
                    return row
        raise ValueError(f"sample_id={sample_id} was not found in {path}")

    def publish_scene(self):
        now = time.time()
        obstacle_payload = {
            "obstacles": self.environment["obstacles"],
            "healthy": True,
            "status": {"source": "env_ros_commands.csv", "sample_id": int(self.row["sample_id"])},
            "frame": "local_ned",
            "timestamp": now,
        }
        self.obstacle_pub.publish(String(data=json.dumps(obstacle_payload)))

        if self.mission_state_pub is not None:
            start = self.environment["start"]
            mission_payload = {
                "state": "HOLDING_FOR_PLAN",
                "position": {
                    "x": float(start["x"]),
                    "y": float(start["y"]),
                    "z": float(self.get_parameter("fixed_z").value),
                },
                "heading_deg": 0.0,
                "failure_reason": None,
                "timestamp": now,
                "source": "dataset_scene_publisher",
            }
            self.mission_state_pub.publish(String(data=json.dumps(mission_payload)))


def main():
    rclpy.init()
    node = DatasetScenePublisher()
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
