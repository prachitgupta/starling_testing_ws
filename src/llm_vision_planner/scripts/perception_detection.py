#!/usr/bin/env python3
import json
import time

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from std_msgs.msg import String

QVIO_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
POINT_CLOUD_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
DETECTION_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

try:
    from voxl_msgs.msg import AiDetection
except ImportError:
    try:
        from voxl_msgs.msg import Aidetection as AiDetection
    except ImportError:
        AiDetection = None


class SemanticObstaclePerception(Node):
    def __init__(self):
        super().__init__("semantic_obstacle_perception")
        if AiDetection is None:
            raise ImportError("voxl_msgs/msg/AiDetection is required for /tflite_data")

        self.declare_parameter("point_cloud_topic", "/voa_pc_out")
        self.declare_parameter("pose_topic", "/qvio")
        self.declare_parameter("detection_topic", "/tflite_data")
        self.declare_parameter("obstacle_topic", "/llm_vision/semantic_obstacles")
        self.declare_parameter("goal_x", 3.0)
        self.declare_parameter("goal_y", 1.0)
        self.declare_parameter("goal_z", -0.2)
        self.declare_parameter("min_range_m", 0.3)
        self.declare_parameter("max_range_m", 6.0)
        self.declare_parameter("dbscan_eps", 0.2)
        self.declare_parameter("dbscan_min_samples", 8)

        self.current_pose = None
        self.obstacles = []
        self.semantic_labels = []
        self.goal = (
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
            float(self.get_parameter("goal_z").value),
        )

        self.pc_sub = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("point_cloud_topic").value),
            self.pc_callback,
            POINT_CLOUD_QOS,
        )
        self.pose_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter("pose_topic").value),
            self.pose_callback,
            QVIO_QOS,
        )
        self.detection_sub = self.create_subscription(
            AiDetection,
            str(self.get_parameter("detection_topic").value),
            self.detection_callback,
            DETECTION_QOS,
        )
        self.obstacle_pub = self.create_publisher(
            String,
            str(self.get_parameter("obstacle_topic").value),
            10,
        )
        self.timer = self.create_timer(0.5, self.publish_obstacles)

    def pose_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
        self.current_pose = (
            round(p.x, 2),
            round(p.y, 2),
            round(p.z, 2),
            round(np.degrees(yaw), 1),
        )

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def detection_callback(self, msg):
        labels = self.extract_labels(msg)
        if labels:
            self.semantic_labels = labels

    def extract_labels(self, msg):
        labels = []
        candidates = getattr(msg, "detections", None)
        if candidates is None:
            candidates = getattr(msg, "detection", None)
        if candidates is None:
            candidates = [msg]
        elif not isinstance(candidates, (list, tuple)):
            candidates = list(candidates)

        for detection in candidates:
            label = self.label_from_detection(detection)
            if label and label not in labels:
                labels.append(label)
        return labels

    @staticmethod
    def label_from_detection(detection):
        for field_name in ("label", "class_name", "name", "object_name", "class_label"):
            value = getattr(detection, field_name, None)
            if value:
                return str(value)

        class_id = getattr(detection, "class_id", None)
        if class_id is not None:
            return f"class_{class_id}"
        return None

    def pc_callback(self, msg):
        if self.current_pose is None:
            return

        pts = np.array(
            [
                [p[0], p[1], p[2]]
                for p in pc2.read_points(
                    msg,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ],
            dtype=float,
        )
        if len(pts) < 10:
            self.obstacles = []
            return

        drone = np.array(self.current_pose[:3], dtype=float)
        dists = np.linalg.norm(pts - drone, axis=1)
        min_range = float(self.get_parameter("min_range_m").value)
        max_range = float(self.get_parameter("max_range_m").value)
        pts = pts[(dists > min_range) & (dists < max_range)]
        if len(pts) < 10:
            self.obstacles = []
            return

        db = DBSCAN(
            eps=float(self.get_parameter("dbscan_eps").value),
            min_samples=int(self.get_parameter("dbscan_min_samples").value),
            n_jobs=-1,
        ).fit(pts)

        obstacles = []
        for index, label in enumerate(sorted(set(db.labels_))):
            if label == -1:
                continue
            cluster = pts[db.labels_ == label]
            semantic_label = self.semantic_label_for_cluster(index)
            obstacles.append(self.describe_obstacle(cluster, drone, semantic_label))

        self.obstacles = sorted(obstacles, key=lambda obstacle: obstacle["distance_m"])

    def semantic_label_for_cluster(self, cluster_index):
        if not self.semantic_labels:
            return "unknown"
        if cluster_index < len(self.semantic_labels):
            return self.semantic_labels[cluster_index]
        return self.semantic_labels[0]

    @staticmethod
    def describe_obstacle(cluster, drone_pos, semantic_label):
        centroid = np.round(cluster.mean(axis=0), 2)
        min_xyz = np.round(cluster.min(axis=0), 2)
        max_xyz = np.round(cluster.max(axis=0), 2)
        size = np.round(max_xyz - min_xyz, 2)
        distance = round(float(np.linalg.norm(centroid - drone_pos)), 2)

        delta = centroid - drone_pos
        bearing = round(float(np.degrees(np.arctan2(delta[1], delta[0]))), 1)

        return {
            "centroid": centroid.tolist(),
            "min_corner": min_xyz.tolist(),
            "max_corner": max_xyz.tolist(),
            "size": size.tolist(),
            "distance_m": distance,
            "bearing_deg": bearing,
            "label": semantic_label,
            "shape": semantic_label,
            "point_count": int(len(cluster)),
        }

    def publish_obstacles(self):
        if self.current_pose is None:
            return

        descriptor = {
            "pose": self.current_pose,
            "obstacles": self.obstacles,
            "semantic_labels": self.semantic_labels,
            "goal": self.goal,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(descriptor)
        self.obstacle_pub.publish(msg)


def main():
    rclpy.init()
    node = SemanticObstaclePerception()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
