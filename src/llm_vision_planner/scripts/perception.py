#!/usr/bin/env python3
import json
import time

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from std_msgs.msg import String

ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
POINT_CLOUD_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class ObstaclePerception(Node):
    def __init__(self):
        super().__init__("obstacle_perception")

        self.declare_parameter("point_cloud_topic", "/voa_pc_out")
        self.declare_parameter("point_cloud_frame", "auto")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("obstacle_topic", "/llm_vision/obstacles")
        self.declare_parameter("goal_x", 2.5)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_z", -0.25)
        self.declare_parameter("min_range_m", 0.20)
        self.declare_parameter("max_range_m", 4.5)
        self.declare_parameter("min_z_m", -2.0)
        self.declare_parameter("max_z_m", 1.0)
        self.declare_parameter("ego_exclusion_radius_m", 0.25)
        self.declare_parameter("dbscan_eps", 0.5)
        self.declare_parameter("dbscan_min_samples", 8)
        self.declare_parameter("debug", False)

        self.current_pose = None
        self.obstacles = []
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
            VehicleOdometry,
            str(self.get_parameter("pose_topic").value),
            self.pose_callback,
            ODOM_QOS,
        )
        self.obstacle_pub = self.create_publisher(
            String,
            str(self.get_parameter("obstacle_topic").value),
            10,
        )
        self.timer = self.create_timer(0.5, self.publish_obstacles)

    def log_warning(self, *args, **kwargs):
        if bool(self.get_parameter("debug").value):
            self.get_logger().warning(*args)

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            self.log_warning("Ignoring VehicleOdometry that is not in NED pose frame.", throttle_duration_sec=5.0)
            return
        yaw = self.quat_to_yaw(msg.q[1], msg.q[2], msg.q[3], msg.q[0])
        self.current_pose = (
            round(float(msg.position[0]), 2),
            round(float(msg.position[1]), 2),
            round(float(msg.position[2]), 2),
            round(np.degrees(yaw), 1),
        )

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def pc_callback(self, msg):
        if self.current_pose is None:
            return

        raw_pts = np.array(
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
        if len(raw_pts) < 10:
            self.obstacles = []
            return

        drone = np.array(self.current_pose[:3], dtype=float)
        yaw = np.radians(float(self.current_pose[3]))
        pts = self.points_to_local_ned(raw_pts, drone, yaw, msg.header.frame_id)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 10:
            self.obstacles = []
            return

        dists = np.linalg.norm(pts - drone, axis=1)
        min_range = float(self.get_parameter("min_range_m").value)
        max_range = float(self.get_parameter("max_range_m").value)
        min_z = float(self.get_parameter("min_z_m").value)
        max_z = float(self.get_parameter("max_z_m").value)
        ego_radius = float(self.get_parameter("ego_exclusion_radius_m").value)
        keep = (
            (dists > max(min_range, ego_radius))
            & (dists < max_range)
            & (pts[:, 2] >= min_z)
            & (pts[:, 2] <= max_z)
        )
        pts = pts[keep]
        if len(pts) < 10:
            self.obstacles = []
            return

        db = DBSCAN(
            eps=float(self.get_parameter("dbscan_eps").value),
            min_samples=int(self.get_parameter("dbscan_min_samples").value),
            n_jobs=-1,
        ).fit(pts)

        obstacles = []
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster = pts[db.labels_ == label]
            obstacles.append(self.describe_obstacle(cluster, drone))

        self.obstacles = sorted(obstacles, key=lambda obstacle: obstacle["distance_m"])

    def points_to_local_ned(self, points, drone_pos, yaw, frame_id):
        source_frame = str(self.get_parameter("point_cloud_frame").value).lower()
        if source_frame == "auto":
            source_frame = self.infer_point_cloud_frame(frame_id)
        if source_frame == "local_ned":
            return points
        if source_frame == "body_ned":
            return drone_pos + (self.yaw_matrix(yaw) @ points.T).T
        self.log_warning(
            f"Unknown point_cloud_frame '{source_frame}', treating points as local_ned.",
            throttle_duration_sec=5.0,
        )
        return points

    @staticmethod
    def infer_point_cloud_frame(frame_id):
        frame = (frame_id or "").lower()
        if any(token in frame for token in ("body", "base_link", "stereo", "tof", "camera")):
            return "body_ned"
        return "local_ned"

    @staticmethod
    def yaw_matrix(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def describe_obstacle(self, cluster, drone_pos):
        centroid = np.round(cluster.mean(axis=0), 2)
        min_xyz = np.round(cluster.min(axis=0), 2)
        max_xyz = np.round(cluster.max(axis=0), 2)
        size = np.round(max_xyz - min_xyz, 2)
        distance = round(float(np.linalg.norm(centroid - drone_pos)), 2)

        delta = centroid - drone_pos
        bearing = round(float(np.degrees(np.arctan2(delta[1], delta[0]))), 1)
        width, depth, height = size

        return {
            "centroid": centroid.tolist(),
            "min_corner": min_xyz.tolist(),
            "max_corner": max_xyz.tolist(),
            "size": size.tolist(),
            "distance_m": distance,
            "bearing_deg": bearing,
            "shape": self.classify_shape(width, depth, height),
            "point_count": int(len(cluster)),
        }

    @staticmethod
    def classify_shape(width, depth, height):
        if height > 1.5 and width < 0.5 and depth < 0.5:
            return "pole"
        if height > 1.2 and width > 1.0 and depth < 0.4:
            return "wall"
        if width > 1.5 and depth > 1.5 and height < 0.5:
            return "ground_obstacle"
        if max(width, depth, height) / (min(width, depth, height) + 1e-3) < 2.0:
            return "compact"
        return "unknown"

    def publish_obstacles(self):
        if self.current_pose is None:
            return

        descriptor = {
            "pose": self.current_pose,
            "obstacles": self.obstacles,
            "goal": self.goal,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(descriptor)
        self.obstacle_pub.publish(msg)


def main():
    rclpy.init()
    node = ObstaclePerception()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
