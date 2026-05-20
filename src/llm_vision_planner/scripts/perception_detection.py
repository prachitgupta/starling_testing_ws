#!/usr/bin/env python3
import json
import math
import time

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

try:
    from voxl_msgs.msg import Aidetection
except ImportError:
    Aidetection = None


BEST_EFFORT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
DETECTION_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=50,
)

DYNAMIC_LABELS = {"person", "dog", "cat", "bicycle", "motorcycle", "car", "truck", "bus"}


class SemanticObstaclePerception(Node):
    def __init__(self):
        super().__init__("semantic_obstacle_perception")
        if Aidetection is None:
            raise ImportError("voxl_msgs/msg/Aidetection is required for /tflite_data")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("detection_topic", "/tflite_data"),
                ("point_cloud_topic", "/tof_pc"),
                ("pose_topic", "/fmu/out/vehicle_odometry"),
                ("pose_msg_type", "px4_vehicle_odometry"),
                ("obstacle_topic", "/llm_vision/semantic_obstacles"),
                ("goal_x", 3.0),
                ("goal_y", 0.0),
                ("goal_z", -0.45),
                ("hires_fx", 459.25),
                ("hires_fy", 459.77),
                ("hires_cx", 640.0),
                ("hires_cy", 400.0),
                ("hires_width", 1280),
                ("hires_height", 800),
                ("cam_body_x_m", 0.066),
                ("cam_body_y_m", 0.009),
                ("cam_body_z_m", -0.012),
                ("cam_roll_deg", 0.0),
                ("cam_pitch_deg", 90.0),
                ("cam_yaw_deg", 180.0),
                ("min_confidence", 0.60),
                ("detection_timeout_s", 1.0),
                ("min_tof_depth_m", 0.20),
                ("max_tof_depth_m", 6.0),
                ("frustum_margin_deg", 5.0),
                ("min_frustum_points", 3),
                ("obstacle_hold_s", 3.0),
                ("obstacle_match_distance_m", 0.75),
                ("publish_hz", 2.0),
                ("debug", True),
                ("print_obstacles_period_s", 1.0),
            ],
        )

        self.goal = (
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
            float(self.get_parameter("goal_z").value),
        )
        self.fx = float(self.get_parameter("hires_fx").value)
        self.fy = float(self.get_parameter("hires_fy").value)
        self.cx = float(self.get_parameter("hires_cx").value)
        self.cy = float(self.get_parameter("hires_cy").value)
        self.image_width = int(self.get_parameter("hires_width").value)
        self.image_height = int(self.get_parameter("hires_height").value)
        self.camera_translation_body = np.array(
            [
                float(self.get_parameter("cam_body_x_m").value),
                float(self.get_parameter("cam_body_y_m").value),
                float(self.get_parameter("cam_body_z_m").value),
            ],
            dtype=float,
        )

        self.rotation_body_to_camera = self.rpy_matrix(
            math.radians(float(self.get_parameter("cam_roll_deg").value)),
            math.radians(float(self.get_parameter("cam_pitch_deg").value)),
            math.radians(float(self.get_parameter("cam_yaw_deg").value)),
        )

        self.detections = []
        self.point_cloud_msg = None
        self.point_cloud_points_world = None
        self.pose = None
        self.detection_count = 0
        self.last_detection_stamp = None
        self.last_point_cloud_stamp = None
        self.last_log_stamp = 0.0
        self.obstacle_tracks = []

        self.create_subscription(
            Aidetection,
            str(self.get_parameter("detection_topic").value),
            self.detection_callback,
            DETECTION_QOS,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("point_cloud_topic").value),
            self.point_cloud_callback,
            BEST_EFFORT_QOS,
        )
        pose_topic = str(self.get_parameter("pose_topic").value)
        pose_msg_type = str(self.get_parameter("pose_msg_type").value).lower()
        if pose_msg_type in ("nav_msgs/odometry", "odometry", "qvio"):
            self.create_subscription(
                Odometry,
                pose_topic,
                self.odom_pose_callback,
                BEST_EFFORT_QOS,
            )
        else:
            self.create_subscription(
                VehicleOdometry,
                pose_topic,
                self.pose_callback,
                BEST_EFFORT_QOS,
            )
        self.obstacle_pub = self.create_publisher(
            String,
            str(self.get_parameter("obstacle_topic").value),
            10,
        )
        publish_hz = max(0.1, float(self.get_parameter("publish_hz").value))
        self.create_timer(1.0 / publish_hz, self.publish_obstacles)

    def detection_callback(self, msg):
        now = time.time()
        self.detections.append((msg, now))
        self.detection_count += 1
        self.last_detection_stamp = now
        self.prune_detections()

    def point_cloud_callback(self, msg):
        self.point_cloud_msg = msg
        self.point_cloud_points_world = None
        self.last_point_cloud_stamp = time.time()

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        q = msg.q
        self.pose = {
            "position": np.array(msg.position[:3], dtype=float),
            "yaw": math.atan2(
                2.0 * (q[0] * q[3] + q[1] * q[2]),
                1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
            ),
        }

    def odom_pose_callback(self, msg):
        q = msg.pose.pose.orientation
        self.pose = {
            "position": np.array(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ],
                dtype=float,
            ),
            "yaw": self.yaw_from_xyzw(q.x, q.y, q.z, q.w),
        }

    def publish_obstacles(self):
        self.prune_detections()
        obstacles = []
        no_depth = []
        low_confidence = []
        labels = []

        for detection, _ in self.current_detections():
            label = str(detection.class_name)
            confidence = self.yolo_confidence(detection)
            if label not in labels:
                labels.append(label)
            if confidence < float(self.get_parameter("min_confidence").value):
                low_confidence.append(label)
                continue

            obstacle, reason = self.build_obstacle(detection, confidence)
            if obstacle is None:
                no_depth.append({"label": label, "reason": reason})
            else:
                obstacles.append(obstacle)

        obstacles = self.update_obstacle_tracks(obstacles)
        obstacles.sort(key=lambda obstacle: obstacle["distance_m"])
        payload = {
            "pose": self.pose_tuple(),
            "obstacles": obstacles,
            "semantic_labels": labels,
            "no_depth": no_depth,
            "low_conf": low_confidence,
            "goal": self.goal,
            "timestamp": time.time(),
            "source": "yolo_tof_frustum",
        }
        self.obstacle_pub.publish(String(data=json.dumps(payload)))
        self.log_summary(obstacles, len(no_depth), len(low_confidence))

    def update_obstacle_tracks(self, observed):
        now = time.time()
        hold_s = float(self.get_parameter("obstacle_hold_s").value)
        match_distance = float(self.get_parameter("obstacle_match_distance_m").value)
        used_tracks = set()

        for obstacle in observed:
            track_index = self.match_track(obstacle, used_tracks, match_distance)
            tracked = dict(obstacle)
            tracked["last_seen_age_s"] = 0.0
            tracked["held"] = False
            if track_index is None:
                self.obstacle_tracks.append({"obstacle": tracked, "last_seen": now})
                used_tracks.add(len(self.obstacle_tracks) - 1)
            else:
                self.obstacle_tracks[track_index] = {"obstacle": tracked, "last_seen": now}
                used_tracks.add(track_index)

        self.obstacle_tracks = [
            track for track in self.obstacle_tracks if now - track["last_seen"] <= hold_s
        ]

        merged = []
        for track in self.obstacle_tracks:
            obstacle = dict(track["obstacle"])
            age = now - track["last_seen"]
            obstacle["last_seen_age_s"] = round(float(age), 2)
            obstacle["held"] = age > 0.05
            if obstacle["held"]:
                obstacle["source"] = f"{obstacle.get('source', 'yolo_tof_frustum')}_held"
            merged.append(obstacle)
        return merged

    def match_track(self, obstacle, used_tracks, match_distance):
        label = obstacle.get("label") or obstacle.get("shape")
        centroid = np.array(obstacle.get("centroid", [0.0, 0.0, 0.0]), dtype=float)
        best_index = None
        best_distance = float("inf")
        for index, track in enumerate(self.obstacle_tracks):
            if index in used_tracks:
                continue
            tracked_obstacle = track["obstacle"]
            tracked_label = tracked_obstacle.get("label") or tracked_obstacle.get("shape")
            if tracked_label != label:
                continue
            tracked_centroid = np.array(tracked_obstacle.get("centroid", [0.0, 0.0, 0.0]), dtype=float)
            distance = float(np.linalg.norm(centroid - tracked_centroid))
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_distance <= match_distance:
            return best_index
        return None

    def build_obstacle(self, detection, confidence):
        if self.pose is None:
            return None, "no pose"

        x1, y1, x2, y2 = self.normalized_box(detection)
        if x2 <= x1 or y2 <= y1:
            return None, "empty bbox"

        u1, u2 = x1 * self.image_width, x2 * self.image_width
        v1, v2 = y1 * self.image_height, y2 * self.image_height
        margin = math.radians(float(self.get_parameter("frustum_margin_deg").value))

        az_left = math.atan2(u1 - self.cx, self.fx) - margin
        az_right = math.atan2(u2 - self.cx, self.fx) + margin
        el_top = math.atan2(v1 - self.cy, self.fy) - margin
        el_bottom = math.atan2(v2 - self.cy, self.fy) + margin

        ray_cam = np.array(
            [((u1 + u2) * 0.5 - self.cx) / self.fx, ((v1 + v2) * 0.5 - self.cy) / self.fy, 1.0],
            dtype=float,
        )
        ray_cam /= np.linalg.norm(ray_cam)
        ray_body = self.rotation_body_to_camera.T @ ray_cam
        ray_world = self.yaw_matrix(self.pose["yaw"]) @ ray_body

        label = str(detection.class_name)
        depth, point_count = self.depth_from_frustum(az_left, az_right, el_top, el_bottom)
        if depth is None:
            return None, f"tof_frustum_points={point_count}"

        drone_pos = self.pose["position"]
        camera_origin_world = drone_pos + self.yaw_matrix(self.pose["yaw"]) @ self.camera_translation_body
        world_pos = camera_origin_world + ray_world * depth
        width_m = max(0.1, (u2 - u1) * depth / self.fx)
        height_m = max(0.1, (v2 - v1) * depth / self.fy)
        half = np.array([0.5 * width_m, 0.5 * width_m, 0.5 * height_m], dtype=float)
        delta = world_pos - drone_pos

        return {
            "centroid": np.round(world_pos, 2).tolist(),
            "min_corner": np.round(world_pos - half, 2).tolist(),
            "max_corner": np.round(world_pos + half, 2).tolist(),
            "size": [round(width_m, 2), round(width_m, 2), round(height_m, 2)],
            "distance_m": round(float(np.linalg.norm(delta)), 2),
            "bearing_deg": round(float(np.degrees(math.atan2(delta[1], delta[0]))), 1),
            "label": label,
            "shape": label,
            "confidence": round(confidence, 2),
            "is_dynamic": label in DYNAMIC_LABELS,
            "depth_source": f"tof_{point_count}pts",
            "source": "yolo_tof_frustum",
        }, None

    def depth_from_frustum(self, az_left, az_right, el_top, el_bottom):
        if self.point_cloud_msg is None or self.pose is None:
            return None, 0

        if self.point_cloud_points_world is None:
            raw_points = list(
                pc2.read_points(self.point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
            )
            if not raw_points:
                return None, 0
            points = np.array([[p[0], p[1], p[2]] for p in raw_points], dtype=float)
            self.point_cloud_points_world = points[np.all(np.isfinite(points), axis=1)]

        points = self.point_cloud_points_world
        if len(points) == 0:
            return None, 0

        camera_origin_world = (
            self.pose["position"] + self.yaw_matrix(self.pose["yaw"]) @ self.camera_translation_body
        )
        v_world = points - camera_origin_world
        distances = np.linalg.norm(v_world, axis=1)
        min_depth = float(self.get_parameter("min_tof_depth_m").value)
        max_depth = float(self.get_parameter("max_tof_depth_m").value)
        depth_mask = (distances > min_depth) & (distances < max_depth)
        v_world = v_world[depth_mask]
        distances = distances[depth_mask]
        if len(v_world) == 0:
            return None, 0

        v_body = (self.yaw_matrix(-self.pose["yaw"]) @ v_world.T).T
        v_cam = (self.rotation_body_to_camera @ v_body.T).T
        front_mask = v_cam[:, 2] > 0.01
        v_cam = v_cam[front_mask]
        distances = distances[front_mask]
        if len(v_cam) == 0:
            return None, 0

        az = np.arctan2(v_cam[:, 0], v_cam[:, 2])
        el = np.arctan2(v_cam[:, 1], v_cam[:, 2])
        in_frustum = (az >= az_left) & (az <= az_right) & (el >= el_top) & (el <= el_bottom)
        count = int(np.sum(in_frustum))
        if count < int(self.get_parameter("min_frustum_points").value):
            return None, count
        return float(np.median(distances[in_frustum])), count

    def normalized_box(self, detection):
        vals = [float(detection.x_min), float(detection.y_min), float(detection.x_max), float(detection.y_max)]
        if max(abs(value) for value in vals) <= 1.5:
            x1, y1, x2, y2 = vals
        else:
            x1 = vals[0] / self.image_width
            y1 = vals[1] / self.image_height
            x2 = vals[2] / self.image_width
            y2 = vals[3] / self.image_height

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)

    def prune_detections(self):
        now = time.time()
        timeout_s = float(self.get_parameter("detection_timeout_s").value)
        self.detections = [(det, stamp) for det, stamp in self.detections if now - stamp <= timeout_s]

    def current_detections(self):
        if not self.detections:
            return []
        try:
            latest_frame_id = max(int(det.frame_id) for det, _ in self.detections)
            return [(det, stamp) for det, stamp in self.detections if int(det.frame_id) == latest_frame_id]
        except (AttributeError, TypeError, ValueError):
            return list(self.detections)

    def log_summary(self, obstacles, no_depth_count, low_conf_count):
        if not bool(self.get_parameter("debug").value):
            return
        now = time.time()
        if now - self.last_log_stamp < float(self.get_parameter("print_obstacles_period_s").value):
            return
        self.last_log_stamp = now

        pc_status = "missing" if self.point_cloud_msg is None else f"{self.point_cloud_msg.width}x{self.point_cloud_msg.height}"
        lines = [
            f"semantic_obstacles={len(obstacles)} no_depth={no_depth_count} low_conf={low_conf_count} "
            f"pc={pc_status} pose={'ok' if self.pose is not None else 'missing'}"
        ]
        for obstacle in obstacles:
            lines.append(
                f"{obstacle['label']} conf={obstacle['confidence']:.2f} "
                f"dist={obstacle['distance_m']:.2f}m pos={obstacle['centroid']} "
                f"box={obstacle['min_corner']}..{obstacle['max_corner']} src={obstacle['depth_source']}"
            )
        self.get_logger().info("\n".join(lines))

    def pose_tuple(self):
        if self.pose is None:
            return None
        p = self.pose["position"]
        return (
            round(float(p[0]), 2),
            round(float(p[1]), 2),
            round(float(p[2]), 2),
            round(float(np.degrees(self.pose["yaw"])), 1),
        )

    @staticmethod
    def yolo_confidence(detection):
        for field_name in ("class_confidence", "detection_confidence"):
            try:
                value = float(getattr(detection, field_name))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                return value
        return -1.0

    @staticmethod
    def rpy_matrix(roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )

    @staticmethod
    def yaw_matrix(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    @staticmethod
    def yaw_from_xyzw(x, y, z, w):
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )


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
