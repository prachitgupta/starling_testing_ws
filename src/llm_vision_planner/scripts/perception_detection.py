#!/usr/bin/env python3
from collections import deque
import json
import math
import time

import numpy as np
import rclpy
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String

try:
    from voxl_msgs.msg import Aidetection
except ImportError:
    Aidetection = None


BEST_EFFORT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
DETECTION_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

DYNAMIC_LABELS = {"person", "dog", "suitcase", "cat", "bicycle", "motorcycle", "car", "truck", "bus"}
HARDCODED_DEPTH_M = {
    "chair": 1.0,
    "person": 1.5,
    "dog": 1.5,
    "suitcase": 1.5,
    "cat": 1.0,
    "bicycle": 2.0,
    "motorcycle": 2.0,
    "car": 3.0,
    "truck": 3.0,
    "bus": 3.0,
}


class SemanticObstaclePerception(Node):
    def __init__(self):
        super().__init__("semantic_obstacle_perception")
        if Aidetection is None:
            raise ImportError("voxl_msgs/msg/Aidetection is required for /tflite_data")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("detection_topic", "/tflite_data"),
                ("detector_image_topic", "/tflite"),
                ("point_cloud_topic", "/tof_pc"),
                ("pose_topic", "/fmu/out/vehicle_odometry"),
                ("obstacle_topic", "/llm_vision/semantic_obstacles"),
                ("goal_x", 0.0),
                ("goal_y", 1.5),
                ("goal_z", -0.5),
                ("hires_fx", 501.5315609739347),
                ("hires_fy", 502.8286520347511),
                ("hires_cx", 508.1806484040712),
                ("hires_cy", 380.6556051674785),
                ("hires_width", 1024),
                ("hires_height", 768),
                ("camera_calibration_valid", True),
                ("detection_camera", "hires_small_color"),
                (
                    "detection_camera_aliases",
                    ["hires_small_color"],
                ),
                ("detection_cam_body_x_m", 0.068),
                ("detection_cam_body_y_m", 0.012),
                ("detection_cam_body_z_m", -0.015),
                ("detection_cam_roll_deg", 0.0),
                ("detection_cam_pitch_deg", 90.0),
                ("detection_cam_yaw_deg", 90.0),
                ("depth_cam_body_x_m", 0.066),
                ("depth_cam_body_y_m", 0.009),
                ("depth_cam_body_z_m", -0.012),
                ("depth_cam_roll_deg", 0.0),
                ("depth_cam_pitch_deg", 90.0),
                ("depth_cam_yaw_deg", 180.0),
                ("point_cloud_frame", "tof_optical"),
                ("min_confidence", 0.70),
                ("detection_timeout_s", 5.0),
                ("detector_timeout_s", 5.0),
                ("point_cloud_timeout_s", 5.0),
                ("pose_timeout_s", 5.0),
                ("max_sync_slop_s", 10.0),
                ("pose_history_size", 2000),
                ("detection_to_depth_offset_s", 0.0),
                ("z_estimation_mode", "depth"),
                ("min_tof_depth_m", 0.20),
                ("max_tof_depth_m", 6.0),
                ("min_valid_point_cloud_points", 100),
                ("min_frustum_points", 10),
                ("bbox_inner_margin_fraction", 0.30),
                ("depth_near_percentile", 25.0),
                ("depth_cluster_tolerance_m", 0.35),
                ("obstacle_hold_s", 10.0),
                ("held_depth_health_grace_s", 10.0),
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
        self.detection_camera_translation_body = np.array(
            [
                float(self.get_parameter("detection_cam_body_x_m").value),
                float(self.get_parameter("detection_cam_body_y_m").value),
                float(self.get_parameter("detection_cam_body_z_m").value),
            ],
            dtype=float,
        )
        self.depth_camera_translation_body = np.array(
            [
                float(self.get_parameter("depth_cam_body_x_m").value),
                float(self.get_parameter("depth_cam_body_y_m").value),
                float(self.get_parameter("depth_cam_body_z_m").value),
            ],
            dtype=float,
        )
        self.rotation_detection_camera_to_body = self.modalai_intrinsic_xyz_matrix(
            math.radians(float(self.get_parameter("detection_cam_roll_deg").value)),
            math.radians(float(self.get_parameter("detection_cam_pitch_deg").value)),
            math.radians(float(self.get_parameter("detection_cam_yaw_deg").value)),
        )
        self.rotation_depth_camera_to_body = self.modalai_intrinsic_xyz_matrix(
            math.radians(float(self.get_parameter("depth_cam_roll_deg").value)),
            math.radians(float(self.get_parameter("depth_cam_pitch_deg").value)),
            math.radians(float(self.get_parameter("depth_cam_yaw_deg").value)),
        )

        self.detections = []
        self.detector_image_samples = deque(maxlen=50)
        self.point_cloud_samples = deque(maxlen=20)
        self.pose_samples = deque(
            maxlen=max(100, int(self.get_parameter("pose_history_size").value))
        )
        self.pose = None
        self.detection_count = 0
        self.last_detection_stamp = None
        self.last_point_cloud_stamp = None
        self.last_pose_stamp = None
        self.last_detector_image_stamp = None
        self.detector_image_size = None
        self.last_sync_basis = "unavailable"
        self.last_point_cloud_sync_delta_s = None
        self.last_pose_sync_delta_s = None
        self.last_log_stamp = 0.0
        self.obstacle_tracks = []

        self.create_subscription(
            Aidetection,
            str(self.get_parameter("detection_topic").value),
            self.detection_callback,
            DETECTION_QOS,
        )
        detector_image_topic = str(self.get_parameter("detector_image_topic").value).strip()
        if detector_image_topic:
            self.create_subscription(
                Image,
                detector_image_topic,
                self.detector_image_callback,
                BEST_EFFORT_QOS,
            )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("point_cloud_topic").value),
            self.point_cloud_callback,
            BEST_EFFORT_QOS,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter("pose_topic").value),
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
        now = time.monotonic()
        self.detections.append((msg, now))
        self.detection_count += 1
        self.last_detection_stamp = now
        self.prune_detections()

    def detector_image_callback(self, msg):
        now = time.monotonic()
        self.last_detector_image_stamp = now
        self.detector_image_size = (int(msg.width), int(msg.height))
        self.detector_image_samples.append(
            {
                "arrival_s": now,
                "source_stamp_ns": self.ros_stamp_ns(msg.header.stamp),
            }
        )

    def point_cloud_callback(self, msg):
        now = time.monotonic()
        raw_points = self.point_cloud_xyz(msg)
        if raw_points.size == 0:
            raw_points = np.empty((0, 3), dtype=float)
        else:
            raw_points = raw_points.reshape((-1, 3))
            raw_points = raw_points[np.all(np.isfinite(raw_points), axis=1)]
        ranges = np.linalg.norm(raw_points, axis=1)
        valid_depth_count = int(
            np.sum(
                (ranges >= float(self.get_parameter("min_tof_depth_m").value))
                & (ranges <= float(self.get_parameter("max_tof_depth_m").value))
            )
        )
        self.point_cloud_samples.append(
            {
                "points": raw_points,
                "valid_depth_count": valid_depth_count,
                "arrival_s": now,
                "source_stamp_ns": self.ros_stamp_ns(msg.header.stamp),
                "frame_id": str(msg.header.frame_id),
                "width": int(msg.width),
                "height": int(msg.height),
            }
        )
        self.last_point_cloud_stamp = now

    @staticmethod
    def point_cloud_xyz(msg):
        fields = {field.name: field for field in msg.fields}
        if any(name not in fields for name in ("x", "y", "z")):
            return np.empty((0, 3), dtype=float)
        if any(fields[name].datatype != PointField.FLOAT32 for name in ("x", "y", "z")):
            return np.empty((0, 3), dtype=float)

        endian = ">" if msg.is_bigendian else "<"
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
                "offsets": [fields[name].offset for name in ("x", "y", "z")],
                "itemsize": msg.point_step,
            }
        )
        rows = []
        for row in range(msg.height):
            start = row * msg.row_step
            rows.append(
                np.frombuffer(
                    msg.data,
                    dtype=dtype,
                    count=msg.width,
                    offset=start,
                )
            )
        if not rows:
            return np.empty((0, 3), dtype=float)
        data = np.concatenate(rows)
        return np.column_stack((data["x"], data["y"], data["z"])).astype(
            float,
            copy=False,
        )

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        rotation_body_to_world = self.quaternion_body_to_world_matrix(msg.q)
        if rotation_body_to_world is None:
            return
        now = time.monotonic()
        pose = {
            "position": np.array(msg.position[:3], dtype=float),
            "rotation_body_to_world": rotation_body_to_world,
            "arrival_s": now,
            "source_stamp_ns": int(msg.timestamp_sample) * 1000,
        }
        self.pose = pose
        self.pose_samples.append(pose)
        self.last_pose_stamp = now

    def publish_obstacles(self):
        self.prune_detections()
        obstacles = []
        no_depth = []
        low_confidence = []
        labels = []
        detections = self.current_detections()
        frame_arrival_s = max((stamp for _, stamp in detections), default=None)
        point_cloud_sample, sync_reason = self.match_point_cloud(frame_arrival_s)
        pose = self.match_pose(point_cloud_sample, frame_arrival_s)

        for detection, _ in detections:
            label = str(detection.class_name)
            confidence = self.detection_confidence(detection)
            if label not in labels:
                labels.append(label)
            if confidence < float(self.get_parameter("min_confidence").value):
                low_confidence.append(label)
                continue

            obstacle, reason = self.build_obstacle(
                detection,
                confidence,
                point_cloud_sample,
                pose,
                sync_reason,
            )
            if obstacle is None:
                no_depth.append({"label": label, "reason": reason})
            else:
                obstacles.append(obstacle)

        obstacles = self.update_obstacle_tracks(obstacles)
        obstacles.sort(key=lambda obstacle: obstacle["distance_m"])
        status = self.health_status(no_depth, obstacles)
        payload = {
            "pose": self.pose_tuple(),
            "obstacles": obstacles,
            "semantic_labels": labels,
            "no_depth": no_depth,
            "low_conf": low_confidence,
            "goal": self.goal,
            "timestamp": time.time(),
            "source": self.perception_source(),
            "healthy": status["healthy"],
            "range_is_measured": status["range_is_measured"],
            "status": status,
            "frame": "local_ned",
        }
        self.obstacle_pub.publish(String(data=json.dumps(payload)))
        self.log_summary(obstacles, len(no_depth), len(low_confidence), status)

    def update_obstacle_tracks(self, observed):
        now = time.monotonic()
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
                obstacle["source"] = f"{obstacle.get('source', 'tflite_tof_projected')}_held"
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

    def build_obstacle(
        self,
        detection,
        confidence,
        point_cloud_sample,
        pose,
        sync_reason,
    ):
        if pose is None:
            return None, "no synchronized pose"
        if not self.detection_camera_matches(getattr(detection, "cam", "")):
            return None, (
                f"camera mismatch: got '{getattr(detection, 'cam', '')}', "
                f"expected one of {sorted(self.accepted_detection_cameras())}"
            )

        x1, y1, x2, y2 = self.normalized_box(detection)
        if x2 <= x1 or y2 <= y1:
            return None, "empty bbox"

        label = str(detection.class_name)
        if self.z_estimation_mode() == "hardcoded":
            return self.build_hardcoded_obstacle(
                x1,
                y1,
                x2,
                y2,
                label,
                confidence,
                pose,
            ), None
        if point_cloud_sample is None:
            return None, sync_reason or "no synchronized point cloud"

        points_body, reason = self.point_cloud_points_body(point_cloud_sample, pose)
        if points_body is None:
            return None, reason

        selected_body, selected_detection, candidate_count = self.depth_points_for_box(
            points_body,
            x1,
            y1,
            x2,
            y2,
        )
        min_points = int(self.get_parameter("min_frustum_points").value)
        if len(selected_body) < min_points:
            return None, f"projected_tof_points={candidate_count}, clustered_points={len(selected_body)}"

        center_body = np.median(selected_body, axis=0)
        optical_depth = float(np.median(selected_detection[:, 2]))
        width_m = max(0.10, (x2 - x1) * self.image_width * optical_depth / self.fx)
        height_m = max(0.10, (y2 - y1) * self.image_height * optical_depth / self.fy)
        depth_m = max(
            0.20,
            float(np.percentile(selected_detection[:, 2], 90.0))
            - float(np.percentile(selected_detection[:, 2], 10.0)),
        )
        corners_detection = np.array(
            [
                [x, y, z]
                for x in (-0.5 * width_m, 0.5 * width_m)
                for y in (-0.5 * height_m, 0.5 * height_m)
                for z in (-0.5 * depth_m, 0.5 * depth_m)
            ],
            dtype=float,
        )
        corners_body = (
            center_body
            + (self.rotation_detection_camera_to_body @ corners_detection.T).T
        )
        return self.obstacle_payload(
            center_body,
            corners_body,
            pose,
            label,
            confidence,
            f"tof_projected_{len(selected_body)}of{candidate_count}pts",
            "tflite_tof_projected",
            point_cloud_sample.get("sync_delta_s"),
            str(getattr(detection, "cam", "")),
            center_body,
            width_m,
            float(np.linalg.norm(center_body - self.detection_camera_translation_body)),
        ), None

    def build_hardcoded_obstacle(self, x1, y1, x2, y2, label, confidence, pose):
        depth = self.hardcoded_depth(label)
        u = 0.5 * (x1 + x2) * self.image_width
        v = 0.5 * (y1 + y2) * self.image_height
        ray_detection = np.array(
            [(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0],
            dtype=float,
        )
        ray_detection /= np.linalg.norm(ray_detection)
        center_body = (
            self.detection_camera_translation_body
            + self.rotation_detection_camera_to_body @ (ray_detection * depth)
        )
        optical_depth = max(0.1, depth * ray_detection[2])
        width_m = max(0.10, (x2 - x1) * self.image_width * optical_depth / self.fx)
        height_m = max(0.10, (y2 - y1) * self.image_height * optical_depth / self.fy)
        depth_m = max(0.20, 0.25 * depth)
        corners_detection = np.array(
            [
                [x, y, z]
                for x in (-0.5 * width_m, 0.5 * width_m)
                for y in (-0.5 * height_m, 0.5 * height_m)
                for z in (-0.5 * depth_m, 0.5 * depth_m)
            ],
            dtype=float,
        )
        corners_body = (
            center_body
            + (self.rotation_detection_camera_to_body @ corners_detection.T).T
        )
        return self.obstacle_payload(
            center_body,
            corners_body,
            pose,
            label,
            confidence,
            "class_prior_not_measured",
            "tflite_class_prior",
            None,
            str(self.get_parameter("detection_camera").value),
            center_body,
            width_m,
            float(np.linalg.norm(center_body - self.detection_camera_translation_body)),
        )

    def obstacle_payload(
        self,
        center_body,
        corners_body,
        pose,
        label,
        confidence,
        depth_source,
        source,
        sync_delta_s,
        camera,
        front_center_body,
        visible_width_m,
        front_range_m,
    ):
        rotation = pose["rotation_body_to_world"]
        drone_pos = pose["position"]
        centroid = drone_pos + rotation @ center_body
        corners_world = drone_pos + (rotation @ corners_body.T).T
        front_surface_center = drone_pos + rotation @ front_center_body
        camera_world = drone_pos + rotation @ self.detection_camera_translation_body
        view_delta_xy = front_surface_center[:2] - camera_world[:2]
        view_norm = float(np.linalg.norm(view_delta_xy))
        if view_norm <= 1e-9:
            view_axis_xy = np.zeros(2, dtype=float)
        else:
            view_axis_xy = view_delta_xy / view_norm
        lateral_axis_xy = np.array([-view_axis_xy[1], view_axis_xy[0]], dtype=float)
        min_corner = corners_world.min(axis=0)
        max_corner = corners_world.max(axis=0)
        size = max_corner - min_corner
        delta = centroid - drone_pos
        payload = {
            "centroid": np.round(centroid, 2).tolist(),
            "front_surface_center": np.round(front_surface_center, 3).tolist(),
            "visible_width_m": round(float(visible_width_m), 3),
            "front_range_m": round(float(front_range_m), 3),
            "view_axis_xy": np.round(view_axis_xy, 6).tolist(),
            "lateral_axis_xy": np.round(lateral_axis_xy, 6).tolist(),
            "min_corner": np.round(min_corner, 2).tolist(),
            "max_corner": np.round(max_corner, 2).tolist(),
            "size": np.round(size, 2).tolist(),
            "distance_m": round(float(np.linalg.norm(delta)), 2),
            "bearing_deg": round(float(np.degrees(math.atan2(delta[1], delta[0]))), 1),
            "label": label,
            "shape": label,
            "confidence": round(confidence, 2),
            "is_dynamic": label.lower() in DYNAMIC_LABELS,
            "depth_source": depth_source,
            "source": source,
            "camera": camera,
        }
        if sync_delta_s is not None:
            payload["sync_delta_s"] = round(float(sync_delta_s), 3)
        return payload

    @staticmethod
    def hardcoded_depth(label):
        return HARDCODED_DEPTH_M.get(str(label).lower(), 1.0)

    def point_cloud_points_body(self, sample, pose):
        points = sample["points"]
        frame = str(self.get_parameter("point_cloud_frame").value).strip().lower()
        if frame == "tof_optical":
            return (
                self.depth_camera_translation_body
                + (self.rotation_depth_camera_to_body @ points.T).T
            ), None
        if frame == "body_frd":
            return points, None
        if frame == "local_ned":
            return (
                pose["rotation_body_to_world"].T
                @ (points - pose["position"]).T
            ).T, None
        return None, f"unsupported point_cloud_frame='{frame}'"

    def depth_points_for_box(self, points_body, x1, y1, x2, y2):
        points_detection = (
            self.rotation_detection_camera_to_body.T
            @ (points_body - self.detection_camera_translation_body).T
        ).T
        ranges = np.linalg.norm(
            points_body - self.depth_camera_translation_body,
            axis=1,
        )
        valid = (
            (points_detection[:, 2] > 0.01)
            & (ranges >= float(self.get_parameter("min_tof_depth_m").value))
            & (ranges <= float(self.get_parameter("max_tof_depth_m").value))
        )
        points_body = points_body[valid]
        points_detection = points_detection[valid]
        ranges = ranges[valid]
        if len(points_body) == 0:
            return points_body, points_detection, 0

        normalized_u = (
            self.fx * points_detection[:, 0] / points_detection[:, 2] + self.cx
        ) / self.image_width
        normalized_v = (
            self.fy * points_detection[:, 1] / points_detection[:, 2] + self.cy
        ) / self.image_height
        full_mask = (
            (normalized_u >= x1)
            & (normalized_u <= x2)
            & (normalized_v >= y1)
            & (normalized_v <= y2)
        )
        margin = max(
            0.0,
            min(0.45, float(self.get_parameter("bbox_inner_margin_fraction").value)),
        )
        inner_mask = (
            (normalized_u >= x1 + margin * (x2 - x1))
            & (normalized_u <= x2 - margin * (x2 - x1))
            & (normalized_v >= y1 + margin * (y2 - y1))
            & (normalized_v <= y2 - margin * (y2 - y1))
        )
        min_points = int(self.get_parameter("min_frustum_points").value)
        roi_mask = inner_mask if int(np.sum(inner_mask)) >= min_points else full_mask
        candidate_count = int(np.sum(roi_mask))
        if candidate_count < min_points:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=float),
                candidate_count,
            )

        roi_ranges = ranges[roi_mask]
        percentile = max(
            0.0,
            min(100.0, float(self.get_parameter("depth_near_percentile").value)),
        )
        seed = float(np.percentile(roi_ranges, percentile))
        tolerance = max(
            0.01,
            float(self.get_parameter("depth_cluster_tolerance_m").value),
        )
        cluster = roi_ranges <= seed + tolerance
        return points_body[roi_mask][cluster], points_detection[roi_mask][cluster], candidate_count

    def normalized_box(self, detection):
        vals = [float(detection.x_min), float(detection.y_min), float(detection.x_max), float(detection.y_max)]
        if all(-0.01 <= value <= 1.01 for value in vals):
            x1, y1, x2, y2 = vals
        else:
            x1 = vals[0] / self.image_width
            y1 = vals[1] / self.image_height
            x2 = vals[2] / self.image_width
            y2 = vals[3] / self.image_height

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)

    def match_point_cloud(self, detection_arrival_s):
        if self.z_estimation_mode() != "depth":
            return None, None
        if detection_arrival_s is None:
            return None, "no current detection frame"
        if not self.point_cloud_samples:
            return None, "no point cloud"

        offset_s = float(self.get_parameter("detection_to_depth_offset_s").value)
        image_sample = None
        if self.detector_image_samples:
            image_sample = min(
                self.detector_image_samples,
                key=lambda candidate: abs(
                    float(candidate["arrival_s"]) - detection_arrival_s
                ),
            )

        source_candidates = [
            candidate
            for candidate in self.point_cloud_samples
            if int(candidate.get("source_stamp_ns", 0)) > 0
        ]
        if (
            image_sample is not None
            and int(image_sample.get("source_stamp_ns", 0)) > 0
            and source_candidates
        ):
            target_ns = int(image_sample["source_stamp_ns"]) + int(offset_s * 1e9)
            sample = min(
                source_candidates,
                key=lambda candidate: abs(
                    int(candidate["source_stamp_ns"]) - target_ns
                ),
            )
            delta_s = abs(int(sample["source_stamp_ns"]) - target_ns) / 1e9
            sync_basis = "tflite_image_and_tof_header"
        else:
            target_s = detection_arrival_s + offset_s
            sample = min(
                self.point_cloud_samples,
                key=lambda candidate: abs(float(candidate["arrival_s"]) - target_s),
            )
            delta_s = abs(float(sample["arrival_s"]) - target_s)
            sync_basis = "local_receipt_time"

        if delta_s > float(self.get_parameter("max_sync_slop_s").value):
            return None, f"point cloud sync delta={delta_s:.3f}s"
        matched = dict(sample)
        matched["sync_delta_s"] = delta_s
        self.last_sync_basis = sync_basis
        self.last_point_cloud_sync_delta_s = delta_s
        return matched, None

    def match_pose(self, point_cloud_sample, fallback_arrival_s):
        if not self.pose_samples:
            return None

        target_stamp_ns = (
            int(point_cloud_sample.get("source_stamp_ns", 0))
            if point_cloud_sample is not None
            else 0
        )
        source_candidates = [
            candidate
            for candidate in self.pose_samples
            if int(candidate.get("source_stamp_ns", 0)) > 0
        ]
        if target_stamp_ns > 0 and source_candidates:
            pose = min(
                source_candidates,
                key=lambda candidate: abs(
                    int(candidate["source_stamp_ns"]) - target_stamp_ns
                ),
            )
            delta_s = abs(int(pose["source_stamp_ns"]) - target_stamp_ns) / 1e9
        elif fallback_arrival_s is not None:
            pose = min(
                self.pose_samples,
                key=lambda candidate: abs(
                    float(candidate["arrival_s"]) - fallback_arrival_s
                ),
            )
            delta_s = abs(float(pose["arrival_s"]) - fallback_arrival_s)
        else:
            return None

        if delta_s > float(self.get_parameter("max_sync_slop_s").value):
            return None
        self.last_pose_sync_delta_s = delta_s
        return pose

    def accepted_detection_cameras(self):
        aliases = {
            self.normalize_camera_name(value)
            for value in self.get_parameter("detection_camera_aliases").value
        }
        aliases.add(
            self.normalize_camera_name(
                str(self.get_parameter("detection_camera").value)
            )
        )
        return {value for value in aliases if value}

    def detection_camera_matches(self, camera):
        return self.normalize_camera_name(camera) in self.accepted_detection_cameras()

    @staticmethod
    def normalize_camera_name(camera):
        return str(camera).strip().strip("/").split("/")[-1].lower()

    def prune_detections(self):
        now = time.monotonic()
        timeout_s = float(self.get_parameter("detection_timeout_s").value)
        self.detections = [(det, stamp) for det, stamp in self.detections if now - stamp <= timeout_s]

    def current_detections(self):
        if not self.detections:
            return []
        try:
            latest_frame_id = int(self.detections[-1][0].frame_id)
            return [(det, stamp) for det, stamp in self.detections if int(det.frame_id) == latest_frame_id]
        except (AttributeError, TypeError, ValueError):
            return list(self.detections)

    @staticmethod
    def held_depth_grace_labels(no_depth, obstacles, grace_s):
        if not no_depth or grace_s <= 0.0:
            return []
        held_labels = {
            str(obstacle.get("label", "")).strip().lower()
            for obstacle in obstacles
            if bool(obstacle.get("held"))
            and float(obstacle.get("last_seen_age_s", math.inf)) <= grace_s
            and str(obstacle.get("depth_source", "")).startswith("tof_projected_")
            and math.isfinite(float(obstacle.get("front_range_m", math.nan)))
        }
        failed_labels = []
        for failure in no_depth:
            label = str(failure.get("label", "")).strip().lower()
            reason = str(failure.get("reason", ""))
            if not reason.startswith("point cloud sync delta=") or label not in held_labels:
                return []
            failed_labels.append(label)
        return sorted(set(failed_labels))

    def health_status(self, no_depth, obstacles):
        now = time.monotonic()
        detector_topic = str(self.get_parameter("detector_image_topic").value).strip()
        detector_timeout = float(self.get_parameter("detector_timeout_s").value)
        point_cloud_timeout = float(self.get_parameter("point_cloud_timeout_s").value)
        pose_timeout = float(self.get_parameter("pose_timeout_s").value)

        if not detector_topic:
            detector = "unmonitored"
        elif self.last_detector_image_stamp is None:
            detector = "missing"
        elif now - self.last_detector_image_stamp > detector_timeout:
            detector = "stale"
        elif self.detector_image_size != (self.image_width, self.image_height):
            detector = (
                f"size_mismatch:{self.detector_image_size[0]}x"
                f"{self.detector_image_size[1]}"
            )
        else:
            detector = "ok"

        if self.last_pose_stamp is None:
            pose = "missing"
        elif now - self.last_pose_stamp > pose_timeout:
            pose = "stale"
        else:
            pose = "ok"

        if self.z_estimation_mode() != "depth":
            point_cloud = "not_used"
        elif self.last_point_cloud_stamp is None:
            point_cloud = "missing"
        elif now - self.last_point_cloud_stamp > point_cloud_timeout:
            point_cloud = "stale"
        elif (
            self.point_cloud_samples
            and self.point_cloud_samples[-1]["valid_depth_count"]
            < int(self.get_parameter("min_valid_point_cloud_points").value)
        ):
            point_cloud = (
                "insufficient_valid_points:"
                f"{self.point_cloud_samples[-1]['valid_depth_count']}"
            )
        else:
            point_cloud = "ok"

        detector_ok = detector in ("ok", "unmonitored")
        depth_ok = point_cloud in ("ok", "not_used")
        calibration_valid = bool(
            self.get_parameter("camera_calibration_valid").value
        )
        grace_s = max(0.0, float(self.get_parameter("held_depth_health_grace_s").value))
        grace_labels = self.held_depth_grace_labels(no_depth, obstacles, grace_s)
        grace_used = bool(no_depth) and bool(grace_labels)
        healthy = (
            detector_ok
            and pose == "ok"
            and depth_ok
            and calibration_valid
            and (not no_depth or grace_used)
        )
        latest_sample = self.point_cloud_samples[-1] if self.point_cloud_samples else None
        current_detections = self.current_detections()
        latest_detection = current_detections[-1][0] if current_detections else None
        return {
            "healthy": bool(healthy),
            "detector": detector,
            "pose": pose,
            "point_cloud": point_cloud,
            "camera_calibration": (
                "verified" if calibration_valid else "unverified"
            ),
            "point_cloud_frame_param": str(
                self.get_parameter("point_cloud_frame").value
            ),
            "point_cloud_header_frame": (
                latest_sample["frame_id"] if latest_sample is not None else None
            ),
            "valid_point_cloud_points": (
                latest_sample["valid_depth_count"]
                if latest_sample is not None
                else 0
            ),
            "detection_camera": str(
                self.get_parameter("detection_camera").value
            ),
            "reported_detection_camera": (
                str(getattr(latest_detection, "cam", ""))
                if latest_detection is not None
                else None
            ),
            "detection_frame_id": (
                int(getattr(latest_detection, "frame_id", 0))
                if latest_detection is not None
                else None
            ),
            "image_size": [self.image_width, self.image_height],
            "range_is_measured": (
                self.z_estimation_mode() == "depth" and calibration_valid
            ),
            "sync_basis": self.last_sync_basis,
            "point_cloud_sync_delta_s": self.last_point_cloud_sync_delta_s,
            "pose_sync_delta_s": self.last_pose_sync_delta_s,
            "held_depth_grace_used": grace_used,
            "held_depth_grace_s": grace_s,
            "held_depth_grace_labels": grace_labels,
        }

    def log_summary(self, obstacles, no_depth_count, low_conf_count, status):
        if not bool(self.get_parameter("debug").value):
            return
        now = time.monotonic()
        if now - self.last_log_stamp < float(self.get_parameter("print_obstacles_period_s").value):
            return
        self.last_log_stamp = now

        z_estimation_mode = self.z_estimation_mode()
        lines = [
            f"semantic_obstacles={len(obstacles)} no_depth={no_depth_count} low_conf={low_conf_count} "
            f"healthy={status['healthy']} z_mode={z_estimation_mode} "
            f"detector={status['detector']} pc={status['point_cloud']} pose={status['pose']}"
        ]
        for obstacle in obstacles:
            min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
            max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
            lines.append(
                f"{obstacle['label']} pos={obstacle['centroid']} "
                f"x=[{min_corner[0]:.2f},{max_corner[0]:.2f}] "
                f"y=[{min_corner[1]:.2f},{max_corner[1]:.2f}] "
                f"z=[{min_corner[2]:.2f},{max_corner[2]:.2f}] "
                f"size={obstacle['size']} distance={obstacle['distance_m']:.2f}m "
                f"prompt='{self.prompt_obstacle_text(obstacle)}'"
            )
        self.get_logger().info("\n".join(lines))

    def z_estimation_mode(self):
        mode = str(self.get_parameter("z_estimation_mode").value).strip().lower()
        return "hardcoded" if mode == "hardcoded" else "depth"

    def perception_source(self):
        if self.z_estimation_mode() == "depth":
            return "tflite_tof_projected"
        return "tflite_class_prior"

    def prompt_obstacle_text(self, obstacle):
        min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
        max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
        label = obstacle.get("label") or obstacle.get("shape") or "unknown"
        return (
            f"{label}: x=[{min_corner[0]:.2f},{max_corner[0]:.2f}], "
            f"y=[{min_corner[1]:.2f},{max_corner[1]:.2f}], "
            f"distance {float(obstacle.get('distance_m', 0.0)):.2f}m, "
            f"size {self.size_phrase(obstacle)}."
        )

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

    def pose_tuple(self):
        if self.pose is None:
            return None
        p = self.pose["position"]
        rotation = self.pose["rotation_body_to_world"]
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        return (
            round(float(p[0]), 2),
            round(float(p[1]), 2),
            round(float(p[2]), 2),
            round(float(np.degrees(yaw)), 1),
        )

    @staticmethod
    def detection_confidence(detection):
        for field_name in ("class_confidence", "detection_confidence"):
            try:
                value = float(getattr(detection, field_name))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                return value
        return -1.0

    @staticmethod
    def modalai_intrinsic_xyz_matrix(roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return rx @ ry @ rz

    @staticmethod
    def quaternion_body_to_world_matrix(quaternion):
        q = np.asarray(quaternion, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            return None
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return None
        w, x, y, z = q / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @staticmethod
    def ros_stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def main():
    rclpy.init()
    node = SemanticObstaclePerception()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
