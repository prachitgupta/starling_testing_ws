#!/usr/bin/env python3
from collections import deque
import json
import math
from pathlib import Path
import sys
import threading
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

try:
    import bag_reader
    import outline_depth as od
    from obstacle_depth import canonical_label, estimate_depth, prior_for, range_window_for
    from outline_segmentation import EFFICIENTSAM_FILE, box_iou, make_segmenter, mask_box
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bag_reader
    import outline_depth as od
    from obstacle_depth import canonical_label, estimate_depth, prior_for, range_window_for
    from outline_segmentation import EFFICIENTSAM_FILE, box_iou, make_segmenter, mask_box


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
                ("detection_timeout_s", 10.0),
                ("detector_timeout_s", 10.0),
                ("point_cloud_timeout_s", 6.0),
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
                # Depth along the line of sight (thickness behind the front):
                # outline = camera outline (EfficientSAM, prompted with the
                # detector box) + ToF + floor, scripts/outline_depth.py;
                # shape_fit = ToF only, scripts/obstacle_depth.py (also the
                # fallback while an outline is computed or without the model);
                # front_spread = legacy p90-p10 of the visible front surface.
                ("depth_estimator", "outline"),
                ("camera_image_topic", "/hires_small_color"),
                ("segmenter", "efficientsam"),
                ("efficientsam_model", ""),
                (
                    "hires_distortion",
                    [-0.28655908953387943, 0.081789942250115405, 0.00052374055289612631,
                     0.00071156224551124046, -0.010172420045783252],
                ),
                ("outline_tof_rotation_deg", [0.0, 0.0, 0.0]),
                ("outline_max_image_gap_s", 0.5),
                ("outline_max_age_s", 10.0),
                ("outline_wait_s", 8.0),
                ("outline_settle_m", 0.08),
                ("outline_async", True),
                ("shape_fit_front_tolerance_m", 0.15),
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
        self.camera_images = deque(maxlen=8)
        self.outline_lock = threading.Lock()
        self.outline_event = threading.Event()
        self.outline_stop = threading.Event()
        self.outline_job = None
        self.outline_last_key = None
        self.outline_results = []
        self.outline_tracks = []
        self.outline_used = set()
        self.outline_status = "not_used"
        self.segmenter = None
        self.outline_camera = None
        if self.depth_estimator_mode() == "outline" and str(self.get_parameter("camera_image_topic").value).strip():
            self.setup_outline()   # an empty camera_image_topic keeps the ToF-only depths

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
        camera_image_topic = str(self.get_parameter("camera_image_topic").value).strip()
        if self.segmenter is not None and camera_image_topic:
            self.create_subscription(Image, camera_image_topic, self.camera_image_callback, BEST_EFFORT_QOS)
            if bool(self.get_parameter("outline_async").value):
                threading.Thread(target=self.outline_worker, daemon=True).start()
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

    def camera_image_callback(self, msg):
        self.camera_images.append(
            {
                "msg": msg,
                "arrival_s": time.monotonic(),
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
        self.outline_used = set()
        obstacles = []
        no_depth = []
        low_confidence = []
        labels = []
        detections = self.current_detections()
        frame_arrival_s = max((stamp for _, stamp in detections), default=None)
        point_cloud_sample, sync_reason = self.match_point_cloud(frame_arrival_s)
        pose = self.match_pose(point_cloud_sample, frame_arrival_s)
        if self.segmenter is not None and point_cloud_sample is not None and pose is not None:
            self.submit_outline_job(detections, frame_arrival_s, point_cloud_sample, pose)

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
        now = time.monotonic()
        self.outline_tracks = [track for track in self.outline_tracks if now - track["seen_s"] <= 1.0]
        status = self.health_status(no_depth, obstacles)
        payload = {
            "pose": self.pose_tuple(),
            "obstacles": obstacles,
            "semantic_labels": labels,
            "no_depth": no_depth,
            "low_conf": low_confidence,
            "goal": self.goal,
            # ROS time: wall time live, the bag's clock when replayed with
            # use_sim_time, so it matches the recorded Vicon stamps.
            "timestamp": self.ros_time_s(),
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

        if self.depth_estimator_mode() in ("outline", "shape_fit"):
            obstacle, reason = self.build_shape_fit_obstacle(
                points_body, x1, y1, x2, y2, label, confidence, point_cloud_sample, pose, detection
            )
            if obstacle is not None:
                if self.depth_estimator_mode() == "outline":
                    self.apply_outline(obstacle, label, (x1, y1, x2, y2), pose, point_cloud_sample["arrival_s"])
                return obstacle, None
            if reason is not None and not reason.startswith("shape_fit_fallback"):
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
        payload = self.obstacle_payload(
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
        )
        if self.depth_estimator_mode() != "front_spread":
            # The ToF fit failed: the visible spread is only a lower bound, so
            # the class prior fills in (never thinner than what was seen).
            payload["measured_depth_along_view_m"] = round(
                max(float(depth_m), float(prior_for(label)["depth"])), 3
            )
            payload["depth_method"] = "prior_fallback"
            payload["depth_confidence"] = 0.0
            payload["depth_ready"] = True
        return payload, None

    def build_shape_fit_obstacle(
        self, points_body, x1, y1, x2, y2, label, confidence, point_cloud_sample, pose, detection
    ):
        """ToF-only depth: every point in the detection box near the object's
        range (the inner-box cluster picks the range), floor removed, fitted by
        scripts/obstacle_depth.py; the box extends back from the nearest surface."""
        seed_body, _, candidate_count = self.depth_points_for_box(points_body, x1, y1, x2, y2)
        min_points = int(self.get_parameter("min_frustum_points").value)
        if len(seed_body) < min_points:
            return None, f"projected_tof_points={candidate_count}, clustered_points={len(seed_body)}"
        front_range = float(
            np.percentile(np.linalg.norm(seed_body - self.depth_camera_translation_body, axis=1), 5.0)
        )
        box_body, box_detection, box_ranges = self.points_in_full_box(points_body, x1, y1, x2, y2)
        tolerance = max(0.0, float(self.get_parameter("shape_fit_front_tolerance_m").value))
        keep = (box_ranges >= front_range - tolerance) & (box_ranges <= front_range + range_window_for(label))
        floor = self.tof_floor(point_cloud_sample, points_body)
        if floor is not None:
            keep &= floor.distance_above(box_detection) > 0.05
        box_detection = box_detection[keep]
        if len(box_detection) < min_points:
            return None, "shape_fit_fallback: too few full-box points"
        optical_depth = float(np.median(box_detection[:, 2]))
        bbox_width_m = max(0.10, (x2 - x1) * self.image_width * optical_depth / self.fx)
        bbox_height_m = max(0.10, (y2 - y1) * self.image_height * optical_depth / self.fy)
        estimate = estimate_depth(box_detection, label, min_width=bbox_width_m, min_height=bbox_height_m)
        if estimate is None:
            return None, "shape_fit_fallback: estimator rejected the points"
        rotation = self.rotation_detection_camera_to_body
        translation = self.detection_camera_translation_body
        corners_body = translation + (rotation @ estimate.corners_cam.T).T
        front_center_body = translation + rotation @ estimate.front_center_cam
        payload = self.obstacle_payload(
            corners_body.mean(axis=0),
            corners_body,
            pose,
            label,
            confidence,
            f"tof_projected_{estimate.points_used}of{len(box_ranges)}pts_{estimate.method}",
            "tflite_tof_shape_fit",
            point_cloud_sample.get("sync_delta_s"),
            str(getattr(detection, "cam", "")),
            front_center_body,
            max(bbox_width_m, estimate.width_m),
            float(np.linalg.norm(front_center_body - translation)),
        )
        payload["measured_depth_along_view_m"] = round(float(estimate.depth_m), 3)
        payload["depth_method"] = estimate.method
        payload["depth_confidence"] = round(float(estimate.confidence), 2)
        payload["depth_ready"] = True
        return payload, None

    def points_in_full_box(self, points_body, x1, y1, x2, y2):
        """Valid ToF points projecting anywhere inside the detection box."""
        points_detection = (
            self.rotation_detection_camera_to_body.T
            @ (points_body - self.detection_camera_translation_body).T
        ).T
        ranges = np.linalg.norm(points_body - self.depth_camera_translation_body, axis=1)
        valid = (
            (points_detection[:, 2] > 0.01)
            & (ranges >= float(self.get_parameter("min_tof_depth_m").value))
            & (ranges <= float(self.get_parameter("max_tof_depth_m").value))
        )
        points_body, points_detection, ranges = points_body[valid], points_detection[valid], ranges[valid]
        u = (self.fx * points_detection[:, 0] / points_detection[:, 2] + self.cx) / self.image_width
        v = (self.fy * points_detection[:, 1] / points_detection[:, 2] + self.cy) / self.image_height
        mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        return points_body[mask], points_detection[mask], ranges[mask]

    def tof_floor(self, point_cloud_sample, points_body):
        """Floor plane from this ToF frame, in the detection camera frame (cached per frame)."""
        if "floor_detection" not in point_cloud_sample:
            points_detection = (
                self.rotation_detection_camera_to_body.T
                @ (points_body - self.detection_camera_translation_body).T
            ).T
            down = self.rotation_detection_camera_to_body.T @ np.array([0.0, 0.0, 1.0])
            point_cloud_sample["floor_detection"] = od.fit_floor_plane(points_detection, down)
        return point_cloud_sample["floor_detection"]

    # ------------------------------------------------------------ outline depth
    def setup_outline(self):
        names = (
            "hires_fx", "hires_fy", "hires_cx", "hires_cy", "hires_width", "hires_height",
            "detection_cam_body_x_m", "detection_cam_body_y_m", "detection_cam_body_z_m",
            "detection_cam_roll_deg", "detection_cam_pitch_deg", "detection_cam_yaw_deg",
        )
        config = {name: float(self.get_parameter(name).value) for name in names}
        distortion = tuple(float(value) for value in self.get_parameter("hires_distortion").value)
        self.outline_camera, _, _ = od.starling_cameras(config, distortion)
        kind = str(self.get_parameter("segmenter").value).strip().lower()
        model = str(self.get_parameter("efficientsam_model").value).strip() or self.default_efficientsam_model()
        try:
            if kind not in ("efficientsam", "sam", "yolo"):
                raise ValueError(f"segmenter must be efficientsam, sam or yolo, not '{kind}'")
            self.segmenter = make_segmenter(kind, efficientsam_model=model)
            # One pass on a blank image: a model this OpenCV/onnxruntime can't run fails here, once.
            self.segmenter.segment(np.zeros((64, 64, 3), dtype=np.uint8), [("check", 1.0, (8.0, 8.0, 56.0, 56.0))])
            self.outline_status = "ok"
        except (SystemExit, Exception) as exc:  # missing model or library: keep the ToF fit
            self.segmenter = None
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            self.outline_status = f"unavailable: {reason}"
            self.get_logger().warning(
                f"outline depth unavailable ({reason}); depths use the ToF-only shape fit"
            )

    @staticmethod
    def default_efficientsam_model():
        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.append(Path(get_package_share_directory("llm_vision_planner")) / "models")
        except Exception:
            pass
        candidates += [Path(__file__).resolve().parents[1] / "models", Path.cwd() / "src" / "llm_vision_planner" / "models"]
        for folder in candidates:
            if (folder / EFFICIENTSAM_FILE).is_file():
                return str(folder / EFFICIENTSAM_FILE)
        return str(candidates[0] / EFFICIENTSAM_FILE)

    def submit_outline_job(self, detections, frame_arrival_s, point_cloud_sample, pose):
        key = (frame_arrival_s, int(point_cloud_sample.get("source_stamp_ns", 0)))
        if key == self.outline_last_key or not self.camera_images:
            return
        self.outline_last_key = key
        cloud_ns = int(point_cloud_sample.get("source_stamp_ns", 0))
        if cloud_ns > 0 and all(int(item["source_stamp_ns"]) > 0 for item in self.camera_images):
            image = min(self.camera_images, key=lambda item: abs(int(item["source_stamp_ns"]) - cloud_ns))
            gap_s = abs(int(image["source_stamp_ns"]) - cloud_ns) / 1e9
        else:
            target = float(point_cloud_sample["arrival_s"])
            image = min(self.camera_images, key=lambda item: abs(float(item["arrival_s"]) - target))
            gap_s = abs(float(image["arrival_s"]) - target)
        if gap_s > float(self.get_parameter("outline_max_image_gap_s").value):
            self.outline_status = f"no camera image within {gap_s:.2f} s of the ToF frame"
            return
        prompts = []
        for detection, _ in detections:
            confidence = self.detection_confidence(detection)
            if confidence < float(self.get_parameter("min_confidence").value):
                continue
            if not self.detection_camera_matches(getattr(detection, "cam", "")):
                continue
            box = self.normalized_box(detection)
            if box[2] > box[0] and box[3] > box[1]:
                prompts.append((str(detection.class_name), confidence, box))
        # TFLite often reports one object several times; outline it once.
        unique = []
        for prompt in sorted(prompts, key=lambda item: -item[1]):
            if all(
                canonical_label(prompt[0]) != canonical_label(kept[0]) or box_iou(prompt[2], kept[2]) < 0.6
                for kept in unique
            ):
                unique.append(prompt)
        prompts = unique
        if not prompts:
            return
        points_body, _ = self.point_cloud_points_body(point_cloud_sample, pose)
        if points_body is None:
            return
        # When the newer of the two was taken: an outline only counts for an
        # object if this frame was taken after the object last moved.
        source_s = min(float(point_cloud_sample["arrival_s"]), float(image["arrival_s"]))
        job = {"image": image["msg"], "points_body": points_body, "pose": pose, "prompts": prompts,
               "source_s": source_s}
        if bool(self.get_parameter("outline_async").value):
            with self.outline_lock:
                self.outline_job = job   # only the newest frame waits
            self.outline_event.set()
        else:
            self.run_outline_job(job)

    def outline_worker(self):
        while not self.outline_stop.is_set():
            if not self.outline_event.wait(0.5):
                continue
            self.outline_event.clear()
            with self.outline_lock:
                job, self.outline_job = self.outline_job, None
            if job is not None:
                self.run_outline_job(job)

    def run_outline_job(self, job):
        try:
            started = time.perf_counter()
            rgb = bag_reader.image_to_rgb(job["image"])
            height, width = rgb.shape[:2]
            camera = self.outline_camera.scaled_to(width, height)
            points_cam = camera.body_to_cam(job["points_body"])
            rotation_deg = [float(value) for value in self.get_parameter("outline_tof_rotation_deg").value]
            if any(rotation_deg):
                points_cam = points_cam @ od.small_rotation(*rotation_deg).T
            floor = od.fit_floor_plane(points_cam, camera.expected_down())
            prompts_px = [
                (label, confidence, (box[0] * width, box[1] * height, box[2] * width, box[3] * height))
                for label, confidence, box in job["prompts"]
            ]
            masks = self.segmenter.segment(rgb, prompts_px)
            results = []
            for (label, _, box), (_, _, box_px) in zip(job["prompts"], prompts_px):
                scored = [
                    (box_iou(mask.box or mask_box(mask.mask), box_px), mask)
                    for mask in masks
                    if canonical_label(mask.label) == canonical_label(label)
                ]
                scored = [item for item in scored if item[0] >= 0.3]
                if not scored:
                    continue
                mask = max(scored, key=lambda item: item[0])[1].mask
                result = od.estimate_from_outline(label, mask, camera, points_cam, floor=floor)
                if result is None:
                    continue
                results.append(
                    {
                        "label": canonical_label(label),
                        "box": box,
                        "source_s": job["source_s"],
                        **self.outline_geometry(result, camera, job["pose"]),
                    }
                )
            max_age = float(self.get_parameter("outline_max_age_s").value)
            now = time.monotonic()
            with self.outline_lock:
                kept = [item for item in self.outline_results if now - item["source_s"] <= max_age]
                self.outline_results = (kept + results)[-32:]
            self.outline_status = f"ok ({1000.0 * (time.perf_counter() - started):.0f} ms per frame)"
        except Exception as exc:  # never let one bad frame stop perception
            self.outline_status = f"error: {type(exc).__name__}: {exc}"
            self.get_logger().warning(f"outline depth failed on a frame: {self.outline_status}")

    @staticmethod
    def outline_geometry(result, camera, pose):
        """The outline box (level frame at the camera) in local NED."""
        level, box = result.level, result.box_level
        front_level = np.array(
            [[0.5 * (box["x0"] + box["x1"]), 0.5 * (box["y_top"] + box["y_bottom"]), box["z0"]]]
        )
        view_cam = level.to_cam(np.array([[0.0, 0.0, 1.0]]))[0]
        rotation, position = pose["rotation_body_to_world"], pose["position"]
        corners_world = position + (rotation @ camera.cam_to_body(result.corners_cam).T).T
        front_world = position + rotation @ camera.cam_to_body(level.to_cam(front_level))[0]
        view_xy = (rotation @ camera.rotation_cam_to_body @ view_cam)[:2]
        view_xy = view_xy / max(1e-9, float(np.linalg.norm(view_xy)))
        camera_world = position + rotation @ camera.translation_cam_body
        return {
            "corners_world": corners_world,
            "front_world": front_world,
            "view_axis_xy": view_xy,
            "visible_width_m": float(box["x1"] - box["x0"]),
            "depth_m": float(result.depth_m),
            "front_range_m": float(np.linalg.norm(front_world - camera_world)),
            "decided_by": str(result.decided_by),
            "depth_observed": bool(result.depth_observed),
        }

    def outline_track(self, name, box, front_xy, frame_s):
        """This object across frames: when its ToF front last moved, and since
        when it has been waiting for an outline. After a move, only an outline
        whose camera image and ToF frame both came after the ToF frame that
        showed the move counts. One track per object, so two people don't
        share a state; a track unseen for 1 s is dropped."""
        settle = float(self.get_parameter("outline_settle_m").value)
        scored = [
            (box_iou(track["box"], box), track)
            for track in self.outline_tracks
            if track["label"] == name and id(track) not in self.outline_used
        ]
        scored = [item for item in scored if item[0] >= 0.3]
        if scored:
            track = max(scored, key=lambda item: item[0])[1]
            if math.dist(front_xy, track["anchor_xy"]) > settle:
                track["anchor_xy"], track["moved_s"] = front_xy, frame_s
        else:
            # Nothing known about where it was before: the image paired with
            # this ToF frame (up to outline_max_image_gap_s older) counts too.
            gap = float(self.get_parameter("outline_max_image_gap_s").value)
            track = {"label": name, "anchor_xy": front_xy, "moved_s": frame_s - gap, "pending_s": None}
            self.outline_tracks.append(track)
        track["box"], track["seen_s"] = box, time.monotonic()
        self.outline_used.add(id(track))
        return track

    def apply_outline(self, payload, label, box, pose, frame_s):
        """Replace the ToF-only geometry with the newest outline estimate made
        from a frame taken after the object last moved."""
        now = time.monotonic()
        name = canonical_label(label)
        track = self.outline_track(name, box, [float(v) for v in payload["front_surface_center"][:2]], frame_s)
        max_age = float(self.get_parameter("outline_max_age_s").value)
        with self.outline_lock:
            candidates = [
                item
                for item in self.outline_results
                if item["label"] == name
                and box_iou(item["box"], box) >= 0.3
                and now - item["source_s"] <= max_age
                and item["source_s"] >= track["moved_s"]
            ]
        if not candidates:
            if self.segmenter is None:
                payload["depth_ready"] = True   # the ToF fit is all there is
                return
            if track["pending_s"] is None:
                track["pending_s"] = now
            payload["depth_ready"] = now - track["pending_s"] >= float(self.get_parameter("outline_wait_s").value)
            payload["outline_pending"] = True
            return
        track["pending_s"] = None
        result = max(candidates, key=lambda item: item["source_s"])
        corners = result["corners_world"]
        centroid = corners.mean(axis=0)
        delta = centroid - pose["position"]
        view_xy = result["view_axis_xy"]
        payload.update(
            {
                "centroid": np.round(centroid, 2).tolist(),
                "min_corner": np.round(corners.min(axis=0), 2).tolist(),
                "max_corner": np.round(corners.max(axis=0), 2).tolist(),
                "size": np.round(corners.max(axis=0) - corners.min(axis=0), 2).tolist(),
                "distance_m": round(float(np.linalg.norm(delta)), 2),
                "bearing_deg": round(float(np.degrees(math.atan2(delta[1], delta[0]))), 1),
                "front_surface_center": np.round(result["front_world"], 3).tolist(),
                "front_range_m": round(result["front_range_m"], 3),
                "visible_width_m": round(result["visible_width_m"], 3),
                "view_axis_xy": np.round(view_xy, 6).tolist(),
                "lateral_axis_xy": np.round([-view_xy[1], view_xy[0]], 6).tolist(),
                "measured_depth_along_view_m": round(result["depth_m"], 3),
                "depth_method": f"outline_{result['decided_by']}",
                "depth_confidence": 1.0 if result["depth_observed"] else 0.5,
                "depth_source": payload["depth_source"].rsplit("pts_", 1)[0] + f"pts_outline_{result['decided_by']}",
                "source": "tflite_tof_outline",
                "depth_ready": True,
                "outline_age_s": round(now - result["source_s"], 2),
            }
        )

    def destroy_node(self):
        self.outline_stop.set()
        self.outline_event.set()
        return super().destroy_node()

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
        payload = self.obstacle_payload(
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
        payload["measured_depth_along_view_m"] = round(float(depth_m), 3)
        payload["depth_method"] = "class_prior"
        payload["depth_confidence"] = 0.0
        return payload

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
            "depth_estimator": self.depth_estimator_mode(),
            "outline": self.outline_status,
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
                f"depth={obstacle.get('measured_depth_along_view_m')}m ({obstacle.get('depth_method', 'front_spread')}) "
                f"prompt='{self.prompt_obstacle_text(obstacle)}'"
            )
        self.get_logger().info("\n".join(lines))

    def depth_estimator_mode(self):
        mode = str(self.get_parameter("depth_estimator").value).strip().lower()
        return mode if mode in ("outline", "shape_fit", "front_spread") else "outline"

    def ros_time_s(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        return now if now > 0.0 else time.time()

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
