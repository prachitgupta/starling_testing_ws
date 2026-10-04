#!/usr/bin/env python3
"""Record raw perception/Vicon pairs for residual conformal calibration.

This node intentionally does not compute a conformity score.  Hardware
collection records only the synchronized environment estimate ``E_hat`` and
ground-truth environment ``E``.  ``postprocess_residual_calibration.py`` later
runs HRRT-star, the Llama planner, the shared QP, and the residual score.
"""

import csv
import json
import math
import re
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


CSV_DELIMITER = ","
RAW_FIELDS = [
    "session_id",
    "capture_id",
    "capture_index",
    "timestamp_s",
    "vicon_timestamp_s",
    "object_id",
    "label",
    "pred_min_x",
    "pred_min_y",
    "pred_max_x",
    "pred_max_y",
    "gt_min_x",
    "gt_min_y",
    "gt_max_x",
    "gt_max_y",
    "missed_detection",
    "stable_pose",
    "gt_center_x",
    "gt_center_y",
    "gt_yaw_rad",
    "gt_corner_0_x", "gt_corner_0_y",
    "gt_corner_1_x", "gt_corner_1_y",
    "gt_corner_2_x", "gt_corner_2_y",
    "gt_corner_3_x", "gt_corner_3_y",
    "pred_front_center_x", "pred_front_center_y",
    "pred_view_axis_x", "pred_view_axis_y",
    "pred_lateral_axis_x", "pred_lateral_axis_y",
    "pred_visible_width_m",
    "pred_chatgpt_depth_m",
    "pred_corner_0_x", "pred_corner_0_y",
    "pred_corner_1_x", "pred_corner_1_y",
    "pred_corner_2_x", "pred_corner_2_y",
    "pred_corner_3_x", "pred_corner_3_y",
    "observer_x",
    "observer_y",
    "observer_z",
    "placeholder",
]

ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=20,
)
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def quaternion_matrix(values):
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion must not be zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_pose(message):
    translation = message.transform.translation
    rotation = message.transform.rotation
    return (
        np.array([translation.x, translation.y, translation.z], dtype=float),
        quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w]),
    )


def px4_pose(message):
    if message.pose_frame != VehicleOdometry.POSE_FRAME_NED:
        raise ValueError("PX4 odometry must use the NED pose frame")
    q = list(message.q)
    return (
        np.asarray(message.position[:3], dtype=float),
        quaternion_matrix([q[1], q[2], q[3], q[0]]),
    )


def marker_from_body(convention):
    if str(convention).lower() == "flu":
        return np.diag([1.0, -1.0, -1.0])
    if str(convention).lower() == "frd":
        return np.eye(3)
    raise ValueError("vicon_vehicle_frame_convention must be 'flu' or 'frd'")


def world_to_ned_candidate(vicon_vehicle_pose, px4_vehicle_pose, convention):
    vicon_position, vicon_rotation = vicon_vehicle_pose
    ned_position, body_to_ned = px4_vehicle_pose
    body_to_vicon_world = vicon_rotation @ marker_from_body(convention)
    rotation = body_to_ned @ body_to_vicon_world.T
    translation = ned_position - rotation @ vicon_position
    return translation, rotation


def average_transforms(candidates):
    translations = np.asarray([item[0] for item in candidates], dtype=float)
    rotations = np.asarray([item[1] for item in candidates], dtype=float)
    left, _, right = np.linalg.svd(np.sum(rotations, axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    translation = np.mean(translations, axis=0)
    translation_error = float(np.max(np.linalg.norm(translations - translation, axis=1)))
    rotation_error = 0.0
    for sample in rotations:
        cosine = min(1.0, max(-1.0, 0.5 * (float(np.trace(rotation.T @ sample)) - 1.0)))
        rotation_error = max(rotation_error, math.degrees(math.acos(cosine)))
    return (translation, rotation), translation_error, rotation_error


def parse_objects(value):
    raw = json.loads(value)
    if not isinstance(raw, list) or not raw:
        raise ValueError("vicon_objects_json must be a non-empty JSON list")
    parsed = []
    for item in raw:
        dimensions = np.asarray(item.get("dimensions_m", []), dtype=float)
        if dimensions.shape not in ((2,), (3,)) or np.any(dimensions <= 0):
            raise ValueError("each Vicon object requires positive dimensions_m")
        parsed.append(
            {
                "object_id": str(item["object_id"]),
                "label": str(item["label"]),
                "topic": str(item["topic"]),
                "dimensions": dimensions,
            }
        )
    return parsed


def ground_truth_aabb(config, marker_pose, world_to_ned):
    marker_position, marker_rotation = marker_pose
    translation, rotation = world_to_ned
    center = translation + rotation @ marker_position
    object_rotation = rotation @ marker_rotation
    width, depth = [float(value) for value in config["dimensions"][:2]]
    local = np.array(
        [[-width / 2, -depth / 2, 0], [-width / 2, depth / 2, 0],
         [width / 2, -depth / 2, 0], [width / 2, depth / 2, 0]],
        dtype=float,
    )
    corners = center + (object_rotation @ local.T).T
    return {
        "min_corner": [float(np.min(corners[:, 0])), float(np.min(corners[:, 1]))],
        "max_corner": [float(np.max(corners[:, 0])), float(np.max(corners[:, 1]))],
        "center": [float(center[0]), float(center[1])],
        "yaw": math.atan2(float(object_rotation[1, 0]), float(object_rotation[0, 0])),
        "corners": [[float(point[0]), float(point[1])] for point in corners],
    }


def angle_difference(first, second):
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def match_prediction(obstacles, config, ground_truth, maximum_distance):
    candidates = []
    for obstacle in obstacles:
        if str(obstacle.get("label", obstacle.get("shape", ""))).lower() != config["label"].lower():
            continue
        minimum = obstacle.get("min_corner", [])
        maximum = obstacle.get("max_corner", [])
        if len(minimum) < 2 or len(maximum) < 2:
            continue
        center = [0.5 * (float(minimum[0]) + float(maximum[0])),
                  0.5 * (float(minimum[1]) + float(maximum[1]))]
        distance = math.dist(center, ground_truth["center"])
        if distance <= maximum_distance:
            candidates.append((str(obstacle.get("object_id")) != config["object_id"], distance, obstacle))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2] if candidates else None


class ResidualCalibrationRecorder(Node):
    """Synchronize each perceived obstacle footprint with Vicon ground truth."""

    def __init__(self):
        super().__init__("residual_calibration_recorder")
        self.declare_parameter("nominal_obstacle_topic", "/llm_vision/nominal_obstacles")
        self.declare_parameter("calibration_status_topic", "/llm_vision/vision_calibration_status")
        self.declare_parameter("trial_id", "unset")
        self.declare_parameter("output_csv", "fine_tuning/datasets/calibration_residual_raw.csv")
        self.declare_parameter("vicon_objects_json", "[]")
        self.declare_parameter("object_id", "obj-1")
        self.declare_parameter("object_label", "chair")
        self.declare_parameter("object_vicon_topic", "/vicon/chair1/chair1")
        self.declare_parameter("object_width_m", 0.0)
        self.declare_parameter("object_depth_m", 0.0)
        self.declare_parameter("vicon_vehicle_topic", "/vicon/Starling2/Starling2")
        self.declare_parameter("vicon_vehicle_frame_convention", "flu")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("frame_calibration_samples", 20)
        self.declare_parameter("frame_sync_tolerance_s", 0.10)
        self.declare_parameter("sync_tolerance_s", 0.10)
        self.declare_parameter("match_distance_m", 0.75)
        self.declare_parameter("object_stability_window_s", 0.50)
        self.declare_parameter("object_stability_min_samples", 5)
        self.declare_parameter("object_stability_position_tolerance_m", 0.02)
        self.declare_parameter("capture_position_change_threshold_m", 0.15)
        self.declare_parameter("capture_yaw_change_threshold_rad", 0.261799)

        self.session_id = str(self.get_parameter("trial_id").value).strip()
        if not self.session_id or self.session_id.lower() in ("unset", "placeholder"):
            raise ValueError("trial_id must identify this independent hardware session")
        self.frame_count = int(self.get_parameter("frame_calibration_samples").value)
        self.frame_sync_tolerance = float(self.get_parameter("frame_sync_tolerance_s").value)
        self.sync_tolerance = float(self.get_parameter("sync_tolerance_s").value)
        self.match_distance = float(self.get_parameter("match_distance_m").value)
        self.stability_window = float(self.get_parameter("object_stability_window_s").value)
        self.stability_min_samples = int(self.get_parameter("object_stability_min_samples").value)
        self.stability_position_tolerance = float(
            self.get_parameter("object_stability_position_tolerance_m").value
        )
        self.capture_position_threshold = float(
            self.get_parameter("capture_position_change_threshold_m").value
        )
        self.capture_yaw_threshold = float(
            self.get_parameter("capture_yaw_change_threshold_rad").value
        )
        if self.capture_position_threshold <= 0.0 or self.capture_yaw_threshold <= 0.0:
            raise ValueError("capture position/yaw change thresholds must be positive")
        self.vehicle_convention = str(self.get_parameter("vicon_vehicle_frame_convention").value)
        marker_from_body(self.vehicle_convention)

        configured = json.loads(str(self.get_parameter("vicon_objects_json").value))
        if configured:
            self.objects = parse_objects(json.dumps(configured))
        else:
            self.objects = parse_objects(
                json.dumps([{
                    "object_id": str(self.get_parameter("object_id").value),
                    "label": str(self.get_parameter("object_label").value),
                    "topic": str(self.get_parameter("object_vicon_topic").value),
                    "dimensions_m": [float(self.get_parameter("object_width_m").value),
                                     float(self.get_parameter("object_depth_m").value)],
                }])
            )

        output = Path(str(self.get_parameter("output_csv").value)).expanduser()
        self.output_csv = output if output.is_absolute() else Path.cwd() / output
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        existed = self.output_csv.exists() and self.output_csv.stat().st_size > 0
        self.stream = self.output_csv.open("a+", newline="", encoding="utf-8")
        if existed:
            self.stream.seek(0)
            if (csv.DictReader(self.stream, delimiter=CSV_DELIMITER).fieldnames or []) != RAW_FIELDS:
                self.stream.close()
                raise ValueError(f"incompatible existing raw CSV header: {self.output_csv}")
            self.stream.seek(0, 2)
        self.writer = csv.DictWriter(
            self.stream, fieldnames=RAW_FIELDS, delimiter=CSV_DELIMITER, lineterminator="\n"
        )
        if not existed:
            self.writer.writeheader()
            self.stream.flush()

        self.capture_index = 0
        self.latest_nominal = None
        self.world_to_ned = None
        self.frame_candidates = deque(maxlen=max(80, self.frame_count * 4))
        self.used_frame_pairs = set()
        self.vehicle_vicon = deque(maxlen=100)
        self.vehicle_px4 = deque(maxlen=100)
        self.object_history = {item["topic"]: deque(maxlen=500) for item in self.objects}
        self.recorded = set()
        self.pending_rows = {}
        self.rejected_captures = set()
        self.last_accepted_scene = None
        self.status = self.create_publisher(
            String, str(self.get_parameter("calibration_status_topic").value), LATCHED_QOS
        )
        self.create_subscription(
            String, str(self.get_parameter("nominal_obstacle_topic").value),
            self.nominal_callback, LATCHED_QOS
        )
        self.create_subscription(
            TransformStamped, str(self.get_parameter("vicon_vehicle_topic").value),
            self.vehicle_vicon_callback, 20
        )
        self.create_subscription(
            VehicleOdometry, str(self.get_parameter("pose_topic").value),
            self.vehicle_px4_callback, ODOM_QOS
        )
        self.object_subscriptions = [
            self.create_subscription(
                TransformStamped, item["topic"],
                lambda message, configured=item: self.object_callback(configured, message), 20
            )
            for item in self.objects
        ]
        self.publish_status("WAITING_FOR_FRAME_ALIGNMENT", "Keep Starling 2 still while frames align.")

    def publish_status(self, status, message, **metadata):
        payload = {"status": status, "message": message, "session_id": self.session_id,
                   "timestamp": time.time(), **metadata}
        self.status.publish(String(data=json.dumps(payload)))

    def vehicle_vicon_callback(self, message):
        try:
            self.vehicle_vicon.append((time.monotonic(), transform_pose(message)))
        except ValueError:
            return
        self.try_align_frames()

    def vehicle_px4_callback(self, message):
        try:
            self.vehicle_px4.append((time.monotonic(), px4_pose(message)))
        except ValueError:
            return
        self.try_align_frames()

    def try_align_frames(self):
        if self.world_to_ned is not None or not self.vehicle_vicon or not self.vehicle_px4:
            return
        vicon_time, vicon_pose = self.vehicle_vicon[-1]
        px4_time, px4_value = min(self.vehicle_px4, key=lambda item: abs(item[0] - vicon_time))
        if abs(px4_time - vicon_time) > self.frame_sync_tolerance:
            return
        pair_key = (vicon_time, px4_time)
        if pair_key in self.used_frame_pairs:
            return
        self.used_frame_pairs.add(pair_key)
        self.frame_candidates.append(
            world_to_ned_candidate(vicon_pose, px4_value, self.vehicle_convention)
        )
        if len(self.frame_candidates) < self.frame_count:
            return
        transform, translation_error, rotation_error = average_transforms(
            list(self.frame_candidates)[-self.frame_count:]
        )
        if translation_error > 0.15 or rotation_error > 5.0:
            return
        self.world_to_ned = transform
        self.publish_status(
            "FRAME_READY", "Frame alignment complete; raw captures are enabled.",
            alignment_translation_error_m=translation_error,
            alignment_rotation_error_deg=rotation_error,
        )

    def nominal_callback(self, message):
        try:
            payload = json.loads(message.data)
            timestamp = float(payload["timestamp"])
            if not isinstance(payload.get("obstacles"), list):
                raise ValueError("missing obstacles")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"ignored invalid nominal-obstacle payload: {exc}")
            return
        self.capture_index += 1
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.session_id)
        self.latest_nominal = {
            "payload": payload,
            "timestamp": timestamp,
            "capture_id": f"{safe}-capture-{self.capture_index:06d}",
        }
        self.pending_rows = {self.latest_nominal["capture_id"]: {}}
        for configured in self.objects:
            self.try_record(configured)

    def object_callback(self, configured, message):
        timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        if timestamp <= 0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        try:
            pose = transform_pose(message)
        except ValueError:
            return
        self.object_history[configured["topic"]].append((timestamp, pose))
        self.try_record(configured)

    @staticmethod
    def observer_fields(payload):
        pose = payload.get("observer_pose", [])
        if not isinstance(pose, (list, tuple)) or len(pose) < 3:
            return {name: "" for name in ("observer_x", "observer_y", "observer_z")}
        return {
            "observer_x": f"{float(pose[0]):.9f}",
            "observer_y": f"{float(pose[1]):.9f}",
            "observer_z": f"{float(pose[2]):.9f}",
        }

    def try_record(self, configured):
        if self.latest_nominal is None or self.world_to_ned is None:
            return
        history = self.object_history[configured["topic"]]
        if not history:
            return
        nominal = self.latest_nominal
        if nominal["capture_id"] in self.rejected_captures:
            return
        half_window = 0.5 * self.stability_window
        if history[-1][0] < nominal["timestamp"] + half_window:
            return
        stability_samples = [
            ground_truth_aabb(configured, pose, self.world_to_ned)
            for timestamp, pose in history
            if abs(timestamp - nominal["timestamp"]) <= half_window
        ]
        if len(stability_samples) < self.stability_min_samples:
            return
        centers = np.asarray([sample["center"] for sample in stability_samples], dtype=float)
        center_deviation = float(np.max(np.linalg.norm(centers - np.mean(centers, axis=0), axis=1)))
        if center_deviation > self.stability_position_tolerance:
            self.rejected_captures.add(nominal["capture_id"])
            self.pending_rows.pop(nominal["capture_id"], None)
            self.publish_status(
                "SKIPPED_MOVING_OBJECT",
                f"Skipped moving object {configured['object_id']} for {nominal['capture_id']}.",
                capture_id=nominal["capture_id"], object_id=configured["object_id"],
            )
            return
        vicon_timestamp, marker_pose = min(
            history, key=lambda item: abs(item[0] - nominal["timestamp"])
        )
        if abs(vicon_timestamp - nominal["timestamp"]) > self.sync_tolerance:
            return
        key = (nominal["capture_id"], configured["object_id"])
        if key in self.recorded:
            return
        ground_truth = ground_truth_aabb(configured, marker_pose, self.world_to_ned)
        predicted = match_prediction(
            nominal["payload"]["obstacles"], configured, ground_truth, self.match_distance
        )
        row = {
            "session_id": self.session_id,
            "capture_id": nominal["capture_id"],
            "capture_index": str(self.capture_index),
            "timestamp_s": f"{nominal['timestamp']:.9f}",
            "vicon_timestamp_s": f"{vicon_timestamp:.9f}",
            "object_id": configured["object_id"],
            "label": configured["label"],
            "gt_min_x": f"{ground_truth['min_corner'][0]:.9f}",
            "gt_min_y": f"{ground_truth['min_corner'][1]:.9f}",
            "gt_max_x": f"{ground_truth['max_corner'][0]:.9f}",
            "gt_max_y": f"{ground_truth['max_corner'][1]:.9f}",
            "missed_detection": str(predicted is None).lower(),
            "stable_pose": "true",
            "gt_center_x": f"{ground_truth['center'][0]:.9f}",
            "gt_center_y": f"{ground_truth['center'][1]:.9f}",
            "gt_yaw_rad": f"{ground_truth['yaw']:.9f}",
            "placeholder": "false",
            **self.observer_fields(nominal["payload"]),
        }
        for index, point in enumerate(ground_truth["corners"]):
            row[f"gt_corner_{index}_x"] = f"{point[0]:.9f}"
            row[f"gt_corner_{index}_y"] = f"{point[1]:.9f}"
        for prefix, source in (("pred", predicted),):
            minimum = source.get("min_corner", []) if source else []
            maximum = source.get("max_corner", []) if source else []
            row[f"{prefix}_min_x"] = f"{float(minimum[0]):.9f}" if len(minimum) >= 2 else ""
            row[f"{prefix}_min_y"] = f"{float(minimum[1]):.9f}" if len(minimum) >= 2 else ""
            row[f"{prefix}_max_x"] = f"{float(maximum[0]):.9f}" if len(maximum) >= 2 else ""
            row[f"{prefix}_max_y"] = f"{float(maximum[1]):.9f}" if len(maximum) >= 2 else ""
        perceived_fields = {
            "pred_front_center_x": ("front_surface_center", 0),
            "pred_front_center_y": ("front_surface_center", 1),
            "pred_view_axis_x": ("view_axis_xy", 0),
            "pred_view_axis_y": ("view_axis_xy", 1),
            "pred_lateral_axis_x": ("lateral_axis_xy", 0),
            "pred_lateral_axis_y": ("lateral_axis_xy", 1),
        }
        for field, (source_field, index) in perceived_fields.items():
            values = predicted.get(source_field, []) if predicted else []
            row[field] = f"{float(values[index]):.9f}" if len(values) > index else ""
        row["pred_visible_width_m"] = (
            f"{float(predicted['visible_width_m']):.9f}"
            if predicted and predicted.get("visible_width_m") is not None else ""
        )
        row["pred_chatgpt_depth_m"] = (
            f"{float(predicted['effective_depth_along_view_m']):.9f}"
            if predicted and predicted.get("effective_depth_along_view_m") is not None else ""
        )
        predicted_corners = predicted.get("nominal_footprint_corners_xy", []) if predicted else []
        for index in range(4):
            point = predicted_corners[index] if len(predicted_corners) > index else []
            row[f"pred_corner_{index}_x"] = f"{float(point[0]):.9f}" if len(point) >= 2 else ""
            row[f"pred_corner_{index}_y"] = f"{float(point[1]):.9f}" if len(point) >= 2 else ""

        pending = self.pending_rows.setdefault(nominal["capture_id"], {})
        pending[configured["object_id"]] = (row, ground_truth)
        if len(pending) != len(self.objects):
            return
        scene = {
            object_id: {"center": value[1]["center"], "yaw": value[1]["yaw"]}
            for object_id, value in pending.items()
        }
        different = self.last_accepted_scene is None or set(scene) != set(self.last_accepted_scene)
        if not different:
            for object_id, pose in scene.items():
                previous = self.last_accepted_scene[object_id]
                if (
                    math.dist(pose["center"], previous["center"])
                    >= self.capture_position_threshold
                    or angle_difference(pose["yaw"], previous["yaw"])
                    >= self.capture_yaw_threshold
                ):
                    different = True
                    break
        if not different:
            self.rejected_captures.add(nominal["capture_id"])
            self.pending_rows.pop(nominal["capture_id"], None)
            self.publish_status(
                "SKIPPED_DUPLICATE_ENVIRONMENT",
                "No tracked object exceeded the configured position or yaw threshold.",
                capture_id=nominal["capture_id"],
                position_threshold_m=self.capture_position_threshold,
                yaw_threshold_rad=self.capture_yaw_threshold,
            )
            return
        for item in self.objects:
            object_id = item["object_id"]
            self.writer.writerow(pending[object_id][0])
            self.recorded.add((nominal["capture_id"], object_id))
        self.stream.flush()
        self.last_accepted_scene = scene
        self.pending_rows.pop(nominal["capture_id"], None)
        self.publish_status(
            "RECORDED", f"Recorded complete environment {nominal['capture_id']}.",
            capture_id=nominal["capture_id"], object_count=len(scene),
            output_csv=str(self.output_csv),
        )

    def destroy_node(self):
        if hasattr(self, "stream") and not self.stream.closed:
            self.stream.flush()
            self.stream.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = ResidualCalibrationRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
