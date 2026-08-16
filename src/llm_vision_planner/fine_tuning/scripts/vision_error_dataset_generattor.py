#!/usr/bin/env python3
"""Record nominal obstacle footprints against synchronized Vicon ground truth."""

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


CSV_FIELDS = [
    "trial_id",
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
    "score_m",
    "missed_detection",
    "placeholder",
]
RAW_CSV_FIELDS = CSV_FIELDS + [
    "session_id",
    "capture_id",
    "capture_index",
    "raw_continuous",
    "stable_pose",
    "gt_center_x",
    "gt_center_y",
    "gt_yaw_rad",
    "observer_x",
    "observer_y",
    "observer_z",
    "observer_yaw_rad",
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


def quaternion_matrix(values, name="quaternion_xyzw"):
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError(f"{name} must not be zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rigid_transform(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    translation = np.asarray(value.get("translation_m"), dtype=float)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError(f"{name}.translation_m must contain three finite values")
    rotation = quaternion_matrix(value.get("quaternion_xyzw"), f"{name}.quaternion_xyzw")
    return translation, rotation


def matrix_to_quaternion(rotation):
    """Return a normalized x-y-z-w quaternion for a proper rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = [
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            values = [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                      (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale]
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            values = [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                      (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale]
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            values = [(matrix[0, 2] + matrix[2, 0]) / scale,
                      (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                      (matrix[1, 0] - matrix[0, 1]) / scale]
    quaternion = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("rotation produced a zero quaternion")
    return (quaternion / norm).tolist()


def marker_from_body_rotation(convention):
    """Known marker-frame coordinates of PX4 body FRD axes."""
    normalized = str(convention).strip().lower()
    if normalized == "flu":
        return np.diag([1.0, -1.0, -1.0])
    if normalized == "frd":
        return np.eye(3)
    raise ValueError("vicon_vehicle_frame_convention must be 'flu' or 'frd'")


def world_to_ned_candidate(vicon_pose, ned_pose, convention="flu"):
    """Derive Vicon-world -> NED from one synchronized vehicle pose pair."""
    vicon_position, vicon_rotation = vicon_pose
    ned_position, body_to_ned_rotation = ned_pose
    marker_to_body = marker_from_body_rotation(convention)
    body_to_vicon_world = np.asarray(vicon_rotation, dtype=float) @ marker_to_body
    world_to_ned_rotation = np.asarray(body_to_ned_rotation, dtype=float) @ body_to_vicon_world.T
    world_to_ned_translation = (
        np.asarray(ned_position, dtype=float)
        - world_to_ned_rotation @ np.asarray(vicon_position, dtype=float)
    )
    return world_to_ned_translation, world_to_ned_rotation


def aggregate_world_to_ned(candidates):
    """Average rigid transforms and report their worst deviation from the mean."""
    if not candidates:
        raise ValueError("at least one frame-transform candidate is required")
    translations = np.asarray([item[0] for item in candidates], dtype=float)
    rotations = np.asarray([item[1] for item in candidates], dtype=float)
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    mean_rotation = u @ correction @ vt
    mean_translation = np.mean(translations, axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)
    rotation_errors = []
    for rotation in rotations:
        cosine = min(1.0, max(-1.0, 0.5 * (float(np.trace(mean_rotation.T @ rotation)) - 1.0)))
        rotation_errors.append(math.degrees(math.acos(cosine)))
    return (mean_translation, mean_rotation), {
        "sample_count": len(candidates),
        "max_translation_deviation_m": float(np.max(translation_errors)),
        "max_rotation_deviation_deg": float(np.max(rotation_errors)),
    }


def parse_vicon_objects(value):
    try:
        raw_objects = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"vicon_objects_json is invalid JSON: {exc}") from exc
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("vicon_objects_json must contain at least one configured object")
    configured = []
    used_topics = set()
    for index, raw in enumerate(raw_objects, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"vicon object {index} must be a JSON object")
        object_id = str(raw.get("object_id", "")).strip()
        label = str(raw.get("label", "")).strip()
        topic = str(raw.get("topic", "")).strip()
        dimensions = np.asarray(raw.get("dimensions_m"), dtype=float)
        if not object_id or not label or not topic.startswith("/"):
            raise ValueError(f"vicon object {index} requires object_id, label, and an absolute topic")
        if topic in used_topics:
            raise ValueError(f"duplicate Vicon topic: {topic}")
        if dimensions.shape not in ((2,), (3,)) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
            raise ValueError(f"vicon object {index} dimensions_m must contain two or three positive values")
        marker_transform = raw.get("marker_to_object")
        if marker_transform in (None, {}):
            marker_translation, marker_rotation = np.zeros(3), np.eye(3)
        else:
            marker_translation, marker_rotation = rigid_transform(
                marker_transform,
                f"vicon object {index}.marker_to_object",
            )
        configured.append(
            {
                "object_id": object_id,
                "label": label,
                "topic": topic,
                "dimensions_m": dimensions,
                "marker_translation": marker_translation,
                "marker_rotation": marker_rotation,
            }
        )
        used_topics.add(topic)
    return configured


def transform_message_pose(msg):
    translation = msg.transform.translation
    rotation = msg.transform.rotation
    position = np.array([translation.x, translation.y, translation.z], dtype=float)
    matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w], "Vicon quaternion")
    if not np.all(np.isfinite(position)):
        raise ValueError("Vicon translation is non-finite")
    return position, matrix


def px4_message_pose(msg):
    if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
        raise ValueError("PX4 odometry pose frame is not NED")
    position = np.asarray(msg.position[:3], dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("PX4 NED position is non-finite")
    quaternion = list(msg.q)
    rotation = quaternion_matrix(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]],
        "PX4 body-to-NED quaternion",
    )
    return position, rotation


def ground_truth_aabb(config, marker_position, marker_rotation, world_to_ned):
    world_to_ned_translation, world_to_ned_rotation = world_to_ned
    object_position_world = marker_position + marker_rotation @ config["marker_translation"]
    object_rotation_world = marker_rotation @ config["marker_rotation"]
    object_position_ned = world_to_ned_translation + world_to_ned_rotation @ object_position_world
    object_rotation_ned = world_to_ned_rotation @ object_rotation_world
    width, depth = (float(value) for value in config["dimensions_m"][:2])
    corners_object = np.array(
        [
            [-0.5 * width, -0.5 * depth, 0.0],
            [-0.5 * width, 0.5 * depth, 0.0],
            [0.5 * width, -0.5 * depth, 0.0],
            [0.5 * width, 0.5 * depth, 0.0],
        ],
        dtype=float,
    )
    corners_ned = object_position_ned + (object_rotation_ned @ corners_object.T).T
    return {
        "min_corner": [float(np.min(corners_ned[:, 0])), float(np.min(corners_ned[:, 1]))],
        "max_corner": [float(np.max(corners_ned[:, 0])), float(np.max(corners_ned[:, 1]))],
        "center_xy": [float(object_position_ned[0]), float(object_position_ned[1])],
        "yaw_rad": math.atan2(float(object_rotation_ned[1, 0]), float(object_rotation_ned[0, 0])),
    }


def wrapped_angle_difference(first, second):
    return abs(math.atan2(math.sin(float(first) - float(second)), math.cos(float(first) - float(second))))


def footprint_pose_stability(ground_truth_samples, position_tolerance_m, yaw_tolerance_rad):
    """Return whether synchronized object poses stayed within configured tolerances."""
    if not ground_truth_samples:
        return False, {"sample_count": 0}
    centers = np.asarray([sample["center_xy"] for sample in ground_truth_samples], dtype=float)
    mean_center = np.mean(centers, axis=0)
    position_deviation = float(np.max(np.linalg.norm(centers - mean_center, axis=1)))
    yaws = [float(sample["yaw_rad"]) for sample in ground_truth_samples]
    mean_yaw = math.atan2(
        sum(math.sin(value) for value in yaws),
        sum(math.cos(value) for value in yaws),
    )
    yaw_deviation = max(wrapped_angle_difference(value, mean_yaw) for value in yaws)
    metrics = {
        "sample_count": len(ground_truth_samples),
        "max_position_deviation_m": position_deviation,
        "max_yaw_deviation_deg": math.degrees(yaw_deviation),
    }
    return (
        position_deviation <= float(position_tolerance_m)
        and yaw_deviation <= float(yaw_tolerance_rad)
    ), metrics


def containment_score(predicted, ground_truth):
    return max(
        0.0,
        float(predicted["min_corner"][0]) - float(ground_truth["min_corner"][0]),
        float(ground_truth["max_corner"][0]) - float(predicted["max_corner"][0]),
        float(predicted["min_corner"][1]) - float(ground_truth["min_corner"][1]),
        float(ground_truth["max_corner"][1]) - float(predicted["max_corner"][1]),
    )


def match_prediction(obstacles, config, ground_truth, maximum_distance_m):
    candidates = []
    for obstacle in obstacles:
        label = str(obstacle.get("label", obstacle.get("shape", ""))).strip().lower()
        if label != config["label"].strip().lower():
            continue
        minimum = obstacle.get("min_corner")
        maximum = obstacle.get("max_corner")
        if not isinstance(minimum, list) or not isinstance(maximum, list) or len(minimum) < 2 or len(maximum) < 2:
            continue
        center = [0.5 * (float(minimum[0]) + float(maximum[0])), 0.5 * (float(minimum[1]) + float(maximum[1]))]
        distance = math.dist(center, ground_truth["center_xy"])
        if distance <= float(maximum_distance_m):
            exact_id = str(obstacle.get("object_id", "")) == config["object_id"]
            candidates.append((not exact_id, distance, obstacle))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if not candidates:
        return None
    if (
        len(candidates) > 1
        and candidates[0][0] == candidates[1][0]
        and math.isclose(candidates[0][1], candidates[1][1], abs_tol=1e-6)
    ):
        return None
    return candidates[0][2]


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class VisionErrorDatasetGenerator(Node):
    def __init__(self):
        super().__init__("vision_error_dataset_generator")
        self.declare_parameter("nominal_obstacle_topic", "/llm_vision/nominal_obstacles")
        self.declare_parameter("calibration_status_topic", "/llm_vision/vision_calibration_status")
        self.declare_parameter("trial_id", "unset")
        self.declare_parameter(
            "output_csv",
            "fine_tuning/datasets/calibration_vision_error.csv",
        )
        self.declare_parameter("vicon_objects_json", "[]")
        self.declare_parameter("object_id", "obj-1")
        self.declare_parameter("object_label", "chair")
        self.declare_parameter("object_vicon_topic", "/vicon/chair1/chair1")
        self.declare_parameter("object_width_m", 0.0)
        self.declare_parameter("object_depth_m", 0.0)
        self.declare_parameter("marker_to_object_json", "{}")
        self.declare_parameter("auto_vicon_world_to_ned", True)
        self.declare_parameter("vicon_world_to_ned_json", "{}")
        self.declare_parameter("vicon_vehicle_topic", "/vicon/Starling2/Starling2")
        self.declare_parameter("vicon_vehicle_frame_convention", "flu")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("frame_sync_tolerance_s", 0.10)
        self.declare_parameter("frame_calibration_samples", 20)
        self.declare_parameter("frame_max_translation_deviation_m", 0.15)
        self.declare_parameter("frame_max_rotation_deviation_deg", 5.0)
        self.declare_parameter("derived_transform_output_json", "")
        self.declare_parameter("sync_tolerance_s", 0.10)
        self.declare_parameter("match_distance_m", 0.75)
        self.declare_parameter("continuous_recording", False)
        self.declare_parameter("object_stability_window_s", 0.50)
        self.declare_parameter("object_stability_min_samples", 5)
        self.declare_parameter("object_stability_position_tolerance_m", 0.02)
        self.declare_parameter("object_stability_yaw_tolerance_deg", 2.0)

        self.trial_id = str(self.get_parameter("trial_id").value).strip()
        if not self.trial_id or self.trial_id.lower() in ("unset", "placeholder"):
            raise ValueError("trial_id must identify one independent, non-placeholder calibration trial")
        self.sync_tolerance_s = float(self.get_parameter("sync_tolerance_s").value)
        self.match_distance_m = float(self.get_parameter("match_distance_m").value)
        self.frame_sync_tolerance_s = float(self.get_parameter("frame_sync_tolerance_s").value)
        self.frame_calibration_samples = int(self.get_parameter("frame_calibration_samples").value)
        self.frame_max_translation_deviation_m = float(
            self.get_parameter("frame_max_translation_deviation_m").value
        )
        self.frame_max_rotation_deviation_deg = float(
            self.get_parameter("frame_max_rotation_deviation_deg").value
        )
        self.continuous_recording = bool(self.get_parameter("continuous_recording").value)
        self.object_stability_window_s = float(
            self.get_parameter("object_stability_window_s").value
        )
        self.object_stability_min_samples = int(
            self.get_parameter("object_stability_min_samples").value
        )
        self.object_stability_position_tolerance_m = float(
            self.get_parameter("object_stability_position_tolerance_m").value
        )
        self.object_stability_yaw_tolerance_rad = math.radians(
            float(self.get_parameter("object_stability_yaw_tolerance_deg").value)
        )
        if min(self.sync_tolerance_s, self.match_distance_m, self.frame_sync_tolerance_s) <= 0.0:
            raise ValueError("synchronization tolerances and match_distance_m must be positive")
        if self.frame_calibration_samples < 3:
            raise ValueError("frame_calibration_samples must be at least three")
        if self.frame_max_translation_deviation_m <= 0.0 or self.frame_max_rotation_deviation_deg <= 0.0:
            raise ValueError("frame transform deviation limits must be positive")
        if (
            self.object_stability_window_s <= 0.0
            or self.object_stability_min_samples < 2
            or self.object_stability_position_tolerance_m <= 0.0
            or self.object_stability_yaw_tolerance_rad <= 0.0
        ):
            raise ValueError("continuous-recording stability parameters must be positive")

        objects_json = str(self.get_parameter("vicon_objects_json").value)
        try:
            configured_objects = json.loads(objects_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"vicon_objects_json is invalid JSON: {exc}") from exc
        if configured_objects:
            self.objects = parse_vicon_objects(objects_json)
        else:
            try:
                marker_to_object = json.loads(str(self.get_parameter("marker_to_object_json").value))
            except json.JSONDecodeError as exc:
                raise ValueError(f"marker_to_object_json is invalid JSON: {exc}") from exc
            self.objects = parse_vicon_objects(
                json.dumps(
                    [
                        {
                            "object_id": str(self.get_parameter("object_id").value),
                            "label": str(self.get_parameter("object_label").value),
                            "topic": str(self.get_parameter("object_vicon_topic").value),
                            "dimensions_m": [
                                float(self.get_parameter("object_width_m").value),
                                float(self.get_parameter("object_depth_m").value),
                            ],
                            "marker_to_object": marker_to_object,
                        }
                    ]
                )
            )

        self.auto_vicon_world_to_ned = bool(self.get_parameter("auto_vicon_world_to_ned").value)
        self.vehicle_frame_convention = str(
            self.get_parameter("vicon_vehicle_frame_convention").value
        ).strip().lower()
        marker_from_body_rotation(self.vehicle_frame_convention)
        self.world_to_ned = None
        self.frame_quality = None
        if not self.auto_vicon_world_to_ned:
            try:
                world_to_ned_raw = json.loads(str(self.get_parameter("vicon_world_to_ned_json").value))
            except json.JSONDecodeError as exc:
                raise ValueError(f"vicon_world_to_ned_json is invalid JSON: {exc}") from exc
            self.world_to_ned = rigid_transform(world_to_ned_raw, "vicon_world_to_ned")
            self.frame_quality = {"sample_count": 0, "source": "configured_json"}

        output = Path(str(self.get_parameter("output_csv").value)).expanduser()
        self.output_csv = output if output.is_absolute() else Path.cwd() / output
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        existed = self.output_csv.exists() and self.output_csv.stat().st_size > 0
        self.csv_fields = RAW_CSV_FIELDS if self.continuous_recording else CSV_FIELDS
        self.capture_index = 0
        self.output_stream = self.output_csv.open("a+", newline="", encoding="utf-8")
        if existed:
            self.output_stream.seek(0)
            reader = csv.DictReader(self.output_stream)
            fields = reader.fieldnames or []
            if fields != self.csv_fields:
                self.output_stream.close()
                raise ValueError(f"existing output CSV has an incompatible header: {self.output_csv}")
            if self.continuous_recording:
                for row in reader:
                    if str(row.get("session_id", "")) == self.trial_id:
                        try:
                            self.capture_index = max(
                                self.capture_index,
                                int(row.get("capture_index", 0)),
                            )
                        except (TypeError, ValueError):
                            self.output_stream.close()
                            raise ValueError(
                                f"existing raw CSV has an invalid capture_index: {self.output_csv}"
                            )
            self.output_stream.seek(0, 2)
        self.writer = csv.DictWriter(
            self.output_stream,
            fieldnames=self.csv_fields,
            lineterminator="\n",
        )
        if not existed:
            self.writer.writeheader()
            self.output_stream.flush()

        self.latest_nominal = None
        self.vicon_history = {
            config["topic"]: deque(maxlen=3000) for config in self.objects
        }
        self.recorded_pairs = set()
        self.vehicle_vicon_samples = deque(maxlen=100)
        self.px4_pose_samples = deque(maxlen=3000)
        self.frame_candidates = deque(maxlen=max(100, self.frame_calibration_samples * 4))
        self.frame_pair_keys = set()
        self.paired_vehicle_vicon_keys = set()
        self.paired_px4_keys = set()
        self.last_frame_rejection_log_s = 0.0
        self.vicon_subscriptions = []
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("calibration_status_topic").value),
            LATCHED_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("nominal_obstacle_topic").value),
            self.nominal_callback,
            LATCHED_QOS,
        )
        for config in self.objects:
            subscription = self.create_subscription(
                TransformStamped,
                config["topic"],
                lambda msg, configured=config: self.vicon_callback(configured, msg),
                10,
            )
            self.vicon_subscriptions.append(subscription)
        if self.auto_vicon_world_to_ned:
            self.vehicle_vicon_subscription = self.create_subscription(
                TransformStamped,
                str(self.get_parameter("vicon_vehicle_topic").value),
                self.vehicle_vicon_callback,
                20,
            )
            self.px4_pose_subscription = self.create_subscription(
                VehicleOdometry,
                str(self.get_parameter("pose_topic").value),
                self.px4_pose_callback,
                ODOM_QOS,
            )
            self.publish_status(
                "WAITING_FOR_FRAME_ALIGNMENT",
                "Collecting synchronized Starling Vicon and PX4 NED poses; keep the vehicle still and visible.",
            )
        else:
            self.save_derived_transform()
            self.publish_status("FRAME_READY", "Using the configured Vicon-world to NED transform.")
        self.get_logger().info(
            f"recording trial_id={self.trial_id} to {self.output_csv} from "
            f"{', '.join(config['topic'] for config in self.objects)}"
        )

    def publish_status(self, status, message, **metadata):
        payload = {
            "status": status,
            "message": message,
            "trial_id": self.trial_id,
            "timestamp": time.time(),
            **metadata,
        }
        self.status_pub.publish(String(data=json.dumps(payload)))
        if status in ("FRAME_READY", "RECORDED"):
            self.get_logger().info(message)

    def save_derived_transform(self):
        if self.world_to_ned is None:
            return
        configured_path = str(self.get_parameter("derived_transform_output_json").value).strip()
        if configured_path:
            output = Path(configured_path).expanduser()
            output = output if output.is_absolute() else Path.cwd() / output
        else:
            safe_trial = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.trial_id)
            output = self.output_csv.with_name(
                f"{self.output_csv.stem}.{safe_trial}.frame.json"
            )
        translation, rotation = self.world_to_ned
        payload = {
            "trial_id": self.trial_id,
            "source": "synchronized_starling_vicon_and_px4_odometry" if self.auto_vicon_world_to_ned else "configured_json",
            "vicon_vehicle_frame_convention": self.vehicle_frame_convention,
            "translation_m": [float(value) for value in translation],
            "quaternion_xyzw": matrix_to_quaternion(rotation),
            "rotation_matrix": np.asarray(rotation, dtype=float).tolist(),
            "quality": self.frame_quality,
            "timestamp": time.time(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.derived_transform_output = output

    def vehicle_vicon_callback(self, msg):
        try:
            pose = transform_message_pose(msg)
        except ValueError as exc:
            self.get_logger().warning(f"ignored invalid Starling Vicon pose: {exc}")
            return
        stamp_key = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        if stamp_key == (0, 0):
            stamp_key = (0, time.monotonic_ns())
        self.vehicle_vicon_samples.append(
            {
                "key": stamp_key,
                "arrival_s": time.monotonic(),
                "pose": pose,
            }
        )
        self.try_frame_pair()

    def px4_pose_callback(self, msg):
        try:
            pose = px4_message_pose(msg)
        except ValueError:
            return
        stamp_key = int(msg.timestamp_sample or msg.timestamp)
        if stamp_key == 0:
            stamp_key = time.monotonic_ns()
        self.px4_pose_samples.append(
            {
                "key": stamp_key,
                "arrival_s": time.monotonic(),
                "pose": pose,
            }
        )
        self.try_frame_pair()

    def try_frame_pair(self):
        if self.world_to_ned is not None or not self.vehicle_vicon_samples or not self.px4_pose_samples:
            return
        vicon = self.vehicle_vicon_samples[-1]
        px4 = min(
            self.px4_pose_samples,
            key=lambda sample: abs(float(sample["arrival_s"]) - float(vicon["arrival_s"])),
        )
        if abs(float(px4["arrival_s"]) - float(vicon["arrival_s"])) > self.frame_sync_tolerance_s:
            return
        if (
            vicon["key"] in self.paired_vehicle_vicon_keys
            or px4["key"] in self.paired_px4_keys
        ):
            return
        pair_key = (vicon["key"], px4["key"])
        if pair_key in self.frame_pair_keys:
            return
        self.frame_pair_keys.add(pair_key)
        self.paired_vehicle_vicon_keys.add(vicon["key"])
        self.paired_px4_keys.add(px4["key"])
        self.frame_candidates.append(
            world_to_ned_candidate(vicon["pose"], px4["pose"], self.vehicle_frame_convention)
        )
        if len(self.frame_candidates) < self.frame_calibration_samples:
            return
        transform, quality = aggregate_world_to_ned(
            list(self.frame_candidates)[-self.frame_calibration_samples :]
        )
        if (
            quality["max_translation_deviation_m"] > self.frame_max_translation_deviation_m
            or quality["max_rotation_deviation_deg"] > self.frame_max_rotation_deviation_deg
        ):
            now = time.monotonic()
            if now - self.last_frame_rejection_log_s >= 1.0:
                self.last_frame_rejection_log_s = now
                self.publish_status(
                    "FRAME_ALIGNMENT_UNSTABLE",
                    "Pose-pair alignment is inconsistent; keep the Starling still and check Vicon/PX4 pose streams.",
                    quality=quality,
                )
            return
        quality["source"] = "synchronized_pose_pairs"
        self.world_to_ned = transform
        self.frame_quality = quality
        self.save_derived_transform()
        self.publish_status(
            "FRAME_READY",
            "Automatically derived Vicon-world to local-NED alignment; calibration capture may proceed.",
            quality=quality,
            derived_transform_json=str(self.derived_transform_output),
        )
        for config in self.objects:
            self.try_record(config)

    def nominal_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid nominal-obstacle JSON: {exc}")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("obstacles"), list):
            self.get_logger().error("nominal-obstacle payload must contain an obstacles list")
            return
        try:
            timestamp = float(payload.get("timestamp"))
        except (TypeError, ValueError):
            self.get_logger().error("nominal-obstacle payload requires a numeric timestamp")
            return
        if not math.isfinite(timestamp):
            self.get_logger().error("nominal-obstacle timestamp is non-finite")
            return
        if self.continuous_recording:
            self.capture_index += 1
            safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.trial_id)
            capture_id = f"{safe_session}-capture-{self.capture_index:06d}"
        else:
            capture_id = self.trial_id
        self.latest_nominal = {
            "payload": payload,
            "timestamp": timestamp,
            "capture_id": capture_id,
            "capture_index": self.capture_index,
        }
        for config in self.objects:
            self.try_record(config)

    def vicon_callback(self, config, msg):
        timestamp = stamp_seconds(msg.header.stamp)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        try:
            position, rotation = transform_message_pose(msg)
        except ValueError as exc:
            self.get_logger().warning(f"ignored invalid Vicon transform on {config['topic']}: {exc}")
            return
        self.vicon_history[config["topic"]].append({
            "timestamp": timestamp,
            "position": position,
            "rotation": rotation,
        })
        self.try_record(config)

    def stable_object_pose(self, config, nominal):
        if not self.continuous_recording:
            return True, None
        history = self.vicon_history[config["topic"]]
        half_window = 0.5 * self.object_stability_window_s
        if not history or history[-1]["timestamp"] < nominal["timestamp"] + half_window:
            return None, None
        samples = [
            sample
            for sample in history
            if abs(float(sample["timestamp"]) - nominal["timestamp"]) <= half_window
        ]
        if len(samples) < self.object_stability_min_samples:
            return False, {"sample_count": len(samples)}
        ground_truth_samples = [
            ground_truth_aabb(
                config,
                sample["position"],
                sample["rotation"],
                self.world_to_ned,
            )
            for sample in samples
        ]
        return footprint_pose_stability(
            ground_truth_samples,
            self.object_stability_position_tolerance_m,
            self.object_stability_yaw_tolerance_rad,
        )

    @staticmethod
    def observer_pose_fields(payload):
        pose = payload.get("observer_pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 4:
            return {field: "" for field in ("observer_x", "observer_y", "observer_z", "observer_yaw_rad")}
        try:
            values = [float(value) for value in pose[:4]]
        except (TypeError, ValueError):
            values = []
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            return {field: "" for field in ("observer_x", "observer_y", "observer_z", "observer_yaw_rad")}
        return {
            "observer_x": f"{values[0]:.9f}",
            "observer_y": f"{values[1]:.9f}",
            "observer_z": f"{values[2]:.9f}",
            "observer_yaw_rad": f"{math.radians(values[3]):.9f}",
        }

    def try_record(self, config):
        nominal = self.latest_nominal
        history = self.vicon_history.get(config["topic"])
        if nominal is None or not history or self.world_to_ned is None:
            return
        vicon = min(
            history,
            key=lambda sample: abs(float(sample["timestamp"]) - nominal["timestamp"]),
        )
        if abs(nominal["timestamp"] - vicon["timestamp"]) > self.sync_tolerance_s:
            return
        pair = (config["object_id"], round(nominal["timestamp"], 6))
        if pair in self.recorded_pairs:
            return
        stable, stability = self.stable_object_pose(config, nominal)
        if stable is None:
            return
        if not stable:
            self.recorded_pairs.add(pair)
            self.publish_status(
                "SKIPPED_MOVING_OBJECT",
                f"Skipped {nominal['capture_id']} because {config['object_id']} was moving.",
                capture_id=nominal["capture_id"],
                object_id=config["object_id"],
                stability=stability,
            )
            return
        ground_truth = ground_truth_aabb(
            config,
            vicon["position"],
            vicon["rotation"],
            self.world_to_ned,
        )
        predicted = match_prediction(
            nominal["payload"]["obstacles"],
            config,
            ground_truth,
            self.match_distance_m,
        )
        row = {
            "trial_id": nominal["capture_id"] if self.continuous_recording else self.trial_id,
            "timestamp_s": f"{nominal['timestamp']:.9f}",
            "vicon_timestamp_s": f"{vicon['timestamp']:.9f}",
            "object_id": config["object_id"],
            "label": config["label"],
            "gt_min_x": f"{ground_truth['min_corner'][0]:.9f}",
            "gt_min_y": f"{ground_truth['min_corner'][1]:.9f}",
            "gt_max_x": f"{ground_truth['max_corner'][0]:.9f}",
            "gt_max_y": f"{ground_truth['max_corner'][1]:.9f}",
            "missed_detection": str(predicted is None).lower(),
            "placeholder": "false",
        }
        if self.continuous_recording:
            row.update(
                {
                    "session_id": self.trial_id,
                    "capture_id": nominal["capture_id"],
                    "capture_index": str(nominal["capture_index"]),
                    "raw_continuous": "true",
                    "stable_pose": "true",
                    "gt_center_x": f"{ground_truth['center_xy'][0]:.9f}",
                    "gt_center_y": f"{ground_truth['center_xy'][1]:.9f}",
                    "gt_yaw_rad": f"{ground_truth['yaw_rad']:.9f}",
                    **self.observer_pose_fields(nominal["payload"]),
                }
            )
        if predicted is None:
            row.update(
                {
                    "pred_min_x": "",
                    "pred_min_y": "",
                    "pred_max_x": "",
                    "pred_max_y": "",
                    "score_m": "",
                }
            )
        else:
            row.update(
                {
                    "pred_min_x": f"{float(predicted['min_corner'][0]):.9f}",
                    "pred_min_y": f"{float(predicted['min_corner'][1]):.9f}",
                    "pred_max_x": f"{float(predicted['max_corner'][0]):.9f}",
                    "pred_max_y": f"{float(predicted['max_corner'][1]):.9f}",
                    "score_m": f"{containment_score(predicted, ground_truth):.9f}",
                }
            )
        self.writer.writerow(row)
        self.output_stream.flush()
        self.recorded_pairs.add(pair)
        outcome = "MISS" if predicted is None else f"score={row['score_m']} m"
        self.get_logger().info(f"recorded {config['object_id']} {outcome}")
        self.publish_status(
            "RECORDED",
            f"Recorded {config['object_id']} for {nominal['capture_id']}: {outcome}.",
            session_id=self.trial_id,
            capture_id=nominal["capture_id"],
            object_id=config["object_id"],
            label=config["label"],
            missed_detection=predicted is None,
            score_m=None if predicted is None else float(row["score_m"]),
            output_csv=str(self.output_csv),
        )

    def destroy_node(self):
        if hasattr(self, "output_stream") and not self.output_stream.closed:
            self.output_stream.flush()
            self.output_stream.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = VisionErrorDatasetGenerator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
