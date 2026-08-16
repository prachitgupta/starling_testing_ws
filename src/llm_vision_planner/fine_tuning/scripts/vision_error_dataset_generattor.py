#!/usr/bin/env python3
"""Record nominal obstacle footprints against synchronized Vicon ground truth."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
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
        marker_translation, marker_rotation = rigid_transform(
            raw.get("marker_to_object"),
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
    }


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
        self.declare_parameter("trial_id", "unset")
        self.declare_parameter(
            "output_csv",
            "fine_tuning/datasets/calibration_vision_error.csv",
        )
        self.declare_parameter("vicon_objects_json", "[]")
        self.declare_parameter("vicon_world_to_ned_json", "{}")
        self.declare_parameter("sync_tolerance_s", 0.10)
        self.declare_parameter("match_distance_m", 0.75)

        self.trial_id = str(self.get_parameter("trial_id").value).strip()
        if not self.trial_id or self.trial_id.lower() in ("unset", "placeholder"):
            raise ValueError("trial_id must identify one independent, non-placeholder calibration trial")
        self.sync_tolerance_s = float(self.get_parameter("sync_tolerance_s").value)
        self.match_distance_m = float(self.get_parameter("match_distance_m").value)
        if self.sync_tolerance_s <= 0.0 or self.match_distance_m <= 0.0:
            raise ValueError("sync_tolerance_s and match_distance_m must be positive")
        self.objects = parse_vicon_objects(str(self.get_parameter("vicon_objects_json").value))
        try:
            world_to_ned_raw = json.loads(str(self.get_parameter("vicon_world_to_ned_json").value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"vicon_world_to_ned_json is invalid JSON: {exc}") from exc
        self.world_to_ned = rigid_transform(world_to_ned_raw, "vicon_world_to_ned")

        output = Path(str(self.get_parameter("output_csv").value)).expanduser()
        self.output_csv = output if output.is_absolute() else Path.cwd() / output
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        existed = self.output_csv.exists() and self.output_csv.stat().st_size > 0
        self.output_stream = self.output_csv.open("a+", newline="", encoding="utf-8")
        if existed:
            self.output_stream.seek(0)
            fields = next(csv.reader(self.output_stream), [])
            if fields != CSV_FIELDS:
                self.output_stream.close()
                raise ValueError(f"existing output CSV has an incompatible header: {self.output_csv}")
            self.output_stream.seek(0, 2)
        self.writer = csv.DictWriter(self.output_stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        if not existed:
            self.writer.writeheader()
            self.output_stream.flush()

        self.latest_nominal = None
        self.latest_vicon = {}
        self.recorded_pairs = set()
        self.vicon_subscriptions = []
        self.create_subscription(
            String,
            str(self.get_parameter("nominal_obstacle_topic").value),
            self.nominal_callback,
            10,
        )
        for config in self.objects:
            subscription = self.create_subscription(
                TransformStamped,
                config["topic"],
                lambda msg, configured=config: self.vicon_callback(configured, msg),
                10,
            )
            self.vicon_subscriptions.append(subscription)
        self.get_logger().info(
            f"recording trial_id={self.trial_id} to {self.output_csv} from "
            f"{', '.join(config['topic'] for config in self.objects)}"
        )

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
        self.latest_nominal = {"payload": payload, "timestamp": timestamp}
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
        self.latest_vicon[config["topic"]] = {
            "timestamp": timestamp,
            "position": position,
            "rotation": rotation,
        }
        self.try_record(config)

    def try_record(self, config):
        nominal = self.latest_nominal
        vicon = self.latest_vicon.get(config["topic"])
        if nominal is None or vicon is None:
            return
        if abs(nominal["timestamp"] - vicon["timestamp"]) > self.sync_tolerance_s:
            return
        pair = (config["object_id"], round(nominal["timestamp"], 6))
        if pair in self.recorded_pairs:
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
            "trial_id": self.trial_id,
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
