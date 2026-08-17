#!/usr/bin/env python3
"""Human-approved mission intent gateway for the existing planner pipeline."""

import copy
import csv
import hashlib
import json
import math
import os
import queue
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import rclpy
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


INSTRUCTIONS = (
    "You are an expert UAV pilot/planner. Generate sparse collision-free 2D routing "
    "waypoints for a quadrotor in NED frame assuming it is airborne at the provided "
    "start location. A separate module will interpolate them and generate dynamically "
    "feasible trajectories, so output only high-level waypoints."
)
INTENT_SYSTEM_PROMPT = """You are a UAV mission intent parser.
Return only the supplied MissionIntent structure.
Do not generate coordinates, waypoints, or a route.
Never invent an object ID; copy it exactly from the detected-object catalog.
Use the operator's language for clarifying_question.
Supported non-motion queries are DESCRIBE_SCENE, LIST_OBJECTS, LOCATE_OBJECT,
EXPLAIN_PROPOSAL, and EXPLAIN_FAILURE. Queries never move the vehicle.
For LOCATE_OBJECT, put the uniquely matched detected ID in query_object_ids.
Supported navigation actions are HOVER and GO_TO with one or more object relations.
Every navigation intent needs at least one NEAR relation to anchor the goal.
NEAR and FAR_FROM may include an explicit minimum, maximum, or exact range in metres.
For an exact distance, set min_distance_m and max_distance_m to the same value.
For an unspecified NEAR distance, leave both distances null so policy defaults apply.
For 'as far as possible', use FAR_FROM with optimize=MAXIMIZE and null distances.
Do not invent a number for vague words such as 'far'; ask for clarification unless
the operator explicitly requests the maximum feasible distance.
If any referenced object is missing or ambiguous, return NEEDS_CLARIFICATION.
HOLD, LAND, RETURN_HOME, and all other flight-control requests are unsupported.
Cancellations use status CANCELLED and do not return READY.
If any safety-relevant meaning is uncertain, return NEEDS_CLARIFICATION.
All detected objects remain mandatory avoidance obstacles.
For every detected object, also estimate its effective physical occupancy depth in
metres from the visible front surface away from the camera. Return exactly one item
in depth_estimates for every detected object ID, including objects not named in the
operator command. This is a scalar size estimate, not a coordinate or waypoint.
If the label or view is insufficient for a defensible estimate, set abstained=true
and effective_depth_along_view_m=null. Never invent or alter an object ID.
"""

ODOM_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class GoalRelation(BaseModel):
    """One natural-language object relation grounded to detected geometry."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    object_label: str
    relation: Literal["NEAR", "FAR_FROM"]
    min_distance_m: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    max_distance_m: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    optimize: Literal["NONE", "MAXIMIZE"]


class ObjectDepthEstimate(BaseModel):
    """One schema-constrained effective-depth estimate for a detected object."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    effective_depth_along_view_m: Optional[float] = Field(default=None, ge=0.05, le=5.0)
    abstained: bool


class MissionIntent(BaseModel):
    """Schema-constrained result returned by the intent model."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["READY", "NEEDS_CLARIFICATION", "CANCELLED", "UNSUPPORTED"]
    intent_type: Literal["NAVIGATION", "QUERY", "NONE"]
    navigation_action: Literal["HOVER", "GO_TO", "NONE"]
    query_type: Literal[
        "DESCRIBE_SCENE",
        "LIST_OBJECTS",
        "LOCATE_OBJECT",
        "EXPLAIN_PROPOSAL",
        "EXPLAIN_FAILURE",
        "NONE",
    ]
    relations: list[GoalRelation] = Field(default_factory=list, max_length=8)
    query_object_ids: list[str] = Field(default_factory=list, max_length=8)
    depth_estimates: list[ObjectDepthEstimate] = Field(default_factory=list, max_length=32)
    clarifying_question: str


class OpenAIIntentParser:
    def __init__(self, model_name):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_name = model_name
        self.client = OpenAI()

    def parse(self, operator_text, object_catalog, conversation):
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "detected_objects": object_catalog,
                            "conversation": conversation[-6:],
                            "operator_command": operator_text,
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            text_format=MissionIntent,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no parsed MissionIntent")
        return response.output_parsed


class MockIntentParser:
    """Offline parser used only by repeatable simulation and unit tests."""

    def parse(self, operator_text, object_catalog, conversation):
        del conversation
        lowered = operator_text.lower()
        if "cancel" in lowered:
            return MissionIntent(
                status="CANCELLED",
                intent_type="NONE",
                navigation_action="NONE",
                query_type="NONE",
                relations=[],
                query_object_ids=[],
                clarifying_question="",
            )
        query_type = self.query_type(lowered)
        matches = [item for item in object_catalog if item["label"].lower() in lowered]
        if query_type != "NONE":
            if query_type == "LOCATE_OBJECT" and len(matches) != 1:
                labels = ", ".join(item["label"] for item in object_catalog) or "none"
                return MissionIntent(
                    status="NEEDS_CLARIFICATION",
                    intent_type="QUERY",
                    navigation_action="NONE",
                    query_type=query_type,
                    relations=[],
                    query_object_ids=[],
                    clarifying_question=f"Which detected object should I locate? I see: {labels}.",
                )
            return MissionIntent(
                status="READY",
                intent_type="QUERY",
                navigation_action="NONE",
                query_type=query_type,
                relations=[],
                query_object_ids=[matches[0]["object_id"]] if query_type == "LOCATE_OBJECT" else [],
                clarifying_question="",
            )
        if any(term in lowered for term in ("hold", "land", "return home", "rth")):
            return MissionIntent(
                status="UNSUPPORTED",
                intent_type="NONE",
                navigation_action="NONE",
                query_type="NONE",
                relations=[],
                query_object_ids=[],
                clarifying_question="Flight-control requests are not supported by the intent gateway.",
            )

        if not matches:
            labels = ", ".join(item["label"] for item in object_catalog) or "none"
            return MissionIntent(
                status="NEEDS_CLARIFICATION",
                intent_type="NAVIGATION",
                navigation_action="NONE",
                query_type="NONE",
                relations=[],
                query_object_ids=[],
                clarifying_question=f"Which detected object do you mean? I see: {labels}.",
            )

        relations = []
        for item in matches:
            label = item["label"].lower()
            far = bool(
                re.search(rf"(?:far(?:thest)?|away)\s+(?:as possible\s+)?from\s+(?:the\s+)?{re.escape(label)}", lowered)
                or re.search(rf"far(?:thest)?[^.]*{re.escape(label)}", lowered)
            )
            distance_match = re.search(
                rf"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b[^.]*{re.escape(label)}",
                lowered,
            )
            distance = float(distance_match.group(1)) if distance_match else None
            maximize = far and any(term in lowered for term in ("as far as possible", "farthest", "maximum distance"))
            relations.append(
                GoalRelation(
                    object_id=item["object_id"],
                    object_label=item["label"],
                    relation="FAR_FROM" if far else "NEAR",
                    min_distance_m=distance,
                    max_distance_m=None if far else distance,
                    optimize="MAXIMIZE" if maximize else "NONE",
                )
            )
        if not any(relation.relation == "NEAR" for relation in relations):
            return MissionIntent(
                status="NEEDS_CLARIFICATION",
                intent_type="NAVIGATION",
                navigation_action="NONE",
                query_type="NONE",
                relations=relations,
                query_object_ids=[],
                clarifying_question="Which detected object should anchor the goal position?",
            )
        return MissionIntent(
            status="READY",
            intent_type="NAVIGATION",
            navigation_action="HOVER" if "hover" in lowered else "GO_TO",
            query_type="NONE",
            relations=relations,
            query_object_ids=[],
            clarifying_question="",
        )

    @staticmethod
    def query_type(lowered):
        if any(term in lowered for term in ("explain the proposal", "why this goal", "explain proposal")):
            return "EXPLAIN_PROPOSAL"
        if any(term in lowered for term in ("explain the failure", "why did it fail", "explain failure")):
            return "EXPLAIN_FAILURE"
        if any(term in lowered for term in ("what do you see", "describe the scene", "describe scene")):
            return "DESCRIBE_SCENE"
        if any(term in lowered for term in ("list objects", "list the objects", "what objects")):
            return "LIST_OBJECTS"
        if any(term in lowered for term in ("where is", "locate", "location of")):
            return "LOCATE_OBJECT"
        return "NONE"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def containment_score(predicted, ground_truth):
    """Smallest scalar x-y expansion that contains the ground-truth AABB."""
    predicted_min = predicted["min_corner"]
    predicted_max = predicted["max_corner"]
    ground_truth_min = ground_truth["min_corner"]
    ground_truth_max = ground_truth["max_corner"]
    return max(
        0.0,
        float(predicted_min[0]) - float(ground_truth_min[0]),
        float(ground_truth_max[0]) - float(predicted_max[0]),
        float(predicted_min[1]) - float(ground_truth_min[1]),
        float(ground_truth_max[1]) - float(predicted_max[1]),
    )


def containment_score_center_extent(predicted, ground_truth):
    """Equivalent containment score expressed using centers and half extents."""
    predicted_center, predicted_size = obstacle_xy_geometry(predicted)
    ground_truth_center, ground_truth_size = obstacle_xy_geometry(ground_truth)
    return max(
        0.0,
        abs(predicted_center[0] - ground_truth_center[0])
        + 0.5 * ground_truth_size[0]
        - 0.5 * predicted_size[0],
        abs(predicted_center[1] - ground_truth_center[1])
        + 0.5 * ground_truth_size[1]
        - 0.5 * predicted_size[1],
    )


def conformal_quantile(values, delta):
    """Finite-sample split-conformal order statistic without silent rank clipping."""
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("vision_error_delta must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("vision calibration requires at least one independent trial")
    rank = math.ceil((len(ordered) + 1) * (1.0 - float(delta)))
    if rank > len(ordered):
        raise ValueError(
            f"vision calibration has {len(ordered)} independent trials, insufficient for delta={float(delta):.3f}"
        )
    return ordered[rank - 1], rank


def resolve_data_file(value):
    requested = Path(value).expanduser()
    candidates = [requested, Path.cwd() / requested]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("llm_vision_planner")) / requested)
    except Exception:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"vision calibration CSV does not exist: {value}")


def parse_csv_bool(value, field):
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError(f"invalid {field} value: {value!r}")


def load_vision_error_certificate(value, delta):
    """Load per-trial maximum containment scores and return a fail-closed certificate."""
    path = resolve_data_file(value)
    required = {
        "trial_id",
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
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing calibration columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no calibration rows")
    if "raw_continuous" in (reader.fieldnames or []) and any(
        parse_csv_bool(row.get("raw_continuous"), "raw_continuous") for row in rows
    ):
        raise ValueError(
            f"{path} contains raw continuous captures; run postprocess_vision_error_dataset.py first"
        )

    trial_scores = {}
    placeholder_flags = set()
    bounds_fields = (
        "pred_min_x",
        "pred_min_y",
        "pred_max_x",
        "pred_max_y",
        "gt_min_x",
        "gt_min_y",
        "gt_max_x",
        "gt_max_y",
    )
    for row_index, row in enumerate(rows, start=2):
        trial_id = str(row.get("trial_id", "")).strip()
        if not trial_id:
            raise ValueError(f"{path}:{row_index} has an empty trial_id")
        placeholder_flags.add(parse_csv_bool(row.get("placeholder"), "placeholder"))
        missed = parse_csv_bool(row.get("missed_detection"), "missed_detection")
        numeric_fields = bounds_fields[4:] if missed else bounds_fields
        try:
            bounds = {field: float(row[field]) for field in numeric_fields}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{row_index} has invalid footprint bounds") from exc
        if not all(math.isfinite(value) for value in bounds.values()):
            raise ValueError(f"{path}:{row_index} has non-finite footprint bounds")
        if not missed and (
            bounds["pred_min_x"] > bounds["pred_max_x"]
            or bounds["pred_min_y"] > bounds["pred_max_y"]
        ):
            raise ValueError(f"{path}:{row_index} has inverted predicted bounds")
        if bounds["gt_min_x"] > bounds["gt_max_x"] or bounds["gt_min_y"] > bounds["gt_max_y"]:
            raise ValueError(f"{path}:{row_index} has inverted ground-truth bounds")
        if missed:
            score = math.inf
        else:
            try:
                score = float(row["score_m"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_index} has an invalid score_m") from exc
            if not math.isfinite(score) or score < 0.0:
                raise ValueError(f"{path}:{row_index} score_m must be finite and non-negative")
            expected_score = containment_score(
                {
                    "min_corner": [bounds["pred_min_x"], bounds["pred_min_y"]],
                    "max_corner": [bounds["pred_max_x"], bounds["pred_max_y"]],
                },
                {
                    "min_corner": [bounds["gt_min_x"], bounds["gt_min_y"]],
                    "max_corner": [bounds["gt_max_x"], bounds["gt_max_y"]],
                },
            )
            if not math.isclose(score, expected_score, abs_tol=1e-6):
                raise ValueError(
                    f"{path}:{row_index} score_m={score} does not match containment score {expected_score}"
                )
        trial_scores[trial_id] = max(score, trial_scores.get(trial_id, 0.0))

    if len(placeholder_flags) != 1:
        raise ValueError(f"{path} mixes placeholder and real calibration rows")
    quantile, rank = conformal_quantile(trial_scores.values(), delta)
    if not math.isfinite(quantile):
        raise ValueError(
            f"{path} selects a non-finite quantile because detection misses exceed the requested risk"
        )
    return {
        "quantile_m": quantile,
        "delta": float(delta),
        "trial_count": len(trial_scores),
        "rank": rank,
        "file": str(path),
        "placeholder": placeholder_flags.pop(),
    }


def nominal_obstacles_from_depth(obstacles, depth_estimates):
    """Extrude each perceived front segment away from the camera into a nominal footprint."""
    estimates = {}
    for raw in depth_estimates:
        estimate = raw.model_dump() if isinstance(raw, ObjectDepthEstimate) else dict(raw)
        object_id = str(estimate.get("object_id", ""))
        if not object_id or object_id in estimates:
            raise ValueError("GPT depth estimates must contain unique detected object IDs")
        estimates[object_id] = estimate
    expected_ids = {str(obstacle["object_id"]) for obstacle in obstacles}
    if set(estimates) != expected_ids:
        missing = sorted(expected_ids - set(estimates))
        extra = sorted(set(estimates) - expected_ids)
        raise ValueError(f"GPT depth estimates do not match the scene (missing={missing}, extra={extra})")

    nominal = []
    for obstacle in obstacles:
        object_id = str(obstacle["object_id"])
        estimate = estimates[object_id]
        depth = estimate.get("effective_depth_along_view_m")
        if bool(estimate.get("abstained")) or depth is None:
            raise ValueError(f"GPT nano abstained from estimating depth for {object_id}")
        depth = float(depth)
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError(f"GPT nano returned invalid depth for {object_id}")

        front = obstacle.get("front_surface_center")
        view = obstacle.get("view_axis_xy")
        lateral = obstacle.get("lateral_axis_xy")
        width = obstacle.get("visible_width_m")
        if not isinstance(front, list) or len(front) < 2:
            raise ValueError(f"perception did not provide front_surface_center for {object_id}")
        if not isinstance(view, list) or len(view) != 2:
            raise ValueError(f"perception did not provide view_axis_xy for {object_id}")
        if not isinstance(lateral, list) or len(lateral) != 2:
            raise ValueError(f"perception did not provide lateral_axis_xy for {object_id}")
        values = [float(front[0]), float(front[1]), float(view[0]), float(view[1]), float(lateral[0]), float(lateral[1]), float(width)]
        if not all(math.isfinite(value) for value in values) or values[-1] <= 0.0:
            raise ValueError(f"perception provided invalid front geometry for {object_id}")
        view_norm = math.hypot(values[2], values[3])
        lateral_norm = math.hypot(values[4], values[5])
        if view_norm <= 1e-9 or lateral_norm <= 1e-9:
            raise ValueError(f"perception provided a zero-length footprint axis for {object_id}")
        view_xy = (values[2] / view_norm, values[3] / view_norm)
        lateral_xy = (values[4] / lateral_norm, values[5] / lateral_norm)
        if abs(view_xy[0] * lateral_xy[0] + view_xy[1] * lateral_xy[1]) > 0.05:
            raise ValueError(f"perception footprint axes are not perpendicular for {object_id}")
        half_width = 0.5 * values[-1]
        front_xy = (values[0], values[1])
        corners = [
            [front_xy[0] - half_width * lateral_xy[0], front_xy[1] - half_width * lateral_xy[1]],
            [front_xy[0] + half_width * lateral_xy[0], front_xy[1] + half_width * lateral_xy[1]],
        ]
        corners.extend([[point[0] + depth * view_xy[0], point[1] + depth * view_xy[1]] for point in corners])
        minimum_z = float(obstacle.get("min_corner", [0.0, 0.0, 0.0])[2])
        maximum_z = float(obstacle.get("max_corner", [0.0, 0.0, 0.0])[2])
        item = copy.deepcopy(obstacle)
        item["min_corner"] = [min(point[0] for point in corners), min(point[1] for point in corners), minimum_z]
        item["max_corner"] = [max(point[0] for point in corners), max(point[1] for point in corners), maximum_z]
        item["centroid"] = [
            front_xy[0] + 0.5 * depth * view_xy[0],
            front_xy[1] + 0.5 * depth * view_xy[1],
            0.5 * (minimum_z + maximum_z),
        ]
        item["size"] = [
            item["max_corner"][0] - item["min_corner"][0],
            item["max_corner"][1] - item["min_corner"][1],
            maximum_z - minimum_z,
        ]
        item["effective_depth_along_view_m"] = depth
        item["depth_estimate_source"] = "gpt_nano"
        item["nominal_footprint_corners_xy"] = corners
        nominal.append(item)
    return nominal


def object_id_for(obstacle, index):
    value = obstacle.get("object_id", obstacle.get("id", index))
    value = str(value)
    return value if value.startswith("obj-") else f"obj-{value}"


def normalize_obstacles(payload):
    normalized = []
    used_ids = set()
    for index, raw in enumerate(payload.get("obstacles", []), start=1):
        obstacle = copy.deepcopy(raw)
        minimum = obstacle.get("min_corner")
        maximum = obstacle.get("max_corner")
        if not isinstance(minimum, list) or not isinstance(maximum, list) or len(minimum) < 2 or len(maximum) < 2:
            raise ValueError("every obstacle requires min_corner and max_corner")
        object_id = object_id_for(obstacle, index)
        if object_id in used_ids:
            raise ValueError(f"duplicate obstacle object_id: {object_id}")
        used_ids.add(object_id)
        obstacle["object_id"] = object_id
        obstacle["label"] = str(obstacle.get("label") or obstacle.get("shape") or "unknown")
        obstacle["min_corner"] = [float(value) for value in minimum]
        obstacle["max_corner"] = [float(value) for value in maximum]
        normalized.append(obstacle)
    return normalized


def scene_signature(obstacles):
    geometry = [
        {
            "object_id": item["object_id"],
            "label": item["label"],
            "min_corner": [round(float(value), 3) for value in item["min_corner"]],
            "max_corner": [round(float(value), 3) for value in item["max_corner"]],
        }
        for item in obstacles
    ]
    geometry.sort(key=lambda item: item["object_id"])
    return hashlib.sha256(canonical_json(geometry).encode("utf-8")).hexdigest()


def obstacle_xy_geometry(obstacle):
    minimum = obstacle["min_corner"]
    maximum = obstacle["max_corner"]
    center = (
        0.5 * (float(minimum[0]) + float(maximum[0])),
        0.5 * (float(minimum[1]) + float(maximum[1])),
    )
    size = (
        float(maximum[0]) - float(minimum[0]),
        float(maximum[1]) - float(minimum[1]),
    )
    return center, size


def scenes_compatible(reference, current, position_tolerance, size_tolerance):
    """Match by label and geometry so sensor jitter and reordered detections are safe."""
    if len(reference) != len(current):
        return False, f"obstacle count changed from {len(reference)} to {len(current)}"
    unmatched = set(range(len(current)))
    for expected in reference:
        expected_center, expected_size = obstacle_xy_geometry(expected)
        candidates = []
        for index in unmatched:
            observed = current[index]
            if observed["label"].strip().lower() != expected["label"].strip().lower():
                continue
            observed_center, observed_size = obstacle_xy_geometry(observed)
            center_delta = math.dist(expected_center, observed_center)
            size_delta = max(abs(a - b) for a, b in zip(expected_size, observed_size))
            candidates.append((center_delta, size_delta, index))
        if not candidates:
            return False, f"missing obstacle label '{expected['label']}'"
        center_delta, size_delta, index = min(candidates)
        if center_delta > float(position_tolerance):
            return False, (
                f"{expected['label']} moved {center_delta:.3f} m "
                f"(allowed {float(position_tolerance):.3f} m)"
            )
        if size_delta > float(size_tolerance):
            return False, (
                f"{expected['label']} size changed {size_delta:.3f} m "
                f"(allowed {float(size_tolerance):.3f} m)"
            )
        unmatched.remove(index)
    return True, ""


def inflate_obstacles_xy(obstacles, padding):
    """Create a conservative planning envelope covering every accepted scene."""
    inflated = copy.deepcopy(obstacles)
    for obstacle in inflated:
        minimum = obstacle["min_corner"]
        maximum = obstacle["max_corner"]
        minimum[0] = float(minimum[0]) - float(padding)
        minimum[1] = float(minimum[1]) - float(padding)
        maximum[0] = float(maximum[0]) + float(padding)
        maximum[1] = float(maximum[1]) + float(padding)
        if isinstance(obstacle.get("size"), list) and len(obstacle["size"]) >= 2:
            obstacle["size"][0] = maximum[0] - minimum[0]
            obstacle["size"][1] = maximum[1] - minimum[1]
    return inflated


def clearance_to_box(point, obstacle):
    minimum = obstacle["min_corner"]
    maximum = obstacle["max_corner"]
    x = float(point["x"])
    y = float(point["y"])
    dx = max(float(minimum[0]) - x, 0.0, x - float(maximum[0]))
    dy = max(float(minimum[1]) - y, 0.0, y - float(maximum[1]))
    if dx == 0.0 and dy == 0.0:
        return -min(
            abs(x - float(minimum[0])),
            abs(float(maximum[0]) - x),
            abs(y - float(minimum[1])),
            abs(float(maximum[1]) - y),
        )
    return math.hypot(dx, dy)


def normalize_goal_relations(
    raw_relations,
    observed_obstacles,
    *,
    default_standoff,
    clearance,
    guard_band,
    default_range_half_width,
    exact_distance_tolerance,
):
    """Validate LLM relations and ground them to immutable observed-object bounds."""
    by_id = {item["object_id"]: item for item in observed_obstacles}
    normalized = []
    used_ids = set()
    for raw in raw_relations:
        relation = raw.model_dump() if isinstance(raw, GoalRelation) else dict(raw)
        object_id = str(relation.get("object_id", ""))
        obstacle = by_id.get(object_id)
        if obstacle is None:
            raise ValueError(f"selected object ID '{object_id}' is not in the current snapshot")
        if obstacle["label"].strip().lower() != str(relation.get("object_label", "")).strip().lower():
            raise ValueError(f"selected object ID '{object_id}' does not match its detected label")
        same_label = [
            item
            for item in observed_obstacles
            if item["label"].strip().lower() == obstacle["label"].strip().lower()
        ]
        if len(same_label) > 1:
            raise ValueError(
                f"I see {len(same_label)} objects labelled {obstacle['label']}; specify which object you mean"
            )
        if object_id in used_ids:
            raise ValueError(f"object '{object_id}' has more than one relation; combine them into one range")
        used_ids.add(object_id)

        minimum = relation.get("min_distance_m")
        maximum = relation.get("max_distance_m")
        minimum = None if minimum is None else float(minimum)
        maximum = None if maximum is None else float(maximum)
        relation_name = str(relation.get("relation", ""))
        optimize = str(relation.get("optimize", "NONE"))
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"distance range for {obstacle['label']} has minimum greater than maximum")

        if relation_name == "NEAR":
            if minimum is None and maximum is None:
                preferred = float(guard_band) + max(float(default_standoff), float(clearance)) + 0.02
                minimum = max(
                    float(guard_band) + float(clearance) + 0.02,
                    preferred - float(default_range_half_width),
                )
                maximum = preferred + float(default_range_half_width)
            elif minimum is not None and maximum is None:
                raise ValueError(f"NEAR relation for {obstacle['label']} also needs a maximum distance")
            elif minimum is None:
                minimum = 0.0
            elif math.isclose(minimum, maximum, abs_tol=1e-9):
                minimum = max(0.0, minimum - float(exact_distance_tolerance))
                maximum += float(exact_distance_tolerance)
        elif relation_name == "FAR_FROM":
            if minimum is None and maximum is None and optimize != "MAXIMIZE":
                raise ValueError(
                    f"FAR_FROM relation for {obstacle['label']} needs a minimum distance or MAXIMIZE"
                )
            minimum = 0.0 if minimum is None else minimum
        else:
            raise ValueError(f"unsupported object relation: {relation_name}")

        normalized.append(
            {
                "object_id": object_id,
                "object_label": obstacle["label"],
                "relation": relation_name,
                "min_distance_m": round(minimum, 3) if minimum is not None else None,
                "max_distance_m": round(maximum, 3) if maximum is not None else None,
                "optimize": optimize,
                "object_min_corner": copy.deepcopy(obstacle["min_corner"]),
                "object_max_corner": copy.deepcopy(obstacle["max_corner"]),
            }
        )
    if not normalized:
        raise ValueError("navigation requires at least one object relation")
    if not any(item["relation"] == "NEAR" for item in normalized):
        raise ValueError("navigation requires at least one NEAR relation to anchor the goal")
    return normalized


def relation_distance(point, relation):
    return clearance_to_box(
        point,
        {
            "min_corner": relation["object_min_corner"],
            "max_corner": relation["object_max_corner"],
        },
    )


def relation_results(point, relations):
    results = []
    for relation in relations:
        distance = relation_distance(point, relation)
        minimum = relation.get("min_distance_m")
        maximum = relation.get("max_distance_m")
        satisfied = (
            (minimum is None or distance >= float(minimum) - 1e-9)
            and (maximum is None or distance <= float(maximum) + 1e-9)
        )
        results.append(
            {
                "object_id": relation["object_id"],
                "object_label": relation["object_label"],
                "relation": relation["relation"],
                "distance_m": round(distance, 3),
                "min_distance_m": minimum,
                "max_distance_m": maximum,
                "optimize": relation.get("optimize", "NONE"),
                "satisfied": satisfied,
            }
        )
    return results


def range_constrained_goal(start, relations, planning_obstacles, workspace, fixed_z, clearance, resolution):
    """Sample the bounded workspace and select a point satisfying every relation."""
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("goal_sample_resolution_m must be finite and positive")
    x_min, x_max = (float(value) for value in workspace["x"])
    y_min, y_max = (float(value) for value in workspace["y"])
    x_count = int(math.ceil((x_max - x_min) / resolution))
    y_count = int(math.ceil((y_max - y_min) / resolution))
    candidates = []
    for x_index in range(x_count + 1):
        x = min(x_max, x_min + x_index * resolution)
        for y_index in range(y_count + 1):
            y = min(y_max, y_min + y_index * resolution)
            point = {"x": round(x, 3), "y": round(y, 3), "z": float(fixed_z)}
            if any(clearance_to_box(point, obstacle) < float(clearance) for obstacle in planning_obstacles):
                continue
            results = relation_results(point, relations)
            if not all(item["satisfied"] for item in results):
                continue
            maximize_score = sum(
                item["distance_m"] for item in results if item["optimize"] == "MAXIMIZE"
            )
            range_error = 0.0
            for item in results:
                minimum = item["min_distance_m"]
                maximum = item["max_distance_m"]
                if minimum is not None and maximum is not None:
                    range_error += abs(item["distance_m"] - 0.5 * (float(minimum) + float(maximum)))
            travel = math.hypot(point["x"] - float(start["x"]), point["y"] - float(start["y"]))
            candidates.append(((maximize_score, -range_error, -travel), point, results))
    if not candidates:
        raise ValueError("no goal satisfies all object-distance ranges and safety constraints")
    _, goal, results = max(candidates, key=lambda item: item[0])
    return goal, results


def safe_standoff_goal(start, target, obstacles, workspace, fixed_z, requested_standoff, default_standoff, clearance):
    minimum = target["min_corner"]
    maximum = target["max_corner"]
    center_x = 0.5 * (float(minimum[0]) + float(maximum[0]))
    center_y = 0.5 * (float(minimum[1]) + float(maximum[1]))
    offset = max(float(requested_standoff), float(default_standoff), float(clearance)) + 0.02
    candidates = [
        {"x": float(minimum[0]) - offset, "y": center_y, "z": fixed_z},
        {"x": float(maximum[0]) + offset, "y": center_y, "z": fixed_z},
        {"x": center_x, "y": float(minimum[1]) - offset, "z": fixed_z},
        {"x": center_x, "y": float(maximum[1]) + offset, "z": fixed_z},
    ]
    x_limits = workspace["x"]
    y_limits = workspace["y"]
    feasible = []
    for point in candidates:
        if not (
            float(x_limits[0]) <= point["x"] <= float(x_limits[1])
            and float(y_limits[0]) <= point["y"] <= float(y_limits[1])
        ):
            continue
        if all(clearance_to_box(point, obstacle) >= float(clearance) for obstacle in obstacles):
            feasible.append({key: round(float(value), 3) for key, value in point.items()})
    if not feasible:
        raise ValueError("no collision-free standoff goal is available in the workspace")
    return min(
        feasible,
        key=lambda point: math.hypot(point["x"] - float(start["x"]), point["y"] - float(start["y"])),
    )


def obstacle_catalog(obstacles):
    return [
        {
            "object_id": item["object_id"],
            "label": item["label"],
            "min_corner": item["min_corner"],
            "max_corner": item["max_corner"],
            "front_surface_center": item.get("front_surface_center"),
            "visible_width_m": item.get("visible_width_m"),
            "front_range_m": item.get("front_range_m"),
            "view_axis_xy": item.get("view_axis_xy"),
            "lateral_axis_xy": item.get("lateral_axis_xy"),
            "confidence": item.get("confidence"),
        }
        for item in obstacles
    ]


def describe_obstacles(obstacles):
    descriptions = []
    for index, obstacle in enumerate(obstacles, start=1):
        minimum = obstacle["min_corner"]
        maximum = obstacle["max_corner"]
        descriptions.append(
            f"{index} {obstacle['label']} ({obstacle['object_id']}): "
            f"x=[{minimum[0]:.2f},{maximum[0]:.2f}], y=[{minimum[1]:.2f},{maximum[1]:.2f}."
        )
    return " ".join(descriptions) if descriptions else "No obstacles currently detected."


def describe_goal_relations(relations):
    descriptions = []
    for relation in relations:
        minimum = relation.get("min_distance_m")
        maximum = relation.get("max_distance_m")
        if minimum is not None and maximum is not None:
            distance = f"range=[{minimum:.2f},{maximum:.2f}]m"
        elif minimum is not None:
            distance = f"distance>={minimum:.2f}m"
        elif maximum is not None:
            distance = f"distance<={maximum:.2f}m"
        else:
            distance = "maximize distance"
        descriptions.append(
            f"{relation['relation']} {relation['object_label']} ({relation['object_id']}), {distance}"
        )
    return "; ".join(descriptions)


def build_planner_prompt(start, goal, workspace, obstacles, clearance, goal_relations=None):
    distance = math.hypot(float(goal["x"]) - float(start["x"]), float(goal["y"]) - float(start["y"]))
    nl_env = (
        "Mission state: the UAV has already taken off and is holding hover at the start position. "
        "Use this hover position as the first waypoint/reference for planning. "
        f"Workspace: x=[{workspace['x'][0]:.2f},{workspace['x'][1]:.2f}]m, "
        f"y=[{workspace['y'][0]:.2f},{workspace['y'][1]:.2f}]m, z={workspace['z']:.2f} fixed. "
        f"Start: ({start['x']:.2f},{start['y']:.2f},{start['z']:.2f}), "
        f"Goal: ({goal['x']:.2f},{goal['y']:.2f},{goal['z']:.2f}), "
        f"distance≈{distance:.2f}m. Obstacles with x-y spans: {describe_obstacles(obstacles)}"
    )
    if goal_relations:
        nl_env += (
            " The deterministic gateway selected the goal to satisfy these approved object-surface "
            f"distance relations: {describe_goal_relations(goal_relations)}."
        )
    constraints = (
        "Constraints:\n"
        f"- all waypoints in NED frame, z must stay {workspace['z']:.2f}\n"
        "- first waypoint must exactly match the provided hover start coordinates\n"
        "- final waypoint must be goal coordinates\n"
        f"- maintain >={clearance:.2f}m clearance from obstacle x-y boxes\n"
        "- stay within workspace\n"
        "- no waypoint should be within obstacle boxes, walls, or near corners\n"
        "- prefer sparse, smooth, monotonic progress through open space\n"
        "- return only the structured output requested by the response model"
    )
    return "\n".join((INSTRUCTIONS, nl_env, constraints)), nl_env


class InteractiveMissionGateway(Node):
    def __init__(self, intent_parser=None):
        super().__init__("interactive_mission_gateway")
        self.declare_parameter("environment", "real")
        self.declare_parameter("intent_provider", "openai")
        self.declare_parameter("openai_intent_model", "gpt-5.4-nano")
        self.declare_parameter("planner_llm_provider", "llama")
        self.declare_parameter("planner_model_name", "rrt_planner")
        self.declare_parameter("visualizer", "contraction")
        self.declare_parameter("operator_command_topic", "/llm_vision/operator_command")
        self.declare_parameter("approval_topic", "/llm_vision/mission_approval")
        self.declare_parameter("launch_approval_topic", "/llm_vision/launch_approval")
        self.declare_parameter("operator_response_topic", "/llm_vision/operator_response")
        self.declare_parameter("mission_proposal_topic", "/llm_vision/mission_proposal")
        self.declare_parameter("launch_proposal_topic", "/llm_vision/launch_proposal")
        self.declare_parameter("executor_command_topic", "/llm_vision/executor_command")
        self.declare_parameter("safety_tube_ready_topic", "/llm_vision/safety_tube_ready")
        self.declare_parameter("semantic_obstacle_topic", "/llm_vision/semantic_obstacles")
        self.declare_parameter("sim_obstacle_topic", "/llm_vision/sim_obstacles")
        self.declare_parameter("nominal_obstacle_topic", "/llm_vision/nominal_obstacles")
        self.declare_parameter("mission_state_topic", "/llm_vision/mission_state")
        self.declare_parameter("pose_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("prompt_topic", "/llm_vision/prompt")
        self.declare_parameter("candidate_verification_topic", "/llm_vision/plan_candidate_verified")
        self.declare_parameter("passed_plan_topic", "/llm_vision/plan_verified")
        self.declare_parameter("calibration_status_topic", "/llm_vision/vision_calibration_status")
        self.declare_parameter("calibration_only", False)
        self.declare_parameter("auto_calibration_capture", False)
        self.declare_parameter("continuous_calibration_capture", False)
        self.declare_parameter("calibration_capture_delay_s", 1.0)
        self.declare_parameter("calibration_capture_interval_s", 3.0)
        self.declare_parameter("required_mission_state", "HOLDING_FOR_PLAN")
        self.declare_parameter("require_mission_state", True)
        self.declare_parameter("workspace_x_min", -3.0)
        self.declare_parameter("workspace_x_max", 3.0)
        self.declare_parameter("workspace_y_min", -3.0)
        self.declare_parameter("workspace_y_max", 3.0)
        self.declare_parameter("fixed_z", -0.5)
        self.declare_parameter("clearance_m", 0.40)
        self.declare_parameter("default_standoff_m", 0.60)
        self.declare_parameter("goal_sample_resolution_m", 0.10)
        self.declare_parameter("default_goal_range_half_width_m", 0.15)
        self.declare_parameter("exact_goal_distance_tolerance_m", 0.10)
        self.declare_parameter("fresh_data_timeout_s", 2.0)
        self.declare_parameter("approval_timeout_s", 30.0)
        self.declare_parameter("freeze_scene_after_approval", True)
        self.declare_parameter("scene_position_tolerance_m", 0.15)
        self.declare_parameter("scene_size_tolerance_m", 0.20)
        self.declare_parameter("obs_safety_bracket", "conformal")
        self.declare_parameter(
            "vision_error_calibration_csv",
            "fine_tuning/datasets/calibration_vision_error_dummy.csv",
        )
        self.declare_parameter("vision_error_delta", 0.10)
        self.declare_parameter("max_start_drift_m", 0.25)
        self.declare_parameter("max_release_start_drift_m", 0.08)
        self.declare_parameter("max_planning_attempts", 3)
        self.declare_parameter("debug", True)

        self.environment = str(self.get_parameter("environment").value).strip().lower()
        self.fixed_z = float(self.get_parameter("fixed_z").value)
        self.clearance_m = float(self.get_parameter("clearance_m").value)
        self.default_standoff_m = float(self.get_parameter("default_standoff_m").value)
        self.goal_sample_resolution_m = float(self.get_parameter("goal_sample_resolution_m").value)
        self.default_goal_range_half_width_m = float(
            self.get_parameter("default_goal_range_half_width_m").value
        )
        self.exact_goal_distance_tolerance_m = float(
            self.get_parameter("exact_goal_distance_tolerance_m").value
        )
        if self.goal_sample_resolution_m <= 0.0:
            raise ValueError("goal_sample_resolution_m must be positive")
        if self.default_goal_range_half_width_m < 0.0 or self.exact_goal_distance_tolerance_m < 0.0:
            raise ValueError("goal range tolerances must be non-negative")
        self.scene_position_tolerance_m = float(self.get_parameter("scene_position_tolerance_m").value)
        self.scene_size_tolerance_m = float(self.get_parameter("scene_size_tolerance_m").value)
        if self.scene_position_tolerance_m < 0.0 or self.scene_size_tolerance_m < 0.0:
            raise ValueError("scene tolerances must be non-negative")
        self.hardcoded_scene_guard_band_m = (
            self.scene_position_tolerance_m + 0.5 * self.scene_size_tolerance_m
        )
        self.obs_safety_bracket = str(self.get_parameter("obs_safety_bracket").value).strip().lower()
        if self.obs_safety_bracket not in ("hardcoded", "conformal"):
            raise ValueError("obs_safety_bracket must be 'hardcoded' or 'conformal'")
        if self.obs_safety_bracket == "conformal":
            self.vision_error_certificate = load_vision_error_certificate(
                str(self.get_parameter("vision_error_calibration_csv").value),
                float(self.get_parameter("vision_error_delta").value),
            )
            self.scene_guard_band_m = float(self.vision_error_certificate["quantile_m"])
        else:
            self.vision_error_certificate = {
                "quantile_m": None,
                "delta": float(self.get_parameter("vision_error_delta").value),
                "trial_count": 0,
                "rank": None,
                "file": "",
                "placeholder": False,
            }
            self.scene_guard_band_m = self.hardcoded_scene_guard_band_m
        self.workspace = {
            "x": [
                float(self.get_parameter("workspace_x_min").value),
                float(self.get_parameter("workspace_x_max").value),
            ],
            "y": [
                float(self.get_parameter("workspace_y_min").value),
                float(self.get_parameter("workspace_y_max").value),
            ],
            "z": self.fixed_z,
        }
        self.state = "WAITING_FOR_COMMAND"
        self.latest_scene = None
        self.latest_scene_received_s = None
        self.latest_mission_state = None
        self.latest_mission_state_received_s = None
        self.current_pose = None
        self.conversation = []
        self.active_mission = None
        self.active_plan_id = None
        self.pending_verified_plan = None
        self.latest_safety_tube_ready = None
        self.last_failure = None
        self.calibration_only = bool(self.get_parameter("calibration_only").value)
        self.auto_calibration_capture = bool(
            self.get_parameter("auto_calibration_capture").value
        )
        self.continuous_calibration_capture = bool(
            self.get_parameter("continuous_calibration_capture").value
        )
        self.calibration_frame_ready_s = None
        self.calibration_capture_count = 0
        self.last_calibration_capture_s = None
        self.require_safety_tubes = (
            str(self.get_parameter("visualizer").value).strip().lower() == "contraction"
        )
        self.intent_request_token = None
        self.intent_results = queue.Queue()
        self.intent_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intent_parser")
        self.intent_parser = intent_parser or self.create_intent_parser()

        obstacle_topic = (
            str(self.get_parameter("sim_obstacle_topic").value)
            if self.environment == "sim"
            else str(self.get_parameter("semantic_obstacle_topic").value)
        )
        self.create_subscription(String, obstacle_topic, self.obstacle_callback, 10)
        self.create_subscription(
            String,
            str(self.get_parameter("mission_state_topic").value),
            self.mission_state_callback,
            10,
        )
        self.create_subscription(
            VehicleOdometry,
            str(self.get_parameter("pose_topic").value),
            self.pose_callback,
            ODOM_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("operator_command_topic").value),
            self.command_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("approval_topic").value),
            self.approval_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("launch_approval_topic").value),
            self.launch_approval_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("candidate_verification_topic").value),
            self.verification_callback,
            LATCHED_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("safety_tube_ready_topic").value),
            self.safety_tube_ready_callback,
            LATCHED_QOS,
        )
        if self.calibration_only:
            self.create_subscription(
                String,
                str(self.get_parameter("calibration_status_topic").value),
                self.calibration_status_callback,
                LATCHED_QOS,
            )
        self.response_pub = self.create_publisher(
            String,
            str(self.get_parameter("operator_response_topic").value),
            LATCHED_QOS,
        )
        self.proposal_pub = self.create_publisher(
            String,
            str(self.get_parameter("mission_proposal_topic").value),
            LATCHED_QOS,
        )
        self.launch_proposal_pub = self.create_publisher(
            String,
            str(self.get_parameter("launch_proposal_topic").value),
            LATCHED_QOS,
        )
        self.executor_command_pub = self.create_publisher(
            String,
            str(self.get_parameter("executor_command_topic").value),
            10,
        )
        self.prompt_pub = self.create_publisher(
            String,
            str(self.get_parameter("prompt_topic").value),
            LATCHED_QOS,
        )
        self.nominal_obstacle_pub = self.create_publisher(
            String,
            str(self.get_parameter("nominal_obstacle_topic").value),
            LATCHED_QOS,
        )
        self.passed_plan_pub = self.create_publisher(
            String,
            str(self.get_parameter("passed_plan_topic").value),
            LATCHED_QOS,
        )
        self.create_timer(0.05, self.drain_intent_results)
        if self.calibration_only and self.auto_calibration_capture:
            self.create_timer(0.20, self.maybe_start_calibration_capture)
        self.get_logger().info(
            f"interactive gateway ready: intent_provider={self.get_parameter('intent_provider').value}, "
            f"obstacles={obstacle_topic}, obs_safety_bracket={self.obs_safety_bracket}, "
            f"scene_guard_band={self.scene_guard_band_m:.3f} m"
        )
        if self.obs_safety_bracket == "conformal" and self.vision_error_certificate["placeholder"]:
            self.get_logger().warning(
                "using the placeholder vision-error certificate; real conformal missions will fail closed"
            )

    def create_intent_parser(self):
        provider = str(self.get_parameter("intent_provider").value).strip().lower()
        if provider == "mock":
            return MockIntentParser()
        if provider != "openai":
            raise ValueError(f"unsupported intent_provider: {provider}")
        return OpenAIIntentParser(str(self.get_parameter("openai_intent_model").value))

    def obstacle_safety_metadata(self):
        certificate = self.vision_error_certificate
        return {
            "obs_safety_bracket": self.obs_safety_bracket,
            "vision_error_quantile_m": certificate["quantile_m"],
            "vision_error_delta": certificate["delta"],
            "vision_error_calibration_trials": certificate["trial_count"],
            "vision_error_calibration_rank": certificate["rank"],
            "vision_error_calibration_csv": certificate["file"],
            "vision_error_calibration_placeholder": certificate["placeholder"],
            "scene_guard_band_m": self.scene_guard_band_m,
        }

    def environment_obstacles(self, scene, intent):
        observed = scene["obstacles"]
        if self.obs_safety_bracket == "hardcoded":
            nominal = copy.deepcopy(observed)
        elif self.environment == "sim":
            nominal = copy.deepcopy(observed)
        else:
            if self.vision_error_certificate["placeholder"]:
                raise ValueError(
                    "real conformal missions require a non-placeholder vision calibration dataset"
                )
            nominal = nominal_obstacles_from_depth(observed, intent.depth_estimates)
        return nominal, inflate_obstacles_xy(nominal, self.scene_guard_band_m)

    def calibration_nominal_obstacles(self, scene, intent, mission_nominal):
        if self.obs_safety_bracket != "hardcoded" or self.environment == "sim":
            return mission_nominal
        try:
            return nominal_obstacles_from_depth(scene["obstacles"], intent.depth_estimates)
        except ValueError as exc:
            self.get_logger().warning(
                f"hardcoded mission remains available, but no calibration snapshot was published: {exc}"
            )
            return None

    def publish_nominal_obstacles(self, scene, nominal, snapshot_id, intent):
        if nominal is None:
            return
        if self.environment == "sim":
            source = "sim_complete_boxes"
        else:
            source = "gpt_nano_front_footprint"
        try:
            observation_timestamp = float(scene.get("timestamp", time.time()))
        except (TypeError, ValueError):
            observation_timestamp = time.time()
        if not math.isfinite(observation_timestamp) or observation_timestamp <= 0.0:
            observation_timestamp = time.time()
        payload = {
            "snapshot_id": snapshot_id,
            "timestamp": observation_timestamp,
            "published_at": time.time(),
            "frame": "local_ned",
            "source": source,
            "obstacles": copy.deepcopy(nominal),
            "observed_obstacles": copy.deepcopy(scene["obstacles"]),
            "observer_pose": copy.deepcopy(scene.get("pose")),
            "depth_estimates": [item.model_dump() for item in intent.depth_estimates],
            **self.obstacle_safety_metadata(),
        }
        self.nominal_obstacle_pub.publish(String(data=json.dumps(payload)))

    def obstacle_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            payload["obstacles"] = normalize_obstacles(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.get_logger().error(f"invalid obstacle snapshot: {exc}")
            return
        self.latest_scene = payload
        self.latest_scene_received_s = time.time()

    def mission_state_callback(self, msg):
        try:
            self.latest_mission_state = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid mission state: {exc}")
            return
        self.latest_mission_state_received_s = time.time()

    def pose_callback(self, msg):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            return
        self.current_pose = {
            "x": float(msg.position[0]),
            "y": float(msg.position[1]),
            "z": float(msg.position[2]),
            "received_s": time.time(),
        }

    def command_callback(self, msg):
        operator_text = self.command_text(msg.data)
        if not operator_text:
            self.publish_response("ERROR", "Operator command is empty.")
            return
        if operator_text.strip().lower() == "cancel":
            self.cancel_active_mission("Mission cancelled by operator.")
            return
        if self.intent_request_token is not None:
            self.publish_response("BUSY", "Another command is still being interpreted.")
            return
        error = self.scene_context_error()
        if error:
            self.publish_response("NOT_READY", error)
            return

        self.start_intent_request(operator_text, copy.deepcopy(self.latest_scene))
        self.publish_response("PARSING_INTENT", "Interpreting the command against the current object snapshot.")

    def start_intent_request(self, operator_text, scene):
        token = uuid.uuid4().hex
        self.intent_request_token = token
        if not self.calibration_only:
            self.conversation.append({"role": "operator", "content": operator_text})
        catalog = obstacle_catalog(scene["obstacles"])
        future = self.intent_executor.submit(
            self.intent_parser.parse,
            operator_text,
            catalog,
            [] if self.calibration_only else list(self.conversation),
        )
        future.add_done_callback(
            lambda completed: self.intent_results.put((token, completed, scene, operator_text))
        )

    def calibration_status_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if payload.get("status") == "FRAME_READY" and self.calibration_frame_ready_s is None:
            self.calibration_frame_ready_s = time.time()
            self.publish_response(
                "CALIBRATION_FRAME_READY",
                "Vicon-world to local-NED alignment was derived automatically; waiting for a stable perception snapshot.",
            )

    def maybe_start_calibration_capture(self):
        if (
            self.intent_request_token is not None
            or self.calibration_frame_ready_s is None
        ):
            return
        if self.calibration_capture_count and not self.continuous_calibration_capture:
            return
        now = time.time()
        if self.last_calibration_capture_s is None:
            delay = max(0.0, float(self.get_parameter("calibration_capture_delay_s").value))
            reference_s = self.calibration_frame_ready_s
        else:
            delay = max(0.1, float(self.get_parameter("calibration_capture_interval_s").value))
            reference_s = self.last_calibration_capture_s
        if now - reference_s < delay:
            return
        if self.context_error() is not None:
            return
        self.calibration_capture_count += 1
        self.last_calibration_capture_s = now
        scene = copy.deepcopy(self.latest_scene)
        if not scene["obstacles"]:
            intent = MissionIntent(
                status="READY",
                intent_type="QUERY",
                navigation_action="NONE",
                query_type="DESCRIBE_SCENE",
                relations=[],
                query_object_ids=[],
                depth_estimates=[],
                clarifying_question="",
            )
            self.handle_calibration_intent(intent, scene)
            return
        self.start_intent_request("Describe all visible objects for vision calibration.", scene)
        self.publish_response(
            "CALIBRATION_ESTIMATING_DEPTH",
            f"Continuous calibration capture {self.calibration_capture_count}: GPT nano is estimating nominal object depths.",
            capture_index=self.calibration_capture_count,
        )

    @staticmethod
    def command_text(raw):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()
        if isinstance(payload, dict):
            return str(payload.get("text", "")).strip()
        return str(payload).strip()

    def drain_intent_results(self):
        while True:
            try:
                token, future, scene, operator_text = self.intent_results.get_nowait()
            except queue.Empty:
                return
            if token != self.intent_request_token:
                continue
            self.intent_request_token = None
            try:
                intent = future.result()
            except Exception as exc:
                self.publish_response("INTENT_ERROR", f"Intent service failed; no mission was released: {exc}")
                continue
            self.handle_intent(intent, scene, operator_text)

    def handle_intent(self, intent, scene, operator_text):
        if intent.status == "CANCELLED":
            self.cancel_active_mission("Mission cancelled by parsed operator intent.")
            return
        if intent.status != "READY":
            default_message = (
                "Flight-control requests are not supported by this interface."
                if intent.status == "UNSUPPORTED"
                else "Please clarify the requested detected object or distance range."
            )
            question = intent.clarifying_question or default_message
            self.conversation.append({"role": "system", "content": question})
            self.publish_response(intent.status, question)
            return
        if self.calibration_only:
            self.handle_calibration_intent(intent, scene)
            return
        if intent.intent_type == "QUERY":
            self.handle_query(intent, scene)
            return

        if intent.intent_type != "NAVIGATION" or intent.navigation_action not in ("HOVER", "GO_TO"):
            self.publish_response(
                "UNSUPPORTED",
                "Supported commands are scene questions and object-relative HOVER or GO_TO navigation.",
            )
            return
        if self.state in (
            "WAITING_FOR_VERIFICATION",
            "FORMING_SAFETY_TUBES",
            "AWAITING_LAUNCH_APPROVAL",
            "LAND_REQUESTED",
            "RELEASED_TO_EXECUTOR",
        ):
            self.publish_response("BUSY", f"Cannot replace the active mission while state={self.state}.")
            return
        error = self.context_error()
        if error:
            self.publish_response("NOT_READY", error)
            return
        start = self.current_start()
        try:
            nominal_obstacles, planning_obstacles = self.environment_obstacles(scene, intent)
        except ValueError as exc:
            self.publish_response("ENVIRONMENT_UNCERTAIN", f"No mission was released: {exc}")
            return
        try:
            goal_relations = normalize_goal_relations(
                intent.relations,
                nominal_obstacles,
                default_standoff=self.default_standoff_m,
                clearance=self.clearance_m,
                guard_band=self.scene_guard_band_m,
                default_range_half_width=self.default_goal_range_half_width_m,
                exact_distance_tolerance=self.exact_goal_distance_tolerance_m,
            )
            goal, goal_relation_results = range_constrained_goal(
                start,
                goal_relations,
                planning_obstacles,
                self.workspace,
                self.fixed_z,
                self.clearance_m,
                self.goal_sample_resolution_m,
            )
        except ValueError as exc:
            message = str(exc)
            status = "NO_SAFE_GOAL" if message.startswith("no goal") else "NEEDS_CLARIFICATION"
            self.publish_response(status, message)
            return

        target_id = next(item["object_id"] for item in goal_relations if item["relation"] == "NEAR")
        target = next(item for item in nominal_obstacles if item["object_id"] == target_id)

        created_at = time.time()
        mission_id = f"M-{int(created_at * 1000)}-{uuid.uuid4().hex[:6]}"
        snapshot_hash = scene_signature(scene["obstacles"])
        snapshot_id = f"S-{snapshot_hash[:12]}"
        calibration_nominal = self.calibration_nominal_obstacles(
            scene,
            intent,
            nominal_obstacles,
        )
        mission = {
            "mission_id": mission_id,
            "snapshot_id": snapshot_id,
            "scene_signature": snapshot_hash,
            "operator_text": operator_text,
            "intent": intent.model_dump(),
            "target": target,
            "start": start,
            "goal": goal,
            "goal_relations": goal_relations,
            "goal_relation_results": goal_relation_results,
            "workspace": copy.deepcopy(self.workspace),
            "observed_obstacles": copy.deepcopy(scene["obstacles"]),
            "nominal_obstacles": copy.deepcopy(nominal_obstacles),
            "obstacles": planning_obstacles,
            "obstacle_safety": self.obstacle_safety_metadata(),
            "created_at": created_at,
            "attempt": 0,
        }
        self.publish_nominal_obstacles(scene, calibration_nominal, snapshot_id, intent)
        # The live hover position is rebound at approval within a configured limit.
        contract = {
            key: mission[key]
            for key in (
                "mission_id",
                "snapshot_id",
                "intent",
                "target",
                "goal",
                "goal_relations",
                "workspace",
                "nominal_obstacles",
                "obstacles",
                "obstacle_safety",
            )
        }
        mission["proposal_hash"] = hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()
        mission["base_prompt"], mission["nl_env"] = build_planner_prompt(
            start,
            goal,
            mission["workspace"],
            mission["obstacles"],
            self.clearance_m,
            mission["goal_relations"],
        )
        self.active_mission = mission
        self.active_plan_id = None
        self.pending_verified_plan = None
        self.latest_safety_tube_ready = None
        self.state = "AWAITING_APPROVAL"
        proposal = self.proposal_payload(mission)
        self.proposal_pub.publish(String(data=json.dumps(proposal)))
        self.publish_response(
            "AWAITING_APPROVAL",
            (
                f"I grounded {len(goal_relations)} object relation(s) and propose goal "
                f"({goal['x']:.2f}, {goal['y']:.2f}, {goal['z']:.2f}). "
                f"Relations: {describe_goal_relations(goal_relations)}. All detected objects remain obstacles. "
                f"Approve mission {mission_id}?"
            ),
            mission_id=mission_id,
        )

    def handle_calibration_intent(self, intent, scene):
        nominal = []
        failures = []
        for obstacle in scene["obstacles"]:
            object_id = str(obstacle["object_id"])
            estimates = [
                estimate
                for estimate in intent.depth_estimates
                if estimate.object_id == object_id
            ]
            try:
                nominal.extend(nominal_obstacles_from_depth([obstacle], estimates))
            except ValueError as exc:
                failures.append({"object_id": object_id, "reason": str(exc)})
        snapshot_id = f"CAL-{scene_signature(scene['obstacles'])[:12]}-{uuid.uuid4().hex[:6]}"
        self.publish_nominal_obstacles(scene, nominal, snapshot_id, intent)
        self.state = "CALIBRATION_SNAPSHOT_PUBLISHED"
        failure_note = (
            f" {len(failures)} invalid or abstaining object(s) were omitted and will be recorded as misses."
            if failures
            else ""
        )
        self.publish_response(
            "CALIBRATION_SNAPSHOT_PUBLISHED",
            f"Published one nominal calibration snapshot containing {len(nominal)} object(s); no mission was planned or released.{failure_note}",
            snapshot_id=snapshot_id,
            object_count=len(nominal),
            failures=failures,
        )

    def handle_query(self, intent, scene):
        query_type = intent.query_type
        obstacles = scene["obstacles"]
        by_id = {item["object_id"]: item for item in obstacles}
        selected = [by_id[object_id] for object_id in intent.query_object_ids if object_id in by_id]
        if query_type == "LOCATE_OBJECT" and len(selected) != len(intent.query_object_ids):
            self.publish_response(
                "NEEDS_CLARIFICATION",
                "The selected object ID is not in the current snapshot; ask about one of the listed objects.",
            )
            return

        if query_type == "DESCRIBE_SCENE":
            details = "; ".join(self.object_location(item) for item in obstacles) or "no detected objects"
            message = f"The current healthy scene contains {len(obstacles)} object(s): {details}."
        elif query_type == "LIST_OBJECTS":
            details = "; ".join(self.object_location(item) for item in obstacles) or "none"
            message = f"Detected objects ({len(obstacles)}): {details}."
        elif query_type == "LOCATE_OBJECT":
            if not selected:
                self.publish_response(
                    "NEEDS_CLARIFICATION",
                    "Which detected object should I locate?",
                )
                return
            message = "Object location: " + "; ".join(self.object_location(item) for item in selected) + "."
        elif query_type == "EXPLAIN_PROPOSAL":
            if self.active_mission is None:
                message = "There is no active mission proposal to explain."
            else:
                mission = self.active_mission
                message = (
                    f"Mission {mission['mission_id']} proposes goal "
                    f"({mission['goal']['x']:.2f}, {mission['goal']['y']:.2f}, {mission['goal']['z']:.2f}) "
                    f"from {len(mission['goal_relations'])} approved relation(s): "
                    f"{describe_goal_relations(mission['goal_relations'])}. The path must keep "
                    f"{self.clearance_m:.2f} m from obstacles after a {self.scene_guard_band_m:.2f} m scene guard band."
                )
        elif query_type == "EXPLAIN_FAILURE":
            if self.last_failure is None:
                message = "No intent, grounding, approval, or verification failure has been recorded."
            else:
                failed = self.last_failure.get("failed_constraints", [])
                suffix = f" Failed constraints: {', '.join(failed)}." if failed else ""
                message = (
                    f"Last failure ({self.last_failure['status']}): "
                    f"{self.last_failure['message'].rstrip('.')}.{suffix}"
                )
        else:
            self.publish_response("UNSUPPORTED", "That non-motion query is not supported.")
            return

        self.conversation.append({"role": "system", "content": message})
        self.publish_response(
            "QUERY_RESULT",
            message,
            query_type=query_type,
            object_ids=[item["object_id"] for item in selected],
        )

    def object_location(self, obstacle):
        minimum = obstacle["min_corner"]
        maximum = obstacle["max_corner"]
        center_x = 0.5 * (float(minimum[0]) + float(maximum[0]))
        center_y = 0.5 * (float(minimum[1]) + float(maximum[1]))
        return f"{obstacle['label']} ({obstacle['object_id']}) centered at ({center_x:.2f}, {center_y:.2f}) m NED"

    def proposal_payload(self, mission):
        target = mission["target"]
        minimum = target["min_corner"]
        maximum = target["max_corner"]
        return {
            "type": "MISSION_PROPOSAL",
            "status": (
                "ENVIRONMENT_CAPTURED"
                if mission.get("environment_frozen", False)
                else "AWAITING_APPROVAL"
            ),
            "mission_id": mission["mission_id"],
            "snapshot_id": mission["snapshot_id"],
            "proposal_hash": mission["proposal_hash"],
            "operator_text": mission["operator_text"],
            "target": {
                "object_id": target["object_id"],
                "label": target["label"],
                "center": [
                    round(0.5 * (float(minimum[0]) + float(maximum[0])), 3),
                    round(0.5 * (float(minimum[1]) + float(maximum[1])), 3),
                    self.fixed_z,
                ],
            },
            "goal": mission["goal"],
            "workspace": copy.deepcopy(mission["workspace"]),
            "goal_relations": mission["goal_relations"],
            "goal_relation_results": mission["goal_relation_results"],
            "clearance_m": self.clearance_m,
            "nominal_obstacles": copy.deepcopy(mission["nominal_obstacles"]),
            "obstacles": copy.deepcopy(mission["obstacles"]),
            "obstacle_ids": [item["object_id"] for item in mission["obstacles"]],
            "created_at": mission["created_at"],
            "observed_obstacles": copy.deepcopy(mission["observed_obstacles"]),
            "environment_frozen": bool(mission.get("environment_frozen", False)),
            "environment_frozen_at": mission.get("environment_frozen_at"),
            **copy.deepcopy(mission["obstacle_safety"]),
        }

    def approval_callback(self, msg):
        try:
            approval = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.publish_response("APPROVAL_REJECTED", f"Approval must be structured JSON: {exc}")
            return
        if self.state != "AWAITING_APPROVAL" or self.active_mission is None:
            self.publish_response("APPROVAL_REJECTED", "There is no mission awaiting approval.")
            return
        decision = str(approval.get("decision", approval.get("type", ""))).upper()
        if decision in ("REJECT", "REJECTED", "CANCEL"):
            self.cancel_active_mission("Mission proposal rejected by operator.")
            return
        if decision not in ("APPROVE", "APPROVED"):
            self.publish_response("APPROVAL_REJECTED", "Decision must be APPROVE or REJECT.")
            return
        mission = self.active_mission
        if approval.get("mission_id") != mission["mission_id"]:
            self.publish_response("APPROVAL_REJECTED", "Approval mission_id does not match the active proposal.")
            return
        if approval.get("proposal_hash") != mission["proposal_hash"]:
            self.publish_response("APPROVAL_REJECTED", "Approval proposal_hash does not match the active proposal.")
            return
        if time.time() - mission["created_at"] > float(self.get_parameter("approval_timeout_s").value):
            self.cancel_active_mission("Mission proposal expired; request a new sensor-grounded proposal.")
            return
        error = self.context_error()
        if error:
            self.cancel_active_mission(f"Approval invalidated: {error}")
            return
        scene_error = self.scene_compatibility_error(mission)
        if scene_error:
            self.cancel_active_mission(f"Scene changed before approval: {scene_error}.")
            return
        current_position = self.current_position()
        drift = math.dist(
            [current_position["x"], current_position["y"], current_position["z"]],
            [mission["start"]["x"], mission["start"]["y"], mission["start"]["z"]],
        )
        if drift > float(self.get_parameter("max_start_drift_m").value):
            self.cancel_active_mission(f"Start moved {drift:.2f} m before approval; request a new proposal.")
            return
        mission["start"] = {
            "x": round(current_position["x"], 3),
            "y": round(current_position["y"], 3),
            "z": self.fixed_z,
        }
        if not self.point_in_workspace(mission["start"]):
            self.cancel_active_mission("Current hover position is outside the planning workspace.")
            return
        if any(clearance_to_box(mission["start"], obstacle) < self.clearance_m for obstacle in mission["obstacles"]):
            self.cancel_active_mission("Current hover position does not satisfy obstacle clearance.")
            return
        mission["base_prompt"], mission["nl_env"] = build_planner_prompt(
            mission["start"],
            mission["goal"],
            mission["workspace"],
            mission["obstacles"],
            self.clearance_m,
            mission["goal_relations"],
        )
        mission["approved_at"] = time.time()
        mission["environment_frozen"] = bool(
            self.get_parameter("freeze_scene_after_approval").value
        )
        mission["environment_frozen_at"] = (
            mission["approved_at"] if mission["environment_frozen"] else None
        )
        if mission["environment_frozen"]:
            self.proposal_pub.publish(String(data=json.dumps(self.proposal_payload(mission))))
        self.publish_planning_attempt()

    def publish_planning_attempt(self, feedback=None):
        mission = self.active_mission
        mission["attempt"] += 1
        plan_id = f"{mission['mission_id']}-A{mission['attempt']:02d}"
        prompt = mission["base_prompt"]
        if feedback:
            failed = feedback.get("failed_constraints", [])
            table = feedback.get("verification_feedback_table") or feedback.get("metrics", {}).get("feedback_table", "")
            prompt += (
                "\nPrevious plan failed verification. Regenerate a sparse waypoint plan for the exact same "
                "approved start, goal, workspace, and obstacle snapshot.\n"
                f"Failed constraints: {', '.join(failed) if failed else 'unknown'}\n{table}\n"
                "Prefer larger obstacle clearance, monotonic goal progress, and smoother segment changes."
            )
        envelope = {
            "mission_id": mission["mission_id"],
            "snapshot_id": mission["snapshot_id"],
            "proposal_hash": mission["proposal_hash"],
            "plan_id": plan_id,
            "attempt": mission["attempt"],
            "prompt": prompt,
            "nl_env": mission["nl_env"],
            "start": mission["start"],
            "goal": mission["goal"],
            "goal_relations": mission["goal_relations"],
            "workspace": mission["workspace"],
            "nominal_obstacles": mission["nominal_obstacles"],
            "obstacles": mission["obstacles"],
            "obstacle_safety": mission["obstacle_safety"],
            **mission["obstacle_safety"],
            "timestamp": time.time(),
            "llm_provider": str(self.get_parameter("planner_llm_provider").value),
            "requested_model": str(self.get_parameter("planner_model_name").value),
        }
        self.active_plan_id = plan_id
        self.state = "WAITING_FOR_VERIFICATION"
        self.prompt_pub.publish(String(data=json.dumps(envelope)))
        self.publish_response(
            "PLANNING",
            f"Approved mission {mission['mission_id']}; planning attempt {mission['attempt']} is running.",
            mission_id=mission["mission_id"],
            plan_id=plan_id,
        )

    def verification_callback(self, msg):
        try:
            candidate = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid candidate verification result: {exc}")
            return
        if self.state != "WAITING_FOR_VERIFICATION" or self.active_mission is None:
            return
        if str(candidate.get("plan_id")) != str(self.active_plan_id):
            self.log_debug(f"ignoring stale candidate plan_id={candidate.get('plan_id')}")
            return
        context_error = self.planning_context_error()
        if context_error:
            self.cancel_active_mission(f"Planning context became unsafe: {context_error}")
            return
        if not self.environment_frozen():
            scene_error = self.scene_compatibility_error(self.active_mission)
            if scene_error:
                self.cancel_active_mission(f"Scene changed during planning: {scene_error}.")
                return
        if not self.candidate_contract_matches(candidate):
            self.cancel_active_mission("Candidate plan contract differs from the approved mission.")
            return
        if bool(candidate.get("passed", False)):
            current_position = self.current_position()
            release_drift = math.dist(
                [current_position["x"], current_position["y"], current_position["z"]],
                [
                    self.active_mission["start"]["x"],
                    self.active_mission["start"]["y"],
                    self.active_mission["start"]["z"],
                ],
            )
            release_limit = float(self.get_parameter("max_release_start_drift_m").value)
            if release_drift > release_limit:
                self.cancel_active_mission(
                    f"Hover moved {release_drift:.2f} m during planning; request a new proposal."
                )
                return
            output = dict(candidate)
            output.update(
                {
                    "mission_id": self.active_mission["mission_id"],
                    "snapshot_id": self.active_mission["snapshot_id"],
                    "proposal_hash": self.active_mission["proposal_hash"],
                    "approved_at": self.active_mission["approved_at"],
                    "observed_obstacles": copy.deepcopy(
                        self.active_mission["observed_obstacles"]
                    ),
                    "nominal_obstacles": copy.deepcopy(
                        self.active_mission["nominal_obstacles"]
                    ),
                    "obstacle_safety": copy.deepcopy(
                        self.active_mission["obstacle_safety"]
                    ),
                    "goal_relations": copy.deepcopy(self.active_mission["goal_relations"]),
                    **copy.deepcopy(self.active_mission["obstacle_safety"]),
                }
            )
            self.pending_verified_plan = output
            self.state = "FORMING_SAFETY_TUBES"
            self.maybe_publish_launch_proposal()
            if self.state == "FORMING_SAFETY_TUBES":
                self.publish_response(
                    "FORMING_SAFETY_TUBES",
                    f"Plan {self.active_plan_id} passed verification; latching its conformal safety tubes.",
                    mission_id=self.active_mission["mission_id"],
                    plan_id=self.active_plan_id,
                )
            return
        self.last_failure = {
            "status": "VERIFICATION_FAILED",
            "message": f"Plan {self.active_plan_id} did not pass deterministic verification",
            "failed_constraints": list(candidate.get("failed_constraints", [])),
            "verification_feedback_table": candidate.get("verification_feedback_table", ""),
            "timestamp": time.time(),
        }
        if self.active_mission["attempt"] >= int(self.get_parameter("max_planning_attempts").value):
            self.state = "PLANNING_FAILED"
            self.publish_response(
                "PLANNING_FAILED",
                "No planning attempt passed verification; the vehicle remains holding.",
                mission_id=self.active_mission["mission_id"],
                plan_id=self.active_plan_id,
                failed_constraints=list(candidate.get("failed_constraints", [])),
                verification_feedback_table=candidate.get("verification_feedback_table", ""),
            )
            return
        self.publish_planning_attempt(feedback=candidate)

    def safety_tube_ready_callback(self, msg):
        try:
            self.latest_safety_tube_ready = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"invalid safety-tube readiness JSON: {exc}")
            return
        tube_status = self.latest_safety_tube_ready
        if (
            self.state == "FORMING_SAFETY_TUBES"
            and tube_status.get("status") == "FAILED"
            and tube_status.get("plan_id") == self.active_plan_id
        ):
            message = str(tube_status.get("reason") or "the swept safety tube intersects an obstacle")
            self.last_failure = {
                "status": "SAFETY_TUBE_FAILED",
                "message": message,
                "failed_constraints": ["swept_tube_clearance"],
                "timestamp": time.time(),
            }
            self.pending_verified_plan = None
            self.state = "SAFETY_TUBE_FAILED"
            self.publish_response(
                "SAFETY_TUBE_FAILED",
                f"Launch blocked: {message}.",
                mission_id=self.active_mission["mission_id"],
                plan_id=self.active_plan_id,
                minimum_clearance_m=tube_status.get("minimum_clearance_m"),
                required_radius_m=tube_status.get("required_radius_m"),
                failed_constraints=["swept_tube_clearance"],
            )
            return
        self.maybe_publish_launch_proposal()

    def maybe_publish_launch_proposal(self):
        if self.state != "FORMING_SAFETY_TUBES" or self.pending_verified_plan is None:
            return
        tube_status = self.latest_safety_tube_ready or {}
        tube_assessment_ready = (
            tube_status.get("status") in ("READY", "WARNING")
            and tube_status.get("plan_id") == self.active_plan_id
            and int(tube_status.get("sample_count", 0)) > 0
        )
        if self.require_safety_tubes and not tube_assessment_ready:
            return
        prediction_certified = bool(
            tube_status.get("tube_gate_passed", tube_status.get("status") == "READY")
        )
        safety_warning = str(tube_status.get("safety_warning") or "")
        if (
            tube_status.get("status") == "WARNING"
            and not prediction_certified
            and not safety_warning
        ):
            safety_warning = (
                "LLM prediction safety not certified: the predicted tube intersects an enlarged obstacle. "
                "Human approval is required to continue."
            )
        launch_proposal = {
            "type": "LAUNCH_PROPOSAL",
            "status": "AWAITING_LAUNCH_APPROVAL",
            "mission_id": self.active_mission["mission_id"],
            "plan_id": self.active_plan_id,
            "proposal_hash": self.active_mission["proposal_hash"],
            "goal": self.active_mission["goal"],
            "goal_relations": self.active_mission["goal_relations"],
            "verified_at": time.time(),
            "conformal_safety_tubes_ready": tube_assessment_ready,
            "safety_tube_samples": tube_status.get("sample_count", 0),
            "tube_gate_passed": prediction_certified,
            "llm_prediction_safety_certified": prediction_certified,
            "safety_warning": safety_warning,
            "tube_colliding_obstacle_ids": tube_status.get("colliding_obstacle_ids", []),
            "tube_minimum_clearance_m": tube_status.get("minimum_clearance_m"),
            "tube_required_radius_m": tube_status.get("required_radius_m"),
            "q_p_scope": tube_status.get("q_p_scope", "simulated_rrt_relative_cross_track"),
            **copy.deepcopy(self.active_mission["obstacle_safety"]),
        }
        self.state = "AWAITING_LAUNCH_APPROVAL"
        self.launch_proposal_pub.publish(String(data=json.dumps(launch_proposal)))
        self.publish_response(
            "AWAITING_LAUNCH_APPROVAL",
            (
                safety_warning
                if safety_warning
                else (
                    f"Plan {self.active_plan_id} passed verification. Review the latched trajectory "
                    "and conformal safety tubes, then approve launch or terminate and land."
                )
            ),
            mission_id=self.active_mission["mission_id"],
            plan_id=self.active_plan_id,
            llm_prediction_safety_certified=prediction_certified,
            safety_warning=safety_warning,
        )

    def launch_approval_callback(self, msg):
        try:
            approval = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.publish_response("LAUNCH_APPROVAL_REJECTED", f"Launch decision must be structured JSON: {exc}")
            return
        if self.state != "AWAITING_LAUNCH_APPROVAL" or self.pending_verified_plan is None:
            self.publish_response("LAUNCH_APPROVAL_REJECTED", "There is no verified plan awaiting launch approval.")
            return
        mission = self.active_mission
        if approval.get("mission_id") != mission["mission_id"]:
            self.publish_response("LAUNCH_APPROVAL_REJECTED", "Launch mission_id does not match the active mission.")
            return
        if approval.get("plan_id") != self.active_plan_id:
            self.publish_response("LAUNCH_APPROVAL_REJECTED", "Launch plan_id does not match the verified plan.")
            return
        if approval.get("proposal_hash") != mission["proposal_hash"]:
            self.publish_response("LAUNCH_APPROVAL_REJECTED", "Launch proposal_hash does not match the active mission.")
            return

        decision = str(approval.get("decision", approval.get("type", ""))).upper()
        if decision in ("DENY", "REJECT", "REJECTED", "TERMINATE", "CANCEL"):
            self.request_executor_land("Verified mission terminated by operator before launch.")
            return
        if decision not in ("APPROVE", "APPROVED", "LAUNCH"):
            self.publish_response("LAUNCH_APPROVAL_REJECTED", "Decision must be APPROVE or DENY.")
            return

        error = self.planning_context_error()
        if error:
            self.publish_response("LAUNCH_APPROVAL_REJECTED", f"Launch context is no longer safe: {error}")
            return
        if not self.environment_frozen():
            scene_error = self.scene_compatibility_error(mission)
            if scene_error:
                self.publish_response("LAUNCH_APPROVAL_REJECTED", f"Scene changed before launch: {scene_error}.")
                return
        current_position = self.current_position()
        release_drift = math.dist(
            [current_position["x"], current_position["y"], current_position["z"]],
            [mission["start"]["x"], mission["start"]["y"], mission["start"]["z"]],
        )
        release_limit = float(self.get_parameter("max_release_start_drift_m").value)
        if release_drift > release_limit:
            self.publish_response(
                "LAUNCH_APPROVAL_REJECTED",
                f"Hover moved {release_drift:.2f} m before launch; terminate and request a new mission.",
            )
            return

        self.passed_plan_pub.publish(String(data=json.dumps(self.pending_verified_plan)))
        self.pending_verified_plan = None
        self.state = "RELEASED_TO_EXECUTOR"
        self.publish_response(
            "LAUNCHED",
            f"Launch approved; plan {self.active_plan_id} was released to the control-law executor.",
            mission_id=mission["mission_id"],
            plan_id=self.active_plan_id,
        )

    def request_executor_land(self, reason):
        mission_id = self.active_mission["mission_id"] if self.active_mission else None
        payload = {
            "command": "LAND",
            "reason": reason,
            "mission_id": mission_id,
            "plan_id": self.active_plan_id,
            "timestamp": time.time(),
        }
        self.executor_command_pub.publish(String(data=json.dumps(payload)))
        self.pending_verified_plan = None
        self.state = "LAND_REQUESTED"
        self.publish_response(
            "LAND_REQUESTED",
            f"{reason} Offboard landing was sent to the control-law executor.",
            mission_id=mission_id,
            plan_id=self.active_plan_id,
        )

    def candidate_contract_matches(self, candidate):
        mission = self.active_mission
        for key in ("start", "goal", "goal_relations", "workspace", "obstacles"):
            if canonical_json(candidate.get(key)) != canonical_json(mission[key]):
                return False
        return True

    def scene_compatibility_error(self, mission):
        compatible, reason = scenes_compatible(
            mission["observed_obstacles"],
            self.latest_scene["obstacles"],
            self.scene_position_tolerance_m,
            self.scene_size_tolerance_m,
        )
        return None if compatible else reason

    def point_in_workspace(self, point):
        return (
            self.workspace["x"][0] <= float(point["x"]) <= self.workspace["x"][1]
            and self.workspace["y"][0] <= float(point["y"]) <= self.workspace["y"][1]
        )

    def context_error(self):
        error = self.scene_context_error()
        if error:
            return error
        return self.vehicle_context_error()

    def planning_context_error(self):
        if self.environment_frozen():
            return self.vehicle_context_error()
        return self.context_error()

    def environment_frozen(self):
        return bool(
            self.active_mission
            and self.active_mission.get("environment_frozen", False)
        )

    def vehicle_context_error(self):
        now = time.time()
        timeout = float(self.get_parameter("fresh_data_timeout_s").value)
        if bool(self.get_parameter("require_mission_state").value):
            if self.latest_mission_state is None or self.latest_mission_state_received_s is None:
                return "No mission-state update is available."
            if now - self.latest_mission_state_received_s > timeout:
                return "The mission-state update is stale."
            required = str(self.get_parameter("required_mission_state").value)
            if self.latest_mission_state.get("state") != required:
                return f"Vehicle must be in {required} before accepting a mission."
        elif self.current_pose is None or now - self.current_pose["received_s"] > timeout:
            return "No fresh NED pose is available."
        return None

    def scene_context_error(self):
        now = time.time()
        timeout = float(self.get_parameter("fresh_data_timeout_s").value)
        if self.latest_scene is None or self.latest_scene_received_s is None:
            return "No obstacle snapshot is available."
        if now - self.latest_scene_received_s > timeout:
            return "The obstacle snapshot is stale."
        if self.latest_scene.get("healthy") is False:
            return "Perception reports an unhealthy obstacle snapshot."
        return None

    def current_start(self):
        position = self.current_position()
        return {
            "x": round(float(position["x"]), 3),
            "y": round(float(position["y"]), 3),
            "z": self.fixed_z,
        }

    def current_position(self):
        if bool(self.get_parameter("require_mission_state").value):
            position = self.latest_mission_state.get("position", {})
            return {key: float(position[key]) for key in ("x", "y", "z")}
        return {key: float(self.current_pose[key]) for key in ("x", "y", "z")}

    def cancel_active_mission(self, reason):
        if self.state == "RELEASED_TO_EXECUTOR":
            self.publish_response("CANCEL_UNAVAILABLE", "The plan has already been released to the executor.")
            return
        if self.state in ("FORMING_SAFETY_TUBES", "AWAITING_LAUNCH_APPROVAL"):
            self.request_executor_land(reason)
            return
        self.intent_request_token = None
        self.active_mission = None
        self.active_plan_id = None
        self.pending_verified_plan = None
        self.state = "WAITING_FOR_COMMAND"
        self.publish_response("CANCELLED", reason)

    def publish_response(self, status, message, **metadata):
        if (
            status in ("INTENT_ERROR", "NO_SAFE_GOAL", "PLANNING_FAILED")
            or status.endswith("_REJECTED")
        ):
            self.last_failure = {
                "status": status,
                "message": message.rstrip("."),
                "failed_constraints": list(metadata.get("failed_constraints", [])),
                "timestamp": time.time(),
            }
        payload = {
            "type": "OPERATOR_RESPONSE",
            "status": status,
            "message": message,
            "gateway_state": self.state,
            "timestamp": time.time(),
        }
        if self.environment_frozen():
            payload.update(
                {
                    "environment_frozen": True,
                    "snapshot_id": self.active_mission["snapshot_id"],
                    "environment_frozen_at": self.active_mission["environment_frozen_at"],
                    "environment_warning": (
                        "Environment captured for static planning. Do not move the vehicle or obstacles "
                        "until launch or termination."
                    ),
                }
            )
        payload.update(metadata)
        self.response_pub.publish(String(data=json.dumps(payload)))
        self.log_debug(f"{status}: {message}")

    def log_debug(self, message):
        if bool(self.get_parameter("debug").value):
            self.get_logger().info(message)

    def destroy_node(self):
        self.intent_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main():
    rclpy.init()
    node = InteractiveMissionGateway()
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
