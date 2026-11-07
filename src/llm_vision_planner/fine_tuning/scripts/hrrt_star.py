#!/usr/bin/env python3
"""H-signature guided RRT-star with finite object-clearance route variants.

The implementation follows the nominal two-dimensional HRRT/HRRT-star ideas
from Hao and Cavusoglu (IROS 2022): every vertex carries a cumulative winding
vector, vertices are expanded in signature-aware subtrees, RRT-star parent and
rewire operations preserve augmented-state identity, and an inter-signature
rewire can seed another subtree.  This workspace adds a finite clearance-bin
vector to the subtree key so one homology side can retain close, medium, and
wide variants without treating clearance as winding.

Inputs are ground-truth axis-aligned obstacle boxes.  No perception-error
inflation or execution safety tube is applied here.  The hard clearance is a
geometric planner parameter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Point = Tuple[float, float]
Signature = Tuple[int, ...]
BinVector = Tuple[int, ...]
CompositeKey = Tuple[Signature, BinVector]

DEFAULT_WORKSPACE = {"x": [-4.0, 4.0], "y": [-3.0, 3.0], "z": -0.5}
DEFAULT_CLEARANCE_M = 0.40
DEFAULT_BIN_THRESHOLDS_M = {
    "person": (0.80, 1.20),
    "chair": (0.60, 0.80),
    "backpack": (0.65, 0.95),
    "bottle": (0.55, 0.75),
    "potted_plant": (0.65, 0.95),
    "bench": (0.65, 0.95),
    "stop_sign": (0.70, 1.00),
    "unknown": (0.75, 1.10),
    "default": (0.65, 0.95),
}
ROUTE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6A3D9A",
    "#1B9E77",
    "#E7298A",
    "#7570B3",
    "#66A61E",
    "#A6761D",
)


@dataclass(frozen=True)
class Obstacle:
    object_id: str
    label: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def center(self) -> Point:
        return 0.5 * (self.min_x + self.max_x), 0.5 * (self.min_y + self.max_y)

    def inflated_bounds(self, margin_m: float) -> Tuple[float, float, float, float]:
        return (
            self.min_x - margin_m,
            self.max_x + margin_m,
            self.min_y - margin_m,
            self.max_y + margin_m,
        )


@dataclass(frozen=True)
class Node:
    q: Point
    parent: Optional[int]
    cost: float
    winding: Tuple[float, ...]
    min_clearances: Tuple[float, ...]
    signature: Signature
    bins: BinVector

    @property
    def key(self) -> CompositeKey:
        return self.signature, self.bins


@dataclass(frozen=True)
class PathClassification:
    signature: Signature
    bins: BinVector
    winding: Tuple[float, ...]
    min_clearances: Tuple[float, ...]
    length_m: float

    @property
    def key(self) -> CompositeKey:
        return self.signature, self.bins


def normalize_label(label: object) -> str:
    return str(label or "unknown").strip().lower().replace(" ", "_").replace("-", "_")


def point_xy(point: object) -> Point:
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    values = list(point)  # type: ignore[arg-type]
    return float(values[0]), float(values[1])


def normalize_obstacles(obstacles: Sequence[Mapping[str, object]]) -> List[Obstacle]:
    normalized = []
    for index, raw in enumerate(obstacles):
        minimum = list(raw.get("min_corner", []))
        maximum = list(raw.get("max_corner", []))
        if len(minimum) < 2 or len(maximum) < 2:
            raise ValueError(f"Obstacle {index} needs min_corner and max_corner with x/y values.")
        min_x, max_x = sorted((float(minimum[0]), float(maximum[0])))
        min_y, max_y = sorted((float(minimum[1]), float(maximum[1])))
        if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
            raise ValueError(f"Obstacle {index} must have non-zero x/y extent.")
        normalized.append(
            Obstacle(
                object_id=str(raw.get("object_id") or f"obstacle-{index + 1}"),
                label=normalize_label(raw.get("label", "unknown")),
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
            )
        )
    return normalized


def workspace_limits(workspace: Mapping[str, object]) -> Tuple[float, float, float, float]:
    x_limits = list(workspace.get("x", DEFAULT_WORKSPACE["x"]))
    y_limits = list(workspace.get("y", DEFAULT_WORKSPACE["y"]))
    if len(x_limits) != 2 or len(y_limits) != 2:
        raise ValueError("Workspace x/y limits must each contain two values.")
    min_x, max_x = sorted((float(x_limits[0]), float(x_limits[1])))
    min_y, max_y = sorted((float(y_limits[0]), float(y_limits[1])))
    if not min_x < max_x or not min_y < max_y:
        raise ValueError("Workspace limits must have positive extent.")
    return min_x, max_x, min_y, max_y


def euclidean(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def in_workspace(point: Point, workspace: Mapping[str, object]) -> bool:
    min_x, max_x, min_y, max_y = workspace_limits(workspace)
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def point_in_bounds(point: Point, bounds: Tuple[float, float, float, float]) -> bool:
    min_x, max_x, min_y, max_y = bounds
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def segment_intersects_bounds(
    start: Point,
    end: Point,
    bounds: Tuple[float, float, float, float],
) -> bool:
    """Inclusive line-segment/AABB intersection using the slab method."""
    min_x, max_x, min_y, max_y = bounds
    t_min, t_max = 0.0, 1.0
    for origin, delta, lower, upper in (
        (start[0], end[0] - start[0], min_x, max_x),
        (start[1], end[1] - start[1], min_y, max_y),
    ):
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            if origin < lower or origin > upper:
                return False
            continue
        first = (lower - origin) / delta
        second = (upper - origin) / delta
        if first > second:
            first, second = second, first
        t_min = max(t_min, first)
        t_max = min(t_max, second)
        if t_min > t_max:
            return False
    return True


def segment_clear(
    start: Point,
    end: Point,
    obstacles: Sequence[Obstacle],
    workspace: Mapping[str, object],
    clearance_m: float,
) -> bool:
    if not in_workspace(start, workspace) or not in_workspace(end, workspace):
        return False
    return not any(
        segment_intersects_bounds(start, end, obstacle.inflated_bounds(clearance_m))
        for obstacle in obstacles
    )


def point_to_obstacle_distance(point: Point, obstacle: Obstacle) -> float:
    dx = max(obstacle.min_x - point[0], 0.0, point[0] - obstacle.max_x)
    dy = max(obstacle.min_y - point[1], 0.0, point[1] - obstacle.max_y)
    return math.hypot(dx, dy)


def edge_clearances(
    start: Point,
    end: Point,
    obstacles: Sequence[Obstacle],
    sample_spacing_m: float,
) -> Tuple[float, ...]:
    distance = euclidean(start, end)
    steps = max(1, int(math.ceil(distance / max(sample_spacing_m, 1e-6))))
    minima = [math.inf] * len(obstacles)
    for step in range(steps + 1):
        ratio = step / steps
        point = (
            start[0] + ratio * (end[0] - start[0]),
            start[1] + ratio * (end[1] - start[1]),
        )
        for index, obstacle in enumerate(obstacles):
            minima[index] = min(minima[index], point_to_obstacle_distance(point, obstacle))
    return tuple(minima)


def edge_winding_increment(start: Point, end: Point, centers: Sequence[Point]) -> Tuple[float, ...]:
    increments = []
    for center_x, center_y in centers:
        first = (start[0] - center_x, start[1] - center_y)
        second = (end[0] - center_x, end[1] - center_y)
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        increments.append(math.atan2(cross, dot) / (2.0 * math.pi))
    return tuple(increments)


def modified_signature_component(value: float, modulus: int = 2, epsilon: float = 1e-9) -> int:
    """Paper-style bounded modified-sign bucket; m=2 gives {-1, 0, +1}."""
    if modulus < 2:
        raise ValueError("signature_modulus must be at least 2")
    if abs(value) <= epsilon:
        return 0
    level = max(1, int(math.ceil(abs(value) - epsilon)))
    if level >= modulus:
        raise ValueError(
            f"Winding component {value:.6f} lies outside the configured "
            f"modified-sign range for modulus {modulus}."
        )
    return level if value > 0.0 else -level


def h_signature(winding: Sequence[float], modulus: int = 2) -> Signature:
    return tuple(modified_signature_component(value, modulus) for value in winding)


def normalized_bin_thresholds(
    overrides: Optional[Mapping[str, Sequence[float]]],
    clearance_m: float,
) -> Dict[str, Tuple[float, float]]:
    values: Dict[str, Tuple[float, float]] = dict(DEFAULT_BIN_THRESHOLDS_M)
    for label, thresholds in (overrides or {}).items():
        pair = list(thresholds)
        if len(pair) != 2:
            raise ValueError(f"Clearance thresholds for {label!r} must contain two values.")
        values[normalize_label(label)] = (float(pair[0]), float(pair[1]))
    minimum_gap = 0.05
    for label, (first, second) in list(values.items()):
        first = max(float(first), clearance_m + minimum_gap)
        second = max(float(second), first + minimum_gap)
        values[label] = (first, second)
    return values


def clearance_bin(
    clearance: float,
    label: str,
    thresholds: Mapping[str, Tuple[float, float]],
) -> int:
    first, second = thresholds.get(normalize_label(label), thresholds["default"])
    if clearance < first:
        return 0
    if clearance < second:
        return 1
    return 2


def clearance_bins(
    clearances: Sequence[float],
    obstacles: Sequence[Obstacle],
    thresholds: Mapping[str, Tuple[float, float]],
) -> BinVector:
    return tuple(
        clearance_bin(distance, obstacle.label, thresholds)
        for distance, obstacle in zip(clearances, obstacles)
    )


def path_length(path: Sequence[Point]) -> float:
    return sum(euclidean(first, second) for first, second in zip(path, path[1:]))


def classify_path(
    path: Sequence[Point],
    obstacles: Sequence[Obstacle],
    thresholds: Mapping[str, Tuple[float, float]],
    signature_modulus: int,
    clearance_sample_spacing_m: float,
) -> PathClassification:
    winding = [0.0] * len(obstacles)
    minima = [math.inf] * len(obstacles)
    centers = [obstacle.center for obstacle in obstacles]
    for start, end in zip(path, path[1:]):
        increment = edge_winding_increment(start, end, centers)
        edge_minima = edge_clearances(start, end, obstacles, clearance_sample_spacing_m)
        winding = [old + delta for old, delta in zip(winding, increment)]
        minima = [min(old, edge) for old, edge in zip(minima, edge_minima)]
    signature = h_signature(winding, signature_modulus)
    bins = clearance_bins(minima, obstacles, thresholds)
    return PathClassification(
        signature=signature,
        bins=bins,
        winding=tuple(winding),
        min_clearances=tuple(minima),
        length_m=path_length(path),
    )


def steer(start: Point, target: Point, step_size_m: float) -> Point:
    distance = euclidean(start, target)
    if distance <= step_size_m:
        return target
    ratio = step_size_m / distance
    return (
        start[0] + ratio * (target[0] - start[0]),
        start[1] + ratio * (target[1] - start[1]),
    )


class HRRTStarPlanner:
    """Finite multi-label HRRT-star planner for fixed-altitude waypoint paths."""

    def __init__(
        self,
        start: object,
        goal: object,
        obstacles: Sequence[Mapping[str, object]],
        workspace: Optional[Mapping[str, object]] = None,
        *,
        clearance_m: float = DEFAULT_CLEARANCE_M,
        clearance_bin_thresholds_m: Optional[Mapping[str, Sequence[float]]] = None,
        step_size_m: float = 0.35,
        rewire_radius_m: float = 0.80,
        goal_connect_radius_m: float = 1.00,
        goal_sample_rate: float = 0.12,
        clearance_sample_rate: float = 0.28,
        clearance_sample_spacing_m: float = 0.04,
        signature_modulus: int = 2,
        max_iterations: int = 300,
        max_candidates: int = 12,
        max_composite_keys: int = 48,
        max_nodes_per_key: int = 260,
        max_total_nodes: int = 12000,
        max_waypoints: int = 12,
        seed: int = 17,
    ):
        self.start = point_xy(start)
        self.goal = point_xy(goal)
        self.obstacles = normalize_obstacles(obstacles)
        self.workspace = dict(workspace or DEFAULT_WORKSPACE)
        self.clearance_m = float(clearance_m)
        self.step_size_m = float(step_size_m)
        self.rewire_radius_m = float(rewire_radius_m)
        self.goal_connect_radius_m = float(goal_connect_radius_m)
        self.goal_sample_rate = float(goal_sample_rate)
        self.clearance_sample_rate = float(clearance_sample_rate)
        self.clearance_sample_spacing_m = float(clearance_sample_spacing_m)
        self.signature_modulus = int(signature_modulus)
        self.max_iterations = int(max_iterations)
        self.max_candidates = int(max_candidates)
        self.max_composite_keys = int(max_composite_keys)
        self.max_nodes_per_key = int(max_nodes_per_key)
        self.max_total_nodes = int(max_total_nodes)
        self.max_waypoints = int(max_waypoints)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.thresholds = normalized_bin_thresholds(clearance_bin_thresholds_m, self.clearance_m)
        self.centers = [obstacle.center for obstacle in self.obstacles]
        self.nodes: List[Node] = []
        self.groups: Dict[CompositeKey, List[int]] = {}
        self.best_goal_paths: Dict[CompositeKey, Tuple[float, List[Point], PathClassification]] = {}

        self._validate_parameters()
        self._initialize_root()

    def _validate_parameters(self) -> None:
        if self.clearance_m < 0.0:
            raise ValueError("clearance_m must be non-negative")
        if self.step_size_m <= 0.0 or self.rewire_radius_m <= 0.0:
            raise ValueError("step_size_m and rewire_radius_m must be positive")
        if self.goal_connect_radius_m <= 0.0:
            raise ValueError("goal_connect_radius_m must be positive")
        if not 0.0 <= self.goal_sample_rate <= 1.0:
            raise ValueError("goal_sample_rate must be in [0, 1]")
        if not 0.0 <= self.clearance_sample_rate <= 1.0 - self.goal_sample_rate:
            raise ValueError("clearance_sample_rate leaves no probability for uniform sampling")
        if self.max_iterations <= 0 or self.max_candidates <= 0:
            raise ValueError("iteration and candidate caps must be positive")
        if not segment_clear(self.start, self.start, self.obstacles, self.workspace, self.clearance_m):
            raise ValueError("Start is outside the workspace or violates hard obstacle clearance.")
        if not segment_clear(self.goal, self.goal, self.obstacles, self.workspace, self.clearance_m):
            raise ValueError("Goal is outside the workspace or violates hard obstacle clearance.")

    def _initialize_root(self) -> None:
        start_clearances = tuple(point_to_obstacle_distance(self.start, item) for item in self.obstacles)
        root = Node(
            q=self.start,
            parent=None,
            cost=0.0,
            winding=(0.0,) * len(self.obstacles),
            min_clearances=start_clearances,
            signature=(0,) * len(self.obstacles),
            bins=clearance_bins(start_clearances, self.obstacles, self.thresholds),
        )
        self.nodes.append(root)
        self.groups[root.key] = [0]

    def _sample(self) -> Point:
        draw = self.rng.random()
        if draw < self.goal_sample_rate:
            return self.goal
        if self.obstacles and draw < self.goal_sample_rate + self.clearance_sample_rate:
            obstacle = self.rng.choice(self.obstacles)
            first, second = self.thresholds.get(obstacle.label, self.thresholds["default"])
            desired_clearance = self.rng.choice(
                (
                    self.clearance_m + 0.05,
                    0.5 * (self.clearance_m + first),
                    0.5 * (first + second),
                    second + 0.20,
                )
            )
            half_diagonal = 0.5 * math.hypot(
                obstacle.max_x - obstacle.min_x,
                obstacle.max_y - obstacle.min_y,
            )
            angle = self.rng.uniform(-math.pi, math.pi)
            radius = half_diagonal + desired_clearance
            candidate = (
                obstacle.center[0] + radius * math.cos(angle),
                obstacle.center[1] + radius * math.sin(angle),
            )
            if in_workspace(candidate, self.workspace):
                return candidate
        min_x, max_x, min_y, max_y = workspace_limits(self.workspace)
        return self.rng.uniform(min_x, max_x), self.rng.uniform(min_y, max_y)

    def _nearest(self, indices: Sequence[int], point: Point) -> int:
        return min(indices, key=lambda index: euclidean(self.nodes[index].q, point))

    def _near(self, indices: Sequence[int], point: Point) -> List[int]:
        nearby = [index for index in indices if euclidean(self.nodes[index].q, point) <= self.rewire_radius_m]
        if not nearby:
            nearby.append(self._nearest(indices, point))
        return nearby

    def _transition(self, parent_index: int, point: Point) -> Optional[Node]:
        parent = self.nodes[parent_index]
        if not segment_clear(parent.q, point, self.obstacles, self.workspace, self.clearance_m):
            return None
        increments = edge_winding_increment(parent.q, point, self.centers)
        winding = tuple(old + delta for old, delta in zip(parent.winding, increments))
        try:
            signature = h_signature(winding, self.signature_modulus)
        except ValueError:
            return None
        edge_minima = edge_clearances(
            parent.q,
            point,
            self.obstacles,
            self.clearance_sample_spacing_m,
        )
        minima = tuple(min(old, edge) for old, edge in zip(parent.min_clearances, edge_minima))
        return Node(
            q=point,
            parent=parent_index,
            cost=parent.cost + euclidean(parent.q, point),
            winding=winding,
            min_clearances=minima,
            signature=signature,
            bins=clearance_bins(minima, self.obstacles, self.thresholds),
        )

    def _can_add_key(self, key: CompositeKey) -> bool:
        return key in self.groups or len(self.groups) < self.max_composite_keys

    def _is_dominated_duplicate(self, candidate: Node, tolerance_m: float = 0.025) -> bool:
        for index in self.groups.get(candidate.key, []):
            existing = self.nodes[index]
            if euclidean(existing.q, candidate.q) <= tolerance_m and existing.cost <= candidate.cost + 1e-9:
                return True
        return False

    def _append_node(self, candidate: Node) -> Optional[int]:
        if len(self.nodes) >= self.max_total_nodes or not self._can_add_key(candidate.key):
            return None
        group = self.groups.setdefault(candidate.key, [])
        if len(group) >= self.max_nodes_per_key or self._is_dominated_duplicate(candidate):
            return None
        index = len(self.nodes)
        self.nodes.append(candidate)
        group.append(index)
        return index

    def _rewire_with_clones(self, new_index: int, limit: int = 2) -> None:
        """Create cheaper immutable labels instead of mutating descendant chains."""
        new_node = self.nodes[new_index]
        improvements = []
        for target_index in list(self.groups.get(new_node.key, [])):
            if target_index == new_index:
                continue
            target = self.nodes[target_index]
            if euclidean(new_node.q, target.q) > self.rewire_radius_m:
                continue
            candidate = self._transition(new_index, target.q)
            if candidate is None or candidate.key != target.key:
                continue
            if candidate.cost + 1e-8 < target.cost:
                improvements.append((target.cost - candidate.cost, candidate))
        for _, candidate in sorted(improvements, key=lambda item: item[0], reverse=True)[:limit]:
            self._append_node(candidate)

    def _inter_signature_rewire(self, new_index: int) -> None:
        """Paper-inspired bridge that may seed a different augmented subtree."""
        new_node = self.nodes[new_index]
        best = None
        for target_index, target in enumerate(self.nodes[:-1]):
            if target.signature == new_node.signature:
                continue
            distance = euclidean(new_node.q, target.q)
            if distance > self.rewire_radius_m:
                continue
            candidate = self._transition(new_index, target.q)
            if candidate is None or candidate.cost + 1e-8 >= target.cost:
                continue
            improvement = target.cost - candidate.cost
            if best is None or improvement > best[0]:
                best = improvement, candidate
        if best is not None:
            self._append_node(best[1])

    def _reconstruct(self, node_index: int) -> List[Point]:
        path = []
        cursor: Optional[int] = node_index
        seen = set()
        while cursor is not None:
            if cursor in seen:
                raise RuntimeError("Cycle detected in HRRT-star parent chain.")
            seen.add(cursor)
            node = self.nodes[cursor]
            path.append(node.q)
            cursor = node.parent
        path.reverse()
        return path

    def _consider_goal(self, node_index: int) -> None:
        node = self.nodes[node_index]
        if euclidean(node.q, self.goal) > self.goal_connect_radius_m:
            return
        if not segment_clear(node.q, self.goal, self.obstacles, self.workspace, self.clearance_m):
            return
        path = self._reconstruct(node_index)
        if euclidean(path[-1], self.goal) > 1e-9:
            path.append(self.goal)
        try:
            classification = classify_path(
                path,
                self.obstacles,
                self.thresholds,
                self.signature_modulus,
                self.clearance_sample_spacing_m,
            )
        except ValueError:
            # The goal edge can push a path outside the bounded modified-sign
            # range even when the parent node itself is valid.  That edge is
            # not a valid augmented-state transition for this planner.
            return
        previous = self.best_goal_paths.get(classification.key)
        if previous is None or classification.length_m + 1e-9 < previous[0]:
            self.best_goal_paths[classification.key] = (classification.length_m, path, classification)

    def _expand_group(self, key: CompositeKey, sample: Point) -> None:
        indices = list(self.groups.get(key, []))
        if not indices:
            return
        nearest_index = self._nearest(indices, sample)
        new_point = steer(self.nodes[nearest_index].q, sample, self.step_size_m)
        if not segment_clear(
            self.nodes[nearest_index].q,
            new_point,
            self.obstacles,
            self.workspace,
            self.clearance_m,
        ):
            return

        best_by_result_key: Dict[CompositeKey, Node] = {}
        for parent_index in self._near(indices, new_point):
            candidate = self._transition(parent_index, new_point)
            if candidate is None:
                continue
            previous = best_by_result_key.get(candidate.key)
            if previous is None or candidate.cost < previous.cost:
                best_by_result_key[candidate.key] = candidate

        for candidate in sorted(best_by_result_key.values(), key=lambda item: (item.cost, item.key))[:3]:
            new_index = self._append_node(candidate)
            if new_index is None:
                continue
            self._rewire_with_clones(new_index)
            self._inter_signature_rewire(new_index)
            self._consider_goal(new_index)

    def _path_key(self, path: Sequence[Point]) -> CompositeKey:
        return classify_path(
            path,
            self.obstacles,
            self.thresholds,
            self.signature_modulus,
            self.clearance_sample_spacing_m,
        ).key

    def _simplify_preserving_key(self, path: Sequence[Point], key: CompositeKey) -> List[Point]:
        if len(path) <= 2:
            return list(path)
        simplified = [path[0]]
        start_index = 0
        while start_index < len(path) - 1:
            chosen = start_index + 1
            for end_index in range(len(path) - 1, start_index, -1):
                if not segment_clear(
                    path[start_index],
                    path[end_index],
                    self.obstacles,
                    self.workspace,
                    self.clearance_m,
                ):
                    continue
                trial = simplified + [path[end_index]] + list(path[end_index + 1 :])
                if self._path_key(trial) == key:
                    chosen = end_index
                    break
            simplified.append(path[chosen])
            start_index = chosen

        while len(simplified) > self.max_waypoints:
            removable = None
            for index in range(1, len(simplified) - 1):
                if not segment_clear(
                    simplified[index - 1],
                    simplified[index + 1],
                    self.obstacles,
                    self.workspace,
                    self.clearance_m,
                ):
                    continue
                trial = simplified[:index] + simplified[index + 1 :]
                if self._path_key(trial) != key:
                    continue
                saving = (
                    euclidean(simplified[index - 1], simplified[index])
                    + euclidean(simplified[index], simplified[index + 1])
                    - euclidean(simplified[index - 1], simplified[index + 1])
                )
                if removable is None or saving < removable[0]:
                    removable = saving, index
            if removable is None:
                break
            simplified.pop(removable[1])
        return simplified

    def _select_goal_paths(self) -> List[Tuple[List[Point], PathClassification]]:
        by_signature: Dict[Signature, List[Tuple[float, List[Point], PathClassification]]] = {}
        for item in self.best_goal_paths.values():
            by_signature.setdefault(item[2].signature, []).append(item)
        for values in by_signature.values():
            values.sort(key=lambda item: (item[0], item[2].bins))

        selected = []
        ordered_signatures = sorted(by_signature)
        while ordered_signatures and len(selected) < self.max_candidates:
            remaining = []
            for signature in ordered_signatures:
                values = by_signature[signature]
                if values and len(selected) < self.max_candidates:
                    _, path, classification = values.pop(0)
                    selected.append((path, classification))
                if values:
                    remaining.append(signature)
            ordered_signatures = remaining
        return selected

    def plan(self) -> Dict[str, object]:
        for _ in range(self.max_iterations):
            if len(self.nodes) >= self.max_total_nodes:
                break
            sample = self._sample()
            active_keys = sorted(self.groups)
            for key in active_keys:
                self._expand_group(key, sample)
                if len(self.nodes) >= self.max_total_nodes:
                    break

        if not self.best_goal_paths:
            for index in sorted(range(len(self.nodes)), key=lambda item: euclidean(self.nodes[item].q, self.goal)):
                self._consider_goal(index)
                if self.best_goal_paths:
                    break
        if not self.best_goal_paths:
            raise RuntimeError("HRRT-star failed to connect any augmented subtree to the goal.")

        fixed_z = float(self.workspace.get("z", DEFAULT_WORKSPACE["z"]))
        routes = []
        for route_index, (raw_path, original) in enumerate(self._select_goal_paths(), start=1):
            path = self._simplify_preserving_key(raw_path, original.key)
            final = classify_path(
                path,
                self.obstacles,
                self.thresholds,
                self.signature_modulus,
                self.clearance_sample_spacing_m,
            )
            if final.key != original.key:
                raise RuntimeError("Signature/bin-preserving simplification changed a route key.")
            routes.append(
                {
                    "route_id": f"route-{route_index:03d}",
                    "color": ROUTE_COLORS[(route_index - 1) % len(ROUTE_COLORS)],
                    "h_signature": list(final.signature),
                    "clearance_bins": list(final.bins),
                    "composite_key": {
                        "h_signature": list(final.signature),
                        "clearance_bins": list(final.bins),
                    },
                    "winding_vector": [round(value, 6) for value in final.winding],
                    "minimum_clearance_m": {
                        obstacle.object_id: round(distance, 4)
                        for obstacle, distance in zip(self.obstacles, final.min_clearances)
                    },
                    "path_length_m": round(final.length_m, 4),
                    "waypoints": [
                        {"x": round(point[0], 4), "y": round(point[1], 4), "z": round(fixed_z, 4)}
                        for point in path
                    ],
                }
            )

        payload: Dict[str, object] = {
            "planner": "hrrt_star_clearance_bins",
            "seed": self.seed,
            "start": {"x": self.start[0], "y": self.start[1], "z": fixed_z},
            "goal": {"x": self.goal[0], "y": self.goal[1], "z": fixed_z},
            "workspace": self.workspace,
            "hard_clearance_m": self.clearance_m,
            "signature_modulus": self.signature_modulus,
            "clearance_bin_thresholds_m": {
                label: list(values) for label, values in sorted(self.thresholds.items())
            },
            "obstacles": [
                {
                    "object_id": item.object_id,
                    "label": item.label,
                    "min_corner": [item.min_x, item.min_y, fixed_z],
                    "max_corner": [item.max_x, item.max_y, fixed_z],
                }
                for item in self.obstacles
            ],
            "routes": routes,
            "stats": {
                "iterations": self.max_iterations,
                "tree_nodes": len(self.nodes),
                "composite_subtrees": len(self.groups),
                "goal_keys_found": len(self.best_goal_paths),
                "routes_returned": len(routes),
            },
        }
        validate_payload(payload, self.clearance_sample_spacing_m)
        return payload


def validate_payload(payload: Mapping[str, object], sample_spacing_m: float = 0.04) -> None:
    workspace = payload["workspace"]
    obstacles = normalize_obstacles(payload["obstacles"])  # type: ignore[arg-type]
    clearance_m = float(payload["hard_clearance_m"])
    signature_modulus = int(payload["signature_modulus"])
    thresholds = normalized_bin_thresholds(payload["clearance_bin_thresholds_m"], clearance_m)  # type: ignore[arg-type]
    start = point_xy(payload["start"])
    goal = point_xy(payload["goal"])
    keys = set()
    routes = list(payload["routes"])  # type: ignore[arg-type]
    for route in routes:
        waypoints = list(route["waypoints"])
        path = [point_xy(point) for point in waypoints]
        if euclidean(path[0], start) > 1e-6 or euclidean(path[-1], goal) > 1e-6:
            raise ValueError(f"{route['route_id']} does not preserve start and goal.")
        for first, second in zip(path, path[1:]):
            if not segment_clear(first, second, obstacles, workspace, clearance_m):  # type: ignore[arg-type]
                raise ValueError(f"{route['route_id']} contains a collision-invalid edge.")
        classification = classify_path(path, obstacles, thresholds, signature_modulus, sample_spacing_m)
        expected = (tuple(route["h_signature"]), tuple(route["clearance_bins"]))
        if classification.key != expected:
            raise ValueError(f"{route['route_id']} metadata does not match its waypoints.")
        if expected in keys:
            raise ValueError(f"Duplicate composite route key: {expected}")
        keys.add(expected)


def plan_hrrt_star(
    start: object,
    goal: object,
    obstacles: Sequence[Mapping[str, object]],
    workspace: Optional[Mapping[str, object]] = None,
    **kwargs: object,
) -> Dict[str, object]:
    return HRRTStarPlanner(start, goal, obstacles, workspace, **kwargs).plan()


def demo_problem() -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, object]], Dict[str, object]]:
    workspace: Dict[str, object] = {"x": [-4.0, 4.0], "y": [-3.0, 3.0], "z": -0.5}
    start = {"x": -3.5, "y": 0.0, "z": -0.5}
    goal = {"x": 3.5, "y": 0.0, "z": -0.5}
    obstacles = [
        {
            "object_id": "person-1",
            "label": "person",
            "min_corner": [-1.45, -0.45, -1.0],
            "max_corner": [-0.65, 0.45, 0.0],
        },
        {
            "object_id": "chair-1",
            "label": "chair",
            "min_corner": [0.65, -0.45, -1.0],
            "max_corner": [1.45, 0.45, 0.0],
        },
    ]
    return start, goal, obstacles, workspace


def plot_payload(payload: Mapping[str, object], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError("matplotlib is required only when --plot-output is used") from exc

    figure, axis = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    obstacles = normalize_obstacles(payload["obstacles"])  # type: ignore[arg-type]
    clearance_m = float(payload["hard_clearance_m"])
    for obstacle in obstacles:
        axis.add_patch(
            Rectangle(
                (obstacle.min_x, obstacle.min_y),
                obstacle.max_x - obstacle.min_x,
                obstacle.max_y - obstacle.min_y,
                facecolor="#6b7280",
                edgecolor="#111827",
                linewidth=1.5,
                zorder=4,
            )
        )
        inflated = obstacle.inflated_bounds(clearance_m)
        axis.add_patch(
            Rectangle(
                (inflated[0], inflated[2]),
                inflated[1] - inflated[0],
                inflated[3] - inflated[2],
                fill=False,
                edgecolor="#6b7280",
                linestyle=":",
                linewidth=1.3,
                zorder=3,
            )
        )
        axis.scatter(*obstacle.center, color="#111827", s=18, zorder=6)
        axis.text(
            obstacle.center[0],
            obstacle.center[1],
            f"{obstacle.label}\n{obstacle.object_id}",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )

    for route in payload["routes"]:  # type: ignore[index]
        points = list(route["waypoints"])
        xs = [float(point["x"]) for point in points]
        ys = [float(point["y"]) for point in points]
        key_label = f"h={tuple(route['h_signature'])}, b={tuple(route['clearance_bins'])}"
        axis.plot(
            xs,
            ys,
            color=route["color"],
            linewidth=2.2,
            marker="o",
            markersize=3.3,
            label=f"{route['route_id']}  {key_label}  L={route['path_length_m']:.2f} m",
            zorder=5,
        )

    start = point_xy(payload["start"])
    goal = point_xy(payload["goal"])
    axis.scatter(*start, color="#111827", s=75, zorder=9)
    axis.scatter(*goal, color="#dc2626", s=75, zorder=9)
    axis.text(start[0], start[1] + 0.14, "Start", ha="center", fontweight="bold")
    axis.text(goal[0], goal[1] + 0.14, "Goal", ha="center", fontweight="bold")
    min_x, max_x, min_y, max_y = workspace_limits(payload["workspace"])  # type: ignore[arg-type]
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_title("HRRT-star route family: H-signature + clearance bins", fontweight="bold")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_json_argument(value: str, name: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use the built-in person/chair example.")
    parser.add_argument("--start", help='JSON point, for example {"x":-3,"y":0,"z":-0.5}.')
    parser.add_argument("--goal", help='JSON point, for example {"x":3,"y":0,"z":-0.5}.')
    parser.add_argument("--obstacles", help="JSON list of ground-truth obstacle boxes.")
    parser.add_argument("--workspace", default=json.dumps(DEFAULT_WORKSPACE), help="JSON x/y limits and fixed z.")
    parser.add_argument("--bin-thresholds", help="Optional JSON label-to-[medium,wide] thresholds.")
    parser.add_argument("--clearance-m", type=float, default=DEFAULT_CLEARANCE_M)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--step-size-m", type=float, default=0.35)
    parser.add_argument("--rewire-radius-m", type=float, default=0.80)
    parser.add_argument("--goal-connect-radius-m", type=float, default=1.00)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-waypoints", type=int, default=12)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.demo:
        start, goal, obstacles, workspace = demo_problem()
    else:
        if not args.start or not args.goal or not args.obstacles:
            raise SystemExit("Provide --demo or all of --start, --goal, and --obstacles.")
        start = parse_json_argument(args.start, "--start")
        goal = parse_json_argument(args.goal, "--goal")
        obstacles = parse_json_argument(args.obstacles, "--obstacles")
        workspace = parse_json_argument(args.workspace, "--workspace")

    thresholds = parse_json_argument(args.bin_thresholds, "--bin-thresholds") if args.bin_thresholds else None
    payload = plan_hrrt_star(
        start,
        goal,
        obstacles,  # type: ignore[arg-type]
        workspace,  # type: ignore[arg-type]
        clearance_m=args.clearance_m,
        clearance_bin_thresholds_m=thresholds,  # type: ignore[arg-type]
        step_size_m=args.step_size_m,
        rewire_radius_m=args.rewire_radius_m,
        goal_connect_radius_m=args.goal_connect_radius_m,
        max_iterations=args.iterations,
        max_candidates=args.max_candidates,
        max_waypoints=args.max_waypoints,
        seed=args.seed,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.plot_output:
        plot_payload(payload, args.plot_output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
