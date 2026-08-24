#!/usr/bin/env python3
"""Any-angle Theta* expert over a label-conditioned semantic cost map."""

import argparse
import heapq
import json
import math


DEFAULT_WORKSPACE = {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": -0.25}
DEFAULT_CLEARANCE_M = 0.40
DEFAULT_GRID_RESOLUTION_M = 0.20
DEFAULT_RISK_SAMPLE_RESOLUTION_M = 0.10
DEFAULT_MAX_WAYPOINTS = 8

# These bounded defaults are deliberately deterministic. A language model may
# propose replacements, but callers should pass them through this same schema.
DEFAULT_SEMANTIC_POLICY = {
    "person": {"hard_margin_m": 0.65, "soft_radius_m": 1.20, "risk_weight": 8.0},
    "chair": {"hard_margin_m": 0.40, "soft_radius_m": 0.55, "risk_weight": 2.0},
    "backpack": {"hard_margin_m": 0.45, "soft_radius_m": 0.65, "risk_weight": 3.0},
    "bottle": {"hard_margin_m": 0.40, "soft_radius_m": 0.45, "risk_weight": 2.0},
    "potted_plant": {"hard_margin_m": 0.45, "soft_radius_m": 0.70, "risk_weight": 3.0},
    "bench": {"hard_margin_m": 0.45, "soft_radius_m": 0.70, "risk_weight": 3.0},
    "stop_sign": {"hard_margin_m": 0.50, "soft_radius_m": 0.85, "risk_weight": 5.0},
    "unknown": {"hard_margin_m": 0.60, "soft_radius_m": 1.00, "risk_weight": 6.0},
    "default": {"hard_margin_m": 0.45, "soft_radius_m": 0.70, "risk_weight": 3.0},
}


def point_xy(point):
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def normalize_label(label):
    return str(label or "unknown").strip().lower().replace(" ", "_").replace("-", "_")


def validated_semantic_policy(policy=None, base_clearance_m=DEFAULT_CLEARANCE_M):
    merged = {label: dict(values) for label, values in DEFAULT_SEMANTIC_POLICY.items()}
    for label, values in (policy or {}).items():
        normalized = normalize_label(label)
        merged.setdefault(normalized, {}).update(values)
    for label, values in merged.items():
        hard_margin = max(float(base_clearance_m), float(values.get("hard_margin_m", base_clearance_m)))
        soft_radius = max(hard_margin, float(values.get("soft_radius_m", hard_margin)))
        risk_weight = max(0.0, float(values.get("risk_weight", 0.0)))
        merged[label] = {
            "hard_margin_m": hard_margin,
            "soft_radius_m": soft_radius,
            "risk_weight": risk_weight,
        }
    return merged


def policy_for_obstacle(obstacle, semantic_policy):
    label = normalize_label(obstacle.get("label", "unknown"))
    return semantic_policy.get(label, semantic_policy["default"])


def obstacle_bounds(obstacle, margin_m=0.0):
    minimum = obstacle.get("min_corner", [0.0, 0.0, 0.0])
    maximum = obstacle.get("max_corner", [0.0, 0.0, 0.0])
    return (
        float(minimum[0]) - margin_m,
        float(maximum[0]) + margin_m,
        float(minimum[1]) - margin_m,
        float(maximum[1]) + margin_m,
    )


def in_workspace(point, workspace):
    x, y = point_xy(point)
    x_limits = workspace.get("x", DEFAULT_WORKSPACE["x"])
    y_limits = workspace.get("y", DEFAULT_WORKSPACE["y"])
    return float(x_limits[0]) <= x <= float(x_limits[1]) and float(y_limits[0]) <= y <= float(y_limits[1])


def point_clear(point, obstacles, workspace, semantic_policy=None, clearance_m=DEFAULT_CLEARANCE_M):
    if not in_workspace(point, workspace):
        return False
    policy = validated_semantic_policy(semantic_policy, clearance_m)
    x, y = point_xy(point)
    for obstacle in obstacles:
        margin = policy_for_obstacle(obstacle, policy)["hard_margin_m"]
        min_x, max_x, min_y, max_y = obstacle_bounds(obstacle, margin)
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return False
    return True


def segment_intersects_aabb(start, end, bounds):
    """Return true for any inclusive intersection, including tangency."""
    start_x, start_y = point_xy(start)
    end_x, end_y = point_xy(end)
    min_x, max_x, min_y, max_y = bounds
    t_min, t_max = 0.0, 1.0
    for origin, delta, lower, upper in (
        (start_x, end_x - start_x, min_x, max_x),
        (start_y, end_y - start_y, min_y, max_y),
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


def segment_clear(start, end, obstacles, workspace, semantic_policy=None, clearance_m=DEFAULT_CLEARANCE_M):
    if not in_workspace(start, workspace) or not in_workspace(end, workspace):
        return False
    policy = validated_semantic_policy(semantic_policy, clearance_m)
    for obstacle in obstacles:
        margin = policy_for_obstacle(obstacle, policy)["hard_margin_m"]
        if segment_intersects_aabb(start, end, obstacle_bounds(obstacle, margin)):
            return False
    return True


def distance_to_obstacle(point, obstacle):
    x, y = point_xy(point)
    min_x, max_x, min_y, max_y = obstacle_bounds(obstacle)
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return math.hypot(dx, dy)


def semantic_risk(point, obstacles, semantic_policy):
    risk = 0.0
    for obstacle in obstacles:
        settings = policy_for_obstacle(obstacle, semantic_policy)
        radius = settings["soft_radius_m"]
        distance = distance_to_obstacle(point, obstacle)
        if radius > 0.0 and distance < radius:
            normalized = 1.0 - distance / radius
            risk += settings["risk_weight"] * normalized * normalized
    return risk


def edge_cost(
    start,
    end,
    obstacles,
    semantic_policy,
    semantic_cost_scale=1.0,
    sample_resolution_m=DEFAULT_RISK_SAMPLE_RESOLUTION_M,
):
    start_x, start_y = point_xy(start)
    end_x, end_y = point_xy(end)
    distance = math.hypot(end_x - start_x, end_y - start_y)
    if distance <= 1e-12:
        return 0.0
    samples = max(1, int(math.ceil(distance / sample_resolution_m)))
    accumulated_risk = 0.0
    for index in range(samples):
        ratio = (index + 0.5) / samples
        point = (start_x + ratio * (end_x - start_x), start_y + ratio * (end_y - start_y))
        accumulated_risk += semantic_risk(point, obstacles, semantic_policy)
    mean_risk = accumulated_risk / samples
    return distance * (1.0 + max(0.0, float(semantic_cost_scale)) * mean_risk)


def reconstruct_path(parent, coordinates, goal_index):
    indices = [goal_index]
    while parent[indices[-1]] != indices[-1]:
        indices.append(parent[indices[-1]])
    indices.reverse()
    return [coordinates[index] for index in indices]


def path_cost(path, obstacles, semantic_policy=None, clearance_m=DEFAULT_CLEARANCE_M, semantic_cost_scale=1.0):
    policy = validated_semantic_policy(semantic_policy, clearance_m)
    return sum(
        edge_cost(first, second, obstacles, policy, semantic_cost_scale)
        for first, second in zip(path, path[1:])
    )


def simplify_path(
    path,
    obstacles,
    workspace,
    semantic_policy,
    clearance_m,
    semantic_cost_scale,
    max_waypoints,
    cost_tolerance=0.02,
):
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    start_index = 0
    while start_index < len(path) - 1:
        chosen = start_index + 1
        original_cost = 0.0
        subpath_costs = {start_index + 1: edge_cost(
            path[start_index], path[start_index + 1], obstacles, semantic_policy, semantic_cost_scale
        )}
        original_cost = subpath_costs[start_index + 1]
        for end_index in range(start_index + 2, len(path)):
            original_cost += edge_cost(
                path[end_index - 1], path[end_index], obstacles, semantic_policy, semantic_cost_scale
            )
            subpath_costs[end_index] = original_cost
        for end_index in range(len(path) - 1, start_index, -1):
            if not segment_clear(path[start_index], path[end_index], obstacles, workspace, semantic_policy, clearance_m):
                continue
            direct_cost = edge_cost(
                path[start_index], path[end_index], obstacles, semantic_policy, semantic_cost_scale
            )
            if direct_cost <= subpath_costs[end_index] * (1.0 + cost_tolerance):
                chosen = end_index
                break
        simplified.append(path[chosen])
        start_index = chosen

    while len(simplified) > max_waypoints:
        best = None
        for index in range(1, len(simplified) - 1):
            if not segment_clear(
                simplified[index - 1], simplified[index + 1], obstacles, workspace, semantic_policy, clearance_m
            ):
                continue
            before = edge_cost(
                simplified[index - 1], simplified[index], obstacles, semantic_policy, semantic_cost_scale
            ) + edge_cost(
                simplified[index], simplified[index + 1], obstacles, semantic_policy, semantic_cost_scale
            )
            after = edge_cost(
                simplified[index - 1], simplified[index + 1], obstacles, semantic_policy, semantic_cost_scale
            )
            candidate = (after - before, index)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            break
        simplified.pop(best[1])
    return simplified


def to_waypoints(path, fixed_z):
    return [
        {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2), "z": round(float(fixed_z), 2)}
        for point in path
    ]


def plan_semantic_theta(
    start,
    goal,
    obstacles,
    workspace=None,
    semantic_policy=None,
    clearance_m=DEFAULT_CLEARANCE_M,
    grid_resolution_m=DEFAULT_GRID_RESOLUTION_M,
    semantic_cost_scale=1.0,
    max_waypoints=DEFAULT_MAX_WAYPOINTS,
):
    """Plan a hard-safe, semantic-cost-aware fixed-altitude path with Basic Theta*."""
    workspace = workspace or DEFAULT_WORKSPACE
    if grid_resolution_m <= 0.0:
        raise ValueError("grid_resolution_m must be positive")
    if max_waypoints < 2:
        raise ValueError("max_waypoints must be at least two")
    policy = validated_semantic_policy(semantic_policy, clearance_m)
    start_xy, goal_xy = point_xy(start), point_xy(goal)
    if not point_clear(start_xy, obstacles, workspace, policy, clearance_m):
        raise ValueError("Start is outside the workspace or inside a semantic hard-safety region.")
    if not point_clear(goal_xy, obstacles, workspace, policy, clearance_m):
        raise ValueError("Goal is outside the workspace or inside a semantic hard-safety region.")

    x_limits = workspace.get("x", DEFAULT_WORKSPACE["x"])
    y_limits = workspace.get("y", DEFAULT_WORKSPACE["y"])
    min_x, max_x = float(x_limits[0]), float(x_limits[1])
    min_y, max_y = float(y_limits[0]), float(y_limits[1])
    x_count = int(math.floor((max_x - min_x) / grid_resolution_m + 1e-9)) + 1
    y_count = int(math.floor((max_y - min_y) / grid_resolution_m + 1e-9)) + 1

    def nearest_index(point):
        x, y = point_xy(point)
        return (
            min(x_count - 1, max(0, int(round((x - min_x) / grid_resolution_m)))),
            min(y_count - 1, max(0, int(round((y - min_y) / grid_resolution_m)))),
        )

    start_index, goal_index = nearest_index(start_xy), nearest_index(goal_xy)
    if start_index == goal_index:
        if segment_clear(start_xy, goal_xy, obstacles, workspace, policy, clearance_m):
            return to_waypoints([start_xy, goal_xy], workspace.get("z", goal.get("z", -0.25) if isinstance(goal, dict) else -0.25))
        raise RuntimeError("Start and goal map to one blocked grid cell at the requested resolution.")

    coordinates = {
        (x_index, y_index): (min_x + x_index * grid_resolution_m, min_y + y_index * grid_resolution_m)
        for x_index in range(x_count)
        for y_index in range(y_count)
    }
    coordinates[start_index] = start_xy
    coordinates[goal_index] = goal_xy

    g_score = {start_index: 0.0}
    parent = {start_index: start_index}
    queue = [(math.dist(start_xy, goal_xy), 0, start_index)]
    closed = set()
    counter = 0

    while queue:
        _, _, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal_index:
            raw_path = reconstruct_path(parent, coordinates, goal_index)
            simplified = simplify_path(
                raw_path,
                obstacles,
                workspace,
                policy,
                clearance_m,
                semantic_cost_scale,
                max_waypoints,
            )
            if len(simplified) > max_waypoints:
                raise RuntimeError(f"Safe path requires {len(simplified)} waypoints; maximum is {max_waypoints}.")
            fixed_z = workspace.get("z", goal.get("z", -0.25) if isinstance(goal, dict) else -0.25)
            return to_waypoints(simplified, fixed_z)
        closed.add(current)
        current_x, current_y = current
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                if offset_x == 0 and offset_y == 0:
                    continue
                neighbor = (current_x + offset_x, current_y + offset_y)
                if not (0 <= neighbor[0] < x_count and 0 <= neighbor[1] < y_count):
                    continue
                if neighbor in closed:
                    continue
                neighbor_point = coordinates[neighbor]
                if not point_clear(neighbor_point, obstacles, workspace, policy, clearance_m):
                    continue

                ancestor = parent[current]
                if segment_clear(coordinates[ancestor], neighbor_point, obstacles, workspace, policy, clearance_m):
                    candidate_parent = ancestor
                else:
                    candidate_parent = current
                tentative = g_score[candidate_parent] + edge_cost(
                    coordinates[candidate_parent],
                    neighbor_point,
                    obstacles,
                    policy,
                    semantic_cost_scale,
                )
                if tentative + 1e-12 >= g_score.get(neighbor, math.inf):
                    continue
                g_score[neighbor] = tentative
                parent[neighbor] = candidate_parent
                heuristic = math.dist(neighbor_point, goal_xy)
                counter += 1
                heapq.heappush(queue, (tentative + heuristic, counter, neighbor))

    raise RuntimeError("Semantic Theta* failed to find a path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help='JSON point, e.g. {"x":0,"y":0,"z":-0.25}')
    parser.add_argument("--goal", required=True, help='JSON point, e.g. {"x":2.5,"y":0,"z":-0.25}')
    parser.add_argument("--obstacles", default="[]", help="JSON semantic obstacle boxes.")
    parser.add_argument("--workspace", default=json.dumps(DEFAULT_WORKSPACE), help="JSON workspace with x/y limits and z.")
    parser.add_argument("--semantic-policy", default="{}", help="JSON per-label hard margins, soft radii, and risk weights.")
    parser.add_argument("--clearance-m", type=float, default=DEFAULT_CLEARANCE_M)
    parser.add_argument("--grid-resolution-m", type=float, default=DEFAULT_GRID_RESOLUTION_M)
    parser.add_argument("--semantic-cost-scale", type=float, default=1.0)
    parser.add_argument("--max-waypoints", type=int, default=DEFAULT_MAX_WAYPOINTS)
    args = parser.parse_args()

    obstacles = json.loads(args.obstacles)
    semantic_policy = validated_semantic_policy(json.loads(args.semantic_policy), args.clearance_m)
    path = plan_semantic_theta(
        json.loads(args.start),
        json.loads(args.goal),
        obstacles,
        workspace=json.loads(args.workspace),
        semantic_policy=semantic_policy,
        clearance_m=args.clearance_m,
        grid_resolution_m=args.grid_resolution_m,
        semantic_cost_scale=args.semantic_cost_scale,
        max_waypoints=args.max_waypoints,
    )
    print(
        json.dumps(
            {
                "waypoints": path,
                "semantic_cost": round(path_cost(path, obstacles, semantic_policy, args.clearance_m, args.semantic_cost_scale), 6),
                "semantic_policy": semantic_policy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
