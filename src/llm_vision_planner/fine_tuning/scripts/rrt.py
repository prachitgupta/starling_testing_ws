#!/usr/bin/env python3
import argparse
import json
import math
import random


DEFAULT_WORKSPACE = {"x": [0.0, 4.0], "y": [0.0, 4.0], "z": -0.25}
DEFAULT_CLEARANCE_M = 0.40


def point_xy(point):
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def obstacle_bounds(obstacle, clearance_m=0.0):
    min_corner = obstacle.get("min_corner", [0.0, 0.0, 0.0])
    max_corner = obstacle.get("max_corner", [0.0, 0.0, 0.0])
    return (
        float(min_corner[0]) - clearance_m,
        float(max_corner[0]) + clearance_m,
        float(min_corner[1]) - clearance_m,
        float(max_corner[1]) + clearance_m,
    )


def in_workspace_xy(x, y, workspace):
    x_limits = workspace.get("x", [0.0, 4.0])
    y_limits = workspace.get("y", [0.0, 4.0])
    return float(x_limits[0]) <= x <= float(x_limits[1]) and float(y_limits[0]) <= y <= float(y_limits[1])


def point_clear(point, obstacles, workspace, clearance_m):
    x, y = point_xy(point)
    if not in_workspace_xy(x, y, workspace):
        return False
    for obstacle in obstacles:
        min_x, max_x, min_y, max_y = obstacle_bounds(obstacle, clearance_m)
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return False
    return True


def segment_clear(a, b, obstacles, workspace, clearance_m, resolution_m=0.05):
    ax, ay = point_xy(a)
    bx, by = point_xy(b)
    distance = math.hypot(bx - ax, by - ay)
    steps = max(1, int(math.ceil(distance / resolution_m)))
    for step in range(steps + 1):
        ratio = step / steps
        point = (ax + (bx - ax) * ratio, ay + (by - ay) * ratio)
        if not point_clear(point, obstacles, workspace, clearance_m):
            return False
    return True


def steer(from_xy, to_xy, step_size_m):
    fx, fy = point_xy(from_xy)
    tx, ty = point_xy(to_xy)
    distance = math.hypot(tx - fx, ty - fy)
    if distance <= step_size_m:
        return tx, ty
    scale = step_size_m / distance
    return fx + (tx - fx) * scale, fy + (ty - fy) * scale


def nearest_node(nodes, sample_xy):
    sx, sy = sample_xy
    return min(range(len(nodes)), key=lambda idx: math.hypot(nodes[idx][0] - sx, nodes[idx][1] - sy))


def reconstruct_path(nodes, parents, goal_index):
    path = []
    index = goal_index
    while index is not None:
        path.append(nodes[index])
        index = parents[index]
    path.reverse()
    return path


def smooth_path(path, obstacles, workspace, clearance_m, passes=80):
    if len(path) <= 2:
        return path
    path = list(path)
    for _ in range(passes):
        if len(path) <= 2:
            break
        i = random.randint(0, len(path) - 3)
        j = random.randint(i + 2, len(path) - 1)
        if segment_clear(path[i], path[j], obstacles, workspace, clearance_m):
            path = path[: i + 1] + path[j:]
    return path


def to_waypoints(path_xy, fixed_z):
    return [{"x": round(x, 2), "y": round(y, 2), "z": round(float(fixed_z), 2)} for x, y in path_xy]


def plan_rrt(
    start,
    goal,
    obstacles,
    workspace=None,
    clearance_m=DEFAULT_CLEARANCE_M,
    step_size_m=0.35,
    goal_sample_rate=0.20,
    max_iterations=3000,
    seed=None,
    smooth=True,
):
    workspace = workspace or DEFAULT_WORKSPACE
    if seed is not None:
        random.seed(seed)

    start_xy = point_xy(start)
    goal_xy = point_xy(goal)
    fixed_z = workspace.get("z", goal.get("z", -0.25) if isinstance(goal, dict) else -0.25)
    if not point_clear(start_xy, obstacles, workspace, clearance_m):
        raise ValueError("Start is outside workspace or inside an inflated obstacle.")
    if not point_clear(goal_xy, obstacles, workspace, clearance_m):
        raise ValueError("Goal is outside workspace or inside an inflated obstacle.")

    nodes = [start_xy]
    parents = [None]
    x_limits = workspace.get("x", [0.0, 4.0])
    y_limits = workspace.get("y", [0.0, 4.0])

    for _ in range(max_iterations):
        if random.random() < goal_sample_rate:
            sample = goal_xy
        else:
            sample = (random.uniform(float(x_limits[0]), float(x_limits[1])), random.uniform(float(y_limits[0]), float(y_limits[1])))

        nearest_index = nearest_node(nodes, sample)
        new_xy = steer(nodes[nearest_index], sample, step_size_m)
        if not segment_clear(nodes[nearest_index], new_xy, obstacles, workspace, clearance_m):
            continue

        nodes.append(new_xy)
        parents.append(nearest_index)
        new_index = len(nodes) - 1
        if math.hypot(new_xy[0] - goal_xy[0], new_xy[1] - goal_xy[1]) <= step_size_m:
            if segment_clear(new_xy, goal_xy, obstacles, workspace, clearance_m):
                nodes.append(goal_xy)
                parents.append(new_index)
                path = reconstruct_path(nodes, parents, len(nodes) - 1)
                if smooth:
                    path = smooth_path(path, obstacles, workspace, clearance_m)
                return to_waypoints(path, fixed_z)

    raise RuntimeError("RRT failed to find a path.")


def main():
    parser = argparse.ArgumentParser(description="Generate an expert collision-free path with RRT.")
    parser.add_argument("--start", required=True, help='JSON point, e.g. {"x":0,"y":0,"z":-0.25}')
    parser.add_argument("--goal", required=True, help='JSON point, e.g. {"x":2.5,"y":0,"z":-0.25}')
    parser.add_argument("--obstacles", default="[]", help="JSON list of obstacle boxes with min_corner/max_corner.")
    parser.add_argument("--workspace", default=json.dumps(DEFAULT_WORKSPACE), help="JSON workspace with x/y limits and z.")
    parser.add_argument("--clearance-m", type=float, default=DEFAULT_CLEARANCE_M)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    path = plan_rrt(
        json.loads(args.start),
        json.loads(args.goal),
        json.loads(args.obstacles),
        workspace=json.loads(args.workspace),
        clearance_m=args.clearance_m,
        seed=args.seed,
    )
    print(json.dumps({"waypoints": path}, indent=2))


if __name__ == "__main__":
    main()
