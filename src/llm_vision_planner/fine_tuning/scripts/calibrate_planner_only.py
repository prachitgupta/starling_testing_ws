#!/usr/bin/env python3
"""Build the fixed Case-A planner-only split-conformal certificate.

This demonstration deliberately removes environment uncertainty: the HRRT
reference and Transformer both receive the same ground-truth obstacle boxes.
One ``shortest`` or ``balanced`` row is selected per held-out layout before
scoring.  The hardware ideal-double-integrator score is

    S = max(S_residual, beta * lambda / projection_gain * language_regret)

where the regret term is zero unless the Transformer changes corridor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, PACKAGE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from min_control_qp import generate_shared_pair  # noqa: E402
from postprocess_residual_calibration import compact, residual_score, safe_path  # noqa: E402
from residual_certificate import certificate, projection_geometry  # noqa: E402
from waypoint_transformer import WaypointTransformerPlanner, refine_waypoints  # noqa: E402


CASE_A_PREFERENCES = ("shortest", "balanced")
DEFAULT_DATASET = (
    PACKAGE_ROOT
    / "fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10/hrrt_sft_test.csv"
)
DEFAULT_CHECKPOINT = (
    PACKAGE_ROOT
    / "fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10"
    / "final_transformer_vertical_teacher_only_80_10_10_best/checkpoint"
)
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "fine_tuning/datasets/planner_only_case_a"
OUTPUT_FIELDS = [
    "session_id",
    "capture_id",
    "scene_id",
    "preference_type",
    "l1_text",
    "l2_text",
    "selected_route_id",
    "reference_route_id",
    "corridor_mismatch",
    "language_regret_m",
    "residual_score",
    "regret_penalty",
    "conformity_score",
    "score_max_time_s",
    "score_max_residual_json",
    "predicted_waypoints_json",
    "reference_waypoints_json",
    "p_matrix_json",
    "k_matrix_json",
    "score_type",
    "dynamics_model",
    "planner_provider",
    "calibration_mode",
    "placeholder",
]


class PlanAbstention(ValueError):
    """The production geometric verifier would refuse to release this plan."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "sample_id", "scene_id", "preference_type", "preference_text",
        "selected_route_id", "environment", "route_cards", "candidate_routes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is not the expected held-out DSS split")
    return rows


def select_case_a_rows(rows, seed: int):
    """Select exactly one eligible preference per layout before scoring."""
    by_scene = defaultdict(list)
    for row in rows:
        if row["preference_type"] in CASE_A_PREFERENCES:
            by_scene[row["scene_id"]].append(row)
    if not by_scene:
        raise ValueError("dataset has no shortest/balanced Case-A rows")
    rng = random.Random(seed)
    selected = []
    for scene_id in sorted(by_scene, key=lambda value: int(value)):
        choices = sorted(by_scene[scene_id], key=lambda row: row["sample_id"])
        if {row["preference_type"] for row in choices} != set(CASE_A_PREFERENCES):
            raise ValueError(f"scene {scene_id} does not contain both Case-A preferences")
        selected.append(rng.choice(choices))
    return selected


def route_xy(waypoints):
    return np.asarray([[float(point["x"]), float(point["y"])] for point in waypoints])


def obstacle_centres(obstacles):
    return np.asarray([
        [
            0.5 * (float(item["min_corner"][0]) + float(item["max_corner"][0])),
            0.5 * (float(item["min_corner"][1]) + float(item["max_corner"][1])),
        ]
        for item in obstacles
    ], dtype=float)


def resample(waypoints, count):
    points = np.asarray(waypoints, dtype=float)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if cumulative[-1] <= 0.0:
        return np.repeat(points[:1], count, axis=0)
    samples = np.linspace(0.0, cumulative[-1], count)
    result = np.column_stack(
        [np.interp(samples, cumulative, points[:, axis]) for axis in range(2)]
    )
    result[0], result[-1] = points[0], points[-1]
    return result


def swept_angles(waypoints, centres, densify=40):
    if len(centres) == 0:
        return np.empty(0)
    points = np.asarray(waypoints, dtype=float)
    fractions = np.linspace(0.0, 1.0, densify, endpoint=False)
    dense = np.concatenate(
        [
            points[index]
            + fractions[:, None] * (points[index + 1] - points[index])
            for index in range(len(points) - 1)
        ]
        + [points[-1:]]
    )
    offsets = dense[None, :, :] - np.asarray(centres)[:, None, :]
    angles = np.unwrap(np.arctan2(offsets[..., 1], offsets[..., 0]), axis=1)
    return angles[:, -1] - angles[:, 0]


def same_corridor(first, second, centres):
    return bool(np.all(np.abs(swept_angles(first, centres) - swept_angles(second, centres)) < np.pi))


def nearest_candidate(predicted_xy, routes):
    distances = [
        np.linalg.norm(predicted_xy - resample(route_xy(route["waypoints"]), len(predicted_xy)), axis=1).mean()
        for route in routes
    ]
    return routes[int(np.argmin(distances))]


def language_regret(preference_type, cards, reference_route_id, selected_route_id):
    if reference_route_id == selected_route_id:
        return 0.0
    by_id = {str(card["route_id"]): card for card in cards}
    reference = by_id[str(reference_route_id)]
    selected = by_id[str(selected_route_id)]
    if preference_type == "shortest":
        value = float(reference["path_length_m"]) - float(selected["path_length_m"])
    elif preference_type == "balanced":
        min_length = max(min(float(card["path_length_m"]) for card in cards), 1e-6)
        max_clearance = max(
            max(float(card["overall_minimum_clearance_m"]) for card in cards), 1e-6
        )
        objective = lambda card: (
            float(card["path_length_m"]) / min_length
            - float(card["overall_minimum_clearance_m"]) / max_clearance
        )
        value = (objective(reference) - objective(selected)) * min_length
    else:
        raise ValueError(f"unsupported Case-A preference: {preference_type}")
    return max(0.0, float(value))


def combined_score(residual, regret_m, beta):
    rate, projection_gain = projection_geometry()
    penalty = float(beta) * rate / projection_gain * float(regret_m)
    return max(float(residual), penalty), penalty


def score_row(row, planner, args):
    environment = json.loads(row["environment"])
    routes = json.loads(row["candidate_routes"])
    cards = json.loads(row["route_cards"])
    start = environment["start"]
    goal = environment["goal"]
    workspace = environment["workspace"]
    obstacles = environment["obstacles"]
    selected = next(
        route for route in routes if str(route["route_id"]) == str(row["selected_route_id"])
    )

    raw_predicted = planner.predict(
        start, goal, obstacles, workspace, row["preference_text"]
    )
    predicted_xy = route_xy(raw_predicted)
    mismatch = not same_corridor(
        predicted_xy, route_xy(selected["waypoints"]), obstacle_centres(obstacles)
    )
    reference = nearest_candidate(predicted_xy, routes) if mismatch else selected
    regret = language_regret(
        row["preference_type"], cards, str(reference["route_id"]), str(selected["route_id"])
    ) if mismatch else 0.0

    predicted = refine_waypoints(
        raw_predicted, workspace, obstacles, args.clearance_m, float(workspace["z"])
    )
    reference_waypoints = refine_waypoints(
        reference["waypoints"], workspace, obstacles, args.clearance_m, float(workspace["z"])
    )
    if not safe_path(predicted, obstacles, workspace, args.clearance_m):
        raise PlanAbstention(
            "refined Transformer route fails ground-truth collision checking"
        )
    expert_trajectory, predicted_trajectory = generate_shared_pair(
        reference_waypoints,
        predicted,
        workspace=workspace,
        obstacles=obstacles,
        dt=args.dt,
        max_velocity_mps=args.max_velocity_mps,
        max_acceleration_mps2=args.max_acceleration_mps2,
        damping=0.0,
    )
    maximum, p_matrix, k_matrix = residual_score(
        expert_trajectory["samples"], predicted_trajectory["samples"]
    )
    score, penalty = combined_score(maximum[0], regret, args.beta)
    return {
        "session_id": "planner-only-case-a-test-derived",
        "capture_id": row["sample_id"],
        "scene_id": row["scene_id"],
        "preference_type": row["preference_type"],
        "l1_text": "Reach the dataset goal while avoiding every obstacle.",
        "l2_text": row["preference_text"],
        "selected_route_id": str(selected["route_id"]),
        "reference_route_id": str(reference["route_id"]),
        "corridor_mismatch": str(mismatch).lower(),
        "language_regret_m": f"{regret:.17g}",
        "residual_score": f"{maximum[0]:.17g}",
        "regret_penalty": f"{penalty:.17g}",
        "conformity_score": f"{score:.17g}",
        "score_max_time_s": f"{maximum[1]:.9f}",
        "score_max_residual_json": compact([float(value) for value in maximum[2]]),
        "predicted_waypoints_json": compact(predicted),
        "reference_waypoints_json": compact(reference_waypoints),
        "p_matrix_json": compact(p_matrix.tolist()),
        "k_matrix_json": compact(k_matrix.tolist()),
        "score_type": "max_ideal_double_integrator_residual_case_a_language_regret",
        "dynamics_model": "ideal_planar_double_integrator",
        "planner_provider": "transformer",
        "calibration_mode": "planner_only",
        "placeholder": "false",
    }


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--clearance-m", type=float, default=0.4)
    parser.add_argument("--max-velocity-mps", type=float, default=0.5)
    parser.add_argument("--max-acceleration-mps2", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="Smoke test only; cannot emit a deployment certificate.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not 0.0 < args.delta < 1.0 or not math.isfinite(args.beta) or args.beta <= 0.0:
        raise SystemExit("--delta must be in (0,1) and --beta must be positive")
    dataset = args.dataset.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_csv = output_dir / "Dcal.scored.csv"
    output_certificate = output_dir / "split_certificate.json"
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.limit is not None and args.limit < len(select_case_a_rows(load_rows(dataset), args.seed)):
        output_csv = output_dir / "smoke.scored.csv"
        output_certificate = None
    if output_csv.exists() and not args.overwrite:
        raise SystemExit(f"{output_csv} exists; pass --overwrite to replace it")

    selected = select_case_a_rows(load_rows(dataset), args.seed)
    if args.limit is not None:
        selected = selected[: args.limit]
    planner = WaypointTransformerPlanner(checkpoint, args.device)
    scored = []
    abstentions = []
    failures = []
    for index, row in enumerate(selected, start=1):
        try:
            scored.append(score_row(row, planner, args))
        except PlanAbstention as exc:
            abstentions.append({"sample_id": row["sample_id"], "reason": str(exc)})
        except Exception as exc:  # computational/data failures refuse the entire certificate
            failures.append({"sample_id": row["sample_id"], "error": str(exc)})
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(
                f"processed {index}/{len(selected)}; accepted={len(scored)}; "
                f"abstained={len(abstentions)}; failures={len(failures)}",
                flush=True,
            )
    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scoring.failures.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in failures),
            encoding="utf-8",
        )
        raise SystemExit("No certificate written because at least one selected layout failed scoring")

    output_dir.mkdir(parents=True, exist_ok=True)
    stale_failures = output_dir / "scoring.failures.jsonl"
    if stale_failures.exists():
        stale_failures.unlink()
    (output_dir / "abstentions.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in abstentions),
        encoding="utf-8",
    )
    write_csv(output_csv, scored)
    if output_certificate is None:
        print(f"wrote smoke scores only: {output_csv}")
        return
    metadata = {
        "calibration_csv": output_csv.name,
        "calibration_dataset_sha256": file_hash(output_csv),
        "source_dataset": str(dataset),
        "source_dataset_sha256": file_hash(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": file_hash(checkpoint / "model.pt"),
        "calibration_mode": "planner_only",
        "coverage_scope": "test-derived Case-A planner-only demonstration; no environment uncertainty",
        "preferences": list(CASE_A_PREFERENCES),
        "selection": "one seeded shortest-or-balanced row per scene_id",
        "selection_seed": args.seed,
        "selected_layout_count": len(selected),
        "accepted_layout_count": len(scored),
        "abstained_layout_count": len(abstentions),
        "acceptance_rate": len(scored) / len(selected),
        "selective_coverage": "conditional on the frozen production collision verifier releasing the plan",
        "score_type": "max_ideal_double_integrator_residual_case_a_language_regret",
        "beta": args.beta,
        "max_velocity_mps": args.max_velocity_mps,
        "max_acceleration_mps2": args.max_acceleration_mps2,
        "qp_dt_s": args.dt,
        "environment_uncertainty_included": False,
        "hardware_validated": False,
    }
    result = certificate(
        [float(row["conformity_score"]) for row in scored], args.delta, metadata
    )
    output_certificate.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
