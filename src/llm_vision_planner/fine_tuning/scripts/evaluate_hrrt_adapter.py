#!/usr/bin/env python3
"""Evaluate an HRRT LoRA adapter on the locked, human-reviewed test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


DISTILLATION_MODES = ("dss", "dss_scott")


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_plan(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def xy(point):
    return float(point["x"]), float(point["y"])


def resample(waypoints, count=50):
    points = [xy(point) for point in waypoints]
    if len(points) < 2:
        return points * count
    lengths = [0.0]
    for first, second in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(first, second))
    if lengths[-1] <= 1e-9:
        return [points[0]] * count
    sampled = []
    segment = 0
    for index in range(count):
        target = lengths[-1] * index / (count - 1)
        while segment + 1 < len(lengths) - 1 and lengths[segment + 1] < target:
            segment += 1
        span = lengths[segment + 1] - lengths[segment]
        ratio = 0.0 if span <= 1e-9 else (target - lengths[segment]) / span
        sampled.append(
            (
                points[segment][0] + ratio * (points[segment + 1][0] - points[segment][0]),
                points[segment][1] + ratio * (points[segment + 1][1] - points[segment][1]),
            )
        )
    return sampled


def trajectory_distance(first, second):
    return sum(math.dist(a, b) for a, b in zip(resample(first), resample(second))) / 50.0


def clearance_to_box(point, obstacle):
    x, y = xy(point)
    minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
    dx = max(float(minimum[0]) - x, 0.0, x - float(maximum[0]))
    dy = max(float(minimum[1]) - y, 0.0, y - float(maximum[1]))
    return math.hypot(dx, dy)


def is_feasible(waypoints, environment, clearance_m):
    if not isinstance(waypoints, list) or not 2 <= len(waypoints) <= 8:
        return False
    workspace = environment["workspace"]
    if any(abs(float(point["z"]) - float(workspace["z"])) > 0.05 for point in waypoints):
        return False
    for point in resample(waypoints, 100):
        x, y = point
        if not (workspace["x"][0] <= x <= workspace["x"][1] and workspace["y"][0] <= y <= workspace["y"][1]):
            return False
        if any(clearance_to_box({"x": x, "y": y}, obstacle) < clearance_m for obstacle in environment["obstacles"]):
            return False
    return True


def infer_route_id(waypoints, routes):
    return min(routes, key=lambda route: trajectory_distance(waypoints, route["waypoints"]))["route_id"]


def generate_predictions(rows, args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    predictions = {}
    for row in rows:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        predictions[row["sample_id"]] = tokenizer.decode(
            generated[0, input_ids.shape[-1] :], skip_special_tokens=True
        )
    return predictions


def load_predictions(path: Path):
    with path.open(encoding="utf-8") as stream:
        return {str(row["sample_id"]): str(row["prediction"]) for row in map(json.loads, stream)}


def evaluate(rows, predictions, clearance_m):
    totals = {
        "samples": len(rows),
        "structured_valid": 0,
        "reasoning_schema_valid": 0,
        "feasible": 0,
        "route_match": 0,
    }
    endpoint_errors, path_errors = [], []
    details = []
    for row in rows:
        prediction = predictions.get(row["sample_id"], "")
        detail = {"sample_id": row["sample_id"]}
        try:
            plan = parse_plan(prediction)
            waypoints = plan["waypoints"]
            reasoning = plan.get("reasoning")
            if not isinstance(reasoning, str) or not isinstance(waypoints, list):
                raise ValueError("missing reasoning or waypoints")
            if not 2 <= len(waypoints) <= 8:
                raise ValueError("waypoints must contain between 2 and 8 points")
            for point in waypoints:
                if not isinstance(point, dict) or not all(axis in point for axis in ("x", "y", "z")):
                    raise ValueError("each waypoint must contain numeric x, y, and z")
                for axis in ("x", "y", "z"):
                    if not math.isfinite(float(point[axis])):
                        raise ValueError("waypoint coordinates must be finite")
            totals["structured_valid"] += 1
            reasoning_schema_valid = all(
                field in reasoning for field in ("Preference:", "Geometry:", "Decision:", "Safety:")
            )
            totals["reasoning_schema_valid"] += int(reasoning_schema_valid)
            environment = json.loads(row["environment"])
            routes = json.loads(row["candidate_routes"])
            target = json.loads(row["completion"])["waypoints"]
            predicted_route_id = infer_route_id(waypoints, routes)
            feasible = is_feasible(waypoints, environment, clearance_m)
            endpoint_error = math.dist(xy(waypoints[-1]), xy(target[-1]))
            path_error = trajectory_distance(waypoints, target)
            totals["feasible"] += int(feasible)
            totals["route_match"] += int(predicted_route_id == row["selected_route_id"])
            endpoint_errors.append(endpoint_error)
            path_errors.append(path_error)
            detail.update(
                structured_valid=True,
                reasoning_schema_valid=reasoning_schema_valid,
                feasible=feasible,
                predicted_route_id=predicted_route_id,
                target_route_id=row["selected_route_id"],
                endpoint_error_m=endpoint_error,
                mean_path_error_m=path_error,
            )
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
            detail.update(structured_valid=False, error=str(exc))
        details.append(detail)
    denominator = max(1, totals["samples"])
    metrics = {
        **totals,
        "structured_valid_rate": totals["structured_valid"] / denominator,
        "reasoning_schema_valid_rate": totals["reasoning_schema_valid"] / denominator,
        "feasible_rate": totals["feasible"] / denominator,
        "route_match_rate": totals["route_match"] / denominator,
        "mean_endpoint_error_m": sum(endpoint_errors) / len(endpoint_errors) if endpoint_errors else None,
        "mean_path_error_m": sum(path_errors) / len(path_errors) if path_errors else None,
    }
    return {"metrics": metrics, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("fine_tuning/datasets/hrrt_sft_test.csv"))
    parser.add_argument("--adapter", type=Path, default=Path("fine_tuning/outputs/llama31_8b_hrrt_lora"))
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--distillation-mode", choices=DISTILLATION_MODES, default="dss")
    parser.add_argument("--predictions", type=Path, help="Optional JSONL predictions for offline scoring.")
    parser.add_argument("--output", type=Path, default=Path("fine_tuning/outputs/llama31_8b_hrrt_lora/test_metrics.json"))
    parser.add_argument("--clearance-m", type=float, default=0.40)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    args = parser.parse_args()
    rows = read_rows(args.test)
    modes = {row.get("distillation_mode", "dss") for row in rows}
    if modes != {args.distillation_mode}:
        raise ValueError(
            f"{args.test} contains distillation modes {sorted(modes)}, expected {args.distillation_mode!r}"
        )
    predictions = load_predictions(args.predictions) if args.predictions else generate_predictions(rows, args)
    report = evaluate(rows, predictions, args.clearance_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
