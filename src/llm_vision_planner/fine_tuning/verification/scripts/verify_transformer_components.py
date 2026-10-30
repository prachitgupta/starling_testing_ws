#!/usr/bin/env python3
"""Verify the HRRT waypoint transformer before and after full training.

This imports the production trainer so the checks exercise the same model,
dataset parser, loss, normalization, and checkpoint format used in training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
TRAINER_DIR = PACKAGE_ROOT / "fine_tuning" / "scripts"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

import train_hrrt_waypoint_transformer as trainer_v1  # noqa: E402
import train_hrrt_waypoint_transformer_2_0 as trainer_v2  # noqa: E402


BatchCollator = trainer_v1.BatchCollator
PlanningExample = trainer_v1.PlanningExample
WaypointDataset = trainer_v1.WaypointDataset
build_model = trainer_v1.build_model
denormalize_predictions = trainer_v1.denormalize_predictions
euclidean_imitation_loss = trainer_v1.euclidean_imitation_loss
load_yaml = trainer_v1.load_yaml
move_batch = trainer_v1.move_batch
read_examples = trainer_v1.read_examples
require_section = trainer_v1.require_section
set_seed = trainer_v1.set_seed


def select_trainer(model_config: Mapping[str, Any]) -> None:
    """Select the checkpoint-compatible trainer without duplicating verification."""
    global BatchCollator, PlanningExample, WaypointDataset, build_model
    global denormalize_predictions, euclidean_imitation_loss, read_examples
    trainer = (
        trainer_v2
        if model_config.get("architecture_version") == "transformer_2_0"
        else trainer_v1
    )
    BatchCollator = trainer.BatchCollator
    PlanningExample = trainer.PlanningExample
    WaypointDataset = trainer.WaypointDataset
    build_model = trainer.build_model
    denormalize_predictions = trainer.denormalize_predictions
    euclidean_imitation_loss = trainer.euclidean_imitation_loss
    read_examples = trainer.read_examples


REQUIRED_COLUMNS = {"scene_id", "preference_text", "environment", "completion"}


def path_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def read_split_metadata(path: Path) -> dict[str, Any]:
    scene_ids: set[str] = set()
    rows = 0
    waypoint_rows = 0
    waypoint_counts: list[int] = []
    coordinate_errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            scene_ids.add(str(row["scene_id"]))
            environment = json.loads(row["environment"])
            completion = json.loads(row["completion"])
            waypoints = completion.get("waypoints")
            if not isinstance(waypoints, list) or len(waypoints) < 2:
                continue
            waypoint_rows += 1
            waypoint_counts.append(len(waypoints))
            workspace = environment["workspace"]
            x0, x1 = map(float, workspace["x"])
            y0, y1 = map(float, workspace["y"])
            for name in ("start", "goal"):
                point = environment[name]
                x, y = float(point["x"]), float(point["y"])
                if not (x0 <= x <= x1 and y0 <= y <= y1):
                    coordinate_errors.append(f"{row.get('sample_id', rows)}:{name}")
            for index, obstacle in enumerate(environment.get("obstacles", []), start=1):
                minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
                if float(minimum[0]) > float(maximum[0]) or float(minimum[1]) > float(maximum[1]):
                    coordinate_errors.append(f"{row.get('sample_id', rows)}:obstacle-{index}")
                if not (
                    x0 <= float(minimum[0]) <= x1
                    and x0 <= float(maximum[0]) <= x1
                    and y0 <= float(minimum[1]) <= y1
                    and y0 <= float(maximum[1]) <= y1
                ):
                    coordinate_errors.append(f"{row.get('sample_id', rows)}:obstacle-{index}-outside-workspace")
            for index, waypoint in enumerate(waypoints):
                x, y = float(waypoint["x"]), float(waypoint["y"])
                if not (x0 <= x <= x1 and y0 <= y <= y1):
                    coordinate_errors.append(f"{row.get('sample_id', rows)}:waypoint-{index}-outside-workspace")
    if waypoint_rows == 0:
        raise ValueError(f"{path} contains no waypoint-labelled rows")
    return {
        "path": str(path),
        "rows": rows,
        "waypoint_rows": waypoint_rows,
        "scene_count": len(scene_ids),
        "scene_ids": scene_ids,
        "original_waypoint_count": {
            "minimum": min(waypoint_counts),
            "maximum": max(waypoint_counts),
            "mean": float(np.mean(waypoint_counts)),
        },
        "coordinate_error_count": len(coordinate_errors),
        "coordinate_error_examples": coordinate_errors[:10],
    }


def dataset_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    reports = {name: read_split_metadata(path) for name, path in paths.items()}
    overlaps: dict[str, list[str]] = {}
    names = list(reports)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}_{right}"] = sorted(reports[left]["scene_ids"] & reports[right]["scene_ids"])[:20]
    for report in reports.values():
        report.pop("scene_ids")
    passed = all(not shared for shared in overlaps.values()) and all(
        report["coordinate_error_count"] == 0 for report in reports.values()
    )
    return {"passed": passed, "splits": reports, "scene_overlap_examples": overlaps}


def make_optimizer(model: Any, training_config: Mapping[str, Any], lr_multiplier: float = 1.0):
    from torch.optim import AdamW

    text_parameters = list(model.text_encoder.parameters())
    text_ids = {id(parameter) for parameter in text_parameters}
    planner_parameters = [parameter for parameter in model.parameters() if id(parameter) not in text_ids]
    return AdamW(
        [
            {
                "params": text_parameters,
                "lr": float(training_config["text_encoder_learning_rate"]) * lr_multiplier,
            },
            {
                "params": planner_parameters,
                "lr": float(training_config["planner_learning_rate"]) * lr_multiplier,
            },
        ],
        weight_decay=float(training_config["weight_decay"]),
    )


def forward_loss(model: Any, batch: Mapping[str, Any]):
    normalized = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["scene_tokens"],
        batch["scene_padding_mask"],
    )
    predicted = denormalize_predictions(normalized, batch["examples"])
    return normalized, euclidean_imitation_loss(predicted, batch["target_xy"])


def grouped_gradient_norms(model: Any) -> dict[str, float]:
    groups = {
        "text_encoder": model.text_encoder,
        "scene_projection": model.scene_projection,
        "fusion_transformer": model.fusion_transformer,
        "waypoint_decoder": model.waypoint_decoder,
        "coordinate_head": model.coordinate_head,
    }
    if hasattr(model, "scene_encoder"):
        groups["scene_encoder"] = model.scene_encoder
    result: dict[str, float] = {}
    for name, module in groups.items():
        squared = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum().item())
        result[name] = math.sqrt(squared)
    route_gradient = model.route_queries.grad
    result["route_queries"] = float(route_gradient.detach().float().norm().item()) if route_gradient is not None else 0.0
    return result


def distinct_examples(examples: Sequence[PlanningExample]) -> tuple[PlanningExample, PlanningExample, PlanningExample]:
    anchor = examples[0]
    different_preference = next(
        (example for example in examples[1:] if example.preference != anchor.preference), examples[-1]
    )
    different_scene = next(
        (example for example in examples[1:] if example.environment != anchor.environment), examples[-1]
    )
    return anchor, different_preference, different_scene


def input_sensitivity(model: Any, collator: BatchCollator, examples: Sequence[PlanningExample], device: Any):
    import torch

    anchor, preference_source, scene_source = distinct_examples(examples)
    preference_variant = replace(anchor, preference=preference_source.preference)
    scene_variant = replace(
        anchor,
        environment=scene_source.environment,
        scene_tokens=scene_source.scene_tokens,
        expert_waypoints=scene_source.expert_waypoints,
        target_xy=scene_source.target_xy,
    )
    batch = move_batch(collator([anchor, preference_variant, scene_variant]), device)
    model.eval()
    with torch.no_grad():
        normalized = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["scene_tokens"],
            batch["scene_padding_mask"],
        )
    text_delta = float(torch.linalg.vector_norm(normalized[0] - normalized[1], dim=-1).mean().item())
    scene_delta = float(torch.linalg.vector_norm(normalized[0] - normalized[2], dim=-1).mean().item())
    return {
        "text_change_normalized_waypoint_delta": text_delta,
        "scene_change_normalized_waypoint_delta": scene_delta,
        "text_response_nonzero": text_delta > 1e-7,
        "scene_response_nonzero": scene_delta > 1e-7,
    }


def tiny_overfit(
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    train_examples: Sequence[PlanningExample],
    tokenizer: Any,
    device: Any,
    sample_count: int,
    updates: int,
    lr_multiplier: float,
    required_reduction: float,
) -> dict[str, Any]:
    import torch

    tiny_examples = list(train_examples[:sample_count])
    collator = BatchCollator(tokenizer, int(model_config["max_text_length"]), int(model_config["max_scene_tokens"]))
    batch = move_batch(collator(tiny_examples), device)
    model = build_model(model_config).to(device)
    optimizer = make_optimizer(model, training_config, lr_multiplier)
    model.eval()
    with torch.no_grad():
        normalized, initial_loss_tensor = forward_loss(model, batch)
    initial_loss = float(initial_loss_tensor.item())
    expected_shape = (len(tiny_examples), int(model_config["waypoint_count"]), 2)
    shape_ok = tuple(normalized.shape) == expected_shape
    best_loss = initial_loss
    for _ in range(updates):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = forward_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["max_gradient_norm"]))
        optimizer.step()
        best_loss = min(best_loss, float(loss.item()))
    model.eval()
    with torch.no_grad():
        _, final_loss_tensor = forward_loss(model, batch)
    final_loss = float(final_loss_tensor.item())
    model.zero_grad(set_to_none=True)
    _, gradient_loss = forward_loss(model, batch)
    gradient_loss.backward()
    gradients = grouped_gradient_norms(model)
    reduction = (initial_loss - final_loss) / max(initial_loss, 1e-12)
    sensitivity = input_sensitivity(model, collator, train_examples, device)
    passed = (
        shape_ok
        and math.isfinite(final_loss)
        and reduction >= required_reduction
        and all(value > 0.0 for value in gradients.values())
        and sensitivity["text_response_nonzero"]
        and sensitivity["scene_response_nonzero"]
    )
    return {
        "passed": passed,
        "samples": len(tiny_examples),
        "updates": updates,
        "learning_rate_multiplier": lr_multiplier,
        "initial_loss_m": initial_loss,
        "final_loss_m": final_loss,
        "best_observed_loss_m": best_loss,
        "relative_loss_reduction": reduction,
        "required_relative_reduction": required_reduction,
        "output_shape": list(normalized.shape),
        "expected_output_shape": list(expected_shape),
        "output_shape_correct": shape_ok,
        "gradient_norms": gradients,
        "sensitivity": sensitivity,
    }


def best_checkpoint(model_output: Path) -> Path:
    candidates: list[tuple[tuple[float, ...], Path]] = []
    for metrics_path in model_output.glob("checkpoint-epoch-*/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        radius = float(metrics.get("safety_radius_m", math.inf))
        imitation = float(metrics["euclidean_imitation_loss_m"])
        key = (0.0, radius, imitation) if math.isfinite(radius) else (1.0, imitation, 0.0)
        candidates.append((key, metrics_path.parent))
    if not candidates:
        raise ValueError(f"no checkpoint metrics found below {model_output}")
    return min(candidates, key=lambda item: item[0])[1]


def imitation_metrics(model: Any, loader: Any, device: Any) -> dict[str, float]:
    import torch

    distances: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            normalized = model(
                batch["input_ids"], batch["attention_mask"], batch["scene_tokens"], batch["scene_padding_mask"]
            )
            predicted = denormalize_predictions(normalized, batch["examples"])
            values = torch.linalg.vector_norm(predicted - batch["target_xy"], ord=2, dim=-1)
            if hasattr(model, "interior_waypoint_count"):
                values = values[:, 1:-1]
            distances.append(values.detach().cpu().numpy().reshape(-1))
    merged = np.concatenate(distances)
    return {
        "mean_euclidean_imitation_loss_m": float(np.mean(merged)),
        "median_waypoint_error_m": float(np.median(merged)),
        "p95_waypoint_error_m": float(np.quantile(merged, 0.95)),
        "maximum_waypoint_error_m": float(np.max(merged)),
    }


def route_examples(
    model: Any,
    collator: BatchCollator,
    examples: Sequence[PlanningExample],
    device: Any,
    limit: int = 6,
) -> list[dict[str, Any]]:
    import torch

    selected = list(examples[:limit])
    batch = move_batch(collator(selected), device)
    model.eval()
    with torch.no_grad():
        normalized = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["scene_tokens"],
            batch["scene_padding_mask"],
        )
        predicted = denormalize_predictions(normalized, selected).detach().cpu().numpy()
    result = []
    for example, predicted_xy in zip(selected, predicted):
        result.append(
            {
                "sample_id": example.sample_id,
                "preference": example.preference,
                "environment": example.environment,
                "expert_xy": example.target_xy.tolist(),
                "predicted_xy": predicted_xy.tolist(),
            }
        )
    return result


def postflight(model_output: Path, validation_path: Path, test_path: Path, device: Any) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    checkpoint = best_checkpoint(model_output)
    effective = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    model_config = dict(effective["model"])
    shared_encoder = model_output / "text_encoder"
    if shared_encoder.is_dir():
        model_config["text_encoder_name"] = str(shared_encoder)
    tokenizer_path = checkpoint / "tokenizer"
    if not tokenizer_path.is_dir():
        tokenizer_path = model_output / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model = build_model(model_config).to(device)
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device), strict=True)
    waypoint_count = int(model_config["waypoint_count"])
    mode = str(effective["data"]["distillation_mode"])
    validation_examples = read_examples(validation_path, mode, waypoint_count)
    test_examples = read_examples(test_path, mode, waypoint_count)
    collator = BatchCollator(tokenizer, int(model_config["max_text_length"]), int(model_config["max_scene_tokens"]))
    batch_size = int(effective["training"]["validation_batch_size"])
    validation_loader = DataLoader(WaypointDataset(validation_examples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(WaypointDataset(test_examples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    sensitivity = input_sensitivity(model, collator, validation_examples, device)
    diagnostic_batch = move_batch(collator(validation_examples[: min(8, len(validation_examples))]), device)
    model.zero_grad(set_to_none=True)
    _, loss = forward_loss(model, diagnostic_batch)
    loss.backward()
    gradients = grouped_gradient_norms(model)
    checkpoint_metrics = json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8"))
    passed = (
        all(math.isfinite(value) and value > 0.0 for value in gradients.values())
        and sensitivity["text_response_nonzero"]
        and sensitivity["scene_response_nonzero"]
    )
    return {
        "passed": passed,
        "selected_checkpoint": str(checkpoint),
        "checkpoint_metrics": checkpoint_metrics,
        "validation": imitation_metrics(model, validation_loader, device),
        "test": imitation_metrics(model, test_loader, device),
        "route_examples": route_examples(model, collator, test_examples, device),
        "gradient_norms": gradients,
        "sensitivity": sensitivity,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "postflight"), required=True)
    parser.add_argument("--config", type=path_arg, required=True)
    parser.add_argument("--train", type=path_arg, required=True)
    parser.add_argument("--validation", type=path_arg, required=True)
    parser.add_argument("--test", type=path_arg, required=True)
    parser.add_argument("--output", type=path_arg, required=True)
    parser.add_argument("--model-output", type=path_arg)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiny-samples", type=int, default=8)
    parser.add_argument("--tiny-updates", type=int, default=250)
    parser.add_argument("--tiny-lr-multiplier", type=float, default=5.0)
    parser.add_argument("--required-overfit-reduction", type=float, default=0.50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from transformers import AutoTokenizer

    config = load_yaml(args.config)
    model_config = dict(require_section(config, "model"))
    select_trainer(model_config)
    training_config = dict(require_section(config, "training"))
    data_config = dict(require_section(config, "data"))
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    set_seed(int(training_config["seed"]))
    random.seed(int(training_config["seed"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"train": args.train, "validation": args.validation, "test": args.test}
    if args.stage == "preflight":
        audit = dataset_audit(paths)
        train_examples = read_examples(args.train, str(data_config["distillation_mode"]), int(model_config["waypoint_count"]))
        tokenizer = AutoTokenizer.from_pretrained(str(model_config["text_encoder_name"]), use_fast=True)
        tiny = tiny_overfit(
            model_config,
            training_config,
            train_examples,
            tokenizer,
            device,
            args.tiny_samples,
            args.tiny_updates,
            args.tiny_lr_multiplier,
            args.required_overfit_reduction,
        )
        report = {"stage": "preflight", "passed": audit["passed"] and tiny["passed"], "dataset": audit, "tiny_overfit": tiny}
    else:
        if args.model_output is None:
            raise SystemExit("--model-output is required for --stage postflight")
        report = {"stage": "postflight", **postflight(args.model_output, args.validation, args.test, device)}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(compact(report), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
