#!/usr/bin/env python3
"""Train Transformer 2.0 for scene-conditioned HRRT waypoint imitation.

Transformer 2.0 keeps the existing CSV/checkpoint interface and eight-point
route contract, but learns only the six interior waypoints.  Start and goal
are copied exactly from numeric scene tokens.  The backpropagated objective is
the mean Euclidean distance over the six learned interior points plus a
teacher-labelled wrong-corridor penalty.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

import train_hrrt_waypoint_transformer as base


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "fine_tuning/configs/hrrt_waypoint_transformer_2_0.yaml"
SCENE_FEATURE_SIZE = 13
LABELS = ("person", "chair", "stop_sign")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


@dataclass
class PlanningExample(base.PlanningExample):
    """Existing training example plus the selected teacher route topology."""

    teacher_h_signature: np.ndarray | None = None


WaypointDataset = base.WaypointDataset
compact_json = base.compact_json
load_yaml = base.load_yaml
move_batch = base.move_batch
require_section = base.require_section
resolve_path = base.resolve_path
resample_waypoints = base.resample_waypoints
set_seed = base.set_seed


def scene_token(token_type: int, label: str | None, bounds: Sequence[float]) -> List[float]:
    """Return [token type, object label, normalized geometry]."""
    token_one_hot = [0.0] * 4
    token_one_hot[token_type] = 1.0
    label_one_hot = [0.0] * len(LABELS)
    if label in LABEL_TO_INDEX:
        label_one_hot[LABEL_TO_INDEX[label]] = 1.0
    return token_one_hot + label_one_hot + [float(value) for value in bounds]


def encode_scene(environment: Mapping[str, Any]) -> np.ndarray:
    """Encode start, goal, workspace, geometry, and obstacle identity numerically."""
    workspace = environment["workspace"]
    x0, x1 = map(float, workspace["x"])
    y0, y1 = map(float, workspace["y"])
    sx = max(x1 - x0, 1e-6)
    sy = max(y1 - y0, 1e-6)

    def nx(value: float) -> float:
        return 2.0 * (float(value) - x0) / sx - 1.0

    def ny(value: float) -> float:
        return 2.0 * (float(value) - y0) / sy - 1.0

    start, goal = environment["start"], environment["goal"]
    tokens = [
        scene_token(0, None, [nx(start["x"]), ny(start["y"]), nx(start["x"]), ny(start["y"]), 0.0, 0.0]),
        scene_token(1, None, [nx(goal["x"]), ny(goal["y"]), nx(goal["x"]), ny(goal["y"]), 0.0, 0.0]),
        scene_token(2, None, [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0]),
    ]
    for obstacle in environment.get("obstacles", []):
        minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
        width = max(0.0, float(maximum[0]) - float(minimum[0])) / sx
        depth = max(0.0, float(maximum[1]) - float(minimum[1])) / sy
        tokens.append(
            scene_token(
                3,
                str(obstacle.get("label", "")).strip().lower(),
                [nx(minimum[0]), ny(minimum[1]), nx(maximum[0]), ny(maximum[1]), width, depth],
            )
        )
    return np.asarray(tokens, dtype=np.float32)


def obstacle_center(obstacle: Mapping[str, Any]) -> np.ndarray:
    minimum = obstacle["min_corner"]
    maximum = obstacle["max_corner"]
    return np.asarray(
        [
            (float(minimum[0]) + float(maximum[0])) / 2.0,
            (float(minimum[1]) + float(maximum[1])) / 2.0,
        ],
        dtype=np.float32,
    )


def infer_teacher_h_signature(
    row: Mapping[str, Any], environment: Mapping[str, Any], target_xy: np.ndarray
) -> np.ndarray:
    """Read the selected HRRT topology, with a geometric fallback for older CSVs."""
    obstacles = environment.get("obstacles", [])
    selected_route_id = str(row.get("selected_route_id", ""))
    try:
        candidates = json.loads(str(row.get("candidate_routes") or "[]"))
    except (json.JSONDecodeError, TypeError):
        candidates = []
    for route in candidates if isinstance(candidates, list) else []:
        if str(route.get("route_id", "")) != selected_route_id:
            continue
        signature = route.get("h_signature")
        if isinstance(signature, list) and len(signature) >= len(obstacles):
            return np.asarray(signature[: len(obstacles)], dtype=np.int8)

    # Compatibility fallback: HRRT's sign convention is the negative of the
    # path's signed lateral side in the start-to-goal frame.
    direction = target_xy[-1] - target_xy[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return np.zeros(len(obstacles), dtype=np.int8)
    unit = direction / norm
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    signature = []
    for obstacle in obstacles:
        center = obstacle_center(obstacle)
        nearest = int(np.argmin(np.abs((target_xy - center) @ unit)))
        lateral = float((target_xy[nearest] - center) @ normal)
        signature.append(-int(np.sign(lateral)))
    return np.asarray(signature, dtype=np.int8)


def read_examples(
    path: Path,
    distillation_mode: str,
    waypoint_count: int,
    limit: int | None = None,
) -> List[PlanningExample]:
    """Read the existing finalized CSV format without changing the dataset."""
    if waypoint_count != 8:
        raise ValueError("Transformer 2.0 requires an eight-point route contract")
    examples: List[PlanningExample] = []
    modes = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            modes.add(row.get("distillation_mode", "dss"))
            try:
                completion = json.loads(row["completion"])
                environment = json.loads(row["environment"])
            except (KeyError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}: malformed row {row.get('sample_id', '?')}") from exc
            waypoints = completion.get("waypoints")
            if not isinstance(waypoints, list) or len(waypoints) < 2:
                continue
            target = resample_waypoints(waypoints, waypoint_count)
            examples.append(
                PlanningExample(
                    sample_id=str(row.get("sample_id", len(examples))),
                    preference=str(row.get("preference_text") or row.get("prompt") or ""),
                    environment=environment,
                    expert_waypoints=[
                        {
                            "x": float(point["x"]),
                            "y": float(point["y"]),
                            "z": float(point.get("z", environment["workspace"]["z"])),
                        }
                        for point in waypoints
                    ],
                    target_xy=target,
                    scene_tokens=encode_scene(environment),
                    teacher_h_signature=infer_teacher_h_signature(row, environment, target),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if modes != {distillation_mode}:
        raise ValueError(f"{path} contains modes {sorted(modes)}, expected {distillation_mode!r}")
    if not examples:
        raise ValueError(f"{path} contains no rows with waypoint labels")
    return examples


class BatchCollator:
    def __init__(self, tokenizer: Any, max_text_length: int, max_scene_tokens: int):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_scene_tokens = max_scene_tokens

    def __call__(self, examples: Sequence[PlanningExample]) -> Dict[str, Any]:
        import torch

        text = self.tokenizer(
            [example.text_context for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        maximum_scene_tokens = max(len(example.scene_tokens) for example in examples)
        if maximum_scene_tokens > self.max_scene_tokens:
            raise ValueError(
                f"scene has {maximum_scene_tokens} tokens, exceeding "
                f"configured max_scene_tokens={self.max_scene_tokens}"
            )
        scene = torch.zeros(
            len(examples), maximum_scene_tokens, SCENE_FEATURE_SIZE, dtype=torch.float32
        )
        scene_mask = torch.ones(len(examples), maximum_scene_tokens, dtype=torch.bool)
        for index, example in enumerate(examples):
            count = len(example.scene_tokens)
            scene[index, :count] = torch.from_numpy(example.scene_tokens)
            scene_mask[index, :count] = False
        return {
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "scene_tokens": scene,
            "scene_padding_mask": scene_mask,
            "target_xy": torch.from_numpy(np.stack([example.target_xy for example in examples])),
            "examples": list(examples),
        }


def build_model(config: Mapping[str, Any]):
    """Build the Transformer 2.0 architecture while retaining the old call signature."""
    import torch
    from torch import nn
    from transformers import AutoModel

    class SceneResidualBlock(nn.Module):
        def __init__(self, hidden_size: int, expansion_size: int, dropout: float) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.layers = nn.Sequential(
                nn.Linear(hidden_size, expansion_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expansion_size, hidden_size),
                nn.Dropout(dropout),
            )

        def forward(self, values):
            return values + self.layers(self.norm(values))

    class PreferenceWaypointTransformer2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.waypoint_count = int(config["waypoint_count"])
            self.interior_waypoint_count = int(config["interior_waypoint_count"])
            if self.waypoint_count != self.interior_waypoint_count + 2:
                raise ValueError("waypoint_count must equal interior_waypoint_count + 2")
            d_model = int(config["hidden_size"])
            heads = int(config["attention_heads"])
            dropout = float(config["dropout"])
            scene_expansion = int(config["scene_feedforward_size"])

            self.text_encoder = AutoModel.from_pretrained(str(config["text_encoder_name"]))
            text_size = int(self.text_encoder.config.hidden_size)
            self.text_projection = nn.Sequential(
                nn.Linear(text_size, d_model), nn.GELU(), nn.LayerNorm(d_model)
            )

            self.scene_input_projection = nn.Sequential(
                nn.Linear(SCENE_FEATURE_SIZE, d_model), nn.GELU(), nn.LayerNorm(d_model)
            )
            self.scene_projection = nn.Sequential(
                *[
                    SceneResidualBlock(d_model, scene_expansion, dropout)
                    for _ in range(int(config["scene_mlp_layers"]))
                ]
            )
            self.scene_summary_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.scene_position_embedding = nn.Embedding(int(config["max_scene_tokens"]) + 1, d_model)
            scene_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=scene_expansion,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.scene_encoder = nn.TransformerEncoder(
                scene_layer,
                num_layers=int(config["scene_encoder_layers"]),
                norm=nn.LayerNorm(d_model),
            )

            self.modality_embedding = nn.Embedding(2, d_model)
            fusion_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=int(config["feedforward_size"]),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fusion_transformer = nn.TransformerEncoder(
                fusion_layer,
                num_layers=int(config["fusion_layers"]),
                norm=nn.LayerNorm(d_model),
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=int(config["feedforward_size"]),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.waypoint_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=int(config["decoder_layers"]),
                norm=nn.LayerNorm(d_model),
            )

            self.route_queries = nn.Parameter(
                torch.randn(self.interior_waypoint_count, d_model) * 0.02
            )
            progress = torch.linspace(
                1.0 / (self.waypoint_count - 1),
                (self.waypoint_count - 2) / (self.waypoint_count - 1),
                self.interior_waypoint_count,
            ).view(1, self.interior_waypoint_count, 1)
            self.register_buffer("waypoint_progress", progress, persistent=True)
            self.progress_projection = nn.Sequential(nn.Linear(1, d_model), nn.GELU())
            self.scene_to_query = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()
            )
            self.coordinate_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2), nn.Tanh()
            )

        def forward(self, input_ids, attention_mask, scene_tokens, scene_padding_mask):
            encoded_text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text = encoded_text.last_hidden_state
            if getattr(encoded_text, "pooler_output", None) is not None:
                text = text + encoded_text.pooler_output.unsqueeze(1)
            text = self.text_projection(text)

            scene = self.scene_projection(self.scene_input_projection(scene_tokens))
            summary = self.scene_summary_token.expand(scene.shape[0], -1, -1)
            scene = torch.cat((summary, scene), dim=1)
            scene_positions = torch.arange(scene.shape[1], device=scene.device)
            scene = scene + self.scene_position_embedding(scene_positions).unsqueeze(0)
            summary_mask = torch.zeros(
                scene_padding_mask.shape[0], 1, dtype=torch.bool, device=scene_padding_mask.device
            )
            encoded_scene_mask = torch.cat((summary_mask, scene_padding_mask), dim=1)
            scene = self.scene_encoder(scene, src_key_padding_mask=encoded_scene_mask)
            scene_summary = scene[:, 0]

            text = text + self.modality_embedding.weight[0].view(1, 1, -1)
            scene = scene + self.modality_embedding.weight[1].view(1, 1, -1)
            memory = torch.cat((text, scene), dim=1)
            padding_mask = torch.cat((attention_mask == 0, encoded_scene_mask), dim=1)
            memory = self.fusion_transformer(memory, src_key_padding_mask=padding_mask)

            queries = self.route_queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            queries = queries + self.progress_projection(self.waypoint_progress)
            queries = queries + self.scene_to_query(scene_summary).unsqueeze(1)
            decoded = self.waypoint_decoder(
                queries, memory, memory_key_padding_mask=padding_mask
            )
            interior = self.coordinate_head(decoded)

            # Geometry starts after 4 token-type and 3 label values.  These
            # normalized coordinates are copied, not predicted by a query.
            start = scene_tokens[:, 0, 7:9].unsqueeze(1)
            goal = scene_tokens[:, 1, 7:9].unsqueeze(1)
            return torch.cat((start, interior, goal), dim=1)

    return PreferenceWaypointTransformer2()


def denormalize_predictions(predicted_normalized, examples: Sequence[PlanningExample]):
    return base.denormalize_predictions(predicted_normalized, examples)


def euclidean_imitation_loss(predicted_xy, target_xy):
    """Mean metric distance over only the six learned interior waypoints."""
    import torch

    return torch.linalg.vector_norm(
        predicted_xy[:, 1:-1] - target_xy[:, 1:-1], ord=2, dim=-1
    ).mean()


def full_route_euclidean_loss(predicted_xy, target_xy):
    import torch

    return torch.linalg.vector_norm(predicted_xy - target_xy, ord=2, dim=-1).mean()


def corridor_side_terms(
    predicted_xy,
    target_xy,
    examples: Sequence[PlanningExample],
    margin_m: float,
    temperature_m: float,
):
    """Return one differentiable hinge term per teacher-labelled obstacle side."""
    import torch

    penalties = []
    signed_laterals = []
    temperature = max(float(temperature_m), 1e-6)
    for batch_index, example in enumerate(examples):
        signature = example.teacher_h_signature
        if signature is None:
            continue
        target = target_xy[batch_index].detach()
        direction = target[-1] - target[0]
        direction_norm = torch.linalg.vector_norm(direction).clamp_min(1e-9)
        unit = direction / direction_norm
        normal = torch.stack((-unit[1], unit[0]))
        obstacles = example.environment.get("obstacles", [])
        for obstacle_index, obstacle in enumerate(obstacles):
            if obstacle_index >= len(signature):
                break
            h_value = int(signature[obstacle_index])
            if h_value == 0:
                continue
            center = torch.as_tensor(
                obstacle_center(obstacle), device=predicted_xy.device, dtype=predicted_xy.dtype
            )
            # Fix the soft correspondence weights from the teacher route so
            # the prediction cannot reduce the penalty by changing which
            # waypoint is considered closest to the obstacle.
            teacher_longitudinal = torch.abs(torch.sum((target - center) * unit, dim=-1))
            weights = torch.softmax(-teacher_longitudinal / temperature, dim=0)
            predicted_lateral = torch.sum(
                weights * torch.sum((predicted_xy[batch_index] - center) * normal, dim=-1)
            )
            desired_side = predicted_lateral.new_tensor(float(-h_value))
            signed_lateral = desired_side * predicted_lateral
            signed_laterals.append(signed_lateral)
            penalties.append(torch.relu(float(margin_m) - signed_lateral))
    if not penalties:
        empty = predicted_xy.new_empty((0,))
        return empty, empty
    return torch.stack(penalties), torch.stack(signed_laterals)


def corridor_side_loss(
    predicted_xy,
    target_xy,
    examples: Sequence[PlanningExample],
    margin_m: float,
    temperature_m: float,
):
    penalties, _ = corridor_side_terms(
        predicted_xy, target_xy, examples, margin_m, temperature_m
    )
    if penalties.numel() == 0:
        return predicted_xy.sum() * 0.0
    return penalties.mean()


def evaluate(model, loader, device, validation_config, loss_config):
    import torch

    model.eval()
    interior_distances: List[np.ndarray] = []
    episode_interior_losses: List[np.ndarray] = []
    full_distances: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    examples: List[PlanningExample] = []
    corridor_penalties: List[np.ndarray] = []
    corridor_wrong = 0
    corridor_decisions = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            normalized = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["scene_tokens"],
                batch["scene_padding_mask"],
            )
            predicted = denormalize_predictions(normalized, batch["examples"])
            distances = torch.linalg.vector_norm(predicted - batch["target_xy"], ord=2, dim=-1)
            interior = distances[:, 1:-1]
            interior_distances.append(interior.detach().cpu().numpy().reshape(-1))
            episode_interior_losses.append(interior.mean(dim=-1).detach().cpu().numpy())
            full_distances.append(distances.detach().cpu().numpy().reshape(-1))
            penalties, signed_laterals = corridor_side_terms(
                predicted,
                batch["target_xy"],
                batch["examples"],
                float(loss_config["corridor_margin_m"]),
                float(loss_config["corridor_temperature_m"]),
            )
            if penalties.numel():
                corridor_penalties.append(penalties.detach().cpu().numpy())
                corridor_wrong += int(torch.count_nonzero(signed_laterals <= 0.0).item())
                corridor_decisions += int(signed_laterals.numel())
            predictions.extend(predicted.detach().cpu().numpy())
            examples.extend(batch["examples"])
    merged_interior = np.concatenate(interior_distances)
    merged_episodes = np.concatenate(episode_interior_losses)
    merged_full = np.concatenate(full_distances)
    mean_corridor_penalty = (
        float(np.mean(np.concatenate(corridor_penalties))) if corridor_penalties else 0.0
    )
    mean_interior_error = float(np.mean(merged_interior))
    safety = base.safety_radius_for_predictions(predictions, examples, validation_config)
    return {
        # Keep the old key so checkpoint selection and plotting remain compatible.
        "euclidean_imitation_loss_m": mean_interior_error,
        "interior_euclidean_imitation_loss_m": mean_interior_error,
        "interior_median_waypoint_error_m": float(np.median(merged_interior)),
        "interior_p80_episode_error_m": float(np.quantile(merged_episodes, 0.80)),
        "interior_p95_waypoint_error_m": float(np.quantile(merged_interior, 0.95)),
        "full_route_euclidean_imitation_loss_m": float(np.mean(merged_full)),
        "corridor_hinge_loss_m": mean_corridor_penalty,
        "wrong_corridor_rate": corridor_wrong / corridor_decisions if corridor_decisions else 0.0,
        "corridor_decision_count": corridor_decisions,
        "validation_objective_loss_m": mean_interior_error
        + float(loss_config["corridor_weight"]) * mean_corridor_penalty,
        **safety,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--safety-radius-threshold-m", type=float)
    parser.add_argument("--corridor-loss-weight", type=float)
    parser.add_argument("--corridor-margin-m", type=float)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    config = load_yaml(args.config)
    data_config = dict(require_section(config, "data"))
    model_config = dict(require_section(config, "model"))
    training_config = dict(require_section(config, "training"))
    loss_config = dict(require_section(config, "loss"))
    validation_config = dict(require_section(config, "validation"))
    if model_config.get("architecture_version") != "transformer_2_0":
        parser.error("model.architecture_version must be 'transformer_2_0'")
    if args.train:
        data_config["train_csv"] = str(args.train)
    if args.validation:
        data_config["validation_csv"] = str(args.validation)
    if args.output_dir:
        training_config["output_dir"] = str(args.output_dir)
    if args.max_epochs:
        training_config["max_epochs"] = args.max_epochs
    if args.corridor_loss_weight is not None:
        loss_config["corridor_weight"] = args.corridor_loss_weight
    if args.corridor_margin_m is not None:
        loss_config["corridor_margin_m"] = args.corridor_margin_m
    for key in ("corridor_weight", "corridor_margin_m", "corridor_temperature_m"):
        value = float(loss_config[key])
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"loss.{key} must be a positive finite value")
    if args.safety_radius_threshold_m is not None:
        if not math.isfinite(args.safety_radius_threshold_m) or args.safety_radius_threshold_m <= 0.0:
            parser.error("--safety-radius-threshold-m must be a positive finite value")
        validation_config["safety_radius_threshold_m"] = args.safety_radius_threshold_m
    if args.smoke_test:
        training_config["max_epochs"] = 1
        data_config["row_limit"] = 8
        validation_config["max_safety_samples"] = 4
    effective_config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "loss": loss_config,
        "validation": validation_config,
    }

    seed = int(training_config["seed"])
    set_seed(seed)
    train_path = resolve_path(data_config["train_csv"])
    validation_path = resolve_path(data_config["validation_csv"])
    output_dir = resolve_path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    waypoint_count = int(model_config["waypoint_count"])
    row_limit = data_config.get("row_limit")
    train_examples = read_examples(
        train_path, str(data_config["distillation_mode"]), waypoint_count, row_limit
    )
    validation_examples = read_examples(
        validation_path, str(data_config["distillation_mode"]), waypoint_count, row_limit
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_config["text_encoder_name"]), use_fast=True)
    collator = BatchCollator(
        tokenizer,
        int(model_config["max_text_length"]),
        int(model_config["max_scene_tokens"]),
    )
    train_loader = DataLoader(
        WaypointDataset(train_examples),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        WaypointDataset(validation_examples),
        batch_size=int(training_config["validation_batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device)
    model.text_encoder.save_pretrained(output_dir / "text_encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    text_parameters = list(model.text_encoder.parameters())
    text_parameter_ids = {id(parameter) for parameter in text_parameters}
    planner_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in text_parameter_ids
    ]
    optimizer = AdamW(
        [
            {"params": text_parameters, "lr": float(training_config["text_encoder_learning_rate"])},
            {"params": planner_parameters, "lr": float(training_config["planner_learning_rate"])},
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    accumulation = int(training_config["gradient_accumulation_steps"])
    max_epochs = int(training_config["max_epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, updates_per_epoch * max_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * float(training_config["warmup_ratio"]))),
        num_training_steps=total_updates,
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    history: List[Dict[str, Any]] = []
    successful_radius_checks = 0
    best_radius = math.inf
    best_key = (math.inf, math.inf, math.inf)
    best_checkpoint = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_objective_losses: List[float] = []
        train_euclidean_losses: List[float] = []
        train_corridor_losses: List[float] = []
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                normalized = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["scene_tokens"],
                    batch["scene_padding_mask"],
                )
                predicted = denormalize_predictions(normalized, batch["examples"])
                euclidean_loss = euclidean_imitation_loss(predicted, batch["target_xy"])
                corridor_loss = corridor_side_loss(
                    predicted,
                    batch["target_xy"],
                    batch["examples"],
                    float(loss_config["corridor_margin_m"]),
                    float(loss_config["corridor_temperature_m"]),
                )
                loss = euclidean_loss + float(loss_config["corridor_weight"]) * corridor_loss
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            train_objective_losses.append(float(loss.item()))
            train_euclidean_losses.append(float(euclidean_loss.item()))
            train_corridor_losses.append(float(corridor_loss.item()))
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_config["max_gradient_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        metrics = evaluate(model, validation_loader, device, validation_config, loss_config)
        metrics.update(
            {
                "epoch": epoch,
                "train_objective_loss_m": float(np.mean(train_objective_losses)),
                "train_euclidean_imitation_loss_m": float(np.mean(train_euclidean_losses)),
                "train_corridor_hinge_loss_m": float(np.mean(train_corridor_losses)),
                "learned_waypoint_count": int(model_config["interior_waypoint_count"]),
            }
        )
        history.append(metrics)
        checkpoint = base.save_checkpoint(
            model, tokenizer, output_dir, epoch, metrics, effective_config
        )
        radius = float(metrics["safety_radius_m"])
        validation_loss = float(metrics["validation_objective_loss_m"])
        checkpoint_key = (
            (0.0, radius, validation_loss)
            if math.isfinite(radius)
            else (1.0, validation_loss, 0.0)
        )
        if best_checkpoint is None or checkpoint_key < best_key:
            best_key = checkpoint_key
            best_radius = radius
            best_checkpoint = checkpoint
        if math.isfinite(radius) and radius <= float(validation_config["safety_radius_threshold_m"]):
            successful_radius_checks += 1
        else:
            successful_radius_checks = 0
        print(compact_json(metrics), flush=True)
        if successful_radius_checks >= int(validation_config["consecutive_checks"]):
            print(f"Safety-radius stopping condition met at epoch {epoch}: {radius:.6f} m", flush=True)
            break

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    summary = {
        "architecture_version": "transformer_2_0",
        "stopping_condition_met": successful_radius_checks
        >= int(validation_config["consecutive_checks"]),
        "best_safety_radius_m": best_radius,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "completed_epochs": len(history),
        "safety_radius_threshold_m": float(validation_config["safety_radius_threshold_m"]),
        "learned_waypoint_count": int(model_config["interior_waypoint_count"]),
        "route_waypoint_count": waypoint_count,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not summary["stopping_condition_met"]:
        print(
            f"Maximum epochs reached; safety target was not met. Best radius={best_radius:.6f} m",
            flush=True,
        )


if __name__ == "__main__":
    main()
