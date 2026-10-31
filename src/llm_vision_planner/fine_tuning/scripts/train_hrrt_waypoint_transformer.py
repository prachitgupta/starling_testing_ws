#!/usr/bin/env python3
"""Train a preference-conditioned transformer to imitate HRRT waypoints.

Only mean Euclidean waypoint distance is differentiated.  The offline
refinement, verification, minimum-control QP, and contraction-tube radius are
evaluated after each epoch and are used only as the stopping certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "fine_tuning/configs/hrrt_waypoint_transformer.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: python3 -m pip install pyyaml") from exc
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def require_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration requires a {name!r} mapping")
    return section


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def resample_waypoints(waypoints: Sequence[Mapping[str, Any]], count: int) -> np.ndarray:
    """Arc-length resample a sparse path to exactly ``count`` x-y points."""
    if count < 2 or len(waypoints) < 2:
        raise ValueError("waypoint resampling requires at least two input and output points")
    points = np.asarray([[float(p["x"]), float(p["y"])] for p in waypoints], dtype=np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    samples = np.linspace(0.0, float(cumulative[-1]), count)
    result = np.empty((count, 2), dtype=np.float32)
    for axis in range(2):
        result[:, axis] = np.interp(samples, cumulative, points[:, axis])
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def scene_token(token_type: int, bounds: Sequence[float]) -> List[float]:
    one_hot = [0.0] * 4
    one_hot[token_type] = 1.0
    return one_hot + [float(value) for value in bounds]


def encode_scene(environment: Mapping[str, Any]) -> np.ndarray:
    """Build start, goal, workspace, and variable obstacle tokens."""
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
        scene_token(0, [nx(start["x"]), ny(start["y"]), nx(start["x"]), ny(start["y"]), 0.0, 0.0]),
        scene_token(1, [nx(goal["x"]), ny(goal["y"]), nx(goal["x"]), ny(goal["y"]), 0.0, 0.0]),
        scene_token(2, [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0]),
    ]
    for obstacle in environment.get("obstacles", []):
        minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
        width = max(0.0, float(maximum[0]) - float(minimum[0])) / sx
        depth = max(0.0, float(maximum[1]) - float(minimum[1])) / sy
        tokens.append(
            scene_token(
                3,
                [nx(minimum[0]), ny(minimum[1]), nx(maximum[0]), ny(maximum[1]), width, depth],
            )
        )
    return np.asarray(tokens, dtype=np.float32)


@dataclass
class PlanningExample:
    sample_id: str
    preference: str
    environment: Dict[str, Any]
    expert_waypoints: List[Dict[str, float]]
    target_xy: np.ndarray
    scene_tokens: np.ndarray

    @property
    def text_context(self) -> str:
        obstacles = self.environment.get("obstacles", [])
        if obstacles:
            identities = "; ".join(
                f"obstacle {index} is {str(obstacle.get('label', 'object')).replace('_', ' ')}"
                for index, obstacle in enumerate(obstacles, start=1)
            )
        else:
            identities = "there are no obstacles"
        preference = self.preference.rstrip().rstrip(".")
        return f"Approved route preference: {preference}. Scene-token identities: {identities}."


def read_examples(
    path: Path,
    distillation_mode: str,
    waypoint_count: int,
    limit: int | None = None,
) -> List[PlanningExample]:
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
            # DSS auxiliary reasoning rows intentionally have no waypoint label.
            if not isinstance(waypoints, list) or len(waypoints) < 2:
                continue
            target = resample_waypoints(waypoints, waypoint_count)
            examples.append(
                PlanningExample(
                    sample_id=str(row.get("sample_id", len(examples))),
                    preference=str(row.get("preference_text") or row.get("prompt") or ""),
                    environment=environment,
                    expert_waypoints=[
                        {"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", environment["workspace"]["z"]))}
                        for p in waypoints
                    ],
                    target_xy=target,
                    scene_tokens=encode_scene(environment),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if modes != {distillation_mode}:
        raise ValueError(f"{path} contains modes {sorted(modes)}, expected {distillation_mode!r}")
    if not examples:
        raise ValueError(f"{path} contains no rows with waypoint labels")
    return examples


class WaypointDataset:
    def __init__(self, examples: Sequence[PlanningExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PlanningExample:
        return self.examples[index]


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
                f"scene has {maximum_scene_tokens} tokens, exceeding configured max_scene_tokens={self.max_scene_tokens}"
            )
        scene = torch.zeros(len(examples), maximum_scene_tokens, 10, dtype=torch.float32)
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
    import torch
    from torch import nn
    from transformers import AutoModel

    class PreferenceWaypointTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.waypoint_count = int(config["waypoint_count"])
            self.fixed_endpoints = bool(config.get("fixed_endpoints", False))
            query_count = self.waypoint_count - 2 if self.fixed_endpoints else self.waypoint_count
            if self.fixed_endpoints:
                self.interior_waypoint_count = query_count
            d_model = int(config["hidden_size"])
            heads = int(config["attention_heads"])
            dropout = float(config["dropout"])
            self.text_encoder = AutoModel.from_pretrained(str(config["text_encoder_name"]))
            text_size = int(self.text_encoder.config.hidden_size)
            self.text_projection = nn.Linear(text_size, d_model)
            self.scene_projection = nn.Sequential(
                nn.Linear(10, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
            )
            self.scene_position_embedding = nn.Embedding(int(config["max_scene_tokens"]), d_model)
            self.modality_embedding = nn.Embedding(2, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=int(config["feedforward_size"]),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fusion_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=int(config["fusion_layers"]), norm=nn.LayerNorm(d_model)
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
                decoder_layer, num_layers=int(config["decoder_layers"]), norm=nn.LayerNorm(d_model)
            )
            self.route_queries = nn.Parameter(torch.randn(query_count, d_model) * 0.02)
            self.coordinate_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2), nn.Tanh()
            )

        def forward(self, input_ids, attention_mask, scene_tokens, scene_padding_mask):
            encoded_text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text = encoded_text.last_hidden_state
            # MiniLM exposes a trainable pooler.  Feed it into fusion so every
            # pretrained encoder tensor participates in full fine-tuning.
            if getattr(encoded_text, "pooler_output", None) is not None:
                text = text + encoded_text.pooler_output.unsqueeze(1)
            text = self.text_projection(text)
            scene = self.scene_projection(scene_tokens)
            scene_positions = torch.arange(scene.shape[1], device=scene.device)
            scene = scene + self.scene_position_embedding(scene_positions).unsqueeze(0)
            text = text + self.modality_embedding.weight[0].view(1, 1, -1)
            scene = scene + self.modality_embedding.weight[1].view(1, 1, -1)
            memory = torch.cat((text, scene), dim=1)
            padding_mask = torch.cat((attention_mask == 0, scene_padding_mask), dim=1)
            memory = self.fusion_transformer(memory, src_key_padding_mask=padding_mask)
            queries = self.route_queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            decoded = self.waypoint_decoder(queries, memory, memory_key_padding_mask=padding_mask)
            predicted = self.coordinate_head(decoded)
            if self.fixed_endpoints:
                # V1 start/goal tokens store normalized x,y at columns 4:6.
                predicted = torch.cat((scene_tokens[:, 0:1, 4:6], predicted,
                                       scene_tokens[:, 1:2, 4:6]), dim=1)
            return predicted

    return PreferenceWaypointTransformer()


def denormalize_predictions(
    predicted_normalized,
    examples: Sequence[PlanningExample],
    enforce_endpoints: bool = False,
):
    import torch

    lower = torch.tensor(
        [[e.environment["workspace"]["x"][0], e.environment["workspace"]["y"][0]] for e in examples],
        dtype=predicted_normalized.dtype,
        device=predicted_normalized.device,
    ).unsqueeze(1)
    upper = torch.tensor(
        [[e.environment["workspace"]["x"][1], e.environment["workspace"]["y"][1]] for e in examples],
        dtype=predicted_normalized.dtype,
        device=predicted_normalized.device,
    ).unsqueeze(1)
    predicted = lower + 0.5 * (predicted_normalized + 1.0) * (upper - lower)
    if enforce_endpoints:
        starts = torch.tensor(
            [[e.environment["start"]["x"], e.environment["start"]["y"]] for e in examples],
            dtype=predicted.dtype,
            device=predicted.device,
        )
        goals = torch.tensor(
            [[e.environment["goal"]["x"], e.environment["goal"]["y"]] for e in examples],
            dtype=predicted.dtype,
            device=predicted.device,
        )
        predicted = torch.cat((starts[:, None], predicted[:, 1:-1], goals[:, None]), dim=1)
    return predicted


def euclidean_imitation_loss(predicted_xy, target_xy):
    """Mean metric waypoint distance; this is the sole backpropagated objective."""
    import torch

    return torch.linalg.vector_norm(predicted_xy - target_xy, ord=2, dim=-1).mean()


def move_batch(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def predicted_waypoint_dicts(predicted_xy: np.ndarray, environment: Mapping[str, Any]):
    z = float(environment["workspace"]["z"])
    result = [{"x": float(x), "y": float(y), "z": z} for x, y in predicted_xy]
    result[0] = {"x": float(environment["start"]["x"]), "y": float(environment["start"]["y"]), "z": z}
    result[-1] = {"x": float(environment["goal"]["x"]), "y": float(environment["goal"]["y"]), "z": z}
    return result


def point_clearance(point: Mapping[str, float], obstacle: Mapping[str, Any]) -> float:
    minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
    dx = max(float(minimum[0]) - point["x"], 0.0, point["x"] - float(maximum[0]))
    dy = max(float(minimum[1]) - point["y"], 0.0, point["y"] - float(maximum[1]))
    return math.hypot(dx, dy)


def refine_waypoints_offline(
    waypoints: Sequence[Mapping[str, float]],
    environment: Mapping[str, Any],
    spacing_m: float,
    clearance_m: float,
    nudge_epsilon_m: float,
) -> List[Dict[str, float]]:
    """Offline equivalent of scripts/refinment.py for non-ROS compute nodes."""
    workspace, obstacles = environment["workspace"], environment.get("obstacles", [])
    z = float(workspace["z"])

    def sanitize(point: Dict[str, float], preserve_goal: bool) -> Dict[str, float]:
        point = {"x": float(point["x"]), "y": float(point["y"]), "z": z}
        if not preserve_goal:
            for obstacle in obstacles:
                minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
                limits = (
                    float(minimum[0]) - clearance_m,
                    float(maximum[0]) + clearance_m,
                    float(minimum[1]) - clearance_m,
                    float(maximum[1]) + clearance_m,
                )
                if limits[0] <= point["x"] <= limits[1] and limits[2] <= point["y"] <= limits[3]:
                    gaps = [point["x"] - limits[0], limits[1] - point["x"], point["y"] - limits[2], limits[3] - point["y"]]
                    side = int(np.argmin(gaps))
                    if side == 0:
                        point["x"] = limits[0] - nudge_epsilon_m
                    elif side == 1:
                        point["x"] = limits[1] + nudge_epsilon_m
                    elif side == 2:
                        point["y"] = limits[2] - nudge_epsilon_m
                    else:
                        point["y"] = limits[3] + nudge_epsilon_m
        point["x"] = min(max(point["x"], float(workspace["x"][0])), float(workspace["x"][1]))
        point["y"] = min(max(point["y"], float(workspace["y"][0])), float(workspace["y"][1]))
        return point

    refined = [sanitize(dict(waypoints[0]), False)]
    for start, end in zip(waypoints, waypoints[1:]):
        dx, dy = float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"])
        steps = max(1, int(math.ceil(math.hypot(dx, dy) / spacing_m)))
        for step in range(1, steps + 1):
            amount = step / steps
            candidate = sanitize(
                {"x": float(start["x"]) + amount * dx, "y": float(start["y"]) + amount * dy, "z": z},
                step == steps,
            )
            if math.hypot(candidate["x"] - refined[-1]["x"], candidate["y"] - refined[-1]["y"]) > 1e-9:
                refined.append(candidate)
    return refined


def verify_waypoints_offline(
    waypoints: Sequence[Mapping[str, float]],
    environment: Mapping[str, Any],
    validation_config: Mapping[str, Any],
) -> bool:
    """Apply the production verifier's geometric and kinematic checks offline."""
    workspace = environment["workspace"]
    start, goal = environment["start"], environment["goal"]
    clearance_m = float(validation_config["clearance_m"])
    if len(waypoints) < 2:
        return False
    if math.dist((waypoints[0]["x"], waypoints[0]["y"]), (start["x"], start["y"])) > 0.10:
        return False
    if math.dist((waypoints[-1]["x"], waypoints[-1]["y"]), (goal["x"], goal["y"])) > 0.05:
        return False
    for point in waypoints:
        if not (
            float(workspace["x"][0]) <= point["x"] <= float(workspace["x"][1])
            and float(workspace["y"][0]) <= point["y"] <= float(workspace["y"][1])
        ):
            return False
        if environment.get("obstacles") and min(point_clearance(point, o) for o in environment["obstacles"]) < clearance_m - 1e-9:
            return False
    progress_tolerance = float(validation_config["progress_tolerance_m"])
    distances = [math.hypot(float(goal["x"]) - p["x"], float(goal["y"]) - p["y"]) for p in waypoints]
    if any(second > first + progress_tolerance for first, second in zip(distances, distances[1:])):
        return False
    nominal_dt = float(validation_config["refinement_spacing_m"]) / float(validation_config["cruise_speed_mps"])
    velocities = [
        ((second["x"] - first["x"]) / nominal_dt, (second["y"] - first["y"]) / nominal_dt)
        for first, second in zip(waypoints, waypoints[1:])
    ]
    max_speed = max((math.hypot(*velocity) for velocity in velocities), default=0.0)
    accelerations = [
        ((second[0] - first[0]) / nominal_dt, (second[1] - first[1]) / nominal_dt)
        for first, second in zip(velocities, velocities[1:])
    ]
    max_acceleration = max((math.hypot(*acceleration) for acceleration in accelerations), default=0.0)
    if max_speed > float(validation_config["max_velocity_mps"]) + 1e-9:
        return False
    if max_acceleration > float(validation_config["max_acceleration_mps2"]) + 1e-9:
        return False
    return True


def conformal_quantile(values: Sequence[float], delta: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return math.inf
    rank = int(math.ceil((len(values) + 1) * (1.0 - delta)))
    if rank > len(values):
        return math.inf
    return values[max(0, rank - 1)]


def projected_radius_from_scores(scores: Sequence[float], delta: float) -> tuple[float, float]:
    """Convert the calibrated single residual score quantile to a 2-D tube radius."""
    if SCRIPT_DIR.as_posix() not in sys.path:
        sys.path.insert(0, SCRIPT_DIR.as_posix())
    from lqr import certified_metric_alpha
    from postprocess_residual_calibration import ideal_metric

    b_matrix, p_matrix, k_matrix = ideal_metric()
    a_matrix = np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=float,
    )
    alpha = certified_metric_alpha(a_matrix, b_matrix, k_matrix, p_matrix)
    position_output = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    projection_gain = math.sqrt(
        float(np.max(np.linalg.eigvalsh(position_output @ np.linalg.inv(p_matrix) @ position_output.T)))
    )
    quantile = conformal_quantile(scores, delta)
    return quantile, projection_gain * quantile / alpha


def safety_radius_for_predictions(
    predictions: Sequence[np.ndarray],
    examples: Sequence[PlanningExample],
    validation_config: Mapping[str, Any],
) -> Dict[str, float]:
    if SCRIPT_DIR.as_posix() not in sys.path:
        sys.path.insert(0, SCRIPT_DIR.as_posix())
    from min_control_qp import generate_shared_pair
    from postprocess_residual_calibration import residual_score

    scores: List[float] = []
    passed = 0
    spacing = float(validation_config["refinement_spacing_m"])
    clearance = float(validation_config["clearance_m"])
    epsilon = float(validation_config["nudge_epsilon_m"])
    maximum_samples = validation_config.get("max_safety_samples")
    pairs = list(zip(predictions, examples))
    if maximum_samples is not None:
        pairs = pairs[: int(maximum_samples)]
    for predicted_xy, example in pairs:
        predicted = predicted_waypoint_dicts(predicted_xy, example.environment)
        expert = refine_waypoints_offline(example.expert_waypoints, example.environment, spacing, clearance, epsilon)
        predicted = refine_waypoints_offline(predicted, example.environment, spacing, clearance, epsilon)
        if not verify_waypoints_offline(predicted, example.environment, validation_config):
            continue
        try:
            expert_qp, predicted_qp = generate_shared_pair(
                expert,
                predicted,
                workspace=example.environment["workspace"],
                obstacles=example.environment.get("obstacles", []),
                dt=float(validation_config["qp_dt"]),
                damping=0.0,
            )
            maximum, _, _ = residual_score(expert_qp["samples"], predicted_qp["samples"])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        scores.append(float(maximum[0]))
        passed += 1
    total = len(pairs)
    if passed != total:
        return {
            "conformity_quantile": math.inf,
            "safety_radius_m": math.inf,
            "verification_pass_rate": passed / max(total, 1),
        }
    quantile, radius = projected_radius_from_scores(scores, float(validation_config["conformal_delta"]))
    return {
        "conformity_quantile": quantile,
        "safety_radius_m": radius,
        "verification_pass_rate": passed / max(total, 1),
    }


def evaluate(model, loader, device, validation_config):
    import torch

    model.eval()
    losses: List[float] = []
    predictions: List[np.ndarray] = []
    examples: List[PlanningExample] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            normalized = model(
                batch["input_ids"], batch["attention_mask"], batch["scene_tokens"], batch["scene_padding_mask"]
            )
            predicted = denormalize_predictions(normalized, batch["examples"])
            loss = euclidean_imitation_loss(predicted, batch["target_xy"])
            losses.append(float(loss.item()))
            predictions.extend(predicted.detach().cpu().numpy())
            examples.extend(batch["examples"])
    safety = safety_radius_for_predictions(predictions, examples, validation_config)
    return {"euclidean_imitation_loss_m": float(np.mean(losses)), **safety}


def save_checkpoint(model, tokenizer, output_dir: Path, epoch: int, metrics: Mapping[str, Any], config):
    import torch

    checkpoint = output_dir / f"checkpoint-epoch-{epoch:03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint / "model.pt")
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    with (checkpoint / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(metrics), stream, indent=2)
    with (checkpoint / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--safety-radius-threshold-m",
        type=float,
        help="Override the YAML safety-radius stopping threshold in metres.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Load only a few rows and run one epoch.")
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    config = load_yaml(args.config)
    data_config = dict(require_section(config, "data"))
    model_config = dict(require_section(config, "model"))
    training_config = dict(require_section(config, "training"))
    validation_config = dict(require_section(config, "validation"))
    if args.train:
        data_config["train_csv"] = str(args.train)
    if args.validation:
        data_config["validation_csv"] = str(args.validation)
    if args.output_dir:
        training_config["output_dir"] = str(args.output_dir)
    if args.max_epochs:
        training_config["max_epochs"] = args.max_epochs
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
    train_examples = read_examples(train_path, str(data_config["distillation_mode"]), waypoint_count, row_limit)
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
    # Keep one local base-encoder copy beside all epoch checkpoints.  The
    # checkpoint model.pt still contains the fine-tuned weights; this folder
    # only lets GPU-server inference rebuild the architecture without an
    # additional Hugging Face download.
    model.text_encoder.save_pretrained(output_dir / "text_encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    text_parameters = list(model.text_encoder.parameters())
    text_parameter_ids = {id(parameter) for parameter in text_parameters}
    planner_parameters = [parameter for parameter in model.parameters() if id(parameter) not in text_parameter_ids]
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("all model weights must remain trainable")
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
    best_checkpoint = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                normalized = model(
                    batch["input_ids"], batch["attention_mask"], batch["scene_tokens"], batch["scene_padding_mask"]
                )
                predicted = denormalize_predictions(normalized, batch["examples"])
                loss = euclidean_imitation_loss(predicted, batch["target_xy"])
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            train_losses.append(float(loss.item()))
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["max_gradient_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        metrics = evaluate(model, validation_loader, device, validation_config)
        metrics.update({"epoch": epoch, "train_euclidean_imitation_loss_m": float(np.mean(train_losses))})
        history.append(metrics)
        checkpoint = save_checkpoint(model, tokenizer, output_dir, epoch, metrics, effective_config)
        radius = float(metrics["safety_radius_m"])
        if best_checkpoint is None or radius < best_radius:
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

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    summary = {
        "stopping_condition_met": successful_radius_checks >= int(validation_config["consecutive_checks"]),
        "best_safety_radius_m": best_radius,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "completed_epochs": len(history),
        "safety_radius_threshold_m": float(validation_config["safety_radius_threshold_m"]),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if not summary["stopping_condition_met"]:
        print(
            f"Maximum epochs reached; safety target was not met. Best radius={best_radius:.6f} m",
            flush=True,
        )


if __name__ == "__main__":
    main()
