#!/usr/bin/env python3
"""Shared runtime for the final vertical V1 waypoint Transformer.

Live ROS planning, batch calibration scoring, and the offline inspection
gateway import this module so they cannot silently use different model loaders
or input encodings.
"""

from __future__ import annotations

import json
from pathlib import Path


FINAL_CHECKPOINT_RELATIVE = Path(
    "fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10/"
    "final_transformer_vertical_teacher_only_80_10_10_best/checkpoint"
)
DEFAULT_ROUTE_PREFERENCE = "Take the quickest collision-free way to the destination."


def resolve_checkpoint(value: str | Path) -> Path:
    """Resolve an explicit/source/install-relative checkpoint directory."""
    requested = Path(value).expanduser()
    candidates = [requested, Path.cwd() / requested]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("llm_vision_planner")) / requested)
    except Exception:
        pass
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Transformer checkpoint directory was not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


class WaypointTransformerPlanner:
    """Load one saved V1 checkpoint and predict sparse NED waypoints."""

    def __init__(self, checkpoint: str | Path, device_name: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer planning requires torch and transformers in the active environment."
            ) from exc

        checkpoint = resolve_checkpoint(checkpoint)
        config_path = checkpoint / "config.json"
        weights_path = checkpoint / "model.pt"
        tokenizer_path = checkpoint / "tokenizer"
        shared_encoder = checkpoint.parent / "text_encoder"
        for path in (config_path, weights_path, tokenizer_path, shared_encoder):
            if not path.exists():
                raise FileNotFoundError(f"Transformer checkpoint is missing {path}")

        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config.get("model"), dict):
            raise ValueError(f"{config_path} does not contain a model configuration")
        model_config = dict(config["model"])
        if model_config.get("architecture_version") == "transformer_2_0":
            raise ValueError("main supports the selected final vertical V1 checkpoint only")
        model_config["text_encoder_name"] = str(shared_encoder)

        from train_hrrt_waypoint_transformer import build_model, encode_scene

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the Transformer, but CUDA is unavailable")

        self.checkpoint = checkpoint
        self.device = torch.device(device_name)
        self.model_config = model_config
        self.encode_scene = encode_scene
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.model = build_model(model_config).to(self.device)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device), strict=True
        )
        self.model.eval()

    @staticmethod
    def text_context(preference, obstacles):
        preference = str(preference or DEFAULT_ROUTE_PREFERENCE).rstrip().rstrip(".")
        if obstacles:
            identities = "; ".join(
                f"obstacle {index} is {str(obstacle.get('label', 'object')).replace('_', ' ')}"
                for index, obstacle in enumerate(obstacles, start=1)
            )
        else:
            identities = "there are no obstacles"
        return f"Approved route preference: {preference}. Scene-token identities: {identities}."

    def predict(self, start, goal, obstacles, workspace, preference=DEFAULT_ROUTE_PREFERENCE):
        import torch

        environment = {
            "start": dict(start),
            "goal": dict(goal),
            "workspace": dict(workspace),
            "obstacles": list(obstacles),
        }
        scene_tokens = self.encode_scene(environment)
        if len(scene_tokens) > int(self.model_config["max_scene_tokens"]):
            raise ValueError(
                f"scene has {len(scene_tokens)} tokens, exceeding checkpoint limit "
                f"{self.model_config['max_scene_tokens']}"
            )
        text = self.tokenizer(
            [self.text_context(preference, obstacles)],
            padding=True,
            truncation=True,
            max_length=int(self.model_config["max_text_length"]),
            return_tensors="pt",
        )
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)
        scene = torch.from_numpy(scene_tokens).unsqueeze(0).to(self.device)
        scene_padding_mask = torch.zeros(
            (1, len(scene_tokens)), dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            normalized = self.model(
                input_ids, attention_mask, scene, scene_padding_mask
            )[0]
        lower = torch.tensor(
            [workspace["x"][0], workspace["y"][0]],
            dtype=normalized.dtype,
            device=self.device,
        )
        upper = torch.tensor(
            [workspace["x"][1], workspace["y"][1]],
            dtype=normalized.dtype,
            device=self.device,
        )
        coordinates = (
            lower + 0.5 * (normalized + 1.0) * (upper - lower)
        ).detach().cpu().numpy()
        z = float(workspace["z"])
        waypoints = [{"x": float(x), "y": float(y), "z": z} for x, y in coordinates]
        waypoints[0] = {"x": float(start["x"]), "y": float(start["y"]), "z": z}
        waypoints[-1] = {"x": float(goal["x"]), "y": float(goal["y"]), "z": z}
        return waypoints
