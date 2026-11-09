"""Shared local V1 waypoint Transformer inference for calibration and ROS planning."""

import json
from pathlib import Path

class WaypointTransformerPlanner:
    """Local inference wrapper for a saved structured waypoint checkpoint."""

    def __init__(self, checkpoint: Path, device_name: str) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The transformer planner requires torch and transformers in the active Python environment."
            ) from exc

        checkpoint = checkpoint.expanduser().resolve()
        config_path = checkpoint / "config.json"
        weights_path = checkpoint / "model.pt"
        tokenizer_path = checkpoint / "tokenizer"
        for path in (config_path, weights_path, tokenizer_path):
            if not path.exists() or (path.is_file() and path.stat().st_size == 0):
                raise FileNotFoundError(f"Transformer checkpoint is missing real assets at {path}; restore weights before inference")
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config.get("model"), dict):
            raise ValueError(f"{config_path} does not contain a model configuration")

        model_config = dict(config["model"])
        shared_encoder = checkpoint.parent / "text_encoder"
        if shared_encoder.is_dir():
            encoder_weights = shared_encoder / "model.safetensors"
            if not encoder_weights.is_file() or encoder_weights.stat().st_size == 0:
                raise FileNotFoundError(f"Restore real text encoder weights at {encoder_weights}")
            model_config["text_encoder_name"] = str(shared_encoder)
        if model_config.get("architecture_version") == "transformer_2_0":
            raise ValueError("This final-testing branch supports the final vertical V1 checkpoint; use the research branch for Transformer 2.0.")
        else:
            from train_hrrt_waypoint_transformer import build_model, encode_scene

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("--transformer-device requests CUDA, but CUDA is unavailable")
        self.device = torch.device(device_name)
        self.model_config = model_config
        self.encode_scene = encode_scene
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.model = build_model(model_config).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=True)
        self.model.eval()

    @staticmethod
    def text_context(preference, obstacles):
        if obstacles:
            identities = "; ".join(
                f"obstacle {index} is {str(obstacle.get('label', 'object')).replace('_', ' ')}"
                for index, obstacle in enumerate(obstacles, start=1)
            )
        else:
            identities = "there are no obstacles"
        return f"Approved route preference: {str(preference).rstrip().rstrip('.')}. Scene-token identities: {identities}."

    def predict(self, start, goal, obstacles, workspace, preference):
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
        scene_padding_mask = torch.zeros((1, len(scene_tokens)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            normalized = self.model(input_ids, attention_mask, scene, scene_padding_mask)[0]
        lower = torch.tensor([workspace["x"][0], workspace["y"][0]], dtype=normalized.dtype, device=self.device)
        upper = torch.tensor([workspace["x"][1], workspace["y"][1]], dtype=normalized.dtype, device=self.device)
        coordinates = (lower + 0.5 * (normalized + 1.0) * (upper - lower)).detach().cpu().numpy()
        z = float(workspace["z"])
        waypoints = [{"x": float(x), "y": float(y), "z": z} for x, y in coordinates]
        waypoints[0] = {"x": float(start["x"]), "y": float(start["y"]), "z": z}
        waypoints[-1] = {"x": float(goal["x"]), "y": float(goal["y"]), "z": z}
        return waypoints



def refine_waypoints(waypoints, workspace, obstacles, clearance, fixed_z):
    """Use the same production refinement as live ROS, without starting a node."""
    try:
        from refinment import PathRefinement
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from refinment import PathRefinement
    refiner = PathRefinement.__new__(PathRefinement)
    refiner.interpolation_spacing_m = 1.0
    refiner.safety_margin_m = float(clearance)
    refiner.nudge_epsilon_m = 0.02
    refiner.fixed_z = float(fixed_z)
    return refiner.interpolate_waypoints(waypoints, workspace, obstacles)
