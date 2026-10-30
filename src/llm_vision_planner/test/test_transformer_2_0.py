#!/usr/bin/env python3
"""Fast architecture-contract tests for Transformer 2.0."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "fine_tuning" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_hrrt_waypoint_transformer_2_0 as transformer_2  # noqa: E402


def environment():
    return {
        "start": {"x": -2.0, "y": -1.0, "z": -0.5},
        "goal": {"x": 2.0, "y": 1.0, "z": -0.5},
        "workspace": {"x": [-3.0, 3.0], "y": [-2.0, 2.0], "z": -0.5},
        "obstacles": [
            {
                "object_id": "chair-1",
                "label": "chair",
                "min_corner": [-0.3, -0.2, -1.0],
                "max_corner": [0.3, 0.2, 0.0],
            }
        ],
    }


def tiny_config():
    return {
        "architecture_version": "transformer_2_0",
        "text_encoder_name": "dummy",
        "max_scene_tokens": 8,
        "waypoint_count": 8,
        "interior_waypoint_count": 6,
        "hidden_size": 32,
        "attention_heads": 4,
        "scene_mlp_layers": 1,
        "scene_encoder_layers": 1,
        "scene_feedforward_size": 64,
        "feedforward_size": 64,
        "fusion_layers": 1,
        "decoder_layers": 1,
        "dropout": 0.0,
    }


def test_scene_tokens_include_numeric_obstacle_label():
    encoded = transformer_2.encode_scene(environment())
    assert encoded.shape == (4, transformer_2.SCENE_FEATURE_SIZE)
    obstacle = encoded[3]
    assert obstacle[:4].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert obstacle[4:7].tolist() == [0.0, 1.0, 0.0]


def test_model_learns_six_points_and_copies_endpoints():
    import torch
    from torch import nn

    class DummyTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16)
            self.embedding = nn.Embedding(32, 16)

        def forward(self, input_ids, attention_mask):
            values = self.embedding(input_ids)
            return SimpleNamespace(last_hidden_state=values, pooler_output=values[:, 0])

    encoded = torch.from_numpy(transformer_2.encode_scene(environment())).unsqueeze(0)
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyTextEncoder()):
        model = transformer_2.build_model(tiny_config())
    output = model(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.ones((1, 3), dtype=torch.long),
        encoded,
        torch.zeros((1, encoded.shape[1]), dtype=torch.bool),
    )
    assert tuple(output.shape) == (1, 8, 2)
    assert model.route_queries.shape[0] == 6
    assert torch.equal(output[:, 0], encoded[:, 0, 7:9])
    assert torch.equal(output[:, -1], encoded[:, 1, 7:9])

    target = output.detach().clone()
    target[:, 1:-1] += 0.2
    loss = transformer_2.euclidean_imitation_loss(output, target)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.scene_encoder.parameters()
    )
    assert model.route_queries.grad is not None
