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
    assert encoded.shape == (3, transformer_2.SCENE_FEATURE_SIZE)
    obstacle = encoded[2]
    assert obstacle[:3].tolist() == [0.0, 0.0, 1.0]
    assert obstacle[3:6].tolist() == [0.0, 1.0, 0.0]


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
    assert torch.equal(output[:, 0], encoded[:, 0, 6:8])
    assert torch.equal(output[:, -1], encoded[:, 1, 6:8])

    target = output.detach().clone()
    target[:, 1:-1] += 0.2
    loss = transformer_2.euclidean_imitation_loss(output, target)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.scene_encoder.parameters()
    )
    assert model.route_queries.grad is not None


def test_wrong_corridor_hinge_is_strong_and_differentiable():
    import numpy as np
    import torch

    scene = environment()
    scene["start"] = {"x": -2.0, "y": 0.0, "z": -0.5}
    scene["goal"] = {"x": 2.0, "y": 0.0, "z": -0.5}
    target = torch.tensor(
        [[[-2.0, 0.0], [-1.4, 0.6], [-0.8, 1.0], [-0.2, 1.0],
          [0.2, 1.0], [0.8, 1.0], [1.4, 0.6], [2.0, 0.0]]]
    )
    wrong = target.clone()
    wrong[:, 1:-1, 1] *= -1.0
    wrong.requires_grad_(True)
    example = transformer_2.PlanningExample(
        sample_id="corridor-test",
        preference="pass above the chair",
        environment=scene,
        expert_waypoints=[],
        target_xy=target[0].numpy(),
        scene_tokens=transformer_2.encode_scene(scene),
        teacher_h_signature=np.asarray([-1], dtype=np.int8),
        teacher_clearance_bins=np.asarray([2], dtype=np.int8),
    )

    wrong_loss = transformer_2.corridor_side_loss(wrong, target, [example], 0.20, 0.35)
    right_loss = transformer_2.corridor_side_loss(target, target, [example], 0.20, 0.35)
    assert wrong_loss.item() > 1.0
    assert right_loss.item() == 0.0
    wrong_loss.backward()
    assert wrong.grad is not None
    assert torch.sum(wrong.grad[0, 1:-1, 1]).item() < 0.0


def test_clearance_bin_hinge_matches_teacher_bin_and_backpropagates():
    import numpy as np
    import torch

    scene = environment()
    # The chair occupies y=[-0.2, 0.2]. A route at y=1.0 has 0.8 m
    # clearance, the lower edge of the chair's wide bin.
    wide_route = torch.tensor(
        [[[-2.0, 1.0], [-1.4, 1.0], [-0.8, 1.0], [-0.2, 1.0],
          [0.2, 1.0], [0.8, 1.0], [1.4, 1.0], [2.0, 1.0]]]
    )
    close_route = wide_route.clone()
    close_route[:, :, 1] = 0.5
    close_route.requires_grad_(True)
    example = transformer_2.PlanningExample(
        sample_id="bin-test",
        preference="take a wide route around the chair",
        environment=scene,
        expert_waypoints=[],
        target_xy=wide_route[0].numpy(),
        scene_tokens=transformer_2.encode_scene(scene),
        teacher_h_signature=np.asarray([-1], dtype=np.int8),
        teacher_clearance_bins=np.asarray([2], dtype=np.int8),
    )

    correct_loss = transformer_2.clearance_bin_loss(
        wide_route, wide_route, [example], 0.40, 8
    )
    close_loss = transformer_2.clearance_bin_loss(
        close_route, wide_route, [example], 0.40, 8
    )
    assert correct_loss.item() < 1e-6
    assert abs(close_loss.item() - 0.5) < 1e-6
    close_loss.backward()
    assert close_route.grad is not None
    assert torch.sum(close_route.grad[0, :, 1]).item() < 0.0


def test_squared_tail_loss_penalizes_large_waypoint_errors_more_strongly():
    import torch

    target = torch.zeros((1, 8, 2), dtype=torch.float32)
    predicted = target.clone()
    predicted[:, 1:-1, 0] = 0.1
    predicted[:, 3, 0] = 2.0
    predicted.requires_grad_(True)

    loss = transformer_2.squared_tail_loss(predicted, target, scale_m=1.0)
    expected = (5 * 0.1**2 + 2.0**2) / 6
    assert abs(loss.item() - expected) < 1e-6
    loss.backward()
    assert predicted.grad is not None
    assert predicted.grad[0, 3, 0].abs() > 10 * predicted.grad[0, 2, 0].abs()
