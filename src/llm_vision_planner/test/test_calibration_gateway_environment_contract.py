from types import SimpleNamespace

from fine_tuning.scripts.calibration_verifier_gateway import (
    requested_mode,
    transformer_environment_pairs,
)
from scripts.interactive_mission_gateway import InteractiveMissionGateway


def test_strict_contract_forces_transformer_to_perceived_environment():
    ground_truth = [{"id": "ground-truth"}]
    perceived = [{"id": "perceived"}]
    args = SimpleNamespace(strict_e_vs_ehat=True)

    assert requested_mode(args, "planner_only") == "env_uncertainty"
    assert transformer_environment_pairs(args, ground_truth, perceived) == (
        ("env_uncertainty", perceived),
    )


def test_legacy_contract_retains_both_modes():
    ground_truth = [{"id": "ground-truth"}]
    perceived = [{"id": "perceived"}]
    args = SimpleNamespace(strict_e_vs_ehat=False)

    assert requested_mode(args, "planner_only") == "planner_only"
    assert transformer_environment_pairs(args, ground_truth, perceived) == (
        ("env_uncertainty", perceived),
        ("planner_only", ground_truth),
    )


def test_live_planner_only_mode_uses_vicon_boxes_without_inflation():
    gateway = InteractiveMissionGateway.__new__(InteractiveMissionGateway)
    gateway.calibration_mode = "planner_only"
    gateway.environment = "real"
    ground_truth = [
        {
            "object_id": "chair-1",
            "label": "chair",
            "min_corner": [-1.0, -0.5, -1.0],
            "max_corner": [0.0, 0.5, 0.0],
        }
    ]
    gateway.latest_vicon_ground_truth = {"obstacles": ground_truth}
    nominal, planning = gateway.environment_obstacles(
        {"obstacles": [{"object_id": "perceived"}]},
        SimpleNamespace(),
    )
    assert nominal == ground_truth
    assert planning == ground_truth
    assert nominal is not ground_truth and planning is not ground_truth
