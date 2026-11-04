from types import SimpleNamespace

from fine_tuning.scripts.calibration_verifier_gateway import (
    requested_mode,
    transformer_environment_pairs,
)
from fine_tuning.verification.scripts.verify_transformer_components import best_checkpoint


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


def test_euclidean_checkpoint_selection_is_independent_of_radius(tmp_path):
    import json

    first = tmp_path / "checkpoint-epoch-001"
    second = tmp_path / "checkpoint-epoch-002"
    first.mkdir()
    second.mkdir()
    (first / "metrics.json").write_text(
        json.dumps({"safety_radius_m": 0.2, "euclidean_imitation_loss_m": 0.8})
    )
    (second / "metrics.json").write_text(
        json.dumps({"safety_radius_m": 0.5, "euclidean_imitation_loss_m": 0.3})
    )

    assert best_checkpoint(tmp_path, "safety_radius") == first
    assert best_checkpoint(tmp_path, "euclidean") == second
