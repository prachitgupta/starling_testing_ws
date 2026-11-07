from types import SimpleNamespace

from fine_tuning.scripts.calibration_verifier_gateway import (
    requested_mode,
    transformer_environment_pairs,
)


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
