#!/usr/bin/env python3
"""Focused checks for the planner-only Case-A calibration path."""

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "fine_tuning/scripts/calibrate_planner_only.py"


def load_module():
    specification = importlib.util.spec_from_file_location("planner_only_calibration", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_case_a_selection_is_one_predeclared_preference_per_layout():
    module = load_module()
    rows = []
    for scene in (1, 2, 3):
        for preference in ("shortest", "balanced", "widest"):
            rows.append(
                {
                    "scene_id": str(scene),
                    "sample_id": f"scene-{scene}-{preference}",
                    "preference_type": preference,
                }
            )
    selected = module.select_case_a_rows(rows, seed=17)
    assert len(selected) == 3
    assert len({row["scene_id"] for row in selected}) == 3
    assert {row["preference_type"] for row in selected} <= {"shortest", "balanced"}


def test_regret_is_applied_only_through_the_max_score():
    module = load_module()
    score, penalty = module.combined_score(residual=0.25, regret_m=0.4, beta=2.0)
    assert score == max(0.25, penalty)
    assert penalty > 0.0
    assert module.combined_score(residual=0.25, regret_m=0.0, beta=2.0) == (0.25, 0.0)


def test_shortest_and_balanced_regret_match_objectives():
    module = load_module()
    cards = [
        {"route_id": "short", "path_length_m": 4.0, "overall_minimum_clearance_m": 0.5},
        {"route_id": "wide", "path_length_m": 5.0, "overall_minimum_clearance_m": 1.0},
    ]
    assert module.language_regret("shortest", cards, "wide", "short") == pytest.approx(1.0)
    expected = ((5.0 / 4.0 - 1.0 / 1.0) - (4.0 / 4.0 - 0.5 / 1.0)) * 4.0
    assert module.language_regret("balanced", cards, "wide", "short") == pytest.approx(
        max(0.0, expected)
    )
    assert math.isclose(module.language_regret("shortest", cards, "short", "short"), 0.0)


if __name__ == "__main__":
    test_case_a_selection_is_one_predeclared_preference_per_layout()
    test_regret_is_applied_only_through_the_max_score()
    test_shortest_and_balanced_regret_match_objectives()
    print("planner-only calibration: PASS")
