"""Evidence-state tests for the live survey display."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
from test_calibration_replay import install_fake_ros  # noqa: E402

install_fake_ros()
sys.path.insert(0, str(ROOT / "fine_tuning" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
from person_workspace_survey import survey_location_states  # noqa: E402


def test_rejected_points_disappear_and_validated_points_latch():
    config = {
        "locations": [
            {"location_id": "pass", "x": 0.0, "y": 0.0},
            {"location_id": "fail", "x": 1.0, "y": 0.0},
        ],
        "visits": 1,
        "checks_per_visit": 2,
        "seed": 7,
    }
    pending, validated, rejected = survey_location_states(config, [])
    assert pending == {"pass", "fail"} and not validated and not rejected

    events = [{"type": "observation", "trial_id": "survey-fail-v00-c000", "errors": ["held detection"]}]
    pending, validated, rejected = survey_location_states(config, events)
    assert rejected == {"fail"} and pending == {"pass"} and not validated

    events += [
        {"type": "observation", "trial_id": "survey-pass-v00-c000", "errors": []},
        {"type": "observation", "trial_id": "survey-pass-v00-c001", "errors": []},
    ]
    pending, validated, rejected = survey_location_states(config, events)
    assert validated == {"pass"} and rejected == {"fail"} and not pending
