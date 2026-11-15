"""Offline contracts for the frozen person-only split-CP experiment."""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fine_tuning" / "scripts"))

import person_experiment as exp
from lqr import A_DOUBLE_INTEGRATOR as A, B_DOUBLE_INTEGRATOR as B, solve_care, certified_metric_alpha
from min_control_qp import discrete_dynamics, propagate_state
from postprocess_residual_calibration import ideal_metric, residual_score
from residual_certificate import (AdaptiveResidualCalibration, certificate, certificate_from_csv,
                                  empirical_quantile, read_scores, split_quantile, split_rank)


def test_ideal_dynamics_and_shared_metric():
    assert np.array_equal(A[2:], np.zeros((2, 4)))
    assert np.array_equal(B[2:], np.eye(2))
    assert np.allclose(propagate_state([0, 0, 1, 2], [2, -1], 0.5), [0.75, 0.875, 2, 1.5])
    assert np.allclose(discrete_dynamics(0.5)[1], [0.125, 0.5])
    p, k, _ = solve_care()
    b2, p2, k2 = ideal_metric()
    assert np.array_equal(B, b2)
    assert np.allclose(p, p2) and np.allclose(k, k2)
    assert certified_metric_alpha(A, B, k, p) > 0
    assert np.max(np.abs(A.T @ p + p @ A - p @ B @ B.T @ p + np.diag([10, 10, 1, 1]))) < 1e-10


@pytest.mark.parametrize("n,delta,rank,finite", [(0, .2, 1, False), (1, .2, 2, False),
    (3, .2, 4, False), (4, .2, 4, True), (9, .1, 9, True), (19, .05, 19, True), (1, .5, 1, True)])
def test_split_rank_boundaries(n, delta, rank, finite):
    scores = list(range(n))
    assert split_rank(n, delta) == rank
    q = split_quantile(scores, delta)
    assert math.isfinite(q) is finite
    result = certificate(scores, delta)
    json.dumps(result, allow_nan=False)
    assert result["rank"] == rank and result["quantile_unbounded"] is not finite
    assert result["calibrated_quantile"] == (rank - 1 if finite else None)


def test_split_ties_and_no_filtering():
    assert split_quantile([2] * 9, .1) == 2
    assert empirical_quantile([1, 2, 3], .2) == 3
    assert math.isinf(split_quantile([1, 2, 3], .2))
    for bad in (math.nan, math.inf, -1):
        with pytest.raises(ValueError):
            split_quantile([1, bad], .2)
    for bad in (0, 1, -0.1, math.nan):
        with pytest.raises(ValueError):
            split_rank(10, bad)


def test_acp_history_cannot_activate_split(tmp_path):
    acp = AdaptiveResidualCalibration([1, 2, 3, 4], .2, .01, tmp_path / "acp.json")
    frozen = acp.snapshot(True)
    acp.update("flight-1", 100, frozen)
    result = acp.snapshot(False)
    assert result["active_calibration"] == "calibrated"
    assert result["active_quantile"] == 4
    saved = (tmp_path / "acp.json").read_bytes()
    loaded = AdaptiveResidualCalibration([1, 2, 3, 4], .2, .01, tmp_path / "acp.json")
    assert loaded.snapshot(False)["active_quantile"] == 4
    assert (tmp_path / "acp.json").read_bytes() == saved
    assert loaded.snapshot(True)["acp_updates"] == 1
    # Even a state belonging to another model cannot prevent frozen split use.
    other = AdaptiveResidualCalibration([2, 3, 4, 5], .2, .01, tmp_path / "acp.json")
    assert other.snapshot(False)["active_quantile"] == 5
    with pytest.raises(ValueError, match="different calibration"):
        other.snapshot(True)


def test_residual_supremum_interior_and_control_jumps():
    # Choose delta x,u so the scalar residual peaks strictly inside [0,2].
    _, p, k = ideal_metric()
    kp, kv = k[0, 0], k[0, 2]
    du = -2.0 / kp
    velocity = (2.0 - kv * du) / kp
    position = (-du - kv * velocity) / kp
    x = [position, 0, velocity, 0]
    end = propagate_state(x, [du, 0], 2.0)
    expert = [{"t": 0., "x": [0.] * 4, "u": [0., 0.]}, {"t": 2., "x": [0.] * 4, "u": [0., 0.]}]
    # Keep terminal residual zero so the interior maximum is decisive.
    end_u = -float(k[0] @ end)
    predicted = [{"t": 0., "x": x, "u": [du, 0.]}, {"t": 2., "x": end.tolist(), "u": [end_u, 0.]}]
    maximum, _, _ = residual_score(expert, predicted)
    assert maximum[1] == pytest.approx(1.0, abs=1e-8)
    assert maximum[0] == pytest.approx(math.sqrt(p[2, 2]), rel=1e-8)
    predicted[-1]["u"] = [100, 0]
    assert residual_score(expert, predicted)[0][1] == 2
    predicted[-1]["t"] = 3
    with pytest.raises(ValueError, match="times differ"):
        residual_score(expert, predicted)


def survey_config(tmp_path):
    return exp.seal(tmp_path / "survey.json", {
        "kind": "survey", "output_dir": str(tmp_path), "seed": 17,
        "source_root": str(ROOT), "params_file": str(ROOT / "config/llm_vision_planner.yaml"),
        "params_sha256": exp.file_hash(ROOT / "config/llm_vision_planner.yaml"),
        "perception": {}, "perception_fingerprint": exp.perception_fingerprint(ROOT, {}),
        "locations": exp.grid_locations([0, .25, 1, 1], .25),
        "visits": 3, "checks_per_visit": 10, "interval_s": 1.,
        "hover": [-3.5, 0, -.5, 0], "person": {"dimensions_m": [.4, .3], "yaw_rad": 0},
        "freshness_s": 1., "sync_tolerance_s": .1, "match_distance_m": .75,
        "min_confidence": .7})


def complete_survey(tmp_path, fail_one=False):
    config = survey_config(tmp_path)
    for index, trial in enumerate(exp.survey_schedule(config)):
        exp.append_event(tmp_path / "survey.events.jsonl", {
            "type": "observation", "trial_id": trial["trial_id"],
            "errors": ["missed person"] if fail_one and index == 0 else [], "frame_id": index})
    return exp.freeze_domain(tmp_path / "survey.json")


def test_survey_excludes_failure_and_does_not_fill_gaps(tmp_path):
    domain = complete_survey(tmp_path, True)
    assert len(domain["locations"]) == 1
    report = exp.load_json(tmp_path / "survey.report.json")
    assert report["passed"] == 1 and report["tested"] == 2
    assert (tmp_path / "survey.map.svg").is_file()
    with pytest.raises(FileExistsError):
        exp.freeze_domain(tmp_path / "survey.json")


def test_incomplete_survey_cannot_freeze(tmp_path):
    survey_config(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        exp.freeze_domain(tmp_path / "survey.json")


def healthy_scene():
    return {"timestamp": 10., "healthy": True, "range_is_measured": True,
            "status": {"detection_frame_id": 1, "point_cloud_sync_delta_s": .01, "pose_sync_delta_s": .01,
                       "detection_age_s": .1, "point_cloud_age_s": .1},
            "obstacles": [{"label": "person", "held": False, "source": "tof",
                           "confidence": .9, "depth_ready": True, "depth_method": "outline"}]}


@pytest.mark.parametrize("failure", ["healthy", "held", "frame", "confidence", "stale", "prior", "depth", "clock", "other", "detection_age_s", "point_cloud_age_s", "outline"])
def test_health_check_rejects_invalid_observations(tmp_path, failure):
    config = survey_config(tmp_path)
    scene = healthy_scene()
    previous = None
    if failure == "healthy": scene["healthy"] = False
    elif failure == "held": scene["obstacles"][0]["held"] = True
    elif failure == "frame": previous = 1
    elif failure == "confidence": scene["obstacles"][0]["confidence"] = .1
    elif failure == "stale": scene["timestamp"] = 0
    elif failure == "prior": scene["obstacles"][0]["depth_method"] = "class_prior"
    elif failure == "depth": scene["obstacles"][0]["depth_ready"] = False
    elif failure == "clock": scene["status"]["point_cloud_sync_delta_s"] = 1.
    elif failure == "other": scene["obstacles"].append({"label": "chair"})
    elif failure == "outline": scene["obstacles"][0].update(depth_method="outline_floor", outline_age_s=9.)
    elif failure.endswith("age_s"): scene["status"][failure] = 10.
    nominal = {"nominal_footprint_corners_xy": [[-.2, .8], [-.2, 1.2], [.2, .8], [.2, 1.2]],
               "effective_depth_along_view_m": .4}
    assert exp.health_errors(scene, nominal, {"center": [0, 1]}, config, 10.1, previous)
    assert exp.health_errors(healthy_scene(), nominal, {"center": [0, 1]}, config, 10.1) == []


def prepared(tmp_path):
    complete_survey(tmp_path)
    return exp.prepare(tmp_path / "domain.json", dict(Dtrain=8, Dval=2, Dcal=9, Dtest=5), .1, 17,
                       {"goal": [3.5, 0], "workspace": [-4, 4, -3, 3], "l1_text": "Go to the goal.",
                        "l2_text": "Stay far from the person.", "clearance_m": .4},
                       {"selector": "heuristic", "model": "", "iterations": 500})


def test_splits_with_replacement_are_frozen_and_disjoint(tmp_path):
    experiment = prepared(tmp_path)
    ids = [s["trial_id"] for split in experiment["splits"].values() for s in split]
    assert len(set(ids)) == len(ids) == 24
    assert len({s["location_id"] for s in experiment["splits"]["Dcal"]}) <= 2
    assert {s["split"] for s in experiment["collection_schedule"]} == {"Dcal", "Dtest"}
    assert exp.checked_manifest(tmp_path / "experiment.json") == experiment
    manifest = exp.load_json(tmp_path / "experiment.json")
    manifest["delta"] = .2
    exp.write_json(tmp_path / "experiment.json", manifest, replace=True)
    with pytest.raises(ValueError, match="modified"):
        exp.checked_manifest(tmp_path / "experiment.json")


def test_small_planned_calibration_is_rejected_before_sampling(tmp_path):
    complete_survey(tmp_path)
    with pytest.raises(ValueError, match="finite split-CP"):
        exp.prepare(tmp_path / "domain.json", dict(Dtrain=1, Dval=1, Dcal=1, Dtest=1), .1, 17, {}, {})
    assert not (tmp_path / "experiment.json").exists()


def test_missing_checkpoint_is_not_faked(tmp_path):
    with pytest.raises(ValueError, match="real checkpoint"):
        exp.checkpoint_hashes(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_strict_scoring_requires_every_trial(tmp_path, monkeypatch):
    experiment = prepared(tmp_path)
    monkeypatch.setattr(exp, "verify_model", lambda _: {})
    with pytest.raises(ValueError, match="complete Dcal/Dtest"):
        exp.score_experiment(tmp_path / "experiment.json")
    exp.append_event(tmp_path / "collection.events.jsonl", {"type": "failure", "trial_id": experiment["collection_schedule"][0]["trial_id"]})
    with pytest.raises(ValueError, match="failed scheduled trial"):
        exp.score_experiment(tmp_path / "experiment.json")
    assert not (tmp_path / "split_certificate.json").exists()


def write_scores(path, scores):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["capture_id", "l1_text", "l2_text", "placeholder", "planner_provider", "dynamics_model", "conformity_score"])
        writer.writeheader()
        for i, score in enumerate(scores):
            writer.writerow(dict(capture_id=str(i), l1_text="goal", l2_text="far", placeholder="false",
                                 planner_provider="transformer", dynamics_model="ideal_planar_double_integrator", conformity_score=score))


def test_invalid_score_cannot_be_silently_dropped(tmp_path):
    path = tmp_path / "scores.csv"
    write_scores(path, [1, "bad", 2, 3])
    with pytest.raises(ValueError, match="refusing subset"):
        read_scores(path)


def test_frozen_delta_and_dataset_hash(tmp_path):
    path = tmp_path / "Dcal.scored.csv"
    write_scores(path, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="frozen split_certificate"):
        certificate_from_csv(path, .2)
    frozen = certificate([1, 2, 3, 4], .2, {"calibration_csv": str(path), "calibration_dataset_sha256": exp.file_hash(path)})
    exp.write_json(tmp_path / "split_certificate.json", frozen)
    assert certificate_from_csv(path, .2)["calibrated_quantile"] == 4
    with pytest.raises(ValueError, match="delta"):
        certificate_from_csv(path, .1)
    write_scores(path, [1, 2, 3, 5])
    with pytest.raises(ValueError, match="changed"):
        certificate_from_csv(path, .2)


def test_synthetic_export_matches_trainer_schema(tmp_path):
    prepared(tmp_path)
    exp.export_synthetic(tmp_path / "experiment.json")
    for split, count in (("Dtrain", 8), ("Dval", 2)):
        with (tmp_path / f"{split}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        assert all(r["distillation_mode"] == "dss" for r in rows)
        assert all(len(json.loads(r["completion"])["waypoints"]) >= 2 for r in rows)
        assert all(len(json.loads(r["environment"])["obstacles"]) == 1 for r in rows)
    with pytest.raises(ValueError, match="replace"):
        exp.export_synthetic(tmp_path / "experiment.json")


@pytest.mark.parametrize("planner_failure", [False, True])
def test_scoring_split_integrity_and_planner_failure(tmp_path, monkeypatch, planner_failure):
    """Stub weights/planner only; exercise actual accounting, files and certificate."""
    import postprocess_residual_calibration as post
    import waypoint_transformer
    experiment = prepared(tmp_path)
    monkeypatch.setattr(exp, "verify_model", lambda _: {"checkpoint": str(tmp_path), "fingerprint": "test-model", "pipeline": {}})
    monkeypatch.setattr(waypoint_transformer, "WaypointTransformerPlanner", lambda *_: object())
    rows = []
    for trial in experiment["collection_schedule"]:
        row = {field: "" for field in post.RAW_REQUIRED_FIELDS}
        row.update(capture_id=trial["trial_id"], label="person", missed_detection="false", placeholder="false")
        rows.append(row)
        exp.append_event(tmp_path / "collection.events.jsonl", {"type": "recorded", "trial_id": trial["trial_id"],
                                                               "raw_row_fingerprint": exp.digest(row)})
    with (tmp_path / "measured.raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def process(trial_id, raw_rows, args):
        if planner_failure and "Dtest" in trial_id:
            raise ValueError("planner rejected test scene")
        result = {field: "" for field in post.SCORED_FIELDS}
        result.update(capture_id=trial_id, l1_text="goal", l2_text="far",
                      conformity_score="1000" if "Dtest" in trial_id else "2",
                      placeholder="false", planner_provider="transformer",
                      dynamics_model="ideal_planar_double_integrator")
        return result
    monkeypatch.setattr(post, "process_capture", process)
    if planner_failure:
        with pytest.raises(ValueError, match="no subset certificate"):
            exp.score_experiment(tmp_path / "experiment.json")
        assert not (tmp_path / "split_certificate.json").exists()
        assert not (tmp_path / "Dcal.scored.csv").exists()
        assert len(exp.read_events(tmp_path / "scoring.failures.jsonl")) == 1
    else:
        result = exp.score_experiment(tmp_path / "experiment.json")
        assert result["calibrated_quantile"] == 2 and result["calibration_sample_count"] == 9
        assert certificate_from_csv(tmp_path / "Dcal.scored.csv", .1)["rank"] == 9
        assert exp.load_json(tmp_path / "test.report.json")["empirical_coverage"] == 0
        with pytest.raises(ValueError, match="frozen Dcal"):
            certificate_from_csv(tmp_path / "Dtest.scored.csv", .1)
