"""Isolated ROS stand-in test of the real guided recorder state machine."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario", ["success", "rotated_marker", "miss", "held", "stale", "clock", "restart", "missing", "interrupted", "survey_pause"])
def test_guided_recording(scenario):
    run = subprocess.run([sys.executable, __file__, scenario], cwd=ROOT, text=True,
                         capture_output=True, timeout=45)
    assert run.returncode == 0, run.stdout + run.stderr


def run_scenario(scenario):
    import csv
    import json
    import tempfile
    import numpy as np
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / "test"))
    import test_calibration_replay as replay
    replay.install_fake_ros()
    for folder in (ROOT / "fine_tuning" / "scripts", ROOT / "scripts"):
        sys.path.insert(0, str(folder))
    import person_experiment as exp
    import person_workspace_survey as guide
    import residual_calibration_recorder as recorder
    guide.time = recorder.time = replay.FakeTime()
    guide.verify_model = lambda manifest: {}  # no weights needed for recorder contract
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        params = ROOT / "config" / "llm_vision_planner.yaml"
        survey = {
            "kind": "survey", "output_dir": str(root), "params_file": str(params),
            "params_sha256": exp.file_hash(params), "hover": [-3.5, 0, -.5, 0],
            "person": {"object_id": "person-1", "label": "person", "topic": "/vicon/person/person",
                       "dimensions_m": [.4, .3], "yaw_rad": 0},
            "locations": [{"location_id": "one", "x": 0., "y": 1.}],
            "seed": 17, "visits": 1, "checks_per_visit": 2, "interval_s": 1.,
            "position_tolerance_m": .05, "yaw_tolerance_rad": .09,
            "freshness_s": 1., "sync_tolerance_s": .1, "min_confidence": .7,
            "match_distance_m": .75, "stability_window_s": .5,
            "stability_min_samples": 5, "stability_position_tolerance_m": .02,
        }
        schedule = [{"trial_id": "repeat-1", "location_id": "one", "x": 0., "y": 1., "split": "Dcal"},
                    {"trial_id": "repeat-2", "location_id": "one", "x": 0., "y": 1., "split": "Dtest"}]
        exp.seal(root / "experiment.json", {"kind": "experiment", "domain": {"survey": survey},
                 "collection_schedule": schedule, "output_dir": str(root)})
        exp.seal(root / "survey.json", survey)
        surveying = scenario == "survey_pause"
        replay.NODE_PARAMS["residual_calibration_recorder"] = {
            "trial_id": "test", "output_csv": str(root / "measured.raw.csv"),
            "collection_mode": "survey" if surveying else "scheduled",
            "experiment_manifest": str(root / ("survey.json" if surveying else "experiment.json")),
            "scheduled_capture": True, "survey_plot_enabled": False,
            "vicon_objects_json": json.dumps([survey["person"]])}
        node = guide.PersonWorkspaceSurvey()
        node.world_to_ned = (np.zeros(3), np.eye(3))
        start = replay.BUS.t
        first_done_at = None
        paused = False
        for i in range(150):
            replay.BUS.t = start + .1*i
            t = replay.BUS.t
            monotonic = guide.time.monotonic()
            if scenario == "missing":
                node.step()
                continue
            node.vehicle_px4.append((monotonic, (np.array([-3.5, 0, -.5]), np.eye(3))))
            node.vehicle_vicon.append((monotonic, (np.array([-3.5, 0, -.5]), np.eye(3))))
            person_x = 0.
            if not surveying and first_done_at is not None and .1 < t - first_done_at < .9:
                person_x = .3
            offset = 100 if scenario == "clock" else 0
            marker_rotation = (np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
                               if scenario == "rotated_marker" else np.eye(3))
            node.object_history["/vicon/person/person"].append(
                (t + offset, (np.array([person_x, 1., 0.]), marker_rotation)))
            node.mission_callback(SimpleNamespace(data=json.dumps({"state": "HOLDING_FOR_PLAN"})))
            scene = {"timestamp": t, "pose": [-3.5, 0, -.5, 0], "healthy": True, "range_is_measured": True,
                     "status": {"detector": "ok", "pose": "ok", "point_cloud": "ok", "detection_frame_id": i,
                                "point_cloud_sync_delta_s": .01, "pose_sync_delta_s": .01,
                                "detection_age_s": .1, "point_cloud_age_s": .1},
                     "obstacles": [{"object_id": "person-1", "label": "person", "confidence": .9,
                          "held": False, "depth_ready": True, "depth_method": "outline", "source": "tof_projected",
                          "measured_depth_along_view_m": .4, "front_surface_center": [person_x-.2, 1., -.5],
                          "view_axis_xy": [1., 0.], "lateral_axis_xy": [0., 1.], "visible_width_m": .3,
                          "min_corner": [person_x-.2, .85, -1.], "max_corner": [person_x+.2, 1.15, 0.]}]}
            if scenario == "miss":
                scene["obstacles"] = []
            if scenario == "held":
                scene["obstacles"][0]["held"] = True
            if scenario == "stale":
                scene["status"]["detection_age_s"] = 9.
            if surveying and node.due is not None and not paused:
                scene["status"]["point_cloud"] = "missing"
                paused = True
            node.scene_callback(SimpleNamespace(data=json.dumps(scene)))
            node.step()
            if scenario == "interrupted" and node.due is not None:
                break
            if node.index == 1 and first_done_at is None:
                first_done_at = t
                if scenario == "restart":
                    node.destroy_node()
                    node = guide.PersonWorkspaceSurvey()
                    node.world_to_ned = (np.zeros(3), np.eye(3))
            if node.stopped or node.index == len(schedule):
                break
        node.destroy_node()
        with (root / "measured.raw.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        events = exp.read_events(root / ("survey.events.jsonl" if surveying else "collection.events.jsonl"))
        if scenario in ("success", "rotated_marker", "restart"):
            assert [r["capture_id"] for r in rows] == ["repeat-1", "repeat-2"], (rows, replay.LOG[-8:])
            assert all(r["missed_detection"] == "false" for r in rows)
            if scenario == "rotated_marker":
                assert rows[0]["gt_yaw_rad"] == "0.000000000"
                assert rows[0]["gt_min_x"] == "-0.200000000"
                assert rows[0]["gt_max_x"] == "0.200000000"
                assert rows[0]["gt_min_y"] == "0.850000000"
                assert rows[0]["gt_max_y"] == "1.150000000"
            assert len([e for e in events if e["type"] == "recorded"]) == 2
            assert any("Snapshot recorded: 2/2" in message for _, _, message in replay.LOG)
        elif scenario in ("miss", "held", "stale"):
            assert rows == []
            assert any(e["type"] == "failure" for e in events)
            try:
                guide.PersonWorkspaceSurvey()
            except ValueError as exc:
                assert "failed" in str(exc)
            else:
                raise AssertionError("failed experiment resumed")
        elif surveying:
            assert rows == []
            assert len([e for e in events if e["type"] == "infrastructure_pause"]) == 1
            observations = [e for e in events if e["type"] == "observation"]
            assert len(observations) == 2 and all(not e["errors"] for e in observations)
        elif scenario == "interrupted":
            try:
                guide.PersonWorkspaceSurvey()
            except ValueError as exc:
                assert "interrupted" in str(exc)
            else:
                raise AssertionError("interrupted trial retried")
        else:
            assert rows == [] and events == []
            assert any("Waiting" in message for _, _, message in replay.LOG)
        print(f"guided recorder {scenario}: PASS")


if __name__ == "__main__":
    run_scenario(sys.argv[1])
