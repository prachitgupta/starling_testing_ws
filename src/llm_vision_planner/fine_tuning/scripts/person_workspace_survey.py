#!/usr/bin/env python3
"""Guided person survey/collection using the production recorder and perception.

Started by vision_error_calibration.launch.py with collection_mode:=survey or
collection_mode:=scheduled. No motion commands are issued by this node.
"""
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from std_msgs.msg import String

from residual_calibration_recorder import ResidualCalibrationRecorder, ground_truth_aabb, LATCHED_QOS
from person_experiment import (angle_distance, append_event, checked_manifest, digest, file_hash,
                               health_errors, read_events, survey_schedule, verify_model)

try:
    from interactive_mission_gateway import nominal_obstacles_from_depth
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from interactive_mission_gateway import nominal_obstacles_from_depth


def survey_location_states(config, events):
    """Return pending/validated/rejected IDs from recorded survey evidence."""
    observations = [event for event in events if event.get("type") == "observation"]
    by_location = {location["location_id"]: [] for location in config["locations"]}
    trial_locations = {item["trial_id"]: item["location_id"] for item in survey_schedule(config)}
    for event in observations:
        location_id = trial_locations.get(event.get("trial_id"))
        if location_id in by_location:
            by_location[location_id].append(event)
    expected = int(config["visits"]) * int(config["checks_per_visit"])
    rejected = {location_id for location_id, rows in by_location.items()
                if any(row.get("errors") for row in rows)}
    validated = {location_id for location_id, rows in by_location.items()
                 if len(rows) == expected and location_id not in rejected}
    pending = set(by_location) - rejected - validated
    return pending, validated, rejected


class SurveyLivePlot:
    """Small operator display; it has no role in survey acceptance decisions."""
    def __init__(self, config, output_path):
        display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        import matplotlib
        if not display_available:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.config = config
        self.output_path = Path(output_path)
        self.figure, self.axes = plt.subplots(figsize=(8, 7))
        self.show_window = (display_available and
                            getattr(self.figure.canvas, "required_interactive_framework", None) is not None)
        if self.show_window:
            plt.ion()
            self.figure.show()

    def render(self, events, target=None, person_xy=None, status="", save=False):
        pending, validated, rejected = survey_location_states(self.config, events)
        locations = {item["location_id"]: item for item in self.config["locations"]}
        axes = self.axes
        axes.clear()

        def scatter(ids, **style):
            points = [locations[location_id] for location_id in ids]
            if points:
                axes.scatter([point["x"] for point in points], [point["y"] for point in points], **style)

        # Rejected locations are deliberately omitted from the grid.
        scatter(sorted(pending), c="#a7adb4", s=28, marker="o", label="pending grid")
        scatter(sorted(validated), c="#228833", s=48, marker="s", label="validated")
        if target is not None:
            axes.scatter([target["x"]], [target["y"]], c="#ee7733", s=180, marker="*",
                         edgecolors="black", linewidths=.7, label="current target", zorder=4)
        if person_xy is not None:
            axes.scatter([person_xy[0]], [person_xy[1]], c="#0077bb", s=90, marker="X",
                         edgecolors="white", linewidths=.7, label="Vicon person", zorder=5)

        all_x = [point["x"] for point in locations.values()]
        all_y = [point["y"] for point in locations.values()]
        margin = max(float(self.config.get("spacing_m", .25)), .25)
        axes.set_xlim(min(all_x) - margin, max(all_x) + margin)
        axes.set_ylim(min(all_y) - margin, max(all_y) + margin)
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, alpha=.25)
        axes.set_xlabel("NED x (m)")
        axes.set_ylabel("NED y (m)")
        axes.set_title(status or "Person workspace survey")
        axes.legend(loc="best")
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        if self.show_window:
            self.figure.canvas.flush_events()
        if save or not self.show_window:
            self.figure.savefig(self.output_path, dpi=120)

    def close(self):
        self.plt.close(self.figure)


class PersonWorkspaceSurvey(ResidualCalibrationRecorder):
    def __init__(self):
        super().__init__()
        self.declare_parameter("collection_mode", "survey")
        self.declare_parameter("experiment_manifest", "")
        self.declare_parameter("semantic_obstacle_topic", "/llm_vision/semantic_obstacles")
        self.declare_parameter("mission_state_topic", "/llm_vision/mission_state")
        self.declare_parameter("required_mission_state", "HOLDING_FOR_PLAN")
        self.declare_parameter("replay", False)
        self.declare_parameter("survey_plot_enabled", True)
        self.declare_parameter("survey_plot_refresh_s", 0.2)
        self.mode = str(self.get_parameter("collection_mode").value)
        if self.mode not in ("survey", "scheduled"):
            raise ValueError("guided collection_mode must be survey or scheduled")
        manifest = checked_manifest(str(self.get_parameter("experiment_manifest").value),
                                    "survey" if self.mode == "survey" else "experiment")
        self.config = manifest if self.mode == "survey" else manifest["domain"]["survey"]
        self.root = Path(self.config["output_dir"])
        # Old sealed manifests used one shared position gate.  New manifests
        # keep the camera fixed more tightly than the human target region.
        self.vehicle_position_tolerance = float(self.config.get(
            "vehicle_position_tolerance_m", self.config.get("position_tolerance_m", 0.10)))
        self.person_position_tolerance = float(self.config.get(
            "person_position_tolerance_m", self.config.get("position_tolerance_m", 0.20)))
        if file_hash(self.config["params_file"]) != self.config["params_sha256"]:
            raise ValueError("YAML changed after survey configuration was frozen")
        if len(self.objects) != 1 or self.objects[0]["label"] != "person":
            raise ValueError("guided collection requires exactly one Vicon person")
        if self.mode == "scheduled":
            verify_model(manifest)
        self.schedule = survey_schedule(manifest) if self.mode == "survey" else manifest["collection_schedule"]
        self.events_path = self.root / ("survey.events.jsonl" if self.mode == "survey" else "collection.events.jsonl")
        events = read_events(self.events_path)
        self.survey_events = list(events) if self.mode == "survey" else []
        if any(e.get("type") == "failure" for e in events):
            raise ValueError("a scheduled trial failed; review this experiment before starting a new version")
        completed_type = "observation" if self.mode == "survey" else "recorded"
        completed = [e for e in events if e.get("type") == completed_type]
        if [e["trial_id"] for e in completed] != [s["trial_id"] for s in self.schedule[:len(completed)]]:
            raise ValueError("event history is not an exact prefix of the frozen schedule")
        finished = {e["trial_id"] for e in completed}
        lifecycle = {e["trial_id"]: e["type"] for e in events if "trial_id" in e}
        if any(state == "started" for state in lifecycle.values()):
            raise ValueError("interrupted trial exists; do not retry or silently omit it; retain evidence and start a new version")
        if self.mode == "scheduled":
            with self.output_csv.open(newline="") as stream:
                raw = list(csv.DictReader(stream))
            if {r["capture_id"] for r in raw} != finished or len(raw) != len(finished):
                raise ValueError("raw CSV/event history mismatch; preserve files and review interrupted recording")
        self.index = len(completed)
        self.previous_frame = completed[-1].get("frame_id") if completed else None
        self.previous_target = self.schedule[self.index - 1] if self.index else None
        self.departed = not bool(self.previous_target)
        self.latest_scene = None
        self.scene_received_s = None
        self.mission = None
        self.mission_received_s = None
        self.pending = None
        self.due = None
        self.stopped = False
        self.done_notice = False
        self.plot = None
        self.plot_status = "Survey starting. Waiting for synchronized inputs."
        self.last_plot_refresh = -math.inf
        if self.mode == "survey" and bool(self.get_parameter("survey_plot_enabled").value):
            try:
                self.plot = SurveyLivePlot(self.config, self.root / "survey.live.png")
                self.refresh_plot(force=True)
                self.get_logger().info(f"Live survey plot: {self.root / 'survey.live.png'}")
            except Exception as exc:
                self.get_logger().warning(f"Live survey plot unavailable; collection will continue: {exc}")
                self.plot = None
        self.create_subscription(String, str(self.get_parameter("semantic_obstacle_topic").value), self.scene_callback, 10)
        self.create_subscription(String, str(self.get_parameter("mission_state_topic").value), self.mission_callback, LATCHED_QOS)
        self.create_timer(0.1, self.step)
        self.publish_status("GUIDE_READY", f"{self.mode}: {self.index}/{len(self.schedule)} complete. Output: {self.root}")

    def nominal_callback(self, message):
        # Only the predetermined guided capture may create a row.
        pass

    def try_record(self, configured):
        # Parent object subscriptions still collect Vicon history.
        # Recording is explicitly invoked below after the scheduled check.
        pass

    def recording_watchdog(self):
        if self.world_to_ned is None:
            super().recording_watchdog()

    def scene_callback(self, message):
        try:
            scene = json.loads(message.data)
            if not isinstance(scene, dict):
                raise ValueError("scene must be an object")
            self.latest_scene = scene
            self.scene_received_s = time.monotonic()
        except (ValueError, TypeError) as exc:
            self.publish_status("INVALID_PERCEPTION", f"Invalid perception payload: {exc}")

    def mission_callback(self, message):
        try:
            self.mission = json.loads(message.data)
            self.mission_received_s = time.monotonic()
        except ValueError:
            self.mission = None

    def wait(self, code, message):
        self.plot_status = message
        self.publish_status(code, message)

    def person_xy(self):
        if self.world_to_ned is None:
            return None
        history = self.object_history[self.objects[0]["topic"]]
        if not history:
            return None
        try:
            return self.ground_truth_geometry(self.objects[0], history[-1][1])["center"]
        except (KeyError, TypeError, ValueError):
            return None

    def ground_truth_geometry(self, configured, marker_pose):
        """Use Vicon translation only; the person footprint stays fixed in NED."""
        return ground_truth_aabb(
            configured,
            marker_pose,
            self.world_to_ned,
            fixed_yaw_rad=float(self.config["person"]["yaw_rad"]),
        )

    def refresh_plot(self, force=False):
        if self.plot is None:
            return
        now = time.monotonic()
        refresh_s = max(float(self.get_parameter("survey_plot_refresh_s").value), 0.05)
        if not force and now - self.last_plot_refresh < refresh_s:
            return
        target = self.schedule[self.index] if self.index < len(self.schedule) else None
        self.plot.render(self.survey_events, target, self.person_xy(), self.plot_status, save=force)
        self.last_plot_refresh = now

    def infrastructure_error(self):
        now = time.monotonic()
        if self.world_to_ned is None:
            return "Waiting for synchronized vehicle pose and frame alignment."
        if not self.vehicle_px4 or now - self.vehicle_px4[-1][0] > self.config["freshness_s"]:
            return "Waiting for fresh PX4 NED pose."
        if not self.vehicle_vicon or now - self.vehicle_vicon[-1][0] > self.config["freshness_s"]:
            return "Waiting for fresh vehicle Vicon pose."
        if not bool(self.get_parameter("replay").value):
            if self.mission is None or self.mission_received_s is None or now - self.mission_received_s > 2.0:
                return "Waiting for mission-state updates."
            required = str(self.get_parameter("required_mission_state").value)
            if self.mission.get("state") != required:
                return f"Waiting for hover state {required}."
        if self.latest_scene is None:
            return "Waiting for perception topic; check camera, detector and ToF bridges."
        if self.scene_received_s is None or now - self.scene_received_s > self.config["freshness_s"]:
            return "Waiting for fresh perception publications."
        status = self.latest_scene.get("status", {})
        for field in ("detector", "pose", "point_cloud"):
            if status.get(field) != "ok":
                return f"Waiting for healthy {field} stream: {status.get(field, 'missing')}."
        history = self.object_history[self.objects[0]["topic"]]
        ros_now = self.get_clock().now().nanoseconds * 1e-9
        if not history or abs(ros_now - history[-1][0]) > self.config["freshness_s"]:
            return f"Waiting for person Vicon at {self.objects[0]['topic']}; check clock offset."
        return None

    def pose_error(self, target):
        position, rotation = self.vehicle_px4[-1][1]
        hover = self.config["hover"]
        if math.dist(position, hover[:3]) > self.vehicle_position_tolerance:
            return "Vehicle drifted from the frozen hover position."
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        if angle_distance(yaw, hover[3]) > self.config["yaw_tolerance_rad"]:
            return "Vehicle heading differs from the frozen camera heading."
        history = self.object_history[self.objects[0]["topic"]]
        ground_truth = self.ground_truth_geometry(self.objects[0], history[-1][1])
        if not self.departed and self.previous_target:
            old = [self.previous_target["x"], self.previous_target["y"]]
            if math.dist(ground_truth["center"], old) <= 0.15:
                return "Step away from the previous location by more than 0.15 m, then move to the next target."
            self.departed = True
        if math.dist(ground_truth["center"], [target["x"], target["y"]]) > self.person_position_tolerance:
            return f"Hover ready. Move to target {self.index + 1}/{len(self.schedule)}: x={target['x']:.2f}, y={target['y']:.2f} (NED)."
        end = history[-1][0]
        samples = [self.ground_truth_geometry(self.objects[0], pose) for stamp, pose in history
                   if end - self.stability_window <= stamp <= end]
        if len(samples) < self.stability_min_samples:
            return "Hold still; waiting for person stability samples."
        centers = np.array([s["center"] for s in samples])
        if np.max(np.linalg.norm(centers - np.mean(centers, axis=0), axis=1)) > self.stability_position_tolerance:
            return "Hold still."
        return None

    def step(self):
        self.refresh_plot()
        if self.stopped:
            return
        if self.index >= len(self.schedule):
            if not self.done_notice:
                self.plot_status = f"Survey complete: {self.index}/{len(self.schedule)} checks accounted for."
                self.refresh_plot(force=True)
                self.publish_status("COMPLETE", f"Complete: {self.index}/{len(self.schedule)}. Files: {self.root}. Stop the launch using the documented vehicle procedure.")
                self.done_notice = True
            return
        target = self.schedule[self.index]
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.pending is not None:
            self.finish_observation(now)
            return
        error = self.infrastructure_error()
        if error:
            if self.due is not None:
                if self.mode == "survey":
                    append_event(self.events_path, {"type": "infrastructure_pause", "trial_id": target["trial_id"],
                                                  "timestamp": now, "reason": error})
                    self.due = None
                    self.wait("WAITING_FOR_INPUT", error)
                else:
                    self.fail_trial(target, error, now)
            else:
                self.wait("WAITING_FOR_INPUT", error)
            return
        error = self.pose_error(target)
        if error:
            if self.due is not None:
                self.fail_trial(target, error, now)
            else:
                self.wait("POSITIONING", error)
            return
        if self.due is None:
            self.due = now + self.config["interval_s"]
            append_event(self.events_path, {"type": "started", "trial_id": target["trial_id"], "timestamp": now,
                                          "scheduled_timestamp": self.due})
            self.wait("HOLD_STILL", f"Hold still. Scheduled check {self.index + 1}/{len(self.schedule)}.")
            return
        if now < self.due:
            return
        scene = copy.deepcopy(self.latest_scene)
        stamp = float(scene.get("status", {}).get("measurement_timestamp") or scene.get("timestamp", now))
        self.pending = {"scene": scene, "timestamp": stamp, "check_time": now, "target": target}
        self.finish_observation(now)

    def fail_trial(self, target, reason, now):
        if self.mode == "survey":
            event = {"type": "observation", "trial_id": target["trial_id"], "timestamp": now,
                     "errors": [reason], "frame_id": None}
            append_event(self.events_path, event)
            self.advance(event)
        else:
            append_event(self.events_path, {"type": "failure", "trial_id": target["trial_id"],
                                          "timestamp": now, "reason": reason})
            self.stopped = True
            self.publish_status("EXPERIMENT_FAILED", f"STOP: trial {target['trial_id']} failed: {reason} Retain the trial; no subset certificate is allowed.")

    def finish_observation(self, now):
        pending = self.pending
        scene, stamp, target = pending["scene"], pending["timestamp"], pending["target"]
        history = self.object_history[self.objects[0]["topic"]]
        if now - pending["check_time"] > 3.0:
            self.fail_trial(target, "timed out waiting for synchronized Vicon around the scheduled observation", now)
            return
        if not history or history[-1][0] < stamp + 0.5 * self.stability_window:
            return
        stamp_gt, pose = min(history, key=lambda item: abs(item[0] - stamp))
        gt = self.ground_truth_geometry(self.objects[0], pose)
        nominal = None
        errors = []
        if abs(stamp_gt - stamp) > self.sync_tolerance:
            errors.append("snapshot/Vicon timestamp mismatch")
        try:
            persons = [dict(o, object_id="person-1") for o in scene.get("obstacles", []) if o.get("label", "").lower() == "person"]
            if len(persons) == 1:
                nominal = nominal_obstacles_from_depth(persons, [{"object_id": "person-1",
                    "effective_depth_along_view_m": persons[0].get("measured_depth_along_view_m"),
                    "source": "perception", "abstained": False}])[0]
            errors += health_errors(scene, nominal, gt, self.config, pending["check_time"], self.previous_frame)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid perception geometry: {exc}")
        window = [self.ground_truth_geometry(self.objects[0], p) for t, p in history
                  if abs(t - stamp) <= 0.5 * self.stability_window]
        if len(window) < self.stability_min_samples:
            errors.append("insufficient synchronized stability samples")
        elif any(math.dist(s["center"], [target["x"], target["y"]]) > self.person_position_tolerance
                 for s in window):
            errors.append("person moved outside the target position tolerance")
        else:
            centers = np.array([s["center"] for s in window])
            if np.max(np.linalg.norm(centers - np.mean(centers, axis=0), axis=1)) > self.stability_position_tolerance:
                errors.append("person moved during capture")
        event = {"type": "observation" if self.mode == "survey" else "recorded",
                 "trial_id": target["trial_id"], "timestamp": stamp, "errors": errors,
                 "frame_id": scene.get("status", {}).get("detection_frame_id"),
                 "scene": scene, "ground_truth": gt}
        if self.mode == "scheduled" and errors:
            append_event(self.events_path, dict(event, type="failure", reason="; ".join(errors)))
            self.stopped = True
            self.publish_status("EXPERIMENT_FAILED", f"STOP: {target['trial_id']}: {'; '.join(errors)}. Evidence retained; no retry/subset certificate.")
            return
        if self.mode == "scheduled":
            self.capture_index += 1
            self.latest_nominal = {"capture_id": target["trial_id"], "timestamp": stamp,
                "payload": {"obstacles": [nominal], "observer_pose": scene.get("pose", [])}}
            self.pending_rows = {target["trial_id"]: {}}
            before = self.recorded_count
            ResidualCalibrationRecorder.try_record(self, self.objects[0])
            if self.recorded_count != before + 1:
                self.fail_trial(target, "recorder did not commit the scheduled capture", now)
                return
            with self.output_csv.open(newline="") as stream:
                row = next((r for r in csv.DictReader(stream) if r["capture_id"] == target["trial_id"]), None)
            if row is None or row["missed_detection"] != "false":
                self.fail_trial(target, "recorded row is missing or contains a missed detection", now)
                return
            event["raw_row_fingerprint"] = digest(row)
        append_event(self.events_path, event)
        self.advance(event)

    def advance(self, event):
        old = self.schedule[self.index]
        self.previous_frame = event.get("frame_id")
        self.previous_target = old
        self.index += 1
        self.pending = self.due = None
        self.departed = (self.mode == "survey" and self.index < len(self.schedule)
                         and self.schedule[self.index]["location_id"] == old["location_id"]
                         and self.schedule[self.index]["visit"] == old["visit"])
        if self.mode == "survey":
            self.survey_events.append(event)
            result = "passed" if not event["errors"] else "failed: " + "; ".join(event["errors"])
            _, validated, rejected = survey_location_states(self.config, self.survey_events)
            location_id = old["location_id"]
            if location_id in rejected:
                point_result = f" Location {location_id} rejected and removed from the grid."
            elif location_id in validated:
                point_result = f" Location {location_id} validated and latched green."
            else:
                point_result = ""
            self.plot_status = f"Survey check {self.index}/{len(self.schedule)} {result}.{point_result}"
            self.refresh_plot(force=True)
            self.publish_status("SURVEY_CHECK", self.plot_status)
        else:
            self.publish_status("RECORDED_GUIDED", f"Snapshot recorded: {self.index}/{len(self.schedule)}. Move to the next location.")

    def destroy_node(self):
        if hasattr(self, "index"):
            self.get_logger().info(f"Guided {self.mode} stopped: {self.index}/{len(self.schedule)} accounted observations; files: {self.root}.")
        if getattr(self, "plot", None) is not None:
            self.refresh_plot(force=True)
            self.plot.close()
            self.plot = None
        return super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = PersonWorkspaceSurvey()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
