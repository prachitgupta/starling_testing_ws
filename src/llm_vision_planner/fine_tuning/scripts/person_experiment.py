#!/usr/bin/env python3
"""Frozen person-location surveys, split manifests, labels and certificates.

The ROS adapter is person_workspace_survey.py. This module has no ROS imports,
so the same validation and sampling rules can be tested offline.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import yaml

from residual_certificate import certificate, split_rank

VERSION = 2
STARLING_FIXED_CAMERA_YAW_RAD = math.radians(84.426)
PIPELINE_FILES = (
    "lqr.py", "min_control_qp.py", "postprocess_residual_calibration.py",
    "residual_certificate.py", "waypoint_transformer.py", "hrrt_star.py",
    "person_experiment.py", "person_workspace_survey.py", "residual_calibration_recorder.py",
    "train_hrrt_waypoint_transformer.py",
    "finalize_hrrt_sft_dataset.py", "generate_hrrt_finetuning_dataset.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def perception_fingerprint(source, perception):
    source = Path(source)
    files = {name: file_hash(source / "scripts" / name) for name in
             ("perception_detection.py", "outline_depth.py", "obstacle_depth.py", "outline_segmentation.py")}
    model = Path(perception.get("efficientsam_model") or source / "models" / "efficientsam_ti_2025april.onnx").expanduser()
    if not model.is_absolute():
        raise ValueError("efficientsam_model must be absolute or empty for the package default")
    # Absence is frozen too: later adding weights changes the observation pipeline.
    files["segmenter_model"] = file_hash(model) if model.is_file() else "absent_measured_shape_fit_fallback"
    return digest(files)


def verify_survey_profile(survey):
    if file_hash(survey["params_file"]) != survey["params_sha256"]:
        raise ValueError("YAML changed after survey freeze")
    if perception_fingerprint(survey["source_root"], survey["perception"]) != survey["perception_fingerprint"]:
        raise ValueError("perception code/assets changed after survey freeze; repeat survey in a new version")


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def append_event(path, event):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        # A partial last line is an explicit recovery error, never a dropped trial.
        return [json.loads(line) for line in stream if line.strip()]


def checked_manifest(path, kind=None):
    value = load_json(path)
    expected = value.pop("fingerprint", None)
    if expected != digest(value) or value.get("version") != VERSION:
        raise ValueError(f"modified or unsupported manifest: {path}")
    value["fingerprint"] = expected
    if kind and value.get("kind") != kind:
        raise ValueError(f"expected {kind} manifest: {path}")
    return value


def seal(path, value):
    value = dict(value, version=VERSION)
    value["fingerprint"] = digest(value)
    write_json(path, value)
    return value


def require_finite(values, name):
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError(f"{name} must be finite")


def grid_locations(bounds, spacing):
    require_finite([*bounds, spacing], "grid")
    xmin, xmax, ymin, ymax = bounds
    if spacing <= 0 or xmin > xmax or ymin > ymax:
        raise ValueError("invalid survey bounds/spacing")
    return [{"location_id": f"P{ix:03d}-{iy:03d}",
             "x": round(xmin + ix * spacing, 9), "y": round(ymin + iy * spacing, 9)}
            for ix in range(int(math.floor((xmax - xmin) / spacing + 1e-9)) + 1)
            for iy in range(int(math.floor((ymax - ymin) / spacing + 1e-9)) + 1)]


def survey_schedule(config):
    rng = random.Random(config["seed"])
    schedule = []
    for visit in range(config["visits"]):
        locations = list(config["locations"])
        rng.shuffle(locations)
        for location in locations:
            for check in range(config["checks_per_visit"]):
                schedule.append({**location, "visit": visit, "check": check,
                                 "trial_id": f"survey-{location['location_id']}-v{visit:02d}-c{check:03d}"})
    return schedule


def angle_distance(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def health_errors(scene, nominal, ground_truth, config, now, previous_frame=None):
    """Test one predetermined observation; caller must not select a better one."""
    errors = []
    if not isinstance(scene, dict):
        return ["no perception observation"]
    status = scene.get("status", {})
    stamp = float(scene.get("timestamp", -math.inf))
    if not math.isfinite(stamp) or not -config["sync_tolerance_s"] <= now - stamp <= config["freshness_s"]:
        errors.append("stale perception timestamp")
    if scene.get("healthy") is not True or scene.get("range_is_measured") is not True:
        errors.append("unhealthy or unmeasured perception")
    frame = status.get("detection_frame_id")
    if frame is None or frame == previous_frame:
        errors.append("missing/reused detector frame")
    for field in ("detection_age_s", "point_cloud_age_s"):
        age = status.get(field)
        if age is None or not math.isfinite(float(age)) or not 0 <= float(age) <= config["freshness_s"]:
            errors.append(f"missing/stale {field}")
    if status.get("held_depth_grace_used"):
        errors.append("held depth grace")
    obstacles = scene.get("obstacles", [])
    persons = [o for o in obstacles if o.get("label", "").lower() == "person"]
    if len(persons) != 1 or len(obstacles) != 1:
        errors.append("expected exactly one person and no other obstacles")
    if len(persons) == 1:
        person = persons[0]
        if person.get("held") or "held" in str(person.get("source", "")):
            errors.append("held detection")
        confidence = float(person.get("confidence", 0))
        if not math.isfinite(confidence) or confidence < config["min_confidence"]:
            errors.append("low confidence")
        if person.get("depth_ready") is not True:
            errors.append("depth not ready")
        method = str(person.get("depth_method", ""))
        if not method or "prior" in method or "hardcoded" in method:
            errors.append("depth is not measured")
        if method.startswith("outline_"):
            age = float(person.get("outline_age_s", math.inf))
            if not math.isfinite(age) or not 0 <= age <= config["freshness_s"]:
                errors.append("stale outline depth")
        for field in ("point_cloud_sync_delta_s", "pose_sync_delta_s"):
            value = status.get(field)
            if value is None or not math.isfinite(float(value)) or abs(float(value)) > config["sync_tolerance_s"]:
                errors.append(f"invalid {field}")
    if nominal is None:
        errors.append("invalid nominal footprint")
    else:
        try:
            corners = np.asarray(nominal["nominal_footprint_corners_xy"], dtype=float)
            if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                raise ValueError("corners")
            center = np.mean(corners, axis=0)
            if math.dist(center, ground_truth["center"]) > config["match_distance_m"]:
                errors.append("person/Vicon association failed")
            if float(nominal["effective_depth_along_view_m"]) <= 0:
                errors.append("invalid measured depth")
        except (KeyError, ValueError, TypeError):
            errors.append("invalid nominal geometry")
    return errors


def freeze_domain(config_path):
    config = checked_manifest(config_path, "survey")
    verify_survey_profile(config)
    root = Path(config["output_dir"])
    events = read_events(root / "survey.events.jsonl")
    expected = survey_schedule(config)
    completed = {event["trial_id"]: event for event in events if event.get("type") == "observation"}
    if len(completed) != len(expected) or set(completed) != {s["trial_id"] for s in expected}:
        raise ValueError("survey incomplete; every scheduled check must be accounted for")
    if len([e for e in events if e.get("type") == "observation"]) != len(completed):
        raise ValueError("duplicate survey observations")
    passed = []
    report = []
    for location in config["locations"]:
        rows = [completed[s["trial_id"]] for s in expected if s["location_id"] == location["location_id"]]
        failures = [row for row in rows if row["errors"]]
        if not failures:
            passed.append(location)
        report.append({**location, "checks": len(rows), "failures": len(failures),
                       "reasons": sorted({reason for row in failures for reason in row["errors"]})})
    if not passed:
        raise ValueError("no location passed all checks; retain survey evidence and start a new survey")
    domain = seal(root / "domain.json", {"kind": "domain", "survey": config,
                  "locations": passed, "survey_events_sha256": file_hash(root / "survey.events.jsonl"),
                  "claim": "zero observed failures at tested locations; no continuous-area or future-perfect-detection guarantee"})
    write_json(root / "survey.report.json", {"passed": len(passed), "tested": len(report), "locations": report})
    with (root / "survey.map.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["location_id", "x", "y", "checks", "failures"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in report)
    # Self-contained SVG: exact measured grid points, never an interpolated polygon.
    xs, ys = [p["x"] for p in report], [p["y"] for p in report]
    sx, sy = max(max(xs) - min(xs), 0.25), max(max(ys) - min(ys), 0.25)
    circles = [f'<circle cx="{40+520*(p["x"]-min(xs))/sx:.2f}" cy="{560-520*(p["y"]-min(ys))/sy:.2f}" r="4" fill="{"#228833" if not p["failures"] else "#cc3311"}"><title>{p["location_id"]}: ({p["x"]}, {p["y"]}), failures={p["failures"]}</title></circle>' for p in report]
    with (root / "survey.map.svg").open("x") as stream:
        stream.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 620"><rect width="600" height="620" fill="white"/><text x="20" y="20">NED person locations: green passed; red excluded</text>' + ''.join(circles) + '</svg>')
    return domain


def prepare(domain_path, counts, delta, seed, task, expert):
    domain = checked_manifest(domain_path, "domain")
    if set(counts) != {"Dtrain", "Dval", "Dcal", "Dtest"} or any(n <= 0 for n in counts.values()):
        raise ValueError("all four explicit split counts must be positive")
    if split_rank(counts["Dcal"], delta) > counts["Dcal"]:
        raise ValueError("planned Dcal cannot produce a finite split-CP quantile; increase n or preselect a larger delta")
    require_finite([*task["goal"], *task["workspace"], task["clearance_m"]], "task")
    if not task["l1_text"].strip() or not task["l2_text"].strip():
        raise ValueError("fixed L1 and L2 are required")
    xmin, xmax, ymin, ymax = task["workspace"]
    if xmin >= xmax or ymin >= ymax or task["clearance_m"] < 0:
        raise ValueError("invalid vehicle navigation bounds or clearance")
    start = domain["survey"]["hover"][:2]
    if any(not (xmin <= x <= xmax and ymin <= y <= ymax) for x, y in (start, task["goal"])):
        raise ValueError("fixed start and goal must lie inside vehicle navigation bounds")
    if math.dist(start, task["goal"]) == 0:
        raise ValueError("fixed start and goal must differ")
    if expert["selector"] not in ("heuristic", "openai") or expert["iterations"] <= 0:
        raise ValueError("explicit expert selector and positive HRRT iterations required")
    root = Path(domain["survey"]["output_dir"])
    rng = random.Random(seed)
    splits = {}
    for split, count in counts.items():
        splits[split] = [{**rng.choice(domain["locations"]), "split": split,
                          "trial_id": f"{root.name}-{split}-{i:06d}"} for i in range(count)]
    measured = splits["Dcal"] + splits["Dtest"]
    rng.shuffle(measured)
    return seal(root / "experiment.json", {
        "kind": "experiment", "domain": domain, "counts": counts, "delta": delta,
        "seed": seed, "task": task, "expert": expert, "splits": splits,
        "collection_schedule": measured, "output_dir": str(root),
        "sampling": "uniform locations with replacement; fixed task and person orientation",
        "observations": {"Dtrain": "synthetic geometry", "Dval": "synthetic geometry",
                         "Dcal": "measured perception/Vicon", "Dtest": "measured perception/Vicon"},
        "tracking_assumption": "perfect_tracking_ideal_double_integrator"})


def scoring_args(experiment):
    task, survey, expert = experiment["task"], experiment["domain"]["survey"], experiment["expert"]
    return SimpleNamespace(
        l1_text=task["l1_text"], l2_text=task["l2_text"],
        start_x=survey["hover"][0], start_y=survey["hover"][1], fixed_z=survey["hover"][2],
        goal_x=task["goal"][0], goal_y=task["goal"][1],
        workspace_x_min=task["workspace"][0], workspace_x_max=task["workspace"][1],
        workspace_y_min=task["workspace"][2], workspace_y_max=task["workspace"][3],
        clearance_m=task["clearance_m"], dt=0.1, max_velocity_mps=0.5, max_acceleration_mps2=0.5,
        seed=experiment["seed"], hrrt_iterations=expert["iterations"], max_candidates=8, max_waypoints=8,
        expert_selector=expert["selector"], expert_model=expert["model"], expert_route_id="",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""), planner_provider="transformer", placeholder=False,
        fixed_start=True)


def synthetic_environment(experiment, trial):
    survey, task = experiment["domain"]["survey"], experiment["task"]
    width, depth = survey["person"]["dimensions_m"]
    yaw = survey["person"]["yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [[trial["x"] + c*x - s*y, trial["y"] + s*x + c*y]
               for x, y in [(-width/2, -depth/2), (-width/2, depth/2), (width/2, -depth/2), (width/2, depth/2)]]
    z = survey["hover"][2]
    return {"start": dict(zip(("x", "y", "z"), survey["hover"][:3])),
            "goal": {"x": task["goal"][0], "y": task["goal"][1], "z": z},
            "workspace": {"x": task["workspace"][:2], "y": task["workspace"][2:], "z": z},
            "obstacles": [{"object_id": "person-1", "label": "person", "yaw_rad": yaw,
                           "footprint_corners_xy": corners,
                           "min_corner": [min(p[0] for p in corners), min(p[1] for p in corners), -1.0],
                           "max_corner": [max(p[0] for p in corners), max(p[1] for p in corners), 0.0]}]}


def export_synthetic(experiment_path):
    from hrrt_star import plan_hrrt_star
    from postprocess_residual_calibration import choose_expert, route_cards
    from finalize_hrrt_sft_dataset import build_prompt, completed_row, write_csv
    experiment = checked_manifest(experiment_path, "experiment")
    root = Path(experiment["output_dir"])
    args = scoring_args(experiment)
    for split in ("Dtrain", "Dval"):
        output = root / f"{split}.csv"
        if output.exists():
            raise ValueError(f"refusing to replace {output}")
    outputs = {}
    for split in ("Dtrain", "Dval"):
        rows = []
        for trial in experiment["splits"][split]:
            env = synthetic_environment(experiment, trial)
            family = plan_hrrt_star(env["start"], env["goal"], env["obstacles"], env["workspace"],
                                   clearance_m=args.clearance_m, max_iterations=args.hrrt_iterations,
                                   max_candidates=args.max_candidates, max_waypoints=args.max_waypoints, seed=args.seed)
            route, selector, _ = choose_expert(family["routes"], env["obstacles"], args)
            raw = {"sample_id": trial["trial_id"], "scene_id": trial["trial_id"], "environment": env,
                   "preference": {"text": args.l2_text, "preference_type": "fixed"},
                   "route_cards": route_cards(family["routes"], env["obstacles"]), "routes": family["routes"]}
            rows.append(completed_row(raw, None, "train" if split == "Dtrain" else "validation",
                                      selector, "dss", trial["trial_id"], "waypoints", route["route_id"],
                                      build_prompt(env, args.l2_text, args.clearance_m), {"waypoints": route["waypoints"]}))
        outputs[split] = rows
    for split, rows in outputs.items():
        write_csv(root / f"{split}.csv", rows)
    write_json(root / "synthetic.provenance.json", {
        "experiment_fingerprint": experiment["fingerprint"],
        "files": {split: file_hash(root / f"{split}.csv") for split in outputs}})


def pipeline_hashes():
    scripts = Path(__file__).resolve().parent
    files = {name: file_hash(scripts / name) for name in PIPELINE_FILES}
    # In a symlink-install __file__ resolves into source; installed copies share lib/.
    for name in ("refinment.py", "verifier.py", "perception_detection.py", "outline_depth.py", "obstacle_depth.py", "outline_segmentation.py", "interactive_mission_gateway.py"):
        candidates = [scripts / name, scripts.parents[1] / "scripts" / name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f"missing production dependency {name}")
        files[name] = file_hash(path)
    return files


def checkpoint_hashes(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    for name in ("model.pt", "config.json", "tokenizer"):
        path = checkpoint / name
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            raise ValueError(f"restore real checkpoint assets at {path}; placeholders cannot be frozen")
    files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
    encoder = checkpoint.parent / "text_encoder"
    if not encoder.is_dir() or not any(encoder.glob("*.safetensors")):
        raise ValueError(f"restore the local text encoder at {encoder}")
    files += sorted(p for p in encoder.rglob("*") if p.is_file())
    return {str(p): file_hash(p) for p in files}


def freeze_checkpoint(experiment_path, checkpoint):
    experiment = checked_manifest(experiment_path, "experiment")
    verify_survey_profile(experiment["domain"]["survey"])
    root = Path(experiment["output_dir"])
    if read_events(root / "collection.events.jsonl"):
        raise ValueError("freeze checkpoint before measured collection")
    provenance = load_json(root / "synthetic.provenance.json")
    if provenance["experiment_fingerprint"] != experiment["fingerprint"]:
        raise ValueError("synthetic data belongs to another experiment")
    for split in ("Dtrain", "Dval"):
        if provenance["files"][split] != file_hash(root / f"{split}.csv"):
            raise ValueError("synthetic training/validation files changed")
    trained_data = load_json(Path(checkpoint).expanduser() / "config.json").get("data", {})
    if trained_data.get("row_limit") is not None:
        raise ValueError("partial-data/smoke-test checkpoints cannot be experimental evidence")
    trained_on = trained_data.get("file_sha256")
    if trained_on != provenance["files"]:
        raise ValueError("checkpoint must be trained on this experiment's Dtrain/Dval; use the updated trainer's saved file hashes")
    from waypoint_transformer import WaypointTransformerPlanner
    WaypointTransformerPlanner(Path(checkpoint), "cpu")  # actual architecture/state compatibility
    return seal(root / "model.freeze.json", {"kind": "model", "experiment_fingerprint": experiment["fingerprint"],
                "checkpoint": str(Path(checkpoint).expanduser().resolve()), "assets": checkpoint_hashes(checkpoint),
                "pipeline": pipeline_hashes(), "synthetic": provenance})


def verify_model(experiment):
    verify_survey_profile(experiment["domain"]["survey"])
    root = Path(experiment["output_dir"])
    frozen = checked_manifest(root / "model.freeze.json", "model")
    if frozen["experiment_fingerprint"] != experiment["fingerprint"] or frozen["pipeline"] != pipeline_hashes():
        raise ValueError("experiment or scoring pipeline changed after checkpoint freeze")
    if frozen["assets"] != checkpoint_hashes(frozen["checkpoint"]):
        raise ValueError("checkpoint changed after freeze")
    if any(file_hash(root / f"{split}.csv") != frozen["synthetic"]["files"][split] for split in ("Dtrain", "Dval")):
        raise ValueError("training/validation data changed after model freeze")
    return frozen


def score_experiment(experiment_path, device="auto"):
    from postprocess_residual_calibration import read_raw, process_capture, write_rows
    from waypoint_transformer import WaypointTransformerPlanner
    experiment = checked_manifest(experiment_path, "experiment")
    root = Path(experiment["output_dir"])
    frozen = verify_model(experiment)
    events = read_events(root / "collection.events.jsonl")
    if any(e.get("type") == "failure" for e in events):
        raise ValueError("experiment has a failed scheduled trial; review and start a new version if needed")
    successes = [e for e in events if e.get("type") == "recorded"]
    expected = {t["trial_id"] for t in experiment["collection_schedule"]}
    if len(successes) != len(expected) or {e["trial_id"] for e in successes} != expected:
        raise ValueError("complete Dcal/Dtest accounting is required before scoring")
    grouped = read_raw(root / "measured.raw.csv")
    if set(grouped) != expected:
        raise ValueError("raw captures differ from scheduled trial IDs")
    if any(len(rows) != 1 or rows[0]["label"] != "person" or rows[0]["missed_detection"].lower() != "false"
           or rows[0].get("placeholder", "").lower() != "false" for rows in grouped.values()):
        raise ValueError("every scheduled scene must contain one real detected person")
    for event in successes:
        if event.get("raw_row_fingerprint") != digest(grouped[event["trial_id"]][0]):
            raise ValueError("raw capture was changed after recording")
    outputs = [root / name for name in ("Dcal.scored.csv", "Dtest.scored.csv", "split_certificate.json", "test.report.json")]
    if any(path.exists() for path in outputs):
        raise ValueError("scoring outputs already exist; do not overwrite frozen results")
    args = scoring_args(experiment)
    args.transformer_planner = WaypointTransformerPlanner(Path(frozen["checkpoint"]), device)
    scored = {}
    for split in ("Dcal", "Dtest"):
        scored[split] = []
        for trial in experiment["splits"][split]:
            try:
                record = process_capture(trial["trial_id"], grouped[trial["trial_id"]], args)
            except (ValueError, RuntimeError) as exc:
                append_event(root / "scoring.failures.jsonl", {"trial_id": trial["trial_id"], "reason": str(exc)})
                raise ValueError(f"scoring stopped at {trial['trial_id']}; no subset certificate written: {exc}") from exc
            scored[split].append(record)
    cal_scores = [float(r["conformity_score"]) for r in scored["Dcal"]]
    result = certificate(cal_scores, experiment["delta"], {
        "experiment_fingerprint": experiment["fingerprint"], "domain_fingerprint": experiment["domain"]["fingerprint"],
        "model_fingerprint": frozen["fingerprint"], "pipeline_fingerprint": digest(frozen["pipeline"]),
        "raw_dataset_sha256": file_hash(root / "measured.raw.csv"), "calibration_split": "Dcal",
        "score_definition": "continuous_ZOH_sup_norm_P_B_du_plus_K_dx", "calibration_csv": str(root / "Dcal.scored.csv")})
    if result["quantile_unbounded"]:
        raise ValueError("finite split-CP quantile unavailable; no usable tube can be certified")
    for split in scored:
        write_rows(root / f"{split}.scored.csv", scored[split])
    result["calibration_dataset_sha256"] = file_hash(root / "Dcal.scored.csv")
    write_json(root / "split_certificate.json", result)
    errors = [r["capture_id"] for r in scored["Dtest"] if float(r["conformity_score"]) > result["calibrated_quantile"]]
    write_json(root / "test.report.json", {"experiment_fingerprint": experiment["fingerprint"],
               "test_count": len(scored["Dtest"]), "miscoverage_count": len(errors), "failed_trial_ids": errors,
               "empirical_coverage": 1 - len(errors)/len(scored["Dtest"]),
               "certificate_sha256": file_hash(root / "split_certificate.json"),
               "scope": "held-out residual-score coverage under perfect tracking; not hardware flight safety"})
    return result


def init_survey(args):
    root = Path(args.output_dir).expanduser().resolve()
    # Explicit source destination: never silently write into an installed package.
    source = Path(args.source_root).expanduser().resolve()
    if not (source / "package.xml").is_file() or not root.is_relative_to(source / "fine_tuning" / "datasets"):
        raise ValueError("output-dir must be below the source package's fine_tuning/datasets")
    if root.exists():
        raise ValueError("use a new versioned output directory")
    params = Path(args.params_file).expanduser().resolve()
    config = yaml.safe_load(params.read_text())
    perception = config["semantic_obstacle_perception"]["ros__parameters"]
    for field in ("detection_topic", "detector_image_topic", "point_cloud_topic", "pose_topic", "obstacle_topic"):
        if not str(perception.get(field, "")).startswith("/"):
            raise ValueError(f"configure absolute perception topic {field} before survey")
    if not args.person_topic.startswith("/"):
        raise ValueError("person-topic must be an absolute ROS topic")
    if perception.get("camera_calibration_valid") is not True or perception.get("z_estimation_mode") != "depth":
        raise ValueError("a verified camera calibration and measured depth are required")
    if not 0 < float(perception.get("min_confidence", 0.7)) <= 1:
        raise ValueError("min_confidence must be in (0,1]")
    require_finite([*args.hover, args.person_width, args.person_depth, args.interval], "survey settings")
    if min(args.person_width, args.person_depth) <= 0 or min(args.visits, args.checks) <= 0 or args.interval < 1:
        raise ValueError("positive dimensions/counts and interval >=1 second required")
    return seal(root / "survey.json", {
        "kind": "survey", "output_dir": str(root), "params_file": str(params), "params_sha256": file_hash(params),
        "source_root": str(source), "perception": perception,
        "perception_fingerprint": perception_fingerprint(source, perception),
        "locations": grid_locations(args.bounds, args.spacing),
        "spacing_m": args.spacing, "visits": args.visits, "checks_per_visit": args.checks,
        "interval_s": args.interval, "seed": args.seed,
        # PX4/NED yaw measured at Starling's fixed takeoff heading.  This is
        # a fixed camera-domain reference, not an operator-provided value.
        "hover": [*args.hover, STARLING_FIXED_CAMERA_YAW_RAD],
        "person": {"object_id": "person-1", "label": "person", "topic": args.person_topic,
                   "dimensions_m": [args.person_width, args.person_depth], "yaw_rad": 0.0},
        "vehicle_position_tolerance_m": 0.10,
        "person_position_tolerance_m": 0.20,
        "yaw_tolerance_rad": math.radians(12),
        "min_confidence": float(perception.get("min_confidence", 0.7)), "freshness_s": 1.0,
        "sync_tolerance_s": 0.1, "match_distance_m": 0.75, "stability_window_s": 0.5,
        "stability_min_samples": 5, "stability_position_tolerance_m": 0.02,
        "capture_rule": "first scheduled fresh observation after Vicon-only settling; never best-of/retry"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    survey = sub.add_parser("init-survey")
    for flag in ("output-dir", "source-root", "params-file"):
        survey.add_argument(f"--{flag}", required=True)
    survey.add_argument("--bounds", type=float, nargs=4, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    survey.add_argument("--hover", type=float, nargs=3, required=True, metavar=("X", "Y", "Z_NED"))
    survey.add_argument("--person-width", type=float, required=True)
    survey.add_argument("--person-depth", type=float, required=True)
    survey.add_argument("--person-topic", default="/vicon/person/person")
    survey.add_argument("--spacing", type=float, default=0.50)
    survey.add_argument("--visits", type=int, default=3)
    survey.add_argument("--checks", type=int, default=10)
    survey.add_argument("--interval", type=float, default=1.0)
    survey.add_argument("--seed", type=int, default=17)
    domain = sub.add_parser("freeze-domain")
    domain.add_argument("--survey", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--domain", required=True)
    for flag in ("n-train", "n-val", "n-cal", "n-test"):
        prep.add_argument(f"--{flag}", type=int, required=True)
    prep.add_argument("--delta", type=float, required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--goal", type=float, nargs=2, required=True)
    prep.add_argument("--workspace", type=float, nargs=4, required=True)
    prep.add_argument("--l1", required=True)
    prep.add_argument("--l2", required=True)
    prep.add_argument("--clearance", type=float, default=0.4)
    prep.add_argument("--expert-selector", choices=("heuristic", "openai"), required=True)
    prep.add_argument("--expert-model", default="gpt-5.4")
    prep.add_argument("--hrrt-iterations", type=int, default=500)
    for name in ("export-synthetic", "freeze-model", "score"):
        child = sub.add_parser(name)
        child.add_argument("--experiment", required=True)
        if name == "freeze-model":
            child.add_argument("--checkpoint", required=True)
        if name == "score":
            child.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        if args.command == "init-survey":
            result = init_survey(args)
        elif args.command == "freeze-domain":
            result = freeze_domain(args.survey)
        elif args.command == "prepare":
            result = prepare(args.domain, dict(Dtrain=args.n_train, Dval=args.n_val, Dcal=args.n_cal, Dtest=args.n_test),
                             args.delta, args.seed, {"goal": args.goal, "workspace": args.workspace,
                             "l1_text": args.l1, "l2_text": args.l2, "clearance_m": args.clearance},
                             {"selector": args.expert_selector, "model": args.expert_model, "iterations": args.hrrt_iterations})
        elif args.command == "export-synthetic":
            export_synthetic(args.experiment)
            result = {"status": "exported Dtrain.csv and Dval.csv"}
        elif args.command == "freeze-model":
            result = freeze_checkpoint(args.experiment, args.checkpoint)
        else:
            result = score_experiment(args.experiment, args.device)
        print(json.dumps({k: result[k] for k in ("kind", "fingerprint", "output_dir", "status", "calibrated_quantile", "calibrated_radius_m") if k in result}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
