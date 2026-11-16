import json
import math
import os
from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, OpaqueFunction
from launch.conditions import UnlessCondition, IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def collection_preflight(context):
    """Fail before vehicle nodes start; report the actual installation and paths."""
    def value(name):
        return LaunchConfiguration(name).perform(context)
    from ament_index_python.packages import get_package_prefix
    prefix = Path(get_package_prefix("llm_vision_planner"))
    params = Path(value("params_file")).expanduser().resolve()
    if not params.is_file():
        raise RuntimeError(f"Calibration YAML does not exist: {params}; source the final-testing install and supply an absolute params_file.")
    mode = value("collection_mode")
    print(f"Calibration package: {prefix}\nConfiguration: {params}\nCollection mode: {mode}", flush=True)
    if mode == "legacy":
        output = Path(value("output_csv")).expanduser()
        if not output.is_absolute():
            raise RuntimeError("output_csv must be an absolute path in the source dataset directory")
        if value("trial_id").lower() in ("", "unset", "placeholder"):
            raise RuntimeError("provide a unique trial_id before starting collection")
        sys.path.insert(0, str(prefix / "lib" / "llm_vision_planner"))
        from residual_calibration_recorder import parse_objects
        objects = json.loads(value("vicon_objects_json")) or [{"object_id": value("object_id"),
            "label": value("object_label"), "topic": value("object_vicon_topic"),
            "dimensions_m": [float(value("object_width_m")), float(value("object_depth_m"))]}]
        parse_objects(json.dumps(objects))
        if not output.parent.is_dir() or not os.access(output.parent, os.W_OK):
            raise RuntimeError(f"create a writable output directory first: {output.parent}")
        return []
    if mode not in ("survey", "scheduled"):
        raise RuntimeError("collection_mode must be legacy, survey, or scheduled")
    sys.path.insert(0, str(prefix / "lib" / "llm_vision_planner"))
    from person_experiment import checked_manifest, file_hash, verify_model, verify_survey_profile
    manifest_path = Path(value("experiment_manifest")).expanduser().resolve()
    manifest = checked_manifest(manifest_path, "survey" if mode == "survey" else "experiment")
    survey = manifest if mode == "survey" else manifest["domain"]["survey"]
    verify_survey_profile(survey)
    if abs(float(survey["person"]["yaw_rad"])) > 1e-12:
        raise RuntimeError("version 2.0 fixes person yaw to 0 degrees NED; create a new survey manifest")
    if not math.isfinite(float(survey["hover"][3])):
        raise RuntimeError("survey hover camera yaw must be finite")
    if file_hash(params) != survey["params_sha256"]:
        raise RuntimeError("params_file differs from frozen survey configuration")
    installed_script = prefix / "lib" / "llm_vision_planner" / "person_experiment.py"
    source_script = Path(survey["source_root"]) / "fine_tuning" / "scripts" / "person_experiment.py"
    if file_hash(installed_script) != file_hash(source_script):
        raise RuntimeError("installed code differs from frozen source package; rebuild and source its install/setup.bash")
    if mode == "scheduled":
        verify_model(manifest)
    replay = value("replay").lower() in ("true", "1")
    use_sim_time = value("use_sim_time").lower() in ("true", "1")
    if replay or use_sim_time:
        raise RuntimeError("guided domain/certification evidence requires live clocks; use legacy replay or isolated tests for validation")
    output = Path(survey["output_dir"]) / ("survey.raw.csv" if mode == "survey" else "measured.raw.csv")
    if not output.parent.is_dir() or not os.access(output.parent, os.W_OK):
        raise RuntimeError(f"output directory is not writable: {output.parent}")
    from residual_calibration_recorder import parse_objects
    parse_objects(json.dumps([survey["person"]]))
    if not output.resolve().is_relative_to(Path(survey["source_root"]) / "fine_tuning" / "datasets"):
        raise RuntimeError("guided output must be inside the source dataset directory")
    recorder_params = {
        "use_sim_time": use_sim_time, "trial_id": output.parent.name,
        "output_csv": str(output), "vicon_objects_json": json.dumps([survey["person"]]),
        "scheduled_capture": True, "collection_mode": mode, "experiment_manifest": str(manifest_path),
        "replay": replay, "sync_tolerance_s": survey["sync_tolerance_s"],
        "object_stability_window_s": survey["stability_window_s"],
        "object_stability_min_samples": survey["stability_min_samples"],
        "object_stability_position_tolerance_m": survey["stability_position_tolerance_m"],
        "match_distance_m": survey["match_distance_m"],
    }
    nodes = [Node(package="llm_vision_planner", executable="person_workspace_survey.py",
                  name="residual_calibration_recorder", output="screen", parameters=[str(params), recorder_params]),
             Node(package="llm_vision_planner", executable="perception_detection.py", name="semantic_obstacle_perception",
                  output="screen", parameters=[str(params), {"use_sim_time": use_sim_time, "debug": False}])]
    if not replay:
        nodes.append(Node(package="llm_vision_planner", executable="control_law_executer.py",
            name="control_law_executer", output="screen", parameters=[str(params), {
                "takeoff_gate_topic": "/llm_vision/vision_calibration_status", "takeoff_gate_status": "FRAME_READY",
                "takeoff_z": survey["hover"][2], "debug": False}]))
    count = len(survey["locations"]) * survey["visits"] * survey["checks_per_visit"] if mode == "survey" else len(manifest["collection_schedule"])
    print(f"Person-only {mode}; {count} scheduled observations; dataset: {output}", flush=True)
    return nodes


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    common_arguments = [
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("llm_vision_planner"), "config", "llm_vision_planner.yaml"]
            ),
        ),
        DeclareLaunchArgument("trial_id", default_value="unset"),
        DeclareLaunchArgument("collection_mode", default_value="legacy"),
        DeclareLaunchArgument("experiment_manifest", default_value=""),
        DeclareLaunchArgument(
            "output_csv",
            default_value="fine_tuning/datasets/calibration_residual_raw.csv",
        ),
        DeclareLaunchArgument("vicon_objects_json", default_value="[]"),
        DeclareLaunchArgument("object_id", default_value="obj-1"),
        DeclareLaunchArgument("object_label", default_value="chair"),
        DeclareLaunchArgument("object_vicon_topic", default_value="/vicon/chair1/chair1"),
        DeclareLaunchArgument("object_width_m", default_value="0.0"),
        DeclareLaunchArgument("object_depth_m", default_value="0.0"),
        DeclareLaunchArgument("vicon_vehicle_topic", default_value="/vicon/Starling2/Starling2"),
        DeclareLaunchArgument("vicon_vehicle_frame_convention", default_value="flu"),
        DeclareLaunchArgument("pose_topic", default_value="/fmu/out/vehicle_odometry"),
        DeclareLaunchArgument("frame_calibration_samples", default_value="20"),
        DeclareLaunchArgument("frame_sync_tolerance_s", default_value="0.10"),
        DeclareLaunchArgument("sync_tolerance_s", default_value="0.10"),
        DeclareLaunchArgument("vicon_time_offset_s", default_value="0.0"),
        DeclareLaunchArgument("match_distance_m", default_value="0.75"),
        DeclareLaunchArgument("capture_position_change_threshold_m", default_value="0.15"),
        DeclareLaunchArgument("capture_yaw_change_threshold_rad", default_value="0.261799"),
        DeclareLaunchArgument("calibration_capture_delay_s", default_value="1.0"),
        DeclareLaunchArgument("calibration_capture_interval_s", default_value="3.0"),
        DeclareLaunchArgument("openai_intent_model", default_value="gpt-5.4-nano"),
        # Depth behind each detected object's front: perception (our outline +
        # ToF method, no API calls) or gpt (GPT nano from the label; needs
        # intent_provider:=openai and OPENAI_API_KEY).
        DeclareLaunchArgument("nominal_depth_source", default_value="perception"),
        DeclareLaunchArgument(
            "intent_provider",
            default_value=PythonExpression(
                ["'openai' if '", LaunchConfiguration("nominal_depth_source"), "' == 'gpt' else 'mock'"]
            ),
        ),
        DeclareLaunchArgument("depth_estimator", default_value="outline"),
        DeclareLaunchArgument("efficientsam_model", default_value=""),
        # replay:=true records from a rosbag instead of a flight: no takeoff
        # node (nothing is sent to a vehicle), no mission-state gate, and
        # use_sim_time follows `ros2 bag play --clock`.
        DeclareLaunchArgument("replay", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value=LaunchConfiguration("replay")),
    ]
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    live = ParameterValue(
        PythonExpression(["'", LaunchConfiguration("replay"), "'.lower() not in ('true', '1')"]),
        value_type=bool,
    )

    recorder = Node(
        package="llm_vision_planner",
        executable="residual_calibration_recorder.py",
        name="residual_calibration_recorder",
        output="screen",
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("collection_mode"), "' == 'legacy'"])),
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "trial_id": LaunchConfiguration("trial_id"),
                "output_csv": LaunchConfiguration("output_csv"),
                "vicon_objects_json": ParameterValue(
                    LaunchConfiguration("vicon_objects_json"), value_type=str
                ),
                "object_id": LaunchConfiguration("object_id"),
                "object_label": LaunchConfiguration("object_label"),
                "object_vicon_topic": LaunchConfiguration("object_vicon_topic"),
                "object_width_m": ParameterValue(
                    LaunchConfiguration("object_width_m"), value_type=float
                ),
                "object_depth_m": ParameterValue(
                    LaunchConfiguration("object_depth_m"), value_type=float
                ),
                "vicon_vehicle_topic": LaunchConfiguration("vicon_vehicle_topic"),
                "vicon_vehicle_frame_convention": LaunchConfiguration(
                    "vicon_vehicle_frame_convention"
                ),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "frame_calibration_samples": ParameterValue(
                    LaunchConfiguration("frame_calibration_samples"), value_type=int
                ),
                "frame_sync_tolerance_s": ParameterValue(
                    LaunchConfiguration("frame_sync_tolerance_s"), value_type=float
                ),
                "sync_tolerance_s": ParameterValue(
                    LaunchConfiguration("sync_tolerance_s"), value_type=float
                ),
                "vicon_time_offset_s": ParameterValue(
                    LaunchConfiguration("vicon_time_offset_s"), value_type=float
                ),
                "match_distance_m": ParameterValue(
                    LaunchConfiguration("match_distance_m"), value_type=float
                ),
                "capture_position_change_threshold_m": ParameterValue(
                    LaunchConfiguration("capture_position_change_threshold_m"), value_type=float
                ),
                "capture_yaw_change_threshold_rad": ParameterValue(
                    LaunchConfiguration("capture_yaw_change_threshold_rad"), value_type=float
                ),
            },
        ],
    )

    perception = Node(
        package="llm_vision_planner",
        executable="perception_detection.py",
        name="semantic_obstacle_perception",
        output="screen",
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("collection_mode"), "' == 'legacy'"])),
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "depth_estimator": LaunchConfiguration("depth_estimator"),
                "efficientsam_model": LaunchConfiguration("efficientsam_model"),
                "debug": False,
            },
        ],
    )

    takeoff = Node(
        package="llm_vision_planner",
        executable="control_law_executer.py",
        name="control_law_executer",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "takeoff_gate_topic": "/llm_vision/vision_calibration_status",
                "takeoff_gate_status": "FRAME_READY",
            },
        ],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("collection_mode"), "' == 'legacy' and '", LaunchConfiguration("replay"), "'.lower() not in ('true', '1')"])),
    )

    nominal_environment = Node(
        package="llm_vision_planner",
        executable="interactive_mission_gateway.py",
        name="interactive_mission_gateway",
        output="screen",
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("collection_mode"), "' == 'legacy'"])),
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "environment": "real",
                "intent_provider": LaunchConfiguration("intent_provider"),
                "openai_intent_model": LaunchConfiguration("openai_intent_model"),
                "nominal_depth_source": LaunchConfiguration("nominal_depth_source"),
                "visualizer": "standard",
                "require_mission_state": live,
                "calibration_only": True,
                "debug": False,
                "auto_calibration_capture": True,
                "continuous_calibration_capture": True,
                "calibration_capture_delay_s": ParameterValue(
                    LaunchConfiguration("calibration_capture_delay_s"), value_type=float
                ),
                "calibration_capture_interval_s": ParameterValue(
                    LaunchConfiguration("calibration_capture_interval_s"), value_type=float
                ),
                "pose_topic": LaunchConfiguration("pose_topic"),
            },
        ],
    )

    return LaunchDescription(
        [
            *common_arguments,
            OpaqueFunction(function=collection_preflight),
            recorder,
            perception,
            takeoff,
            TimerAction(period=1.0, actions=[nominal_environment]),
        ]
    )
