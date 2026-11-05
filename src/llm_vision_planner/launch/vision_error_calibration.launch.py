from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


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
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "depth_estimator": LaunchConfiguration("depth_estimator"),
                "efficientsam_model": LaunchConfiguration("efficientsam_model"),
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
        condition=UnlessCondition(LaunchConfiguration("replay")),
    )

    nominal_environment = Node(
        package="llm_vision_planner",
        executable="interactive_mission_gateway.py",
        name="vision_calibration_environment",
        output="screen",
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
            recorder,
            perception,
            takeoff,
            TimerAction(period=1.0, actions=[nominal_environment]),
        ]
    )
